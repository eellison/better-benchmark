class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[16, 512][512, 1]cuda:0", primals_4: "i64[1, 512][512, 1]cuda:0", primals_7: "f32[1024][1]cuda:0", primals_17: "f32[1024][1]cuda:0", primals_23: "f32[1024][1]cuda:0", primals_33: "f32[1024][1]cuda:0", primals_39: "f32[1024][1]cuda:0", primals_49: "f32[1024][1]cuda:0", primals_55: "f32[1024][1]cuda:0", primals_65: "f32[1024][1]cuda:0", primals_71: "f32[1024][1]cuda:0", primals_81: "f32[1024][1]cuda:0", primals_87: "f32[1024][1]cuda:0", primals_97: "f32[1024][1]cuda:0", primals_103: "f32[1024][1]cuda:0", primals_113: "f32[1024][1]cuda:0", primals_119: "f32[1024][1]cuda:0", primals_129: "f32[1024][1]cuda:0", primals_135: "f32[1024][1]cuda:0", primals_145: "f32[1024][1]cuda:0", primals_151: "f32[1024][1]cuda:0", primals_161: "f32[1024][1]cuda:0", primals_167: "f32[1024][1]cuda:0", primals_177: "f32[1024][1]cuda:0", primals_183: "f32[1024][1]cuda:0", primals_193: "f32[1024][1]cuda:0", primals_199: "f32[1024][1]cuda:0", primals_209: "f32[1024][1]cuda:0", primals_215: "f32[1024][1]cuda:0", primals_225: "f32[1024][1]cuda:0", primals_231: "f32[1024][1]cuda:0", primals_241: "f32[1024][1]cuda:0", primals_247: "f32[1024][1]cuda:0", primals_257: "f32[1024][1]cuda:0", primals_263: "f32[1024][1]cuda:0", primals_273: "f32[1024][1]cuda:0", primals_279: "f32[1024][1]cuda:0", primals_289: "f32[1024][1]cuda:0", primals_295: "f32[1024][1]cuda:0", primals_305: "f32[1024][1]cuda:0", primals_311: "f32[1024][1]cuda:0", primals_321: "f32[1024][1]cuda:0", primals_327: "f32[1024][1]cuda:0", primals_337: "f32[1024][1]cuda:0", primals_343: "f32[1024][1]cuda:0", primals_353: "f32[1024][1]cuda:0", primals_359: "f32[1024][1]cuda:0", primals_369: "f32[1024][1]cuda:0", primals_375: "f32[1024][1]cuda:0", primals_385: "f32[1024][1]cuda:0", primals_391: "f32[1024][1]cuda:0", primals_395: "f32[1024][1]cuda:0", full_default: "i64[16, 512][512, 1]cuda:0", gt: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_3: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_138: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_139: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_140: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_261: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_262: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_263: "u64[2][1]cuda:0", getitem_264: "u64[][]cuda:0", view_16: "bf16[8192, 1024][1024, 1]cuda:0", gt_2: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_9: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_18: "bf16[8192, 1024][1024, 1]cuda:0", addmm_4: "bf16[8192, 4096][4096, 1]cuda:0", view_20: "bf16[8192, 4096][4096, 1]cuda:0", gt_3: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_16: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_22: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_132: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_133: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_134: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_254: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_255: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_256: "u64[2][1]cuda:0", getitem_257: "u64[][]cuda:0", view_38: "bf16[8192, 1024][1024, 1]cuda:0", gt_5: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_22: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_40: "bf16[8192, 1024][1024, 1]cuda:0", addmm_10: "bf16[8192, 4096][4096, 1]cuda:0", view_42: "bf16[8192, 4096][4096, 1]cuda:0", gt_6: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_29: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_44: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_126: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_127: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_128: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_247: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_248: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_249: "u64[2][1]cuda:0", getitem_250: "u64[][]cuda:0", view_60: "bf16[8192, 1024][1024, 1]cuda:0", gt_8: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_35: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_62: "bf16[8192, 1024][1024, 1]cuda:0", addmm_16: "bf16[8192, 4096][4096, 1]cuda:0", view_64: "bf16[8192, 4096][4096, 1]cuda:0", gt_9: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_42: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_66: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_120: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_121: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_122: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_240: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_241: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_242: "u64[2][1]cuda:0", getitem_243: "u64[][]cuda:0", view_82: "bf16[8192, 1024][1024, 1]cuda:0", gt_11: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_48: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_84: "bf16[8192, 1024][1024, 1]cuda:0", addmm_22: "bf16[8192, 4096][4096, 1]cuda:0", view_86: "bf16[8192, 4096][4096, 1]cuda:0", gt_12: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_55: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_88: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_114: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_115: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_116: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_233: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_234: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_235: "u64[2][1]cuda:0", getitem_236: "u64[][]cuda:0", view_104: "bf16[8192, 1024][1024, 1]cuda:0", gt_14: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_61: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_106: "bf16[8192, 1024][1024, 1]cuda:0", addmm_28: "bf16[8192, 4096][4096, 1]cuda:0", view_108: "bf16[8192, 4096][4096, 1]cuda:0", gt_15: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_68: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_110: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_108: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_109: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_110: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_226: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_227: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_228: "u64[2][1]cuda:0", getitem_229: "u64[][]cuda:0", view_126: "bf16[8192, 1024][1024, 1]cuda:0", gt_17: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_74: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_128: "bf16[8192, 1024][1024, 1]cuda:0", addmm_34: "bf16[8192, 4096][4096, 1]cuda:0", view_130: "bf16[8192, 4096][4096, 1]cuda:0", gt_18: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_81: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_132: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_102: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_103: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_104: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_219: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_220: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_221: "u64[2][1]cuda:0", getitem_222: "u64[][]cuda:0", view_148: "bf16[8192, 1024][1024, 1]cuda:0", gt_20: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_87: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_150: "bf16[8192, 1024][1024, 1]cuda:0", addmm_40: "bf16[8192, 4096][4096, 1]cuda:0", view_152: "bf16[8192, 4096][4096, 1]cuda:0", gt_21: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_94: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_154: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_96: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_97: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_98: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_212: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_213: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_214: "u64[2][1]cuda:0", getitem_215: "u64[][]cuda:0", view_170: "bf16[8192, 1024][1024, 1]cuda:0", gt_23: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_100: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_172: "bf16[8192, 1024][1024, 1]cuda:0", addmm_46: "bf16[8192, 4096][4096, 1]cuda:0", view_174: "bf16[8192, 4096][4096, 1]cuda:0", gt_24: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_176: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_90: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_91: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_92: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_205: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_206: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_207: "u64[2][1]cuda:0", getitem_208: "u64[][]cuda:0", view_192: "bf16[8192, 1024][1024, 1]cuda:0", gt_26: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_113: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_194: "bf16[8192, 1024][1024, 1]cuda:0", addmm_52: "bf16[8192, 4096][4096, 1]cuda:0", view_196: "bf16[8192, 4096][4096, 1]cuda:0", gt_27: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_120: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_198: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_84: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_85: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_86: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_198: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_199: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_200: "u64[2][1]cuda:0", getitem_201: "u64[][]cuda:0", view_214: "bf16[8192, 1024][1024, 1]cuda:0", gt_29: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_126: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_216: "bf16[8192, 1024][1024, 1]cuda:0", addmm_58: "bf16[8192, 4096][4096, 1]cuda:0", view_218: "bf16[8192, 4096][4096, 1]cuda:0", gt_30: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_133: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_220: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_78: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_79: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_80: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_191: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_192: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_193: "u64[2][1]cuda:0", getitem_194: "u64[][]cuda:0", view_236: "bf16[8192, 1024][1024, 1]cuda:0", gt_32: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_139: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_238: "bf16[8192, 1024][1024, 1]cuda:0", addmm_64: "bf16[8192, 4096][4096, 1]cuda:0", view_240: "bf16[8192, 4096][4096, 1]cuda:0", gt_33: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_146: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_242: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_72: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_73: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_74: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_184: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_185: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_186: "u64[2][1]cuda:0", getitem_187: "u64[][]cuda:0", view_258: "bf16[8192, 1024][1024, 1]cuda:0", gt_35: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_152: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_260: "bf16[8192, 1024][1024, 1]cuda:0", addmm_70: "bf16[8192, 4096][4096, 1]cuda:0", view_262: "bf16[8192, 4096][4096, 1]cuda:0", gt_36: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_159: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_264: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_66: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_67: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_68: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_177: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_178: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_179: "u64[2][1]cuda:0", getitem_180: "u64[][]cuda:0", view_280: "bf16[8192, 1024][1024, 1]cuda:0", gt_38: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_165: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_282: "bf16[8192, 1024][1024, 1]cuda:0", addmm_76: "bf16[8192, 4096][4096, 1]cuda:0", view_284: "bf16[8192, 4096][4096, 1]cuda:0", gt_39: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_172: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_286: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_60: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_61: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_62: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_170: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_171: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_172: "u64[2][1]cuda:0", getitem_173: "u64[][]cuda:0", view_302: "bf16[8192, 1024][1024, 1]cuda:0", gt_41: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_178: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_304: "bf16[8192, 1024][1024, 1]cuda:0", addmm_82: "bf16[8192, 4096][4096, 1]cuda:0", view_306: "bf16[8192, 4096][4096, 1]cuda:0", gt_42: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_185: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_308: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_54: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_55: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_56: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_163: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_164: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_165: "u64[2][1]cuda:0", getitem_166: "u64[][]cuda:0", view_324: "bf16[8192, 1024][1024, 1]cuda:0", gt_44: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_191: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_326: "bf16[8192, 1024][1024, 1]cuda:0", addmm_88: "bf16[8192, 4096][4096, 1]cuda:0", view_328: "bf16[8192, 4096][4096, 1]cuda:0", gt_45: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_198: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_330: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_48: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_49: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_50: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_156: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_157: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_158: "u64[2][1]cuda:0", getitem_159: "u64[][]cuda:0", view_346: "bf16[8192, 1024][1024, 1]cuda:0", gt_47: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_204: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_348: "bf16[8192, 1024][1024, 1]cuda:0", addmm_94: "bf16[8192, 4096][4096, 1]cuda:0", view_350: "bf16[8192, 4096][4096, 1]cuda:0", gt_48: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_211: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_352: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_42: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_43: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_44: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_149: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_150: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_151: "u64[2][1]cuda:0", getitem_152: "u64[][]cuda:0", view_368: "bf16[8192, 1024][1024, 1]cuda:0", gt_50: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_217: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_370: "bf16[8192, 1024][1024, 1]cuda:0", addmm_100: "bf16[8192, 4096][4096, 1]cuda:0", view_372: "bf16[8192, 4096][4096, 1]cuda:0", gt_51: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_224: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_374: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_36: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_37: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_38: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_142: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_143: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_144: "u64[2][1]cuda:0", getitem_145: "u64[][]cuda:0", view_390: "bf16[8192, 1024][1024, 1]cuda:0", gt_53: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_230: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_392: "bf16[8192, 1024][1024, 1]cuda:0", addmm_106: "bf16[8192, 4096][4096, 1]cuda:0", view_394: "bf16[8192, 4096][4096, 1]cuda:0", gt_54: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_237: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_396: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_30: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_31: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_32: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_135: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_136: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_137: "u64[2][1]cuda:0", getitem_138: "u64[][]cuda:0", view_412: "bf16[8192, 1024][1024, 1]cuda:0", gt_56: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_243: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_414: "bf16[8192, 1024][1024, 1]cuda:0", addmm_112: "bf16[8192, 4096][4096, 1]cuda:0", view_416: "bf16[8192, 4096][4096, 1]cuda:0", gt_57: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_250: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_418: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_24: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_25: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_26: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_128: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_129: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_130: "u64[2][1]cuda:0", getitem_131: "u64[][]cuda:0", view_434: "bf16[8192, 1024][1024, 1]cuda:0", gt_59: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_256: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_436: "bf16[8192, 1024][1024, 1]cuda:0", addmm_118: "bf16[8192, 4096][4096, 1]cuda:0", view_438: "bf16[8192, 4096][4096, 1]cuda:0", gt_60: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_263: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_440: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_18: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_19: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_20: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_121: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_122: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_123: "u64[2][1]cuda:0", getitem_124: "u64[][]cuda:0", view_456: "bf16[8192, 1024][1024, 1]cuda:0", gt_62: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_269: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_458: "bf16[8192, 1024][1024, 1]cuda:0", addmm_124: "bf16[8192, 4096][4096, 1]cuda:0", view_460: "bf16[8192, 4096][4096, 1]cuda:0", gt_63: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_276: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_462: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_12: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_13: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_14: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_114: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_115: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_116: "u64[2][1]cuda:0", getitem_117: "u64[][]cuda:0", view_478: "bf16[8192, 1024][1024, 1]cuda:0", gt_65: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_282: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_480: "bf16[8192, 1024][1024, 1]cuda:0", addmm_130: "bf16[8192, 4096][4096, 1]cuda:0", view_482: "bf16[8192, 4096][4096, 1]cuda:0", gt_66: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_289: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_484: "bf16[8192, 1024][1024, 1]cuda:0", permute_default_6: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_7: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_8: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_107: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_108: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_109: "u64[2][1]cuda:0", getitem_110: "u64[][]cuda:0", view_500: "bf16[8192, 1024][1024, 1]cuda:0", gt_68: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_295: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_502: "bf16[8192, 1024][1024, 1]cuda:0", addmm_136: "bf16[8192, 4096][4096, 1]cuda:0", view_504: "bf16[8192, 4096][4096, 1]cuda:0", gt_69: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_302: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_506: "bf16[8192, 1024][1024, 1]cuda:0", permute_default: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_1: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", permute_default_2: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_100: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0", getitem_101: "f32[16, 16, 512][8192, 512, 1]cuda:0", getitem_102: "u64[2][1]cuda:0", getitem_103: "u64[][]cuda:0", view_522: "bf16[8192, 1024][1024, 1]cuda:0", gt_71: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_308: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_524: "bf16[8192, 1024][1024, 1]cuda:0", addmm_142: "bf16[8192, 4096][4096, 1]cuda:0", view_526: "bf16[8192, 4096][4096, 1]cuda:0", gt_72: "b8[16, 512, 1024][524288, 1024, 1]cuda:0", mul_315: "f32[16, 512, 1024][524288, 1024, 1]cuda:0", view_528: "bf16[8192, 1024][1024, 1]cuda:0", addmm_144: "bf16[8192, 1024][1024, 1]cuda:0", getitem_99: "f32[16, 512, 1][512, 1, 1]cuda:0", rsqrt_49: "f32[16, 512, 1][512, 1, 1]cuda:0", view_530: "bf16[8192, 1024][1024, 1]cuda:0", view_531: "bf16[16, 512, 29056][14876672, 29056, 1]cuda:0", constant_pad_nd: "i64[16, 513][513, 1]cuda:0", amax_24: "f32[8192, 1][1, 1]cuda:0", log: "f32[8192, 1][1, 1]cuda:0", convert_element_type_1000: "f32[][]cuda:0", permute_266: "bf16[29056, 1024][1024, 1]cuda:0", permute_270: "bf16[1024, 1024][1024, 1]cuda:0", div_51: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_274: "bf16[1024, 4096][4096, 1]cuda:0", permute_278: "bf16[4096, 1024][1024, 1]cuda:0", div_52: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_282: "bf16[1024, 1024][1024, 1]cuda:0", permute_293: "bf16[1024, 1024][1024, 1]cuda:0", permute_298: "bf16[1024, 1024][1024, 1]cuda:0", permute_303: "bf16[1024, 1024][1024, 1]cuda:0", div_54: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_307: "bf16[1024, 4096][4096, 1]cuda:0", permute_311: "bf16[4096, 1024][1024, 1]cuda:0", div_55: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_315: "bf16[1024, 1024][1024, 1]cuda:0", permute_326: "bf16[1024, 1024][1024, 1]cuda:0", permute_331: "bf16[1024, 1024][1024, 1]cuda:0", permute_336: "bf16[1024, 1024][1024, 1]cuda:0", div_57: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_340: "bf16[1024, 4096][4096, 1]cuda:0", permute_344: "bf16[4096, 1024][1024, 1]cuda:0", div_58: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_348: "bf16[1024, 1024][1024, 1]cuda:0", permute_359: "bf16[1024, 1024][1024, 1]cuda:0", permute_364: "bf16[1024, 1024][1024, 1]cuda:0", permute_369: "bf16[1024, 1024][1024, 1]cuda:0", div_60: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_373: "bf16[1024, 4096][4096, 1]cuda:0", permute_377: "bf16[4096, 1024][1024, 1]cuda:0", div_61: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_381: "bf16[1024, 1024][1024, 1]cuda:0", permute_392: "bf16[1024, 1024][1024, 1]cuda:0", permute_397: "bf16[1024, 1024][1024, 1]cuda:0", permute_402: "bf16[1024, 1024][1024, 1]cuda:0", div_63: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_406: "bf16[1024, 4096][4096, 1]cuda:0", permute_410: "bf16[4096, 1024][1024, 1]cuda:0", div_64: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_414: "bf16[1024, 1024][1024, 1]cuda:0", permute_425: "bf16[1024, 1024][1024, 1]cuda:0", permute_430: "bf16[1024, 1024][1024, 1]cuda:0", permute_435: "bf16[1024, 1024][1024, 1]cuda:0", div_66: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_439: "bf16[1024, 4096][4096, 1]cuda:0", permute_443: "bf16[4096, 1024][1024, 1]cuda:0", div_67: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_447: "bf16[1024, 1024][1024, 1]cuda:0", permute_458: "bf16[1024, 1024][1024, 1]cuda:0", permute_463: "bf16[1024, 1024][1024, 1]cuda:0", permute_468: "bf16[1024, 1024][1024, 1]cuda:0", div_69: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_472: "bf16[1024, 4096][4096, 1]cuda:0", permute_476: "bf16[4096, 1024][1024, 1]cuda:0", div_70: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_480: "bf16[1024, 1024][1024, 1]cuda:0", permute_491: "bf16[1024, 1024][1024, 1]cuda:0", permute_496: "bf16[1024, 1024][1024, 1]cuda:0", permute_501: "bf16[1024, 1024][1024, 1]cuda:0", div_72: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_505: "bf16[1024, 4096][4096, 1]cuda:0", permute_509: "bf16[4096, 1024][1024, 1]cuda:0", div_73: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_513: "bf16[1024, 1024][1024, 1]cuda:0", permute_524: "bf16[1024, 1024][1024, 1]cuda:0", permute_529: "bf16[1024, 1024][1024, 1]cuda:0", permute_534: "bf16[1024, 1024][1024, 1]cuda:0", div_75: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_538: "bf16[1024, 4096][4096, 1]cuda:0", permute_542: "bf16[4096, 1024][1024, 1]cuda:0", div_76: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_546: "bf16[1024, 1024][1024, 1]cuda:0", permute_557: "bf16[1024, 1024][1024, 1]cuda:0", permute_562: "bf16[1024, 1024][1024, 1]cuda:0", permute_567: "bf16[1024, 1024][1024, 1]cuda:0", div_78: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_571: "bf16[1024, 4096][4096, 1]cuda:0", permute_575: "bf16[4096, 1024][1024, 1]cuda:0", div_79: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_579: "bf16[1024, 1024][1024, 1]cuda:0", permute_590: "bf16[1024, 1024][1024, 1]cuda:0", permute_595: "bf16[1024, 1024][1024, 1]cuda:0", permute_600: "bf16[1024, 1024][1024, 1]cuda:0", div_81: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_604: "bf16[1024, 4096][4096, 1]cuda:0", permute_608: "bf16[4096, 1024][1024, 1]cuda:0", div_82: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_612: "bf16[1024, 1024][1024, 1]cuda:0", permute_623: "bf16[1024, 1024][1024, 1]cuda:0", permute_628: "bf16[1024, 1024][1024, 1]cuda:0", permute_633: "bf16[1024, 1024][1024, 1]cuda:0", div_84: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_637: "bf16[1024, 4096][4096, 1]cuda:0", permute_641: "bf16[4096, 1024][1024, 1]cuda:0", div_85: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_645: "bf16[1024, 1024][1024, 1]cuda:0", permute_656: "bf16[1024, 1024][1024, 1]cuda:0", permute_661: "bf16[1024, 1024][1024, 1]cuda:0", permute_666: "bf16[1024, 1024][1024, 1]cuda:0", div_87: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_670: "bf16[1024, 4096][4096, 1]cuda:0", permute_674: "bf16[4096, 1024][1024, 1]cuda:0", div_88: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_678: "bf16[1024, 1024][1024, 1]cuda:0", permute_689: "bf16[1024, 1024][1024, 1]cuda:0", permute_694: "bf16[1024, 1024][1024, 1]cuda:0", permute_699: "bf16[1024, 1024][1024, 1]cuda:0", div_90: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_703: "bf16[1024, 4096][4096, 1]cuda:0", permute_707: "bf16[4096, 1024][1024, 1]cuda:0", div_91: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_711: "bf16[1024, 1024][1024, 1]cuda:0", permute_722: "bf16[1024, 1024][1024, 1]cuda:0", permute_727: "bf16[1024, 1024][1024, 1]cuda:0", permute_732: "bf16[1024, 1024][1024, 1]cuda:0", div_93: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_736: "bf16[1024, 4096][4096, 1]cuda:0", permute_740: "bf16[4096, 1024][1024, 1]cuda:0", div_94: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_744: "bf16[1024, 1024][1024, 1]cuda:0", permute_755: "bf16[1024, 1024][1024, 1]cuda:0", permute_760: "bf16[1024, 1024][1024, 1]cuda:0", permute_765: "bf16[1024, 1024][1024, 1]cuda:0", div_96: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_769: "bf16[1024, 4096][4096, 1]cuda:0", permute_773: "bf16[4096, 1024][1024, 1]cuda:0", div_97: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_777: "bf16[1024, 1024][1024, 1]cuda:0", permute_788: "bf16[1024, 1024][1024, 1]cuda:0", permute_793: "bf16[1024, 1024][1024, 1]cuda:0", permute_798: "bf16[1024, 1024][1024, 1]cuda:0", div_99: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_802: "bf16[1024, 4096][4096, 1]cuda:0", permute_806: "bf16[4096, 1024][1024, 1]cuda:0", div_100: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_810: "bf16[1024, 1024][1024, 1]cuda:0", permute_821: "bf16[1024, 1024][1024, 1]cuda:0", permute_826: "bf16[1024, 1024][1024, 1]cuda:0", permute_831: "bf16[1024, 1024][1024, 1]cuda:0", div_102: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_835: "bf16[1024, 4096][4096, 1]cuda:0", permute_839: "bf16[4096, 1024][1024, 1]cuda:0", div_103: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_843: "bf16[1024, 1024][1024, 1]cuda:0", permute_854: "bf16[1024, 1024][1024, 1]cuda:0", permute_859: "bf16[1024, 1024][1024, 1]cuda:0", permute_864: "bf16[1024, 1024][1024, 1]cuda:0", div_105: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_868: "bf16[1024, 4096][4096, 1]cuda:0", permute_872: "bf16[4096, 1024][1024, 1]cuda:0", div_106: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_876: "bf16[1024, 1024][1024, 1]cuda:0", permute_887: "bf16[1024, 1024][1024, 1]cuda:0", permute_892: "bf16[1024, 1024][1024, 1]cuda:0", permute_897: "bf16[1024, 1024][1024, 1]cuda:0", div_108: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_901: "bf16[1024, 4096][4096, 1]cuda:0", permute_905: "bf16[4096, 1024][1024, 1]cuda:0", div_109: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_909: "bf16[1024, 1024][1024, 1]cuda:0", permute_920: "bf16[1024, 1024][1024, 1]cuda:0", permute_925: "bf16[1024, 1024][1024, 1]cuda:0", permute_930: "bf16[1024, 1024][1024, 1]cuda:0", div_111: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_934: "bf16[1024, 4096][4096, 1]cuda:0", permute_938: "bf16[4096, 1024][1024, 1]cuda:0", div_112: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_942: "bf16[1024, 1024][1024, 1]cuda:0", permute_953: "bf16[1024, 1024][1024, 1]cuda:0", permute_958: "bf16[1024, 1024][1024, 1]cuda:0", permute_963: "bf16[1024, 1024][1024, 1]cuda:0", div_114: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_967: "bf16[1024, 4096][4096, 1]cuda:0", permute_971: "bf16[4096, 1024][1024, 1]cuda:0", div_115: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_975: "bf16[1024, 1024][1024, 1]cuda:0", permute_986: "bf16[1024, 1024][1024, 1]cuda:0", permute_991: "bf16[1024, 1024][1024, 1]cuda:0", permute_996: "bf16[1024, 1024][1024, 1]cuda:0", div_117: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_1000: "bf16[1024, 4096][4096, 1]cuda:0", permute_1004: "bf16[4096, 1024][1024, 1]cuda:0", div_118: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_1008: "bf16[1024, 1024][1024, 1]cuda:0", permute_1019: "bf16[1024, 1024][1024, 1]cuda:0", permute_1024: "bf16[1024, 1024][1024, 1]cuda:0", permute_1029: "bf16[1024, 1024][1024, 1]cuda:0", div_120: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_1033: "bf16[1024, 4096][4096, 1]cuda:0", permute_1037: "bf16[4096, 1024][1024, 1]cuda:0", div_121: "f32[16, 512, 1][512, 1, 1]cuda:0", permute_1041: "bf16[1024, 1024][1024, 1]cuda:0", permute_1052: "bf16[1024, 1024][1024, 1]cuda:0", permute_1057: "bf16[1024, 1024][1024, 1]cuda:0", permute_1062: "bf16[1024, 1024][1024, 1]cuda:0", div_123: "f32[16, 512, 1][512, 1, 1]cuda:0", tangents_1: "f32[][]cuda:0", tangents_2: "bf16[16, 512, 29056][14876672, 29056, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_49: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_1000);  tangents_1 = convert_element_type_1000 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[16, 512][513, 1]cuda:0" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_96: "i64[16, 512][512, 1]cuda:0" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_533: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(clone_96, [-1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_3: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_533, 1);  view_533 = None
        ne_3: "b8[8192, 1][1, 1]cuda:0" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
        full_default_2: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, unsqueeze_3, full_default_2);  unsqueeze_3 = full_default_2 = None

        # No stacktrace found for following nodes
        iota_default: "i64[29056][1]cuda:0" = torch.ops.prims.iota.default(29056, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 29056][29056, 1]cuda:0" = torch.ops.aten.reshape.default(iota_default, [1, 29056]);  iota_default = None
        expand_default: "i64[8192, 29056][1, 0]cuda:0" = torch.ops.aten.expand.default(where_2, [8192, 29056]);  where_2 = None
        eq_tensor: "b8[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[][]cuda:0" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_3: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ne_3, div_49, full_default_3);  ne_3 = div_49 = full_default_3 = None
        mul_322: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_999: "f32[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_532: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_999, [-1, 29056]);  convert_element_type_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_75: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        sub_76: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        exp_25: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_28: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_322, [1], True)
        mul_323: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_77: "f32[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_534: "f32[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.aten.reshape.default(sub_77, [16, 512, 29056]);  sub_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:56 in ForCausalLMLoss, code: logits = logits.float()
        convert_element_type_1001: "bf16[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_534, torch.bfloat16);  view_534 = None
        add_199: "bf16[16, 512, 29056][14876672, 29056, 1]cuda:0" = torch.ops.aten.add.Tensor(tangents_2, convert_element_type_1001);  tangents_2 = convert_element_type_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        view_535: "bf16[8192, 29056][29056, 1]cuda:0" = torch.ops.aten.reshape.default(add_199, [8192, 29056]);  add_199 = None
        mm: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_266);  permute_266 = None
        permute_267: "bf16[29056, 8192][1, 29056]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_1: "bf16[29056, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_267, view_530);  permute_267 = view_530 = None
        sum_29: "f32[1, 29056][29056, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_535, [0], True, dtype = torch.float32);  view_535 = None
        view_536: "f32[29056][1]cuda:0" = torch.ops.aten.reshape.default(sum_29, [29056]);  sum_29 = None
        convert_element_type_1006: "bf16[29056][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.bfloat16);  view_536 = None
        view_537: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm, [16, 512, 1024]);  mm = None
        convert_element_type_1007: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None
        convert_element_type_1008: "f32[29056, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_1, torch.float32);  mm_1 = None
        convert_element_type_1009: "f32[29056][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1006, torch.float32);  convert_element_type_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_325: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1007, primals_395);  primals_395 = None
        mul_326: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, 1024)
        sum_30: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_144, [16, 512, 1024]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_990: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_317: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, 0.5)
        mul_318: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, 0.7071067811865476)
        erf_24: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_196: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_319: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, add_196);  mul_317 = None
        convert_element_type_991: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        convert_element_type_992: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_991, torch.float32);  convert_element_type_991 = None
        sub_74: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_992, getitem_99);  convert_element_type_992 = getitem_99 = None
        mul_320: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
        mul_327: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, mul_320);  mul_325 = None
        sum_31: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
        mul_328: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_320, sum_31);  sum_31 = None
        sub_79: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_326, sum_30);  mul_326 = sum_30 = None
        sub_80: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_79, mul_328);  sub_79 = mul_328 = None
        div_50: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None
        mul_329: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_80);  div_50 = sub_80 = None
        mul_330: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1007, mul_320);  mul_320 = None
        sum_32: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
        sum_33: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1007, [0, 1]);  convert_element_type_1007 = None
        convert_element_type_1010: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_329, torch.bfloat16);  mul_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1011: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1010, torch.float32);  convert_element_type_1010 = None
        mul_332: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
        mul_333: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, convert_element_type_990)
        mul_334: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, -0.5);  mul_333 = None
        exp_26: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.exp.default(mul_334);  mul_334 = None
        mul_335: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_336: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_990, mul_335);  convert_element_type_990 = mul_335 = None
        add_201: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None
        mul_337: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1011, add_201);  convert_element_type_1011 = add_201 = None
        convert_element_type_1013: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_337, torch.bfloat16);  mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        view_538: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1013, [8192, 1024]);  convert_element_type_1013 = None
        mm_2: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
        permute_271: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_3: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
        sum_34: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_538, [0], True, dtype = torch.float32);  view_538 = None
        view_539: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [1024]);  sum_34 = None
        convert_element_type_1018: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.bfloat16);  view_539 = None
        view_540: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_2, [16, 512, 1024]);  mm_2 = None
        convert_element_type_1019: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_540, torch.float32);  view_540 = None
        convert_element_type_1020: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_3, torch.float32);  mm_3 = None
        convert_element_type_1021: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1018, torch.float32);  convert_element_type_1018 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        mul_339: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1019, primals_391);  primals_391 = None
        mul_340: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, 1024)
        sum_35: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
        mul_341: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_339, mul_315);  mul_339 = None
        sum_36: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
        mul_342: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, sum_36);  sum_36 = None
        sub_82: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_340, sum_35);  mul_340 = sum_35 = None
        sub_83: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_82, mul_342);  sub_82 = mul_342 = None
        mul_343: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_83);  div_51 = sub_83 = None
        mul_344: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1019, mul_315);  mul_315 = None
        sum_37: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
        sum_38: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1019, [0, 1]);  convert_element_type_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1022: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_343, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1023: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_72, torch.bfloat16);  gt_72 = None
        mul_345: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, 1.1111111111111112);  convert_element_type_1023 = None
        mul_346: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1022, mul_345);  convert_element_type_1022 = mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_541: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_346, [8192, 1024]);  mul_346 = None
        mm_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
        permute_275: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_5: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
        sum_39: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_541, [0], True, dtype = torch.float32);  view_541 = None
        view_542: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [1024]);  sum_39 = None
        convert_element_type_1028: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_542, torch.bfloat16);  view_542 = None
        view_543: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_4, [16, 512, 4096]);  mm_4 = None
        convert_element_type_1029: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_5, torch.float32);  mm_5 = None
        convert_element_type_1030: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1028, torch.float32);  convert_element_type_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1031: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_543, torch.float32);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_525: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_142, [16, 512, 4096]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_977: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_525, torch.float32);  view_525 = None
        mul_311: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, 0.7071067811865476)
        erf_23: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_192: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_348: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
        mul_349: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, convert_element_type_977)
        mul_350: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_349, -0.5);  mul_349 = None
        exp_27: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_350);  mul_350 = None
        mul_351: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_352: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_977, mul_351);  convert_element_type_977 = mul_351 = None
        add_203: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_348, mul_352);  mul_348 = mul_352 = None
        mul_353: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1031, add_203);  convert_element_type_1031 = add_203 = None
        convert_element_type_1033: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_353, torch.bfloat16);  mul_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1033, [8192, 4096]);  convert_element_type_1033 = None
        mm_6: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
        permute_279: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_7: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
        sum_40: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_544, [0], True, dtype = torch.float32);  view_544 = None
        view_545: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [4096]);  sum_40 = None
        convert_element_type_1038: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_545, torch.bfloat16);  view_545 = None
        view_546: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_6, [16, 512, 1024]);  mm_6 = None
        convert_element_type_1039: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_546, torch.float32);  view_546 = None
        convert_element_type_1040: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_7, torch.float32);  mm_7 = None
        convert_element_type_1041: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1038, torch.float32);  convert_element_type_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_355: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1039, primals_385);  primals_385 = None
        mul_356: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_355, 1024)
        sum_41: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
        mul_357: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_355, mul_308);  mul_355 = None
        sum_42: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
        mul_358: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, sum_42);  sum_42 = None
        sub_85: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_356, sum_41);  mul_356 = sum_41 = None
        sub_86: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_85, mul_358);  sub_85 = mul_358 = None
        mul_359: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_86);  div_52 = sub_86 = None
        mul_360: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1039, mul_308);  mul_308 = None
        sum_43: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
        sum_44: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1039, [0, 1]);  convert_element_type_1039 = None
        add_204: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_343, mul_359);  mul_343 = mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1042: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1043: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_71, torch.bfloat16);  gt_71 = None
        mul_361: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1043, 1.1111111111111112);  convert_element_type_1043 = None
        mul_362: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1042, mul_361);  convert_element_type_1042 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_547: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_362, [8192, 1024]);  mul_362 = None
        mm_8: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
        permute_283: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_547, [1, 0])
        mm_9: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
        sum_45: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_547, [0], True, dtype = torch.float32);  view_547 = None
        view_548: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [1024]);  sum_45 = None
        convert_element_type_1048: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_548, torch.bfloat16);  view_548 = None
        view_549: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_8, [16, 512, 1024]);  mm_8 = None
        convert_element_type_1049: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_9, torch.float32);  mm_9 = None
        convert_element_type_1050: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1048, torch.float32);  convert_element_type_1048 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_550: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_549, [16, 512, 16, 64]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_286: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_99: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_99, permute_default, permute_default_1, permute_default_2, getitem_100, getitem_101, None, None, 512, 512, 0.1, False, getitem_102, getitem_103, scale = 0.125);  clone_99 = permute_default = permute_default_1 = permute_default_2 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = None
        getitem_104: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default[0]
        getitem_105: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default[1]
        getitem_106: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default[2];  _scaled_dot_product_flash_attention_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_5: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_4: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_3: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        clone_101: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_557: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_101, [16, 512, 1024]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_558: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_557, [8192, 1024]);  view_557 = None
        mm_10: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_558, permute_293);  permute_293 = None
        permute_294: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_11: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        sum_47: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_558, [0], True, dtype = torch.float32);  view_558 = None
        view_559: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_47, [1024]);  sum_47 = None
        convert_element_type_1066: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_559, torch.bfloat16);  view_559 = None
        view_560: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_10, [16, 512, 1024]);  mm_10 = None
        convert_element_type_1067: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_560, torch.float32);  view_560 = None
        convert_element_type_1068: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_11, torch.float32);  mm_11 = None
        convert_element_type_1069: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1066, torch.float32);  convert_element_type_1066 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_561: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_4, [16, 512, 1024]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_102: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_561, memory_format = torch.contiguous_format);  view_561 = None
        view_562: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_102, [8192, 1024]);  clone_102 = None
        mm_12: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_562, permute_298);  permute_298 = None
        permute_299: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_13: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_299, view_506);  permute_299 = None
        sum_48: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_562, [0], True, dtype = torch.float32);  view_562 = None
        view_563: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_48, [1024]);  sum_48 = None
        convert_element_type_1074: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_563, torch.bfloat16);  view_563 = None
        view_564: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_12, [16, 512, 1024]);  mm_12 = None
        convert_element_type_1075: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_564, torch.float32);  view_564 = None
        add_205: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1067, convert_element_type_1075);  convert_element_type_1067 = convert_element_type_1075 = None
        convert_element_type_1076: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_13, torch.float32);  mm_13 = None
        convert_element_type_1077: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1074, torch.float32);  convert_element_type_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_103: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        view_565: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_103, [16, 512, 1024]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_566: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_565, [8192, 1024]);  view_565 = None
        mm_14: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
        permute_304: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_15: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
        sum_49: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_566, [0], True, dtype = torch.float32);  view_566 = None
        view_567: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_49, [1024]);  sum_49 = None
        convert_element_type_1082: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.bfloat16);  view_567 = None
        view_568: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_14, [16, 512, 1024]);  mm_14 = None
        convert_element_type_1083: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_568, torch.float32);  view_568 = None
        add_206: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, convert_element_type_1083);  add_205 = convert_element_type_1083 = None
        convert_element_type_1084: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_1085: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1082, torch.float32);  convert_element_type_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_367: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, primals_375);  primals_375 = None
        mul_368: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, 1024)
        sum_50: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
        mul_369: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_367, mul_302);  mul_367 = None
        sum_51: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
        mul_370: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_302, sum_51);  sum_51 = None
        sub_88: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_368, sum_50);  mul_368 = sum_50 = None
        sub_89: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_88, mul_370);  sub_88 = mul_370 = None
        mul_371: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_89);  div_54 = sub_89 = None
        mul_372: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_206, mul_302);  mul_302 = None
        sum_52: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
        sum_53: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None
        add_207: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_204, mul_371);  add_204 = mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1086: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1087: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_69, torch.bfloat16);  gt_69 = None
        mul_373: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1087, 1.1111111111111112);  convert_element_type_1087 = None
        mul_374: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1086, mul_373);  convert_element_type_1086 = mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_569: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_374, [8192, 1024]);  mul_374 = None
        mm_16: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
        permute_308: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_569, [1, 0])
        mm_17: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
        sum_54: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_569, [0], True, dtype = torch.float32);  view_569 = None
        view_570: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_54, [1024]);  sum_54 = None
        convert_element_type_1092: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_570, torch.bfloat16);  view_570 = None
        view_571: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [16, 512, 4096]);  mm_16 = None
        convert_element_type_1093: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_1094: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1092, torch.float32);  convert_element_type_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1095: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_571, torch.float32);  view_571 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_136, [16, 512, 4096]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_936: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_503, torch.float32);  view_503 = None
        mul_298: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, 0.7071067811865476)
        erf_22: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
        add_184: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_376: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
        mul_377: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, convert_element_type_936)
        mul_378: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
        exp_28: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
        mul_379: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_380: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_936, mul_379);  convert_element_type_936 = mul_379 = None
        add_209: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
        mul_381: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1095, add_209);  convert_element_type_1095 = add_209 = None
        convert_element_type_1097: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_381, torch.bfloat16);  mul_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1097, [8192, 4096]);  convert_element_type_1097 = None
        mm_18: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
        permute_312: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_19: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
        sum_55: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_572, [0], True, dtype = torch.float32);  view_572 = None
        view_573: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_55, [4096]);  sum_55 = None
        convert_element_type_1102: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_573, torch.bfloat16);  view_573 = None
        view_574: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [16, 512, 1024]);  mm_18 = None
        convert_element_type_1103: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_574, torch.float32);  view_574 = None
        convert_element_type_1104: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_1105: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1102, torch.float32);  convert_element_type_1102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_383: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1103, primals_369);  primals_369 = None
        mul_384: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_383, 1024)
        sum_56: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
        mul_385: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_383, mul_295);  mul_383 = None
        sum_57: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
        mul_386: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, sum_57);  sum_57 = None
        sub_91: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_384, sum_56);  mul_384 = sum_56 = None
        sub_92: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, mul_386);  sub_91 = mul_386 = None
        mul_387: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_92);  div_55 = sub_92 = None
        mul_388: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1103, mul_295);  mul_295 = None
        sum_58: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
        sum_59: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1103, [0, 1]);  convert_element_type_1103 = None
        add_210: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_207, mul_387);  add_207 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1106: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1107: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_68, torch.bfloat16);  gt_68 = None
        mul_389: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1107, 1.1111111111111112);  convert_element_type_1107 = None
        mul_390: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1106, mul_389);  convert_element_type_1106 = mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_575: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_390, [8192, 1024]);  mul_390 = None
        mm_20: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
        permute_316: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_575, [1, 0])
        mm_21: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
        sum_60: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_575, [0], True, dtype = torch.float32);  view_575 = None
        view_576: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_60, [1024]);  sum_60 = None
        convert_element_type_1112: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_576, torch.bfloat16);  view_576 = None
        view_577: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [16, 512, 1024]);  mm_20 = None
        convert_element_type_1113: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_1114: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1112, torch.float32);  convert_element_type_1112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_578: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_577, [16, 512, 16, 64]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_319: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_106: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_106, permute_default_6, permute_default_7, permute_default_8, getitem_107, getitem_108, None, None, 512, 512, 0.1, False, getitem_109, getitem_110, scale = 0.125);  clone_106 = permute_default_6 = permute_default_7 = permute_default_8 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = None
        getitem_111: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_1[0]
        getitem_112: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_1[1]
        getitem_113: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_1[2];  _scaled_dot_product_flash_attention_backward_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_11: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_10: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_9: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
        clone_108: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_585: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_108, [16, 512, 1024]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_586: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_585, [8192, 1024]);  view_585 = None
        mm_22: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_586, permute_326);  permute_326 = None
        permute_327: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_23: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        sum_62: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_586, [0], True, dtype = torch.float32);  view_586 = None
        view_587: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_62, [1024]);  sum_62 = None
        convert_element_type_1130: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.bfloat16);  view_587 = None
        view_588: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [16, 512, 1024]);  mm_22 = None
        convert_element_type_1131: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_588, torch.float32);  view_588 = None
        convert_element_type_1132: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_1133: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1130, torch.float32);  convert_element_type_1130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_589: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_10, [16, 512, 1024]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_109: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_589, memory_format = torch.contiguous_format);  view_589 = None
        view_590: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_109, [8192, 1024]);  clone_109 = None
        mm_24: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_331);  permute_331 = None
        permute_332: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_25: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_332, view_484);  permute_332 = None
        sum_63: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_590, [0], True, dtype = torch.float32);  view_590 = None
        view_591: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [1024]);  sum_63 = None
        convert_element_type_1138: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.bfloat16);  view_591 = None
        view_592: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [16, 512, 1024]);  mm_24 = None
        convert_element_type_1139: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_592, torch.float32);  view_592 = None
        add_211: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1131, convert_element_type_1139);  convert_element_type_1131 = convert_element_type_1139 = None
        convert_element_type_1140: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_1141: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1138, torch.float32);  convert_element_type_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_110: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_11, memory_format = torch.contiguous_format);  permute_default_11 = None
        view_593: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_110, [16, 512, 1024]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_594: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [8192, 1024]);  view_593 = None
        mm_26: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
        permute_337: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_27: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
        sum_64: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_594, [0], True, dtype = torch.float32);  view_594 = None
        view_595: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [1024]);  sum_64 = None
        convert_element_type_1146: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_595, torch.bfloat16);  view_595 = None
        view_596: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [16, 512, 1024]);  mm_26 = None
        convert_element_type_1147: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_596, torch.float32);  view_596 = None
        add_212: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_211, convert_element_type_1147);  add_211 = convert_element_type_1147 = None
        convert_element_type_1148: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_1149: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1146, torch.float32);  convert_element_type_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_395: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, primals_359);  primals_359 = None
        mul_396: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, 1024)
        sum_65: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_395, mul_289);  mul_395 = None
        sum_66: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_289, sum_66);  sum_66 = None
        sub_94: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_396, sum_65);  mul_396 = sum_65 = None
        sub_95: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, mul_398);  sub_94 = mul_398 = None
        mul_399: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_57, sub_95);  div_57 = sub_95 = None
        mul_400: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, mul_289);  mul_289 = None
        sum_67: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_68: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
        add_213: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_210, mul_399);  add_210 = mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1150: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_213, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1151: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_66, torch.bfloat16);  gt_66 = None
        mul_401: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1151, 1.1111111111111112);  convert_element_type_1151 = None
        mul_402: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1150, mul_401);  convert_element_type_1150 = mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_597: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_402, [8192, 1024]);  mul_402 = None
        mm_28: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
        permute_341: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_597, [1, 0])
        mm_29: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
        sum_69: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_597, [0], True, dtype = torch.float32);  view_597 = None
        view_598: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [1024]);  sum_69 = None
        convert_element_type_1156: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_598, torch.bfloat16);  view_598 = None
        view_599: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [16, 512, 4096]);  mm_28 = None
        convert_element_type_1157: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_1158: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1156, torch.float32);  convert_element_type_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1159: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_599, torch.float32);  view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_130, [16, 512, 4096]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_895: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_481, torch.float32);  view_481 = None
        mul_285: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, 0.7071067811865476)
        erf_21: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
        add_176: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_404: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
        mul_405: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, convert_element_type_895)
        mul_406: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_405, -0.5);  mul_405 = None
        exp_29: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_406);  mul_406 = None
        mul_407: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_408: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_895, mul_407);  convert_element_type_895 = mul_407 = None
        add_215: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_404, mul_408);  mul_404 = mul_408 = None
        mul_409: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1159, add_215);  convert_element_type_1159 = add_215 = None
        convert_element_type_1161: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_409, torch.bfloat16);  mul_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_600: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1161, [8192, 4096]);  convert_element_type_1161 = None
        mm_30: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
        permute_345: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_31: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
        sum_70: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_600, [0], True, dtype = torch.float32);  view_600 = None
        view_601: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [4096]);  sum_70 = None
        convert_element_type_1166: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_601, torch.bfloat16);  view_601 = None
        view_602: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [16, 512, 1024]);  mm_30 = None
        convert_element_type_1167: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_602, torch.float32);  view_602 = None
        convert_element_type_1168: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_1169: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1166, torch.float32);  convert_element_type_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_411: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1167, primals_353);  primals_353 = None
        mul_412: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, 1024)
        sum_71: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
        mul_413: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_411, mul_282);  mul_411 = None
        sum_72: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        mul_414: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_282, sum_72);  sum_72 = None
        sub_97: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_412, sum_71);  mul_412 = sum_71 = None
        sub_98: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_97, mul_414);  sub_97 = mul_414 = None
        mul_415: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_98);  div_58 = sub_98 = None
        mul_416: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1167, mul_282);  mul_282 = None
        sum_73: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
        sum_74: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1167, [0, 1]);  convert_element_type_1167 = None
        add_216: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_213, mul_415);  add_213 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1170: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1171: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_65, torch.bfloat16);  gt_65 = None
        mul_417: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1171, 1.1111111111111112);  convert_element_type_1171 = None
        mul_418: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1170, mul_417);  convert_element_type_1170 = mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_603: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_418, [8192, 1024]);  mul_418 = None
        mm_32: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
        permute_349: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_603, [1, 0])
        mm_33: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
        sum_75: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_603, [0], True, dtype = torch.float32);  view_603 = None
        view_604: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [1024]);  sum_75 = None
        convert_element_type_1176: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_604, torch.bfloat16);  view_604 = None
        view_605: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [16, 512, 1024]);  mm_32 = None
        convert_element_type_1177: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_1178: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1176, torch.float32);  convert_element_type_1176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_606: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_605, [16, 512, 16, 64]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_352: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_113: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_113, permute_default_12, permute_default_13, permute_default_14, getitem_114, getitem_115, None, None, 512, 512, 0.1, False, getitem_116, getitem_117, scale = 0.125);  clone_113 = permute_default_12 = permute_default_13 = permute_default_14 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = None
        getitem_118: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_2[0]
        getitem_119: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_2[1]
        getitem_120: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_2[2];  _scaled_dot_product_flash_attention_backward_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_17: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_16: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_15: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        clone_115: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        view_613: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_115, [16, 512, 1024]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_614: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_613, [8192, 1024]);  view_613 = None
        mm_34: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_614, permute_359);  permute_359 = None
        permute_360: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_35: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        sum_77: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_614, [0], True, dtype = torch.float32);  view_614 = None
        view_615: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_77, [1024]);  sum_77 = None
        convert_element_type_1194: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_615, torch.bfloat16);  view_615 = None
        view_616: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [16, 512, 1024]);  mm_34 = None
        convert_element_type_1195: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_616, torch.float32);  view_616 = None
        convert_element_type_1196: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_1197: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1194, torch.float32);  convert_element_type_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_617: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_16, [16, 512, 1024]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_116: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_617, memory_format = torch.contiguous_format);  view_617 = None
        view_618: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_116, [8192, 1024]);  clone_116 = None
        mm_36: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_618, permute_364);  permute_364 = None
        permute_365: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_37: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_365, view_462);  permute_365 = None
        sum_78: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_618, [0], True, dtype = torch.float32);  view_618 = None
        view_619: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_78, [1024]);  sum_78 = None
        convert_element_type_1202: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_619, torch.bfloat16);  view_619 = None
        view_620: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [16, 512, 1024]);  mm_36 = None
        convert_element_type_1203: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_620, torch.float32);  view_620 = None
        add_217: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1195, convert_element_type_1203);  convert_element_type_1195 = convert_element_type_1203 = None
        convert_element_type_1204: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_1205: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1202, torch.float32);  convert_element_type_1202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_117: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_17, memory_format = torch.contiguous_format);  permute_default_17 = None
        view_621: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_117, [16, 512, 1024]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_622: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_621, [8192, 1024]);  view_621 = None
        mm_38: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
        permute_370: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_39: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
        sum_79: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_622, [0], True, dtype = torch.float32);  view_622 = None
        view_623: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_79, [1024]);  sum_79 = None
        convert_element_type_1210: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_623, torch.bfloat16);  view_623 = None
        view_624: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [16, 512, 1024]);  mm_38 = None
        convert_element_type_1211: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_624, torch.float32);  view_624 = None
        add_218: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_217, convert_element_type_1211);  add_217 = convert_element_type_1211 = None
        convert_element_type_1212: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_1213: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1210, torch.float32);  convert_element_type_1210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_423: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_218, primals_343);  primals_343 = None
        mul_424: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, 1024)
        sum_80: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True)
        mul_425: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_423, mul_276);  mul_423 = None
        sum_81: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
        mul_426: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, sum_81);  sum_81 = None
        sub_100: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_424, sum_80);  mul_424 = sum_80 = None
        sub_101: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_100, mul_426);  sub_100 = mul_426 = None
        mul_427: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_101);  div_60 = sub_101 = None
        mul_428: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_218, mul_276);  mul_276 = None
        sum_82: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
        sum_83: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
        add_219: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_216, mul_427);  add_216 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1214: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1215: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_63, torch.bfloat16);  gt_63 = None
        mul_429: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1215, 1.1111111111111112);  convert_element_type_1215 = None
        mul_430: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1214, mul_429);  convert_element_type_1214 = mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_625: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_430, [8192, 1024]);  mul_430 = None
        mm_40: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
        permute_374: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_41: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
        sum_84: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_625, [0], True, dtype = torch.float32);  view_625 = None
        view_626: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_84, [1024]);  sum_84 = None
        convert_element_type_1220: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_626, torch.bfloat16);  view_626 = None
        view_627: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [16, 512, 4096]);  mm_40 = None
        convert_element_type_1221: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_1222: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1220, torch.float32);  convert_element_type_1220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1223: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.float32);  view_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_459: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_124, [16, 512, 4096]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_854: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_459, torch.float32);  view_459 = None
        mul_272: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, 0.7071067811865476)
        erf_20: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_272);  mul_272 = None
        add_168: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_432: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
        mul_433: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, convert_element_type_854)
        mul_434: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_433, -0.5);  mul_433 = None
        exp_30: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_434);  mul_434 = None
        mul_435: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_436: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_854, mul_435);  convert_element_type_854 = mul_435 = None
        add_221: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_432, mul_436);  mul_432 = mul_436 = None
        mul_437: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1223, add_221);  convert_element_type_1223 = add_221 = None
        convert_element_type_1225: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_437, torch.bfloat16);  mul_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1225, [8192, 4096]);  convert_element_type_1225 = None
        mm_42: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
        permute_378: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_43: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
        sum_85: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_628, [0], True, dtype = torch.float32);  view_628 = None
        view_629: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_85, [4096]);  sum_85 = None
        convert_element_type_1230: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_629, torch.bfloat16);  view_629 = None
        view_630: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [16, 512, 1024]);  mm_42 = None
        convert_element_type_1231: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_630, torch.float32);  view_630 = None
        convert_element_type_1232: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_1233: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1230, torch.float32);  convert_element_type_1230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_439: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1231, primals_337);  primals_337 = None
        mul_440: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_439, 1024)
        sum_86: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True)
        mul_441: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_439, mul_269);  mul_439 = None
        sum_87: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True);  mul_441 = None
        mul_442: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, sum_87);  sum_87 = None
        sub_103: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_440, sum_86);  mul_440 = sum_86 = None
        sub_104: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_103, mul_442);  sub_103 = mul_442 = None
        mul_443: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_61, sub_104);  div_61 = sub_104 = None
        mul_444: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1231, mul_269);  mul_269 = None
        sum_88: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1]);  mul_444 = None
        sum_89: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1231, [0, 1]);  convert_element_type_1231 = None
        add_222: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_219, mul_443);  add_219 = mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1234: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_222, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1235: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_62, torch.bfloat16);  gt_62 = None
        mul_445: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1235, 1.1111111111111112);  convert_element_type_1235 = None
        mul_446: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1234, mul_445);  convert_element_type_1234 = mul_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_631: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_446, [8192, 1024]);  mul_446 = None
        mm_44: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
        permute_382: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_45: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
        sum_90: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_631, [0], True, dtype = torch.float32);  view_631 = None
        view_632: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_90, [1024]);  sum_90 = None
        convert_element_type_1240: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.bfloat16);  view_632 = None
        view_633: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [16, 512, 1024]);  mm_44 = None
        convert_element_type_1241: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_1242: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1240, torch.float32);  convert_element_type_1240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_634: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_633, [16, 512, 16, 64]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_385: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_120: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_120, permute_default_18, permute_default_19, permute_default_20, getitem_121, getitem_122, None, None, 512, 512, 0.1, False, getitem_123, getitem_124, scale = 0.125);  clone_120 = permute_default_18 = permute_default_19 = permute_default_20 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = None
        getitem_125: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_3[0]
        getitem_126: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_3[1]
        getitem_127: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_3[2];  _scaled_dot_product_flash_attention_backward_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_23: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_125, [0, 2, 1, 3]);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_22: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_21: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None
        clone_122: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_21, memory_format = torch.contiguous_format);  permute_default_21 = None
        view_641: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_122, [16, 512, 1024]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_642: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_641, [8192, 1024]);  view_641 = None
        mm_46: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_642, permute_392);  permute_392 = None
        permute_393: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_47: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        sum_92: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_642, [0], True, dtype = torch.float32);  view_642 = None
        view_643: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_92, [1024]);  sum_92 = None
        convert_element_type_1258: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_643, torch.bfloat16);  view_643 = None
        view_644: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [16, 512, 1024]);  mm_46 = None
        convert_element_type_1259: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_644, torch.float32);  view_644 = None
        convert_element_type_1260: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_1261: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1258, torch.float32);  convert_element_type_1258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_645: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_22, [16, 512, 1024]);  permute_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_123: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_645, memory_format = torch.contiguous_format);  view_645 = None
        view_646: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_123, [8192, 1024]);  clone_123 = None
        mm_48: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_646, permute_397);  permute_397 = None
        permute_398: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_49: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_398, view_440);  permute_398 = None
        sum_93: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_646, [0], True, dtype = torch.float32);  view_646 = None
        view_647: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [1024]);  sum_93 = None
        convert_element_type_1266: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_647, torch.bfloat16);  view_647 = None
        view_648: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [16, 512, 1024]);  mm_48 = None
        convert_element_type_1267: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_648, torch.float32);  view_648 = None
        add_223: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1259, convert_element_type_1267);  convert_element_type_1259 = convert_element_type_1267 = None
        convert_element_type_1268: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_1269: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1266, torch.float32);  convert_element_type_1266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_124: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_23, memory_format = torch.contiguous_format);  permute_default_23 = None
        view_649: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_124, [16, 512, 1024]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_650: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_649, [8192, 1024]);  view_649 = None
        mm_50: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
        permute_403: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_51: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
        sum_94: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_650, [0], True, dtype = torch.float32);  view_650 = None
        view_651: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [1024]);  sum_94 = None
        convert_element_type_1274: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.bfloat16);  view_651 = None
        view_652: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [16, 512, 1024]);  mm_50 = None
        convert_element_type_1275: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        add_224: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_223, convert_element_type_1275);  add_223 = convert_element_type_1275 = None
        convert_element_type_1276: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_1277: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1274, torch.float32);  convert_element_type_1274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_451: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, primals_327);  primals_327 = None
        mul_452: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, 1024)
        sum_95: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True)
        mul_453: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_451, mul_263);  mul_451 = None
        sum_96: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
        mul_454: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, sum_96);  sum_96 = None
        sub_106: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_452, sum_95);  mul_452 = sum_95 = None
        sub_107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_106, mul_454);  sub_106 = mul_454 = None
        mul_455: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_63, sub_107);  div_63 = sub_107 = None
        mul_456: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_224, mul_263);  mul_263 = None
        sum_97: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1]);  mul_456 = None
        sum_98: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
        add_225: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_222, mul_455);  add_222 = mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1278: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_225, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1279: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_60, torch.bfloat16);  gt_60 = None
        mul_457: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1279, 1.1111111111111112);  convert_element_type_1279 = None
        mul_458: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1278, mul_457);  convert_element_type_1278 = mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_653: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_458, [8192, 1024]);  mul_458 = None
        mm_52: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
        permute_407: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_53: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
        sum_99: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_653, [0], True, dtype = torch.float32);  view_653 = None
        view_654: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [1024]);  sum_99 = None
        convert_element_type_1284: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_654, torch.bfloat16);  view_654 = None
        view_655: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [16, 512, 4096]);  mm_52 = None
        convert_element_type_1285: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_1286: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1284, torch.float32);  convert_element_type_1284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1287: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_655, torch.float32);  view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_437: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_118, [16, 512, 4096]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_813: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_437, torch.float32);  view_437 = None
        mul_259: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_813, 0.7071067811865476)
        erf_19: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_259);  mul_259 = None
        add_160: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_460: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
        mul_461: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_813, convert_element_type_813)
        mul_462: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
        exp_31: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
        mul_463: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_464: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_813, mul_463);  convert_element_type_813 = mul_463 = None
        add_227: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
        mul_465: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1287, add_227);  convert_element_type_1287 = add_227 = None
        convert_element_type_1289: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_465, torch.bfloat16);  mul_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_656: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1289, [8192, 4096]);  convert_element_type_1289 = None
        mm_54: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
        permute_411: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_55: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
        sum_100: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_656, [0], True, dtype = torch.float32);  view_656 = None
        view_657: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [4096]);  sum_100 = None
        convert_element_type_1294: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_657, torch.bfloat16);  view_657 = None
        view_658: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [16, 512, 1024]);  mm_54 = None
        convert_element_type_1295: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_658, torch.float32);  view_658 = None
        convert_element_type_1296: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_1297: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1294, torch.float32);  convert_element_type_1294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_467: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1295, primals_321);  primals_321 = None
        mul_468: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, 1024)
        sum_101: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
        mul_469: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_467, mul_256);  mul_467 = None
        sum_102: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
        mul_470: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, sum_102);  sum_102 = None
        sub_109: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_468, sum_101);  mul_468 = sum_101 = None
        sub_110: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_109, mul_470);  sub_109 = mul_470 = None
        mul_471: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_64, sub_110);  div_64 = sub_110 = None
        mul_472: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1295, mul_256);  mul_256 = None
        sum_103: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
        sum_104: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1295, [0, 1]);  convert_element_type_1295 = None
        add_228: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_225, mul_471);  add_225 = mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1298: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1299: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_59, torch.bfloat16);  gt_59 = None
        mul_473: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1299, 1.1111111111111112);  convert_element_type_1299 = None
        mul_474: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1298, mul_473);  convert_element_type_1298 = mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_659: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_474, [8192, 1024]);  mul_474 = None
        mm_56: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
        permute_415: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_57: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
        sum_105: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_659, [0], True, dtype = torch.float32);  view_659 = None
        view_660: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [1024]);  sum_105 = None
        convert_element_type_1304: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_660, torch.bfloat16);  view_660 = None
        view_661: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [16, 512, 1024]);  mm_56 = None
        convert_element_type_1305: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_1306: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1304, torch.float32);  convert_element_type_1304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_662: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_661, [16, 512, 16, 64]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_418: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_127: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_127, permute_default_24, permute_default_25, permute_default_26, getitem_128, getitem_129, None, None, 512, 512, 0.1, False, getitem_130, getitem_131, scale = 0.125);  clone_127 = permute_default_24 = permute_default_25 = permute_default_26 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = None
        getitem_132: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_4[0]
        getitem_133: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_4[1]
        getitem_134: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_4[2];  _scaled_dot_product_flash_attention_backward_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_29: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_28: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_27: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        clone_129: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_27, memory_format = torch.contiguous_format);  permute_default_27 = None
        view_669: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_129, [16, 512, 1024]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_670: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [8192, 1024]);  view_669 = None
        mm_58: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_670, permute_425);  permute_425 = None
        permute_426: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_59: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        sum_107: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_670, [0], True, dtype = torch.float32);  view_670 = None
        view_671: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_107, [1024]);  sum_107 = None
        convert_element_type_1322: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_671, torch.bfloat16);  view_671 = None
        view_672: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [16, 512, 1024]);  mm_58 = None
        convert_element_type_1323: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_672, torch.float32);  view_672 = None
        convert_element_type_1324: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_1325: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1322, torch.float32);  convert_element_type_1322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_673: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_28, [16, 512, 1024]);  permute_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_130: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_673, memory_format = torch.contiguous_format);  view_673 = None
        view_674: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_130, [8192, 1024]);  clone_130 = None
        mm_60: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_674, permute_430);  permute_430 = None
        permute_431: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_674, [1, 0])
        mm_61: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_431, view_418);  permute_431 = None
        sum_108: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_674, [0], True, dtype = torch.float32);  view_674 = None
        view_675: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_108, [1024]);  sum_108 = None
        convert_element_type_1330: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_675, torch.bfloat16);  view_675 = None
        view_676: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [16, 512, 1024]);  mm_60 = None
        convert_element_type_1331: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_676, torch.float32);  view_676 = None
        add_229: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1323, convert_element_type_1331);  convert_element_type_1323 = convert_element_type_1331 = None
        convert_element_type_1332: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_1333: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1330, torch.float32);  convert_element_type_1330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_131: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_29, memory_format = torch.contiguous_format);  permute_default_29 = None
        view_677: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_131, [16, 512, 1024]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_678: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_677, [8192, 1024]);  view_677 = None
        mm_62: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
        permute_436: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_63: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
        sum_109: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_678, [0], True, dtype = torch.float32);  view_678 = None
        view_679: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_109, [1024]);  sum_109 = None
        convert_element_type_1338: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_679, torch.bfloat16);  view_679 = None
        view_680: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [16, 512, 1024]);  mm_62 = None
        convert_element_type_1339: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_680, torch.float32);  view_680 = None
        add_230: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_229, convert_element_type_1339);  add_229 = convert_element_type_1339 = None
        convert_element_type_1340: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_1341: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1338, torch.float32);  convert_element_type_1338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_479: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_230, primals_311);  primals_311 = None
        mul_480: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, 1024)
        sum_110: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True)
        mul_481: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_479, mul_250);  mul_479 = None
        sum_111: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True);  mul_481 = None
        mul_482: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, sum_111);  sum_111 = None
        sub_112: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_480, sum_110);  mul_480 = sum_110 = None
        sub_113: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, mul_482);  sub_112 = mul_482 = None
        mul_483: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_66, sub_113);  div_66 = sub_113 = None
        mul_484: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_230, mul_250);  mul_250 = None
        sum_112: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 1]);  mul_484 = None
        sum_113: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
        add_231: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, mul_483);  add_228 = mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1342: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1343: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_57, torch.bfloat16);  gt_57 = None
        mul_485: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1343, 1.1111111111111112);  convert_element_type_1343 = None
        mul_486: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1342, mul_485);  convert_element_type_1342 = mul_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_681: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_486, [8192, 1024]);  mul_486 = None
        mm_64: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
        permute_440: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_65: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
        sum_114: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_681, [0], True, dtype = torch.float32);  view_681 = None
        view_682: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_114, [1024]);  sum_114 = None
        convert_element_type_1348: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_682, torch.bfloat16);  view_682 = None
        view_683: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [16, 512, 4096]);  mm_64 = None
        convert_element_type_1349: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_1350: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1348, torch.float32);  convert_element_type_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1351: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_683, torch.float32);  view_683 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_415: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_112, [16, 512, 4096]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_772: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_246: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, 0.7071067811865476)
        erf_18: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_152: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_488: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
        mul_489: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, convert_element_type_772)
        mul_490: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_489, -0.5);  mul_489 = None
        exp_32: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_490);  mul_490 = None
        mul_491: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_492: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_772, mul_491);  convert_element_type_772 = mul_491 = None
        add_233: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_488, mul_492);  mul_488 = mul_492 = None
        mul_493: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1351, add_233);  convert_element_type_1351 = add_233 = None
        convert_element_type_1353: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_493, torch.bfloat16);  mul_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_684: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1353, [8192, 4096]);  convert_element_type_1353 = None
        mm_66: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
        permute_444: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_684, [1, 0])
        mm_67: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
        sum_115: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_684, [0], True, dtype = torch.float32);  view_684 = None
        view_685: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_115, [4096]);  sum_115 = None
        convert_element_type_1358: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_685, torch.bfloat16);  view_685 = None
        view_686: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [16, 512, 1024]);  mm_66 = None
        convert_element_type_1359: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_686, torch.float32);  view_686 = None
        convert_element_type_1360: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_1361: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1358, torch.float32);  convert_element_type_1358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_495: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1359, primals_305);  primals_305 = None
        mul_496: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, 1024)
        sum_116: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_495, mul_243);  mul_495 = None
        sum_117: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, sum_117);  sum_117 = None
        sub_115: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_496, sum_116);  mul_496 = sum_116 = None
        sub_116: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, mul_498);  sub_115 = mul_498 = None
        mul_499: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_67, sub_116);  div_67 = sub_116 = None
        mul_500: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1359, mul_243);  mul_243 = None
        sum_118: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_119: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1359, [0, 1]);  convert_element_type_1359 = None
        add_234: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_231, mul_499);  add_231 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1362: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1363: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_56, torch.bfloat16);  gt_56 = None
        mul_501: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1363, 1.1111111111111112);  convert_element_type_1363 = None
        mul_502: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1362, mul_501);  convert_element_type_1362 = mul_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_502, [8192, 1024]);  mul_502 = None
        mm_68: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
        permute_448: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_69: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
        sum_120: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_687, [0], True, dtype = torch.float32);  view_687 = None
        view_688: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_120, [1024]);  sum_120 = None
        convert_element_type_1368: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_688, torch.bfloat16);  view_688 = None
        view_689: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [16, 512, 1024]);  mm_68 = None
        convert_element_type_1369: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_1370: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1368, torch.float32);  convert_element_type_1368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_690: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_689, [16, 512, 16, 64]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_451: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_134: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_134, permute_default_30, permute_default_31, permute_default_32, getitem_135, getitem_136, None, None, 512, 512, 0.1, False, getitem_137, getitem_138, scale = 0.125);  clone_134 = permute_default_30 = permute_default_31 = permute_default_32 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = None
        getitem_139: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_5[0]
        getitem_140: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_5[1]
        getitem_141: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_5[2];  _scaled_dot_product_flash_attention_backward_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_35: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_34: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_33: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1, 3]);  getitem_141 = None
        clone_136: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_33, memory_format = torch.contiguous_format);  permute_default_33 = None
        view_697: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [16, 512, 1024]);  clone_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_698: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_697, [8192, 1024]);  view_697 = None
        mm_70: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_698, permute_458);  permute_458 = None
        permute_459: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_698, [1, 0])
        mm_71: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        sum_122: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_698, [0], True, dtype = torch.float32);  view_698 = None
        view_699: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_122, [1024]);  sum_122 = None
        convert_element_type_1386: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_699, torch.bfloat16);  view_699 = None
        view_700: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [16, 512, 1024]);  mm_70 = None
        convert_element_type_1387: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_700, torch.float32);  view_700 = None
        convert_element_type_1388: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_1389: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1386, torch.float32);  convert_element_type_1386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_701: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_34, [16, 512, 1024]);  permute_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_137: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_701, memory_format = torch.contiguous_format);  view_701 = None
        view_702: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [8192, 1024]);  clone_137 = None
        mm_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_702, permute_463);  permute_463 = None
        permute_464: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_702, [1, 0])
        mm_73: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_464, view_396);  permute_464 = None
        sum_123: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_702, [0], True, dtype = torch.float32);  view_702 = None
        view_703: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [1024]);  sum_123 = None
        convert_element_type_1394: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_703, torch.bfloat16);  view_703 = None
        view_704: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [16, 512, 1024]);  mm_72 = None
        convert_element_type_1395: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_704, torch.float32);  view_704 = None
        add_235: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1387, convert_element_type_1395);  convert_element_type_1387 = convert_element_type_1395 = None
        convert_element_type_1396: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_1397: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1394, torch.float32);  convert_element_type_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_138: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_35, memory_format = torch.contiguous_format);  permute_default_35 = None
        view_705: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [16, 512, 1024]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_706: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_705, [8192, 1024]);  view_705 = None
        mm_74: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
        permute_469: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_75: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
        sum_124: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_706, [0], True, dtype = torch.float32);  view_706 = None
        view_707: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [1024]);  sum_124 = None
        convert_element_type_1402: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_707, torch.bfloat16);  view_707 = None
        view_708: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [16, 512, 1024]);  mm_74 = None
        convert_element_type_1403: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_708, torch.float32);  view_708 = None
        add_236: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, convert_element_type_1403);  add_235 = convert_element_type_1403 = None
        convert_element_type_1404: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_1405: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1402, torch.float32);  convert_element_type_1402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_507: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_236, primals_295);  primals_295 = None
        mul_508: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, 1024)
        sum_125: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True)
        mul_509: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_507, mul_237);  mul_507 = None
        sum_126: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True);  mul_509 = None
        mul_510: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, sum_126);  sum_126 = None
        sub_118: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_508, sum_125);  mul_508 = sum_125 = None
        sub_119: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_118, mul_510);  sub_118 = mul_510 = None
        mul_511: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_69, sub_119);  div_69 = sub_119 = None
        mul_512: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_236, mul_237);  mul_237 = None
        sum_127: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 1]);  mul_512 = None
        sum_128: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
        add_237: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_234, mul_511);  add_234 = mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1406: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1407: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_54, torch.bfloat16);  gt_54 = None
        mul_513: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1407, 1.1111111111111112);  convert_element_type_1407 = None
        mul_514: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1406, mul_513);  convert_element_type_1406 = mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_709: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_514, [8192, 1024]);  mul_514 = None
        mm_76: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
        permute_473: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_77: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
        sum_129: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_709, [0], True, dtype = torch.float32);  view_709 = None
        view_710: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [1024]);  sum_129 = None
        convert_element_type_1412: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_710, torch.bfloat16);  view_710 = None
        view_711: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [16, 512, 4096]);  mm_76 = None
        convert_element_type_1413: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_1414: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1412, torch.float32);  convert_element_type_1412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1415: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_106, [16, 512, 4096]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_731: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_393, torch.float32);  view_393 = None
        mul_233: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, 0.7071067811865476)
        erf_17: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_144: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_516: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
        mul_517: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, convert_element_type_731)
        mul_518: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_517, -0.5);  mul_517 = None
        exp_33: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_518);  mul_518 = None
        mul_519: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_520: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, mul_519);  convert_element_type_731 = mul_519 = None
        add_239: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_516, mul_520);  mul_516 = mul_520 = None
        mul_521: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1415, add_239);  convert_element_type_1415 = add_239 = None
        convert_element_type_1417: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_521, torch.bfloat16);  mul_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1417, [8192, 4096]);  convert_element_type_1417 = None
        mm_78: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
        permute_477: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_79: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
        sum_130: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_712, [0], True, dtype = torch.float32);  view_712 = None
        view_713: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [4096]);  sum_130 = None
        convert_element_type_1422: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_713, torch.bfloat16);  view_713 = None
        view_714: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [16, 512, 1024]);  mm_78 = None
        convert_element_type_1423: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_714, torch.float32);  view_714 = None
        convert_element_type_1424: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_1425: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1422, torch.float32);  convert_element_type_1422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_523: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1423, primals_289);  primals_289 = None
        mul_524: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_523, 1024)
        sum_131: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
        mul_525: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_523, mul_230);  mul_523 = None
        sum_132: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
        mul_526: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, sum_132);  sum_132 = None
        sub_121: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_524, sum_131);  mul_524 = sum_131 = None
        sub_122: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_121, mul_526);  sub_121 = mul_526 = None
        mul_527: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_70, sub_122);  div_70 = sub_122 = None
        mul_528: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1423, mul_230);  mul_230 = None
        sum_133: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
        sum_134: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1423, [0, 1]);  convert_element_type_1423 = None
        add_240: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_237, mul_527);  add_237 = mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1426: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1427: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_53, torch.bfloat16);  gt_53 = None
        mul_529: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1427, 1.1111111111111112);  convert_element_type_1427 = None
        mul_530: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1426, mul_529);  convert_element_type_1426 = mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_715: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_530, [8192, 1024]);  mul_530 = None
        mm_80: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
        permute_481: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_81: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
        sum_135: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_715, [0], True, dtype = torch.float32);  view_715 = None
        view_716: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [1024]);  sum_135 = None
        convert_element_type_1432: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_716, torch.bfloat16);  view_716 = None
        view_717: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [16, 512, 1024]);  mm_80 = None
        convert_element_type_1433: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_1434: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1432, torch.float32);  convert_element_type_1432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_718: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_717, [16, 512, 16, 64]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_484: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_141: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_141, permute_default_36, permute_default_37, permute_default_38, getitem_142, getitem_143, None, None, 512, 512, 0.1, False, getitem_144, getitem_145, scale = 0.125);  clone_141 = permute_default_36 = permute_default_37 = permute_default_38 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = None
        getitem_146: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_6[0]
        getitem_147: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_6[1]
        getitem_148: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_6[2];  _scaled_dot_product_flash_attention_backward_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_41: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_40: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_39: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        clone_143: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_39, memory_format = torch.contiguous_format);  permute_default_39 = None
        view_725: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [16, 512, 1024]);  clone_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_726: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_725, [8192, 1024]);  view_725 = None
        mm_82: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_726, permute_491);  permute_491 = None
        permute_492: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_726, [1, 0])
        mm_83: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        sum_137: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_726, [0], True, dtype = torch.float32);  view_726 = None
        view_727: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_137, [1024]);  sum_137 = None
        convert_element_type_1450: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_727, torch.bfloat16);  view_727 = None
        view_728: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [16, 512, 1024]);  mm_82 = None
        convert_element_type_1451: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_728, torch.float32);  view_728 = None
        convert_element_type_1452: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_1453: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1450, torch.float32);  convert_element_type_1450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_729: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_40, [16, 512, 1024]);  permute_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_144: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_730: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [8192, 1024]);  clone_144 = None
        mm_84: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_730, permute_496);  permute_496 = None
        permute_497: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_730, [1, 0])
        mm_85: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_497, view_374);  permute_497 = None
        sum_138: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_730, [0], True, dtype = torch.float32);  view_730 = None
        view_731: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_138, [1024]);  sum_138 = None
        convert_element_type_1458: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_731, torch.bfloat16);  view_731 = None
        view_732: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [16, 512, 1024]);  mm_84 = None
        convert_element_type_1459: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_732, torch.float32);  view_732 = None
        add_241: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1451, convert_element_type_1459);  convert_element_type_1451 = convert_element_type_1459 = None
        convert_element_type_1460: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_1461: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1458, torch.float32);  convert_element_type_1458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_145: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_41, memory_format = torch.contiguous_format);  permute_default_41 = None
        view_733: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [16, 512, 1024]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_734: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_733, [8192, 1024]);  view_733 = None
        mm_86: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
        permute_502: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_87: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
        sum_139: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_734, [0], True, dtype = torch.float32);  view_734 = None
        view_735: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_139, [1024]);  sum_139 = None
        convert_element_type_1466: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_735, torch.bfloat16);  view_735 = None
        view_736: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [16, 512, 1024]);  mm_86 = None
        convert_element_type_1467: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_736, torch.float32);  view_736 = None
        add_242: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_241, convert_element_type_1467);  add_241 = convert_element_type_1467 = None
        convert_element_type_1468: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_1469: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1466, torch.float32);  convert_element_type_1466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_535: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, primals_279);  primals_279 = None
        mul_536: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, 1024)
        sum_140: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True)
        mul_537: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, mul_224);  mul_535 = None
        sum_141: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True);  mul_537 = None
        mul_538: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, sum_141);  sum_141 = None
        sub_124: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_536, sum_140);  mul_536 = sum_140 = None
        sub_125: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_124, mul_538);  sub_124 = mul_538 = None
        mul_539: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_72, sub_125);  div_72 = sub_125 = None
        mul_540: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_242, mul_224);  mul_224 = None
        sum_142: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 1]);  mul_540 = None
        sum_143: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
        add_243: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, mul_539);  add_240 = mul_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1470: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1471: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_51, torch.bfloat16);  gt_51 = None
        mul_541: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1471, 1.1111111111111112);  convert_element_type_1471 = None
        mul_542: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1470, mul_541);  convert_element_type_1470 = mul_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_737: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_542, [8192, 1024]);  mul_542 = None
        mm_88: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
        permute_506: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_89: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
        sum_144: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_737, [0], True, dtype = torch.float32);  view_737 = None
        view_738: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_144, [1024]);  sum_144 = None
        convert_element_type_1476: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_738, torch.bfloat16);  view_738 = None
        view_739: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [16, 512, 4096]);  mm_88 = None
        convert_element_type_1477: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_1478: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1476, torch.float32);  convert_element_type_1476 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1479: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_739, torch.float32);  view_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_371: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_100, [16, 512, 4096]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_690: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_371, torch.float32);  view_371 = None
        mul_220: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, 0.7071067811865476)
        erf_16: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
        add_136: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_544: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
        mul_545: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, convert_element_type_690)
        mul_546: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_545, -0.5);  mul_545 = None
        exp_34: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_546);  mul_546 = None
        mul_547: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_548: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, mul_547);  convert_element_type_690 = mul_547 = None
        add_245: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_544, mul_548);  mul_544 = mul_548 = None
        mul_549: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1479, add_245);  convert_element_type_1479 = add_245 = None
        convert_element_type_1481: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_549, torch.bfloat16);  mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_740: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1481, [8192, 4096]);  convert_element_type_1481 = None
        mm_90: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
        permute_510: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_91: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
        sum_145: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_740, [0], True, dtype = torch.float32);  view_740 = None
        view_741: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_145, [4096]);  sum_145 = None
        convert_element_type_1486: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_741, torch.bfloat16);  view_741 = None
        view_742: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [16, 512, 1024]);  mm_90 = None
        convert_element_type_1487: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_742, torch.float32);  view_742 = None
        convert_element_type_1488: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_1489: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1486, torch.float32);  convert_element_type_1486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_551: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1487, primals_273);  primals_273 = None
        mul_552: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_551, 1024)
        sum_146: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
        mul_553: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_551, mul_217);  mul_551 = None
        sum_147: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
        mul_554: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, sum_147);  sum_147 = None
        sub_127: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_552, sum_146);  mul_552 = sum_146 = None
        sub_128: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_127, mul_554);  sub_127 = mul_554 = None
        mul_555: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_73, sub_128);  div_73 = sub_128 = None
        mul_556: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1487, mul_217);  mul_217 = None
        sum_148: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
        sum_149: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1487, [0, 1]);  convert_element_type_1487 = None
        add_246: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_243, mul_555);  add_243 = mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1490: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1491: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_50, torch.bfloat16);  gt_50 = None
        mul_557: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1491, 1.1111111111111112);  convert_element_type_1491 = None
        mul_558: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1490, mul_557);  convert_element_type_1490 = mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_743: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_558, [8192, 1024]);  mul_558 = None
        mm_92: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
        permute_514: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_93: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
        sum_150: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_743, [0], True, dtype = torch.float32);  view_743 = None
        view_744: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_150, [1024]);  sum_150 = None
        convert_element_type_1496: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_744, torch.bfloat16);  view_744 = None
        view_745: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [16, 512, 1024]);  mm_92 = None
        convert_element_type_1497: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_1498: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1496, torch.float32);  convert_element_type_1496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_746: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_745, [16, 512, 16, 64]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_517: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_148: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_148, permute_default_42, permute_default_43, permute_default_44, getitem_149, getitem_150, None, None, 512, 512, 0.1, False, getitem_151, getitem_152, scale = 0.125);  clone_148 = permute_default_42 = permute_default_43 = permute_default_44 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = None
        getitem_153: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_7[0]
        getitem_154: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_7[1]
        getitem_155: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_7[2];  _scaled_dot_product_flash_attention_backward_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_47: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_46: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_45: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        clone_150: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_45, memory_format = torch.contiguous_format);  permute_default_45 = None
        view_753: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [16, 512, 1024]);  clone_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_754: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_753, [8192, 1024]);  view_753 = None
        mm_94: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_754, permute_524);  permute_524 = None
        permute_525: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_754, [1, 0])
        mm_95: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        sum_152: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_754, [0], True, dtype = torch.float32);  view_754 = None
        view_755: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_152, [1024]);  sum_152 = None
        convert_element_type_1514: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_755, torch.bfloat16);  view_755 = None
        view_756: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [16, 512, 1024]);  mm_94 = None
        convert_element_type_1515: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_756, torch.float32);  view_756 = None
        convert_element_type_1516: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_1517: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1514, torch.float32);  convert_element_type_1514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_757: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_46, [16, 512, 1024]);  permute_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_151: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_757, memory_format = torch.contiguous_format);  view_757 = None
        view_758: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [8192, 1024]);  clone_151 = None
        mm_96: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_758, permute_529);  permute_529 = None
        permute_530: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_97: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_530, view_352);  permute_530 = None
        sum_153: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_758, [0], True, dtype = torch.float32);  view_758 = None
        view_759: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [1024]);  sum_153 = None
        convert_element_type_1522: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_759, torch.bfloat16);  view_759 = None
        view_760: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [16, 512, 1024]);  mm_96 = None
        convert_element_type_1523: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_760, torch.float32);  view_760 = None
        add_247: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1515, convert_element_type_1523);  convert_element_type_1515 = convert_element_type_1523 = None
        convert_element_type_1524: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_1525: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1522, torch.float32);  convert_element_type_1522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_152: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_47, memory_format = torch.contiguous_format);  permute_default_47 = None
        view_761: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [16, 512, 1024]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_762: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_761, [8192, 1024]);  view_761 = None
        mm_98: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
        permute_535: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_99: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
        sum_154: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_762, [0], True, dtype = torch.float32);  view_762 = None
        view_763: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_154, [1024]);  sum_154 = None
        convert_element_type_1530: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_763, torch.bfloat16);  view_763 = None
        view_764: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [16, 512, 1024]);  mm_98 = None
        convert_element_type_1531: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_764, torch.float32);  view_764 = None
        add_248: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_247, convert_element_type_1531);  add_247 = convert_element_type_1531 = None
        convert_element_type_1532: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_1533: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1530, torch.float32);  convert_element_type_1530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_563: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_248, primals_263);  primals_263 = None
        mul_564: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, 1024)
        sum_155: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True)
        mul_565: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_563, mul_211);  mul_563 = None
        sum_156: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True);  mul_565 = None
        mul_566: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, sum_156);  sum_156 = None
        sub_130: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_564, sum_155);  mul_564 = sum_155 = None
        sub_131: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_130, mul_566);  sub_130 = mul_566 = None
        mul_567: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_75, sub_131);  div_75 = sub_131 = None
        mul_568: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_248, mul_211);  mul_211 = None
        sum_157: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 1]);  mul_568 = None
        sum_158: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
        add_249: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_246, mul_567);  add_246 = mul_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1534: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_249, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1535: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_48, torch.bfloat16);  gt_48 = None
        mul_569: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1535, 1.1111111111111112);  convert_element_type_1535 = None
        mul_570: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1534, mul_569);  convert_element_type_1534 = mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_765: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_570, [8192, 1024]);  mul_570 = None
        mm_100: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
        permute_539: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_101: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
        sum_159: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_765, [0], True, dtype = torch.float32);  view_765 = None
        view_766: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [1024]);  sum_159 = None
        convert_element_type_1540: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_766, torch.bfloat16);  view_766 = None
        view_767: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [16, 512, 4096]);  mm_100 = None
        convert_element_type_1541: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_1542: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1540, torch.float32);  convert_element_type_1540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1543: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_767, torch.float32);  view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_94, [16, 512, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_649: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_349, torch.float32);  view_349 = None
        mul_207: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, 0.7071067811865476)
        erf_15: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_128: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_572: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
        mul_573: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, convert_element_type_649)
        mul_574: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, -0.5);  mul_573 = None
        exp_35: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_574);  mul_574 = None
        mul_575: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_576: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_649, mul_575);  convert_element_type_649 = mul_575 = None
        add_251: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_572, mul_576);  mul_572 = mul_576 = None
        mul_577: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1543, add_251);  convert_element_type_1543 = add_251 = None
        convert_element_type_1545: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_577, torch.bfloat16);  mul_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_768: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1545, [8192, 4096]);  convert_element_type_1545 = None
        mm_102: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
        permute_543: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_103: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
        sum_160: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_768, [0], True, dtype = torch.float32);  view_768 = None
        view_769: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_160, [4096]);  sum_160 = None
        convert_element_type_1550: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_769, torch.bfloat16);  view_769 = None
        view_770: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [16, 512, 1024]);  mm_102 = None
        convert_element_type_1551: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_770, torch.float32);  view_770 = None
        convert_element_type_1552: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_1553: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1550, torch.float32);  convert_element_type_1550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_579: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1551, primals_257);  primals_257 = None
        mul_580: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, 1024)
        sum_161: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
        mul_581: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_579, mul_204);  mul_579 = None
        sum_162: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        mul_582: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, sum_162);  sum_162 = None
        sub_133: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_580, sum_161);  mul_580 = sum_161 = None
        sub_134: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_133, mul_582);  sub_133 = mul_582 = None
        mul_583: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_76, sub_134);  div_76 = sub_134 = None
        mul_584: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1551, mul_204);  mul_204 = None
        sum_163: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
        sum_164: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1551, [0, 1]);  convert_element_type_1551 = None
        add_252: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_249, mul_583);  add_249 = mul_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1554: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1555: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_47, torch.bfloat16);  gt_47 = None
        mul_585: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1555, 1.1111111111111112);  convert_element_type_1555 = None
        mul_586: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1554, mul_585);  convert_element_type_1554 = mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_771: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_586, [8192, 1024]);  mul_586 = None
        mm_104: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
        permute_547: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_105: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
        sum_165: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_771, [0], True, dtype = torch.float32);  view_771 = None
        view_772: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [1024]);  sum_165 = None
        convert_element_type_1560: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_772, torch.bfloat16);  view_772 = None
        view_773: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [16, 512, 1024]);  mm_104 = None
        convert_element_type_1561: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_1562: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1560, torch.float32);  convert_element_type_1560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_774: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_773, [16, 512, 16, 64]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_550: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_155: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_550, memory_format = torch.contiguous_format);  permute_550 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_155, permute_default_48, permute_default_49, permute_default_50, getitem_156, getitem_157, None, None, 512, 512, 0.1, False, getitem_158, getitem_159, scale = 0.125);  clone_155 = permute_default_48 = permute_default_49 = permute_default_50 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = None
        getitem_160: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_8[0]
        getitem_161: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_8[1]
        getitem_162: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_8[2];  _scaled_dot_product_flash_attention_backward_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_53: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_52: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_51: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        clone_157: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_51, memory_format = torch.contiguous_format);  permute_default_51 = None
        view_781: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [16, 512, 1024]);  clone_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_782: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_781, [8192, 1024]);  view_781 = None
        mm_106: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_782, permute_557);  permute_557 = None
        permute_558: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_782, [1, 0])
        mm_107: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        sum_167: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_782, [0], True, dtype = torch.float32);  view_782 = None
        view_783: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_167, [1024]);  sum_167 = None
        convert_element_type_1578: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_783, torch.bfloat16);  view_783 = None
        view_784: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [16, 512, 1024]);  mm_106 = None
        convert_element_type_1579: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_784, torch.float32);  view_784 = None
        convert_element_type_1580: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_1581: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1578, torch.float32);  convert_element_type_1578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_785: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_52, [16, 512, 1024]);  permute_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_158: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_785, memory_format = torch.contiguous_format);  view_785 = None
        view_786: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [8192, 1024]);  clone_158 = None
        mm_108: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_786, permute_562);  permute_562 = None
        permute_563: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_109: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_563, view_330);  permute_563 = None
        sum_168: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_786, [0], True, dtype = torch.float32);  view_786 = None
        view_787: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_168, [1024]);  sum_168 = None
        convert_element_type_1586: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_787, torch.bfloat16);  view_787 = None
        view_788: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [16, 512, 1024]);  mm_108 = None
        convert_element_type_1587: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_788, torch.float32);  view_788 = None
        add_253: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1579, convert_element_type_1587);  convert_element_type_1579 = convert_element_type_1587 = None
        convert_element_type_1588: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_1589: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1586, torch.float32);  convert_element_type_1586 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_159: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_53, memory_format = torch.contiguous_format);  permute_default_53 = None
        view_789: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [16, 512, 1024]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_790: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_789, [8192, 1024]);  view_789 = None
        mm_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
        permute_568: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_111: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
        sum_169: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_790, [0], True, dtype = torch.float32);  view_790 = None
        view_791: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_169, [1024]);  sum_169 = None
        convert_element_type_1594: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_791, torch.bfloat16);  view_791 = None
        view_792: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [16, 512, 1024]);  mm_110 = None
        convert_element_type_1595: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_792, torch.float32);  view_792 = None
        add_254: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, convert_element_type_1595);  add_253 = convert_element_type_1595 = None
        convert_element_type_1596: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_1597: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1594, torch.float32);  convert_element_type_1594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_591: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_254, primals_247);  primals_247 = None
        mul_592: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, 1024)
        sum_170: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_591, [2], True)
        mul_593: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_591, mul_198);  mul_591 = None
        sum_171: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True);  mul_593 = None
        mul_594: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_198, sum_171);  sum_171 = None
        sub_136: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_592, sum_170);  mul_592 = sum_170 = None
        sub_137: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, mul_594);  sub_136 = mul_594 = None
        mul_595: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_78, sub_137);  div_78 = sub_137 = None
        mul_596: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_254, mul_198);  mul_198 = None
        sum_172: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 1]);  mul_596 = None
        sum_173: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None
        add_255: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_252, mul_595);  add_252 = mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1598: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_255, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1599: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_45, torch.bfloat16);  gt_45 = None
        mul_597: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1599, 1.1111111111111112);  convert_element_type_1599 = None
        mul_598: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1598, mul_597);  convert_element_type_1598 = mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_793: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_598, [8192, 1024]);  mul_598 = None
        mm_112: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
        permute_572: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_113: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
        sum_174: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_793, [0], True, dtype = torch.float32);  view_793 = None
        view_794: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_174, [1024]);  sum_174 = None
        convert_element_type_1604: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_794, torch.bfloat16);  view_794 = None
        view_795: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_112, [16, 512, 4096]);  mm_112 = None
        convert_element_type_1605: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_1606: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1604, torch.float32);  convert_element_type_1604 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1607: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_795, torch.float32);  view_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_88, [16, 512, 4096]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_608: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_327, torch.float32);  view_327 = None
        mul_194: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, 0.7071067811865476)
        erf_14: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
        add_120: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_600: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
        mul_601: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, convert_element_type_608)
        mul_602: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_601, -0.5);  mul_601 = None
        exp_36: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_602);  mul_602 = None
        mul_603: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_604: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, mul_603);  convert_element_type_608 = mul_603 = None
        add_257: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_600, mul_604);  mul_600 = mul_604 = None
        mul_605: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1607, add_257);  convert_element_type_1607 = add_257 = None
        convert_element_type_1609: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_605, torch.bfloat16);  mul_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1609, [8192, 4096]);  convert_element_type_1609 = None
        mm_114: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
        permute_576: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_115: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
        sum_175: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_796, [0], True, dtype = torch.float32);  view_796 = None
        view_797: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_175, [4096]);  sum_175 = None
        convert_element_type_1614: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_797, torch.bfloat16);  view_797 = None
        view_798: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [16, 512, 1024]);  mm_114 = None
        convert_element_type_1615: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_798, torch.float32);  view_798 = None
        convert_element_type_1616: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_1617: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1614, torch.float32);  convert_element_type_1614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_607: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1615, primals_241);  primals_241 = None
        mul_608: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_607, 1024)
        sum_176: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True)
        mul_609: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_607, mul_191);  mul_607 = None
        sum_177: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True);  mul_609 = None
        mul_610: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_191, sum_177);  sum_177 = None
        sub_139: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_608, sum_176);  mul_608 = sum_176 = None
        sub_140: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, mul_610);  sub_139 = mul_610 = None
        mul_611: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_79, sub_140);  div_79 = sub_140 = None
        mul_612: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1615, mul_191);  mul_191 = None
        sum_178: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
        sum_179: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1615, [0, 1]);  convert_element_type_1615 = None
        add_258: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_255, mul_611);  add_255 = mul_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1618: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_258, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1619: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_44, torch.bfloat16);  gt_44 = None
        mul_613: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1619, 1.1111111111111112);  convert_element_type_1619 = None
        mul_614: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1618, mul_613);  convert_element_type_1618 = mul_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_799: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_614, [8192, 1024]);  mul_614 = None
        mm_116: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
        permute_580: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_117: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
        sum_180: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_799, [0], True, dtype = torch.float32);  view_799 = None
        view_800: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_180, [1024]);  sum_180 = None
        convert_element_type_1624: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_800, torch.bfloat16);  view_800 = None
        view_801: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [16, 512, 1024]);  mm_116 = None
        convert_element_type_1625: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_1626: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1624, torch.float32);  convert_element_type_1624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_802: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_801, [16, 512, 16, 64]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_583: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_162: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_583, memory_format = torch.contiguous_format);  permute_583 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_162, permute_default_54, permute_default_55, permute_default_56, getitem_163, getitem_164, None, None, 512, 512, 0.1, False, getitem_165, getitem_166, scale = 0.125);  clone_162 = permute_default_54 = permute_default_55 = permute_default_56 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = None
        getitem_167: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_9[0]
        getitem_168: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_9[1]
        getitem_169: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_9[2];  _scaled_dot_product_flash_attention_backward_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_59: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_58: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_57: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None
        clone_164: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_57, memory_format = torch.contiguous_format);  permute_default_57 = None
        view_809: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [16, 512, 1024]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_810: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_809, [8192, 1024]);  view_809 = None
        mm_118: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_810, permute_590);  permute_590 = None
        permute_591: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_810, [1, 0])
        mm_119: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        sum_182: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_810, [0], True, dtype = torch.float32);  view_810 = None
        view_811: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [1024]);  sum_182 = None
        convert_element_type_1642: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_811, torch.bfloat16);  view_811 = None
        view_812: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [16, 512, 1024]);  mm_118 = None
        convert_element_type_1643: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_812, torch.float32);  view_812 = None
        convert_element_type_1644: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_1645: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1642, torch.float32);  convert_element_type_1642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_813: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_58, [16, 512, 1024]);  permute_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_165: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_813, memory_format = torch.contiguous_format);  view_813 = None
        view_814: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [8192, 1024]);  clone_165 = None
        mm_120: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_814, permute_595);  permute_595 = None
        permute_596: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_814, [1, 0])
        mm_121: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_596, view_308);  permute_596 = None
        sum_183: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_814, [0], True, dtype = torch.float32);  view_814 = None
        view_815: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [1024]);  sum_183 = None
        convert_element_type_1650: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_815, torch.bfloat16);  view_815 = None
        view_816: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_120, [16, 512, 1024]);  mm_120 = None
        convert_element_type_1651: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_816, torch.float32);  view_816 = None
        add_259: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1643, convert_element_type_1651);  convert_element_type_1643 = convert_element_type_1651 = None
        convert_element_type_1652: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_121, torch.float32);  mm_121 = None
        convert_element_type_1653: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1650, torch.float32);  convert_element_type_1650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_166: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_59, memory_format = torch.contiguous_format);  permute_default_59 = None
        view_817: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [16, 512, 1024]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_818: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_817, [8192, 1024]);  view_817 = None
        mm_122: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
        permute_601: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_123: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
        sum_184: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_818, [0], True, dtype = torch.float32);  view_818 = None
        view_819: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_184, [1024]);  sum_184 = None
        convert_element_type_1658: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_819, torch.bfloat16);  view_819 = None
        view_820: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [16, 512, 1024]);  mm_122 = None
        convert_element_type_1659: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_820, torch.float32);  view_820 = None
        add_260: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_259, convert_element_type_1659);  add_259 = convert_element_type_1659 = None
        convert_element_type_1660: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_1661: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1658, torch.float32);  convert_element_type_1658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_619: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_260, primals_231);  primals_231 = None
        mul_620: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, 1024)
        sum_185: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_619, [2], True)
        mul_621: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_619, mul_185);  mul_619 = None
        sum_186: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True);  mul_621 = None
        mul_622: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, sum_186);  sum_186 = None
        sub_142: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_620, sum_185);  mul_620 = sum_185 = None
        sub_143: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_142, mul_622);  sub_142 = mul_622 = None
        mul_623: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_81, sub_143);  div_81 = sub_143 = None
        mul_624: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_260, mul_185);  mul_185 = None
        sum_187: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 1]);  mul_624 = None
        sum_188: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
        add_261: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, mul_623);  add_258 = mul_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1662: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_261, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1663: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_42, torch.bfloat16);  gt_42 = None
        mul_625: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1663, 1.1111111111111112);  convert_element_type_1663 = None
        mul_626: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1662, mul_625);  convert_element_type_1662 = mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_821: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_626, [8192, 1024]);  mul_626 = None
        mm_124: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
        permute_605: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_125: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
        sum_189: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_821, [0], True, dtype = torch.float32);  view_821 = None
        view_822: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [1024]);  sum_189 = None
        convert_element_type_1668: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_822, torch.bfloat16);  view_822 = None
        view_823: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [16, 512, 4096]);  mm_124 = None
        convert_element_type_1669: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_1670: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1668, torch.float32);  convert_element_type_1668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1671: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_823, torch.float32);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [16, 512, 4096]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_567: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_305, torch.float32);  view_305 = None
        mul_181: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, 0.7071067811865476)
        erf_13: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_112: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_628: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
        mul_629: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, convert_element_type_567)
        mul_630: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_629, -0.5);  mul_629 = None
        exp_37: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_630);  mul_630 = None
        mul_631: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_632: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_567, mul_631);  convert_element_type_567 = mul_631 = None
        add_263: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_628, mul_632);  mul_628 = mul_632 = None
        mul_633: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1671, add_263);  convert_element_type_1671 = add_263 = None
        convert_element_type_1673: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_633, torch.bfloat16);  mul_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1673, [8192, 4096]);  convert_element_type_1673 = None
        mm_126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
        permute_609: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_127: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
        sum_190: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_824, [0], True, dtype = torch.float32);  view_824 = None
        view_825: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_190, [4096]);  sum_190 = None
        convert_element_type_1678: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_825, torch.bfloat16);  view_825 = None
        view_826: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [16, 512, 1024]);  mm_126 = None
        convert_element_type_1679: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_826, torch.float32);  view_826 = None
        convert_element_type_1680: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_1681: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1678, torch.float32);  convert_element_type_1678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_635: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1679, primals_225);  primals_225 = None
        mul_636: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_635, 1024)
        sum_191: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
        mul_637: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_635, mul_178);  mul_635 = None
        sum_192: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
        mul_638: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, sum_192);  sum_192 = None
        sub_145: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_636, sum_191);  mul_636 = sum_191 = None
        sub_146: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_145, mul_638);  sub_145 = mul_638 = None
        mul_639: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_82, sub_146);  div_82 = sub_146 = None
        mul_640: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1679, mul_178);  mul_178 = None
        sum_193: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
        sum_194: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1679, [0, 1]);  convert_element_type_1679 = None
        add_264: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_261, mul_639);  add_261 = mul_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1682: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1683: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_41, torch.bfloat16);  gt_41 = None
        mul_641: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1683, 1.1111111111111112);  convert_element_type_1683 = None
        mul_642: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1682, mul_641);  convert_element_type_1682 = mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_827: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_642, [8192, 1024]);  mul_642 = None
        mm_128: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
        permute_613: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_827, [1, 0])
        mm_129: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
        sum_195: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_827, [0], True, dtype = torch.float32);  view_827 = None
        view_828: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_195, [1024]);  sum_195 = None
        convert_element_type_1688: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_828, torch.bfloat16);  view_828 = None
        view_829: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_128, [16, 512, 1024]);  mm_128 = None
        convert_element_type_1689: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_129, torch.float32);  mm_129 = None
        convert_element_type_1690: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1688, torch.float32);  convert_element_type_1688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_830: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_829, [16, 512, 16, 64]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_616: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_169: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_169, permute_default_60, permute_default_61, permute_default_62, getitem_170, getitem_171, None, None, 512, 512, 0.1, False, getitem_172, getitem_173, scale = 0.125);  clone_169 = permute_default_60 = permute_default_61 = permute_default_62 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = None
        getitem_174: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_10[0]
        getitem_175: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_10[1]
        getitem_176: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_10[2];  _scaled_dot_product_flash_attention_backward_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_65: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_64: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_63: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        clone_171: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_63, memory_format = torch.contiguous_format);  permute_default_63 = None
        view_837: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [16, 512, 1024]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_838: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_837, [8192, 1024]);  view_837 = None
        mm_130: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_838, permute_623);  permute_623 = None
        permute_624: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_838, [1, 0])
        mm_131: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        sum_197: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_838, [0], True, dtype = torch.float32);  view_838 = None
        view_839: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_197, [1024]);  sum_197 = None
        convert_element_type_1706: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_839, torch.bfloat16);  view_839 = None
        view_840: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [16, 512, 1024]);  mm_130 = None
        convert_element_type_1707: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_840, torch.float32);  view_840 = None
        convert_element_type_1708: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_1709: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1706, torch.float32);  convert_element_type_1706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_841: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_64, [16, 512, 1024]);  permute_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_172: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_841, memory_format = torch.contiguous_format);  view_841 = None
        view_842: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [8192, 1024]);  clone_172 = None
        mm_132: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_842, permute_628);  permute_628 = None
        permute_629: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_842, [1, 0])
        mm_133: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_629, view_286);  permute_629 = None
        sum_198: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_842, [0], True, dtype = torch.float32);  view_842 = None
        view_843: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_198, [1024]);  sum_198 = None
        convert_element_type_1714: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_843, torch.bfloat16);  view_843 = None
        view_844: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [16, 512, 1024]);  mm_132 = None
        convert_element_type_1715: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_844, torch.float32);  view_844 = None
        add_265: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1707, convert_element_type_1715);  convert_element_type_1707 = convert_element_type_1715 = None
        convert_element_type_1716: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_1717: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1714, torch.float32);  convert_element_type_1714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_173: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_65, memory_format = torch.contiguous_format);  permute_default_65 = None
        view_845: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [16, 512, 1024]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_846: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_845, [8192, 1024]);  view_845 = None
        mm_134: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
        permute_634: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_135: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
        sum_199: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_846, [0], True, dtype = torch.float32);  view_846 = None
        view_847: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_199, [1024]);  sum_199 = None
        convert_element_type_1722: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_847, torch.bfloat16);  view_847 = None
        view_848: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [16, 512, 1024]);  mm_134 = None
        convert_element_type_1723: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_848, torch.float32);  view_848 = None
        add_266: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, convert_element_type_1723);  add_265 = convert_element_type_1723 = None
        convert_element_type_1724: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1725: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1722, torch.float32);  convert_element_type_1722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_647: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, primals_215);  primals_215 = None
        mul_648: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_647, 1024)
        sum_200: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True)
        mul_649: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_647, mul_172);  mul_647 = None
        sum_201: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True);  mul_649 = None
        mul_650: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_172, sum_201);  sum_201 = None
        sub_148: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_648, sum_200);  mul_648 = sum_200 = None
        sub_149: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_148, mul_650);  sub_148 = mul_650 = None
        mul_651: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_84, sub_149);  div_84 = sub_149 = None
        mul_652: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_266, mul_172);  mul_172 = None
        sum_202: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 1]);  mul_652 = None
        sum_203: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
        add_267: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_264, mul_651);  add_264 = mul_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1726: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_267, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1727: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_39, torch.bfloat16);  gt_39 = None
        mul_653: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1727, 1.1111111111111112);  convert_element_type_1727 = None
        mul_654: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1726, mul_653);  convert_element_type_1726 = mul_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_849: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_654, [8192, 1024]);  mul_654 = None
        mm_136: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
        permute_638: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_137: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
        sum_204: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_849, [0], True, dtype = torch.float32);  view_849 = None
        view_850: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_204, [1024]);  sum_204 = None
        convert_element_type_1732: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_850, torch.bfloat16);  view_850 = None
        view_851: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_136, [16, 512, 4096]);  mm_136 = None
        convert_element_type_1733: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_137, torch.float32);  mm_137 = None
        convert_element_type_1734: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1732, torch.float32);  convert_element_type_1732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1735: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_851, torch.float32);  view_851 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_76, [16, 512, 4096]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_526: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_283, torch.float32);  view_283 = None
        mul_168: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, 0.7071067811865476)
        erf_12: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_104: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_656: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
        mul_657: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, convert_element_type_526)
        mul_658: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_657, -0.5);  mul_657 = None
        exp_38: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_658);  mul_658 = None
        mul_659: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_660: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_526, mul_659);  convert_element_type_526 = mul_659 = None
        add_269: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_656, mul_660);  mul_656 = mul_660 = None
        mul_661: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1735, add_269);  convert_element_type_1735 = add_269 = None
        convert_element_type_1737: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_661, torch.bfloat16);  mul_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_852: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1737, [8192, 4096]);  convert_element_type_1737 = None
        mm_138: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
        permute_642: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_139: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
        sum_205: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_852, [0], True, dtype = torch.float32);  view_852 = None
        view_853: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_205, [4096]);  sum_205 = None
        convert_element_type_1742: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_853, torch.bfloat16);  view_853 = None
        view_854: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [16, 512, 1024]);  mm_138 = None
        convert_element_type_1743: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_854, torch.float32);  view_854 = None
        convert_element_type_1744: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1745: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1742, torch.float32);  convert_element_type_1742 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_663: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1743, primals_209);  primals_209 = None
        mul_664: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, 1024)
        sum_206: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
        mul_665: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_663, mul_165);  mul_663 = None
        sum_207: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
        mul_666: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_165, sum_207);  sum_207 = None
        sub_151: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_664, sum_206);  mul_664 = sum_206 = None
        sub_152: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_151, mul_666);  sub_151 = mul_666 = None
        mul_667: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_85, sub_152);  div_85 = sub_152 = None
        mul_668: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1743, mul_165);  mul_165 = None
        sum_208: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
        sum_209: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1743, [0, 1]);  convert_element_type_1743 = None
        add_270: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_267, mul_667);  add_267 = mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1746: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1747: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_38, torch.bfloat16);  gt_38 = None
        mul_669: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1747, 1.1111111111111112);  convert_element_type_1747 = None
        mul_670: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1746, mul_669);  convert_element_type_1746 = mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_855: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_670, [8192, 1024]);  mul_670 = None
        mm_140: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
        permute_646: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_855, [1, 0])
        mm_141: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
        sum_210: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_855, [0], True, dtype = torch.float32);  view_855 = None
        view_856: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_210, [1024]);  sum_210 = None
        convert_element_type_1752: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_856, torch.bfloat16);  view_856 = None
        view_857: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [16, 512, 1024]);  mm_140 = None
        convert_element_type_1753: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1754: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1752, torch.float32);  convert_element_type_1752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_858: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_857, [16, 512, 16, 64]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_649: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_176: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_176, permute_default_66, permute_default_67, permute_default_68, getitem_177, getitem_178, None, None, 512, 512, 0.1, False, getitem_179, getitem_180, scale = 0.125);  clone_176 = permute_default_66 = permute_default_67 = permute_default_68 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = None
        getitem_181: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_11[0]
        getitem_182: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_11[1]
        getitem_183: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_11[2];  _scaled_dot_product_flash_attention_backward_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_71: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_70: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_69: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        clone_178: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_69, memory_format = torch.contiguous_format);  permute_default_69 = None
        view_865: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [16, 512, 1024]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_866: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_865, [8192, 1024]);  view_865 = None
        mm_142: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_866, permute_656);  permute_656 = None
        permute_657: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_866, [1, 0])
        mm_143: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        sum_212: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_866, [0], True, dtype = torch.float32);  view_866 = None
        view_867: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_212, [1024]);  sum_212 = None
        convert_element_type_1770: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_867, torch.bfloat16);  view_867 = None
        view_868: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [16, 512, 1024]);  mm_142 = None
        convert_element_type_1771: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_868, torch.float32);  view_868 = None
        convert_element_type_1772: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1773: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1770, torch.float32);  convert_element_type_1770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_869: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_70, [16, 512, 1024]);  permute_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_179: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_869, memory_format = torch.contiguous_format);  view_869 = None
        view_870: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [8192, 1024]);  clone_179 = None
        mm_144: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_870, permute_661);  permute_661 = None
        permute_662: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_870, [1, 0])
        mm_145: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_662, view_264);  permute_662 = None
        sum_213: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_870, [0], True, dtype = torch.float32);  view_870 = None
        view_871: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_213, [1024]);  sum_213 = None
        convert_element_type_1778: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_871, torch.bfloat16);  view_871 = None
        view_872: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_144, [16, 512, 1024]);  mm_144 = None
        convert_element_type_1779: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_872, torch.float32);  view_872 = None
        add_271: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1771, convert_element_type_1779);  convert_element_type_1771 = convert_element_type_1779 = None
        convert_element_type_1780: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_145, torch.float32);  mm_145 = None
        convert_element_type_1781: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1778, torch.float32);  convert_element_type_1778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_180: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_71, memory_format = torch.contiguous_format);  permute_default_71 = None
        view_873: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [16, 512, 1024]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_874: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_873, [8192, 1024]);  view_873 = None
        mm_146: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
        permute_667: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_874, [1, 0])
        mm_147: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
        sum_214: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_874, [0], True, dtype = torch.float32);  view_874 = None
        view_875: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_214, [1024]);  sum_214 = None
        convert_element_type_1786: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_875, torch.bfloat16);  view_875 = None
        view_876: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [16, 512, 1024]);  mm_146 = None
        convert_element_type_1787: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_876, torch.float32);  view_876 = None
        add_272: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_271, convert_element_type_1787);  add_271 = convert_element_type_1787 = None
        convert_element_type_1788: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1789: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1786, torch.float32);  convert_element_type_1786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_675: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_272, primals_199);  primals_199 = None
        mul_676: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_675, 1024)
        sum_215: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True)
        mul_677: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_675, mul_159);  mul_675 = None
        sum_216: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True);  mul_677 = None
        mul_678: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_159, sum_216);  sum_216 = None
        sub_154: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_676, sum_215);  mul_676 = sum_215 = None
        sub_155: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_154, mul_678);  sub_154 = mul_678 = None
        mul_679: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_87, sub_155);  div_87 = sub_155 = None
        mul_680: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_272, mul_159);  mul_159 = None
        sum_217: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 1]);  mul_680 = None
        sum_218: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None
        add_273: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, mul_679);  add_270 = mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1790: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_273, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1791: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_36, torch.bfloat16);  gt_36 = None
        mul_681: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1791, 1.1111111111111112);  convert_element_type_1791 = None
        mul_682: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1790, mul_681);  convert_element_type_1790 = mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_877: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_682, [8192, 1024]);  mul_682 = None
        mm_148: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
        permute_671: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_149: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
        sum_219: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_877, [0], True, dtype = torch.float32);  view_877 = None
        view_878: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_219, [1024]);  sum_219 = None
        convert_element_type_1796: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_878, torch.bfloat16);  view_878 = None
        view_879: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [16, 512, 4096]);  mm_148 = None
        convert_element_type_1797: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_1798: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1796, torch.float32);  convert_element_type_1796 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1799: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_879, torch.float32);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [16, 512, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_485: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_261, torch.float32);  view_261 = None
        mul_155: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, 0.7071067811865476)
        erf_11: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_684: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_685: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, convert_element_type_485)
        mul_686: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_685, -0.5);  mul_685 = None
        exp_39: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_686);  mul_686 = None
        mul_687: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_688: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_485, mul_687);  convert_element_type_485 = mul_687 = None
        add_275: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_684, mul_688);  mul_684 = mul_688 = None
        mul_689: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1799, add_275);  convert_element_type_1799 = add_275 = None
        convert_element_type_1801: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_689, torch.bfloat16);  mul_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_880: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1801, [8192, 4096]);  convert_element_type_1801 = None
        mm_150: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
        permute_675: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_151: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
        sum_220: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_880, [0], True, dtype = torch.float32);  view_880 = None
        view_881: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_220, [4096]);  sum_220 = None
        convert_element_type_1806: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_881, torch.bfloat16);  view_881 = None
        view_882: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [16, 512, 1024]);  mm_150 = None
        convert_element_type_1807: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_882, torch.float32);  view_882 = None
        convert_element_type_1808: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_1809: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1806, torch.float32);  convert_element_type_1806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_691: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1807, primals_193);  primals_193 = None
        mul_692: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_691, 1024)
        sum_221: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_691, [2], True)
        mul_693: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_691, mul_152);  mul_691 = None
        sum_222: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True);  mul_693 = None
        mul_694: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_152, sum_222);  sum_222 = None
        sub_157: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_692, sum_221);  mul_692 = sum_221 = None
        sub_158: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_157, mul_694);  sub_157 = mul_694 = None
        mul_695: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_88, sub_158);  div_88 = sub_158 = None
        mul_696: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1807, mul_152);  mul_152 = None
        sum_223: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 1]);  mul_696 = None
        sum_224: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1807, [0, 1]);  convert_element_type_1807 = None
        add_276: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_273, mul_695);  add_273 = mul_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1810: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_276, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1811: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_35, torch.bfloat16);  gt_35 = None
        mul_697: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1811, 1.1111111111111112);  convert_element_type_1811 = None
        mul_698: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1810, mul_697);  convert_element_type_1810 = mul_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_883: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_698, [8192, 1024]);  mul_698 = None
        mm_152: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
        permute_679: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_153: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
        sum_225: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_883, [0], True, dtype = torch.float32);  view_883 = None
        view_884: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [1024]);  sum_225 = None
        convert_element_type_1816: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_884, torch.bfloat16);  view_884 = None
        view_885: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_152, [16, 512, 1024]);  mm_152 = None
        convert_element_type_1817: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_153, torch.float32);  mm_153 = None
        convert_element_type_1818: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1816, torch.float32);  convert_element_type_1816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_886: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_885, [16, 512, 16, 64]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_682: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_183: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_12 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_183, permute_default_72, permute_default_73, permute_default_74, getitem_184, getitem_185, None, None, 512, 512, 0.1, False, getitem_186, getitem_187, scale = 0.125);  clone_183 = permute_default_72 = permute_default_73 = permute_default_74 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = None
        getitem_188: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_12[0]
        getitem_189: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_12[1]
        getitem_190: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_12[2];  _scaled_dot_product_flash_attention_backward_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_77: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_76: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_75: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_190, [0, 2, 1, 3]);  getitem_190 = None
        clone_185: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_75, memory_format = torch.contiguous_format);  permute_default_75 = None
        view_893: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [16, 512, 1024]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_894: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_893, [8192, 1024]);  view_893 = None
        mm_154: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_894, permute_689);  permute_689 = None
        permute_690: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_894, [1, 0])
        mm_155: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        sum_227: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_894, [0], True, dtype = torch.float32);  view_894 = None
        view_895: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_227, [1024]);  sum_227 = None
        convert_element_type_1834: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_895, torch.bfloat16);  view_895 = None
        view_896: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [16, 512, 1024]);  mm_154 = None
        convert_element_type_1835: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_896, torch.float32);  view_896 = None
        convert_element_type_1836: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None
        convert_element_type_1837: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1834, torch.float32);  convert_element_type_1834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_897: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_76, [16, 512, 1024]);  permute_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_186: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_897, memory_format = torch.contiguous_format);  view_897 = None
        view_898: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [8192, 1024]);  clone_186 = None
        mm_156: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_898, permute_694);  permute_694 = None
        permute_695: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_157: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_695, view_242);  permute_695 = None
        sum_228: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_898, [0], True, dtype = torch.float32);  view_898 = None
        view_899: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_228, [1024]);  sum_228 = None
        convert_element_type_1842: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_899, torch.bfloat16);  view_899 = None
        view_900: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [16, 512, 1024]);  mm_156 = None
        convert_element_type_1843: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_900, torch.float32);  view_900 = None
        add_277: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1835, convert_element_type_1843);  convert_element_type_1835 = convert_element_type_1843 = None
        convert_element_type_1844: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_1845: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1842, torch.float32);  convert_element_type_1842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_187: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_77, memory_format = torch.contiguous_format);  permute_default_77 = None
        view_901: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [16, 512, 1024]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_902: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_901, [8192, 1024]);  view_901 = None
        mm_158: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
        permute_700: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_159: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
        sum_229: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_902, [0], True, dtype = torch.float32);  view_902 = None
        view_903: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_229, [1024]);  sum_229 = None
        convert_element_type_1850: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_903, torch.bfloat16);  view_903 = None
        view_904: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [16, 512, 1024]);  mm_158 = None
        convert_element_type_1851: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_904, torch.float32);  view_904 = None
        add_278: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_277, convert_element_type_1851);  add_277 = convert_element_type_1851 = None
        convert_element_type_1852: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_1853: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1850, torch.float32);  convert_element_type_1850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_703: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_278, primals_183);  primals_183 = None
        mul_704: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_703, 1024)
        sum_230: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True)
        mul_705: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_703, mul_146);  mul_703 = None
        sum_231: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True);  mul_705 = None
        mul_706: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, sum_231);  sum_231 = None
        sub_160: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_704, sum_230);  mul_704 = sum_230 = None
        sub_161: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, mul_706);  sub_160 = mul_706 = None
        mul_707: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_90, sub_161);  div_90 = sub_161 = None
        mul_708: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_278, mul_146);  mul_146 = None
        sum_232: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 1]);  mul_708 = None
        sum_233: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
        add_279: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_276, mul_707);  add_276 = mul_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1854: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_279, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1855: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_33, torch.bfloat16);  gt_33 = None
        mul_709: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1855, 1.1111111111111112);  convert_element_type_1855 = None
        mul_710: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1854, mul_709);  convert_element_type_1854 = mul_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_905: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_710, [8192, 1024]);  mul_710 = None
        mm_160: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
        permute_704: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_161: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
        sum_234: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_905, [0], True, dtype = torch.float32);  view_905 = None
        view_906: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_234, [1024]);  sum_234 = None
        convert_element_type_1860: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_906, torch.bfloat16);  view_906 = None
        view_907: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_160, [16, 512, 4096]);  mm_160 = None
        convert_element_type_1861: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_161, torch.float32);  mm_161 = None
        convert_element_type_1862: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1860, torch.float32);  convert_element_type_1860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1863: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_907, torch.float32);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_64, [16, 512, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_444: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_239, torch.float32);  view_239 = None
        mul_142: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, 0.7071067811865476)
        erf_10: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_712: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_713: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, convert_element_type_444)
        mul_714: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_713, -0.5);  mul_713 = None
        exp_40: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_714);  mul_714 = None
        mul_715: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_716: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_444, mul_715);  convert_element_type_444 = mul_715 = None
        add_281: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_712, mul_716);  mul_712 = mul_716 = None
        mul_717: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1863, add_281);  convert_element_type_1863 = add_281 = None
        convert_element_type_1865: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_717, torch.bfloat16);  mul_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1865, [8192, 4096]);  convert_element_type_1865 = None
        mm_162: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
        permute_708: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_163: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
        sum_235: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_908, [0], True, dtype = torch.float32);  view_908 = None
        view_909: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_235, [4096]);  sum_235 = None
        convert_element_type_1870: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.bfloat16);  view_909 = None
        view_910: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [16, 512, 1024]);  mm_162 = None
        convert_element_type_1871: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_910, torch.float32);  view_910 = None
        convert_element_type_1872: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_1873: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1870, torch.float32);  convert_element_type_1870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_719: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1871, primals_177);  primals_177 = None
        mul_720: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_719, 1024)
        sum_236: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_719, [2], True)
        mul_721: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_719, mul_139);  mul_719 = None
        sum_237: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_721, [2], True);  mul_721 = None
        mul_722: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, sum_237);  sum_237 = None
        sub_163: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_720, sum_236);  mul_720 = sum_236 = None
        sub_164: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_163, mul_722);  sub_163 = mul_722 = None
        mul_723: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_91, sub_164);  div_91 = sub_164 = None
        mul_724: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1871, mul_139);  mul_139 = None
        sum_238: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 1]);  mul_724 = None
        sum_239: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1871, [0, 1]);  convert_element_type_1871 = None
        add_282: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_279, mul_723);  add_279 = mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1874: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_282, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1875: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_32, torch.bfloat16);  gt_32 = None
        mul_725: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1875, 1.1111111111111112);  convert_element_type_1875 = None
        mul_726: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1874, mul_725);  convert_element_type_1874 = mul_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_911: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_726, [8192, 1024]);  mul_726 = None
        mm_164: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
        permute_712: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_911, [1, 0])
        mm_165: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
        sum_240: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_911, [0], True, dtype = torch.float32);  view_911 = None
        view_912: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_240, [1024]);  sum_240 = None
        convert_element_type_1880: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_912, torch.bfloat16);  view_912 = None
        view_913: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [16, 512, 1024]);  mm_164 = None
        convert_element_type_1881: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_1882: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1880, torch.float32);  convert_element_type_1880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_914: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_913, [16, 512, 16, 64]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_715: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_190: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_13 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_190, permute_default_78, permute_default_79, permute_default_80, getitem_191, getitem_192, None, None, 512, 512, 0.1, False, getitem_193, getitem_194, scale = 0.125);  clone_190 = permute_default_78 = permute_default_79 = permute_default_80 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = None
        getitem_195: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_13[0]
        getitem_196: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_13[1]
        getitem_197: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_13[2];  _scaled_dot_product_flash_attention_backward_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_83: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_82: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_81: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
        clone_192: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_81, memory_format = torch.contiguous_format);  permute_default_81 = None
        view_921: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_192, [16, 512, 1024]);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_922: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_921, [8192, 1024]);  view_921 = None
        mm_166: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_922, permute_722);  permute_722 = None
        permute_723: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_922, [1, 0])
        mm_167: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        sum_242: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_922, [0], True, dtype = torch.float32);  view_922 = None
        view_923: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_242, [1024]);  sum_242 = None
        convert_element_type_1898: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_923, torch.bfloat16);  view_923 = None
        view_924: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [16, 512, 1024]);  mm_166 = None
        convert_element_type_1899: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_924, torch.float32);  view_924 = None
        convert_element_type_1900: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        convert_element_type_1901: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1898, torch.float32);  convert_element_type_1898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_925: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_82, [16, 512, 1024]);  permute_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_193: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_925, memory_format = torch.contiguous_format);  view_925 = None
        view_926: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_193, [8192, 1024]);  clone_193 = None
        mm_168: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_926, permute_727);  permute_727 = None
        permute_728: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_926, [1, 0])
        mm_169: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_728, view_220);  permute_728 = None
        sum_243: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_926, [0], True, dtype = torch.float32);  view_926 = None
        view_927: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_243, [1024]);  sum_243 = None
        convert_element_type_1906: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_927, torch.bfloat16);  view_927 = None
        view_928: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_168, [16, 512, 1024]);  mm_168 = None
        convert_element_type_1907: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_928, torch.float32);  view_928 = None
        add_283: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1899, convert_element_type_1907);  convert_element_type_1899 = convert_element_type_1907 = None
        convert_element_type_1908: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_169, torch.float32);  mm_169 = None
        convert_element_type_1909: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1906, torch.float32);  convert_element_type_1906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_194: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_83, memory_format = torch.contiguous_format);  permute_default_83 = None
        view_929: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_194, [16, 512, 1024]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_930: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_929, [8192, 1024]);  view_929 = None
        mm_170: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
        permute_733: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_171: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
        sum_244: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_930, [0], True, dtype = torch.float32);  view_930 = None
        view_931: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_244, [1024]);  sum_244 = None
        convert_element_type_1914: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_931, torch.bfloat16);  view_931 = None
        view_932: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [16, 512, 1024]);  mm_170 = None
        convert_element_type_1915: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None
        add_284: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, convert_element_type_1915);  add_283 = convert_element_type_1915 = None
        convert_element_type_1916: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_1917: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1914, torch.float32);  convert_element_type_1914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_731: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_284, primals_167);  primals_167 = None
        mul_732: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, 1024)
        sum_245: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True)
        mul_733: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_731, mul_133);  mul_731 = None
        sum_246: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
        mul_734: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_133, sum_246);  sum_246 = None
        sub_166: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_732, sum_245);  mul_732 = sum_245 = None
        sub_167: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_166, mul_734);  sub_166 = mul_734 = None
        mul_735: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_93, sub_167);  div_93 = sub_167 = None
        mul_736: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_284, mul_133);  mul_133 = None
        sum_247: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1]);  mul_736 = None
        sum_248: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None
        add_285: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, mul_735);  add_282 = mul_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1918: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_285, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1919: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_30, torch.bfloat16);  gt_30 = None
        mul_737: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1919, 1.1111111111111112);  convert_element_type_1919 = None
        mul_738: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1918, mul_737);  convert_element_type_1918 = mul_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_933: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_738, [8192, 1024]);  mul_738 = None
        mm_172: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
        permute_737: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_173: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
        sum_249: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_933, [0], True, dtype = torch.float32);  view_933 = None
        view_934: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_249, [1024]);  sum_249 = None
        convert_element_type_1924: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_934, torch.bfloat16);  view_934 = None
        view_935: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [16, 512, 4096]);  mm_172 = None
        convert_element_type_1925: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_1926: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1924, torch.float32);  convert_element_type_1924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1927: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_935, torch.float32);  view_935 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [16, 512, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_403: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_217, torch.float32);  view_217 = None
        mul_129: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, 0.7071067811865476)
        erf_9: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_740: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
        mul_741: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, convert_element_type_403)
        mul_742: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_741, -0.5);  mul_741 = None
        exp_41: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_742);  mul_742 = None
        mul_743: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_744: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_403, mul_743);  convert_element_type_403 = mul_743 = None
        add_287: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_740, mul_744);  mul_740 = mul_744 = None
        mul_745: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1927, add_287);  convert_element_type_1927 = add_287 = None
        convert_element_type_1929: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_745, torch.bfloat16);  mul_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_936: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1929, [8192, 4096]);  convert_element_type_1929 = None
        mm_174: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
        permute_741: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_936, [1, 0])
        mm_175: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
        sum_250: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_936, [0], True, dtype = torch.float32);  view_936 = None
        view_937: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_250, [4096]);  sum_250 = None
        convert_element_type_1934: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_937, torch.bfloat16);  view_937 = None
        view_938: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [16, 512, 1024]);  mm_174 = None
        convert_element_type_1935: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_938, torch.float32);  view_938 = None
        convert_element_type_1936: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_1937: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1934, torch.float32);  convert_element_type_1934 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_747: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1935, primals_161);  primals_161 = None
        mul_748: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, 1024)
        sum_251: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
        mul_749: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_747, mul_126);  mul_747 = None
        sum_252: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
        mul_750: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_126, sum_252);  sum_252 = None
        sub_169: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_748, sum_251);  mul_748 = sum_251 = None
        sub_170: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_169, mul_750);  sub_169 = mul_750 = None
        mul_751: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_94, sub_170);  div_94 = sub_170 = None
        mul_752: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1935, mul_126);  mul_126 = None
        sum_253: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
        sum_254: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1935, [0, 1]);  convert_element_type_1935 = None
        add_288: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_285, mul_751);  add_285 = mul_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_1938: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_288, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1939: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_29, torch.bfloat16);  gt_29 = None
        mul_753: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1939, 1.1111111111111112);  convert_element_type_1939 = None
        mul_754: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1938, mul_753);  convert_element_type_1938 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_939: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_754, [8192, 1024]);  mul_754 = None
        mm_176: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
        permute_745: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_939, [1, 0])
        mm_177: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
        sum_255: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_939, [0], True, dtype = torch.float32);  view_939 = None
        view_940: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_255, [1024]);  sum_255 = None
        convert_element_type_1944: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_940, torch.bfloat16);  view_940 = None
        view_941: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_176, [16, 512, 1024]);  mm_176 = None
        convert_element_type_1945: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_177, torch.float32);  mm_177 = None
        convert_element_type_1946: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1944, torch.float32);  convert_element_type_1944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_942: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_941, [16, 512, 16, 64]);  view_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_748: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_197: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_14 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_197, permute_default_84, permute_default_85, permute_default_86, getitem_198, getitem_199, None, None, 512, 512, 0.1, False, getitem_200, getitem_201, scale = 0.125);  clone_197 = permute_default_84 = permute_default_85 = permute_default_86 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = None
        getitem_202: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_14[0]
        getitem_203: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_14[1]
        getitem_204: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_14[2];  _scaled_dot_product_flash_attention_backward_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_89: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_88: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_87: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
        clone_199: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_87, memory_format = torch.contiguous_format);  permute_default_87 = None
        view_949: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_199, [16, 512, 1024]);  clone_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_950: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_949, [8192, 1024]);  view_949 = None
        mm_178: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_950, permute_755);  permute_755 = None
        permute_756: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_950, [1, 0])
        mm_179: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        sum_257: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_950, [0], True, dtype = torch.float32);  view_950 = None
        view_951: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_257, [1024]);  sum_257 = None
        convert_element_type_1962: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_951, torch.bfloat16);  view_951 = None
        view_952: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [16, 512, 1024]);  mm_178 = None
        convert_element_type_1963: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_952, torch.float32);  view_952 = None
        convert_element_type_1964: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        convert_element_type_1965: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1962, torch.float32);  convert_element_type_1962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_953: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_88, [16, 512, 1024]);  permute_default_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_200: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_953, memory_format = torch.contiguous_format);  view_953 = None
        view_954: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_200, [8192, 1024]);  clone_200 = None
        mm_180: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_954, permute_760);  permute_760 = None
        permute_761: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_954, [1, 0])
        mm_181: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_761, view_198);  permute_761 = None
        sum_258: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_954, [0], True, dtype = torch.float32);  view_954 = None
        view_955: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_258, [1024]);  sum_258 = None
        convert_element_type_1970: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_955, torch.bfloat16);  view_955 = None
        view_956: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [16, 512, 1024]);  mm_180 = None
        convert_element_type_1971: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_956, torch.float32);  view_956 = None
        add_289: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1963, convert_element_type_1971);  convert_element_type_1963 = convert_element_type_1971 = None
        convert_element_type_1972: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_1973: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1970, torch.float32);  convert_element_type_1970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_201: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_89, memory_format = torch.contiguous_format);  permute_default_89 = None
        view_957: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_201, [16, 512, 1024]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_958: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_957, [8192, 1024]);  view_957 = None
        mm_182: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
        permute_766: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_183: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
        sum_259: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_958, [0], True, dtype = torch.float32);  view_958 = None
        view_959: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_259, [1024]);  sum_259 = None
        convert_element_type_1978: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_959, torch.bfloat16);  view_959 = None
        view_960: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [16, 512, 1024]);  mm_182 = None
        convert_element_type_1979: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_960, torch.float32);  view_960 = None
        add_290: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_289, convert_element_type_1979);  add_289 = convert_element_type_1979 = None
        convert_element_type_1980: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_1981: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1978, torch.float32);  convert_element_type_1978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_759: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_290, primals_151);  primals_151 = None
        mul_760: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, 1024)
        sum_260: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_759, [2], True)
        mul_761: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_759, mul_120);  mul_759 = None
        sum_261: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True);  mul_761 = None
        mul_762: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_120, sum_261);  sum_261 = None
        sub_172: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_760, sum_260);  mul_760 = sum_260 = None
        sub_173: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_172, mul_762);  sub_172 = mul_762 = None
        mul_763: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_96, sub_173);  div_96 = sub_173 = None
        mul_764: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_290, mul_120);  mul_120 = None
        sum_262: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 1]);  mul_764 = None
        sum_263: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
        add_291: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_288, mul_763);  add_288 = mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_1982: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_291, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1983: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_27, torch.bfloat16);  gt_27 = None
        mul_765: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1983, 1.1111111111111112);  convert_element_type_1983 = None
        mul_766: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1982, mul_765);  convert_element_type_1982 = mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_961: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_766, [8192, 1024]);  mul_766 = None
        mm_184: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
        permute_770: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_961, [1, 0])
        mm_185: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
        sum_264: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_961, [0], True, dtype = torch.float32);  view_961 = None
        view_962: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_264, [1024]);  sum_264 = None
        convert_element_type_1988: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_962, torch.bfloat16);  view_962 = None
        view_963: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_184, [16, 512, 4096]);  mm_184 = None
        convert_element_type_1989: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_185, torch.float32);  mm_185 = None
        convert_element_type_1990: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1988, torch.float32);  convert_element_type_1988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1991: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_963, torch.float32);  view_963 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_52, [16, 512, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_362: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_195, torch.float32);  view_195 = None
        mul_116: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, 0.7071067811865476)
        erf_8: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_768: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        mul_769: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, convert_element_type_362)
        mul_770: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_769, -0.5);  mul_769 = None
        exp_42: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_770);  mul_770 = None
        mul_771: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_772: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_362, mul_771);  convert_element_type_362 = mul_771 = None
        add_293: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_768, mul_772);  mul_768 = mul_772 = None
        mul_773: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1991, add_293);  convert_element_type_1991 = add_293 = None
        convert_element_type_1993: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_773, torch.bfloat16);  mul_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_964: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1993, [8192, 4096]);  convert_element_type_1993 = None
        mm_186: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
        permute_774: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_964, [1, 0])
        mm_187: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
        sum_265: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_964, [0], True, dtype = torch.float32);  view_964 = None
        view_965: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_265, [4096]);  sum_265 = None
        convert_element_type_1998: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_965, torch.bfloat16);  view_965 = None
        view_966: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [16, 512, 1024]);  mm_186 = None
        convert_element_type_1999: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_966, torch.float32);  view_966 = None
        convert_element_type_2000: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_2001: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1998, torch.float32);  convert_element_type_1998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_775: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1999, primals_145);  primals_145 = None
        mul_776: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_775, 1024)
        sum_266: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True)
        mul_777: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_775, mul_113);  mul_775 = None
        sum_267: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_777, [2], True);  mul_777 = None
        mul_778: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_113, sum_267);  sum_267 = None
        sub_175: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_776, sum_266);  mul_776 = sum_266 = None
        sub_176: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_175, mul_778);  sub_175 = mul_778 = None
        mul_779: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_97, sub_176);  div_97 = sub_176 = None
        mul_780: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1999, mul_113);  mul_113 = None
        sum_268: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 1]);  mul_780 = None
        sum_269: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1999, [0, 1]);  convert_element_type_1999 = None
        add_294: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_291, mul_779);  add_291 = mul_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2002: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2003: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_26, torch.bfloat16);  gt_26 = None
        mul_781: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2003, 1.1111111111111112);  convert_element_type_2003 = None
        mul_782: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2002, mul_781);  convert_element_type_2002 = mul_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_967: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_782, [8192, 1024]);  mul_782 = None
        mm_188: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
        permute_778: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_189: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
        sum_270: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_967, [0], True, dtype = torch.float32);  view_967 = None
        view_968: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_270, [1024]);  sum_270 = None
        convert_element_type_2008: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_968, torch.bfloat16);  view_968 = None
        view_969: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [16, 512, 1024]);  mm_188 = None
        convert_element_type_2009: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_2010: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2008, torch.float32);  convert_element_type_2008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_970: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_969, [16, 512, 16, 64]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_781: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_204: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_15 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_204, permute_default_90, permute_default_91, permute_default_92, getitem_205, getitem_206, None, None, 512, 512, 0.1, False, getitem_207, getitem_208, scale = 0.125);  clone_204 = permute_default_90 = permute_default_91 = permute_default_92 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = None
        getitem_209: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_15[0]
        getitem_210: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_15[1]
        getitem_211: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_15[2];  _scaled_dot_product_flash_attention_backward_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_95: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_94: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_93: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
        clone_206: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_93, memory_format = torch.contiguous_format);  permute_default_93 = None
        view_977: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_206, [16, 512, 1024]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_978: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_977, [8192, 1024]);  view_977 = None
        mm_190: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_978, permute_788);  permute_788 = None
        permute_789: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_978, [1, 0])
        mm_191: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        sum_272: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_978, [0], True, dtype = torch.float32);  view_978 = None
        view_979: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_272, [1024]);  sum_272 = None
        convert_element_type_2026: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_979, torch.bfloat16);  view_979 = None
        view_980: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [16, 512, 1024]);  mm_190 = None
        convert_element_type_2027: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_980, torch.float32);  view_980 = None
        convert_element_type_2028: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        convert_element_type_2029: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2026, torch.float32);  convert_element_type_2026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_981: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_94, [16, 512, 1024]);  permute_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_207: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_981, memory_format = torch.contiguous_format);  view_981 = None
        view_982: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_207, [8192, 1024]);  clone_207 = None
        mm_192: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_982, permute_793);  permute_793 = None
        permute_794: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_982, [1, 0])
        mm_193: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_794, view_176);  permute_794 = None
        sum_273: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_982, [0], True, dtype = torch.float32);  view_982 = None
        view_983: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_273, [1024]);  sum_273 = None
        convert_element_type_2034: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_983, torch.bfloat16);  view_983 = None
        view_984: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_192, [16, 512, 1024]);  mm_192 = None
        convert_element_type_2035: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_984, torch.float32);  view_984 = None
        add_295: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2027, convert_element_type_2035);  convert_element_type_2027 = convert_element_type_2035 = None
        convert_element_type_2036: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_193, torch.float32);  mm_193 = None
        convert_element_type_2037: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2034, torch.float32);  convert_element_type_2034 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_208: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_95, memory_format = torch.contiguous_format);  permute_default_95 = None
        view_985: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_208, [16, 512, 1024]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_986: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_985, [8192, 1024]);  view_985 = None
        mm_194: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
        permute_799: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_986, [1, 0])
        mm_195: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
        sum_274: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_986, [0], True, dtype = torch.float32);  view_986 = None
        view_987: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_274, [1024]);  sum_274 = None
        convert_element_type_2042: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_987, torch.bfloat16);  view_987 = None
        view_988: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [16, 512, 1024]);  mm_194 = None
        convert_element_type_2043: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_988, torch.float32);  view_988 = None
        add_296: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_295, convert_element_type_2043);  add_295 = convert_element_type_2043 = None
        convert_element_type_2044: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_2045: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2042, torch.float32);  convert_element_type_2042 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_787: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_296, primals_135);  primals_135 = None
        mul_788: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_787, 1024)
        sum_275: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_787, [2], True)
        mul_789: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_787, mul_107);  mul_787 = None
        sum_276: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True);  mul_789 = None
        mul_790: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, sum_276);  sum_276 = None
        sub_178: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_788, sum_275);  mul_788 = sum_275 = None
        sub_179: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, mul_790);  sub_178 = mul_790 = None
        mul_791: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_99, sub_179);  div_99 = sub_179 = None
        mul_792: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_296, mul_107);  mul_107 = None
        sum_277: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 1]);  mul_792 = None
        sum_278: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
        add_297: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_294, mul_791);  add_294 = mul_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2046: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_297, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2047: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_24, torch.bfloat16);  gt_24 = None
        mul_793: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2047, 1.1111111111111112);  convert_element_type_2047 = None
        mul_794: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2046, mul_793);  convert_element_type_2046 = mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_989: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_794, [8192, 1024]);  mul_794 = None
        mm_196: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
        permute_803: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_989, [1, 0])
        mm_197: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
        sum_279: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_989, [0], True, dtype = torch.float32);  view_989 = None
        view_990: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_279, [1024]);  sum_279 = None
        convert_element_type_2052: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_990, torch.bfloat16);  view_990 = None
        view_991: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_196, [16, 512, 4096]);  mm_196 = None
        convert_element_type_2053: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_2054: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2052, torch.float32);  convert_element_type_2052 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2055: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_991, torch.float32);  view_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [16, 512, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_321: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_173, torch.float32);  view_173 = None
        mul_103: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, 0.7071067811865476)
        erf_7: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_796: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
        mul_797: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, convert_element_type_321)
        mul_798: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_797, -0.5);  mul_797 = None
        exp_43: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_798);  mul_798 = None
        mul_799: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_800: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_321, mul_799);  convert_element_type_321 = mul_799 = None
        add_299: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_796, mul_800);  mul_796 = mul_800 = None
        mul_801: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2055, add_299);  convert_element_type_2055 = add_299 = None
        convert_element_type_2057: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_801, torch.bfloat16);  mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_992: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2057, [8192, 4096]);  convert_element_type_2057 = None
        mm_198: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
        permute_807: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_992, [1, 0])
        mm_199: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
        sum_280: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_992, [0], True, dtype = torch.float32);  view_992 = None
        view_993: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_280, [4096]);  sum_280 = None
        convert_element_type_2062: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_993, torch.bfloat16);  view_993 = None
        view_994: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [16, 512, 1024]);  mm_198 = None
        convert_element_type_2063: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_994, torch.float32);  view_994 = None
        convert_element_type_2064: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_2065: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2062, torch.float32);  convert_element_type_2062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_803: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2063, primals_129);  primals_129 = None
        mul_804: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_803, 1024)
        sum_281: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True)
        mul_805: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_803, mul_100);  mul_803 = None
        sum_282: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
        mul_806: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, sum_282);  sum_282 = None
        sub_181: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_804, sum_281);  mul_804 = sum_281 = None
        sub_182: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_181, mul_806);  sub_181 = mul_806 = None
        mul_807: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_100, sub_182);  div_100 = sub_182 = None
        mul_808: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2063, mul_100);  mul_100 = None
        sum_283: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_808, [0, 1]);  mul_808 = None
        sum_284: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2063, [0, 1]);  convert_element_type_2063 = None
        add_300: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_297, mul_807);  add_297 = mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2066: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_300, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2067: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_23, torch.bfloat16);  gt_23 = None
        mul_809: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2067, 1.1111111111111112);  convert_element_type_2067 = None
        mul_810: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2066, mul_809);  convert_element_type_2066 = mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_995: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_810, [8192, 1024]);  mul_810 = None
        mm_200: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
        permute_811: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_995, [1, 0])
        mm_201: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
        sum_285: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_995, [0], True, dtype = torch.float32);  view_995 = None
        view_996: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_285, [1024]);  sum_285 = None
        convert_element_type_2072: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_996, torch.bfloat16);  view_996 = None
        view_997: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_200, [16, 512, 1024]);  mm_200 = None
        convert_element_type_2073: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_201, torch.float32);  mm_201 = None
        convert_element_type_2074: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2072, torch.float32);  convert_element_type_2072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_998: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_997, [16, 512, 16, 64]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_814: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_211: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_16 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_211, permute_default_96, permute_default_97, permute_default_98, getitem_212, getitem_213, None, None, 512, 512, 0.1, False, getitem_214, getitem_215, scale = 0.125);  clone_211 = permute_default_96 = permute_default_97 = permute_default_98 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = None
        getitem_216: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_16[0]
        getitem_217: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_16[1]
        getitem_218: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_16[2];  _scaled_dot_product_flash_attention_backward_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_101: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_100: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_99: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        clone_213: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_99, memory_format = torch.contiguous_format);  permute_default_99 = None
        view_1005: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_213, [16, 512, 1024]);  clone_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1006: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1005, [8192, 1024]);  view_1005 = None
        mm_202: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1006, permute_821);  permute_821 = None
        permute_822: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1006, [1, 0])
        mm_203: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        sum_287: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1006, [0], True, dtype = torch.float32);  view_1006 = None
        view_1007: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_287, [1024]);  sum_287 = None
        convert_element_type_2090: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1007, torch.bfloat16);  view_1007 = None
        view_1008: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [16, 512, 1024]);  mm_202 = None
        convert_element_type_2091: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1008, torch.float32);  view_1008 = None
        convert_element_type_2092: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None
        convert_element_type_2093: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2090, torch.float32);  convert_element_type_2090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1009: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_100, [16, 512, 1024]);  permute_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_214: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1009, memory_format = torch.contiguous_format);  view_1009 = None
        view_1010: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_214, [8192, 1024]);  clone_214 = None
        mm_204: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1010, permute_826);  permute_826 = None
        permute_827: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_205: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_827, view_154);  permute_827 = None
        sum_288: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1010, [0], True, dtype = torch.float32);  view_1010 = None
        view_1011: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_288, [1024]);  sum_288 = None
        convert_element_type_2098: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1011, torch.bfloat16);  view_1011 = None
        view_1012: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [16, 512, 1024]);  mm_204 = None
        convert_element_type_2099: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1012, torch.float32);  view_1012 = None
        add_301: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2091, convert_element_type_2099);  convert_element_type_2091 = convert_element_type_2099 = None
        convert_element_type_2100: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None
        convert_element_type_2101: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2098, torch.float32);  convert_element_type_2098 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_215: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_101, memory_format = torch.contiguous_format);  permute_default_101 = None
        view_1013: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_215, [16, 512, 1024]);  clone_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1014: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1013, [8192, 1024]);  view_1013 = None
        mm_206: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
        permute_832: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_207: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
        sum_289: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True, dtype = torch.float32);  view_1014 = None
        view_1015: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_289, [1024]);  sum_289 = None
        convert_element_type_2106: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1015, torch.bfloat16);  view_1015 = None
        view_1016: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [16, 512, 1024]);  mm_206 = None
        convert_element_type_2107: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1016, torch.float32);  view_1016 = None
        add_302: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_301, convert_element_type_2107);  add_301 = convert_element_type_2107 = None
        convert_element_type_2108: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_2109: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2106, torch.float32);  convert_element_type_2106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_815: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_302, primals_119);  primals_119 = None
        mul_816: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_815, 1024)
        sum_290: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_815, [2], True)
        mul_817: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_815, mul_94);  mul_815 = None
        sum_291: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True);  mul_817 = None
        mul_818: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_94, sum_291);  sum_291 = None
        sub_184: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_816, sum_290);  mul_816 = sum_290 = None
        sub_185: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_184, mul_818);  sub_184 = mul_818 = None
        mul_819: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_102, sub_185);  div_102 = sub_185 = None
        mul_820: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_302, mul_94);  mul_94 = None
        sum_292: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 1]);  mul_820 = None
        sum_293: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
        add_303: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_300, mul_819);  add_300 = mul_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2110: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_303, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2111: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_21, torch.bfloat16);  gt_21 = None
        mul_821: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2111, 1.1111111111111112);  convert_element_type_2111 = None
        mul_822: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2110, mul_821);  convert_element_type_2110 = mul_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1017: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_822, [8192, 1024]);  mul_822 = None
        mm_208: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
        permute_836: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1017, [1, 0])
        mm_209: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
        sum_294: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True, dtype = torch.float32);  view_1017 = None
        view_1018: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_294, [1024]);  sum_294 = None
        convert_element_type_2116: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1018, torch.bfloat16);  view_1018 = None
        view_1019: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_208, [16, 512, 4096]);  mm_208 = None
        convert_element_type_2117: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_209, torch.float32);  mm_209 = None
        convert_element_type_2118: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2116, torch.float32);  convert_element_type_2116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2119: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1019, torch.float32);  view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [16, 512, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_280: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_151, torch.float32);  view_151 = None
        mul_90: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, 0.7071067811865476)
        erf_6: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_824: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_825: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, convert_element_type_280)
        mul_826: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_825, -0.5);  mul_825 = None
        exp_44: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_826);  mul_826 = None
        mul_827: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_828: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_280, mul_827);  convert_element_type_280 = mul_827 = None
        add_305: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_824, mul_828);  mul_824 = mul_828 = None
        mul_829: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2119, add_305);  convert_element_type_2119 = add_305 = None
        convert_element_type_2121: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_829, torch.bfloat16);  mul_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1020: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2121, [8192, 4096]);  convert_element_type_2121 = None
        mm_210: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
        permute_840: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1020, [1, 0])
        mm_211: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
        sum_295: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True, dtype = torch.float32);  view_1020 = None
        view_1021: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_295, [4096]);  sum_295 = None
        convert_element_type_2126: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1021, torch.bfloat16);  view_1021 = None
        view_1022: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_210, [16, 512, 1024]);  mm_210 = None
        convert_element_type_2127: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1022, torch.float32);  view_1022 = None
        convert_element_type_2128: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_211, torch.float32);  mm_211 = None
        convert_element_type_2129: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2126, torch.float32);  convert_element_type_2126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_831: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2127, primals_113);  primals_113 = None
        mul_832: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_831, 1024)
        sum_296: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
        mul_833: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_831, mul_87);  mul_831 = None
        sum_297: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
        mul_834: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, sum_297);  sum_297 = None
        sub_187: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_832, sum_296);  mul_832 = sum_296 = None
        sub_188: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_187, mul_834);  sub_187 = mul_834 = None
        mul_835: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_103, sub_188);  div_103 = sub_188 = None
        mul_836: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2127, mul_87);  mul_87 = None
        sum_298: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
        sum_299: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2127, [0, 1]);  convert_element_type_2127 = None
        add_306: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_303, mul_835);  add_303 = mul_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2130: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_306, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2131: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_20, torch.bfloat16);  gt_20 = None
        mul_837: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2131, 1.1111111111111112);  convert_element_type_2131 = None
        mul_838: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2130, mul_837);  convert_element_type_2130 = mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1023: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_838, [8192, 1024]);  mul_838 = None
        mm_212: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
        permute_844: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1023, [1, 0])
        mm_213: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
        sum_300: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True, dtype = torch.float32);  view_1023 = None
        view_1024: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_300, [1024]);  sum_300 = None
        convert_element_type_2136: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1024, torch.bfloat16);  view_1024 = None
        view_1025: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_212, [16, 512, 1024]);  mm_212 = None
        convert_element_type_2137: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_213, torch.float32);  mm_213 = None
        convert_element_type_2138: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2136, torch.float32);  convert_element_type_2136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1026: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1025, [16, 512, 16, 64]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_847: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_218: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_17 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_218, permute_default_102, permute_default_103, permute_default_104, getitem_219, getitem_220, None, None, 512, 512, 0.1, False, getitem_221, getitem_222, scale = 0.125);  clone_218 = permute_default_102 = permute_default_103 = permute_default_104 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = None
        getitem_223: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_17[0]
        getitem_224: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_17[1]
        getitem_225: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_17[2];  _scaled_dot_product_flash_attention_backward_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_107: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_106: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_224, [0, 2, 1, 3]);  getitem_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_105: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None
        clone_220: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_105, memory_format = torch.contiguous_format);  permute_default_105 = None
        view_1033: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_220, [16, 512, 1024]);  clone_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1034: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1033, [8192, 1024]);  view_1033 = None
        mm_214: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1034, permute_854);  permute_854 = None
        permute_855: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1034, [1, 0])
        mm_215: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        sum_302: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True, dtype = torch.float32);  view_1034 = None
        view_1035: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_302, [1024]);  sum_302 = None
        convert_element_type_2154: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1035, torch.bfloat16);  view_1035 = None
        view_1036: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_214, [16, 512, 1024]);  mm_214 = None
        convert_element_type_2155: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1036, torch.float32);  view_1036 = None
        convert_element_type_2156: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_215, torch.float32);  mm_215 = None
        convert_element_type_2157: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2154, torch.float32);  convert_element_type_2154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1037: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_106, [16, 512, 1024]);  permute_default_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_221: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1037, memory_format = torch.contiguous_format);  view_1037 = None
        view_1038: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_221, [8192, 1024]);  clone_221 = None
        mm_216: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1038, permute_859);  permute_859 = None
        permute_860: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1038, [1, 0])
        mm_217: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_860, view_132);  permute_860 = None
        sum_303: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1038, [0], True, dtype = torch.float32);  view_1038 = None
        view_1039: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_303, [1024]);  sum_303 = None
        convert_element_type_2162: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1039, torch.bfloat16);  view_1039 = None
        view_1040: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_216, [16, 512, 1024]);  mm_216 = None
        convert_element_type_2163: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1040, torch.float32);  view_1040 = None
        add_307: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2155, convert_element_type_2163);  convert_element_type_2155 = convert_element_type_2163 = None
        convert_element_type_2164: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_217, torch.float32);  mm_217 = None
        convert_element_type_2165: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2162, torch.float32);  convert_element_type_2162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_222: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_107, memory_format = torch.contiguous_format);  permute_default_107 = None
        view_1041: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_222, [16, 512, 1024]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1042: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1041, [8192, 1024]);  view_1041 = None
        mm_218: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
        permute_865: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1042, [1, 0])
        mm_219: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
        sum_304: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True, dtype = torch.float32);  view_1042 = None
        view_1043: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_304, [1024]);  sum_304 = None
        convert_element_type_2170: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1043, torch.bfloat16);  view_1043 = None
        view_1044: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_218, [16, 512, 1024]);  mm_218 = None
        convert_element_type_2171: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1044, torch.float32);  view_1044 = None
        add_308: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_307, convert_element_type_2171);  add_307 = convert_element_type_2171 = None
        convert_element_type_2172: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_219, torch.float32);  mm_219 = None
        convert_element_type_2173: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2170, torch.float32);  convert_element_type_2170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_843: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_308, primals_103);  primals_103 = None
        mul_844: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, 1024)
        sum_305: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_843, [2], True)
        mul_845: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_843, mul_81);  mul_843 = None
        sum_306: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_845, [2], True);  mul_845 = None
        mul_846: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_81, sum_306);  sum_306 = None
        sub_190: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_844, sum_305);  mul_844 = sum_305 = None
        sub_191: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_190, mul_846);  sub_190 = mul_846 = None
        mul_847: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_105, sub_191);  div_105 = sub_191 = None
        mul_848: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_308, mul_81);  mul_81 = None
        sum_307: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_848, [0, 1]);  mul_848 = None
        sum_308: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
        add_309: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_306, mul_847);  add_306 = mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2174: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_309, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2175: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_18, torch.bfloat16);  gt_18 = None
        mul_849: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2175, 1.1111111111111112);  convert_element_type_2175 = None
        mul_850: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2174, mul_849);  convert_element_type_2174 = mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1045: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_850, [8192, 1024]);  mul_850 = None
        mm_220: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
        permute_869: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1045, [1, 0])
        mm_221: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
        sum_309: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True, dtype = torch.float32);  view_1045 = None
        view_1046: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_309, [1024]);  sum_309 = None
        convert_element_type_2180: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1046, torch.bfloat16);  view_1046 = None
        view_1047: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_220, [16, 512, 4096]);  mm_220 = None
        convert_element_type_2181: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_221, torch.float32);  mm_221 = None
        convert_element_type_2182: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2180, torch.float32);  convert_element_type_2180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2183: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1047, torch.float32);  view_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [16, 512, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_239: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_129, torch.float32);  view_129 = None
        mul_77: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, 0.7071067811865476)
        erf_5: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_852: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_853: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, convert_element_type_239)
        mul_854: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
        exp_45: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
        mul_855: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_856: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_239, mul_855);  convert_element_type_239 = mul_855 = None
        add_311: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
        mul_857: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2183, add_311);  convert_element_type_2183 = add_311 = None
        convert_element_type_2185: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_857, torch.bfloat16);  mul_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1048: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2185, [8192, 4096]);  convert_element_type_2185 = None
        mm_222: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
        permute_873: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_223: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
        sum_310: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True, dtype = torch.float32);  view_1048 = None
        view_1049: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_310, [4096]);  sum_310 = None
        convert_element_type_2190: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1049, torch.bfloat16);  view_1049 = None
        view_1050: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_222, [16, 512, 1024]);  mm_222 = None
        convert_element_type_2191: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1050, torch.float32);  view_1050 = None
        convert_element_type_2192: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_223, torch.float32);  mm_223 = None
        convert_element_type_2193: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2190, torch.float32);  convert_element_type_2190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_859: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2191, primals_97);  primals_97 = None
        mul_860: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_859, 1024)
        sum_311: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
        mul_861: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_859, mul_74);  mul_859 = None
        sum_312: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
        mul_862: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_74, sum_312);  sum_312 = None
        sub_193: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_860, sum_311);  mul_860 = sum_311 = None
        sub_194: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_193, mul_862);  sub_193 = mul_862 = None
        mul_863: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_106, sub_194);  div_106 = sub_194 = None
        mul_864: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2191, mul_74);  mul_74 = None
        sum_313: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
        sum_314: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2191, [0, 1]);  convert_element_type_2191 = None
        add_312: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_309, mul_863);  add_309 = mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2194: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_312, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2195: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_17, torch.bfloat16);  gt_17 = None
        mul_865: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2195, 1.1111111111111112);  convert_element_type_2195 = None
        mul_866: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2194, mul_865);  convert_element_type_2194 = mul_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1051: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_866, [8192, 1024]);  mul_866 = None
        mm_224: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
        permute_877: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1051, [1, 0])
        mm_225: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
        sum_315: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True, dtype = torch.float32);  view_1051 = None
        view_1052: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_315, [1024]);  sum_315 = None
        convert_element_type_2200: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1052, torch.bfloat16);  view_1052 = None
        view_1053: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_224, [16, 512, 1024]);  mm_224 = None
        convert_element_type_2201: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_225, torch.float32);  mm_225 = None
        convert_element_type_2202: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2200, torch.float32);  convert_element_type_2200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1054: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1053, [16, 512, 16, 64]);  view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_880: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_225: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_18 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_225, permute_default_108, permute_default_109, permute_default_110, getitem_226, getitem_227, None, None, 512, 512, 0.1, False, getitem_228, getitem_229, scale = 0.125);  clone_225 = permute_default_108 = permute_default_109 = permute_default_110 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = None
        getitem_230: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_18[0]
        getitem_231: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_18[1]
        getitem_232: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_18[2];  _scaled_dot_product_flash_attention_backward_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_113: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_230, [0, 2, 1, 3]);  getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_112: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_231, [0, 2, 1, 3]);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_111: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_232, [0, 2, 1, 3]);  getitem_232 = None
        clone_227: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_111, memory_format = torch.contiguous_format);  permute_default_111 = None
        view_1061: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_227, [16, 512, 1024]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1062: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1061, [8192, 1024]);  view_1061 = None
        mm_226: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1062, permute_887);  permute_887 = None
        permute_888: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_227: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        sum_317: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True, dtype = torch.float32);  view_1062 = None
        view_1063: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_317, [1024]);  sum_317 = None
        convert_element_type_2218: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1063, torch.bfloat16);  view_1063 = None
        view_1064: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_226, [16, 512, 1024]);  mm_226 = None
        convert_element_type_2219: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1064, torch.float32);  view_1064 = None
        convert_element_type_2220: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_227, torch.float32);  mm_227 = None
        convert_element_type_2221: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2218, torch.float32);  convert_element_type_2218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1065: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_112, [16, 512, 1024]);  permute_default_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_228: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1065, memory_format = torch.contiguous_format);  view_1065 = None
        view_1066: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_228, [8192, 1024]);  clone_228 = None
        mm_228: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1066, permute_892);  permute_892 = None
        permute_893: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1066, [1, 0])
        mm_229: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_893, view_110);  permute_893 = None
        sum_318: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True, dtype = torch.float32);  view_1066 = None
        view_1067: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_318, [1024]);  sum_318 = None
        convert_element_type_2226: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1067, torch.bfloat16);  view_1067 = None
        view_1068: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_228, [16, 512, 1024]);  mm_228 = None
        convert_element_type_2227: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1068, torch.float32);  view_1068 = None
        add_313: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2219, convert_element_type_2227);  convert_element_type_2219 = convert_element_type_2227 = None
        convert_element_type_2228: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_229, torch.float32);  mm_229 = None
        convert_element_type_2229: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2226, torch.float32);  convert_element_type_2226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_229: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_113, memory_format = torch.contiguous_format);  permute_default_113 = None
        view_1069: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_229, [16, 512, 1024]);  clone_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1070: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1069, [8192, 1024]);  view_1069 = None
        mm_230: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
        permute_898: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1070, [1, 0])
        mm_231: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
        sum_319: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True, dtype = torch.float32);  view_1070 = None
        view_1071: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
        convert_element_type_2234: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1071, torch.bfloat16);  view_1071 = None
        view_1072: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_230, [16, 512, 1024]);  mm_230 = None
        convert_element_type_2235: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1072, torch.float32);  view_1072 = None
        add_314: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_313, convert_element_type_2235);  add_313 = convert_element_type_2235 = None
        convert_element_type_2236: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_231, torch.float32);  mm_231 = None
        convert_element_type_2237: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2234, torch.float32);  convert_element_type_2234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_871: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_314, primals_87);  primals_87 = None
        mul_872: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_871, 1024)
        sum_320: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_871, [2], True)
        mul_873: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_871, mul_68);  mul_871 = None
        sum_321: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True);  mul_873 = None
        mul_874: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, sum_321);  sum_321 = None
        sub_196: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_872, sum_320);  mul_872 = sum_320 = None
        sub_197: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_196, mul_874);  sub_196 = mul_874 = None
        mul_875: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_108, sub_197);  div_108 = sub_197 = None
        mul_876: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_314, mul_68);  mul_68 = None
        sum_322: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 1]);  mul_876 = None
        sum_323: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
        add_315: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_312, mul_875);  add_312 = mul_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2238: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_315, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2239: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_15, torch.bfloat16);  gt_15 = None
        mul_877: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2239, 1.1111111111111112);  convert_element_type_2239 = None
        mul_878: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2238, mul_877);  convert_element_type_2238 = mul_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1073: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_878, [8192, 1024]);  mul_878 = None
        mm_232: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
        permute_902: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1073, [1, 0])
        mm_233: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
        sum_324: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True, dtype = torch.float32);  view_1073 = None
        view_1074: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_324, [1024]);  sum_324 = None
        convert_element_type_2244: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1074, torch.bfloat16);  view_1074 = None
        view_1075: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_232, [16, 512, 4096]);  mm_232 = None
        convert_element_type_2245: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_233, torch.float32);  mm_233 = None
        convert_element_type_2246: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2244, torch.float32);  convert_element_type_2244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2247: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1075, torch.float32);  view_1075 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [16, 512, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_198: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_107, torch.float32);  view_107 = None
        mul_64: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, 0.7071067811865476)
        erf_4: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_880: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
        mul_881: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, convert_element_type_198)
        mul_882: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_881, -0.5);  mul_881 = None
        exp_46: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_882);  mul_882 = None
        mul_883: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_884: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_198, mul_883);  convert_element_type_198 = mul_883 = None
        add_317: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_880, mul_884);  mul_880 = mul_884 = None
        mul_885: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2247, add_317);  convert_element_type_2247 = add_317 = None
        convert_element_type_2249: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_885, torch.bfloat16);  mul_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1076: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2249, [8192, 4096]);  convert_element_type_2249 = None
        mm_234: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
        permute_906: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_235: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
        sum_325: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True, dtype = torch.float32);  view_1076 = None
        view_1077: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_325, [4096]);  sum_325 = None
        convert_element_type_2254: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1077, torch.bfloat16);  view_1077 = None
        view_1078: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_234, [16, 512, 1024]);  mm_234 = None
        convert_element_type_2255: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1078, torch.float32);  view_1078 = None
        convert_element_type_2256: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_235, torch.float32);  mm_235 = None
        convert_element_type_2257: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2254, torch.float32);  convert_element_type_2254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_887: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2255, primals_81);  primals_81 = None
        mul_888: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_887, 1024)
        sum_326: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
        mul_889: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_887, mul_61);  mul_887 = None
        sum_327: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
        mul_890: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, sum_327);  sum_327 = None
        sub_199: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_888, sum_326);  mul_888 = sum_326 = None
        sub_200: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_199, mul_890);  sub_199 = mul_890 = None
        mul_891: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_109, sub_200);  div_109 = sub_200 = None
        mul_892: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2255, mul_61);  mul_61 = None
        sum_328: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
        sum_329: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2255, [0, 1]);  convert_element_type_2255 = None
        add_318: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_315, mul_891);  add_315 = mul_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2258: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_318, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2259: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_14, torch.bfloat16);  gt_14 = None
        mul_893: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2259, 1.1111111111111112);  convert_element_type_2259 = None
        mul_894: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2258, mul_893);  convert_element_type_2258 = mul_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1079: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_894, [8192, 1024]);  mul_894 = None
        mm_236: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
        permute_910: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1079, [1, 0])
        mm_237: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
        sum_330: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True, dtype = torch.float32);  view_1079 = None
        view_1080: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_330, [1024]);  sum_330 = None
        convert_element_type_2264: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1080, torch.bfloat16);  view_1080 = None
        view_1081: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_236, [16, 512, 1024]);  mm_236 = None
        convert_element_type_2265: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_237, torch.float32);  mm_237 = None
        convert_element_type_2266: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2264, torch.float32);  convert_element_type_2264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1082: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1081, [16, 512, 16, 64]);  view_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_913: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_232: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_19 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_232, permute_default_114, permute_default_115, permute_default_116, getitem_233, getitem_234, None, None, 512, 512, 0.1, False, getitem_235, getitem_236, scale = 0.125);  clone_232 = permute_default_114 = permute_default_115 = permute_default_116 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = None
        getitem_237: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_19[0]
        getitem_238: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_19[1]
        getitem_239: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_19[2];  _scaled_dot_product_flash_attention_backward_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_119: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_237, [0, 2, 1, 3]);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_118: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_117: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_239, [0, 2, 1, 3]);  getitem_239 = None
        clone_234: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_117, memory_format = torch.contiguous_format);  permute_default_117 = None
        view_1089: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_234, [16, 512, 1024]);  clone_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1090: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1089, [8192, 1024]);  view_1089 = None
        mm_238: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1090, permute_920);  permute_920 = None
        permute_921: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1090, [1, 0])
        mm_239: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        sum_332: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True, dtype = torch.float32);  view_1090 = None
        view_1091: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_332, [1024]);  sum_332 = None
        convert_element_type_2282: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1091, torch.bfloat16);  view_1091 = None
        view_1092: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_238, [16, 512, 1024]);  mm_238 = None
        convert_element_type_2283: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1092, torch.float32);  view_1092 = None
        convert_element_type_2284: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_239, torch.float32);  mm_239 = None
        convert_element_type_2285: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2282, torch.float32);  convert_element_type_2282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1093: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_118, [16, 512, 1024]);  permute_default_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_235: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1093, memory_format = torch.contiguous_format);  view_1093 = None
        view_1094: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_235, [8192, 1024]);  clone_235 = None
        mm_240: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1094, permute_925);  permute_925 = None
        permute_926: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1094, [1, 0])
        mm_241: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_926, view_88);  permute_926 = None
        sum_333: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1094, [0], True, dtype = torch.float32);  view_1094 = None
        view_1095: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
        convert_element_type_2290: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1095, torch.bfloat16);  view_1095 = None
        view_1096: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_240, [16, 512, 1024]);  mm_240 = None
        convert_element_type_2291: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1096, torch.float32);  view_1096 = None
        add_319: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2283, convert_element_type_2291);  convert_element_type_2283 = convert_element_type_2291 = None
        convert_element_type_2292: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_241, torch.float32);  mm_241 = None
        convert_element_type_2293: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2290, torch.float32);  convert_element_type_2290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_236: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_119, memory_format = torch.contiguous_format);  permute_default_119 = None
        view_1097: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_236, [16, 512, 1024]);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1098: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1097, [8192, 1024]);  view_1097 = None
        mm_242: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
        permute_931: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1098, [1, 0])
        mm_243: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
        sum_334: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True, dtype = torch.float32);  view_1098 = None
        view_1099: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_334, [1024]);  sum_334 = None
        convert_element_type_2298: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1099, torch.bfloat16);  view_1099 = None
        view_1100: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_242, [16, 512, 1024]);  mm_242 = None
        convert_element_type_2299: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1100, torch.float32);  view_1100 = None
        add_320: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_319, convert_element_type_2299);  add_319 = convert_element_type_2299 = None
        convert_element_type_2300: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_243, torch.float32);  mm_243 = None
        convert_element_type_2301: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2298, torch.float32);  convert_element_type_2298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_899: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_320, primals_71);  primals_71 = None
        mul_900: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_899, 1024)
        sum_335: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_899, [2], True)
        mul_901: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_899, mul_55);  mul_899 = None
        sum_336: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_901, [2], True);  mul_901 = None
        mul_902: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, sum_336);  sum_336 = None
        sub_202: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_900, sum_335);  mul_900 = sum_335 = None
        sub_203: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, mul_902);  sub_202 = mul_902 = None
        mul_903: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_111, sub_203);  div_111 = sub_203 = None
        mul_904: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_320, mul_55);  mul_55 = None
        sum_337: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 1]);  mul_904 = None
        sum_338: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None
        add_321: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_318, mul_903);  add_318 = mul_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2302: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_321, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2303: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_12, torch.bfloat16);  gt_12 = None
        mul_905: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2303, 1.1111111111111112);  convert_element_type_2303 = None
        mul_906: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2302, mul_905);  convert_element_type_2302 = mul_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1101: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_906, [8192, 1024]);  mul_906 = None
        mm_244: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
        permute_935: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_245: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
        sum_339: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True, dtype = torch.float32);  view_1101 = None
        view_1102: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_339, [1024]);  sum_339 = None
        convert_element_type_2308: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1102, torch.bfloat16);  view_1102 = None
        view_1103: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_244, [16, 512, 4096]);  mm_244 = None
        convert_element_type_2309: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_245, torch.float32);  mm_245 = None
        convert_element_type_2310: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2308, torch.float32);  convert_element_type_2308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2311: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1103, torch.float32);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [16, 512, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_157: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_85, torch.float32);  view_85 = None
        mul_51: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, 0.7071067811865476)
        erf_3: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_908: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_909: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, convert_element_type_157)
        mul_910: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_909, -0.5);  mul_909 = None
        exp_47: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_910);  mul_910 = None
        mul_911: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_912: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_157, mul_911);  convert_element_type_157 = mul_911 = None
        add_323: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_908, mul_912);  mul_908 = mul_912 = None
        mul_913: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2311, add_323);  convert_element_type_2311 = add_323 = None
        convert_element_type_2313: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_913, torch.bfloat16);  mul_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1104: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2313, [8192, 4096]);  convert_element_type_2313 = None
        mm_246: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
        permute_939: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_247: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
        sum_340: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True, dtype = torch.float32);  view_1104 = None
        view_1105: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_340, [4096]);  sum_340 = None
        convert_element_type_2318: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1105, torch.bfloat16);  view_1105 = None
        view_1106: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_246, [16, 512, 1024]);  mm_246 = None
        convert_element_type_2319: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1106, torch.float32);  view_1106 = None
        convert_element_type_2320: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_247, torch.float32);  mm_247 = None
        convert_element_type_2321: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2318, torch.float32);  convert_element_type_2318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_915: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2319, primals_65);  primals_65 = None
        mul_916: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_915, 1024)
        sum_341: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_915, [2], True)
        mul_917: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_915, mul_48);  mul_915 = None
        sum_342: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_917, [2], True);  mul_917 = None
        mul_918: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, sum_342);  sum_342 = None
        sub_205: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_916, sum_341);  mul_916 = sum_341 = None
        sub_206: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_205, mul_918);  sub_205 = mul_918 = None
        mul_919: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_112, sub_206);  div_112 = sub_206 = None
        mul_920: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2319, mul_48);  mul_48 = None
        sum_343: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_920, [0, 1]);  mul_920 = None
        sum_344: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2319, [0, 1]);  convert_element_type_2319 = None
        add_324: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_321, mul_919);  add_321 = mul_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2322: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2323: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_11, torch.bfloat16);  gt_11 = None
        mul_921: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2323, 1.1111111111111112);  convert_element_type_2323 = None
        mul_922: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2322, mul_921);  convert_element_type_2322 = mul_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1107: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_922, [8192, 1024]);  mul_922 = None
        mm_248: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
        permute_943: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1107, [1, 0])
        mm_249: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
        sum_345: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True, dtype = torch.float32);  view_1107 = None
        view_1108: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_345, [1024]);  sum_345 = None
        convert_element_type_2328: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1108, torch.bfloat16);  view_1108 = None
        view_1109: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_248, [16, 512, 1024]);  mm_248 = None
        convert_element_type_2329: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_249, torch.float32);  mm_249 = None
        convert_element_type_2330: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2328, torch.float32);  convert_element_type_2328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1110: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1109, [16, 512, 16, 64]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_946: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_239: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_946, memory_format = torch.contiguous_format);  permute_946 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_20 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_239, permute_default_120, permute_default_121, permute_default_122, getitem_240, getitem_241, None, None, 512, 512, 0.1, False, getitem_242, getitem_243, scale = 0.125);  clone_239 = permute_default_120 = permute_default_121 = permute_default_122 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = None
        getitem_244: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_20[0]
        getitem_245: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_20[1]
        getitem_246: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_20[2];  _scaled_dot_product_flash_attention_backward_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_125: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3]);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_124: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_245, [0, 2, 1, 3]);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_123: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_246, [0, 2, 1, 3]);  getitem_246 = None
        clone_241: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_123, memory_format = torch.contiguous_format);  permute_default_123 = None
        view_1117: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_241, [16, 512, 1024]);  clone_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1118: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1117, [8192, 1024]);  view_1117 = None
        mm_250: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1118, permute_953);  permute_953 = None
        permute_954: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_251: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        sum_347: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True, dtype = torch.float32);  view_1118 = None
        view_1119: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_347, [1024]);  sum_347 = None
        convert_element_type_2346: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1119, torch.bfloat16);  view_1119 = None
        view_1120: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_250, [16, 512, 1024]);  mm_250 = None
        convert_element_type_2347: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1120, torch.float32);  view_1120 = None
        convert_element_type_2348: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_251, torch.float32);  mm_251 = None
        convert_element_type_2349: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2346, torch.float32);  convert_element_type_2346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1121: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_124, [16, 512, 1024]);  permute_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_242: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1121, memory_format = torch.contiguous_format);  view_1121 = None
        view_1122: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_242, [8192, 1024]);  clone_242 = None
        mm_252: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1122, permute_958);  permute_958 = None
        permute_959: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1122, [1, 0])
        mm_253: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_959, view_66);  permute_959 = None
        sum_348: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1122, [0], True, dtype = torch.float32);  view_1122 = None
        view_1123: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_348, [1024]);  sum_348 = None
        convert_element_type_2354: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1123, torch.bfloat16);  view_1123 = None
        view_1124: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_252, [16, 512, 1024]);  mm_252 = None
        convert_element_type_2355: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1124, torch.float32);  view_1124 = None
        add_325: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2347, convert_element_type_2355);  convert_element_type_2347 = convert_element_type_2355 = None
        convert_element_type_2356: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_253, torch.float32);  mm_253 = None
        convert_element_type_2357: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2354, torch.float32);  convert_element_type_2354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_243: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_125, memory_format = torch.contiguous_format);  permute_default_125 = None
        view_1125: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_243, [16, 512, 1024]);  clone_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1126: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1125, [8192, 1024]);  view_1125 = None
        mm_254: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
        permute_964: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_255: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
        sum_349: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True, dtype = torch.float32);  view_1126 = None
        view_1127: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_349, [1024]);  sum_349 = None
        convert_element_type_2362: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1127, torch.bfloat16);  view_1127 = None
        view_1128: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_254, [16, 512, 1024]);  mm_254 = None
        convert_element_type_2363: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1128, torch.float32);  view_1128 = None
        add_326: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_325, convert_element_type_2363);  add_325 = convert_element_type_2363 = None
        convert_element_type_2364: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_255, torch.float32);  mm_255 = None
        convert_element_type_2365: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2362, torch.float32);  convert_element_type_2362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_927: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_326, primals_55);  primals_55 = None
        mul_928: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_927, 1024)
        sum_350: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_927, [2], True)
        mul_929: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_927, mul_42);  mul_927 = None
        sum_351: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_929, [2], True);  mul_929 = None
        mul_930: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_42, sum_351);  sum_351 = None
        sub_208: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_928, sum_350);  mul_928 = sum_350 = None
        sub_209: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_208, mul_930);  sub_208 = mul_930 = None
        mul_931: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_114, sub_209);  div_114 = sub_209 = None
        mul_932: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_326, mul_42);  mul_42 = None
        sum_352: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_932, [0, 1]);  mul_932 = None
        sum_353: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None
        add_327: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_324, mul_931);  add_324 = mul_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2366: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_327, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2367: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_9, torch.bfloat16);  gt_9 = None
        mul_933: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2367, 1.1111111111111112);  convert_element_type_2367 = None
        mul_934: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2366, mul_933);  convert_element_type_2366 = mul_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1129: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_934, [8192, 1024]);  mul_934 = None
        mm_256: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
        permute_968: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_257: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
        sum_354: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True, dtype = torch.float32);  view_1129 = None
        view_1130: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_354, [1024]);  sum_354 = None
        convert_element_type_2372: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1130, torch.bfloat16);  view_1130 = None
        view_1131: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_256, [16, 512, 4096]);  mm_256 = None
        convert_element_type_2373: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_257, torch.float32);  mm_257 = None
        convert_element_type_2374: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2372, torch.float32);  convert_element_type_2372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2375: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1131, torch.float32);  view_1131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [16, 512, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_116: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_63, torch.float32);  view_63 = None
        mul_38: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, 0.7071067811865476)
        erf_2: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_936: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_937: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, convert_element_type_116)
        mul_938: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_937, -0.5);  mul_937 = None
        exp_48: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_938);  mul_938 = None
        mul_939: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_940: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_116, mul_939);  convert_element_type_116 = mul_939 = None
        add_329: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_936, mul_940);  mul_936 = mul_940 = None
        mul_941: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2375, add_329);  convert_element_type_2375 = add_329 = None
        convert_element_type_2377: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_941, torch.bfloat16);  mul_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1132: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2377, [8192, 4096]);  convert_element_type_2377 = None
        mm_258: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
        permute_972: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_259: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
        sum_355: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True, dtype = torch.float32);  view_1132 = None
        view_1133: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_355, [4096]);  sum_355 = None
        convert_element_type_2382: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1133, torch.bfloat16);  view_1133 = None
        view_1134: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_258, [16, 512, 1024]);  mm_258 = None
        convert_element_type_2383: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1134, torch.float32);  view_1134 = None
        convert_element_type_2384: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_259, torch.float32);  mm_259 = None
        convert_element_type_2385: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2382, torch.float32);  convert_element_type_2382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_943: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2383, primals_49);  primals_49 = None
        mul_944: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_943, 1024)
        sum_356: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_943, [2], True)
        mul_945: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_943, mul_35);  mul_943 = None
        sum_357: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_945, [2], True);  mul_945 = None
        mul_946: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_35, sum_357);  sum_357 = None
        sub_211: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_944, sum_356);  mul_944 = sum_356 = None
        sub_212: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_211, mul_946);  sub_211 = mul_946 = None
        mul_947: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_115, sub_212);  div_115 = sub_212 = None
        mul_948: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2383, mul_35);  mul_35 = None
        sum_358: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_948, [0, 1]);  mul_948 = None
        sum_359: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2383, [0, 1]);  convert_element_type_2383 = None
        add_330: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_327, mul_947);  add_327 = mul_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2386: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_330, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2387: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_8, torch.bfloat16);  gt_8 = None
        mul_949: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2387, 1.1111111111111112);  convert_element_type_2387 = None
        mul_950: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2386, mul_949);  convert_element_type_2386 = mul_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1135: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_950, [8192, 1024]);  mul_950 = None
        mm_260: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
        permute_976: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_261: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
        sum_360: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True, dtype = torch.float32);  view_1135 = None
        view_1136: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_360, [1024]);  sum_360 = None
        convert_element_type_2392: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1136, torch.bfloat16);  view_1136 = None
        view_1137: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_260, [16, 512, 1024]);  mm_260 = None
        convert_element_type_2393: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_261, torch.float32);  mm_261 = None
        convert_element_type_2394: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2392, torch.float32);  convert_element_type_2392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1138: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1137, [16, 512, 16, 64]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_979: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_246: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_979, memory_format = torch.contiguous_format);  permute_979 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_21 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_246, permute_default_126, permute_default_127, permute_default_128, getitem_247, getitem_248, None, None, 512, 512, 0.1, False, getitem_249, getitem_250, scale = 0.125);  clone_246 = permute_default_126 = permute_default_127 = permute_default_128 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = None
        getitem_251: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_21[0]
        getitem_252: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_21[1]
        getitem_253: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_21[2];  _scaled_dot_product_flash_attention_backward_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_131: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_251, [0, 2, 1, 3]);  getitem_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_130: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_129: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_253, [0, 2, 1, 3]);  getitem_253 = None
        clone_248: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_129, memory_format = torch.contiguous_format);  permute_default_129 = None
        view_1145: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_248, [16, 512, 1024]);  clone_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1146: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1145, [8192, 1024]);  view_1145 = None
        mm_262: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1146, permute_986);  permute_986 = None
        permute_987: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1146, [1, 0])
        mm_263: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        sum_362: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1146, [0], True, dtype = torch.float32);  view_1146 = None
        view_1147: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_362, [1024]);  sum_362 = None
        convert_element_type_2410: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1147, torch.bfloat16);  view_1147 = None
        view_1148: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_262, [16, 512, 1024]);  mm_262 = None
        convert_element_type_2411: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1148, torch.float32);  view_1148 = None
        convert_element_type_2412: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_263, torch.float32);  mm_263 = None
        convert_element_type_2413: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2410, torch.float32);  convert_element_type_2410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1149: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_130, [16, 512, 1024]);  permute_default_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_249: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1149, memory_format = torch.contiguous_format);  view_1149 = None
        view_1150: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_249, [8192, 1024]);  clone_249 = None
        mm_264: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1150, permute_991);  permute_991 = None
        permute_992: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1150, [1, 0])
        mm_265: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_992, view_44);  permute_992 = None
        sum_363: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1150, [0], True, dtype = torch.float32);  view_1150 = None
        view_1151: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_363, [1024]);  sum_363 = None
        convert_element_type_2418: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1151, torch.bfloat16);  view_1151 = None
        view_1152: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_264, [16, 512, 1024]);  mm_264 = None
        convert_element_type_2419: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1152, torch.float32);  view_1152 = None
        add_331: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2411, convert_element_type_2419);  convert_element_type_2411 = convert_element_type_2419 = None
        convert_element_type_2420: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_265, torch.float32);  mm_265 = None
        convert_element_type_2421: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2418, torch.float32);  convert_element_type_2418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_250: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_131, memory_format = torch.contiguous_format);  permute_default_131 = None
        view_1153: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_250, [16, 512, 1024]);  clone_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1154: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1153, [8192, 1024]);  view_1153 = None
        mm_266: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
        permute_997: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_267: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
        sum_364: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True, dtype = torch.float32);  view_1154 = None
        view_1155: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_364, [1024]);  sum_364 = None
        convert_element_type_2426: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1155, torch.bfloat16);  view_1155 = None
        view_1156: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_266, [16, 512, 1024]);  mm_266 = None
        convert_element_type_2427: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1156, torch.float32);  view_1156 = None
        add_332: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_331, convert_element_type_2427);  add_331 = convert_element_type_2427 = None
        convert_element_type_2428: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_267, torch.float32);  mm_267 = None
        convert_element_type_2429: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2426, torch.float32);  convert_element_type_2426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_955: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_332, primals_39);  primals_39 = None
        mul_956: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_955, 1024)
        sum_365: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_955, [2], True)
        mul_957: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_955, mul_29);  mul_955 = None
        sum_366: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_957, [2], True);  mul_957 = None
        mul_958: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, sum_366);  sum_366 = None
        sub_214: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_956, sum_365);  mul_956 = sum_365 = None
        sub_215: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_214, mul_958);  sub_214 = mul_958 = None
        mul_959: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_117, sub_215);  div_117 = sub_215 = None
        mul_960: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_332, mul_29);  mul_29 = None
        sum_367: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 1]);  mul_960 = None
        sum_368: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None
        add_333: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_330, mul_959);  add_330 = mul_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2430: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_333, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2431: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_6, torch.bfloat16);  gt_6 = None
        mul_961: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2431, 1.1111111111111112);  convert_element_type_2431 = None
        mul_962: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2430, mul_961);  convert_element_type_2430 = mul_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1157: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_962, [8192, 1024]);  mul_962 = None
        mm_268: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
        permute_1001: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1157, [1, 0])
        mm_269: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
        sum_369: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True, dtype = torch.float32);  view_1157 = None
        view_1158: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_369, [1024]);  sum_369 = None
        convert_element_type_2436: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1158, torch.bfloat16);  view_1158 = None
        view_1159: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_268, [16, 512, 4096]);  mm_268 = None
        convert_element_type_2437: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_269, torch.float32);  mm_269 = None
        convert_element_type_2438: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2436, torch.float32);  convert_element_type_2436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2439: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1159, torch.float32);  view_1159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [16, 512, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_75: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_41, torch.float32);  view_41 = None
        mul_25: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, 0.7071067811865476)
        erf_1: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_964: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
        mul_965: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, convert_element_type_75)
        mul_966: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_965, -0.5);  mul_965 = None
        exp_49: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_966);  mul_966 = None
        mul_967: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
        mul_968: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_75, mul_967);  convert_element_type_75 = mul_967 = None
        add_335: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_964, mul_968);  mul_964 = mul_968 = None
        mul_969: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2439, add_335);  convert_element_type_2439 = add_335 = None
        convert_element_type_2441: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_969, torch.bfloat16);  mul_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1160: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2441, [8192, 4096]);  convert_element_type_2441 = None
        mm_270: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
        permute_1005: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1160, [1, 0])
        mm_271: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
        sum_370: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True, dtype = torch.float32);  view_1160 = None
        view_1161: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_370, [4096]);  sum_370 = None
        convert_element_type_2446: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1161, torch.bfloat16);  view_1161 = None
        view_1162: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_270, [16, 512, 1024]);  mm_270 = None
        convert_element_type_2447: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1162, torch.float32);  view_1162 = None
        convert_element_type_2448: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_271, torch.float32);  mm_271 = None
        convert_element_type_2449: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2446, torch.float32);  convert_element_type_2446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_971: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2447, primals_33);  primals_33 = None
        mul_972: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_971, 1024)
        sum_371: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_971, [2], True)
        mul_973: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_971, mul_22);  mul_971 = None
        sum_372: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_973, [2], True);  mul_973 = None
        mul_974: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, sum_372);  sum_372 = None
        sub_217: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_972, sum_371);  mul_972 = sum_371 = None
        sub_218: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_217, mul_974);  sub_217 = mul_974 = None
        mul_975: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_118, sub_218);  div_118 = sub_218 = None
        mul_976: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2447, mul_22);  mul_22 = None
        sum_373: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 1]);  mul_976 = None
        sum_374: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2447, [0, 1]);  convert_element_type_2447 = None
        add_336: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_333, mul_975);  add_333 = mul_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2450: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_336, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2451: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_5, torch.bfloat16);  gt_5 = None
        mul_977: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2451, 1.1111111111111112);  convert_element_type_2451 = None
        mul_978: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2450, mul_977);  convert_element_type_2450 = mul_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1163: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_978, [8192, 1024]);  mul_978 = None
        mm_272: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
        permute_1009: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_273: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
        sum_375: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True, dtype = torch.float32);  view_1163 = None
        view_1164: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_375, [1024]);  sum_375 = None
        convert_element_type_2456: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1164, torch.bfloat16);  view_1164 = None
        view_1165: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_272, [16, 512, 1024]);  mm_272 = None
        convert_element_type_2457: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_273, torch.float32);  mm_273 = None
        convert_element_type_2458: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2456, torch.float32);  convert_element_type_2456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1166: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1165, [16, 512, 16, 64]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_1012: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_253: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_22 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_253, permute_default_132, permute_default_133, permute_default_134, getitem_254, getitem_255, None, None, 512, 512, 0.1, False, getitem_256, getitem_257, scale = 0.125);  clone_253 = permute_default_132 = permute_default_133 = permute_default_134 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = None
        getitem_258: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_22[0]
        getitem_259: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_22[1]
        getitem_260: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_22[2];  _scaled_dot_product_flash_attention_backward_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_137: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3]);  getitem_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_136: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_259, [0, 2, 1, 3]);  getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_135: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        clone_255: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_135, memory_format = torch.contiguous_format);  permute_default_135 = None
        view_1173: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_255, [16, 512, 1024]);  clone_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1174: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1173, [8192, 1024]);  view_1173 = None
        mm_274: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1174, permute_1019);  permute_1019 = None
        permute_1020: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1174, [1, 0])
        mm_275: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        sum_377: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True, dtype = torch.float32);  view_1174 = None
        view_1175: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_377, [1024]);  sum_377 = None
        convert_element_type_2474: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1175, torch.bfloat16);  view_1175 = None
        view_1176: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_274, [16, 512, 1024]);  mm_274 = None
        convert_element_type_2475: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1176, torch.float32);  view_1176 = None
        convert_element_type_2476: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_275, torch.float32);  mm_275 = None
        convert_element_type_2477: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2474, torch.float32);  convert_element_type_2474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1177: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_136, [16, 512, 1024]);  permute_default_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_256: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1177, memory_format = torch.contiguous_format);  view_1177 = None
        view_1178: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_256, [8192, 1024]);  clone_256 = None
        mm_276: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1178, permute_1024);  permute_1024 = None
        permute_1025: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1178, [1, 0])
        mm_277: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1025, view_22);  permute_1025 = None
        sum_378: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1178, [0], True, dtype = torch.float32);  view_1178 = None
        view_1179: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_378, [1024]);  sum_378 = None
        convert_element_type_2482: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1179, torch.bfloat16);  view_1179 = None
        view_1180: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_276, [16, 512, 1024]);  mm_276 = None
        convert_element_type_2483: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1180, torch.float32);  view_1180 = None
        add_337: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2475, convert_element_type_2483);  convert_element_type_2475 = convert_element_type_2483 = None
        convert_element_type_2484: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_277, torch.float32);  mm_277 = None
        convert_element_type_2485: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2482, torch.float32);  convert_element_type_2482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_257: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_137, memory_format = torch.contiguous_format);  permute_default_137 = None
        view_1181: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_257, [16, 512, 1024]);  clone_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1182: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1181, [8192, 1024]);  view_1181 = None
        mm_278: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
        permute_1030: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_279: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
        sum_379: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True, dtype = torch.float32);  view_1182 = None
        view_1183: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_379, [1024]);  sum_379 = None
        convert_element_type_2490: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1183, torch.bfloat16);  view_1183 = None
        view_1184: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_278, [16, 512, 1024]);  mm_278 = None
        convert_element_type_2491: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1184, torch.float32);  view_1184 = None
        add_338: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_337, convert_element_type_2491);  add_337 = convert_element_type_2491 = None
        convert_element_type_2492: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_279, torch.float32);  mm_279 = None
        convert_element_type_2493: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2490, torch.float32);  convert_element_type_2490 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_983: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_338, primals_23);  primals_23 = None
        mul_984: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_983, 1024)
        sum_380: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_983, [2], True)
        mul_985: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_983, mul_16);  mul_983 = None
        sum_381: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_985, [2], True);  mul_985 = None
        mul_986: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_381);  sum_381 = None
        sub_220: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_984, sum_380);  mul_984 = sum_380 = None
        sub_221: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_220, mul_986);  sub_220 = mul_986 = None
        mul_987: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_120, sub_221);  div_120 = sub_221 = None
        mul_988: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_338, mul_16);  mul_16 = None
        sum_382: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_988, [0, 1]);  mul_988 = None
        sum_383: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
        add_339: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_336, mul_987);  add_336 = mul_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:263 in forward, code: return input_tensor + hidden_states
        convert_element_type_2494: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_339, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2495: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_3, torch.bfloat16);  gt_3 = None
        mul_989: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2495, 1.1111111111111112);  convert_element_type_2495 = None
        mul_990: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2494, mul_989);  convert_element_type_2494 = mul_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1185: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_990, [8192, 1024]);  mul_990 = None
        mm_280: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
        permute_1034: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_281: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
        sum_384: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True, dtype = torch.float32);  view_1185 = None
        view_1186: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_384, [1024]);  sum_384 = None
        convert_element_type_2500: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1186, torch.bfloat16);  view_1186 = None
        view_1187: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(mm_280, [16, 512, 4096]);  mm_280 = None
        convert_element_type_2501: "f32[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_281, torch.float32);  mm_281 = None
        convert_element_type_2502: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2500, torch.float32);  convert_element_type_2500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_2503: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1187, torch.float32);  view_1187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [16, 512, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_34: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_19, torch.float32);  view_19 = None
        mul_12: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, 0.7071067811865476)
        erf: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_992: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_993: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, convert_element_type_34)
        mul_994: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_993, -0.5);  mul_993 = None
        exp_50: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.exp.default(mul_994);  mul_994 = None
        mul_995: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_996: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_34, mul_995);  convert_element_type_34 = mul_995 = None
        add_341: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_992, mul_996);  mul_992 = mul_996 = None
        mul_997: "f32[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2503, add_341);  convert_element_type_2503 = add_341 = None
        convert_element_type_2505: "bf16[16, 512, 4096][2097152, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_997, torch.bfloat16);  mul_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1188: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_2505, [8192, 4096]);  convert_element_type_2505 = None
        mm_282: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
        permute_1038: "bf16[4096, 8192][1, 4096]cuda:0" = torch.ops.aten.permute.default(view_1188, [1, 0])
        mm_283: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
        sum_385: "f32[1, 4096][4096, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True, dtype = torch.float32);  view_1188 = None
        view_1189: "f32[4096][1]cuda:0" = torch.ops.aten.reshape.default(sum_385, [4096]);  sum_385 = None
        convert_element_type_2510: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1189, torch.bfloat16);  view_1189 = None
        view_1190: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_282, [16, 512, 1024]);  mm_282 = None
        convert_element_type_2511: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1190, torch.float32);  view_1190 = None
        convert_element_type_2512: "f32[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_283, torch.float32);  mm_283 = None
        convert_element_type_2513: "f32[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2510, torch.float32);  convert_element_type_2510 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_999: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2511, primals_17);  primals_17 = None
        mul_1000: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_999, 1024)
        sum_386: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_999, [2], True)
        mul_1001: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_999, mul_9);  mul_999 = None
        sum_387: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1001, [2], True);  mul_1001 = None
        mul_1002: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, sum_387);  sum_387 = None
        sub_223: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1000, sum_386);  mul_1000 = sum_386 = None
        sub_224: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, mul_1002);  sub_223 = mul_1002 = None
        mul_1003: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_121, sub_224);  div_121 = sub_224 = None
        mul_1004: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2511, mul_9);  mul_9 = None
        sum_388: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 1]);  mul_1004 = None
        sum_389: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_2511, [0, 1]);  convert_element_type_2511 = None
        add_342: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_339, mul_1003);  add_339 = mul_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:204 in forward, code: return residual + hidden_states
        convert_element_type_2514: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_342, torch.bfloat16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2515: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt_2, torch.bfloat16);  gt_2 = None
        mul_1005: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2515, 1.1111111111111112);  convert_element_type_2515 = None
        mul_1006: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2514, mul_1005);  convert_element_type_2514 = mul_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1191: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(mul_1006, [8192, 1024]);  mul_1006 = None
        mm_284: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
        permute_1042: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1191, [1, 0])
        mm_285: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
        sum_390: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True, dtype = torch.float32);  view_1191 = None
        view_1192: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_390, [1024]);  sum_390 = None
        convert_element_type_2520: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1192, torch.bfloat16);  view_1192 = None
        view_1193: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_284, [16, 512, 1024]);  mm_284 = None
        convert_element_type_2521: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_285, torch.float32);  mm_285 = None
        convert_element_type_2522: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2520, torch.float32);  convert_element_type_2520 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1194: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(view_1193, [16, 512, 16, 64]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_1045: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_260: "bf16[16, 16, 512, 64][524288, 32768, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_1045, memory_format = torch.contiguous_format);  permute_1045 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_flash_attention_backward_default_23 = torch.ops.aten._scaled_dot_product_flash_attention_backward.default(clone_260, permute_default_138, permute_default_139, permute_default_140, getitem_261, getitem_262, None, None, 512, 512, 0.1, False, getitem_263, getitem_264, scale = 0.125);  clone_260 = permute_default_138 = permute_default_139 = permute_default_140 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = None
        getitem_265: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_23[0]
        getitem_266: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_23[1]
        getitem_267: "bf16[16, 16, 512, 64][524288, 64, 1024, 1]cuda:0" = _scaled_dot_product_flash_attention_backward_default_23[2];  _scaled_dot_product_flash_attention_backward_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_143: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_265, [0, 2, 1, 3]);  getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_142: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_266, [0, 2, 1, 3]);  getitem_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_141: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_267, [0, 2, 1, 3]);  getitem_267 = None
        clone_262: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_141, memory_format = torch.contiguous_format);  permute_default_141 = None
        view_1201: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_262, [16, 512, 1024]);  clone_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1202: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1201, [8192, 1024]);  view_1201 = None
        mm_286: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1202, permute_1052);  permute_1052 = None
        permute_1053: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1202, [1, 0])
        mm_287: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        sum_392: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1202, [0], True, dtype = torch.float32);  view_1202 = None
        view_1203: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_392, [1024]);  sum_392 = None
        convert_element_type_2538: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1203, torch.bfloat16);  view_1203 = None
        view_1204: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_286, [16, 512, 1024]);  mm_286 = None
        convert_element_type_2539: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1204, torch.float32);  view_1204 = None
        convert_element_type_2540: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_287, torch.float32);  mm_287 = None
        convert_element_type_2541: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2538, torch.float32);  convert_element_type_2538 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1205: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_default_142, [16, 512, 1024]);  permute_default_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_263: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(view_1205, memory_format = torch.contiguous_format);  view_1205 = None
        view_1206: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_263, [8192, 1024]);  clone_263 = None
        mm_288: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1206, permute_1057);  permute_1057 = None
        permute_1058: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1206, [1, 0])
        mm_289: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1058, view);  permute_1058 = None
        sum_393: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1206, [0], True, dtype = torch.float32);  view_1206 = None
        view_1207: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_393, [1024]);  sum_393 = None
        convert_element_type_2546: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1207, torch.bfloat16);  view_1207 = None
        view_1208: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_288, [16, 512, 1024]);  mm_288 = None
        convert_element_type_2547: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1208, torch.float32);  view_1208 = None
        add_343: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_2539, convert_element_type_2547);  convert_element_type_2539 = convert_element_type_2547 = None
        convert_element_type_2548: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_289, torch.float32);  mm_289 = None
        convert_element_type_2549: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2546, torch.float32);  convert_element_type_2546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_264: "bf16[16, 512, 16, 64][524288, 1024, 64, 1]cuda:0" = torch.ops.aten.clone.default(permute_default_143, memory_format = torch.contiguous_format);  permute_default_143 = None
        view_1209: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_264, [16, 512, 1024]);  clone_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1210: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(view_1209, [8192, 1024]);  view_1209 = None
        mm_290: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
        permute_1063: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_291: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
        sum_394: "f32[1, 1024][1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True, dtype = torch.float32);  view_1210 = None
        view_1211: "f32[1024][1]cuda:0" = torch.ops.aten.reshape.default(sum_394, [1024]);  sum_394 = None
        convert_element_type_2554: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1211, torch.bfloat16);  view_1211 = None
        view_1212: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(mm_290, [16, 512, 1024]);  mm_290 = None
        convert_element_type_2555: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_1212, torch.float32);  view_1212 = None
        add_344: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_343, convert_element_type_2555);  add_343 = convert_element_type_2555 = None
        convert_element_type_2556: "f32[1024, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_291, torch.float32);  mm_291 = None
        convert_element_type_2557: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_2554, torch.float32);  convert_element_type_2554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_1011: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_344, primals_7);  primals_7 = None
        mul_1012: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1011, 1024)
        sum_395: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1011, [2], True)
        mul_1013: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_1011, mul_3);  mul_1011 = None
        sum_396: "f32[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1013, [2], True);  mul_1013 = None
        mul_1014: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_3, sum_396);  sum_396 = None
        sub_226: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_1012, sum_395);  mul_1012 = sum_395 = None
        sub_227: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, mul_1014);  sub_226 = mul_1014 = None
        mul_1015: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_123, sub_227);  div_123 = sub_227 = None
        mul_1016: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_344, mul_3);  mul_3 = None
        sum_397: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1016, [0, 1]);  mul_1016 = None
        sum_398: "f32[1024][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_344, [0, 1]);  add_344 = None
        add_345: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(add_342, mul_1015);  add_342 = mul_1015 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:98 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_2558: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1017: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_2558, 1.1111111111111112);  convert_element_type_2558 = None
        mul_1018: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_345, mul_1017);  add_345 = mul_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        sum_399: "f32[1, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_1018, [0], True, dtype = torch.float32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        ge: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_4, 0)
        lt: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_4, 512)
        bitwise_and: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge, lt);  ge = lt = None
        ne_5: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_4, -1)
        bitwise_and_1: "b8[1, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and, ne_5);  bitwise_and = ne_5 = None
        unsqueeze_4: "b8[1, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_1, -1);  bitwise_and_1 = None
        full_default_7: "f32[512, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate: "f32[512, 1024][1024, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_7, unsqueeze_4, [primals_4], sum_399);  full_default_7 = unsqueeze_4 = primals_4 = sum_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_8: "b8[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.full.default([16, 512, 1], True, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[2, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_1: "f32[2, 1024][1024, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_9, full_default_8, [full_default], mul_1018);  full_default_9 = full_default_8 = full_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        ge_2: "b8[16, 512][512, 1]cuda:0" = torch.ops.aten.ge.Scalar(primals_2, 0)
        lt_2: "b8[16, 512][512, 1]cuda:0" = torch.ops.aten.lt.Scalar(primals_2, 29056)
        bitwise_and_4: "b8[16, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(ge_2, lt_2);  ge_2 = lt_2 = None
        ne_7: "b8[16, 512][512, 1]cuda:0" = torch.ops.aten.ne.Scalar(primals_2, 0)
        bitwise_and_5: "b8[16, 512][512, 1]cuda:0" = torch.ops.aten.bitwise_and.Tensor(bitwise_and_4, ne_7);  bitwise_and_4 = ne_7 = None
        unsqueeze_6: "b8[16, 512, 1][512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(bitwise_and_5, -1);  bitwise_and_5 = None
        full_default_10: "f32[29056, 1024][1024, 1]cuda:0" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _unsafe_masked_index_put_accumulate_2: "f32[29056, 1024][1024, 1]cuda:0" = torch.ops.aten._unsafe_masked_index_put_accumulate.default(full_default_10, unsqueeze_6, [primals_2], mul_1018);  full_default_10 = unsqueeze_6 = primals_2 = mul_1018 = None
        add_346: "f32[29056, 1024][1024, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_1008, _unsafe_masked_index_put_accumulate_2);  convert_element_type_1008 = _unsafe_masked_index_put_accumulate_2 = None
        return (None, None, add_346, None, _unsafe_masked_index_put_accumulate_1, _unsafe_masked_index_put_accumulate, sum_397, sum_398, convert_element_type_2556, convert_element_type_2557, convert_element_type_2548, convert_element_type_2549, convert_element_type_2540, convert_element_type_2541, convert_element_type_2521, convert_element_type_2522, sum_388, sum_389, convert_element_type_2512, convert_element_type_2513, convert_element_type_2501, convert_element_type_2502, sum_382, sum_383, convert_element_type_2492, convert_element_type_2493, convert_element_type_2484, convert_element_type_2485, convert_element_type_2476, convert_element_type_2477, convert_element_type_2457, convert_element_type_2458, sum_373, sum_374, convert_element_type_2448, convert_element_type_2449, convert_element_type_2437, convert_element_type_2438, sum_367, sum_368, convert_element_type_2428, convert_element_type_2429, convert_element_type_2420, convert_element_type_2421, convert_element_type_2412, convert_element_type_2413, convert_element_type_2393, convert_element_type_2394, sum_358, sum_359, convert_element_type_2384, convert_element_type_2385, convert_element_type_2373, convert_element_type_2374, sum_352, sum_353, convert_element_type_2364, convert_element_type_2365, convert_element_type_2356, convert_element_type_2357, convert_element_type_2348, convert_element_type_2349, convert_element_type_2329, convert_element_type_2330, sum_343, sum_344, convert_element_type_2320, convert_element_type_2321, convert_element_type_2309, convert_element_type_2310, sum_337, sum_338, convert_element_type_2300, convert_element_type_2301, convert_element_type_2292, convert_element_type_2293, convert_element_type_2284, convert_element_type_2285, convert_element_type_2265, convert_element_type_2266, sum_328, sum_329, convert_element_type_2256, convert_element_type_2257, convert_element_type_2245, convert_element_type_2246, sum_322, sum_323, convert_element_type_2236, convert_element_type_2237, convert_element_type_2228, convert_element_type_2229, convert_element_type_2220, convert_element_type_2221, convert_element_type_2201, convert_element_type_2202, sum_313, sum_314, convert_element_type_2192, convert_element_type_2193, convert_element_type_2181, convert_element_type_2182, sum_307, sum_308, convert_element_type_2172, convert_element_type_2173, convert_element_type_2164, convert_element_type_2165, convert_element_type_2156, convert_element_type_2157, convert_element_type_2137, convert_element_type_2138, sum_298, sum_299, convert_element_type_2128, convert_element_type_2129, convert_element_type_2117, convert_element_type_2118, sum_292, sum_293, convert_element_type_2108, convert_element_type_2109, convert_element_type_2100, convert_element_type_2101, convert_element_type_2092, convert_element_type_2093, convert_element_type_2073, convert_element_type_2074, sum_283, sum_284, convert_element_type_2064, convert_element_type_2065, convert_element_type_2053, convert_element_type_2054, sum_277, sum_278, convert_element_type_2044, convert_element_type_2045, convert_element_type_2036, convert_element_type_2037, convert_element_type_2028, convert_element_type_2029, convert_element_type_2009, convert_element_type_2010, sum_268, sum_269, convert_element_type_2000, convert_element_type_2001, convert_element_type_1989, convert_element_type_1990, sum_262, sum_263, convert_element_type_1980, convert_element_type_1981, convert_element_type_1972, convert_element_type_1973, convert_element_type_1964, convert_element_type_1965, convert_element_type_1945, convert_element_type_1946, sum_253, sum_254, convert_element_type_1936, convert_element_type_1937, convert_element_type_1925, convert_element_type_1926, sum_247, sum_248, convert_element_type_1916, convert_element_type_1917, convert_element_type_1908, convert_element_type_1909, convert_element_type_1900, convert_element_type_1901, convert_element_type_1881, convert_element_type_1882, sum_238, sum_239, convert_element_type_1872, convert_element_type_1873, convert_element_type_1861, convert_element_type_1862, sum_232, sum_233, convert_element_type_1852, convert_element_type_1853, convert_element_type_1844, convert_element_type_1845, convert_element_type_1836, convert_element_type_1837, convert_element_type_1817, convert_element_type_1818, sum_223, sum_224, convert_element_type_1808, convert_element_type_1809, convert_element_type_1797, convert_element_type_1798, sum_217, sum_218, convert_element_type_1788, convert_element_type_1789, convert_element_type_1780, convert_element_type_1781, convert_element_type_1772, convert_element_type_1773, convert_element_type_1753, convert_element_type_1754, sum_208, sum_209, convert_element_type_1744, convert_element_type_1745, convert_element_type_1733, convert_element_type_1734, sum_202, sum_203, convert_element_type_1724, convert_element_type_1725, convert_element_type_1716, convert_element_type_1717, convert_element_type_1708, convert_element_type_1709, convert_element_type_1689, convert_element_type_1690, sum_193, sum_194, convert_element_type_1680, convert_element_type_1681, convert_element_type_1669, convert_element_type_1670, sum_187, sum_188, convert_element_type_1660, convert_element_type_1661, convert_element_type_1652, convert_element_type_1653, convert_element_type_1644, convert_element_type_1645, convert_element_type_1625, convert_element_type_1626, sum_178, sum_179, convert_element_type_1616, convert_element_type_1617, convert_element_type_1605, convert_element_type_1606, sum_172, sum_173, convert_element_type_1596, convert_element_type_1597, convert_element_type_1588, convert_element_type_1589, convert_element_type_1580, convert_element_type_1581, convert_element_type_1561, convert_element_type_1562, sum_163, sum_164, convert_element_type_1552, convert_element_type_1553, convert_element_type_1541, convert_element_type_1542, sum_157, sum_158, convert_element_type_1532, convert_element_type_1533, convert_element_type_1524, convert_element_type_1525, convert_element_type_1516, convert_element_type_1517, convert_element_type_1497, convert_element_type_1498, sum_148, sum_149, convert_element_type_1488, convert_element_type_1489, convert_element_type_1477, convert_element_type_1478, sum_142, sum_143, convert_element_type_1468, convert_element_type_1469, convert_element_type_1460, convert_element_type_1461, convert_element_type_1452, convert_element_type_1453, convert_element_type_1433, convert_element_type_1434, sum_133, sum_134, convert_element_type_1424, convert_element_type_1425, convert_element_type_1413, convert_element_type_1414, sum_127, sum_128, convert_element_type_1404, convert_element_type_1405, convert_element_type_1396, convert_element_type_1397, convert_element_type_1388, convert_element_type_1389, convert_element_type_1369, convert_element_type_1370, sum_118, sum_119, convert_element_type_1360, convert_element_type_1361, convert_element_type_1349, convert_element_type_1350, sum_112, sum_113, convert_element_type_1340, convert_element_type_1341, convert_element_type_1332, convert_element_type_1333, convert_element_type_1324, convert_element_type_1325, convert_element_type_1305, convert_element_type_1306, sum_103, sum_104, convert_element_type_1296, convert_element_type_1297, convert_element_type_1285, convert_element_type_1286, sum_97, sum_98, convert_element_type_1276, convert_element_type_1277, convert_element_type_1268, convert_element_type_1269, convert_element_type_1260, convert_element_type_1261, convert_element_type_1241, convert_element_type_1242, sum_88, sum_89, convert_element_type_1232, convert_element_type_1233, convert_element_type_1221, convert_element_type_1222, sum_82, sum_83, convert_element_type_1212, convert_element_type_1213, convert_element_type_1204, convert_element_type_1205, convert_element_type_1196, convert_element_type_1197, convert_element_type_1177, convert_element_type_1178, sum_73, sum_74, convert_element_type_1168, convert_element_type_1169, convert_element_type_1157, convert_element_type_1158, sum_67, sum_68, convert_element_type_1148, convert_element_type_1149, convert_element_type_1140, convert_element_type_1141, convert_element_type_1132, convert_element_type_1133, convert_element_type_1113, convert_element_type_1114, sum_58, sum_59, convert_element_type_1104, convert_element_type_1105, convert_element_type_1093, convert_element_type_1094, sum_52, sum_53, convert_element_type_1084, convert_element_type_1085, convert_element_type_1076, convert_element_type_1077, convert_element_type_1068, convert_element_type_1069, convert_element_type_1049, convert_element_type_1050, sum_43, sum_44, convert_element_type_1040, convert_element_type_1041, convert_element_type_1029, convert_element_type_1030, sum_37, sum_38, convert_element_type_1020, convert_element_type_1021, sum_32, sum_33, convert_element_type_1009)

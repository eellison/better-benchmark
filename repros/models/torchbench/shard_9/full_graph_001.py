class GraphModule(torch.nn.Module):
    def forward(self, primals_4: "f32[50, 768][768, 1]cuda:0", primals_5: "f32[768][1]cuda:0", primals_6: "f32[768][1]cuda:0", primals_7: "f32[768][1]cuda:0", primals_13: "f32[768][1]cuda:0", primals_19: "f32[768][1]cuda:0", primals_25: "f32[768][1]cuda:0", primals_31: "f32[768][1]cuda:0", primals_37: "f32[768][1]cuda:0", primals_43: "f32[768][1]cuda:0", primals_49: "f32[768][1]cuda:0", primals_55: "f32[768][1]cuda:0", primals_61: "f32[768][1]cuda:0", primals_67: "f32[768][1]cuda:0", primals_73: "f32[768][1]cuda:0", primals_79: "f32[768][1]cuda:0", primals_85: "f32[768][1]cuda:0", primals_91: "f32[768][1]cuda:0", primals_97: "f32[768][1]cuda:0", primals_103: "f32[768][1]cuda:0", primals_109: "f32[768][1]cuda:0", primals_115: "f32[768][1]cuda:0", primals_121: "f32[768][1]cuda:0", primals_127: "f32[768][1]cuda:0", primals_133: "f32[768][1]cuda:0", primals_139: "f32[768][1]cuda:0", primals_145: "f32[768][1]cuda:0", primals_151: "f32[768][1]cuda:0", primals_154: "i64[32, 77][77, 1]cuda:0", primals_156: "f32[77, 512][512, 1]cuda:0", primals_158: "f32[512][1]cuda:0", primals_164: "f32[512][1]cuda:0", primals_170: "f32[512][1]cuda:0", primals_176: "f32[512][1]cuda:0", primals_182: "f32[512][1]cuda:0", primals_188: "f32[512][1]cuda:0", primals_194: "f32[512][1]cuda:0", primals_200: "f32[512][1]cuda:0", primals_206: "f32[512][1]cuda:0", primals_212: "f32[512][1]cuda:0", primals_218: "f32[512][1]cuda:0", primals_224: "f32[512][1]cuda:0", primals_230: "f32[512][1]cuda:0", primals_236: "f32[512][1]cuda:0", primals_242: "f32[512][1]cuda:0", primals_248: "f32[512][1]cuda:0", primals_254: "f32[512][1]cuda:0", primals_260: "f32[512][1]cuda:0", primals_266: "f32[512][1]cuda:0", primals_272: "f32[512][1]cuda:0", primals_278: "f32[512][1]cuda:0", primals_284: "f32[512][1]cuda:0", primals_290: "f32[512][1]cuda:0", primals_296: "f32[512][1]cuda:0", primals_302: "f32[512][1]cuda:0", convert_element_type: "f16[768, 3, 32, 32][3072, 1024, 32, 1]cuda:0", convert_element_type_1: "f16[32, 3, 224, 224][150528, 50176, 224, 1]cuda:0", cat: "f32[32, 50, 768][38400, 768, 1]cuda:0", getitem_1: "f32[32, 50, 1][50, 1, 1]cuda:0", rsqrt: "f32[32, 50, 1][50, 1, 1]cuda:0", getitem_3: "f32[32, 50, 1][50, 1, 1]cuda:0", rsqrt_1: "f32[32, 50, 1][50, 1, 1]cuda:0", view_1: "f16[1600, 768][768, 1]cuda:0", view_7: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_8: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_9: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_4: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_5: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_10: "i64[][]cuda:0", getitem_11: "i64[][]cuda:0", mul_4: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_12: "f16[1600, 768][768, 1]cuda:0", addmm_1: "f16[1600, 3072][3072, 1]cuda:0", view_14: "f16[1600, 3072][3072, 1]cuda:0", mul_8: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_16: "f16[1600, 768][768, 1]cuda:0", view_22: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_23: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_24: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_17: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_18: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_23: "i64[][]cuda:0", getitem_24: "i64[][]cuda:0", mul_10: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_27: "f16[1600, 768][768, 1]cuda:0", addmm_4: "f16[1600, 3072][3072, 1]cuda:0", view_29: "f16[1600, 3072][3072, 1]cuda:0", mul_14: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_31: "f16[1600, 768][768, 1]cuda:0", view_37: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_38: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_39: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_30: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_31: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_36: "i64[][]cuda:0", getitem_37: "i64[][]cuda:0", mul_16: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_42: "f16[1600, 768][768, 1]cuda:0", addmm_7: "f16[1600, 3072][3072, 1]cuda:0", view_44: "f16[1600, 3072][3072, 1]cuda:0", mul_20: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_46: "f16[1600, 768][768, 1]cuda:0", view_52: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_53: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_54: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_43: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_44: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_49: "i64[][]cuda:0", getitem_50: "i64[][]cuda:0", mul_22: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_57: "f16[1600, 768][768, 1]cuda:0", addmm_10: "f16[1600, 3072][3072, 1]cuda:0", view_59: "f16[1600, 3072][3072, 1]cuda:0", mul_26: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_61: "f16[1600, 768][768, 1]cuda:0", view_67: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_68: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_69: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_56: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_57: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_62: "i64[][]cuda:0", getitem_63: "i64[][]cuda:0", mul_28: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_72: "f16[1600, 768][768, 1]cuda:0", addmm_13: "f16[1600, 3072][3072, 1]cuda:0", view_74: "f16[1600, 3072][3072, 1]cuda:0", mul_32: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_76: "f16[1600, 768][768, 1]cuda:0", view_82: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_83: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_84: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_69: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_70: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_75: "i64[][]cuda:0", getitem_76: "i64[][]cuda:0", mul_34: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_87: "f16[1600, 768][768, 1]cuda:0", addmm_16: "f16[1600, 3072][3072, 1]cuda:0", view_89: "f16[1600, 3072][3072, 1]cuda:0", mul_38: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_91: "f16[1600, 768][768, 1]cuda:0", view_97: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_98: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_99: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_82: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_83: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_88: "i64[][]cuda:0", getitem_89: "i64[][]cuda:0", mul_40: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_102: "f16[1600, 768][768, 1]cuda:0", addmm_19: "f16[1600, 3072][3072, 1]cuda:0", view_104: "f16[1600, 3072][3072, 1]cuda:0", mul_44: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_106: "f16[1600, 768][768, 1]cuda:0", view_112: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_113: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_114: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_95: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_96: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_101: "i64[][]cuda:0", getitem_102: "i64[][]cuda:0", mul_46: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_117: "f16[1600, 768][768, 1]cuda:0", addmm_22: "f16[1600, 3072][3072, 1]cuda:0", view_119: "f16[1600, 3072][3072, 1]cuda:0", mul_50: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_121: "f16[1600, 768][768, 1]cuda:0", view_127: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_128: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_129: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_108: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_109: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_114: "i64[][]cuda:0", getitem_115: "i64[][]cuda:0", mul_52: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_132: "f16[1600, 768][768, 1]cuda:0", addmm_25: "f16[1600, 3072][3072, 1]cuda:0", view_134: "f16[1600, 3072][3072, 1]cuda:0", mul_56: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_136: "f16[1600, 768][768, 1]cuda:0", view_142: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_143: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_144: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_121: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_122: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_127: "i64[][]cuda:0", getitem_128: "i64[][]cuda:0", mul_58: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_147: "f16[1600, 768][768, 1]cuda:0", addmm_28: "f16[1600, 3072][3072, 1]cuda:0", view_149: "f16[1600, 3072][3072, 1]cuda:0", mul_62: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_151: "f16[1600, 768][768, 1]cuda:0", view_157: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_158: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_159: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_134: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_135: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_140: "i64[][]cuda:0", getitem_141: "i64[][]cuda:0", mul_64: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_162: "f16[1600, 768][768, 1]cuda:0", addmm_31: "f16[1600, 3072][3072, 1]cuda:0", view_164: "f16[1600, 3072][3072, 1]cuda:0", mul_68: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_166: "f16[1600, 768][768, 1]cuda:0", view_172: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_173: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", view_174: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_147: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0", getitem_148: "f32[32, 12, 50, 1][600, 50, 1, 1]cuda:0", getitem_153: "i64[][]cuda:0", getitem_154: "i64[][]cuda:0", mul_70: "f32[32, 50, 768][38400, 768, 1]cuda:0", view_177: "f16[1600, 768][768, 1]cuda:0", addmm_34: "f16[1600, 3072][3072, 1]cuda:0", view_179: "f16[1600, 3072][3072, 1]cuda:0", mm_12: "f16[32, 512][512, 1]cuda:0", embedding: "f32[32, 77, 512][39424, 512, 1]cuda:0", getitem_161: "f32[77, 32, 1][32, 1, 1]cuda:0", rsqrt_26: "f32[77, 32, 1][32, 1, 1]cuda:0", view_181: "f16[2464, 512][512, 1]cuda:0", view_187: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_188: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_189: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_162: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_163: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_168: "i64[][]cuda:0", getitem_169: "i64[][]cuda:0", view_192: "f16[2464, 512][512, 1]cuda:0", addmm_38: "f16[2464, 2048][2048, 1]cuda:0", view_194: "f16[2464, 2048][2048, 1]cuda:0", view_196: "f16[2464, 512][512, 1]cuda:0", view_202: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_203: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_204: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_175: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_176: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_181: "i64[][]cuda:0", getitem_182: "i64[][]cuda:0", view_207: "f16[2464, 512][512, 1]cuda:0", addmm_42: "f16[2464, 2048][2048, 1]cuda:0", view_209: "f16[2464, 2048][2048, 1]cuda:0", view_211: "f16[2464, 512][512, 1]cuda:0", view_217: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_218: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_219: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_188: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_189: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_194: "i64[][]cuda:0", getitem_195: "i64[][]cuda:0", view_222: "f16[2464, 512][512, 1]cuda:0", addmm_46: "f16[2464, 2048][2048, 1]cuda:0", view_224: "f16[2464, 2048][2048, 1]cuda:0", view_226: "f16[2464, 512][512, 1]cuda:0", view_232: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_233: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_234: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_201: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_202: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_207: "i64[][]cuda:0", getitem_208: "i64[][]cuda:0", view_237: "f16[2464, 512][512, 1]cuda:0", addmm_50: "f16[2464, 2048][2048, 1]cuda:0", view_239: "f16[2464, 2048][2048, 1]cuda:0", view_241: "f16[2464, 512][512, 1]cuda:0", view_247: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_248: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_249: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_214: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_215: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_220: "i64[][]cuda:0", getitem_221: "i64[][]cuda:0", view_252: "f16[2464, 512][512, 1]cuda:0", addmm_54: "f16[2464, 2048][2048, 1]cuda:0", view_254: "f16[2464, 2048][2048, 1]cuda:0", view_256: "f16[2464, 512][512, 1]cuda:0", view_262: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_263: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_264: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_227: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_228: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_233: "i64[][]cuda:0", getitem_234: "i64[][]cuda:0", view_267: "f16[2464, 512][512, 1]cuda:0", addmm_58: "f16[2464, 2048][2048, 1]cuda:0", view_269: "f16[2464, 2048][2048, 1]cuda:0", view_271: "f16[2464, 512][512, 1]cuda:0", view_277: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_278: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_279: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_240: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_241: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_246: "i64[][]cuda:0", getitem_247: "i64[][]cuda:0", view_282: "f16[2464, 512][512, 1]cuda:0", addmm_62: "f16[2464, 2048][2048, 1]cuda:0", view_284: "f16[2464, 2048][2048, 1]cuda:0", view_286: "f16[2464, 512][512, 1]cuda:0", view_292: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_293: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_294: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_253: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_254: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_259: "i64[][]cuda:0", getitem_260: "i64[][]cuda:0", view_297: "f16[2464, 512][512, 1]cuda:0", addmm_66: "f16[2464, 2048][2048, 1]cuda:0", view_299: "f16[2464, 2048][2048, 1]cuda:0", view_301: "f16[2464, 512][512, 1]cuda:0", view_307: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_308: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_309: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_266: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_267: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_272: "i64[][]cuda:0", getitem_273: "i64[][]cuda:0", view_312: "f16[2464, 512][512, 1]cuda:0", addmm_70: "f16[2464, 2048][2048, 1]cuda:0", view_314: "f16[2464, 2048][2048, 1]cuda:0", view_316: "f16[2464, 512][512, 1]cuda:0", view_322: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_323: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_324: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_279: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_280: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_285: "i64[][]cuda:0", getitem_286: "i64[][]cuda:0", view_327: "f16[2464, 512][512, 1]cuda:0", addmm_74: "f16[2464, 2048][2048, 1]cuda:0", view_329: "f16[2464, 2048][2048, 1]cuda:0", view_331: "f16[2464, 512][512, 1]cuda:0", view_337: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_338: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_339: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_292: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_293: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_298: "i64[][]cuda:0", getitem_299: "i64[][]cuda:0", view_342: "f16[2464, 512][512, 1]cuda:0", addmm_78: "f16[2464, 2048][2048, 1]cuda:0", view_344: "f16[2464, 2048][2048, 1]cuda:0", view_346: "f16[2464, 512][512, 1]cuda:0", view_352: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_353: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", view_354: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_305: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0", getitem_306: "f32[32, 8, 77, 1][616, 77, 1, 1]cuda:0", getitem_311: "i64[][]cuda:0", getitem_312: "i64[][]cuda:0", view_357: "f16[2464, 512][512, 1]cuda:0", addmm_82: "f16[2464, 2048][2048, 1]cuda:0", view_359: "f16[2464, 2048][2048, 1]cuda:0", mul_148: "f32[32, 77, 512][39424, 512, 1]cuda:0", iota_default: "i64[32][1]cuda:0", argmax: "i64[32][1]cuda:0", convert_element_type_523: "f16[32, 512][512, 1]cuda:0", mm_13: "f16[32, 512][512, 1]cuda:0", pow_2: "f32[32, 1][1, 1]cuda:0", pow_4: "f32[32, 1][1, 1]cuda:0", permute_246: "f16[512, 512][512, 1]cuda:0", div_10: "f32[32, 77, 1][77, 1, 1]cuda:0", permute_249: "f16[512, 2048][2048, 1]cuda:0", permute_253: "f16[2048, 512][512, 1]cuda:0", mul_166: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_11: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_257: "f16[512, 512][512, 1]cuda:0", permute_266: "f16[1536, 512][512, 1]cuda:0", mul_173: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_12: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_270: "f16[512, 2048][2048, 1]cuda:0", permute_274: "f16[2048, 512][512, 1]cuda:0", mul_185: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_13: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_278: "f16[512, 512][512, 1]cuda:0", permute_287: "f16[1536, 512][512, 1]cuda:0", mul_192: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_14: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_291: "f16[512, 2048][2048, 1]cuda:0", permute_295: "f16[2048, 512][512, 1]cuda:0", mul_204: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_15: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_299: "f16[512, 512][512, 1]cuda:0", permute_308: "f16[1536, 512][512, 1]cuda:0", mul_211: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_16: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_312: "f16[512, 2048][2048, 1]cuda:0", permute_316: "f16[2048, 512][512, 1]cuda:0", mul_223: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_17: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_320: "f16[512, 512][512, 1]cuda:0", permute_329: "f16[1536, 512][512, 1]cuda:0", mul_230: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_18: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_333: "f16[512, 2048][2048, 1]cuda:0", permute_337: "f16[2048, 512][512, 1]cuda:0", mul_242: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_19: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_341: "f16[512, 512][512, 1]cuda:0", permute_350: "f16[1536, 512][512, 1]cuda:0", mul_249: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_20: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_354: "f16[512, 2048][2048, 1]cuda:0", permute_358: "f16[2048, 512][512, 1]cuda:0", mul_261: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_21: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_362: "f16[512, 512][512, 1]cuda:0", permute_371: "f16[1536, 512][512, 1]cuda:0", mul_268: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_22: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_375: "f16[512, 2048][2048, 1]cuda:0", permute_379: "f16[2048, 512][512, 1]cuda:0", mul_280: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_23: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_383: "f16[512, 512][512, 1]cuda:0", permute_392: "f16[1536, 512][512, 1]cuda:0", mul_287: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_24: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_396: "f16[512, 2048][2048, 1]cuda:0", permute_400: "f16[2048, 512][512, 1]cuda:0", mul_299: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_25: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_404: "f16[512, 512][512, 1]cuda:0", permute_413: "f16[1536, 512][512, 1]cuda:0", mul_306: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_26: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_417: "f16[512, 2048][2048, 1]cuda:0", permute_421: "f16[2048, 512][512, 1]cuda:0", mul_318: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_27: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_425: "f16[512, 512][512, 1]cuda:0", permute_434: "f16[1536, 512][512, 1]cuda:0", mul_325: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_28: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_438: "f16[512, 2048][2048, 1]cuda:0", permute_442: "f16[2048, 512][512, 1]cuda:0", mul_337: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_29: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_446: "f16[512, 512][512, 1]cuda:0", permute_455: "f16[1536, 512][512, 1]cuda:0", mul_344: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_30: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_459: "f16[512, 2048][2048, 1]cuda:0", permute_463: "f16[2048, 512][512, 1]cuda:0", mul_356: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_31: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_467: "f16[512, 512][512, 1]cuda:0", permute_476: "f16[1536, 512][512, 1]cuda:0", mul_363: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_32: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_480: "f16[512, 2048][2048, 1]cuda:0", permute_484: "f16[2048, 512][512, 1]cuda:0", mul_375: "f32[77, 32, 512][512, 39424, 1]cuda:0", div_33: "f32[77, 32, 1][32, 1, 1]cuda:0", permute_488: "f16[512, 512][512, 1]cuda:0", permute_497: "f16[1536, 512][512, 1]cuda:0", permute_502: "f16[768, 32][1, 768]cuda:0", permute_503: "f16[512, 768][1, 512]cuda:0", mul_389: "f32[32, 768][768, 1]cuda:0", div_35: "f32[32, 1][1, 1]cuda:0", permute_504: "f16[768, 3072][3072, 1]cuda:0", permute_508: "f16[3072, 768][768, 1]cuda:0", div_36: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_513: "f16[768, 768][768, 1]cuda:0", permute_524: "f16[2304, 768][768, 1]cuda:0", div_37: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_527: "f16[768, 3072][3072, 1]cuda:0", permute_531: "f16[3072, 768][768, 1]cuda:0", div_38: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_536: "f16[768, 768][768, 1]cuda:0", permute_547: "f16[2304, 768][768, 1]cuda:0", div_39: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_550: "f16[768, 3072][3072, 1]cuda:0", permute_554: "f16[3072, 768][768, 1]cuda:0", div_40: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_559: "f16[768, 768][768, 1]cuda:0", permute_570: "f16[2304, 768][768, 1]cuda:0", div_41: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_573: "f16[768, 3072][3072, 1]cuda:0", permute_577: "f16[3072, 768][768, 1]cuda:0", div_42: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_582: "f16[768, 768][768, 1]cuda:0", permute_593: "f16[2304, 768][768, 1]cuda:0", div_43: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_596: "f16[768, 3072][3072, 1]cuda:0", permute_600: "f16[3072, 768][768, 1]cuda:0", div_44: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_605: "f16[768, 768][768, 1]cuda:0", permute_616: "f16[2304, 768][768, 1]cuda:0", div_45: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_619: "f16[768, 3072][3072, 1]cuda:0", permute_623: "f16[3072, 768][768, 1]cuda:0", div_46: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_628: "f16[768, 768][768, 1]cuda:0", permute_639: "f16[2304, 768][768, 1]cuda:0", div_47: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_642: "f16[768, 3072][3072, 1]cuda:0", permute_646: "f16[3072, 768][768, 1]cuda:0", div_48: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_651: "f16[768, 768][768, 1]cuda:0", permute_662: "f16[2304, 768][768, 1]cuda:0", div_49: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_665: "f16[768, 3072][3072, 1]cuda:0", permute_669: "f16[3072, 768][768, 1]cuda:0", div_50: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_674: "f16[768, 768][768, 1]cuda:0", permute_685: "f16[2304, 768][768, 1]cuda:0", div_51: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_688: "f16[768, 3072][3072, 1]cuda:0", permute_692: "f16[3072, 768][768, 1]cuda:0", div_52: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_697: "f16[768, 768][768, 1]cuda:0", permute_708: "f16[2304, 768][768, 1]cuda:0", div_53: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_711: "f16[768, 3072][3072, 1]cuda:0", permute_715: "f16[3072, 768][768, 1]cuda:0", div_54: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_720: "f16[768, 768][768, 1]cuda:0", permute_731: "f16[2304, 768][768, 1]cuda:0", div_55: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_734: "f16[768, 3072][3072, 1]cuda:0", permute_738: "f16[3072, 768][768, 1]cuda:0", div_56: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_743: "f16[768, 768][768, 1]cuda:0", permute_754: "f16[2304, 768][768, 1]cuda:0", div_57: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_757: "f16[768, 3072][3072, 1]cuda:0", permute_761: "f16[3072, 768][768, 1]cuda:0", div_58: "f32[32, 50, 1][50, 1, 1]cuda:0", permute_766: "f16[768, 768][768, 1]cuda:0", permute_777: "f16[2304, 768][768, 1]cuda:0", tangents_1: "f32[32, 512][512, 1]cuda:0", tangents_2: "f32[32, 512][512, 1]cuda:0"):
        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/model.py:73 in forward, code: embeddings_b = F.normalize(embeddings_b)
        neg: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.neg.default(tangents_2)
        clamp_min_1: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.clamp_min.default(pow_4, 1e-12)
        expand_2: "f32[32, 512][1, 0]cuda:0" = torch.ops.aten.expand.default(clamp_min_1, [32, 512]);  clamp_min_1 = None
        div_2: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(mm_13, expand_2)
        div_3: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(div_2, expand_2);  div_2 = None
        mul_150: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg, div_3);  neg = div_3 = None
        div_4: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(tangents_2, expand_2);  tangents_2 = expand_2 = None
        convert_element_type_528: "f16[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_4, torch.float16);  div_4 = None
        sum_3: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_150, [1], True);  mul_150 = None
        full_default: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        ge: "b8[32, 1][1, 1]cuda:0" = torch.ops.aten.ge.Scalar(pow_4, 1e-12)
        where: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ge, sum_3, full_default);  ge = sum_3 = None
        div_5: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(mm_13, pow_4);  mm_13 = None
        eq: "b8[32, 1][1, 1]cuda:0" = torch.ops.aten.eq.Scalar(pow_4, 0);  pow_4 = None
        where_1: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.where.self(eq, full_default, div_5);  eq = div_5 = None
        mul_151: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(where, where_1);  where = where_1 = None
        convert_element_type_529: "f16[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_151, torch.float16);  mul_151 = None
        add_164: "f16[32, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_528, convert_element_type_529);  convert_element_type_528 = convert_element_type_529 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/model.py:72 in forward, code: embeddings_a = F.normalize(embeddings_a)
        clamp_min: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.clamp_min.default(pow_2, 1e-12)
        expand_1: "f32[32, 512][1, 0]cuda:0" = torch.ops.aten.expand.default(clamp_min, [32, 512]);  clamp_min = None
        div_6: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(mm_12, expand_1)
        div_7: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(div_6, expand_1);  div_6 = None
        neg_1: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.neg.default(tangents_1)
        mul_152: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(neg_1, div_7);  neg_1 = div_7 = None
        div_8: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(tangents_1, expand_1);  tangents_1 = expand_1 = None
        convert_element_type_530: "f16[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(div_8, torch.float16);  div_8 = None
        sum_4: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_152, [1], True);  mul_152 = None
        ge_1: "b8[32, 1][1, 1]cuda:0" = torch.ops.aten.ge.Scalar(pow_2, 1e-12)
        where_2: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.where.self(ge_1, sum_4, full_default);  ge_1 = sum_4 = None
        div_9: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.div.Tensor(mm_12, pow_2);  mm_12 = None
        eq_1: "b8[32, 1][1, 1]cuda:0" = torch.ops.aten.eq.Scalar(pow_2, 0);  pow_2 = None
        where_3: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.where.self(eq_1, full_default, div_9);  eq_1 = div_9 = None
        mul_153: "f32[32, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(where_2, where_3);  where_2 = where_3 = None
        convert_element_type_531: "f16[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_153, torch.float16);  mul_153 = None
        add_165: "f16[32, 512][512, 1]cuda:0" = torch.ops.aten.add.Tensor(convert_element_type_530, convert_element_type_531);  convert_element_type_530 = convert_element_type_531 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:130 in forward, code: projected_embeddings = self.projection(
        permute_244: "f16[512, 32][1, 512]cuda:0" = torch.ops.aten.permute.default(add_164, [1, 0])
        mm_14: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_244, convert_element_type_523);  permute_244 = convert_element_type_523 = None
        mm_15: "f16[32, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(add_164, permute_246);  add_164 = permute_246 = None
        convert_element_type_536: "f32[32, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_15, torch.float32);  mm_15 = None
        convert_element_type_537: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_14, torch.float32);  mm_14 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:131 in forward, code: hidden_state[torch.arange(hidden_state.shape[0]), text.argmax(dim=-1)]
        full_default_4: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.full.default([32, 77, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.index_put_.default(full_default_4, [iota_default, argmax], convert_element_type_536, True);  full_default_4 = iota_default = argmax = convert_element_type_536 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        mul_155: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_put, primals_302);  primals_302 = None
        mul_156: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, 512)
        sum_5: "f32[32, 77, 1][77, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_155, [2], True)
        mul_157: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, mul_148);  mul_155 = None
        sum_6: "f32[32, 77, 1][77, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_157, [2], True);  mul_157 = None
        mul_158: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, sum_6);  sum_6 = None
        sub_52: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_156, sum_5);  mul_156 = sum_5 = None
        sub_53: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_52, mul_158);  sub_52 = mul_158 = None
        mul_159: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_10, sub_53);  div_10 = sub_53 = None
        mul_160: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(index_put, mul_148);  mul_148 = None
        sum_7: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_160, [0, 1]);  mul_160 = None
        sum_8: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(index_put, [0, 1]);  index_put = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:124 in forward, code: embeddings = torch.permute(embeddings, (1, 0, 2))
        permute_248: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.permute.default(mul_159, [1, 0, 2]);  mul_159 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        convert_element_type_538: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(permute_248, torch.float16)
        clone_133: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_538, memory_format = torch.contiguous_format);  convert_element_type_538 = None
        view_361: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_133, [2464, 512]);  clone_133 = None
        mm_16: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_361, permute_249);  permute_249 = None
        permute_250: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_361, [1, 0])
        mm_17: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_250, view_359);  permute_250 = view_359 = None
        sum_9: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_361, [0], True);  view_361 = None
        view_362: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_9, [512]);  sum_9 = None
        view_363: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_16, [77, 32, 2048]);  mm_16 = None
        convert_element_type_543: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_17, torch.float32);  mm_17 = None
        convert_element_type_544: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_362, torch.float32);  view_362 = None
        view_358: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_82, [77, 32, 2048]);  addmm_82 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_146: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_358, 1.702)
        sigmoid_23: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_146);  mul_146 = None
        mul_161: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_363, sigmoid_23)
        mul_162: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_363, view_358);  view_363 = view_358 = None
        convert_element_type_545: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_162, torch.float32);  mul_162 = None
        convert_element_type_546: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_23, torch.float32);  sigmoid_23 = None
        sub_54: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_546)
        mul_163: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_546, sub_54);  convert_element_type_546 = sub_54 = None
        mul_164: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_545, mul_163);  convert_element_type_545 = mul_163 = None
        convert_element_type_547: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_164, torch.float16);  mul_164 = None
        mul_165: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_547, 1.702);  convert_element_type_547 = None
        add_166: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_161, mul_165);  mul_161 = mul_165 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_364: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_166, [2464, 2048]);  add_166 = None
        mm_18: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_364, permute_253);  permute_253 = None
        permute_254: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_364, [1, 0])
        mm_19: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_254, view_357);  permute_254 = view_357 = None
        sum_10: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_364, [0], True);  view_364 = None
        view_365: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_10, [2048]);  sum_10 = None
        view_366: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_18, [77, 32, 512]);  mm_18 = None
        convert_element_type_552: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_366, torch.float32);  view_366 = None
        convert_element_type_553: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_19, torch.float32);  mm_19 = None
        convert_element_type_554: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_365, torch.float32);  view_365 = None
        mul_167: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_552, primals_296);  primals_296 = None
        mul_168: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, 512)
        sum_11: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_167, [2], True)
        mul_169: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, mul_166);  mul_167 = None
        sum_12: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_169, [2], True);  mul_169 = None
        mul_170: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_166, sum_12);  sum_12 = None
        sub_56: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_168, sum_11);  mul_168 = sum_11 = None
        sub_57: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_56, mul_170);  sub_56 = mul_170 = None
        mul_171: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_11, sub_57);  div_11 = sub_57 = None
        mul_172: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_552, mul_166);  mul_166 = None
        sum_13: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_172, [0, 1]);  mul_172 = None
        sum_14: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_552, [0, 1]);  convert_element_type_552 = None
        add_167: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(permute_248, mul_171);  permute_248 = mul_171 = None
        convert_element_type_555: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_167, torch.float16)
        clone_134: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_555, memory_format = torch.contiguous_format);  convert_element_type_555 = None
        view_367: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_134, [2464, 512]);  clone_134 = None
        mm_20: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_367, permute_257);  permute_257 = None
        permute_258: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_367, [1, 0])
        permute_238: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_305, [2, 0, 1, 3])
        view_355: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_238, [2464, 512]);  permute_238 = None
        mm_21: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_258, view_355);  permute_258 = view_355 = None
        sum_15: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_367, [0], True);  view_367 = None
        view_368: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_15, [512]);  sum_15 = None
        convert_element_type_560: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_21, torch.float32);  mm_21 = None
        convert_element_type_561: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_368, torch.float32);  view_368 = None
        view_369: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_20, [77, 32, 8, 64]);  mm_20 = None
        permute_261: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_369, [1, 2, 0, 3]);  view_369 = None
        _scaled_dot_product_cudnn_attention_backward = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_261, view_352, view_353, view_354, getitem_305, getitem_306, getitem_311, getitem_312, None, None, None, 77, 77, 0.0, True);  permute_261 = view_352 = view_353 = view_354 = getitem_305 = getitem_306 = getitem_311 = getitem_312 = None
        getitem_318: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[0]
        getitem_319: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[1]
        getitem_320: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward[2];  _scaled_dot_product_cudnn_attention_backward = None
        view_370: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_320, [256, 77, 64]);  getitem_320 = None
        view_371: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_319, [256, 77, 64]);  getitem_319 = None
        view_372: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_318, [256, 77, 64]);  getitem_318 = None
        permute_262: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_370, [1, 0, 2]);  view_370 = None
        view_373: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_262, [77, 32, 512]);  permute_262 = None
        permute_263: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_371, [1, 0, 2]);  view_371 = None
        view_374: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_263, [77, 32, 512]);  permute_263 = None
        permute_264: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_372, [1, 0, 2]);  view_372 = None
        view_375: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_264, [77, 32, 512]);  permute_264 = None
        full_default_5: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.full.default([3, 77, 32, 512], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        select_scatter_default: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_373, 0, 2);  view_373 = None
        select_scatter_default_1: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_374, 0, 1);  view_374 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_168: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default, select_scatter_default_1);  select_scatter_default = select_scatter_default_1 = None

        # No stacktrace found for following nodes
        select_scatter_default_2: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_375, 0, 0);  view_375 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_169: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_168, select_scatter_default_2);  add_168 = select_scatter_default_2 = None
        unsqueeze_25: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_169, 3);  add_169 = None
        permute_265: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_25, [3, 1, 2, 0, 4]);  unsqueeze_25 = None
        squeeze_24: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_265, 0);  permute_265 = None
        clone_135: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_24, memory_format = torch.contiguous_format);  squeeze_24 = None
        view_376: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_135, [77, 32, 1536]);  clone_135 = None
        view_377: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_376, [2464, 1536]);  view_376 = None
        mm_22: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_377, permute_266);  permute_266 = None
        permute_267: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_377, [1, 0])
        mm_23: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_267, view_346);  permute_267 = view_346 = None
        sum_16: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_377, [0], True);  view_377 = None
        view_378: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_16, [1536]);  sum_16 = None
        view_379: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_22, [77, 32, 512]);  mm_22 = None
        convert_element_type_566: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_379, torch.float32);  view_379 = None
        convert_element_type_567: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_23, torch.float32);  mm_23 = None
        convert_element_type_568: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_378, torch.float32);  view_378 = None
        mul_174: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, primals_290);  primals_290 = None
        mul_175: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, 512)
        sum_17: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_174, [2], True)
        mul_176: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_174, mul_173);  mul_174 = None
        sum_18: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_176, [2], True);  mul_176 = None
        mul_177: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, sum_18);  sum_18 = None
        sub_59: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_175, sum_17);  mul_175 = sum_17 = None
        sub_60: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_59, mul_177);  sub_59 = mul_177 = None
        mul_178: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_12, sub_60);  div_12 = sub_60 = None
        mul_179: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_566, mul_173);  mul_173 = None
        sum_19: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_179, [0, 1]);  mul_179 = None
        sum_20: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_566, [0, 1]);  convert_element_type_566 = None
        add_170: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, mul_178);  add_167 = mul_178 = None
        convert_element_type_569: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_170, torch.float16)
        clone_136: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_569, memory_format = torch.contiguous_format);  convert_element_type_569 = None
        view_380: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_136, [2464, 512]);  clone_136 = None
        mm_24: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_380, permute_270);  permute_270 = None
        permute_271: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_380, [1, 0])
        mm_25: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_271, view_344);  permute_271 = view_344 = None
        sum_21: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_380, [0], True);  view_380 = None
        view_381: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_21, [512]);  sum_21 = None
        view_382: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_24, [77, 32, 2048]);  mm_24 = None
        convert_element_type_574: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_25, torch.float32);  mm_25 = None
        convert_element_type_575: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_381, torch.float32);  view_381 = None
        view_343: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_78, [77, 32, 2048]);  addmm_78 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_140: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_343, 1.702)
        sigmoid_22: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_140);  mul_140 = None
        mul_180: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_382, sigmoid_22)
        mul_181: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_382, view_343);  view_382 = view_343 = None
        convert_element_type_576: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_181, torch.float32);  mul_181 = None
        convert_element_type_577: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_22, torch.float32);  sigmoid_22 = None
        sub_61: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_577)
        mul_182: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_577, sub_61);  convert_element_type_577 = sub_61 = None
        mul_183: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_576, mul_182);  convert_element_type_576 = mul_182 = None
        convert_element_type_578: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_183, torch.float16);  mul_183 = None
        mul_184: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_578, 1.702);  convert_element_type_578 = None
        add_171: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_180, mul_184);  mul_180 = mul_184 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_383: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_171, [2464, 2048]);  add_171 = None
        mm_26: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_383, permute_274);  permute_274 = None
        permute_275: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_383, [1, 0])
        mm_27: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_275, view_342);  permute_275 = view_342 = None
        sum_22: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_383, [0], True);  view_383 = None
        view_384: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_22, [2048]);  sum_22 = None
        view_385: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_26, [77, 32, 512]);  mm_26 = None
        convert_element_type_583: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_385, torch.float32);  view_385 = None
        convert_element_type_584: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_27, torch.float32);  mm_27 = None
        convert_element_type_585: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_384, torch.float32);  view_384 = None
        mul_186: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, primals_284);  primals_284 = None
        mul_187: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, 512)
        sum_23: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_186, [2], True)
        mul_188: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_186, mul_185);  mul_186 = None
        sum_24: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_188, [2], True);  mul_188 = None
        mul_189: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, sum_24);  sum_24 = None
        sub_63: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_187, sum_23);  mul_187 = sum_23 = None
        sub_64: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_63, mul_189);  sub_63 = mul_189 = None
        mul_190: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_13, sub_64);  div_13 = sub_64 = None
        mul_191: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_583, mul_185);  mul_185 = None
        sum_25: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_191, [0, 1]);  mul_191 = None
        sum_26: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_583, [0, 1]);  convert_element_type_583 = None
        add_172: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_170, mul_190);  add_170 = mul_190 = None
        convert_element_type_586: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_172, torch.float16)
        clone_137: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_586, memory_format = torch.contiguous_format);  convert_element_type_586 = None
        view_386: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_137, [2464, 512]);  clone_137 = None
        mm_28: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_386, permute_278);  permute_278 = None
        permute_279: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_386, [1, 0])
        permute_229: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_292, [2, 0, 1, 3])
        view_340: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_229, [2464, 512]);  permute_229 = None
        mm_29: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_279, view_340);  permute_279 = view_340 = None
        sum_27: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_386, [0], True);  view_386 = None
        view_387: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_27, [512]);  sum_27 = None
        convert_element_type_591: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_29, torch.float32);  mm_29 = None
        convert_element_type_592: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_387, torch.float32);  view_387 = None
        view_388: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_28, [77, 32, 8, 64]);  mm_28 = None
        permute_282: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_388, [1, 2, 0, 3]);  view_388 = None
        _scaled_dot_product_cudnn_attention_backward_1 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_282, view_337, view_338, view_339, getitem_292, getitem_293, getitem_298, getitem_299, None, None, None, 77, 77, 0.0, True);  permute_282 = view_337 = view_338 = view_339 = getitem_292 = getitem_293 = getitem_298 = getitem_299 = None
        getitem_321: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[0]
        getitem_322: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[1]
        getitem_323: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_1[2];  _scaled_dot_product_cudnn_attention_backward_1 = None
        view_389: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_323, [256, 77, 64]);  getitem_323 = None
        view_390: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_322, [256, 77, 64]);  getitem_322 = None
        view_391: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_321, [256, 77, 64]);  getitem_321 = None
        permute_283: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_389, [1, 0, 2]);  view_389 = None
        view_392: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_283, [77, 32, 512]);  permute_283 = None
        permute_284: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_390, [1, 0, 2]);  view_390 = None
        view_393: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_284, [77, 32, 512]);  permute_284 = None
        permute_285: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_391, [1, 0, 2]);  view_391 = None
        view_394: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_285, [77, 32, 512]);  permute_285 = None

        # No stacktrace found for following nodes
        select_scatter_default_3: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_392, 0, 2);  view_392 = None
        select_scatter_default_4: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_393, 0, 1);  view_393 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_173: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_3, select_scatter_default_4);  select_scatter_default_3 = select_scatter_default_4 = None

        # No stacktrace found for following nodes
        select_scatter_default_5: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_394, 0, 0);  view_394 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_174: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_173, select_scatter_default_5);  add_173 = select_scatter_default_5 = None
        unsqueeze_26: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_174, 3);  add_174 = None
        permute_286: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_26, [3, 1, 2, 0, 4]);  unsqueeze_26 = None
        squeeze_25: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_286, 0);  permute_286 = None
        clone_138: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_25, memory_format = torch.contiguous_format);  squeeze_25 = None
        view_395: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_138, [77, 32, 1536]);  clone_138 = None
        view_396: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_395, [2464, 1536]);  view_395 = None
        mm_30: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_396, permute_287);  permute_287 = None
        permute_288: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_396, [1, 0])
        mm_31: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_288, view_331);  permute_288 = view_331 = None
        sum_28: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_396, [0], True);  view_396 = None
        view_397: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_28, [1536]);  sum_28 = None
        view_398: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_30, [77, 32, 512]);  mm_30 = None
        convert_element_type_597: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_398, torch.float32);  view_398 = None
        convert_element_type_598: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_31, torch.float32);  mm_31 = None
        convert_element_type_599: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_397, torch.float32);  view_397 = None
        mul_193: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_597, primals_278);  primals_278 = None
        mul_194: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, 512)
        sum_29: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_193, [2], True)
        mul_195: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_193, mul_192);  mul_193 = None
        sum_30: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_195, [2], True);  mul_195 = None
        mul_196: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, sum_30);  sum_30 = None
        sub_66: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_194, sum_29);  mul_194 = sum_29 = None
        sub_67: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_66, mul_196);  sub_66 = mul_196 = None
        mul_197: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_14, sub_67);  div_14 = sub_67 = None
        mul_198: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_597, mul_192);  mul_192 = None
        sum_31: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_198, [0, 1]);  mul_198 = None
        sum_32: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_597, [0, 1]);  convert_element_type_597 = None
        add_175: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_172, mul_197);  add_172 = mul_197 = None
        convert_element_type_600: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.float16)
        clone_139: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_600, memory_format = torch.contiguous_format);  convert_element_type_600 = None
        view_399: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_139, [2464, 512]);  clone_139 = None
        mm_32: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_399, permute_291);  permute_291 = None
        permute_292: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_399, [1, 0])
        mm_33: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_292, view_329);  permute_292 = view_329 = None
        sum_33: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_399, [0], True);  view_399 = None
        view_400: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_33, [512]);  sum_33 = None
        view_401: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_32, [77, 32, 2048]);  mm_32 = None
        convert_element_type_605: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_33, torch.float32);  mm_33 = None
        convert_element_type_606: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_400, torch.float32);  view_400 = None
        view_328: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_74, [77, 32, 2048]);  addmm_74 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_134: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_328, 1.702)
        sigmoid_21: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_134);  mul_134 = None
        mul_199: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_401, sigmoid_21)
        mul_200: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_401, view_328);  view_401 = view_328 = None
        convert_element_type_607: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_200, torch.float32);  mul_200 = None
        convert_element_type_608: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_21, torch.float32);  sigmoid_21 = None
        sub_68: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_608)
        mul_201: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_608, sub_68);  convert_element_type_608 = sub_68 = None
        mul_202: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_607, mul_201);  convert_element_type_607 = mul_201 = None
        convert_element_type_609: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_202, torch.float16);  mul_202 = None
        mul_203: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_609, 1.702);  convert_element_type_609 = None
        add_176: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_199, mul_203);  mul_199 = mul_203 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_402: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_176, [2464, 2048]);  add_176 = None
        mm_34: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_402, permute_295);  permute_295 = None
        permute_296: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_402, [1, 0])
        mm_35: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_296, view_327);  permute_296 = view_327 = None
        sum_34: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_402, [0], True);  view_402 = None
        view_403: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_34, [2048]);  sum_34 = None
        view_404: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_34, [77, 32, 512]);  mm_34 = None
        convert_element_type_614: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_404, torch.float32);  view_404 = None
        convert_element_type_615: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_35, torch.float32);  mm_35 = None
        convert_element_type_616: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_403, torch.float32);  view_403 = None
        mul_205: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, primals_272);  primals_272 = None
        mul_206: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, 512)
        sum_35: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_205, [2], True)
        mul_207: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, mul_204);  mul_205 = None
        sum_36: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_207, [2], True);  mul_207 = None
        mul_208: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_204, sum_36);  sum_36 = None
        sub_70: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_206, sum_35);  mul_206 = sum_35 = None
        sub_71: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_70, mul_208);  sub_70 = mul_208 = None
        mul_209: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_15, sub_71);  div_15 = sub_71 = None
        mul_210: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_614, mul_204);  mul_204 = None
        sum_37: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_210, [0, 1]);  mul_210 = None
        sum_38: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_614, [0, 1]);  convert_element_type_614 = None
        add_177: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_175, mul_209);  add_175 = mul_209 = None
        convert_element_type_617: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.float16)
        clone_140: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_617, memory_format = torch.contiguous_format);  convert_element_type_617 = None
        view_405: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_140, [2464, 512]);  clone_140 = None
        mm_36: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_405, permute_299);  permute_299 = None
        permute_300: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_405, [1, 0])
        permute_220: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_279, [2, 0, 1, 3])
        view_325: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_220, [2464, 512]);  permute_220 = None
        mm_37: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_300, view_325);  permute_300 = view_325 = None
        sum_39: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_405, [0], True);  view_405 = None
        view_406: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None
        convert_element_type_622: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_37, torch.float32);  mm_37 = None
        convert_element_type_623: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_406, torch.float32);  view_406 = None
        view_407: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_36, [77, 32, 8, 64]);  mm_36 = None
        permute_303: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_407, [1, 2, 0, 3]);  view_407 = None
        _scaled_dot_product_cudnn_attention_backward_2 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_303, view_322, view_323, view_324, getitem_279, getitem_280, getitem_285, getitem_286, None, None, None, 77, 77, 0.0, True);  permute_303 = view_322 = view_323 = view_324 = getitem_279 = getitem_280 = getitem_285 = getitem_286 = None
        getitem_324: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[0]
        getitem_325: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[1]
        getitem_326: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_2[2];  _scaled_dot_product_cudnn_attention_backward_2 = None
        view_408: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_326, [256, 77, 64]);  getitem_326 = None
        view_409: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_325, [256, 77, 64]);  getitem_325 = None
        view_410: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_324, [256, 77, 64]);  getitem_324 = None
        permute_304: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_408, [1, 0, 2]);  view_408 = None
        view_411: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_304, [77, 32, 512]);  permute_304 = None
        permute_305: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_409, [1, 0, 2]);  view_409 = None
        view_412: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_305, [77, 32, 512]);  permute_305 = None
        permute_306: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_410, [1, 0, 2]);  view_410 = None
        view_413: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_306, [77, 32, 512]);  permute_306 = None

        # No stacktrace found for following nodes
        select_scatter_default_6: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_411, 0, 2);  view_411 = None
        select_scatter_default_7: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_412, 0, 1);  view_412 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_178: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_6, select_scatter_default_7);  select_scatter_default_6 = select_scatter_default_7 = None

        # No stacktrace found for following nodes
        select_scatter_default_8: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_413, 0, 0);  view_413 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_179: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, select_scatter_default_8);  add_178 = select_scatter_default_8 = None
        unsqueeze_27: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_179, 3);  add_179 = None
        permute_307: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_27, [3, 1, 2, 0, 4]);  unsqueeze_27 = None
        squeeze_26: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_307, 0);  permute_307 = None
        clone_141: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_26, memory_format = torch.contiguous_format);  squeeze_26 = None
        view_414: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_141, [77, 32, 1536]);  clone_141 = None
        view_415: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_414, [2464, 1536]);  view_414 = None
        mm_38: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_415, permute_308);  permute_308 = None
        permute_309: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_415, [1, 0])
        mm_39: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_309, view_316);  permute_309 = view_316 = None
        sum_40: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_415, [0], True);  view_415 = None
        view_416: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_40, [1536]);  sum_40 = None
        view_417: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_38, [77, 32, 512]);  mm_38 = None
        convert_element_type_628: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_417, torch.float32);  view_417 = None
        convert_element_type_629: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_39, torch.float32);  mm_39 = None
        convert_element_type_630: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_416, torch.float32);  view_416 = None
        mul_212: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_628, primals_266);  primals_266 = None
        mul_213: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, 512)
        sum_41: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_212, [2], True)
        mul_214: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, mul_211);  mul_212 = None
        sum_42: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_214, [2], True);  mul_214 = None
        mul_215: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_211, sum_42);  sum_42 = None
        sub_73: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_213, sum_41);  mul_213 = sum_41 = None
        sub_74: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_73, mul_215);  sub_73 = mul_215 = None
        mul_216: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_16, sub_74);  div_16 = sub_74 = None
        mul_217: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_628, mul_211);  mul_211 = None
        sum_43: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_217, [0, 1]);  mul_217 = None
        sum_44: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_628, [0, 1]);  convert_element_type_628 = None
        add_180: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_177, mul_216);  add_177 = mul_216 = None
        convert_element_type_631: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_180, torch.float16)
        clone_142: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_631, memory_format = torch.contiguous_format);  convert_element_type_631 = None
        view_418: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_142, [2464, 512]);  clone_142 = None
        mm_40: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_418, permute_312);  permute_312 = None
        permute_313: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_418, [1, 0])
        mm_41: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_313, view_314);  permute_313 = view_314 = None
        sum_45: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_418, [0], True);  view_418 = None
        view_419: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_45, [512]);  sum_45 = None
        view_420: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_40, [77, 32, 2048]);  mm_40 = None
        convert_element_type_636: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_41, torch.float32);  mm_41 = None
        convert_element_type_637: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_419, torch.float32);  view_419 = None
        view_313: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_70, [77, 32, 2048]);  addmm_70 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_128: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_313, 1.702)
        sigmoid_20: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_128);  mul_128 = None
        mul_218: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_420, sigmoid_20)
        mul_219: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_420, view_313);  view_420 = view_313 = None
        convert_element_type_638: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_219, torch.float32);  mul_219 = None
        convert_element_type_639: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_20, torch.float32);  sigmoid_20 = None
        sub_75: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_639)
        mul_220: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_639, sub_75);  convert_element_type_639 = sub_75 = None
        mul_221: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_638, mul_220);  convert_element_type_638 = mul_220 = None
        convert_element_type_640: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_221, torch.float16);  mul_221 = None
        mul_222: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_640, 1.702);  convert_element_type_640 = None
        add_181: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, mul_222);  mul_218 = mul_222 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_421: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_181, [2464, 2048]);  add_181 = None
        mm_42: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_421, permute_316);  permute_316 = None
        permute_317: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_421, [1, 0])
        mm_43: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_317, view_312);  permute_317 = view_312 = None
        sum_46: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_421, [0], True);  view_421 = None
        view_422: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_46, [2048]);  sum_46 = None
        view_423: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_42, [77, 32, 512]);  mm_42 = None
        convert_element_type_645: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_423, torch.float32);  view_423 = None
        convert_element_type_646: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_43, torch.float32);  mm_43 = None
        convert_element_type_647: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_422, torch.float32);  view_422 = None
        mul_224: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_645, primals_260);  primals_260 = None
        mul_225: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, 512)
        sum_47: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_224, [2], True)
        mul_226: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, mul_223);  mul_224 = None
        sum_48: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_226, [2], True);  mul_226 = None
        mul_227: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_223, sum_48);  sum_48 = None
        sub_77: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_225, sum_47);  mul_225 = sum_47 = None
        sub_78: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_77, mul_227);  sub_77 = mul_227 = None
        mul_228: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_17, sub_78);  div_17 = sub_78 = None
        mul_229: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_645, mul_223);  mul_223 = None
        sum_49: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_229, [0, 1]);  mul_229 = None
        sum_50: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_645, [0, 1]);  convert_element_type_645 = None
        add_182: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_180, mul_228);  add_180 = mul_228 = None
        convert_element_type_648: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_182, torch.float16)
        clone_143: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_648, memory_format = torch.contiguous_format);  convert_element_type_648 = None
        view_424: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_143, [2464, 512]);  clone_143 = None
        mm_44: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_424, permute_320);  permute_320 = None
        permute_321: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_424, [1, 0])
        permute_211: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_266, [2, 0, 1, 3])
        view_310: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_211, [2464, 512]);  permute_211 = None
        mm_45: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_321, view_310);  permute_321 = view_310 = None
        sum_51: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_424, [0], True);  view_424 = None
        view_425: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_51, [512]);  sum_51 = None
        convert_element_type_653: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_45, torch.float32);  mm_45 = None
        convert_element_type_654: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_425, torch.float32);  view_425 = None
        view_426: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_44, [77, 32, 8, 64]);  mm_44 = None
        permute_324: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_426, [1, 2, 0, 3]);  view_426 = None
        _scaled_dot_product_cudnn_attention_backward_3 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_324, view_307, view_308, view_309, getitem_266, getitem_267, getitem_272, getitem_273, None, None, None, 77, 77, 0.0, True);  permute_324 = view_307 = view_308 = view_309 = getitem_266 = getitem_267 = getitem_272 = getitem_273 = None
        getitem_327: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[0]
        getitem_328: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[1]
        getitem_329: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_3[2];  _scaled_dot_product_cudnn_attention_backward_3 = None
        view_427: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_329, [256, 77, 64]);  getitem_329 = None
        view_428: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_328, [256, 77, 64]);  getitem_328 = None
        view_429: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_327, [256, 77, 64]);  getitem_327 = None
        permute_325: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_427, [1, 0, 2]);  view_427 = None
        view_430: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_325, [77, 32, 512]);  permute_325 = None
        permute_326: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_428, [1, 0, 2]);  view_428 = None
        view_431: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_326, [77, 32, 512]);  permute_326 = None
        permute_327: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_429, [1, 0, 2]);  view_429 = None
        view_432: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_327, [77, 32, 512]);  permute_327 = None

        # No stacktrace found for following nodes
        select_scatter_default_9: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_430, 0, 2);  view_430 = None
        select_scatter_default_10: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_431, 0, 1);  view_431 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_183: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_9, select_scatter_default_10);  select_scatter_default_9 = select_scatter_default_10 = None

        # No stacktrace found for following nodes
        select_scatter_default_11: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_432, 0, 0);  view_432 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_184: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_183, select_scatter_default_11);  add_183 = select_scatter_default_11 = None
        unsqueeze_28: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_184, 3);  add_184 = None
        permute_328: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_28, [3, 1, 2, 0, 4]);  unsqueeze_28 = None
        squeeze_27: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_328, 0);  permute_328 = None
        clone_144: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_27, memory_format = torch.contiguous_format);  squeeze_27 = None
        view_433: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_144, [77, 32, 1536]);  clone_144 = None
        view_434: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_433, [2464, 1536]);  view_433 = None
        mm_46: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_434, permute_329);  permute_329 = None
        permute_330: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_434, [1, 0])
        mm_47: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_330, view_301);  permute_330 = view_301 = None
        sum_52: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_434, [0], True);  view_434 = None
        view_435: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_52, [1536]);  sum_52 = None
        view_436: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_46, [77, 32, 512]);  mm_46 = None
        convert_element_type_659: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_436, torch.float32);  view_436 = None
        convert_element_type_660: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_47, torch.float32);  mm_47 = None
        convert_element_type_661: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_435, torch.float32);  view_435 = None
        mul_231: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, primals_254);  primals_254 = None
        mul_232: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, 512)
        sum_53: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_231, [2], True)
        mul_233: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, mul_230);  mul_231 = None
        sum_54: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_233, [2], True);  mul_233 = None
        mul_234: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_230, sum_54);  sum_54 = None
        sub_80: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_232, sum_53);  mul_232 = sum_53 = None
        sub_81: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_80, mul_234);  sub_80 = mul_234 = None
        mul_235: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_18, sub_81);  div_18 = sub_81 = None
        mul_236: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_659, mul_230);  mul_230 = None
        sum_55: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_236, [0, 1]);  mul_236 = None
        sum_56: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_659, [0, 1]);  convert_element_type_659 = None
        add_185: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_182, mul_235);  add_182 = mul_235 = None
        convert_element_type_662: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_185, torch.float16)
        clone_145: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_662, memory_format = torch.contiguous_format);  convert_element_type_662 = None
        view_437: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_145, [2464, 512]);  clone_145 = None
        mm_48: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_437, permute_333);  permute_333 = None
        permute_334: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_437, [1, 0])
        mm_49: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_334, view_299);  permute_334 = view_299 = None
        sum_57: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_437, [0], True);  view_437 = None
        view_438: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_57, [512]);  sum_57 = None
        view_439: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_48, [77, 32, 2048]);  mm_48 = None
        convert_element_type_667: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_49, torch.float32);  mm_49 = None
        convert_element_type_668: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_438, torch.float32);  view_438 = None
        view_298: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_66, [77, 32, 2048]);  addmm_66 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_122: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_298, 1.702)
        sigmoid_19: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_122);  mul_122 = None
        mul_237: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_439, sigmoid_19)
        mul_238: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_439, view_298);  view_439 = view_298 = None
        convert_element_type_669: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_238, torch.float32);  mul_238 = None
        convert_element_type_670: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_19, torch.float32);  sigmoid_19 = None
        sub_82: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_670)
        mul_239: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_670, sub_82);  convert_element_type_670 = sub_82 = None
        mul_240: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_669, mul_239);  convert_element_type_669 = mul_239 = None
        convert_element_type_671: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_240, torch.float16);  mul_240 = None
        mul_241: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_671, 1.702);  convert_element_type_671 = None
        add_186: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_237, mul_241);  mul_237 = mul_241 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_440: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_186, [2464, 2048]);  add_186 = None
        mm_50: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_440, permute_337);  permute_337 = None
        permute_338: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_440, [1, 0])
        mm_51: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_338, view_297);  permute_338 = view_297 = None
        sum_58: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_440, [0], True);  view_440 = None
        view_441: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_58, [2048]);  sum_58 = None
        view_442: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_50, [77, 32, 512]);  mm_50 = None
        convert_element_type_676: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_442, torch.float32);  view_442 = None
        convert_element_type_677: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_51, torch.float32);  mm_51 = None
        convert_element_type_678: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_441, torch.float32);  view_441 = None
        mul_243: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_676, primals_248);  primals_248 = None
        mul_244: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, 512)
        sum_59: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_243, [2], True)
        mul_245: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_243, mul_242);  mul_243 = None
        sum_60: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_245, [2], True);  mul_245 = None
        mul_246: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, sum_60);  sum_60 = None
        sub_84: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_244, sum_59);  mul_244 = sum_59 = None
        sub_85: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_84, mul_246);  sub_84 = mul_246 = None
        mul_247: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_19, sub_85);  div_19 = sub_85 = None
        mul_248: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_676, mul_242);  mul_242 = None
        sum_61: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_248, [0, 1]);  mul_248 = None
        sum_62: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_676, [0, 1]);  convert_element_type_676 = None
        add_187: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_185, mul_247);  add_185 = mul_247 = None
        convert_element_type_679: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.float16)
        clone_146: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_679, memory_format = torch.contiguous_format);  convert_element_type_679 = None
        view_443: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_146, [2464, 512]);  clone_146 = None
        mm_52: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_443, permute_341);  permute_341 = None
        permute_342: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_443, [1, 0])
        permute_202: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_253, [2, 0, 1, 3])
        view_295: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_202, [2464, 512]);  permute_202 = None
        mm_53: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_342, view_295);  permute_342 = view_295 = None
        sum_63: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_443, [0], True);  view_443 = None
        view_444: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_63, [512]);  sum_63 = None
        convert_element_type_684: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_53, torch.float32);  mm_53 = None
        convert_element_type_685: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_444, torch.float32);  view_444 = None
        view_445: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_52, [77, 32, 8, 64]);  mm_52 = None
        permute_345: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_445, [1, 2, 0, 3]);  view_445 = None
        _scaled_dot_product_cudnn_attention_backward_4 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_345, view_292, view_293, view_294, getitem_253, getitem_254, getitem_259, getitem_260, None, None, None, 77, 77, 0.0, True);  permute_345 = view_292 = view_293 = view_294 = getitem_253 = getitem_254 = getitem_259 = getitem_260 = None
        getitem_330: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[0]
        getitem_331: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[1]
        getitem_332: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_4[2];  _scaled_dot_product_cudnn_attention_backward_4 = None
        view_446: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_332, [256, 77, 64]);  getitem_332 = None
        view_447: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_331, [256, 77, 64]);  getitem_331 = None
        view_448: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_330, [256, 77, 64]);  getitem_330 = None
        permute_346: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_446, [1, 0, 2]);  view_446 = None
        view_449: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_346, [77, 32, 512]);  permute_346 = None
        permute_347: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_447, [1, 0, 2]);  view_447 = None
        view_450: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_347, [77, 32, 512]);  permute_347 = None
        permute_348: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_448, [1, 0, 2]);  view_448 = None
        view_451: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_348, [77, 32, 512]);  permute_348 = None

        # No stacktrace found for following nodes
        select_scatter_default_12: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_449, 0, 2);  view_449 = None
        select_scatter_default_13: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_450, 0, 1);  view_450 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_188: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_12, select_scatter_default_13);  select_scatter_default_12 = select_scatter_default_13 = None

        # No stacktrace found for following nodes
        select_scatter_default_14: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_451, 0, 0);  view_451 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_189: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_188, select_scatter_default_14);  add_188 = select_scatter_default_14 = None
        unsqueeze_29: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_189, 3);  add_189 = None
        permute_349: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_29, [3, 1, 2, 0, 4]);  unsqueeze_29 = None
        squeeze_28: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_349, 0);  permute_349 = None
        clone_147: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_28, memory_format = torch.contiguous_format);  squeeze_28 = None
        view_452: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_147, [77, 32, 1536]);  clone_147 = None
        view_453: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_452, [2464, 1536]);  view_452 = None
        mm_54: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_453, permute_350);  permute_350 = None
        permute_351: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_453, [1, 0])
        mm_55: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_351, view_286);  permute_351 = view_286 = None
        sum_64: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_453, [0], True);  view_453 = None
        view_454: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_64, [1536]);  sum_64 = None
        view_455: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_54, [77, 32, 512]);  mm_54 = None
        convert_element_type_690: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_455, torch.float32);  view_455 = None
        convert_element_type_691: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_55, torch.float32);  mm_55 = None
        convert_element_type_692: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_454, torch.float32);  view_454 = None
        mul_250: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, primals_242);  primals_242 = None
        mul_251: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, 512)
        sum_65: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_250, [2], True)
        mul_252: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_250, mul_249);  mul_250 = None
        sum_66: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_252, [2], True);  mul_252 = None
        mul_253: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, sum_66);  sum_66 = None
        sub_87: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_251, sum_65);  mul_251 = sum_65 = None
        sub_88: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_87, mul_253);  sub_87 = mul_253 = None
        mul_254: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_20, sub_88);  div_20 = sub_88 = None
        mul_255: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_690, mul_249);  mul_249 = None
        sum_67: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_255, [0, 1]);  mul_255 = None
        sum_68: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_690, [0, 1]);  convert_element_type_690 = None
        add_190: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_187, mul_254);  add_187 = mul_254 = None
        convert_element_type_693: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.float16)
        clone_148: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_693, memory_format = torch.contiguous_format);  convert_element_type_693 = None
        view_456: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_148, [2464, 512]);  clone_148 = None
        mm_56: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_456, permute_354);  permute_354 = None
        permute_355: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_456, [1, 0])
        mm_57: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_355, view_284);  permute_355 = view_284 = None
        sum_69: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_456, [0], True);  view_456 = None
        view_457: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_69, [512]);  sum_69 = None
        view_458: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_56, [77, 32, 2048]);  mm_56 = None
        convert_element_type_698: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_57, torch.float32);  mm_57 = None
        convert_element_type_699: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_457, torch.float32);  view_457 = None
        view_283: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_62, [77, 32, 2048]);  addmm_62 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_116: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_283, 1.702)
        sigmoid_18: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_116);  mul_116 = None
        mul_256: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_458, sigmoid_18)
        mul_257: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_458, view_283);  view_458 = view_283 = None
        convert_element_type_700: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_257, torch.float32);  mul_257 = None
        convert_element_type_701: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_18, torch.float32);  sigmoid_18 = None
        sub_89: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_701)
        mul_258: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_701, sub_89);  convert_element_type_701 = sub_89 = None
        mul_259: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_700, mul_258);  convert_element_type_700 = mul_258 = None
        convert_element_type_702: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_259, torch.float16);  mul_259 = None
        mul_260: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_702, 1.702);  convert_element_type_702 = None
        add_191: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_256, mul_260);  mul_256 = mul_260 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_459: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_191, [2464, 2048]);  add_191 = None
        mm_58: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_459, permute_358);  permute_358 = None
        permute_359: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_459, [1, 0])
        mm_59: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_359, view_282);  permute_359 = view_282 = None
        sum_70: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_459, [0], True);  view_459 = None
        view_460: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_70, [2048]);  sum_70 = None
        view_461: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_58, [77, 32, 512]);  mm_58 = None
        convert_element_type_707: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_461, torch.float32);  view_461 = None
        convert_element_type_708: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_59, torch.float32);  mm_59 = None
        convert_element_type_709: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_460, torch.float32);  view_460 = None
        mul_262: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_707, primals_236);  primals_236 = None
        mul_263: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, 512)
        sum_71: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_262, [2], True)
        mul_264: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_262, mul_261);  mul_262 = None
        sum_72: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_264, [2], True);  mul_264 = None
        mul_265: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_261, sum_72);  sum_72 = None
        sub_91: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_263, sum_71);  mul_263 = sum_71 = None
        sub_92: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_91, mul_265);  sub_91 = mul_265 = None
        mul_266: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_21, sub_92);  div_21 = sub_92 = None
        mul_267: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_707, mul_261);  mul_261 = None
        sum_73: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_267, [0, 1]);  mul_267 = None
        sum_74: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_707, [0, 1]);  convert_element_type_707 = None
        add_192: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_190, mul_266);  add_190 = mul_266 = None
        convert_element_type_710: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_192, torch.float16)
        clone_149: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_710, memory_format = torch.contiguous_format);  convert_element_type_710 = None
        view_462: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_149, [2464, 512]);  clone_149 = None
        mm_60: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_462, permute_362);  permute_362 = None
        permute_363: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_462, [1, 0])
        permute_193: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_240, [2, 0, 1, 3])
        view_280: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_193, [2464, 512]);  permute_193 = None
        mm_61: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_363, view_280);  permute_363 = view_280 = None
        sum_75: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_462, [0], True);  view_462 = None
        view_463: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_75, [512]);  sum_75 = None
        convert_element_type_715: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_61, torch.float32);  mm_61 = None
        convert_element_type_716: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_463, torch.float32);  view_463 = None
        view_464: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_60, [77, 32, 8, 64]);  mm_60 = None
        permute_366: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_464, [1, 2, 0, 3]);  view_464 = None
        _scaled_dot_product_cudnn_attention_backward_5 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_366, view_277, view_278, view_279, getitem_240, getitem_241, getitem_246, getitem_247, None, None, None, 77, 77, 0.0, True);  permute_366 = view_277 = view_278 = view_279 = getitem_240 = getitem_241 = getitem_246 = getitem_247 = None
        getitem_333: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[0]
        getitem_334: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[1]
        getitem_335: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_5[2];  _scaled_dot_product_cudnn_attention_backward_5 = None
        view_465: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_335, [256, 77, 64]);  getitem_335 = None
        view_466: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_334, [256, 77, 64]);  getitem_334 = None
        view_467: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_333, [256, 77, 64]);  getitem_333 = None
        permute_367: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_465, [1, 0, 2]);  view_465 = None
        view_468: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_367, [77, 32, 512]);  permute_367 = None
        permute_368: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_466, [1, 0, 2]);  view_466 = None
        view_469: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_368, [77, 32, 512]);  permute_368 = None
        permute_369: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_467, [1, 0, 2]);  view_467 = None
        view_470: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_369, [77, 32, 512]);  permute_369 = None

        # No stacktrace found for following nodes
        select_scatter_default_15: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_468, 0, 2);  view_468 = None
        select_scatter_default_16: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_469, 0, 1);  view_469 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_193: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_15, select_scatter_default_16);  select_scatter_default_15 = select_scatter_default_16 = None

        # No stacktrace found for following nodes
        select_scatter_default_17: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_470, 0, 0);  view_470 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_194: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_193, select_scatter_default_17);  add_193 = select_scatter_default_17 = None
        unsqueeze_30: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_194, 3);  add_194 = None
        permute_370: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_30, [3, 1, 2, 0, 4]);  unsqueeze_30 = None
        squeeze_29: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_370, 0);  permute_370 = None
        clone_150: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_29, memory_format = torch.contiguous_format);  squeeze_29 = None
        view_471: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_150, [77, 32, 1536]);  clone_150 = None
        view_472: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_471, [2464, 1536]);  view_471 = None
        mm_62: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_472, permute_371);  permute_371 = None
        permute_372: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_472, [1, 0])
        mm_63: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_372, view_271);  permute_372 = view_271 = None
        sum_76: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_472, [0], True);  view_472 = None
        view_473: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_76, [1536]);  sum_76 = None
        view_474: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_62, [77, 32, 512]);  mm_62 = None
        convert_element_type_721: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_474, torch.float32);  view_474 = None
        convert_element_type_722: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_63, torch.float32);  mm_63 = None
        convert_element_type_723: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_473, torch.float32);  view_473 = None
        mul_269: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_721, primals_230);  primals_230 = None
        mul_270: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, 512)
        sum_77: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_269, [2], True)
        mul_271: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, mul_268);  mul_269 = None
        sum_78: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_271, [2], True);  mul_271 = None
        mul_272: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_268, sum_78);  sum_78 = None
        sub_94: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_270, sum_77);  mul_270 = sum_77 = None
        sub_95: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_94, mul_272);  sub_94 = mul_272 = None
        mul_273: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_22, sub_95);  div_22 = sub_95 = None
        mul_274: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_721, mul_268);  mul_268 = None
        sum_79: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_274, [0, 1]);  mul_274 = None
        sum_80: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_721, [0, 1]);  convert_element_type_721 = None
        add_195: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_192, mul_273);  add_192 = mul_273 = None
        convert_element_type_724: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_195, torch.float16)
        clone_151: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_724, memory_format = torch.contiguous_format);  convert_element_type_724 = None
        view_475: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_151, [2464, 512]);  clone_151 = None
        mm_64: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_475, permute_375);  permute_375 = None
        permute_376: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_475, [1, 0])
        mm_65: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_376, view_269);  permute_376 = view_269 = None
        sum_81: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_475, [0], True);  view_475 = None
        view_476: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_81, [512]);  sum_81 = None
        view_477: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_64, [77, 32, 2048]);  mm_64 = None
        convert_element_type_729: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_65, torch.float32);  mm_65 = None
        convert_element_type_730: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_476, torch.float32);  view_476 = None
        view_268: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_58, [77, 32, 2048]);  addmm_58 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_110: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_268, 1.702)
        sigmoid_17: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_110);  mul_110 = None
        mul_275: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_477, sigmoid_17)
        mul_276: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_477, view_268);  view_477 = view_268 = None
        convert_element_type_731: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_276, torch.float32);  mul_276 = None
        convert_element_type_732: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_17, torch.float32);  sigmoid_17 = None
        sub_96: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_732)
        mul_277: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_732, sub_96);  convert_element_type_732 = sub_96 = None
        mul_278: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_731, mul_277);  convert_element_type_731 = mul_277 = None
        convert_element_type_733: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_278, torch.float16);  mul_278 = None
        mul_279: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_733, 1.702);  convert_element_type_733 = None
        add_196: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, mul_279);  mul_275 = mul_279 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_478: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_196, [2464, 2048]);  add_196 = None
        mm_66: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_478, permute_379);  permute_379 = None
        permute_380: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_478, [1, 0])
        mm_67: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_380, view_267);  permute_380 = view_267 = None
        sum_82: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_478, [0], True);  view_478 = None
        view_479: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_82, [2048]);  sum_82 = None
        view_480: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_66, [77, 32, 512]);  mm_66 = None
        convert_element_type_738: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_480, torch.float32);  view_480 = None
        convert_element_type_739: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_67, torch.float32);  mm_67 = None
        convert_element_type_740: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_479, torch.float32);  view_479 = None
        mul_281: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_738, primals_224);  primals_224 = None
        mul_282: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, 512)
        sum_83: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_281, [2], True)
        mul_283: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, mul_280);  mul_281 = None
        sum_84: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_283, [2], True);  mul_283 = None
        mul_284: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_280, sum_84);  sum_84 = None
        sub_98: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_282, sum_83);  mul_282 = sum_83 = None
        sub_99: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_98, mul_284);  sub_98 = mul_284 = None
        mul_285: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_23, sub_99);  div_23 = sub_99 = None
        mul_286: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_738, mul_280);  mul_280 = None
        sum_85: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_286, [0, 1]);  mul_286 = None
        sum_86: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_738, [0, 1]);  convert_element_type_738 = None
        add_197: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_195, mul_285);  add_195 = mul_285 = None
        convert_element_type_741: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.float16)
        clone_152: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_741, memory_format = torch.contiguous_format);  convert_element_type_741 = None
        view_481: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_152, [2464, 512]);  clone_152 = None
        mm_68: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_481, permute_383);  permute_383 = None
        permute_384: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_481, [1, 0])
        permute_184: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_227, [2, 0, 1, 3])
        view_265: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_184, [2464, 512]);  permute_184 = None
        mm_69: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_384, view_265);  permute_384 = view_265 = None
        sum_87: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_481, [0], True);  view_481 = None
        view_482: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_87, [512]);  sum_87 = None
        convert_element_type_746: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_69, torch.float32);  mm_69 = None
        convert_element_type_747: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_482, torch.float32);  view_482 = None
        view_483: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_68, [77, 32, 8, 64]);  mm_68 = None
        permute_387: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_483, [1, 2, 0, 3]);  view_483 = None
        _scaled_dot_product_cudnn_attention_backward_6 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_387, view_262, view_263, view_264, getitem_227, getitem_228, getitem_233, getitem_234, None, None, None, 77, 77, 0.0, True);  permute_387 = view_262 = view_263 = view_264 = getitem_227 = getitem_228 = getitem_233 = getitem_234 = None
        getitem_336: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[0]
        getitem_337: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[1]
        getitem_338: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_6[2];  _scaled_dot_product_cudnn_attention_backward_6 = None
        view_484: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_338, [256, 77, 64]);  getitem_338 = None
        view_485: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_337, [256, 77, 64]);  getitem_337 = None
        view_486: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_336, [256, 77, 64]);  getitem_336 = None
        permute_388: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_484, [1, 0, 2]);  view_484 = None
        view_487: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_388, [77, 32, 512]);  permute_388 = None
        permute_389: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_485, [1, 0, 2]);  view_485 = None
        view_488: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_389, [77, 32, 512]);  permute_389 = None
        permute_390: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_486, [1, 0, 2]);  view_486 = None
        view_489: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_390, [77, 32, 512]);  permute_390 = None

        # No stacktrace found for following nodes
        select_scatter_default_18: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_487, 0, 2);  view_487 = None
        select_scatter_default_19: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_488, 0, 1);  view_488 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_198: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_18, select_scatter_default_19);  select_scatter_default_18 = select_scatter_default_19 = None

        # No stacktrace found for following nodes
        select_scatter_default_20: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_489, 0, 0);  view_489 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_199: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_198, select_scatter_default_20);  add_198 = select_scatter_default_20 = None
        unsqueeze_31: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_199, 3);  add_199 = None
        permute_391: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_31, [3, 1, 2, 0, 4]);  unsqueeze_31 = None
        squeeze_30: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_391, 0);  permute_391 = None
        clone_153: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_30, memory_format = torch.contiguous_format);  squeeze_30 = None
        view_490: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_153, [77, 32, 1536]);  clone_153 = None
        view_491: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_490, [2464, 1536]);  view_490 = None
        mm_70: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_491, permute_392);  permute_392 = None
        permute_393: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_491, [1, 0])
        mm_71: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_393, view_256);  permute_393 = view_256 = None
        sum_88: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_491, [0], True);  view_491 = None
        view_492: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_88, [1536]);  sum_88 = None
        view_493: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_70, [77, 32, 512]);  mm_70 = None
        convert_element_type_752: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_493, torch.float32);  view_493 = None
        convert_element_type_753: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_71, torch.float32);  mm_71 = None
        convert_element_type_754: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_492, torch.float32);  view_492 = None
        mul_288: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_752, primals_218);  primals_218 = None
        mul_289: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, 512)
        sum_89: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_288, [2], True)
        mul_290: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, mul_287);  mul_288 = None
        sum_90: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_290, [2], True);  mul_290 = None
        mul_291: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_287, sum_90);  sum_90 = None
        sub_101: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_289, sum_89);  mul_289 = sum_89 = None
        sub_102: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_101, mul_291);  sub_101 = mul_291 = None
        mul_292: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_24, sub_102);  div_24 = sub_102 = None
        mul_293: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_752, mul_287);  mul_287 = None
        sum_91: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_293, [0, 1]);  mul_293 = None
        sum_92: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_752, [0, 1]);  convert_element_type_752 = None
        add_200: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_197, mul_292);  add_197 = mul_292 = None
        convert_element_type_755: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_200, torch.float16)
        clone_154: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_755, memory_format = torch.contiguous_format);  convert_element_type_755 = None
        view_494: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_154, [2464, 512]);  clone_154 = None
        mm_72: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_494, permute_396);  permute_396 = None
        permute_397: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_494, [1, 0])
        mm_73: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_397, view_254);  permute_397 = view_254 = None
        sum_93: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_494, [0], True);  view_494 = None
        view_495: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_93, [512]);  sum_93 = None
        view_496: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_72, [77, 32, 2048]);  mm_72 = None
        convert_element_type_760: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_73, torch.float32);  mm_73 = None
        convert_element_type_761: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_495, torch.float32);  view_495 = None
        view_253: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_54, [77, 32, 2048]);  addmm_54 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_104: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_253, 1.702)
        sigmoid_16: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_104);  mul_104 = None
        mul_294: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_496, sigmoid_16)
        mul_295: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_496, view_253);  view_496 = view_253 = None
        convert_element_type_762: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_295, torch.float32);  mul_295 = None
        convert_element_type_763: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_16, torch.float32);  sigmoid_16 = None
        sub_103: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_763)
        mul_296: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_763, sub_103);  convert_element_type_763 = sub_103 = None
        mul_297: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_762, mul_296);  convert_element_type_762 = mul_296 = None
        convert_element_type_764: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_297, torch.float16);  mul_297 = None
        mul_298: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_764, 1.702);  convert_element_type_764 = None
        add_201: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_294, mul_298);  mul_294 = mul_298 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_497: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_201, [2464, 2048]);  add_201 = None
        mm_74: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_497, permute_400);  permute_400 = None
        permute_401: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_497, [1, 0])
        mm_75: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_401, view_252);  permute_401 = view_252 = None
        sum_94: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_497, [0], True);  view_497 = None
        view_498: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_94, [2048]);  sum_94 = None
        view_499: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_74, [77, 32, 512]);  mm_74 = None
        convert_element_type_769: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_499, torch.float32);  view_499 = None
        convert_element_type_770: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_75, torch.float32);  mm_75 = None
        convert_element_type_771: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_498, torch.float32);  view_498 = None
        mul_300: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, primals_212);  primals_212 = None
        mul_301: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, 512)
        sum_95: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_300, [2], True)
        mul_302: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_300, mul_299);  mul_300 = None
        sum_96: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_302, [2], True);  mul_302 = None
        mul_303: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, sum_96);  sum_96 = None
        sub_105: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_301, sum_95);  mul_301 = sum_95 = None
        sub_106: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_105, mul_303);  sub_105 = mul_303 = None
        mul_304: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_25, sub_106);  div_25 = sub_106 = None
        mul_305: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_769, mul_299);  mul_299 = None
        sum_97: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_305, [0, 1]);  mul_305 = None
        sum_98: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_769, [0, 1]);  convert_element_type_769 = None
        add_202: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, mul_304);  add_200 = mul_304 = None
        convert_element_type_772: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_202, torch.float16)
        clone_155: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_772, memory_format = torch.contiguous_format);  convert_element_type_772 = None
        view_500: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_155, [2464, 512]);  clone_155 = None
        mm_76: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_500, permute_404);  permute_404 = None
        permute_405: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_500, [1, 0])
        permute_175: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_214, [2, 0, 1, 3])
        view_250: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_175, [2464, 512]);  permute_175 = None
        mm_77: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_405, view_250);  permute_405 = view_250 = None
        sum_99: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        view_501: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_99, [512]);  sum_99 = None
        convert_element_type_777: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_77, torch.float32);  mm_77 = None
        convert_element_type_778: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_501, torch.float32);  view_501 = None
        view_502: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_76, [77, 32, 8, 64]);  mm_76 = None
        permute_408: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_502, [1, 2, 0, 3]);  view_502 = None
        _scaled_dot_product_cudnn_attention_backward_7 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_408, view_247, view_248, view_249, getitem_214, getitem_215, getitem_220, getitem_221, None, None, None, 77, 77, 0.0, True);  permute_408 = view_247 = view_248 = view_249 = getitem_214 = getitem_215 = getitem_220 = getitem_221 = None
        getitem_339: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[0]
        getitem_340: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[1]
        getitem_341: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_7[2];  _scaled_dot_product_cudnn_attention_backward_7 = None
        view_503: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_341, [256, 77, 64]);  getitem_341 = None
        view_504: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_340, [256, 77, 64]);  getitem_340 = None
        view_505: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_339, [256, 77, 64]);  getitem_339 = None
        permute_409: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_503, [1, 0, 2]);  view_503 = None
        view_506: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_409, [77, 32, 512]);  permute_409 = None
        permute_410: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_504, [1, 0, 2]);  view_504 = None
        view_507: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_410, [77, 32, 512]);  permute_410 = None
        permute_411: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_505, [1, 0, 2]);  view_505 = None
        view_508: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_411, [77, 32, 512]);  permute_411 = None

        # No stacktrace found for following nodes
        select_scatter_default_21: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_506, 0, 2);  view_506 = None
        select_scatter_default_22: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_507, 0, 1);  view_507 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_203: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_21, select_scatter_default_22);  select_scatter_default_21 = select_scatter_default_22 = None

        # No stacktrace found for following nodes
        select_scatter_default_23: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_508, 0, 0);  view_508 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_204: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_203, select_scatter_default_23);  add_203 = select_scatter_default_23 = None
        unsqueeze_32: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_204, 3);  add_204 = None
        permute_412: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_32, [3, 1, 2, 0, 4]);  unsqueeze_32 = None
        squeeze_31: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_412, 0);  permute_412 = None
        clone_156: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_31, memory_format = torch.contiguous_format);  squeeze_31 = None
        view_509: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_156, [77, 32, 1536]);  clone_156 = None
        view_510: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_509, [2464, 1536]);  view_509 = None
        mm_78: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_510, permute_413);  permute_413 = None
        permute_414: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_510, [1, 0])
        mm_79: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_414, view_241);  permute_414 = view_241 = None
        sum_100: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_510, [0], True);  view_510 = None
        view_511: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_100, [1536]);  sum_100 = None
        view_512: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_78, [77, 32, 512]);  mm_78 = None
        convert_element_type_783: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_512, torch.float32);  view_512 = None
        convert_element_type_784: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_79, torch.float32);  mm_79 = None
        convert_element_type_785: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_511, torch.float32);  view_511 = None
        mul_307: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_783, primals_206);  primals_206 = None
        mul_308: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_307, 512)
        sum_101: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_307, [2], True)
        mul_309: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_307, mul_306);  mul_307 = None
        sum_102: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_309, [2], True);  mul_309 = None
        mul_310: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, sum_102);  sum_102 = None
        sub_108: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_308, sum_101);  mul_308 = sum_101 = None
        sub_109: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_108, mul_310);  sub_108 = mul_310 = None
        mul_311: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_26, sub_109);  div_26 = sub_109 = None
        mul_312: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_783, mul_306);  mul_306 = None
        sum_103: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_312, [0, 1]);  mul_312 = None
        sum_104: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_783, [0, 1]);  convert_element_type_783 = None
        add_205: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_202, mul_311);  add_202 = mul_311 = None
        convert_element_type_786: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_205, torch.float16)
        clone_157: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_786, memory_format = torch.contiguous_format);  convert_element_type_786 = None
        view_513: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_157, [2464, 512]);  clone_157 = None
        mm_80: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_513, permute_417);  permute_417 = None
        permute_418: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_513, [1, 0])
        mm_81: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_418, view_239);  permute_418 = view_239 = None
        sum_105: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_513, [0], True);  view_513 = None
        view_514: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_105, [512]);  sum_105 = None
        view_515: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_80, [77, 32, 2048]);  mm_80 = None
        convert_element_type_791: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_81, torch.float32);  mm_81 = None
        convert_element_type_792: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_514, torch.float32);  view_514 = None
        view_238: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_50, [77, 32, 2048]);  addmm_50 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_98: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_238, 1.702)
        sigmoid_15: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_98);  mul_98 = None
        mul_313: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_515, sigmoid_15)
        mul_314: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_515, view_238);  view_515 = view_238 = None
        convert_element_type_793: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_314, torch.float32);  mul_314 = None
        convert_element_type_794: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_15, torch.float32);  sigmoid_15 = None
        sub_110: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_794)
        mul_315: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_794, sub_110);  convert_element_type_794 = sub_110 = None
        mul_316: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_793, mul_315);  convert_element_type_793 = mul_315 = None
        convert_element_type_795: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_316, torch.float16);  mul_316 = None
        mul_317: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_795, 1.702);  convert_element_type_795 = None
        add_206: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_313, mul_317);  mul_313 = mul_317 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_516: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_206, [2464, 2048]);  add_206 = None
        mm_82: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_516, permute_421);  permute_421 = None
        permute_422: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_516, [1, 0])
        mm_83: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_422, view_237);  permute_422 = view_237 = None
        sum_106: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_516, [0], True);  view_516 = None
        view_517: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_106, [2048]);  sum_106 = None
        view_518: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_82, [77, 32, 512]);  mm_82 = None
        convert_element_type_800: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_518, torch.float32);  view_518 = None
        convert_element_type_801: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_83, torch.float32);  mm_83 = None
        convert_element_type_802: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_517, torch.float32);  view_517 = None
        mul_319: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_800, primals_200);  primals_200 = None
        mul_320: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, 512)
        sum_107: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_319, [2], True)
        mul_321: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_319, mul_318);  mul_319 = None
        sum_108: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_321, [2], True);  mul_321 = None
        mul_322: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_318, sum_108);  sum_108 = None
        sub_112: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_320, sum_107);  mul_320 = sum_107 = None
        sub_113: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_112, mul_322);  sub_112 = mul_322 = None
        mul_323: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_27, sub_113);  div_27 = sub_113 = None
        mul_324: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_800, mul_318);  mul_318 = None
        sum_109: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_324, [0, 1]);  mul_324 = None
        sum_110: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_800, [0, 1]);  convert_element_type_800 = None
        add_207: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_205, mul_323);  add_205 = mul_323 = None
        convert_element_type_803: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_207, torch.float16)
        clone_158: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_803, memory_format = torch.contiguous_format);  convert_element_type_803 = None
        view_519: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_158, [2464, 512]);  clone_158 = None
        mm_84: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_519, permute_425);  permute_425 = None
        permute_426: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_519, [1, 0])
        permute_166: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_201, [2, 0, 1, 3])
        view_235: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_166, [2464, 512]);  permute_166 = None
        mm_85: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_426, view_235);  permute_426 = view_235 = None
        sum_111: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
        view_520: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_111, [512]);  sum_111 = None
        convert_element_type_808: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_85, torch.float32);  mm_85 = None
        convert_element_type_809: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_520, torch.float32);  view_520 = None
        view_521: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_84, [77, 32, 8, 64]);  mm_84 = None
        permute_429: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_521, [1, 2, 0, 3]);  view_521 = None
        _scaled_dot_product_cudnn_attention_backward_8 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_429, view_232, view_233, view_234, getitem_201, getitem_202, getitem_207, getitem_208, None, None, None, 77, 77, 0.0, True);  permute_429 = view_232 = view_233 = view_234 = getitem_201 = getitem_202 = getitem_207 = getitem_208 = None
        getitem_342: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[0]
        getitem_343: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[1]
        getitem_344: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_8[2];  _scaled_dot_product_cudnn_attention_backward_8 = None
        view_522: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_344, [256, 77, 64]);  getitem_344 = None
        view_523: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_343, [256, 77, 64]);  getitem_343 = None
        view_524: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_342, [256, 77, 64]);  getitem_342 = None
        permute_430: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_522, [1, 0, 2]);  view_522 = None
        view_525: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_430, [77, 32, 512]);  permute_430 = None
        permute_431: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_523, [1, 0, 2]);  view_523 = None
        view_526: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_431, [77, 32, 512]);  permute_431 = None
        permute_432: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_524, [1, 0, 2]);  view_524 = None
        view_527: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_432, [77, 32, 512]);  permute_432 = None

        # No stacktrace found for following nodes
        select_scatter_default_24: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_525, 0, 2);  view_525 = None
        select_scatter_default_25: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_526, 0, 1);  view_526 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_208: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_24, select_scatter_default_25);  select_scatter_default_24 = select_scatter_default_25 = None

        # No stacktrace found for following nodes
        select_scatter_default_26: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_527, 0, 0);  view_527 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_209: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_208, select_scatter_default_26);  add_208 = select_scatter_default_26 = None
        unsqueeze_33: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_209, 3);  add_209 = None
        permute_433: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_33, [3, 1, 2, 0, 4]);  unsqueeze_33 = None
        squeeze_32: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_433, 0);  permute_433 = None
        clone_159: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_32, memory_format = torch.contiguous_format);  squeeze_32 = None
        view_528: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_159, [77, 32, 1536]);  clone_159 = None
        view_529: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_528, [2464, 1536]);  view_528 = None
        mm_86: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_529, permute_434);  permute_434 = None
        permute_435: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_529, [1, 0])
        mm_87: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_435, view_226);  permute_435 = view_226 = None
        sum_112: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_529, [0], True);  view_529 = None
        view_530: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_112, [1536]);  sum_112 = None
        view_531: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_86, [77, 32, 512]);  mm_86 = None
        convert_element_type_814: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_531, torch.float32);  view_531 = None
        convert_element_type_815: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_87, torch.float32);  mm_87 = None
        convert_element_type_816: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_530, torch.float32);  view_530 = None
        mul_326: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_814, primals_194);  primals_194 = None
        mul_327: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, 512)
        sum_113: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_326, [2], True)
        mul_328: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_326, mul_325);  mul_326 = None
        sum_114: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_328, [2], True);  mul_328 = None
        mul_329: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_325, sum_114);  sum_114 = None
        sub_115: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_327, sum_113);  mul_327 = sum_113 = None
        sub_116: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_115, mul_329);  sub_115 = mul_329 = None
        mul_330: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_28, sub_116);  div_28 = sub_116 = None
        mul_331: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_814, mul_325);  mul_325 = None
        sum_115: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_331, [0, 1]);  mul_331 = None
        sum_116: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_814, [0, 1]);  convert_element_type_814 = None
        add_210: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_207, mul_330);  add_207 = mul_330 = None
        convert_element_type_817: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.float16)
        clone_160: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_817, memory_format = torch.contiguous_format);  convert_element_type_817 = None
        view_532: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_160, [2464, 512]);  clone_160 = None
        mm_88: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_532, permute_438);  permute_438 = None
        permute_439: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_532, [1, 0])
        mm_89: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_439, view_224);  permute_439 = view_224 = None
        sum_117: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
        view_533: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_117, [512]);  sum_117 = None
        view_534: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_88, [77, 32, 2048]);  mm_88 = None
        convert_element_type_822: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_89, torch.float32);  mm_89 = None
        convert_element_type_823: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_533, torch.float32);  view_533 = None
        view_223: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [77, 32, 2048]);  addmm_46 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_92: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_223, 1.702)
        sigmoid_14: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_92);  mul_92 = None
        mul_332: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_534, sigmoid_14)
        mul_333: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_534, view_223);  view_534 = view_223 = None
        convert_element_type_824: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_333, torch.float32);  mul_333 = None
        convert_element_type_825: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_14, torch.float32);  sigmoid_14 = None
        sub_117: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_825)
        mul_334: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_825, sub_117);  convert_element_type_825 = sub_117 = None
        mul_335: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_824, mul_334);  convert_element_type_824 = mul_334 = None
        convert_element_type_826: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_335, torch.float16);  mul_335 = None
        mul_336: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_826, 1.702);  convert_element_type_826 = None
        add_211: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_535: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_211, [2464, 2048]);  add_211 = None
        mm_90: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_535, permute_442);  permute_442 = None
        permute_443: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_91: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_443, view_222);  permute_443 = view_222 = None
        sum_118: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_118, [2048]);  sum_118 = None
        view_537: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_90, [77, 32, 512]);  mm_90 = None
        convert_element_type_831: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_537, torch.float32);  view_537 = None
        convert_element_type_832: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_91, torch.float32);  mm_91 = None
        convert_element_type_833: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_536, torch.float32);  view_536 = None
        mul_338: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_831, primals_188);  primals_188 = None
        mul_339: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, 512)
        sum_119: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_338, [2], True)
        mul_340: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, mul_337);  mul_338 = None
        sum_120: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_340, [2], True);  mul_340 = None
        mul_341: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_337, sum_120);  sum_120 = None
        sub_119: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_339, sum_119);  mul_339 = sum_119 = None
        sub_120: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_119, mul_341);  sub_119 = mul_341 = None
        mul_342: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_29, sub_120);  div_29 = sub_120 = None
        mul_343: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_831, mul_337);  mul_337 = None
        sum_121: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_343, [0, 1]);  mul_343 = None
        sum_122: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_831, [0, 1]);  convert_element_type_831 = None
        add_212: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_210, mul_342);  add_210 = mul_342 = None
        convert_element_type_834: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.float16)
        clone_161: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_834, memory_format = torch.contiguous_format);  convert_element_type_834 = None
        view_538: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_161, [2464, 512]);  clone_161 = None
        mm_92: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_538, permute_446);  permute_446 = None
        permute_447: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_538, [1, 0])
        permute_157: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_188, [2, 0, 1, 3])
        view_220: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_157, [2464, 512]);  permute_157 = None
        mm_93: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_447, view_220);  permute_447 = view_220 = None
        sum_123: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        view_539: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_123, [512]);  sum_123 = None
        convert_element_type_839: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_93, torch.float32);  mm_93 = None
        convert_element_type_840: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_539, torch.float32);  view_539 = None
        view_540: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_92, [77, 32, 8, 64]);  mm_92 = None
        permute_450: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_540, [1, 2, 0, 3]);  view_540 = None
        _scaled_dot_product_cudnn_attention_backward_9 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_450, view_217, view_218, view_219, getitem_188, getitem_189, getitem_194, getitem_195, None, None, None, 77, 77, 0.0, True);  permute_450 = view_217 = view_218 = view_219 = getitem_188 = getitem_189 = getitem_194 = getitem_195 = None
        getitem_345: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[0]
        getitem_346: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[1]
        getitem_347: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_9[2];  _scaled_dot_product_cudnn_attention_backward_9 = None
        view_541: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_347, [256, 77, 64]);  getitem_347 = None
        view_542: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_346, [256, 77, 64]);  getitem_346 = None
        view_543: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_345, [256, 77, 64]);  getitem_345 = None
        permute_451: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_541, [1, 0, 2]);  view_541 = None
        view_544: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_451, [77, 32, 512]);  permute_451 = None
        permute_452: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_542, [1, 0, 2]);  view_542 = None
        view_545: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_452, [77, 32, 512]);  permute_452 = None
        permute_453: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_543, [1, 0, 2]);  view_543 = None
        view_546: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_453, [77, 32, 512]);  permute_453 = None

        # No stacktrace found for following nodes
        select_scatter_default_27: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_544, 0, 2);  view_544 = None
        select_scatter_default_28: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_545, 0, 1);  view_545 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_213: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_27, select_scatter_default_28);  select_scatter_default_27 = select_scatter_default_28 = None

        # No stacktrace found for following nodes
        select_scatter_default_29: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_546, 0, 0);  view_546 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_214: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_213, select_scatter_default_29);  add_213 = select_scatter_default_29 = None
        unsqueeze_34: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_214, 3);  add_214 = None
        permute_454: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_34, [3, 1, 2, 0, 4]);  unsqueeze_34 = None
        squeeze_33: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_454, 0);  permute_454 = None
        clone_162: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_33, memory_format = torch.contiguous_format);  squeeze_33 = None
        view_547: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_162, [77, 32, 1536]);  clone_162 = None
        view_548: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_547, [2464, 1536]);  view_547 = None
        mm_94: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_548, permute_455);  permute_455 = None
        permute_456: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_548, [1, 0])
        mm_95: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_456, view_211);  permute_456 = view_211 = None
        sum_124: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_124, [1536]);  sum_124 = None
        view_550: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_94, [77, 32, 512]);  mm_94 = None
        convert_element_type_845: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_550, torch.float32);  view_550 = None
        convert_element_type_846: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_95, torch.float32);  mm_95 = None
        convert_element_type_847: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_549, torch.float32);  view_549 = None
        mul_345: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_845, primals_182);  primals_182 = None
        mul_346: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, 512)
        sum_125: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_345, [2], True)
        mul_347: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, mul_344);  mul_345 = None
        sum_126: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_347, [2], True);  mul_347 = None
        mul_348: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_344, sum_126);  sum_126 = None
        sub_122: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_346, sum_125);  mul_346 = sum_125 = None
        sub_123: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_122, mul_348);  sub_122 = mul_348 = None
        mul_349: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_30, sub_123);  div_30 = sub_123 = None
        mul_350: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_845, mul_344);  mul_344 = None
        sum_127: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_350, [0, 1]);  mul_350 = None
        sum_128: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_845, [0, 1]);  convert_element_type_845 = None
        add_215: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_212, mul_349);  add_212 = mul_349 = None
        convert_element_type_848: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_215, torch.float16)
        clone_163: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_848, memory_format = torch.contiguous_format);  convert_element_type_848 = None
        view_551: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_163, [2464, 512]);  clone_163 = None
        mm_96: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_551, permute_459);  permute_459 = None
        permute_460: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_551, [1, 0])
        mm_97: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_460, view_209);  permute_460 = view_209 = None
        sum_129: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
        view_552: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_129, [512]);  sum_129 = None
        view_553: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_96, [77, 32, 2048]);  mm_96 = None
        convert_element_type_853: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_97, torch.float32);  mm_97 = None
        convert_element_type_854: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_552, torch.float32);  view_552 = None
        view_208: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [77, 32, 2048]);  addmm_42 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_86: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_208, 1.702)
        sigmoid_13: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_86);  mul_86 = None
        mul_351: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_553, sigmoid_13)
        mul_352: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_553, view_208);  view_553 = view_208 = None
        convert_element_type_855: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_352, torch.float32);  mul_352 = None
        convert_element_type_856: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_13, torch.float32);  sigmoid_13 = None
        sub_124: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_856)
        mul_353: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_856, sub_124);  convert_element_type_856 = sub_124 = None
        mul_354: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_855, mul_353);  convert_element_type_855 = mul_353 = None
        convert_element_type_857: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_354, torch.float16);  mul_354 = None
        mul_355: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_857, 1.702);  convert_element_type_857 = None
        add_216: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_351, mul_355);  mul_351 = mul_355 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_554: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_216, [2464, 2048]);  add_216 = None
        mm_98: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_554, permute_463);  permute_463 = None
        permute_464: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_554, [1, 0])
        mm_99: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_464, view_207);  permute_464 = view_207 = None
        sum_130: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
        view_555: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_130, [2048]);  sum_130 = None
        view_556: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_98, [77, 32, 512]);  mm_98 = None
        convert_element_type_862: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_556, torch.float32);  view_556 = None
        convert_element_type_863: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_99, torch.float32);  mm_99 = None
        convert_element_type_864: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_555, torch.float32);  view_555 = None
        mul_357: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_862, primals_176);  primals_176 = None
        mul_358: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, 512)
        sum_131: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True)
        mul_359: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_357, mul_356);  mul_357 = None
        sum_132: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_359, [2], True);  mul_359 = None
        mul_360: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, sum_132);  sum_132 = None
        sub_126: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_358, sum_131);  mul_358 = sum_131 = None
        sub_127: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_126, mul_360);  sub_126 = mul_360 = None
        mul_361: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_31, sub_127);  div_31 = sub_127 = None
        mul_362: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_862, mul_356);  mul_356 = None
        sum_133: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_362, [0, 1]);  mul_362 = None
        sum_134: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_862, [0, 1]);  convert_element_type_862 = None
        add_217: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_215, mul_361);  add_215 = mul_361 = None
        convert_element_type_865: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_217, torch.float16)
        clone_164: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_865, memory_format = torch.contiguous_format);  convert_element_type_865 = None
        view_557: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_164, [2464, 512]);  clone_164 = None
        mm_100: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_557, permute_467);  permute_467 = None
        permute_468: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_557, [1, 0])
        permute_148: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_175, [2, 0, 1, 3])
        view_205: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_148, [2464, 512]);  permute_148 = None
        mm_101: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_468, view_205);  permute_468 = view_205 = None
        sum_135: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
        view_558: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_135, [512]);  sum_135 = None
        convert_element_type_870: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_101, torch.float32);  mm_101 = None
        convert_element_type_871: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_558, torch.float32);  view_558 = None
        view_559: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_100, [77, 32, 8, 64]);  mm_100 = None
        permute_471: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_559, [1, 2, 0, 3]);  view_559 = None
        _scaled_dot_product_cudnn_attention_backward_10 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_471, view_202, view_203, view_204, getitem_175, getitem_176, getitem_181, getitem_182, None, None, None, 77, 77, 0.0, True);  permute_471 = view_202 = view_203 = view_204 = getitem_175 = getitem_176 = getitem_181 = getitem_182 = None
        getitem_348: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[0]
        getitem_349: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[1]
        getitem_350: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_10[2];  _scaled_dot_product_cudnn_attention_backward_10 = None
        view_560: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_350, [256, 77, 64]);  getitem_350 = None
        view_561: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_349, [256, 77, 64]);  getitem_349 = None
        view_562: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_348, [256, 77, 64]);  getitem_348 = None
        permute_472: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_560, [1, 0, 2]);  view_560 = None
        view_563: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_472, [77, 32, 512]);  permute_472 = None
        permute_473: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_561, [1, 0, 2]);  view_561 = None
        view_564: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_473, [77, 32, 512]);  permute_473 = None
        permute_474: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_562, [1, 0, 2]);  view_562 = None
        view_565: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_474, [77, 32, 512]);  permute_474 = None

        # No stacktrace found for following nodes
        select_scatter_default_30: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_563, 0, 2);  view_563 = None
        select_scatter_default_31: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_564, 0, 1);  view_564 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_218: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_30, select_scatter_default_31);  select_scatter_default_30 = select_scatter_default_31 = None

        # No stacktrace found for following nodes
        select_scatter_default_32: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_565, 0, 0);  view_565 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_219: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_218, select_scatter_default_32);  add_218 = select_scatter_default_32 = None
        unsqueeze_35: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_219, 3);  add_219 = None
        permute_475: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_35, [3, 1, 2, 0, 4]);  unsqueeze_35 = None
        squeeze_34: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_475, 0);  permute_475 = None
        clone_165: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_34, memory_format = torch.contiguous_format);  squeeze_34 = None
        view_566: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_165, [77, 32, 1536]);  clone_165 = None
        view_567: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_566, [2464, 1536]);  view_566 = None
        mm_102: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_567, permute_476);  permute_476 = None
        permute_477: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_567, [1, 0])
        mm_103: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_477, view_196);  permute_477 = view_196 = None
        sum_136: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
        view_568: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_136, [1536]);  sum_136 = None
        view_569: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_102, [77, 32, 512]);  mm_102 = None
        convert_element_type_876: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_569, torch.float32);  view_569 = None
        convert_element_type_877: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_103, torch.float32);  mm_103 = None
        convert_element_type_878: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_568, torch.float32);  view_568 = None
        mul_364: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_876, primals_170);  primals_170 = None
        mul_365: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, 512)
        sum_137: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True)
        mul_366: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_364, mul_363);  mul_364 = None
        sum_138: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_366, [2], True);  mul_366 = None
        mul_367: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_363, sum_138);  sum_138 = None
        sub_129: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_365, sum_137);  mul_365 = sum_137 = None
        sub_130: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_129, mul_367);  sub_129 = mul_367 = None
        mul_368: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_32, sub_130);  div_32 = sub_130 = None
        mul_369: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_876, mul_363);  mul_363 = None
        sum_139: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_369, [0, 1]);  mul_369 = None
        sum_140: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_876, [0, 1]);  convert_element_type_876 = None
        add_220: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_217, mul_368);  add_217 = mul_368 = None
        convert_element_type_879: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_220, torch.float16)
        clone_166: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_879, memory_format = torch.contiguous_format);  convert_element_type_879 = None
        view_570: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_166, [2464, 512]);  clone_166 = None
        mm_104: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(view_570, permute_480);  permute_480 = None
        permute_481: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_570, [1, 0])
        mm_105: "f16[512, 2048][2048, 1]cuda:0" = torch.ops.aten.mm.default(permute_481, view_194);  permute_481 = view_194 = None
        sum_141: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
        view_571: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_141, [512]);  sum_141 = None
        view_572: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(mm_104, [77, 32, 2048]);  mm_104 = None
        convert_element_type_884: "f32[512, 2048][2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_105, torch.float32);  mm_105 = None
        convert_element_type_885: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_571, torch.float32);  view_571 = None
        view_193: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [77, 32, 2048]);  addmm_38 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_80: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_193, 1.702)
        sigmoid_12: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_80);  mul_80 = None
        mul_370: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_572, sigmoid_12)
        mul_371: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_572, view_193);  view_572 = view_193 = None
        convert_element_type_886: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_371, torch.float32);  mul_371 = None
        convert_element_type_887: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_12, torch.float32);  sigmoid_12 = None
        sub_131: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_887)
        mul_372: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_887, sub_131);  convert_element_type_887 = sub_131 = None
        mul_373: "f32[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_886, mul_372);  convert_element_type_886 = mul_372 = None
        convert_element_type_888: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_373, torch.float16);  mul_373 = None
        mul_374: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_888, 1.702);  convert_element_type_888 = None
        add_221: "f16[77, 32, 2048][65536, 2048, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_370, mul_374);  mul_370 = mul_374 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        view_573: "f16[2464, 2048][2048, 1]cuda:0" = torch.ops.aten.reshape.default(add_221, [2464, 2048]);  add_221 = None
        mm_106: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_573, permute_484);  permute_484 = None
        permute_485: "f16[2048, 2464][1, 2048]cuda:0" = torch.ops.aten.permute.default(view_573, [1, 0])
        mm_107: "f16[2048, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_485, view_192);  permute_485 = view_192 = None
        sum_142: "f16[1, 2048][2048, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
        view_574: "f16[2048][1]cuda:0" = torch.ops.aten.reshape.default(sum_142, [2048]);  sum_142 = None
        view_575: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_106, [77, 32, 512]);  mm_106 = None
        convert_element_type_893: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_575, torch.float32);  view_575 = None
        convert_element_type_894: "f32[2048, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_107, torch.float32);  mm_107 = None
        convert_element_type_895: "f32[2048][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_574, torch.float32);  view_574 = None
        mul_376: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, primals_164);  primals_164 = None
        mul_377: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, 512)
        sum_143: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_376, [2], True)
        mul_378: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_376, mul_375);  mul_376 = None
        sum_144: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_378, [2], True);  mul_378 = None
        mul_379: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, sum_144);  sum_144 = None
        sub_133: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_377, sum_143);  mul_377 = sum_143 = None
        sub_134: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_133, mul_379);  sub_133 = mul_379 = None
        mul_380: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_33, sub_134);  div_33 = sub_134 = None
        mul_381: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_893, mul_375);  mul_375 = None
        sum_145: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_381, [0, 1]);  mul_381 = None
        sum_146: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_893, [0, 1]);  convert_element_type_893 = None
        add_222: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_220, mul_380);  add_220 = mul_380 = None
        convert_element_type_896: "f16[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_222, torch.float16)
        clone_167: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.clone.default(convert_element_type_896, memory_format = torch.contiguous_format);  convert_element_type_896 = None
        view_576: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(clone_167, [2464, 512]);  clone_167 = None
        mm_108: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_576, permute_488);  permute_488 = None
        permute_489: "f16[512, 2464][1, 512]cuda:0" = torch.ops.aten.permute.default(view_576, [1, 0])
        permute_139: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_162, [2, 0, 1, 3])
        view_190: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_139, [2464, 512]);  permute_139 = None
        mm_109: "f16[512, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_489, view_190);  permute_489 = view_190 = None
        sum_147: "f16[1, 512][512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_576, [0], True);  view_576 = None
        view_577: "f16[512][1]cuda:0" = torch.ops.aten.reshape.default(sum_147, [512]);  sum_147 = None
        convert_element_type_901: "f32[512, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_109, torch.float32);  mm_109 = None
        convert_element_type_902: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_577, torch.float32);  view_577 = None
        view_578: "f16[77, 32, 8, 64][16384, 512, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_108, [77, 32, 8, 64]);  mm_108 = None
        permute_492: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_578, [1, 2, 0, 3]);  view_578 = None
        _scaled_dot_product_cudnn_attention_backward_11 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_492, view_187, view_188, view_189, getitem_162, getitem_163, getitem_168, getitem_169, None, None, None, 77, 77, 0.0, True);  permute_492 = view_187 = view_188 = view_189 = getitem_162 = getitem_163 = getitem_168 = getitem_169 = None
        getitem_351: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[0]
        getitem_352: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[1]
        getitem_353: "f16[32, 8, 77, 64][512, 64, 16384, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_11[2];  _scaled_dot_product_cudnn_attention_backward_11 = None
        view_579: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_353, [256, 77, 64]);  getitem_353 = None
        view_580: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_352, [256, 77, 64]);  getitem_352 = None
        view_581: "f16[256, 77, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_351, [256, 77, 64]);  getitem_351 = None
        permute_493: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_579, [1, 0, 2]);  view_579 = None
        view_582: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_493, [77, 32, 512]);  permute_493 = None
        permute_494: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_580, [1, 0, 2]);  view_580 = None
        view_583: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_494, [77, 32, 512]);  permute_494 = None
        permute_495: "f16[77, 256, 64][16384, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_581, [1, 0, 2]);  view_581 = None
        view_584: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_495, [77, 32, 512]);  permute_495 = None

        # No stacktrace found for following nodes
        select_scatter_default_33: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_582, 0, 2);  view_582 = None
        select_scatter_default_34: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_583, 0, 1);  view_583 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_223: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_33, select_scatter_default_34);  select_scatter_default_33 = select_scatter_default_34 = None

        # No stacktrace found for following nodes
        select_scatter_default_35: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_5, view_584, 0, 0);  full_default_5 = view_584 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        add_224: "f16[3, 77, 32, 512][1261568, 16384, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_223, select_scatter_default_35);  add_223 = select_scatter_default_35 = None
        unsqueeze_36: "f16[3, 77, 32, 1, 512][1261568, 16384, 512, 512, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_224, 3);  add_224 = None
        permute_496: "f16[1, 77, 32, 3, 512][512, 16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_36, [3, 1, 2, 0, 4]);  unsqueeze_36 = None
        squeeze_35: "f16[77, 32, 3, 512][16384, 512, 1261568, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_496, 0);  permute_496 = None
        clone_168: "f16[77, 32, 3, 512][49152, 1536, 512, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_35, memory_format = torch.contiguous_format);  squeeze_35 = None
        view_585: "f16[77, 32, 1536][49152, 1536, 1]cuda:0" = torch.ops.aten.reshape.default(clone_168, [77, 32, 1536]);  clone_168 = None
        view_586: "f16[2464, 1536][1536, 1]cuda:0" = torch.ops.aten.reshape.default(view_585, [2464, 1536]);  view_585 = None
        mm_110: "f16[2464, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(view_586, permute_497);  permute_497 = None
        permute_498: "f16[1536, 2464][1, 1536]cuda:0" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_111: "f16[1536, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_498, view_181);  permute_498 = view_181 = None
        sum_148: "f16[1, 1536][1536, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
        view_587: "f16[1536][1]cuda:0" = torch.ops.aten.reshape.default(sum_148, [1536]);  sum_148 = None
        view_588: "f16[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.reshape.default(mm_110, [77, 32, 512]);  mm_110 = None
        convert_element_type_907: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_588, torch.float32);  view_588 = None
        convert_element_type_908: "f32[1536, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_111, torch.float32);  mm_111 = None
        convert_element_type_909: "f32[1536][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_587, torch.float32);  view_587 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:119 in forward, code: embeddings = embeddings + self.positional_embedding
        add_89: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(embedding, primals_156);  embedding = primals_156 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:120 in forward, code: embeddings = embeddings.permute(1, 0, 2)
        permute_133: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.permute.default(add_89, [1, 0, 2]);  add_89 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:121 in forward, code: embeddings = self.encoder(embeddings, mask=self.mask, is_causal=True)
        sub_135: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.sub.Tensor(permute_133, getitem_161);  permute_133 = getitem_161 = None
        mul_382: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_135, rsqrt_26);  sub_135 = None
        mul_383: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_907, primals_158);  primals_158 = None
        mul_384: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_383, 512)
        sum_149: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
        mul_385: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_383, mul_382);  mul_383 = None
        sum_150: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
        mul_386: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_382, sum_150);  sum_150 = None
        sub_136: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_384, sum_149);  mul_384 = sum_149 = None
        sub_137: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_136, mul_386);  sub_136 = mul_386 = None
        div_34: "f32[77, 32, 1][32, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_26, 512);  rsqrt_26 = None
        mul_387: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_34, sub_137);  div_34 = sub_137 = None
        mul_388: "f32[77, 32, 512][16384, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_907, mul_382);  mul_382 = None
        sum_151: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
        sum_152: "f32[512][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_907, [0, 1]);  convert_element_type_907 = None
        add_225: "f32[77, 32, 512][512, 39424, 1]cuda:0" = torch.ops.aten.add.Tensor(add_222, mul_387);  add_222 = mul_387 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:120 in forward, code: embeddings = embeddings.permute(1, 0, 2)
        permute_501: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.permute.default(add_225, [1, 0, 2]);  add_225 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:119 in forward, code: embeddings = embeddings + self.positional_embedding
        sum_153: "f32[1, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_501, [0], True)
        view_589: "f32[77, 512][512, 1]cuda:0" = torch.ops.aten.reshape.default(sum_153, [77, 512]);  sum_153 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/text_encoder.py:118 in forward, code: embeddings = self.token_embedding(text)
        eq_2: "b8[32, 77][77, 1]cuda:0" = torch.ops.aten.eq.Scalar(primals_154, -1)
        unsqueeze_37: "b8[32, 77, 1][77, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_4: "f32[32, 77, 512][39424, 512, 1]cuda:0" = torch.ops.aten.where.self(unsqueeze_37, full_default, permute_501);  unsqueeze_37 = full_default = permute_501 = None
        full_default_42: "f32[49408, 512][512, 1]cuda:0" = torch.ops.aten.full.default([49408, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[49408, 512][512, 1]cuda:0" = torch.ops.aten.index_put_.default(full_default_42, [primals_154], where_4, True);  full_default_42 = primals_154 = where_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:112 in forward, code: x = x @ self.projection
        mm_112: "f16[768, 512][512, 1]cuda:0" = torch.ops.aten.mm.default(permute_502, add_165);  permute_502 = None
        mm_113: "f16[32, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(add_165, permute_503);  add_165 = permute_503 = None
        convert_element_type_914: "f32[32, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_113, torch.float32);  mm_113 = None
        convert_element_type_915: "f32[768, 512][512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_112, torch.float32);  mm_112 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        mul_390: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_914, primals_151);  primals_151 = None
        mul_391: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, 768)
        sum_154: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_390, [1], True)
        mul_392: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, mul_389);  mul_390 = None
        sum_155: "f32[32, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_392, [1], True);  mul_392 = None
        mul_393: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_389, sum_155);  sum_155 = None
        sub_139: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_391, sum_154);  mul_391 = sum_154 = None
        sub_140: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_139, mul_393);  sub_139 = mul_393 = None
        mul_394: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_35, sub_140);  div_35 = sub_140 = None
        mul_395: "f32[32, 768][768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_914, mul_389);  mul_389 = None
        sum_156: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_395, [0]);  mul_395 = None
        sum_157: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_914, [0]);  convert_element_type_914 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:111 in forward, code: x = self.ln_post(x[:, 0, :])
        full_default_43: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.full.default([32, 50, 768], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        select_scatter_default_36: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_43, mul_394, 1, 0);  full_default_43 = mul_394 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        convert_element_type_916: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(select_scatter_default_36, torch.float16)
        view_590: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_916, [1600, 768]);  convert_element_type_916 = None
        mm_114: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_590, permute_504);  permute_504 = None
        permute_505: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_115: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_505, view_179);  permute_505 = view_179 = None
        sum_158: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_158, [768]);  sum_158 = None
        view_592: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_114, [32, 50, 3072]);  mm_114 = None
        convert_element_type_921: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_115, torch.float32);  mm_115 = None
        convert_element_type_922: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_591, torch.float32);  view_591 = None
        view_178: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [32, 50, 3072]);  addmm_34 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_72: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_178, 1.702)
        sigmoid_11: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_72);  mul_72 = None
        mul_396: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_592, sigmoid_11)
        mul_397: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_592, view_178);  view_592 = view_178 = None
        convert_element_type_923: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_397, torch.float32);  mul_397 = None
        convert_element_type_924: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_11, torch.float32);  sigmoid_11 = None
        sub_141: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_924)
        mul_398: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_924, sub_141);  convert_element_type_924 = sub_141 = None
        mul_399: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_923, mul_398);  convert_element_type_923 = mul_398 = None
        convert_element_type_925: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_399, torch.float16);  mul_399 = None
        mul_400: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_925, 1.702);  convert_element_type_925 = None
        add_226: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_396, mul_400);  mul_396 = mul_400 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_593: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_226, [1600, 3072]);  add_226 = None
        mm_116: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_593, permute_508);  permute_508 = None
        permute_509: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_593, [1, 0])
        mm_117: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_509, view_177);  permute_509 = view_177 = None
        sum_159: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_593, [0], True);  view_593 = None
        view_594: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_159, [3072]);  sum_159 = None
        view_595: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_116, [32, 50, 768]);  mm_116 = None
        convert_element_type_930: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_595, torch.float32);  view_595 = None
        convert_element_type_931: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_117, torch.float32);  mm_117 = None
        convert_element_type_932: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_594, torch.float32);  view_594 = None
        mul_402: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_930, primals_145);  primals_145 = None
        mul_403: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, 768)
        sum_160: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_402, [2], True)
        mul_404: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_402, mul_70);  mul_402 = None
        sum_161: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_404, [2], True);  mul_404 = None
        mul_405: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_70, sum_161);  sum_161 = None
        sub_143: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_403, sum_160);  mul_403 = sum_160 = None
        sub_144: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_143, mul_405);  sub_143 = mul_405 = None
        mul_406: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_36, sub_144);  div_36 = sub_144 = None
        mul_407: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_930, mul_70);  mul_70 = None
        sum_162: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_407, [0, 1]);  mul_407 = None
        sum_163: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_930, [0, 1]);  convert_element_type_930 = None
        add_227: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_36, mul_406);  select_scatter_default_36 = mul_406 = None
        convert_element_type_933: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_227, torch.float16)
        permute_512: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_933, [1, 0, 2]);  convert_element_type_933 = None
        clone_169: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_512, memory_format = torch.contiguous_format);  permute_512 = None
        view_596: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_169, [1600, 768]);  clone_169 = None
        mm_118: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_596, permute_513);  permute_513 = None
        permute_514: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_596, [1, 0])
        permute_128: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_147, [2, 0, 1, 3])
        view_175: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_128, [1600, 768]);  permute_128 = None
        mm_119: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_514, view_175);  permute_514 = view_175 = None
        sum_164: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_596, [0], True);  view_596 = None
        view_597: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_164, [768]);  sum_164 = None
        convert_element_type_938: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_119, torch.float32);  mm_119 = None
        convert_element_type_939: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_597, torch.float32);  view_597 = None
        view_598: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_118, [50, 32, 12, 64]);  mm_118 = None
        permute_517: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_598, [1, 2, 0, 3]);  view_598 = None
        _scaled_dot_product_cudnn_attention_backward_12 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_517, view_172, view_173, view_174, getitem_147, getitem_148, getitem_153, getitem_154, None, None, None, 50, 50, 0.0, False);  permute_517 = view_172 = view_173 = view_174 = getitem_147 = getitem_148 = getitem_153 = getitem_154 = None
        getitem_354: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_12[0]
        getitem_355: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_12[1]
        getitem_356: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_12[2];  _scaled_dot_product_cudnn_attention_backward_12 = None
        view_599: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_356, [384, 50, 64]);  getitem_356 = None
        view_600: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_355, [384, 50, 64]);  getitem_355 = None
        view_601: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_354, [384, 50, 64]);  getitem_354 = None
        permute_518: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_599, [1, 0, 2]);  view_599 = None
        view_602: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_518, [50, 32, 768]);  permute_518 = None
        permute_519: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_600, [1, 0, 2]);  view_600 = None
        view_603: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_519, [50, 32, 768]);  permute_519 = None
        permute_520: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_601, [1, 0, 2]);  view_601 = None
        view_604: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_520, [50, 32, 768]);  permute_520 = None
        full_default_44: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.full.default([3, 50, 32, 768], 0, dtype = torch.float16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # No stacktrace found for following nodes
        select_scatter_default_37: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_602, 0, 2);  view_602 = None
        select_scatter_default_38: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_603, 0, 1);  view_603 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_228: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_37, select_scatter_default_38);  select_scatter_default_37 = select_scatter_default_38 = None

        # No stacktrace found for following nodes
        select_scatter_default_39: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_604, 0, 0);  view_604 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_229: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_228, select_scatter_default_39);  add_228 = select_scatter_default_39 = None
        unsqueeze_38: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_229, 3);  add_229 = None
        permute_521: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_38, [3, 1, 2, 0, 4]);  unsqueeze_38 = None
        squeeze_36: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_521, 0);  permute_521 = None
        clone_170: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_36, memory_format = torch.contiguous_format);  squeeze_36 = None
        view_605: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_170, [50, 32, 2304]);  clone_170 = None
        sum_165: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_605, [0, 1], True)
        view_606: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_165, [2304]);  sum_165 = None
        view_607: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_605, [1600, 2304]);  view_605 = None
        permute_522: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_607, [1, 0])
        mm_120: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_522, view_166);  permute_522 = view_166 = None
        mm_121: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_607, permute_524);  view_607 = permute_524 = None
        view_608: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_121, [50, 32, 768]);  mm_121 = None
        convert_element_type_944: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_608, torch.float32);  view_608 = None
        convert_element_type_945: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_120, torch.float32);  mm_120 = None
        convert_element_type_946: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_606, torch.float32);  view_606 = None
        permute_526: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_944, [1, 0, 2]);  convert_element_type_944 = None
        mul_409: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_526, primals_139);  primals_139 = None
        mul_410: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, 768)
        sum_166: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_409, [2], True)
        mul_411: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_409, mul_68);  mul_409 = None
        sum_167: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True);  mul_411 = None
        mul_412: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, sum_167);  sum_167 = None
        sub_146: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_410, sum_166);  mul_410 = sum_166 = None
        sub_147: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_146, mul_412);  sub_146 = mul_412 = None
        mul_413: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_37, sub_147);  div_37 = sub_147 = None
        mul_414: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_526, mul_68);  mul_68 = None
        sum_168: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_414, [0, 1]);  mul_414 = None
        sum_169: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_526, [0, 1]);  permute_526 = None
        add_230: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_227, mul_413);  add_227 = mul_413 = None
        convert_element_type_947: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_230, torch.float16)
        view_609: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_947, [1600, 768]);  convert_element_type_947 = None
        mm_122: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_609, permute_527);  permute_527 = None
        permute_528: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_609, [1, 0])
        mm_123: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_528, view_164);  permute_528 = view_164 = None
        sum_170: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_609, [0], True);  view_609 = None
        view_610: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_170, [768]);  sum_170 = None
        view_611: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_122, [32, 50, 3072]);  mm_122 = None
        convert_element_type_952: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_123, torch.float32);  mm_123 = None
        convert_element_type_953: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_610, torch.float32);  view_610 = None
        view_163: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [32, 50, 3072]);  addmm_31 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_66: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_163, 1.702)
        sigmoid_10: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_66);  mul_66 = None
        mul_415: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_611, sigmoid_10)
        mul_416: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_611, view_163);  view_611 = view_163 = None
        convert_element_type_954: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_416, torch.float32);  mul_416 = None
        convert_element_type_955: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_10, torch.float32);  sigmoid_10 = None
        sub_148: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_955)
        mul_417: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_955, sub_148);  convert_element_type_955 = sub_148 = None
        mul_418: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_954, mul_417);  convert_element_type_954 = mul_417 = None
        convert_element_type_956: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_418, torch.float16);  mul_418 = None
        mul_419: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_956, 1.702);  convert_element_type_956 = None
        add_231: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_415, mul_419);  mul_415 = mul_419 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_612: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_231, [1600, 3072]);  add_231 = None
        mm_124: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_612, permute_531);  permute_531 = None
        permute_532: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_125: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_532, view_162);  permute_532 = view_162 = None
        sum_171: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        view_613: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_171, [3072]);  sum_171 = None
        view_614: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_124, [32, 50, 768]);  mm_124 = None
        convert_element_type_961: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_614, torch.float32);  view_614 = None
        convert_element_type_962: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_125, torch.float32);  mm_125 = None
        convert_element_type_963: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_613, torch.float32);  view_613 = None
        mul_421: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, primals_133);  primals_133 = None
        mul_422: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, 768)
        sum_172: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_421, [2], True)
        mul_423: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_421, mul_64);  mul_421 = None
        sum_173: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True);  mul_423 = None
        mul_424: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, sum_173);  sum_173 = None
        sub_150: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_422, sum_172);  mul_422 = sum_172 = None
        sub_151: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_150, mul_424);  sub_150 = mul_424 = None
        mul_425: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_38, sub_151);  div_38 = sub_151 = None
        mul_426: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_961, mul_64);  mul_64 = None
        sum_174: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_426, [0, 1]);  mul_426 = None
        sum_175: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_961, [0, 1]);  convert_element_type_961 = None
        add_232: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_230, mul_425);  add_230 = mul_425 = None
        convert_element_type_964: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_232, torch.float16)
        permute_535: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_964, [1, 0, 2]);  convert_element_type_964 = None
        clone_171: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
        view_615: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_171, [1600, 768]);  clone_171 = None
        mm_126: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_615, permute_536);  permute_536 = None
        permute_537: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_615, [1, 0])
        permute_117: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_134, [2, 0, 1, 3])
        view_160: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_117, [1600, 768]);  permute_117 = None
        mm_127: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_537, view_160);  permute_537 = view_160 = None
        sum_176: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        view_616: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_176, [768]);  sum_176 = None
        convert_element_type_969: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_127, torch.float32);  mm_127 = None
        convert_element_type_970: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_616, torch.float32);  view_616 = None
        view_617: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_126, [50, 32, 12, 64]);  mm_126 = None
        permute_540: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_617, [1, 2, 0, 3]);  view_617 = None
        _scaled_dot_product_cudnn_attention_backward_13 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_540, view_157, view_158, view_159, getitem_134, getitem_135, getitem_140, getitem_141, None, None, None, 50, 50, 0.0, False);  permute_540 = view_157 = view_158 = view_159 = getitem_134 = getitem_135 = getitem_140 = getitem_141 = None
        getitem_357: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_13[0]
        getitem_358: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_13[1]
        getitem_359: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_13[2];  _scaled_dot_product_cudnn_attention_backward_13 = None
        view_618: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_359, [384, 50, 64]);  getitem_359 = None
        view_619: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_358, [384, 50, 64]);  getitem_358 = None
        view_620: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_357, [384, 50, 64]);  getitem_357 = None
        permute_541: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_618, [1, 0, 2]);  view_618 = None
        view_621: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_541, [50, 32, 768]);  permute_541 = None
        permute_542: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_619, [1, 0, 2]);  view_619 = None
        view_622: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_542, [50, 32, 768]);  permute_542 = None
        permute_543: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_620, [1, 0, 2]);  view_620 = None
        view_623: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_543, [50, 32, 768]);  permute_543 = None

        # No stacktrace found for following nodes
        select_scatter_default_40: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_621, 0, 2);  view_621 = None
        select_scatter_default_41: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_622, 0, 1);  view_622 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_233: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_40, select_scatter_default_41);  select_scatter_default_40 = select_scatter_default_41 = None

        # No stacktrace found for following nodes
        select_scatter_default_42: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_623, 0, 0);  view_623 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_234: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_233, select_scatter_default_42);  add_233 = select_scatter_default_42 = None
        unsqueeze_39: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_234, 3);  add_234 = None
        permute_544: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_39, [3, 1, 2, 0, 4]);  unsqueeze_39 = None
        squeeze_37: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_544, 0);  permute_544 = None
        clone_172: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_37, memory_format = torch.contiguous_format);  squeeze_37 = None
        view_624: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_172, [50, 32, 2304]);  clone_172 = None
        sum_177: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_624, [0, 1], True)
        view_625: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_177, [2304]);  sum_177 = None
        view_626: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_624, [1600, 2304]);  view_624 = None
        permute_545: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_626, [1, 0])
        mm_128: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_545, view_151);  permute_545 = view_151 = None
        mm_129: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_626, permute_547);  view_626 = permute_547 = None
        view_627: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_129, [50, 32, 768]);  mm_129 = None
        convert_element_type_975: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_627, torch.float32);  view_627 = None
        convert_element_type_976: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_128, torch.float32);  mm_128 = None
        convert_element_type_977: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_625, torch.float32);  view_625 = None
        permute_549: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_975, [1, 0, 2]);  convert_element_type_975 = None
        mul_428: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_549, primals_127);  primals_127 = None
        mul_429: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_428, 768)
        sum_178: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_428, [2], True)
        mul_430: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_428, mul_62);  mul_428 = None
        sum_179: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_430, [2], True);  mul_430 = None
        mul_431: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_62, sum_179);  sum_179 = None
        sub_153: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_429, sum_178);  mul_429 = sum_178 = None
        sub_154: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_153, mul_431);  sub_153 = mul_431 = None
        mul_432: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_39, sub_154);  div_39 = sub_154 = None
        mul_433: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_549, mul_62);  mul_62 = None
        sum_180: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_433, [0, 1]);  mul_433 = None
        sum_181: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_549, [0, 1]);  permute_549 = None
        add_235: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_232, mul_432);  add_232 = mul_432 = None
        convert_element_type_978: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_235, torch.float16)
        view_628: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_978, [1600, 768]);  convert_element_type_978 = None
        mm_130: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_628, permute_550);  permute_550 = None
        permute_551: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_131: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_551, view_149);  permute_551 = view_149 = None
        sum_182: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        view_629: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_182, [768]);  sum_182 = None
        view_630: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_130, [32, 50, 3072]);  mm_130 = None
        convert_element_type_983: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_131, torch.float32);  mm_131 = None
        convert_element_type_984: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_629, torch.float32);  view_629 = None
        view_148: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [32, 50, 3072]);  addmm_28 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_60: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_148, 1.702)
        sigmoid_9: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_60);  mul_60 = None
        mul_434: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_630, sigmoid_9)
        mul_435: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_630, view_148);  view_630 = view_148 = None
        convert_element_type_985: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_435, torch.float32);  mul_435 = None
        convert_element_type_986: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_9, torch.float32);  sigmoid_9 = None
        sub_155: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_986)
        mul_436: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_986, sub_155);  convert_element_type_986 = sub_155 = None
        mul_437: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_985, mul_436);  convert_element_type_985 = mul_436 = None
        convert_element_type_987: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_437, torch.float16);  mul_437 = None
        mul_438: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_987, 1.702);  convert_element_type_987 = None
        add_236: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_434, mul_438);  mul_434 = mul_438 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_631: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_236, [1600, 3072]);  add_236 = None
        mm_132: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_631, permute_554);  permute_554 = None
        permute_555: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_133: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_555, view_147);  permute_555 = view_147 = None
        sum_183: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        view_632: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_183, [3072]);  sum_183 = None
        view_633: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_132, [32, 50, 768]);  mm_132 = None
        convert_element_type_992: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_633, torch.float32);  view_633 = None
        convert_element_type_993: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_133, torch.float32);  mm_133 = None
        convert_element_type_994: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_632, torch.float32);  view_632 = None
        mul_440: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_992, primals_121);  primals_121 = None
        mul_441: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, 768)
        sum_184: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_440, [2], True)
        mul_442: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_440, mul_58);  mul_440 = None
        sum_185: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_442, [2], True);  mul_442 = None
        mul_443: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_58, sum_185);  sum_185 = None
        sub_157: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_441, sum_184);  mul_441 = sum_184 = None
        sub_158: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_157, mul_443);  sub_157 = mul_443 = None
        mul_444: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_40, sub_158);  div_40 = sub_158 = None
        mul_445: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_992, mul_58);  mul_58 = None
        sum_186: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_445, [0, 1]);  mul_445 = None
        sum_187: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_992, [0, 1]);  convert_element_type_992 = None
        add_237: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_235, mul_444);  add_235 = mul_444 = None
        convert_element_type_995: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.float16)
        permute_558: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_995, [1, 0, 2]);  convert_element_type_995 = None
        clone_173: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_558, memory_format = torch.contiguous_format);  permute_558 = None
        view_634: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_173, [1600, 768]);  clone_173 = None
        mm_134: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_634, permute_559);  permute_559 = None
        permute_560: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_634, [1, 0])
        permute_106: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_121, [2, 0, 1, 3])
        view_145: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [1600, 768]);  permute_106 = None
        mm_135: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_560, view_145);  permute_560 = view_145 = None
        sum_188: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_634, [0], True);  view_634 = None
        view_635: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_188, [768]);  sum_188 = None
        convert_element_type_1000: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_135, torch.float32);  mm_135 = None
        convert_element_type_1001: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_635, torch.float32);  view_635 = None
        view_636: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_134, [50, 32, 12, 64]);  mm_134 = None
        permute_563: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_636, [1, 2, 0, 3]);  view_636 = None
        _scaled_dot_product_cudnn_attention_backward_14 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_563, view_142, view_143, view_144, getitem_121, getitem_122, getitem_127, getitem_128, None, None, None, 50, 50, 0.0, False);  permute_563 = view_142 = view_143 = view_144 = getitem_121 = getitem_122 = getitem_127 = getitem_128 = None
        getitem_360: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_14[0]
        getitem_361: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_14[1]
        getitem_362: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_14[2];  _scaled_dot_product_cudnn_attention_backward_14 = None
        view_637: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_362, [384, 50, 64]);  getitem_362 = None
        view_638: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_361, [384, 50, 64]);  getitem_361 = None
        view_639: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_360, [384, 50, 64]);  getitem_360 = None
        permute_564: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_637, [1, 0, 2]);  view_637 = None
        view_640: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_564, [50, 32, 768]);  permute_564 = None
        permute_565: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_638, [1, 0, 2]);  view_638 = None
        view_641: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_565, [50, 32, 768]);  permute_565 = None
        permute_566: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_639, [1, 0, 2]);  view_639 = None
        view_642: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_566, [50, 32, 768]);  permute_566 = None

        # No stacktrace found for following nodes
        select_scatter_default_43: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_640, 0, 2);  view_640 = None
        select_scatter_default_44: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_641, 0, 1);  view_641 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_238: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_43, select_scatter_default_44);  select_scatter_default_43 = select_scatter_default_44 = None

        # No stacktrace found for following nodes
        select_scatter_default_45: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_642, 0, 0);  view_642 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_239: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_238, select_scatter_default_45);  add_238 = select_scatter_default_45 = None
        unsqueeze_40: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_239, 3);  add_239 = None
        permute_567: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_40, [3, 1, 2, 0, 4]);  unsqueeze_40 = None
        squeeze_38: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_567, 0);  permute_567 = None
        clone_174: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_38, memory_format = torch.contiguous_format);  squeeze_38 = None
        view_643: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_174, [50, 32, 2304]);  clone_174 = None
        sum_189: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_643, [0, 1], True)
        view_644: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_189, [2304]);  sum_189 = None
        view_645: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_643, [1600, 2304]);  view_643 = None
        permute_568: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_645, [1, 0])
        mm_136: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_568, view_136);  permute_568 = view_136 = None
        mm_137: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_645, permute_570);  view_645 = permute_570 = None
        view_646: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_137, [50, 32, 768]);  mm_137 = None
        convert_element_type_1006: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_646, torch.float32);  view_646 = None
        convert_element_type_1007: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_136, torch.float32);  mm_136 = None
        convert_element_type_1008: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_644, torch.float32);  view_644 = None
        permute_572: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1006, [1, 0, 2]);  convert_element_type_1006 = None
        mul_447: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_572, primals_115);  primals_115 = None
        mul_448: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, 768)
        sum_190: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_447, [2], True)
        mul_449: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_447, mul_56);  mul_447 = None
        sum_191: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_449, [2], True);  mul_449 = None
        mul_450: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_56, sum_191);  sum_191 = None
        sub_160: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_448, sum_190);  mul_448 = sum_190 = None
        sub_161: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_160, mul_450);  sub_160 = mul_450 = None
        mul_451: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_41, sub_161);  div_41 = sub_161 = None
        mul_452: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_572, mul_56);  mul_56 = None
        sum_192: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_452, [0, 1]);  mul_452 = None
        sum_193: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_572, [0, 1]);  permute_572 = None
        add_240: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_237, mul_451);  add_237 = mul_451 = None
        convert_element_type_1009: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_240, torch.float16)
        view_647: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1009, [1600, 768]);  convert_element_type_1009 = None
        mm_138: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_647, permute_573);  permute_573 = None
        permute_574: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_647, [1, 0])
        mm_139: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_574, view_134);  permute_574 = view_134 = None
        sum_194: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_647, [0], True);  view_647 = None
        view_648: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_194, [768]);  sum_194 = None
        view_649: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_138, [32, 50, 3072]);  mm_138 = None
        convert_element_type_1014: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_139, torch.float32);  mm_139 = None
        convert_element_type_1015: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_648, torch.float32);  view_648 = None
        view_133: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [32, 50, 3072]);  addmm_25 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_54: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_133, 1.702)
        sigmoid_8: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_54);  mul_54 = None
        mul_453: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_649, sigmoid_8)
        mul_454: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_649, view_133);  view_649 = view_133 = None
        convert_element_type_1016: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_454, torch.float32);  mul_454 = None
        convert_element_type_1017: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_8, torch.float32);  sigmoid_8 = None
        sub_162: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1017)
        mul_455: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1017, sub_162);  convert_element_type_1017 = sub_162 = None
        mul_456: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1016, mul_455);  convert_element_type_1016 = mul_455 = None
        convert_element_type_1018: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_456, torch.float16);  mul_456 = None
        mul_457: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1018, 1.702);  convert_element_type_1018 = None
        add_241: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_453, mul_457);  mul_453 = mul_457 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_650: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_241, [1600, 3072]);  add_241 = None
        mm_140: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_650, permute_577);  permute_577 = None
        permute_578: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_141: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_578, view_132);  permute_578 = view_132 = None
        sum_195: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        view_651: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_195, [3072]);  sum_195 = None
        view_652: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_140, [32, 50, 768]);  mm_140 = None
        convert_element_type_1023: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        convert_element_type_1024: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_141, torch.float32);  mm_141 = None
        convert_element_type_1025: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_651, torch.float32);  view_651 = None
        mul_459: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, primals_109);  primals_109 = None
        mul_460: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, 768)
        sum_196: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_459, [2], True)
        mul_461: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_459, mul_52);  mul_459 = None
        sum_197: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_461, [2], True);  mul_461 = None
        mul_462: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, sum_197);  sum_197 = None
        sub_164: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_460, sum_196);  mul_460 = sum_196 = None
        sub_165: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_164, mul_462);  sub_164 = mul_462 = None
        mul_463: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_42, sub_165);  div_42 = sub_165 = None
        mul_464: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1023, mul_52);  mul_52 = None
        sum_198: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_464, [0, 1]);  mul_464 = None
        sum_199: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1023, [0, 1]);  convert_element_type_1023 = None
        add_242: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_240, mul_463);  add_240 = mul_463 = None
        convert_element_type_1026: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.float16)
        permute_581: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1026, [1, 0, 2]);  convert_element_type_1026 = None
        clone_175: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_581, memory_format = torch.contiguous_format);  permute_581 = None
        view_653: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_175, [1600, 768]);  clone_175 = None
        mm_142: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_653, permute_582);  permute_582 = None
        permute_583: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_653, [1, 0])
        permute_95: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_108, [2, 0, 1, 3])
        view_130: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_95, [1600, 768]);  permute_95 = None
        mm_143: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_583, view_130);  permute_583 = view_130 = None
        sum_200: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        view_654: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_200, [768]);  sum_200 = None
        convert_element_type_1031: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_143, torch.float32);  mm_143 = None
        convert_element_type_1032: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_654, torch.float32);  view_654 = None
        view_655: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_142, [50, 32, 12, 64]);  mm_142 = None
        permute_586: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_655, [1, 2, 0, 3]);  view_655 = None
        _scaled_dot_product_cudnn_attention_backward_15 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_586, view_127, view_128, view_129, getitem_108, getitem_109, getitem_114, getitem_115, None, None, None, 50, 50, 0.0, False);  permute_586 = view_127 = view_128 = view_129 = getitem_108 = getitem_109 = getitem_114 = getitem_115 = None
        getitem_363: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_15[0]
        getitem_364: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_15[1]
        getitem_365: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_15[2];  _scaled_dot_product_cudnn_attention_backward_15 = None
        view_656: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_365, [384, 50, 64]);  getitem_365 = None
        view_657: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_364, [384, 50, 64]);  getitem_364 = None
        view_658: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_363, [384, 50, 64]);  getitem_363 = None
        permute_587: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_656, [1, 0, 2]);  view_656 = None
        view_659: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_587, [50, 32, 768]);  permute_587 = None
        permute_588: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_657, [1, 0, 2]);  view_657 = None
        view_660: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_588, [50, 32, 768]);  permute_588 = None
        permute_589: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_658, [1, 0, 2]);  view_658 = None
        view_661: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_589, [50, 32, 768]);  permute_589 = None

        # No stacktrace found for following nodes
        select_scatter_default_46: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_659, 0, 2);  view_659 = None
        select_scatter_default_47: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_660, 0, 1);  view_660 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_243: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_46, select_scatter_default_47);  select_scatter_default_46 = select_scatter_default_47 = None

        # No stacktrace found for following nodes
        select_scatter_default_48: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_661, 0, 0);  view_661 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_244: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_243, select_scatter_default_48);  add_243 = select_scatter_default_48 = None
        unsqueeze_41: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_244, 3);  add_244 = None
        permute_590: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_41, [3, 1, 2, 0, 4]);  unsqueeze_41 = None
        squeeze_39: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_590, 0);  permute_590 = None
        clone_176: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_39, memory_format = torch.contiguous_format);  squeeze_39 = None
        view_662: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_176, [50, 32, 2304]);  clone_176 = None
        sum_201: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_662, [0, 1], True)
        view_663: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_201, [2304]);  sum_201 = None
        view_664: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_662, [1600, 2304]);  view_662 = None
        permute_591: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_664, [1, 0])
        mm_144: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_591, view_121);  permute_591 = view_121 = None
        mm_145: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_664, permute_593);  view_664 = permute_593 = None
        view_665: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_145, [50, 32, 768]);  mm_145 = None
        convert_element_type_1037: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_665, torch.float32);  view_665 = None
        convert_element_type_1038: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_144, torch.float32);  mm_144 = None
        convert_element_type_1039: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_663, torch.float32);  view_663 = None
        permute_595: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1037, [1, 0, 2]);  convert_element_type_1037 = None
        mul_466: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_595, primals_103);  primals_103 = None
        mul_467: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, 768)
        sum_202: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True)
        mul_468: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_466, mul_50);  mul_466 = None
        sum_203: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_468, [2], True);  mul_468 = None
        mul_469: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, sum_203);  sum_203 = None
        sub_167: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_467, sum_202);  mul_467 = sum_202 = None
        sub_168: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_167, mul_469);  sub_167 = mul_469 = None
        mul_470: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_43, sub_168);  div_43 = sub_168 = None
        mul_471: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_595, mul_50);  mul_50 = None
        sum_204: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_471, [0, 1]);  mul_471 = None
        sum_205: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_595, [0, 1]);  permute_595 = None
        add_245: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_242, mul_470);  add_242 = mul_470 = None
        convert_element_type_1040: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_245, torch.float16)
        view_666: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1040, [1600, 768]);  convert_element_type_1040 = None
        mm_146: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_666, permute_596);  permute_596 = None
        permute_597: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_666, [1, 0])
        mm_147: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_597, view_119);  permute_597 = view_119 = None
        sum_206: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_666, [0], True);  view_666 = None
        view_667: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_206, [768]);  sum_206 = None
        view_668: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_146, [32, 50, 3072]);  mm_146 = None
        convert_element_type_1045: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_147, torch.float32);  mm_147 = None
        convert_element_type_1046: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_667, torch.float32);  view_667 = None
        view_118: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [32, 50, 3072]);  addmm_22 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_48: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_118, 1.702)
        sigmoid_7: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_48);  mul_48 = None
        mul_472: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_668, sigmoid_7)
        mul_473: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_668, view_118);  view_668 = view_118 = None
        convert_element_type_1047: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_473, torch.float32);  mul_473 = None
        convert_element_type_1048: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_7, torch.float32);  sigmoid_7 = None
        sub_169: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1048)
        mul_474: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1048, sub_169);  convert_element_type_1048 = sub_169 = None
        mul_475: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1047, mul_474);  convert_element_type_1047 = mul_474 = None
        convert_element_type_1049: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_475, torch.float16);  mul_475 = None
        mul_476: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1049, 1.702);  convert_element_type_1049 = None
        add_246: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_472, mul_476);  mul_472 = mul_476 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_669: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_246, [1600, 3072]);  add_246 = None
        mm_148: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_669, permute_600);  permute_600 = None
        permute_601: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_669, [1, 0])
        mm_149: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_601, view_117);  permute_601 = view_117 = None
        sum_207: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_669, [0], True);  view_669 = None
        view_670: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_207, [3072]);  sum_207 = None
        view_671: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_148, [32, 50, 768]);  mm_148 = None
        convert_element_type_1054: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_671, torch.float32);  view_671 = None
        convert_element_type_1055: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_149, torch.float32);  mm_149 = None
        convert_element_type_1056: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_670, torch.float32);  view_670 = None
        mul_478: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1054, primals_97);  primals_97 = None
        mul_479: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, 768)
        sum_208: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_478, [2], True)
        mul_480: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_478, mul_46);  mul_478 = None
        sum_209: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_480, [2], True);  mul_480 = None
        mul_481: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_46, sum_209);  sum_209 = None
        sub_171: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_479, sum_208);  mul_479 = sum_208 = None
        sub_172: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_171, mul_481);  sub_171 = mul_481 = None
        mul_482: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_44, sub_172);  div_44 = sub_172 = None
        mul_483: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1054, mul_46);  mul_46 = None
        sum_210: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_483, [0, 1]);  mul_483 = None
        sum_211: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1054, [0, 1]);  convert_element_type_1054 = None
        add_247: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_245, mul_482);  add_245 = mul_482 = None
        convert_element_type_1057: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_247, torch.float16)
        permute_604: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1057, [1, 0, 2]);  convert_element_type_1057 = None
        clone_177: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_604, memory_format = torch.contiguous_format);  permute_604 = None
        view_672: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_177, [1600, 768]);  clone_177 = None
        mm_150: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_672, permute_605);  permute_605 = None
        permute_606: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_672, [1, 0])
        permute_84: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_95, [2, 0, 1, 3])
        view_115: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [1600, 768]);  permute_84 = None
        mm_151: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_606, view_115);  permute_606 = view_115 = None
        sum_212: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_672, [0], True);  view_672 = None
        view_673: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_212, [768]);  sum_212 = None
        convert_element_type_1062: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_151, torch.float32);  mm_151 = None
        convert_element_type_1063: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_673, torch.float32);  view_673 = None
        view_674: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_150, [50, 32, 12, 64]);  mm_150 = None
        permute_609: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_674, [1, 2, 0, 3]);  view_674 = None
        _scaled_dot_product_cudnn_attention_backward_16 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_609, view_112, view_113, view_114, getitem_95, getitem_96, getitem_101, getitem_102, None, None, None, 50, 50, 0.0, False);  permute_609 = view_112 = view_113 = view_114 = getitem_95 = getitem_96 = getitem_101 = getitem_102 = None
        getitem_366: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_16[0]
        getitem_367: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_16[1]
        getitem_368: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_16[2];  _scaled_dot_product_cudnn_attention_backward_16 = None
        view_675: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_368, [384, 50, 64]);  getitem_368 = None
        view_676: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_367, [384, 50, 64]);  getitem_367 = None
        view_677: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_366, [384, 50, 64]);  getitem_366 = None
        permute_610: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_675, [1, 0, 2]);  view_675 = None
        view_678: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_610, [50, 32, 768]);  permute_610 = None
        permute_611: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_676, [1, 0, 2]);  view_676 = None
        view_679: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_611, [50, 32, 768]);  permute_611 = None
        permute_612: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_677, [1, 0, 2]);  view_677 = None
        view_680: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_612, [50, 32, 768]);  permute_612 = None

        # No stacktrace found for following nodes
        select_scatter_default_49: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_678, 0, 2);  view_678 = None
        select_scatter_default_50: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_679, 0, 1);  view_679 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_248: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_49, select_scatter_default_50);  select_scatter_default_49 = select_scatter_default_50 = None

        # No stacktrace found for following nodes
        select_scatter_default_51: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_680, 0, 0);  view_680 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_249: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_248, select_scatter_default_51);  add_248 = select_scatter_default_51 = None
        unsqueeze_42: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_249, 3);  add_249 = None
        permute_613: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_42, [3, 1, 2, 0, 4]);  unsqueeze_42 = None
        squeeze_40: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_613, 0);  permute_613 = None
        clone_178: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_40, memory_format = torch.contiguous_format);  squeeze_40 = None
        view_681: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_178, [50, 32, 2304]);  clone_178 = None
        sum_213: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_681, [0, 1], True)
        view_682: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_213, [2304]);  sum_213 = None
        view_683: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_681, [1600, 2304]);  view_681 = None
        permute_614: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_683, [1, 0])
        mm_152: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_614, view_106);  permute_614 = view_106 = None
        mm_153: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_683, permute_616);  view_683 = permute_616 = None
        view_684: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_153, [50, 32, 768]);  mm_153 = None
        convert_element_type_1068: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_684, torch.float32);  view_684 = None
        convert_element_type_1069: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_152, torch.float32);  mm_152 = None
        convert_element_type_1070: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_682, torch.float32);  view_682 = None
        permute_618: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1068, [1, 0, 2]);  convert_element_type_1068 = None
        mul_485: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_618, primals_91);  primals_91 = None
        mul_486: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, 768)
        sum_214: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_485, [2], True)
        mul_487: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_485, mul_44);  mul_485 = None
        sum_215: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_487, [2], True);  mul_487 = None
        mul_488: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_44, sum_215);  sum_215 = None
        sub_174: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_486, sum_214);  mul_486 = sum_214 = None
        sub_175: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_174, mul_488);  sub_174 = mul_488 = None
        mul_489: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_45, sub_175);  div_45 = sub_175 = None
        mul_490: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_618, mul_44);  mul_44 = None
        sum_216: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_490, [0, 1]);  mul_490 = None
        sum_217: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_618, [0, 1]);  permute_618 = None
        add_250: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_247, mul_489);  add_247 = mul_489 = None
        convert_element_type_1071: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_250, torch.float16)
        view_685: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1071, [1600, 768]);  convert_element_type_1071 = None
        mm_154: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_685, permute_619);  permute_619 = None
        permute_620: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_685, [1, 0])
        mm_155: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_620, view_104);  permute_620 = view_104 = None
        sum_218: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_685, [0], True);  view_685 = None
        view_686: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_218, [768]);  sum_218 = None
        view_687: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_154, [32, 50, 3072]);  mm_154 = None
        convert_element_type_1076: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_155, torch.float32);  mm_155 = None
        convert_element_type_1077: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_686, torch.float32);  view_686 = None
        view_103: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [32, 50, 3072]);  addmm_19 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_42: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_103, 1.702)
        sigmoid_6: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_42);  mul_42 = None
        mul_491: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_687, sigmoid_6)
        mul_492: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_687, view_103);  view_687 = view_103 = None
        convert_element_type_1078: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_492, torch.float32);  mul_492 = None
        convert_element_type_1079: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_6, torch.float32);  sigmoid_6 = None
        sub_176: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1079)
        mul_493: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1079, sub_176);  convert_element_type_1079 = sub_176 = None
        mul_494: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1078, mul_493);  convert_element_type_1078 = mul_493 = None
        convert_element_type_1080: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_494, torch.float16);  mul_494 = None
        mul_495: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1080, 1.702);  convert_element_type_1080 = None
        add_251: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_491, mul_495);  mul_491 = mul_495 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_688: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_251, [1600, 3072]);  add_251 = None
        mm_156: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_688, permute_623);  permute_623 = None
        permute_624: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_688, [1, 0])
        mm_157: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_624, view_102);  permute_624 = view_102 = None
        sum_219: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_688, [0], True);  view_688 = None
        view_689: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_219, [3072]);  sum_219 = None
        view_690: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_156, [32, 50, 768]);  mm_156 = None
        convert_element_type_1085: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_690, torch.float32);  view_690 = None
        convert_element_type_1086: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_157, torch.float32);  mm_157 = None
        convert_element_type_1087: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_689, torch.float32);  view_689 = None
        mul_497: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1085, primals_85);  primals_85 = None
        mul_498: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, 768)
        sum_220: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True)
        mul_499: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_497, mul_40);  mul_497 = None
        sum_221: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_499, [2], True);  mul_499 = None
        mul_500: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_40, sum_221);  sum_221 = None
        sub_178: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_498, sum_220);  mul_498 = sum_220 = None
        sub_179: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_178, mul_500);  sub_178 = mul_500 = None
        mul_501: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_46, sub_179);  div_46 = sub_179 = None
        mul_502: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1085, mul_40);  mul_40 = None
        sum_222: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_502, [0, 1]);  mul_502 = None
        sum_223: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1085, [0, 1]);  convert_element_type_1085 = None
        add_252: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_250, mul_501);  add_250 = mul_501 = None
        convert_element_type_1088: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.float16)
        permute_627: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1088, [1, 0, 2]);  convert_element_type_1088 = None
        clone_179: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_627, memory_format = torch.contiguous_format);  permute_627 = None
        view_691: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_179, [1600, 768]);  clone_179 = None
        mm_158: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_691, permute_628);  permute_628 = None
        permute_629: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_691, [1, 0])
        permute_73: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_82, [2, 0, 1, 3])
        view_100: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_73, [1600, 768]);  permute_73 = None
        mm_159: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_629, view_100);  permute_629 = view_100 = None
        sum_224: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_691, [0], True);  view_691 = None
        view_692: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_224, [768]);  sum_224 = None
        convert_element_type_1093: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_159, torch.float32);  mm_159 = None
        convert_element_type_1094: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None
        view_693: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_158, [50, 32, 12, 64]);  mm_158 = None
        permute_632: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_693, [1, 2, 0, 3]);  view_693 = None
        _scaled_dot_product_cudnn_attention_backward_17 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_632, view_97, view_98, view_99, getitem_82, getitem_83, getitem_88, getitem_89, None, None, None, 50, 50, 0.0, False);  permute_632 = view_97 = view_98 = view_99 = getitem_82 = getitem_83 = getitem_88 = getitem_89 = None
        getitem_369: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_17[0]
        getitem_370: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_17[1]
        getitem_371: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_17[2];  _scaled_dot_product_cudnn_attention_backward_17 = None
        view_694: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_371, [384, 50, 64]);  getitem_371 = None
        view_695: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_370, [384, 50, 64]);  getitem_370 = None
        view_696: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_369, [384, 50, 64]);  getitem_369 = None
        permute_633: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_694, [1, 0, 2]);  view_694 = None
        view_697: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_633, [50, 32, 768]);  permute_633 = None
        permute_634: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_695, [1, 0, 2]);  view_695 = None
        view_698: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_634, [50, 32, 768]);  permute_634 = None
        permute_635: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_696, [1, 0, 2]);  view_696 = None
        view_699: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_635, [50, 32, 768]);  permute_635 = None

        # No stacktrace found for following nodes
        select_scatter_default_52: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_697, 0, 2);  view_697 = None
        select_scatter_default_53: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_698, 0, 1);  view_698 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_253: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_52, select_scatter_default_53);  select_scatter_default_52 = select_scatter_default_53 = None

        # No stacktrace found for following nodes
        select_scatter_default_54: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_699, 0, 0);  view_699 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_254: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_253, select_scatter_default_54);  add_253 = select_scatter_default_54 = None
        unsqueeze_43: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_254, 3);  add_254 = None
        permute_636: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_43, [3, 1, 2, 0, 4]);  unsqueeze_43 = None
        squeeze_41: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_636, 0);  permute_636 = None
        clone_180: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_41, memory_format = torch.contiguous_format);  squeeze_41 = None
        view_700: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_180, [50, 32, 2304]);  clone_180 = None
        sum_225: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_700, [0, 1], True)
        view_701: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_225, [2304]);  sum_225 = None
        view_702: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_700, [1600, 2304]);  view_700 = None
        permute_637: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_702, [1, 0])
        mm_160: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_637, view_91);  permute_637 = view_91 = None
        mm_161: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_702, permute_639);  view_702 = permute_639 = None
        view_703: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_161, [50, 32, 768]);  mm_161 = None
        convert_element_type_1099: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_703, torch.float32);  view_703 = None
        convert_element_type_1100: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_160, torch.float32);  mm_160 = None
        convert_element_type_1101: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_701, torch.float32);  view_701 = None
        permute_641: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1099, [1, 0, 2]);  convert_element_type_1099 = None
        mul_504: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_641, primals_79);  primals_79 = None
        mul_505: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, 768)
        sum_226: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_504, [2], True)
        mul_506: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_504, mul_38);  mul_504 = None
        sum_227: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_506, [2], True);  mul_506 = None
        mul_507: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_38, sum_227);  sum_227 = None
        sub_181: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_505, sum_226);  mul_505 = sum_226 = None
        sub_182: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_181, mul_507);  sub_181 = mul_507 = None
        mul_508: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_47, sub_182);  div_47 = sub_182 = None
        mul_509: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_641, mul_38);  mul_38 = None
        sum_228: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_509, [0, 1]);  mul_509 = None
        sum_229: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_641, [0, 1]);  permute_641 = None
        add_255: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_252, mul_508);  add_252 = mul_508 = None
        convert_element_type_1102: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_255, torch.float16)
        view_704: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1102, [1600, 768]);  convert_element_type_1102 = None
        mm_162: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_704, permute_642);  permute_642 = None
        permute_643: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_704, [1, 0])
        mm_163: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_643, view_89);  permute_643 = view_89 = None
        sum_230: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_704, [0], True);  view_704 = None
        view_705: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_230, [768]);  sum_230 = None
        view_706: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_162, [32, 50, 3072]);  mm_162 = None
        convert_element_type_1107: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_163, torch.float32);  mm_163 = None
        convert_element_type_1108: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_705, torch.float32);  view_705 = None
        view_88: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [32, 50, 3072]);  addmm_16 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_36: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_88, 1.702)
        sigmoid_5: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_36);  mul_36 = None
        mul_510: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_706, sigmoid_5)
        mul_511: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_706, view_88);  view_706 = view_88 = None
        convert_element_type_1109: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_511, torch.float32);  mul_511 = None
        convert_element_type_1110: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_5, torch.float32);  sigmoid_5 = None
        sub_183: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1110)
        mul_512: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1110, sub_183);  convert_element_type_1110 = sub_183 = None
        mul_513: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1109, mul_512);  convert_element_type_1109 = mul_512 = None
        convert_element_type_1111: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_513, torch.float16);  mul_513 = None
        mul_514: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1111, 1.702);  convert_element_type_1111 = None
        add_256: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_510, mul_514);  mul_510 = mul_514 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_707: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_256, [1600, 3072]);  add_256 = None
        mm_164: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_707, permute_646);  permute_646 = None
        permute_647: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_707, [1, 0])
        mm_165: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_647, view_87);  permute_647 = view_87 = None
        sum_231: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_707, [0], True);  view_707 = None
        view_708: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_231, [3072]);  sum_231 = None
        view_709: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_164, [32, 50, 768]);  mm_164 = None
        convert_element_type_1116: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_709, torch.float32);  view_709 = None
        convert_element_type_1117: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_165, torch.float32);  mm_165 = None
        convert_element_type_1118: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_708, torch.float32);  view_708 = None
        mul_516: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1116, primals_73);  primals_73 = None
        mul_517: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, 768)
        sum_232: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_516, [2], True)
        mul_518: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_516, mul_34);  mul_516 = None
        sum_233: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True);  mul_518 = None
        mul_519: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, sum_233);  sum_233 = None
        sub_185: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_517, sum_232);  mul_517 = sum_232 = None
        sub_186: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_185, mul_519);  sub_185 = mul_519 = None
        mul_520: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_48, sub_186);  div_48 = sub_186 = None
        mul_521: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1116, mul_34);  mul_34 = None
        sum_234: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_521, [0, 1]);  mul_521 = None
        sum_235: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1116, [0, 1]);  convert_element_type_1116 = None
        add_257: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_255, mul_520);  add_255 = mul_520 = None
        convert_element_type_1119: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_257, torch.float16)
        permute_650: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1119, [1, 0, 2]);  convert_element_type_1119 = None
        clone_181: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_650, memory_format = torch.contiguous_format);  permute_650 = None
        view_710: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_181, [1600, 768]);  clone_181 = None
        mm_166: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_710, permute_651);  permute_651 = None
        permute_652: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_710, [1, 0])
        permute_62: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_69, [2, 0, 1, 3])
        view_85: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_62, [1600, 768]);  permute_62 = None
        mm_167: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_652, view_85);  permute_652 = view_85 = None
        sum_236: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_710, [0], True);  view_710 = None
        view_711: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_236, [768]);  sum_236 = None
        convert_element_type_1124: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_167, torch.float32);  mm_167 = None
        convert_element_type_1125: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_711, torch.float32);  view_711 = None
        view_712: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_166, [50, 32, 12, 64]);  mm_166 = None
        permute_655: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_712, [1, 2, 0, 3]);  view_712 = None
        _scaled_dot_product_cudnn_attention_backward_18 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_655, view_82, view_83, view_84, getitem_69, getitem_70, getitem_75, getitem_76, None, None, None, 50, 50, 0.0, False);  permute_655 = view_82 = view_83 = view_84 = getitem_69 = getitem_70 = getitem_75 = getitem_76 = None
        getitem_372: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_18[0]
        getitem_373: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_18[1]
        getitem_374: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_18[2];  _scaled_dot_product_cudnn_attention_backward_18 = None
        view_713: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_374, [384, 50, 64]);  getitem_374 = None
        view_714: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_373, [384, 50, 64]);  getitem_373 = None
        view_715: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_372, [384, 50, 64]);  getitem_372 = None
        permute_656: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_713, [1, 0, 2]);  view_713 = None
        view_716: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_656, [50, 32, 768]);  permute_656 = None
        permute_657: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_714, [1, 0, 2]);  view_714 = None
        view_717: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_657, [50, 32, 768]);  permute_657 = None
        permute_658: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_715, [1, 0, 2]);  view_715 = None
        view_718: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_658, [50, 32, 768]);  permute_658 = None

        # No stacktrace found for following nodes
        select_scatter_default_55: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_716, 0, 2);  view_716 = None
        select_scatter_default_56: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_717, 0, 1);  view_717 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_258: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_55, select_scatter_default_56);  select_scatter_default_55 = select_scatter_default_56 = None

        # No stacktrace found for following nodes
        select_scatter_default_57: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_718, 0, 0);  view_718 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_259: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_258, select_scatter_default_57);  add_258 = select_scatter_default_57 = None
        unsqueeze_44: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_259, 3);  add_259 = None
        permute_659: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_44, [3, 1, 2, 0, 4]);  unsqueeze_44 = None
        squeeze_42: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_659, 0);  permute_659 = None
        clone_182: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_42, memory_format = torch.contiguous_format);  squeeze_42 = None
        view_719: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_182, [50, 32, 2304]);  clone_182 = None
        sum_237: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_719, [0, 1], True)
        view_720: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_237, [2304]);  sum_237 = None
        view_721: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_719, [1600, 2304]);  view_719 = None
        permute_660: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_721, [1, 0])
        mm_168: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_660, view_76);  permute_660 = view_76 = None
        mm_169: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_721, permute_662);  view_721 = permute_662 = None
        view_722: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_169, [50, 32, 768]);  mm_169 = None
        convert_element_type_1130: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_722, torch.float32);  view_722 = None
        convert_element_type_1131: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_168, torch.float32);  mm_168 = None
        convert_element_type_1132: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_720, torch.float32);  view_720 = None
        permute_664: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1130, [1, 0, 2]);  convert_element_type_1130 = None
        mul_523: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_664, primals_67);  primals_67 = None
        mul_524: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_523, 768)
        sum_238: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
        mul_525: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_523, mul_32);  mul_523 = None
        sum_239: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
        mul_526: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, sum_239);  sum_239 = None
        sub_188: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_524, sum_238);  mul_524 = sum_238 = None
        sub_189: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_188, mul_526);  sub_188 = mul_526 = None
        mul_527: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_49, sub_189);  div_49 = sub_189 = None
        mul_528: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_664, mul_32);  mul_32 = None
        sum_240: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
        sum_241: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_664, [0, 1]);  permute_664 = None
        add_260: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_257, mul_527);  add_257 = mul_527 = None
        convert_element_type_1133: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_260, torch.float16)
        view_723: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1133, [1600, 768]);  convert_element_type_1133 = None
        mm_170: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_723, permute_665);  permute_665 = None
        permute_666: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_723, [1, 0])
        mm_171: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_666, view_74);  permute_666 = view_74 = None
        sum_242: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_723, [0], True);  view_723 = None
        view_724: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_242, [768]);  sum_242 = None
        view_725: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_170, [32, 50, 3072]);  mm_170 = None
        convert_element_type_1138: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_171, torch.float32);  mm_171 = None
        convert_element_type_1139: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_724, torch.float32);  view_724 = None
        view_73: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [32, 50, 3072]);  addmm_13 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_30: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_73, 1.702)
        sigmoid_4: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_30);  mul_30 = None
        mul_529: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_725, sigmoid_4)
        mul_530: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_725, view_73);  view_725 = view_73 = None
        convert_element_type_1140: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_530, torch.float32);  mul_530 = None
        convert_element_type_1141: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_4, torch.float32);  sigmoid_4 = None
        sub_190: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1141)
        mul_531: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1141, sub_190);  convert_element_type_1141 = sub_190 = None
        mul_532: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1140, mul_531);  convert_element_type_1140 = mul_531 = None
        convert_element_type_1142: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_532, torch.float16);  mul_532 = None
        mul_533: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1142, 1.702);  convert_element_type_1142 = None
        add_261: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_529, mul_533);  mul_529 = mul_533 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_726: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_261, [1600, 3072]);  add_261 = None
        mm_172: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_726, permute_669);  permute_669 = None
        permute_670: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_726, [1, 0])
        mm_173: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_670, view_72);  permute_670 = view_72 = None
        sum_243: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
        view_727: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_243, [3072]);  sum_243 = None
        view_728: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_172, [32, 50, 768]);  mm_172 = None
        convert_element_type_1147: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_728, torch.float32);  view_728 = None
        convert_element_type_1148: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_173, torch.float32);  mm_173 = None
        convert_element_type_1149: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_727, torch.float32);  view_727 = None
        mul_535: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1147, primals_61);  primals_61 = None
        mul_536: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, 768)
        sum_244: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True)
        mul_537: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_535, mul_28);  mul_535 = None
        sum_245: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True);  mul_537 = None
        mul_538: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_28, sum_245);  sum_245 = None
        sub_192: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_536, sum_244);  mul_536 = sum_244 = None
        sub_193: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_192, mul_538);  sub_192 = mul_538 = None
        mul_539: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_50, sub_193);  div_50 = sub_193 = None
        mul_540: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1147, mul_28);  mul_28 = None
        sum_246: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 1]);  mul_540 = None
        sum_247: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1147, [0, 1]);  convert_element_type_1147 = None
        add_262: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_260, mul_539);  add_260 = mul_539 = None
        convert_element_type_1150: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_262, torch.float16)
        permute_673: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1150, [1, 0, 2]);  convert_element_type_1150 = None
        clone_183: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_673, memory_format = torch.contiguous_format);  permute_673 = None
        view_729: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_183, [1600, 768]);  clone_183 = None
        mm_174: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_729, permute_674);  permute_674 = None
        permute_675: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_729, [1, 0])
        permute_51: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_56, [2, 0, 1, 3])
        view_70: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_51, [1600, 768]);  permute_51 = None
        mm_175: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_675, view_70);  permute_675 = view_70 = None
        sum_248: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
        view_730: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_248, [768]);  sum_248 = None
        convert_element_type_1155: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_175, torch.float32);  mm_175 = None
        convert_element_type_1156: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_730, torch.float32);  view_730 = None
        view_731: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_174, [50, 32, 12, 64]);  mm_174 = None
        permute_678: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_731, [1, 2, 0, 3]);  view_731 = None
        _scaled_dot_product_cudnn_attention_backward_19 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_678, view_67, view_68, view_69, getitem_56, getitem_57, getitem_62, getitem_63, None, None, None, 50, 50, 0.0, False);  permute_678 = view_67 = view_68 = view_69 = getitem_56 = getitem_57 = getitem_62 = getitem_63 = None
        getitem_375: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_19[0]
        getitem_376: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_19[1]
        getitem_377: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_19[2];  _scaled_dot_product_cudnn_attention_backward_19 = None
        view_732: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_377, [384, 50, 64]);  getitem_377 = None
        view_733: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_376, [384, 50, 64]);  getitem_376 = None
        view_734: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_375, [384, 50, 64]);  getitem_375 = None
        permute_679: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_732, [1, 0, 2]);  view_732 = None
        view_735: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_679, [50, 32, 768]);  permute_679 = None
        permute_680: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_733, [1, 0, 2]);  view_733 = None
        view_736: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_680, [50, 32, 768]);  permute_680 = None
        permute_681: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_734, [1, 0, 2]);  view_734 = None
        view_737: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_681, [50, 32, 768]);  permute_681 = None

        # No stacktrace found for following nodes
        select_scatter_default_58: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_735, 0, 2);  view_735 = None
        select_scatter_default_59: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_736, 0, 1);  view_736 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_263: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_58, select_scatter_default_59);  select_scatter_default_58 = select_scatter_default_59 = None

        # No stacktrace found for following nodes
        select_scatter_default_60: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_737, 0, 0);  view_737 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_264: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_263, select_scatter_default_60);  add_263 = select_scatter_default_60 = None
        unsqueeze_45: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_264, 3);  add_264 = None
        permute_682: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_45, [3, 1, 2, 0, 4]);  unsqueeze_45 = None
        squeeze_43: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_682, 0);  permute_682 = None
        clone_184: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_43, memory_format = torch.contiguous_format);  squeeze_43 = None
        view_738: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_184, [50, 32, 2304]);  clone_184 = None
        sum_249: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_738, [0, 1], True)
        view_739: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_249, [2304]);  sum_249 = None
        view_740: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_738, [1600, 2304]);  view_738 = None
        permute_683: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_176: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_683, view_61);  permute_683 = view_61 = None
        mm_177: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_740, permute_685);  view_740 = permute_685 = None
        view_741: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_177, [50, 32, 768]);  mm_177 = None
        convert_element_type_1161: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_741, torch.float32);  view_741 = None
        convert_element_type_1162: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_176, torch.float32);  mm_176 = None
        convert_element_type_1163: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_739, torch.float32);  view_739 = None
        permute_687: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1161, [1, 0, 2]);  convert_element_type_1161 = None
        mul_542: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_687, primals_55);  primals_55 = None
        mul_543: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, 768)
        sum_250: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_542, [2], True)
        mul_544: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_542, mul_26);  mul_542 = None
        sum_251: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_544, [2], True);  mul_544 = None
        mul_545: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_26, sum_251);  sum_251 = None
        sub_195: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_543, sum_250);  mul_543 = sum_250 = None
        sub_196: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_195, mul_545);  sub_195 = mul_545 = None
        mul_546: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_51, sub_196);  div_51 = sub_196 = None
        mul_547: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_687, mul_26);  mul_26 = None
        sum_252: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_547, [0, 1]);  mul_547 = None
        sum_253: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_687, [0, 1]);  permute_687 = None
        add_265: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_262, mul_546);  add_262 = mul_546 = None
        convert_element_type_1164: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_265, torch.float16)
        view_742: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1164, [1600, 768]);  convert_element_type_1164 = None
        mm_178: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_742, permute_688);  permute_688 = None
        permute_689: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_742, [1, 0])
        mm_179: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_689, view_59);  permute_689 = view_59 = None
        sum_254: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_742, [0], True);  view_742 = None
        view_743: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_254, [768]);  sum_254 = None
        view_744: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_178, [32, 50, 3072]);  mm_178 = None
        convert_element_type_1169: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_179, torch.float32);  mm_179 = None
        convert_element_type_1170: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_743, torch.float32);  view_743 = None
        view_58: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [32, 50, 3072]);  addmm_10 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_24: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_58, 1.702)
        sigmoid_3: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_24);  mul_24 = None
        mul_548: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_744, sigmoid_3)
        mul_549: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_744, view_58);  view_744 = view_58 = None
        convert_element_type_1171: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_549, torch.float32);  mul_549 = None
        convert_element_type_1172: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_3, torch.float32);  sigmoid_3 = None
        sub_197: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1172)
        mul_550: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1172, sub_197);  convert_element_type_1172 = sub_197 = None
        mul_551: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1171, mul_550);  convert_element_type_1171 = mul_550 = None
        convert_element_type_1173: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_551, torch.float16);  mul_551 = None
        mul_552: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1173, 1.702);  convert_element_type_1173 = None
        add_266: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_548, mul_552);  mul_548 = mul_552 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_745: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_266, [1600, 3072]);  add_266 = None
        mm_180: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_745, permute_692);  permute_692 = None
        permute_693: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_745, [1, 0])
        mm_181: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_693, view_57);  permute_693 = view_57 = None
        sum_255: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_745, [0], True);  view_745 = None
        view_746: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_255, [3072]);  sum_255 = None
        view_747: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_180, [32, 50, 768]);  mm_180 = None
        convert_element_type_1178: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_747, torch.float32);  view_747 = None
        convert_element_type_1179: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_181, torch.float32);  mm_181 = None
        convert_element_type_1180: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_746, torch.float32);  view_746 = None
        mul_554: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1178, primals_49);  primals_49 = None
        mul_555: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_554, 768)
        sum_256: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_554, [2], True)
        mul_556: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_554, mul_22);  mul_554 = None
        sum_257: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_556, [2], True);  mul_556 = None
        mul_557: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_22, sum_257);  sum_257 = None
        sub_199: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_555, sum_256);  mul_555 = sum_256 = None
        sub_200: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_199, mul_557);  sub_199 = mul_557 = None
        mul_558: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_52, sub_200);  div_52 = sub_200 = None
        mul_559: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1178, mul_22);  mul_22 = None
        sum_258: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1]);  mul_559 = None
        sum_259: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1178, [0, 1]);  convert_element_type_1178 = None
        add_267: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_265, mul_558);  add_265 = mul_558 = None
        convert_element_type_1181: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_267, torch.float16)
        permute_696: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1181, [1, 0, 2]);  convert_element_type_1181 = None
        clone_185: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_696, memory_format = torch.contiguous_format);  permute_696 = None
        view_748: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_185, [1600, 768]);  clone_185 = None
        mm_182: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_748, permute_697);  permute_697 = None
        permute_698: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_748, [1, 0])
        permute_40: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_43, [2, 0, 1, 3])
        view_55: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_40, [1600, 768]);  permute_40 = None
        mm_183: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_698, view_55);  permute_698 = view_55 = None
        sum_260: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_748, [0], True);  view_748 = None
        view_749: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_260, [768]);  sum_260 = None
        convert_element_type_1186: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_183, torch.float32);  mm_183 = None
        convert_element_type_1187: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_749, torch.float32);  view_749 = None
        view_750: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_182, [50, 32, 12, 64]);  mm_182 = None
        permute_701: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_750, [1, 2, 0, 3]);  view_750 = None
        _scaled_dot_product_cudnn_attention_backward_20 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_701, view_52, view_53, view_54, getitem_43, getitem_44, getitem_49, getitem_50, None, None, None, 50, 50, 0.0, False);  permute_701 = view_52 = view_53 = view_54 = getitem_43 = getitem_44 = getitem_49 = getitem_50 = None
        getitem_378: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_20[0]
        getitem_379: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_20[1]
        getitem_380: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_20[2];  _scaled_dot_product_cudnn_attention_backward_20 = None
        view_751: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_380, [384, 50, 64]);  getitem_380 = None
        view_752: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_379, [384, 50, 64]);  getitem_379 = None
        view_753: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_378, [384, 50, 64]);  getitem_378 = None
        permute_702: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_751, [1, 0, 2]);  view_751 = None
        view_754: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_702, [50, 32, 768]);  permute_702 = None
        permute_703: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_752, [1, 0, 2]);  view_752 = None
        view_755: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_703, [50, 32, 768]);  permute_703 = None
        permute_704: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_753, [1, 0, 2]);  view_753 = None
        view_756: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_704, [50, 32, 768]);  permute_704 = None

        # No stacktrace found for following nodes
        select_scatter_default_61: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_754, 0, 2);  view_754 = None
        select_scatter_default_62: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_755, 0, 1);  view_755 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_268: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_61, select_scatter_default_62);  select_scatter_default_61 = select_scatter_default_62 = None

        # No stacktrace found for following nodes
        select_scatter_default_63: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_756, 0, 0);  view_756 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_269: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_268, select_scatter_default_63);  add_268 = select_scatter_default_63 = None
        unsqueeze_46: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_269, 3);  add_269 = None
        permute_705: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_46, [3, 1, 2, 0, 4]);  unsqueeze_46 = None
        squeeze_44: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_705, 0);  permute_705 = None
        clone_186: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_44, memory_format = torch.contiguous_format);  squeeze_44 = None
        view_757: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_186, [50, 32, 2304]);  clone_186 = None
        sum_261: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_757, [0, 1], True)
        view_758: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_261, [2304]);  sum_261 = None
        view_759: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_757, [1600, 2304]);  view_757 = None
        permute_706: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_759, [1, 0])
        mm_184: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_706, view_46);  permute_706 = view_46 = None
        mm_185: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_759, permute_708);  view_759 = permute_708 = None
        view_760: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_185, [50, 32, 768]);  mm_185 = None
        convert_element_type_1192: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_760, torch.float32);  view_760 = None
        convert_element_type_1193: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_184, torch.float32);  mm_184 = None
        convert_element_type_1194: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_758, torch.float32);  view_758 = None
        permute_710: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1192, [1, 0, 2]);  convert_element_type_1192 = None
        mul_561: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_710, primals_43);  primals_43 = None
        mul_562: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_561, 768)
        sum_262: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_561, [2], True)
        mul_563: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_561, mul_20);  mul_561 = None
        sum_263: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True);  mul_563 = None
        mul_564: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, sum_263);  sum_263 = None
        sub_202: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_562, sum_262);  mul_562 = sum_262 = None
        sub_203: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_202, mul_564);  sub_202 = mul_564 = None
        mul_565: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_53, sub_203);  div_53 = sub_203 = None
        mul_566: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_710, mul_20);  mul_20 = None
        sum_264: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_566, [0, 1]);  mul_566 = None
        sum_265: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_710, [0, 1]);  permute_710 = None
        add_270: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_267, mul_565);  add_267 = mul_565 = None
        convert_element_type_1195: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_270, torch.float16)
        view_761: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1195, [1600, 768]);  convert_element_type_1195 = None
        mm_186: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_761, permute_711);  permute_711 = None
        permute_712: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_761, [1, 0])
        mm_187: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_712, view_44);  permute_712 = view_44 = None
        sum_266: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_761, [0], True);  view_761 = None
        view_762: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_266, [768]);  sum_266 = None
        view_763: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_186, [32, 50, 3072]);  mm_186 = None
        convert_element_type_1200: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_187, torch.float32);  mm_187 = None
        convert_element_type_1201: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_762, torch.float32);  view_762 = None
        view_43: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [32, 50, 3072]);  addmm_7 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_18: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_43, 1.702)
        sigmoid_2: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_18);  mul_18 = None
        mul_567: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_763, sigmoid_2)
        mul_568: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_763, view_43);  view_763 = view_43 = None
        convert_element_type_1202: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_568, torch.float32);  mul_568 = None
        convert_element_type_1203: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_2, torch.float32);  sigmoid_2 = None
        sub_204: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1203)
        mul_569: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1203, sub_204);  convert_element_type_1203 = sub_204 = None
        mul_570: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1202, mul_569);  convert_element_type_1202 = mul_569 = None
        convert_element_type_1204: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_570, torch.float16);  mul_570 = None
        mul_571: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1204, 1.702);  convert_element_type_1204 = None
        add_271: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_567, mul_571);  mul_567 = mul_571 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_764: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_271, [1600, 3072]);  add_271 = None
        mm_188: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_764, permute_715);  permute_715 = None
        permute_716: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_764, [1, 0])
        mm_189: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_716, view_42);  permute_716 = view_42 = None
        sum_267: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_764, [0], True);  view_764 = None
        view_765: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_267, [3072]);  sum_267 = None
        view_766: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_188, [32, 50, 768]);  mm_188 = None
        convert_element_type_1209: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_766, torch.float32);  view_766 = None
        convert_element_type_1210: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_189, torch.float32);  mm_189 = None
        convert_element_type_1211: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_765, torch.float32);  view_765 = None
        mul_573: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1209, primals_37);  primals_37 = None
        mul_574: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, 768)
        sum_268: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True)
        mul_575: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_573, mul_16);  mul_573 = None
        sum_269: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_575, [2], True);  mul_575 = None
        mul_576: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, sum_269);  sum_269 = None
        sub_206: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_574, sum_268);  mul_574 = sum_268 = None
        sub_207: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_206, mul_576);  sub_206 = mul_576 = None
        mul_577: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_54, sub_207);  div_54 = sub_207 = None
        mul_578: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1209, mul_16);  mul_16 = None
        sum_270: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_578, [0, 1]);  mul_578 = None
        sum_271: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1209, [0, 1]);  convert_element_type_1209 = None
        add_272: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_270, mul_577);  add_270 = mul_577 = None
        convert_element_type_1212: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_272, torch.float16)
        permute_719: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1212, [1, 0, 2]);  convert_element_type_1212 = None
        clone_187: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_719, memory_format = torch.contiguous_format);  permute_719 = None
        view_767: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_187, [1600, 768]);  clone_187 = None
        mm_190: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_767, permute_720);  permute_720 = None
        permute_721: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_767, [1, 0])
        permute_29: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_30, [2, 0, 1, 3])
        view_40: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_29, [1600, 768]);  permute_29 = None
        mm_191: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_721, view_40);  permute_721 = view_40 = None
        sum_272: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_767, [0], True);  view_767 = None
        view_768: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_272, [768]);  sum_272 = None
        convert_element_type_1217: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_191, torch.float32);  mm_191 = None
        convert_element_type_1218: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_768, torch.float32);  view_768 = None
        view_769: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_190, [50, 32, 12, 64]);  mm_190 = None
        permute_724: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_769, [1, 2, 0, 3]);  view_769 = None
        _scaled_dot_product_cudnn_attention_backward_21 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_724, view_37, view_38, view_39, getitem_30, getitem_31, getitem_36, getitem_37, None, None, None, 50, 50, 0.0, False);  permute_724 = view_37 = view_38 = view_39 = getitem_30 = getitem_31 = getitem_36 = getitem_37 = None
        getitem_381: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_21[0]
        getitem_382: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_21[1]
        getitem_383: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_21[2];  _scaled_dot_product_cudnn_attention_backward_21 = None
        view_770: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_383, [384, 50, 64]);  getitem_383 = None
        view_771: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_382, [384, 50, 64]);  getitem_382 = None
        view_772: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_381, [384, 50, 64]);  getitem_381 = None
        permute_725: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_770, [1, 0, 2]);  view_770 = None
        view_773: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_725, [50, 32, 768]);  permute_725 = None
        permute_726: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_771, [1, 0, 2]);  view_771 = None
        view_774: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_726, [50, 32, 768]);  permute_726 = None
        permute_727: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_772, [1, 0, 2]);  view_772 = None
        view_775: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_727, [50, 32, 768]);  permute_727 = None

        # No stacktrace found for following nodes
        select_scatter_default_64: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_773, 0, 2);  view_773 = None
        select_scatter_default_65: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_774, 0, 1);  view_774 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_273: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_64, select_scatter_default_65);  select_scatter_default_64 = select_scatter_default_65 = None

        # No stacktrace found for following nodes
        select_scatter_default_66: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_775, 0, 0);  view_775 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_274: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_273, select_scatter_default_66);  add_273 = select_scatter_default_66 = None
        unsqueeze_47: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_274, 3);  add_274 = None
        permute_728: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_47, [3, 1, 2, 0, 4]);  unsqueeze_47 = None
        squeeze_45: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_728, 0);  permute_728 = None
        clone_188: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_45, memory_format = torch.contiguous_format);  squeeze_45 = None
        view_776: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_188, [50, 32, 2304]);  clone_188 = None
        sum_273: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_776, [0, 1], True)
        view_777: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_273, [2304]);  sum_273 = None
        view_778: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_776, [1600, 2304]);  view_776 = None
        permute_729: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_778, [1, 0])
        mm_192: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_729, view_31);  permute_729 = view_31 = None
        mm_193: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_778, permute_731);  view_778 = permute_731 = None
        view_779: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_193, [50, 32, 768]);  mm_193 = None
        convert_element_type_1223: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_779, torch.float32);  view_779 = None
        convert_element_type_1224: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_192, torch.float32);  mm_192 = None
        convert_element_type_1225: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_777, torch.float32);  view_777 = None
        permute_733: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1223, [1, 0, 2]);  convert_element_type_1223 = None
        mul_580: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_733, primals_31);  primals_31 = None
        mul_581: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_580, 768)
        sum_274: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_580, [2], True)
        mul_582: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_580, mul_14);  mul_580 = None
        sum_275: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_582, [2], True);  mul_582 = None
        mul_583: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_14, sum_275);  sum_275 = None
        sub_209: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_581, sum_274);  mul_581 = sum_274 = None
        sub_210: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_209, mul_583);  sub_209 = mul_583 = None
        mul_584: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_55, sub_210);  div_55 = sub_210 = None
        mul_585: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_733, mul_14);  mul_14 = None
        sum_276: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 1]);  mul_585 = None
        sum_277: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_733, [0, 1]);  permute_733 = None
        add_275: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_272, mul_584);  add_272 = mul_584 = None
        convert_element_type_1226: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_275, torch.float16)
        view_780: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1226, [1600, 768]);  convert_element_type_1226 = None
        mm_194: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_780, permute_734);  permute_734 = None
        permute_735: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_780, [1, 0])
        mm_195: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_735, view_29);  permute_735 = view_29 = None
        sum_278: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
        view_781: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_278, [768]);  sum_278 = None
        view_782: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_194, [32, 50, 3072]);  mm_194 = None
        convert_element_type_1231: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_195, torch.float32);  mm_195 = None
        convert_element_type_1232: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_781, torch.float32);  view_781 = None
        view_28: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [32, 50, 3072]);  addmm_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_12: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_28, 1.702)
        sigmoid_1: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_12);  mul_12 = None
        mul_586: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_782, sigmoid_1)
        mul_587: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_782, view_28);  view_782 = view_28 = None
        convert_element_type_1233: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_587, torch.float32);  mul_587 = None
        convert_element_type_1234: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid_1, torch.float32);  sigmoid_1 = None
        sub_211: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1234)
        mul_588: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1234, sub_211);  convert_element_type_1234 = sub_211 = None
        mul_589: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1233, mul_588);  convert_element_type_1233 = mul_588 = None
        convert_element_type_1235: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_589, torch.float16);  mul_589 = None
        mul_590: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1235, 1.702);  convert_element_type_1235 = None
        add_276: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_586, mul_590);  mul_586 = mul_590 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_783: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_276, [1600, 3072]);  add_276 = None
        mm_196: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_783, permute_738);  permute_738 = None
        permute_739: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_783, [1, 0])
        mm_197: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_739, view_27);  permute_739 = view_27 = None
        sum_279: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_783, [0], True);  view_783 = None
        view_784: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_279, [3072]);  sum_279 = None
        view_785: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_196, [32, 50, 768]);  mm_196 = None
        convert_element_type_1240: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_785, torch.float32);  view_785 = None
        convert_element_type_1241: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_197, torch.float32);  mm_197 = None
        convert_element_type_1242: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_784, torch.float32);  view_784 = None
        mul_592: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1240, primals_25);  primals_25 = None
        mul_593: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, 768)
        sum_280: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_592, [2], True)
        mul_594: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_592, mul_10);  mul_592 = None
        sum_281: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_594, [2], True);  mul_594 = None
        mul_595: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_10, sum_281);  sum_281 = None
        sub_213: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_593, sum_280);  mul_593 = sum_280 = None
        sub_214: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_213, mul_595);  sub_213 = mul_595 = None
        mul_596: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_56, sub_214);  div_56 = sub_214 = None
        mul_597: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1240, mul_10);  mul_10 = None
        sum_282: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_597, [0, 1]);  mul_597 = None
        sum_283: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1240, [0, 1]);  convert_element_type_1240 = None
        add_277: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_275, mul_596);  add_275 = mul_596 = None
        convert_element_type_1243: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_277, torch.float16)
        permute_742: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1243, [1, 0, 2]);  convert_element_type_1243 = None
        clone_189: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_742, memory_format = torch.contiguous_format);  permute_742 = None
        view_786: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_189, [1600, 768]);  clone_189 = None
        mm_198: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_786, permute_743);  permute_743 = None
        permute_744: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_786, [1, 0])
        permute_18: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_17, [2, 0, 1, 3])
        view_25: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_18, [1600, 768]);  permute_18 = None
        mm_199: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_744, view_25);  permute_744 = view_25 = None
        sum_284: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
        view_787: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_284, [768]);  sum_284 = None
        convert_element_type_1248: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_199, torch.float32);  mm_199 = None
        convert_element_type_1249: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_787, torch.float32);  view_787 = None
        view_788: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_198, [50, 32, 12, 64]);  mm_198 = None
        permute_747: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_788, [1, 2, 0, 3]);  view_788 = None
        _scaled_dot_product_cudnn_attention_backward_22 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_747, view_22, view_23, view_24, getitem_17, getitem_18, getitem_23, getitem_24, None, None, None, 50, 50, 0.0, False);  permute_747 = view_22 = view_23 = view_24 = getitem_17 = getitem_18 = getitem_23 = getitem_24 = None
        getitem_384: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_22[0]
        getitem_385: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_22[1]
        getitem_386: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_22[2];  _scaled_dot_product_cudnn_attention_backward_22 = None
        view_789: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_386, [384, 50, 64]);  getitem_386 = None
        view_790: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_385, [384, 50, 64]);  getitem_385 = None
        view_791: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_384, [384, 50, 64]);  getitem_384 = None
        permute_748: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_789, [1, 0, 2]);  view_789 = None
        view_792: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_748, [50, 32, 768]);  permute_748 = None
        permute_749: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_790, [1, 0, 2]);  view_790 = None
        view_793: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_749, [50, 32, 768]);  permute_749 = None
        permute_750: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_791, [1, 0, 2]);  view_791 = None
        view_794: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_750, [50, 32, 768]);  permute_750 = None

        # No stacktrace found for following nodes
        select_scatter_default_67: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_792, 0, 2);  view_792 = None
        select_scatter_default_68: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_793, 0, 1);  view_793 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_278: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_67, select_scatter_default_68);  select_scatter_default_67 = select_scatter_default_68 = None

        # No stacktrace found for following nodes
        select_scatter_default_69: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_794, 0, 0);  view_794 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_279: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_278, select_scatter_default_69);  add_278 = select_scatter_default_69 = None
        unsqueeze_48: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_279, 3);  add_279 = None
        permute_751: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_48, [3, 1, 2, 0, 4]);  unsqueeze_48 = None
        squeeze_46: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_751, 0);  permute_751 = None
        clone_190: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_46, memory_format = torch.contiguous_format);  squeeze_46 = None
        view_795: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_190, [50, 32, 2304]);  clone_190 = None
        sum_285: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_795, [0, 1], True)
        view_796: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_285, [2304]);  sum_285 = None
        view_797: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_795, [1600, 2304]);  view_795 = None
        permute_752: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_797, [1, 0])
        mm_200: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_752, view_16);  permute_752 = view_16 = None
        mm_201: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_797, permute_754);  view_797 = permute_754 = None
        view_798: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_201, [50, 32, 768]);  mm_201 = None
        convert_element_type_1254: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_798, torch.float32);  view_798 = None
        convert_element_type_1255: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_200, torch.float32);  mm_200 = None
        convert_element_type_1256: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_796, torch.float32);  view_796 = None
        permute_756: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1254, [1, 0, 2]);  convert_element_type_1254 = None
        mul_599: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_756, primals_19);  primals_19 = None
        mul_600: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_599, 768)
        sum_286: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True)
        mul_601: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_599, mul_8);  mul_599 = None
        sum_287: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True);  mul_601 = None
        mul_602: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_8, sum_287);  sum_287 = None
        sub_216: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_600, sum_286);  mul_600 = sum_286 = None
        sub_217: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_216, mul_602);  sub_216 = mul_602 = None
        mul_603: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_57, sub_217);  div_57 = sub_217 = None
        mul_604: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_756, mul_8);  mul_8 = None
        sum_288: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 1]);  mul_604 = None
        sum_289: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_756, [0, 1]);  permute_756 = None
        add_280: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_277, mul_603);  add_277 = mul_603 = None
        convert_element_type_1257: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_280, torch.float16)
        view_799: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1257, [1600, 768]);  convert_element_type_1257 = None
        mm_202: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(view_799, permute_757);  permute_757 = None
        permute_758: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_203: "f16[768, 3072][3072, 1]cuda:0" = torch.ops.aten.mm.default(permute_758, view_14);  permute_758 = view_14 = None
        sum_290: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        view_800: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_290, [768]);  sum_290 = None
        view_801: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(mm_202, [32, 50, 3072]);  mm_202 = None
        convert_element_type_1262: "f32[768, 3072][3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_203, torch.float32);  mm_203 = None
        convert_element_type_1263: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_800, torch.float32);  view_800 = None
        view_13: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [32, 50, 3072]);  addmm_1 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/activation.py:25 in forward, code: return torch.sigmoid(1.702 * x) * x
        mul_6: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_13, 1.702)
        sigmoid: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sigmoid.default(mul_6);  mul_6 = None
        mul_605: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_801, sigmoid)
        mul_606: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(view_801, view_13);  view_801 = view_13 = None
        convert_element_type_1264: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_606, torch.float32);  mul_606 = None
        convert_element_type_1265: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sigmoid, torch.float32);  sigmoid = None
        sub_218: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.sub.Tensor(1, convert_element_type_1265)
        mul_607: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1265, sub_218);  convert_element_type_1265 = sub_218 = None
        mul_608: "f32[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1264, mul_607);  convert_element_type_1264 = mul_607 = None
        convert_element_type_1266: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_608, torch.float16);  mul_608 = None
        mul_609: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1266, 1.702);  convert_element_type_1266 = None
        add_281: "f16[32, 50, 3072][153600, 3072, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_605, mul_609);  mul_605 = mul_609 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        view_802: "f16[1600, 3072][3072, 1]cuda:0" = torch.ops.aten.reshape.default(add_281, [1600, 3072]);  add_281 = None
        mm_204: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_802, permute_761);  permute_761 = None
        permute_762: "f16[3072, 1600][1, 3072]cuda:0" = torch.ops.aten.permute.default(view_802, [1, 0])
        mm_205: "f16[3072, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_762, view_12);  permute_762 = view_12 = None
        sum_291: "f16[1, 3072][3072, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_802, [0], True);  view_802 = None
        view_803: "f16[3072][1]cuda:0" = torch.ops.aten.reshape.default(sum_291, [3072]);  sum_291 = None
        view_804: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_204, [32, 50, 768]);  mm_204 = None
        convert_element_type_1271: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_804, torch.float32);  view_804 = None
        convert_element_type_1272: "f32[3072, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_205, torch.float32);  mm_205 = None
        convert_element_type_1273: "f32[3072][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_803, torch.float32);  view_803 = None
        mul_611: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1271, primals_13);  primals_13 = None
        mul_612: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_611, 768)
        sum_292: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_611, [2], True)
        mul_613: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_611, mul_4);  mul_611 = None
        sum_293: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_613, [2], True);  mul_613 = None
        mul_614: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, sum_293);  sum_293 = None
        sub_220: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_612, sum_292);  mul_612 = sum_292 = None
        sub_221: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_220, mul_614);  sub_220 = mul_614 = None
        mul_615: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_58, sub_221);  div_58 = sub_221 = None
        mul_616: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1271, mul_4);  mul_4 = None
        sum_294: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_616, [0, 1]);  mul_616 = None
        sum_295: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(convert_element_type_1271, [0, 1]);  convert_element_type_1271 = None
        add_282: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_280, mul_615);  add_280 = mul_615 = None
        convert_element_type_1274: "f16[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_282, torch.float16)
        permute_765: "f16[50, 32, 768][768, 38400, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1274, [1, 0, 2]);  convert_element_type_1274 = None
        clone_191: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.clone.default(permute_765, memory_format = torch.contiguous_format);  permute_765 = None
        view_805: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(clone_191, [1600, 768]);  clone_191 = None
        mm_206: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_805, permute_766);  permute_766 = None
        permute_767: "f16[768, 1600][1, 768]cuda:0" = torch.ops.aten.permute.default(view_805, [1, 0])
        permute_7: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.permute.default(getitem_4, [2, 0, 1, 3])
        view_10: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [1600, 768]);  permute_7 = None
        mm_207: "f16[768, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_767, view_10);  permute_767 = view_10 = None
        sum_296: "f16[1, 768][768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_805, [0], True);  view_805 = None
        view_806: "f16[768][1]cuda:0" = torch.ops.aten.reshape.default(sum_296, [768]);  sum_296 = None
        convert_element_type_1279: "f32[768, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_207, torch.float32);  mm_207 = None
        convert_element_type_1280: "f32[768][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_806, torch.float32);  view_806 = None
        view_807: "f16[50, 32, 12, 64][24576, 768, 64, 1]cuda:0" = torch.ops.aten.reshape.default(mm_206, [50, 32, 12, 64]);  mm_206 = None
        permute_770: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = torch.ops.aten.permute.default(view_807, [1, 2, 0, 3]);  view_807 = None
        _scaled_dot_product_cudnn_attention_backward_23 = torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default(permute_770, view_7, view_8, view_9, getitem_4, getitem_5, getitem_10, getitem_11, None, None, None, 50, 50, 0.0, False);  permute_770 = view_7 = view_8 = view_9 = getitem_4 = getitem_5 = getitem_10 = getitem_11 = None
        getitem_387: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_23[0]
        getitem_388: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_23[1]
        getitem_389: "f16[32, 12, 50, 64][768, 64, 24576, 1]cuda:0" = _scaled_dot_product_cudnn_attention_backward_23[2];  _scaled_dot_product_cudnn_attention_backward_23 = None
        view_808: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_389, [384, 50, 64]);  getitem_389 = None
        view_809: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_388, [384, 50, 64]);  getitem_388 = None
        view_810: "f16[384, 50, 64][64, 24576, 1]cuda:0" = torch.ops.aten.reshape.default(getitem_387, [384, 50, 64]);  getitem_387 = None
        permute_771: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_808, [1, 0, 2]);  view_808 = None
        view_811: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_771, [50, 32, 768]);  permute_771 = None
        permute_772: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_809, [1, 0, 2]);  view_809 = None
        view_812: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_772, [50, 32, 768]);  permute_772 = None
        permute_773: "f16[50, 384, 64][24576, 64, 1]cuda:0" = torch.ops.aten.permute.default(view_810, [1, 0, 2]);  view_810 = None
        view_813: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_773, [50, 32, 768]);  permute_773 = None

        # No stacktrace found for following nodes
        select_scatter_default_70: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_811, 0, 2);  view_811 = None
        select_scatter_default_71: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_812, 0, 1);  view_812 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_283: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(select_scatter_default_70, select_scatter_default_71);  select_scatter_default_70 = select_scatter_default_71 = None

        # No stacktrace found for following nodes
        select_scatter_default_72: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.select_scatter.default(full_default_44, view_813, 0, 0);  full_default_44 = view_813 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        add_284: "f16[3, 50, 32, 768][1228800, 24576, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_283, select_scatter_default_72);  add_283 = select_scatter_default_72 = None
        unsqueeze_49: "f16[3, 50, 32, 1, 768][1228800, 24576, 768, 768, 1]cuda:0" = torch.ops.aten.unsqueeze.default(add_284, 3);  add_284 = None
        permute_774: "f16[1, 50, 32, 3, 768][768, 24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_49, [3, 1, 2, 0, 4]);  unsqueeze_49 = None
        squeeze_47: "f16[50, 32, 3, 768][24576, 768, 1228800, 1]cuda:0" = torch.ops.aten.squeeze.dim(permute_774, 0);  permute_774 = None
        clone_192: "f16[50, 32, 3, 768][73728, 2304, 768, 1]cuda:0" = torch.ops.aten.clone.default(squeeze_47, memory_format = torch.contiguous_format);  squeeze_47 = None
        view_814: "f16[50, 32, 2304][73728, 2304, 1]cuda:0" = torch.ops.aten.reshape.default(clone_192, [50, 32, 2304]);  clone_192 = None
        sum_297: "f16[1, 1, 2304][2304, 2304, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(view_814, [0, 1], True)
        view_815: "f16[2304][1]cuda:0" = torch.ops.aten.reshape.default(sum_297, [2304]);  sum_297 = None
        view_816: "f16[1600, 2304][2304, 1]cuda:0" = torch.ops.aten.reshape.default(view_814, [1600, 2304]);  view_814 = None
        permute_775: "f16[2304, 1600][1, 2304]cuda:0" = torch.ops.aten.permute.default(view_816, [1, 0])
        mm_208: "f16[2304, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(permute_775, view_1);  permute_775 = view_1 = None
        mm_209: "f16[1600, 768][768, 1]cuda:0" = torch.ops.aten.mm.default(view_816, permute_777);  view_816 = permute_777 = None
        view_817: "f16[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.aten.reshape.default(mm_209, [50, 32, 768]);  mm_209 = None
        convert_element_type_1285: "f32[50, 32, 768][24576, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_817, torch.float32);  view_817 = None
        convert_element_type_1286: "f32[2304, 768][768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mm_208, torch.float32);  mm_208 = None
        convert_element_type_1287: "f32[2304][1]cuda:0" = torch.ops.prims.convert_element_type.default(view_815, torch.float32);  view_815 = None
        permute_779: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1285, [1, 0, 2]);  convert_element_type_1285 = None
        mul_618: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_779, primals_7);  primals_7 = None
        mul_619: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_618, 768)
        sum_298: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_618, [2], True)

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:105 in forward, code: x = x + self.positional_embedding
        add: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(cat, primals_4);  cat = primals_4 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        sub: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add, getitem_1);  add = getitem_1 = None
        mul: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = None
        mul_1: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, primals_5)
        add_2: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_1, primals_6);  mul_1 = primals_6 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:108 in forward, code: x = self.encoder(x)
        sub_1: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_2, getitem_3);  add_2 = getitem_3 = None
        mul_2: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt_1);  sub_1 = None
        mul_620: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_618, mul_2);  mul_618 = None
        sum_299: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_620, [2], True);  mul_620 = None
        mul_621: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_2, sum_299);  sum_299 = None
        sub_223: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_619, sum_298);  mul_619 = sum_298 = None
        sub_224: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_223, mul_621);  sub_223 = mul_621 = None
        div_59: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 768);  rsqrt_1 = None
        mul_622: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_59, sub_224);  div_59 = sub_224 = None
        mul_623: "f32[32, 50, 768][768, 24576, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_779, mul_2);  mul_2 = None
        sum_300: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_623, [0, 1]);  mul_623 = None
        sum_301: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(permute_779, [0, 1]);  permute_779 = None
        add_285: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.add.Tensor(add_282, mul_622);  add_282 = mul_622 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/modules/layers/normalizations.py:18 in forward, code: output = nn.functional.layer_norm(
        mul_625: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_285, primals_5);  primals_5 = None
        mul_626: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_625, 768)
        sum_302: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_625, [2], True)
        mul_627: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_625, mul);  mul_625 = None
        sum_303: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_627, [2], True);  mul_627 = None
        mul_628: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, sum_303);  sum_303 = None
        sub_226: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_626, sum_302);  mul_626 = sum_302 = None
        sub_227: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_226, mul_628);  sub_226 = mul_628 = None
        div_60: "f32[32, 50, 1][50, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 768);  rsqrt = None
        mul_629: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(div_60, sub_227);  div_60 = sub_227 = None
        mul_630: "f32[32, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_285, mul);  mul = None
        sum_304: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_630, [0, 1]);  mul_630 = None
        sum_305: "f32[768][1]cuda:0" = torch.ops.aten.sum.dim_IntList(add_285, [0, 1]);  add_285 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:105 in forward, code: x = x + self.positional_embedding
        sum_306: "f32[1, 50, 768][38400, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(mul_629, [0], True)
        view_818: "f32[50, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(sum_306, [50, 768]);  sum_306 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:98 in forward, code: x = torch.cat(
        slice_1: "f32[32, 1, 768][38400, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(mul_629, 1, 0, 1)
        slice_2: "f32[32, 49, 768][38400, 768, 1]cuda:0" = torch.ops.aten.slice.Tensor(mul_629, 1, 1, 50);  mul_629 = None
        convert_element_type_1288: "f16[32, 49, 768][37632, 768, 1]cuda:0" = torch.ops.prims.convert_element_type.default(slice_2, torch.float16);  slice_2 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:100 in forward, code: self.cls_token_embedding.unsqueeze(0).expand(x.shape[0], -1, -1),
        sum_307: "f32[1, 1, 768][768, 768, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(slice_1, [0], True);  slice_1 = None
        view_819: "f32[1, 768][768, 1]cuda:0" = torch.ops.aten.reshape.default(sum_307, [1, 768]);  sum_307 = None
        squeeze_48: "f32[768][1]cuda:0" = torch.ops.aten.squeeze.dim(view_819, 0);  view_819 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:97 in forward, code: x = x.permute(0, 2, 1)
        permute_780: "f16[32, 768, 49][37632, 1, 768]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1288, [0, 2, 1]);  convert_element_type_1288 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:94 in forward, code: x = torch.flatten(x, start_dim=2)
        view_820: "f16[32, 768, 7, 7][37632, 1, 5376, 768]cuda:0" = torch.ops.aten.reshape.default(permute_780, [32, 768, 7, 7]);  permute_780 = None

        # File: /opt/conda/envs/py_3.10/lib/python3.10/site-packages/torchmultimodal/models/clip/image_encoder.py:91 in forward, code: x = self.conv(x)
        convolution_backward = torch.ops.aten.convolution_backward.default(view_820, convert_element_type_1, convert_element_type, [0], [32, 32], [0, 0], [1, 1], False, [0, 0], 1, [False, True, False]);  view_820 = convert_element_type_1 = convert_element_type = None
        getitem_391: "f16[768, 3, 32, 32][3072, 1024, 32, 1]cuda:0" = convolution_backward[1];  convolution_backward = None
        convert_element_type_1289: "f32[768, 3, 32, 32][3072, 1024, 32, 1]cuda:0" = torch.ops.prims.convert_element_type.default(getitem_391, torch.float32);  getitem_391 = None
        return (None, convert_element_type_1289, squeeze_48, view_818, sum_304, sum_305, sum_300, sum_301, convert_element_type_1287, convert_element_type_1286, convert_element_type_1279, convert_element_type_1280, sum_294, sum_295, convert_element_type_1272, convert_element_type_1273, convert_element_type_1262, convert_element_type_1263, sum_288, sum_289, convert_element_type_1256, convert_element_type_1255, convert_element_type_1248, convert_element_type_1249, sum_282, sum_283, convert_element_type_1241, convert_element_type_1242, convert_element_type_1231, convert_element_type_1232, sum_276, sum_277, convert_element_type_1225, convert_element_type_1224, convert_element_type_1217, convert_element_type_1218, sum_270, sum_271, convert_element_type_1210, convert_element_type_1211, convert_element_type_1200, convert_element_type_1201, sum_264, sum_265, convert_element_type_1194, convert_element_type_1193, convert_element_type_1186, convert_element_type_1187, sum_258, sum_259, convert_element_type_1179, convert_element_type_1180, convert_element_type_1169, convert_element_type_1170, sum_252, sum_253, convert_element_type_1163, convert_element_type_1162, convert_element_type_1155, convert_element_type_1156, sum_246, sum_247, convert_element_type_1148, convert_element_type_1149, convert_element_type_1138, convert_element_type_1139, sum_240, sum_241, convert_element_type_1132, convert_element_type_1131, convert_element_type_1124, convert_element_type_1125, sum_234, sum_235, convert_element_type_1117, convert_element_type_1118, convert_element_type_1107, convert_element_type_1108, sum_228, sum_229, convert_element_type_1101, convert_element_type_1100, convert_element_type_1093, convert_element_type_1094, sum_222, sum_223, convert_element_type_1086, convert_element_type_1087, convert_element_type_1076, convert_element_type_1077, sum_216, sum_217, convert_element_type_1070, convert_element_type_1069, convert_element_type_1062, convert_element_type_1063, sum_210, sum_211, convert_element_type_1055, convert_element_type_1056, convert_element_type_1045, convert_element_type_1046, sum_204, sum_205, convert_element_type_1039, convert_element_type_1038, convert_element_type_1031, convert_element_type_1032, sum_198, sum_199, convert_element_type_1024, convert_element_type_1025, convert_element_type_1014, convert_element_type_1015, sum_192, sum_193, convert_element_type_1008, convert_element_type_1007, convert_element_type_1000, convert_element_type_1001, sum_186, sum_187, convert_element_type_993, convert_element_type_994, convert_element_type_983, convert_element_type_984, sum_180, sum_181, convert_element_type_977, convert_element_type_976, convert_element_type_969, convert_element_type_970, sum_174, sum_175, convert_element_type_962, convert_element_type_963, convert_element_type_952, convert_element_type_953, sum_168, sum_169, convert_element_type_946, convert_element_type_945, convert_element_type_938, convert_element_type_939, sum_162, sum_163, convert_element_type_931, convert_element_type_932, convert_element_type_921, convert_element_type_922, sum_156, sum_157, convert_element_type_915, None, index_put_1, view_589, None, sum_151, sum_152, convert_element_type_909, convert_element_type_908, convert_element_type_901, convert_element_type_902, sum_145, sum_146, convert_element_type_894, convert_element_type_895, convert_element_type_884, convert_element_type_885, sum_139, sum_140, convert_element_type_878, convert_element_type_877, convert_element_type_870, convert_element_type_871, sum_133, sum_134, convert_element_type_863, convert_element_type_864, convert_element_type_853, convert_element_type_854, sum_127, sum_128, convert_element_type_847, convert_element_type_846, convert_element_type_839, convert_element_type_840, sum_121, sum_122, convert_element_type_832, convert_element_type_833, convert_element_type_822, convert_element_type_823, sum_115, sum_116, convert_element_type_816, convert_element_type_815, convert_element_type_808, convert_element_type_809, sum_109, sum_110, convert_element_type_801, convert_element_type_802, convert_element_type_791, convert_element_type_792, sum_103, sum_104, convert_element_type_785, convert_element_type_784, convert_element_type_777, convert_element_type_778, sum_97, sum_98, convert_element_type_770, convert_element_type_771, convert_element_type_760, convert_element_type_761, sum_91, sum_92, convert_element_type_754, convert_element_type_753, convert_element_type_746, convert_element_type_747, sum_85, sum_86, convert_element_type_739, convert_element_type_740, convert_element_type_729, convert_element_type_730, sum_79, sum_80, convert_element_type_723, convert_element_type_722, convert_element_type_715, convert_element_type_716, sum_73, sum_74, convert_element_type_708, convert_element_type_709, convert_element_type_698, convert_element_type_699, sum_67, sum_68, convert_element_type_692, convert_element_type_691, convert_element_type_684, convert_element_type_685, sum_61, sum_62, convert_element_type_677, convert_element_type_678, convert_element_type_667, convert_element_type_668, sum_55, sum_56, convert_element_type_661, convert_element_type_660, convert_element_type_653, convert_element_type_654, sum_49, sum_50, convert_element_type_646, convert_element_type_647, convert_element_type_636, convert_element_type_637, sum_43, sum_44, convert_element_type_630, convert_element_type_629, convert_element_type_622, convert_element_type_623, sum_37, sum_38, convert_element_type_615, convert_element_type_616, convert_element_type_605, convert_element_type_606, sum_31, sum_32, convert_element_type_599, convert_element_type_598, convert_element_type_591, convert_element_type_592, sum_25, sum_26, convert_element_type_584, convert_element_type_585, convert_element_type_574, convert_element_type_575, sum_19, sum_20, convert_element_type_568, convert_element_type_567, convert_element_type_560, convert_element_type_561, sum_13, sum_14, convert_element_type_553, convert_element_type_554, convert_element_type_543, convert_element_type_544, sum_7, sum_8, convert_element_type_537)

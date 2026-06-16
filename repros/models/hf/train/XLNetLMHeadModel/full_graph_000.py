class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[16, 512][512, 1]cuda:0", primals_2: "f32[32000, 1024][1024, 1]cuda:0", primals_3: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_4: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_5: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_6: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_7: "f32[16, 64][64, 1]cuda:0", primals_8: "f32[16, 64][64, 1]cuda:0", primals_9: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_10: "f32[1024][1]cuda:0", primals_11: "f32[1024][1]cuda:0", primals_12: "f32[4096, 1024][1024, 1]cuda:0", primals_13: "f32[4096][1]cuda:0", primals_14: "f32[1024, 4096][4096, 1]cuda:0", primals_15: "f32[1024][1]cuda:0", primals_16: "f32[1024][1]cuda:0", primals_17: "f32[1024][1]cuda:0", primals_18: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_19: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_20: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_21: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_22: "f32[16, 64][64, 1]cuda:0", primals_23: "f32[16, 64][64, 1]cuda:0", primals_24: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_25: "f32[1024][1]cuda:0", primals_26: "f32[1024][1]cuda:0", primals_27: "f32[4096, 1024][1024, 1]cuda:0", primals_28: "f32[4096][1]cuda:0", primals_29: "f32[1024, 4096][4096, 1]cuda:0", primals_30: "f32[1024][1]cuda:0", primals_31: "f32[1024][1]cuda:0", primals_32: "f32[1024][1]cuda:0", primals_33: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_34: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_35: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_36: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_37: "f32[16, 64][64, 1]cuda:0", primals_38: "f32[16, 64][64, 1]cuda:0", primals_39: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_40: "f32[1024][1]cuda:0", primals_41: "f32[1024][1]cuda:0", primals_42: "f32[4096, 1024][1024, 1]cuda:0", primals_43: "f32[4096][1]cuda:0", primals_44: "f32[1024, 4096][4096, 1]cuda:0", primals_45: "f32[1024][1]cuda:0", primals_46: "f32[1024][1]cuda:0", primals_47: "f32[1024][1]cuda:0", primals_48: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_49: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_50: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_51: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_52: "f32[16, 64][64, 1]cuda:0", primals_53: "f32[16, 64][64, 1]cuda:0", primals_54: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_55: "f32[1024][1]cuda:0", primals_56: "f32[1024][1]cuda:0", primals_57: "f32[4096, 1024][1024, 1]cuda:0", primals_58: "f32[4096][1]cuda:0", primals_59: "f32[1024, 4096][4096, 1]cuda:0", primals_60: "f32[1024][1]cuda:0", primals_61: "f32[1024][1]cuda:0", primals_62: "f32[1024][1]cuda:0", primals_63: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_64: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_65: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_66: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_67: "f32[16, 64][64, 1]cuda:0", primals_68: "f32[16, 64][64, 1]cuda:0", primals_69: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_70: "f32[1024][1]cuda:0", primals_71: "f32[1024][1]cuda:0", primals_72: "f32[4096, 1024][1024, 1]cuda:0", primals_73: "f32[4096][1]cuda:0", primals_74: "f32[1024, 4096][4096, 1]cuda:0", primals_75: "f32[1024][1]cuda:0", primals_76: "f32[1024][1]cuda:0", primals_77: "f32[1024][1]cuda:0", primals_78: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_79: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_80: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_81: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_82: "f32[16, 64][64, 1]cuda:0", primals_83: "f32[16, 64][64, 1]cuda:0", primals_84: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_85: "f32[1024][1]cuda:0", primals_86: "f32[1024][1]cuda:0", primals_87: "f32[4096, 1024][1024, 1]cuda:0", primals_88: "f32[4096][1]cuda:0", primals_89: "f32[1024, 4096][4096, 1]cuda:0", primals_90: "f32[1024][1]cuda:0", primals_91: "f32[1024][1]cuda:0", primals_92: "f32[1024][1]cuda:0", primals_93: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_94: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_95: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_96: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_97: "f32[16, 64][64, 1]cuda:0", primals_98: "f32[16, 64][64, 1]cuda:0", primals_99: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_100: "f32[1024][1]cuda:0", primals_101: "f32[1024][1]cuda:0", primals_102: "f32[4096, 1024][1024, 1]cuda:0", primals_103: "f32[4096][1]cuda:0", primals_104: "f32[1024, 4096][4096, 1]cuda:0", primals_105: "f32[1024][1]cuda:0", primals_106: "f32[1024][1]cuda:0", primals_107: "f32[1024][1]cuda:0", primals_108: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_109: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_110: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_111: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_112: "f32[16, 64][64, 1]cuda:0", primals_113: "f32[16, 64][64, 1]cuda:0", primals_114: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_115: "f32[1024][1]cuda:0", primals_116: "f32[1024][1]cuda:0", primals_117: "f32[4096, 1024][1024, 1]cuda:0", primals_118: "f32[4096][1]cuda:0", primals_119: "f32[1024, 4096][4096, 1]cuda:0", primals_120: "f32[1024][1]cuda:0", primals_121: "f32[1024][1]cuda:0", primals_122: "f32[1024][1]cuda:0", primals_123: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_124: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_125: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_126: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_127: "f32[16, 64][64, 1]cuda:0", primals_128: "f32[16, 64][64, 1]cuda:0", primals_129: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_130: "f32[1024][1]cuda:0", primals_131: "f32[1024][1]cuda:0", primals_132: "f32[4096, 1024][1024, 1]cuda:0", primals_133: "f32[4096][1]cuda:0", primals_134: "f32[1024, 4096][4096, 1]cuda:0", primals_135: "f32[1024][1]cuda:0", primals_136: "f32[1024][1]cuda:0", primals_137: "f32[1024][1]cuda:0", primals_138: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_139: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_140: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_141: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_142: "f32[16, 64][64, 1]cuda:0", primals_143: "f32[16, 64][64, 1]cuda:0", primals_144: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_145: "f32[1024][1]cuda:0", primals_146: "f32[1024][1]cuda:0", primals_147: "f32[4096, 1024][1024, 1]cuda:0", primals_148: "f32[4096][1]cuda:0", primals_149: "f32[1024, 4096][4096, 1]cuda:0", primals_150: "f32[1024][1]cuda:0", primals_151: "f32[1024][1]cuda:0", primals_152: "f32[1024][1]cuda:0", primals_153: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_154: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_155: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_156: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_157: "f32[16, 64][64, 1]cuda:0", primals_158: "f32[16, 64][64, 1]cuda:0", primals_159: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_160: "f32[1024][1]cuda:0", primals_161: "f32[1024][1]cuda:0", primals_162: "f32[4096, 1024][1024, 1]cuda:0", primals_163: "f32[4096][1]cuda:0", primals_164: "f32[1024, 4096][4096, 1]cuda:0", primals_165: "f32[1024][1]cuda:0", primals_166: "f32[1024][1]cuda:0", primals_167: "f32[1024][1]cuda:0", primals_168: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_169: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_170: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_171: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_172: "f32[16, 64][64, 1]cuda:0", primals_173: "f32[16, 64][64, 1]cuda:0", primals_174: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_175: "f32[1024][1]cuda:0", primals_176: "f32[1024][1]cuda:0", primals_177: "f32[4096, 1024][1024, 1]cuda:0", primals_178: "f32[4096][1]cuda:0", primals_179: "f32[1024, 4096][4096, 1]cuda:0", primals_180: "f32[1024][1]cuda:0", primals_181: "f32[1024][1]cuda:0", primals_182: "f32[1024][1]cuda:0", primals_183: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_184: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_185: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_186: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_187: "f32[16, 64][64, 1]cuda:0", primals_188: "f32[16, 64][64, 1]cuda:0", primals_189: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_190: "f32[1024][1]cuda:0", primals_191: "f32[1024][1]cuda:0", primals_192: "f32[4096, 1024][1024, 1]cuda:0", primals_193: "f32[4096][1]cuda:0", primals_194: "f32[1024, 4096][4096, 1]cuda:0", primals_195: "f32[1024][1]cuda:0", primals_196: "f32[1024][1]cuda:0", primals_197: "f32[1024][1]cuda:0", primals_198: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_199: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_200: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_201: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_202: "f32[16, 64][64, 1]cuda:0", primals_203: "f32[16, 64][64, 1]cuda:0", primals_204: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_205: "f32[1024][1]cuda:0", primals_206: "f32[1024][1]cuda:0", primals_207: "f32[4096, 1024][1024, 1]cuda:0", primals_208: "f32[4096][1]cuda:0", primals_209: "f32[1024, 4096][4096, 1]cuda:0", primals_210: "f32[1024][1]cuda:0", primals_211: "f32[1024][1]cuda:0", primals_212: "f32[1024][1]cuda:0", primals_213: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_214: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_215: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_216: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_217: "f32[16, 64][64, 1]cuda:0", primals_218: "f32[16, 64][64, 1]cuda:0", primals_219: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_220: "f32[1024][1]cuda:0", primals_221: "f32[1024][1]cuda:0", primals_222: "f32[4096, 1024][1024, 1]cuda:0", primals_223: "f32[4096][1]cuda:0", primals_224: "f32[1024, 4096][4096, 1]cuda:0", primals_225: "f32[1024][1]cuda:0", primals_226: "f32[1024][1]cuda:0", primals_227: "f32[1024][1]cuda:0", primals_228: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_229: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_230: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_231: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_232: "f32[16, 64][64, 1]cuda:0", primals_233: "f32[16, 64][64, 1]cuda:0", primals_234: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_235: "f32[1024][1]cuda:0", primals_236: "f32[1024][1]cuda:0", primals_237: "f32[4096, 1024][1024, 1]cuda:0", primals_238: "f32[4096][1]cuda:0", primals_239: "f32[1024, 4096][4096, 1]cuda:0", primals_240: "f32[1024][1]cuda:0", primals_241: "f32[1024][1]cuda:0", primals_242: "f32[1024][1]cuda:0", primals_243: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_244: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_245: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_246: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_247: "f32[16, 64][64, 1]cuda:0", primals_248: "f32[16, 64][64, 1]cuda:0", primals_249: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_250: "f32[1024][1]cuda:0", primals_251: "f32[1024][1]cuda:0", primals_252: "f32[4096, 1024][1024, 1]cuda:0", primals_253: "f32[4096][1]cuda:0", primals_254: "f32[1024, 4096][4096, 1]cuda:0", primals_255: "f32[1024][1]cuda:0", primals_256: "f32[1024][1]cuda:0", primals_257: "f32[1024][1]cuda:0", primals_258: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_259: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_260: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_261: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_262: "f32[16, 64][64, 1]cuda:0", primals_263: "f32[16, 64][64, 1]cuda:0", primals_264: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_265: "f32[1024][1]cuda:0", primals_266: "f32[1024][1]cuda:0", primals_267: "f32[4096, 1024][1024, 1]cuda:0", primals_268: "f32[4096][1]cuda:0", primals_269: "f32[1024, 4096][4096, 1]cuda:0", primals_270: "f32[1024][1]cuda:0", primals_271: "f32[1024][1]cuda:0", primals_272: "f32[1024][1]cuda:0", primals_273: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_274: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_275: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_276: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_277: "f32[16, 64][64, 1]cuda:0", primals_278: "f32[16, 64][64, 1]cuda:0", primals_279: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_280: "f32[1024][1]cuda:0", primals_281: "f32[1024][1]cuda:0", primals_282: "f32[4096, 1024][1024, 1]cuda:0", primals_283: "f32[4096][1]cuda:0", primals_284: "f32[1024, 4096][4096, 1]cuda:0", primals_285: "f32[1024][1]cuda:0", primals_286: "f32[1024][1]cuda:0", primals_287: "f32[1024][1]cuda:0", primals_288: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_289: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_290: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_291: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_292: "f32[16, 64][64, 1]cuda:0", primals_293: "f32[16, 64][64, 1]cuda:0", primals_294: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_295: "f32[1024][1]cuda:0", primals_296: "f32[1024][1]cuda:0", primals_297: "f32[4096, 1024][1024, 1]cuda:0", primals_298: "f32[4096][1]cuda:0", primals_299: "f32[1024, 4096][4096, 1]cuda:0", primals_300: "f32[1024][1]cuda:0", primals_301: "f32[1024][1]cuda:0", primals_302: "f32[1024][1]cuda:0", primals_303: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_304: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_305: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_306: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_307: "f32[16, 64][64, 1]cuda:0", primals_308: "f32[16, 64][64, 1]cuda:0", primals_309: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_310: "f32[1024][1]cuda:0", primals_311: "f32[1024][1]cuda:0", primals_312: "f32[4096, 1024][1024, 1]cuda:0", primals_313: "f32[4096][1]cuda:0", primals_314: "f32[1024, 4096][4096, 1]cuda:0", primals_315: "f32[1024][1]cuda:0", primals_316: "f32[1024][1]cuda:0", primals_317: "f32[1024][1]cuda:0", primals_318: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_319: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_320: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_321: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_322: "f32[16, 64][64, 1]cuda:0", primals_323: "f32[16, 64][64, 1]cuda:0", primals_324: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_325: "f32[1024][1]cuda:0", primals_326: "f32[1024][1]cuda:0", primals_327: "f32[4096, 1024][1024, 1]cuda:0", primals_328: "f32[4096][1]cuda:0", primals_329: "f32[1024, 4096][4096, 1]cuda:0", primals_330: "f32[1024][1]cuda:0", primals_331: "f32[1024][1]cuda:0", primals_332: "f32[1024][1]cuda:0", primals_333: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_334: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_335: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_336: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_337: "f32[16, 64][64, 1]cuda:0", primals_338: "f32[16, 64][64, 1]cuda:0", primals_339: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_340: "f32[1024][1]cuda:0", primals_341: "f32[1024][1]cuda:0", primals_342: "f32[4096, 1024][1024, 1]cuda:0", primals_343: "f32[4096][1]cuda:0", primals_344: "f32[1024, 4096][4096, 1]cuda:0", primals_345: "f32[1024][1]cuda:0", primals_346: "f32[1024][1]cuda:0", primals_347: "f32[1024][1]cuda:0", primals_348: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_349: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_350: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_351: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_352: "f32[16, 64][64, 1]cuda:0", primals_353: "f32[16, 64][64, 1]cuda:0", primals_354: "f32[1024, 16, 64][1024, 64, 1]cuda:0", primals_355: "f32[1024][1]cuda:0", primals_356: "f32[1024][1]cuda:0", primals_357: "f32[4096, 1024][1024, 1]cuda:0", primals_358: "f32[4096][1]cuda:0", primals_359: "f32[1024, 4096][4096, 1]cuda:0", primals_360: "f32[1024][1]cuda:0", primals_361: "f32[1024][1]cuda:0", primals_362: "f32[1024][1]cuda:0", primals_363: "f32[32000][1]cuda:0", primals_364: "i64[16, 512][512, 1]cuda:0"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1046 in forward, code: input_ids = input_ids.transpose(0, 1).contiguous()
        permute: "i64[512, 16][1, 512]cuda:0" = torch.ops.aten.permute.default(primals_1, [1, 0]);  primals_1 = None
        clone: "i64[512, 16][16, 1]cuda:0" = torch.ops.aten.clone.default(permute, memory_format = torch.contiguous_format);  permute = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1115 in forward, code: word_emb_k = self.word_embedding(input_ids)
        embedding: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.embedding.default(primals_2, clone)

        # No stacktrace found for following nodes
        inductor_seeds_default: "i64[99][1]cuda:0" = torch.ops.prims.inductor_seeds.default(99, device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1116 in forward, code: output_h = self.dropout(word_emb_k)
        inductor_lookup_seed_default: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 0)
        inductor_random_default_98: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default, 'rand');  inductor_lookup_seed_default = None
        gt: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_98, 0.1);  inductor_random_default_98 = None
        mul: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:942 in relative_positional_encoding, code: freq_seq = torch.arange(0, self.d_model, 2.0, dtype=torch.int64, device=device).float()
        iota: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 2, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type: "f32[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(iota, torch.float32);  iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:943 in relative_positional_encoding, code: inv_freq = 1 / torch.pow(10000, (freq_seq / self.d_model))
        div: "f32[512][1]cuda:0" = torch.ops.aten.div.Tensor(convert_element_type, 1024);  convert_element_type = None
        pow_1: "f32[512][1]cuda:0" = torch.ops.aten.pow.Scalar(10000, div);  div = None
        reciprocal: "f32[512][1]cuda:0" = torch.ops.aten.reciprocal.default(pow_1);  pow_1 = None
        mul_2: "f32[512][1]cuda:0" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:971 in relative_positional_encoding, code: fwd_pos_seq = torch.arange(beg, end, -1.0, dtype=torch.int64, device=device).float()
        iota_1: "i64[1024][1]cuda:0" = torch.ops.prims.iota.default(1024, start = 512, step = -1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        convert_element_type_1: "f32[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(iota_1, torch.float32);  iota_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:931 in positional_embedding, code: sinusoid_inp = torch.einsum("i,d->id", pos_seq, inv_freq)
        convert_element_type_2: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1, torch.bfloat16);  convert_element_type_1 = None
        convert_element_type_3: "bf16[512][1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_2, torch.bfloat16);  mul_2 = None
        unsqueeze: "bf16[1024, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_2, 1);  convert_element_type_2 = None
        permute_1: "bf16[1024, 1][1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze, [0, 1]);  unsqueeze = None
        unsqueeze_1: "bf16[512, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_3, 1);  convert_element_type_3 = None
        permute_2: "bf16[1, 512][1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_1, [1, 0]);  unsqueeze_1 = None
        mul_3: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.mul.Tensor(permute_1, permute_2);  permute_1 = permute_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:932 in positional_embedding, code: pos_emb = torch.cat([torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)], dim=-1)
        sin: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.sin.default(mul_3)
        cos: "bf16[1024, 512][512, 1]cuda:0" = torch.ops.aten.cos.default(mul_3);  mul_3 = None
        cat: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.cat.default([sin, cos], -1);  sin = cos = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:933 in positional_embedding, code: pos_emb = pos_emb[:, None, :]
        unsqueeze_2: "bf16[1024, 1, 1024][1024, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(cat, 1);  cat = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:936 in positional_embedding, code: pos_emb = pos_emb.expand(-1, bsz, -1)
        expand: "bf16[1024, 16, 1024][1024, 0, 1]cuda:0" = torch.ops.aten.expand.default(unsqueeze_2, [-1, 16, -1]);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1143 in forward, code: pos_emb = self.dropout(pos_emb)
        inductor_lookup_seed_default_1: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 1)
        inductor_random_default_97: "f32[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([1024, 16, 1024], inductor_lookup_seed_default_1, 'rand');  inductor_lookup_seed_default_1 = None
        convert_element_type_default_144: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_97, torch.bfloat16);  inductor_random_default_97 = None
        gt_1: "b8[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_144, 0.1);  convert_element_type_default_144 = None
        mul_4: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_1, expand);  gt_1 = expand = None
        mul_5: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_4, 1.1111111111111112);  mul_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_4: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_1, torch.bfloat16)
        convert_element_type_5: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_3, torch.bfloat16);  primals_3 = None
        unsqueeze_3: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_4, 3);  convert_element_type_4 = None
        unsqueeze_4: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 4);  unsqueeze_3 = None
        unsqueeze_5: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_5, 3);  convert_element_type_5 = None
        unsqueeze_6: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_5, 4);  unsqueeze_5 = None
        view: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_4, [1, 8192, 1024]);  unsqueeze_4 = None
        view_1: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_6, [1, 1024, 1024]);  unsqueeze_6 = None
        squeeze_dim_670: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view, 0)
        squeeze_dim_671: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_1, 0)
        mm_default_335: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_670, squeeze_dim_671);  squeeze_dim_671 = None
        unsqueeze_default_335: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_335, 0);  mm_default_335 = None
        view_2: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_335, [512, 16, 1, 16, 64]);  unsqueeze_default_335 = None
        permute_7: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 4, 2]);  view_2 = None
        view_3: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_7, [512, 16, 16, 64]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_9: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_4, torch.bfloat16);  primals_4 = None
        unsqueeze_9: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_9, 3);  convert_element_type_9 = None
        unsqueeze_10: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 4);  unsqueeze_9 = None
        view_5: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_10, [1, 1024, 1024]);  unsqueeze_10 = None
        squeeze_dim_669: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_5, 0)
        mm_default_334: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_670, squeeze_dim_669);  squeeze_dim_669 = None
        unsqueeze_default_334: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_334, 0);  mm_default_334 = None
        view_6: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_334, [512, 16, 1, 16, 64]);  unsqueeze_default_334 = None
        permute_12: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_6, [0, 1, 3, 4, 2]);  view_6 = None
        view_7: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_12, [512, 16, 16, 64]);  permute_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_13: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_5, torch.bfloat16);  primals_5 = None
        unsqueeze_13: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_13, 3);  convert_element_type_13 = None
        unsqueeze_14: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_13, 4);  unsqueeze_13 = None
        view_9: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_14, [1, 1024, 1024]);  unsqueeze_14 = None
        squeeze_dim_667: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_9, 0)
        mm_default_333: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_670, squeeze_dim_667);  squeeze_dim_670 = squeeze_dim_667 = None
        unsqueeze_default_333: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_333, 0);  mm_default_333 = None
        view_10: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_333, [512, 16, 1, 16, 64]);  unsqueeze_default_333 = None
        permute_17: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_10, [0, 1, 3, 4, 2]);  view_10 = None
        view_11: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_17, [512, 16, 16, 64]);  permute_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_default_23: "bf16[1024, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_5, torch.bfloat16);  mul_5 = None
        convert_element_type_18: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_6, torch.bfloat16);  primals_6 = None
        unsqueeze_15: "bf16[1024, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_default_23, 3);  convert_element_type_default_23 = None
        unsqueeze_16: "bf16[1024, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_15, 4);  unsqueeze_15 = None
        unsqueeze_17: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_18, 3);  convert_element_type_18 = None
        unsqueeze_18: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_17, 4);  unsqueeze_17 = None
        view_12: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_16, [1, 16384, 1024]);  unsqueeze_16 = None
        view_13: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_18, [1, 1024, 1024]);  unsqueeze_18 = None
        squeeze_dim_664: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_12, 0)
        squeeze_dim_665: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_13, 0);  view_13 = None
        mm_default_332: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_665);  squeeze_dim_665 = None
        unsqueeze_default_332: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_332, 0);  mm_default_332 = None
        view_14: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_332, [1024, 16, 1, 16, 64]);  unsqueeze_default_332 = None
        permute_22: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_14, [0, 1, 3, 4, 2]);  view_14 = None
        view_15: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_22, [1024, 16, 16, 64]);  permute_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_3, primals_7);  primals_7 = None
        convert_element_type_21: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add, torch.bfloat16);  add = None
        unsqueeze_19: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_21, 4);  convert_element_type_21 = None
        permute_23: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_19, [1, 2, 0, 4, 3]);  unsqueeze_19 = None
        unsqueeze_20: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_7, 4);  view_7 = None
        permute_24: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_20, [1, 2, 4, 0, 3]);  unsqueeze_20 = None
        permute_25: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_23, [0, 1, 2, 4, 3]);  permute_23 = None
        view_16: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_25, [256, 512, 64]);  permute_25 = None
        permute_26: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_24, [0, 1, 4, 3, 2]);  permute_24 = None
        view_17: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_26, [256, 64, 512]);  permute_26 = None
        bmm_4: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_16, view_17)
        view_18: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_4, [16, 16, 512, 1, 512]);  bmm_4 = None
        permute_27: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_18, [0, 1, 2, 4, 3]);  view_18 = None
        view_19: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_27, [16, 16, 512, 512]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_1: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_3, primals_8);  view_3 = primals_8 = None
        convert_element_type_24: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_1, torch.bfloat16);  add_1 = None
        unsqueeze_21: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_24, 4);  convert_element_type_24 = None
        permute_28: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_21, [1, 2, 0, 4, 3]);  unsqueeze_21 = None
        unsqueeze_22: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_15, 4);  view_15 = None
        permute_29: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_22, [1, 2, 4, 0, 3]);  unsqueeze_22 = None
        permute_30: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_28, [0, 1, 2, 4, 3]);  permute_28 = None
        view_20: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_30, [256, 512, 64]);  permute_30 = None
        permute_31: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_29, [0, 1, 4, 3, 2]);  permute_29 = None
        view_21: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_31, [256, 64, 1024]);  permute_31 = None
        bmm_5: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_20, view_21)
        view_22: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_5, [16, 16, 512, 1, 1024]);  bmm_5 = None
        permute_32: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_22, [0, 1, 2, 4, 3]);  view_22 = None
        view_23: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_32, [16, 16, 512, 1024]);  permute_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_24: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_23, [16, 16, 1024, 512]);  view_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_1: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_24, 2, 1, 9223372036854775807);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_25: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_1, [16, 16, 512, 1023]);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        iota_2: "i64[512][1]cuda:0" = torch.ops.prims.iota.default(512, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        index: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_25, [None, None, None, iota_2]);  view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_2: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_19, index);  view_19 = index = None
        add_3: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_2, 0);  add_2 = None

        # No stacktrace found for following nodes
        mul_tensor_92: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_3, 0.125)
        convert_element_type_default_70: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_92, torch.float32);  mul_tensor_92 = None
        convert_element_type_default_71: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_3, torch.float32)
        mul_tensor_93: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_71, 1);  convert_element_type_default_71 = None
        amax_default_46: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_93, [3], True)
        sub_tensor_46: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_93, amax_default_46);  mul_tensor_93 = None
        mul_tensor_94: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_46, 0.125);  sub_tensor_46 = None
        amax_default_47: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_70, [3], True)
        sub_tensor_47: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_70, amax_default_47)
        abs_default_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_70)
        ne_scalar_23: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_23, inf);  abs_default_23 = None
        eq_tensor_24: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_70, convert_element_type_default_70);  convert_element_type_default_70 = None
        mul_tensor_95: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_24, ne_scalar_23);  eq_tensor_24 = ne_scalar_23 = None
        logical_not_default_46: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_95);  mul_tensor_95 = None
        any_dims_23: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_46, [3], True);  logical_not_default_46 = None
        logical_not_default_47: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_23);  any_dims_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_47, mul_tensor_94, sub_tensor_47);  mul_tensor_94 = sub_tensor_47 = None
        exp: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_24);  where_self_24 = None
        sum_1: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp, [3], True)
        div_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_2: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 2)
        inductor_random_default_96: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_2, 'rand');  inductor_lookup_seed_default_2 = None
        gt_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_96, 0.1);  inductor_random_default_96 = None
        mul_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_2, div_1);  div_1 = None
        mul_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_7, 1.1111111111111112);  mul_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_28: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_8, torch.bfloat16);  mul_8 = None
        unsqueeze_23: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_28, 4);  convert_element_type_28 = None
        unsqueeze_24: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_11, 4);  view_11 = None
        permute_34: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_24, [4, 1, 2, 3, 0]);  unsqueeze_24 = None
        view_26: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_23, [256, 512, 512]);  unsqueeze_23 = None
        permute_36: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_34, [1, 2, 4, 3, 0]);  permute_34 = None
        view_27: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_36, [256, 512, 64]);  permute_36 = None
        bmm_6: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_26, view_27)
        view_28: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_6, [16, 16, 512, 1, 64]);  bmm_6 = None
        permute_37: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_28, [2, 0, 1, 4, 3]);  view_28 = None
        view_29: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_37, [512, 16, 16, 64]);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_31: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_9, torch.bfloat16);  primals_9 = None
        unsqueeze_25: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_29, 4);  view_29 = None
        permute_38: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_25, [0, 1, 4, 3, 2]);  unsqueeze_25 = None
        unsqueeze_26: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_31, 3);  convert_element_type_31 = None
        unsqueeze_27: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_26, 4);  unsqueeze_26 = None
        permute_39: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_27, [3, 4, 0, 2, 1]);  unsqueeze_27 = None
        permute_40: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_38, [0, 1, 3, 4, 2]);  permute_38 = None
        clone_1: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_40, memory_format = torch.contiguous_format);  permute_40 = None
        view_30: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_1, [1, 8192, 1024]);  clone_1 = None
        permute_41: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_39, [3, 4, 2, 0, 1]);  permute_39 = None
        clone_2: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_41, memory_format = torch.contiguous_format);  permute_41 = None
        view_31: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_2, [1, 1024, 1024]);  clone_2 = None
        squeeze_dim_662: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_30, 0)
        squeeze_dim_663: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_31, 0)
        mm_default_331: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_662, squeeze_dim_663);  squeeze_dim_662 = squeeze_dim_663 = None
        unsqueeze_default_331: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_331, 0);  mm_default_331 = None
        view_32: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_331, [512, 16, 1, 1, 1024]);  unsqueeze_default_331 = None
        permute_42: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_32, [0, 1, 4, 2, 3]);  view_32 = None
        view_33: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_42, [512, 16, 1024]);  permute_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_3: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 3)
        inductor_random_default_95: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_3, 'rand');  inductor_lookup_seed_default_3 = None
        convert_element_type_default_143: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_95, torch.bfloat16);  inductor_random_default_95 = None
        gt_3: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_143, 0.1);  convert_element_type_default_143 = None
        mul_9: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_3, view_33);  view_33 = None
        mul_10: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_9, 1.1111111111111112);  mul_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_4: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_10, mul_1);  mul_10 = mul_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean = torch.ops.aten.var_mean.correction(add_4, [2], correction = 0, keepdim = True)
        getitem: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean[0]
        getitem_1: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean[1];  var_mean = None
        add_5: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_5);  add_5 = None
        sub_1: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_4, getitem_1);  add_4 = getitem_1 = None
        mul_11: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = None
        mul_12: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_11, primals_10)
        add_6: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_12, primals_11);  mul_12 = primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_34: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_13, torch.bfloat16);  primals_13 = None
        convert_element_type_35: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_12, torch.bfloat16);  primals_12 = None
        convert_element_type_36: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16)
        view_34: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_36, [8192, 1024]);  convert_element_type_36 = None
        permute_43: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_35, [1, 0]);  convert_element_type_35 = None
        addmm: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_34, view_34, permute_43);  convert_element_type_34 = None
        view_35: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_40: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_35, torch.float32);  view_35 = None
        mul_13: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, 0.5)
        mul_14: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_40, 0.7071067811865476);  convert_element_type_40 = None
        erf: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_14);  mul_14 = None
        add_7: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_15: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_13, add_7);  mul_13 = add_7 = None
        convert_element_type_41: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_15, torch.bfloat16);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_4: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 4)
        inductor_random_default_94: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_4, 'rand');  inductor_lookup_seed_default_4 = None
        convert_element_type_default_142: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_94, torch.bfloat16);  inductor_random_default_94 = None
        gt_4: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_142, 0.1);  convert_element_type_default_142 = None
        mul_16: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_4, convert_element_type_41);  convert_element_type_41 = None
        mul_17: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_16, 1.1111111111111112);  mul_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_42: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_15, torch.bfloat16);  primals_15 = None
        convert_element_type_43: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_14, torch.bfloat16);  primals_14 = None
        view_36: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_17, [8192, 4096]);  mul_17 = None
        permute_44: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_43, [1, 0]);  convert_element_type_43 = None
        addmm_1: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_42, view_36, permute_44);  convert_element_type_42 = None
        view_37: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_1, [512, 16, 1024]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_5: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 5)
        inductor_random_default_93: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_5, 'rand');  inductor_lookup_seed_default_5 = None
        convert_element_type_default_141: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_93, torch.bfloat16);  inductor_random_default_93 = None
        gt_5: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_141, 0.1);  convert_element_type_default_141 = None
        mul_18: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_5, view_37);  view_37 = None
        mul_19: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_18, 1.1111111111111112);  mul_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_8: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_19, add_6);  mul_19 = add_6 = None
        var_mean_1 = torch.ops.aten.var_mean.correction(add_8, [2], correction = 0, keepdim = True)
        getitem_2: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_1[0]
        getitem_3: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_1[1];  var_mean_1 = None
        add_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_2, 1e-12);  getitem_2 = None
        rsqrt_1: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_9);  add_9 = None
        sub_2: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_8, getitem_3);  add_8 = getitem_3 = None
        mul_20: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = None
        mul_21: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_20, primals_16)
        add_10: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_21, primals_17);  mul_21 = primals_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_47: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16)
        convert_element_type_48: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_18, torch.bfloat16);  primals_18 = None
        unsqueeze_28: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_47, 3);  convert_element_type_47 = None
        unsqueeze_29: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_28, 4);  unsqueeze_28 = None
        unsqueeze_30: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_48, 3);  convert_element_type_48 = None
        unsqueeze_31: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_30, 4);  unsqueeze_30 = None
        view_38: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_29, [1, 8192, 1024]);  unsqueeze_29 = None
        view_39: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_31, [1, 1024, 1024]);  unsqueeze_31 = None
        squeeze_dim_660: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_38, 0)
        squeeze_dim_661: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_39, 0)
        mm_default_330: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_660, squeeze_dim_661);  squeeze_dim_661 = None
        unsqueeze_default_330: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_330, 0);  mm_default_330 = None
        view_40: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_330, [512, 16, 1, 16, 64]);  unsqueeze_default_330 = None
        permute_49: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_40, [0, 1, 3, 4, 2]);  view_40 = None
        view_41: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_49, [512, 16, 16, 64]);  permute_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_52: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_19, torch.bfloat16);  primals_19 = None
        unsqueeze_34: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_52, 3);  convert_element_type_52 = None
        unsqueeze_35: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_34, 4);  unsqueeze_34 = None
        view_43: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_35, [1, 1024, 1024]);  unsqueeze_35 = None
        squeeze_dim_659: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_43, 0)
        mm_default_329: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_660, squeeze_dim_659);  squeeze_dim_659 = None
        unsqueeze_default_329: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_329, 0);  mm_default_329 = None
        view_44: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_329, [512, 16, 1, 16, 64]);  unsqueeze_default_329 = None
        permute_54: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_44, [0, 1, 3, 4, 2]);  view_44 = None
        view_45: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_54, [512, 16, 16, 64]);  permute_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_56: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_20, torch.bfloat16);  primals_20 = None
        unsqueeze_38: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_56, 3);  convert_element_type_56 = None
        unsqueeze_39: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_38, 4);  unsqueeze_38 = None
        view_47: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_39, [1, 1024, 1024]);  unsqueeze_39 = None
        squeeze_dim_657: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_47, 0)
        mm_default_328: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_660, squeeze_dim_657);  squeeze_dim_660 = squeeze_dim_657 = None
        unsqueeze_default_328: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_328, 0);  mm_default_328 = None
        view_48: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_328, [512, 16, 1, 16, 64]);  unsqueeze_default_328 = None
        permute_59: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_48, [0, 1, 3, 4, 2]);  view_48 = None
        view_49: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_59, [512, 16, 16, 64]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_61: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_21, torch.bfloat16);  primals_21 = None
        unsqueeze_42: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_61, 3);  convert_element_type_61 = None
        unsqueeze_43: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_42, 4);  unsqueeze_42 = None
        view_51: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_43, [1, 1024, 1024]);  unsqueeze_43 = None
        squeeze_dim_655: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_51, 0);  view_51 = None
        mm_default_327: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_655);  squeeze_dim_655 = None
        unsqueeze_default_327: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_327, 0);  mm_default_327 = None
        view_52: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_327, [1024, 16, 1, 16, 64]);  unsqueeze_default_327 = None
        permute_64: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_52, [0, 1, 3, 4, 2]);  view_52 = None
        view_53: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_64, [1024, 16, 16, 64]);  permute_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_11: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, primals_22);  primals_22 = None
        convert_element_type_64: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_11, torch.bfloat16);  add_11 = None
        unsqueeze_44: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_64, 4);  convert_element_type_64 = None
        permute_65: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_44, [1, 2, 0, 4, 3]);  unsqueeze_44 = None
        unsqueeze_45: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_45, 4);  view_45 = None
        permute_66: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_45, [1, 2, 4, 0, 3]);  unsqueeze_45 = None
        permute_67: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_65, [0, 1, 2, 4, 3]);  permute_65 = None
        view_54: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_67, [256, 512, 64]);  permute_67 = None
        permute_68: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_66, [0, 1, 4, 3, 2]);  permute_66 = None
        view_55: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_68, [256, 64, 512]);  permute_68 = None
        bmm_12: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_54, view_55)
        view_56: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_12, [16, 16, 512, 1, 512]);  bmm_12 = None
        permute_69: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_56, [0, 1, 2, 4, 3]);  view_56 = None
        view_57: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_69, [16, 16, 512, 512]);  permute_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_12: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_41, primals_23);  view_41 = primals_23 = None
        convert_element_type_67: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16);  add_12 = None
        unsqueeze_46: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_67, 4);  convert_element_type_67 = None
        permute_70: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_46, [1, 2, 0, 4, 3]);  unsqueeze_46 = None
        unsqueeze_47: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_53, 4);  view_53 = None
        permute_71: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_47, [1, 2, 4, 0, 3]);  unsqueeze_47 = None
        permute_72: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_70, [0, 1, 2, 4, 3]);  permute_70 = None
        view_58: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_72, [256, 512, 64]);  permute_72 = None
        permute_73: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_71, [0, 1, 4, 3, 2]);  permute_71 = None
        view_59: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_73, [256, 64, 1024]);  permute_73 = None
        bmm_13: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_58, view_59)
        view_60: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_13, [16, 16, 512, 1, 1024]);  bmm_13 = None
        permute_74: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_60, [0, 1, 2, 4, 3]);  view_60 = None
        view_61: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_74, [16, 16, 512, 1024]);  permute_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_62: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_61, [16, 16, 1024, 512]);  view_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_2: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_62, 2, 1, 9223372036854775807);  view_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_63: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_2, [16, 16, 512, 1023]);  slice_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_1: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_63, [None, None, None, iota_2]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_13: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_57, index_1);  view_57 = index_1 = None
        add_14: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_13, 0);  add_13 = None

        # No stacktrace found for following nodes
        mul_tensor_88: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_14, 0.125)
        convert_element_type_default_68: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_88, torch.float32);  mul_tensor_88 = None
        convert_element_type_default_69: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_14, torch.float32)
        mul_tensor_89: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_69, 1);  convert_element_type_default_69 = None
        amax_default_44: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_89, [3], True)
        sub_tensor_44: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_89, amax_default_44);  mul_tensor_89 = None
        mul_tensor_90: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_44, 0.125);  sub_tensor_44 = None
        amax_default_45: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_68, [3], True)
        sub_tensor_45: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_68, amax_default_45)
        abs_default_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_68)
        ne_scalar_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_22, inf);  abs_default_22 = None
        eq_tensor_23: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_68, convert_element_type_default_68);  convert_element_type_default_68 = None
        mul_tensor_91: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_23, ne_scalar_22);  eq_tensor_23 = ne_scalar_22 = None
        logical_not_default_44: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_91);  mul_tensor_91 = None
        any_dims_22: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_44, [3], True);  logical_not_default_44 = None
        logical_not_default_45: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_22);  any_dims_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_45, mul_tensor_90, sub_tensor_45);  mul_tensor_90 = sub_tensor_45 = None
        exp_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_23);  where_self_23 = None
        sum_2: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_1, [3], True)
        div_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_6: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 6)
        inductor_random_default_92: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_6, 'rand');  inductor_lookup_seed_default_6 = None
        gt_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_92, 0.1);  inductor_random_default_92 = None
        mul_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_6, div_2);  div_2 = None
        mul_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_23, 1.1111111111111112);  mul_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_71: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_24, torch.bfloat16);  mul_24 = None
        unsqueeze_48: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_71, 4);  convert_element_type_71 = None
        unsqueeze_49: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_49, 4);  view_49 = None
        permute_76: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_49, [4, 1, 2, 3, 0]);  unsqueeze_49 = None
        view_64: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_48, [256, 512, 512]);  unsqueeze_48 = None
        permute_78: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_76, [1, 2, 4, 3, 0]);  permute_76 = None
        view_65: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_78, [256, 512, 64]);  permute_78 = None
        bmm_14: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_64, view_65)
        view_66: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_14, [16, 16, 512, 1, 64]);  bmm_14 = None
        permute_79: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_66, [2, 0, 1, 4, 3]);  view_66 = None
        view_67: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_79, [512, 16, 16, 64]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_74: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_24, torch.bfloat16);  primals_24 = None
        unsqueeze_50: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_67, 4);  view_67 = None
        permute_80: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_50, [0, 1, 4, 3, 2]);  unsqueeze_50 = None
        unsqueeze_51: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_74, 3);  convert_element_type_74 = None
        unsqueeze_52: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_51, 4);  unsqueeze_51 = None
        permute_81: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_52, [3, 4, 0, 2, 1]);  unsqueeze_52 = None
        permute_82: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_80, [0, 1, 3, 4, 2]);  permute_80 = None
        clone_3: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_82, memory_format = torch.contiguous_format);  permute_82 = None
        view_68: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_3, [1, 8192, 1024]);  clone_3 = None
        permute_83: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_81, [3, 4, 2, 0, 1]);  permute_81 = None
        clone_4: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_83, memory_format = torch.contiguous_format);  permute_83 = None
        view_69: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_4, [1, 1024, 1024]);  clone_4 = None
        squeeze_dim_652: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_68, 0)
        squeeze_dim_653: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_69, 0)
        mm_default_326: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_652, squeeze_dim_653);  squeeze_dim_652 = squeeze_dim_653 = None
        unsqueeze_default_326: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_326, 0);  mm_default_326 = None
        view_70: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_326, [512, 16, 1, 1, 1024]);  unsqueeze_default_326 = None
        permute_84: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_70, [0, 1, 4, 2, 3]);  view_70 = None
        view_71: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_84, [512, 16, 1024]);  permute_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_7: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 7)
        inductor_random_default_91: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_7, 'rand');  inductor_lookup_seed_default_7 = None
        convert_element_type_default_140: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_91, torch.bfloat16);  inductor_random_default_91 = None
        gt_7: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_140, 0.1);  convert_element_type_default_140 = None
        mul_25: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_7, view_71);  view_71 = None
        mul_26: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_25, 1.1111111111111112);  mul_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_15: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_26, add_10);  mul_26 = add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_15, [2], correction = 0, keepdim = True)
        getitem_4: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_2[0]
        getitem_5: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_2[1];  var_mean_2 = None
        add_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_4, 1e-12);  getitem_4 = None
        rsqrt_2: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_16);  add_16 = None
        sub_4: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_15, getitem_5);  add_15 = getitem_5 = None
        mul_27: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_2);  sub_4 = None
        mul_28: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_27, primals_25)
        add_17: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_28, primals_26);  mul_28 = primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_77: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_28, torch.bfloat16);  primals_28 = None
        convert_element_type_78: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_27, torch.bfloat16);  primals_27 = None
        convert_element_type_79: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_17, torch.bfloat16)
        view_72: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_79, [8192, 1024]);  convert_element_type_79 = None
        permute_85: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_78, [1, 0]);  convert_element_type_78 = None
        addmm_2: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_77, view_72, permute_85);  convert_element_type_77 = None
        view_73: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_2, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_83: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_73, torch.float32);  view_73 = None
        mul_29: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, 0.5)
        mul_30: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_83, 0.7071067811865476);  convert_element_type_83 = None
        erf_1: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_30);  mul_30 = None
        add_18: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_31: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_29, add_18);  mul_29 = add_18 = None
        convert_element_type_84: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_31, torch.bfloat16);  mul_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_8: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 8)
        inductor_random_default_90: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_8, 'rand');  inductor_lookup_seed_default_8 = None
        convert_element_type_default_139: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_90, torch.bfloat16);  inductor_random_default_90 = None
        gt_8: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_139, 0.1);  convert_element_type_default_139 = None
        mul_32: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_8, convert_element_type_84);  convert_element_type_84 = None
        mul_33: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_32, 1.1111111111111112);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_85: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_30, torch.bfloat16);  primals_30 = None
        convert_element_type_86: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_29, torch.bfloat16);  primals_29 = None
        view_74: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_33, [8192, 4096]);  mul_33 = None
        permute_86: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_86, [1, 0]);  convert_element_type_86 = None
        addmm_3: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_85, view_74, permute_86);  convert_element_type_85 = None
        view_75: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_3, [512, 16, 1024]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_9: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 9)
        inductor_random_default_89: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_9, 'rand');  inductor_lookup_seed_default_9 = None
        convert_element_type_default_138: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_89, torch.bfloat16);  inductor_random_default_89 = None
        gt_9: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_138, 0.1);  convert_element_type_default_138 = None
        mul_34: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_9, view_75);  view_75 = None
        mul_35: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_34, 1.1111111111111112);  mul_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_19: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_35, add_17);  mul_35 = add_17 = None
        var_mean_3 = torch.ops.aten.var_mean.correction(add_19, [2], correction = 0, keepdim = True)
        getitem_6: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_3[0]
        getitem_7: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_3[1];  var_mean_3 = None
        add_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_6, 1e-12);  getitem_6 = None
        rsqrt_3: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_20);  add_20 = None
        sub_5: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_19, getitem_7);  add_19 = getitem_7 = None
        mul_36: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = None
        mul_37: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_36, primals_31)
        add_21: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_37, primals_32);  mul_37 = primals_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_90: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_21, torch.bfloat16)
        convert_element_type_91: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_33, torch.bfloat16);  primals_33 = None
        unsqueeze_53: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_90, 3);  convert_element_type_90 = None
        unsqueeze_54: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_53, 4);  unsqueeze_53 = None
        unsqueeze_55: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_91, 3);  convert_element_type_91 = None
        unsqueeze_56: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_55, 4);  unsqueeze_55 = None
        view_76: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_54, [1, 8192, 1024]);  unsqueeze_54 = None
        view_77: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_56, [1, 1024, 1024]);  unsqueeze_56 = None
        squeeze_dim_650: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_76, 0)
        squeeze_dim_651: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_77, 0)
        mm_default_325: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_650, squeeze_dim_651);  squeeze_dim_651 = None
        unsqueeze_default_325: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_325, 0);  mm_default_325 = None
        view_78: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_325, [512, 16, 1, 16, 64]);  unsqueeze_default_325 = None
        permute_91: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_78, [0, 1, 3, 4, 2]);  view_78 = None
        view_79: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_91, [512, 16, 16, 64]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_95: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_34, torch.bfloat16);  primals_34 = None
        unsqueeze_59: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_95, 3);  convert_element_type_95 = None
        unsqueeze_60: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_59, 4);  unsqueeze_59 = None
        view_81: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_60, [1, 1024, 1024]);  unsqueeze_60 = None
        squeeze_dim_649: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_81, 0)
        mm_default_324: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_650, squeeze_dim_649);  squeeze_dim_649 = None
        unsqueeze_default_324: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_324, 0);  mm_default_324 = None
        view_82: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_324, [512, 16, 1, 16, 64]);  unsqueeze_default_324 = None
        permute_96: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_82, [0, 1, 3, 4, 2]);  view_82 = None
        view_83: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_96, [512, 16, 16, 64]);  permute_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_99: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_35, torch.bfloat16);  primals_35 = None
        unsqueeze_63: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_99, 3);  convert_element_type_99 = None
        unsqueeze_64: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_63, 4);  unsqueeze_63 = None
        view_85: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_64, [1, 1024, 1024]);  unsqueeze_64 = None
        squeeze_dim_647: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_85, 0)
        mm_default_323: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_650, squeeze_dim_647);  squeeze_dim_650 = squeeze_dim_647 = None
        unsqueeze_default_323: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_323, 0);  mm_default_323 = None
        view_86: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_323, [512, 16, 1, 16, 64]);  unsqueeze_default_323 = None
        permute_101: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_86, [0, 1, 3, 4, 2]);  view_86 = None
        view_87: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_101, [512, 16, 16, 64]);  permute_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_104: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_36, torch.bfloat16);  primals_36 = None
        unsqueeze_67: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_104, 3);  convert_element_type_104 = None
        unsqueeze_68: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_67, 4);  unsqueeze_67 = None
        view_89: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_68, [1, 1024, 1024]);  unsqueeze_68 = None
        squeeze_dim_645: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_89, 0);  view_89 = None
        mm_default_322: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_645);  squeeze_dim_645 = None
        unsqueeze_default_322: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_322, 0);  mm_default_322 = None
        view_90: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_322, [1024, 16, 1, 16, 64]);  unsqueeze_default_322 = None
        permute_106: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_90, [0, 1, 3, 4, 2]);  view_90 = None
        view_91: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_106, [1024, 16, 16, 64]);  permute_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_22: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_79, primals_37);  primals_37 = None
        convert_element_type_107: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16);  add_22 = None
        unsqueeze_69: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_107, 4);  convert_element_type_107 = None
        permute_107: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_69, [1, 2, 0, 4, 3]);  unsqueeze_69 = None
        unsqueeze_70: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_83, 4);  view_83 = None
        permute_108: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_70, [1, 2, 4, 0, 3]);  unsqueeze_70 = None
        permute_109: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_107, [0, 1, 2, 4, 3]);  permute_107 = None
        view_92: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_109, [256, 512, 64]);  permute_109 = None
        permute_110: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_108, [0, 1, 4, 3, 2]);  permute_108 = None
        view_93: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_110, [256, 64, 512]);  permute_110 = None
        bmm_20: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_92, view_93)
        view_94: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_20, [16, 16, 512, 1, 512]);  bmm_20 = None
        permute_111: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_94, [0, 1, 2, 4, 3]);  view_94 = None
        view_95: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_111, [16, 16, 512, 512]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_23: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_79, primals_38);  view_79 = primals_38 = None
        convert_element_type_110: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_23, torch.bfloat16);  add_23 = None
        unsqueeze_71: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_110, 4);  convert_element_type_110 = None
        permute_112: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_71, [1, 2, 0, 4, 3]);  unsqueeze_71 = None
        unsqueeze_72: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_91, 4);  view_91 = None
        permute_113: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_72, [1, 2, 4, 0, 3]);  unsqueeze_72 = None
        permute_114: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_112, [0, 1, 2, 4, 3]);  permute_112 = None
        view_96: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_114, [256, 512, 64]);  permute_114 = None
        permute_115: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_113, [0, 1, 4, 3, 2]);  permute_113 = None
        view_97: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_115, [256, 64, 1024]);  permute_115 = None
        bmm_21: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_96, view_97)
        view_98: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_21, [16, 16, 512, 1, 1024]);  bmm_21 = None
        permute_116: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_98, [0, 1, 2, 4, 3]);  view_98 = None
        view_99: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_116, [16, 16, 512, 1024]);  permute_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_100: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_99, [16, 16, 1024, 512]);  view_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_3: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_100, 2, 1, 9223372036854775807);  view_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_101: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_3, [16, 16, 512, 1023]);  slice_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_2: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_101, [None, None, None, iota_2]);  view_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_24: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_95, index_2);  view_95 = index_2 = None
        add_25: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_24, 0);  add_24 = None

        # No stacktrace found for following nodes
        mul_tensor_84: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_25, 0.125)
        convert_element_type_default_66: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_84, torch.float32);  mul_tensor_84 = None
        convert_element_type_default_67: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_25, torch.float32)
        mul_tensor_85: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_67, 1);  convert_element_type_default_67 = None
        amax_default_42: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_85, [3], True)
        sub_tensor_42: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_85, amax_default_42);  mul_tensor_85 = None
        mul_tensor_86: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_42, 0.125);  sub_tensor_42 = None
        amax_default_43: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_66, [3], True)
        sub_tensor_43: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_66, amax_default_43)
        abs_default_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_66)
        ne_scalar_21: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_21, inf);  abs_default_21 = None
        eq_tensor_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_66, convert_element_type_default_66);  convert_element_type_default_66 = None
        mul_tensor_87: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_22, ne_scalar_21);  eq_tensor_22 = ne_scalar_21 = None
        logical_not_default_42: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_87);  mul_tensor_87 = None
        any_dims_21: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_42, [3], True);  logical_not_default_42 = None
        logical_not_default_43: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_21);  any_dims_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_43, mul_tensor_86, sub_tensor_43);  mul_tensor_86 = sub_tensor_43 = None
        exp_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_22);  where_self_22 = None
        sum_3: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_2, [3], True)
        div_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_10: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 10)
        inductor_random_default_88: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_10, 'rand');  inductor_lookup_seed_default_10 = None
        gt_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_88, 0.1);  inductor_random_default_88 = None
        mul_39: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_10, div_3);  div_3 = None
        mul_40: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_39, 1.1111111111111112);  mul_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_114: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_40, torch.bfloat16);  mul_40 = None
        unsqueeze_73: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_114, 4);  convert_element_type_114 = None
        unsqueeze_74: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_87, 4);  view_87 = None
        permute_118: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_74, [4, 1, 2, 3, 0]);  unsqueeze_74 = None
        view_102: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_73, [256, 512, 512]);  unsqueeze_73 = None
        permute_120: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_118, [1, 2, 4, 3, 0]);  permute_118 = None
        view_103: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_120, [256, 512, 64]);  permute_120 = None
        bmm_22: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_102, view_103)
        view_104: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_22, [16, 16, 512, 1, 64]);  bmm_22 = None
        permute_121: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_104, [2, 0, 1, 4, 3]);  view_104 = None
        view_105: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_121, [512, 16, 16, 64]);  permute_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_117: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_39, torch.bfloat16);  primals_39 = None
        unsqueeze_75: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_105, 4);  view_105 = None
        permute_122: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_75, [0, 1, 4, 3, 2]);  unsqueeze_75 = None
        unsqueeze_76: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_117, 3);  convert_element_type_117 = None
        unsqueeze_77: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_76, 4);  unsqueeze_76 = None
        permute_123: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_77, [3, 4, 0, 2, 1]);  unsqueeze_77 = None
        permute_124: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_122, [0, 1, 3, 4, 2]);  permute_122 = None
        clone_5: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_124, memory_format = torch.contiguous_format);  permute_124 = None
        view_106: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_5, [1, 8192, 1024]);  clone_5 = None
        permute_125: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_123, [3, 4, 2, 0, 1]);  permute_123 = None
        clone_6: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_125, memory_format = torch.contiguous_format);  permute_125 = None
        view_107: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_6, [1, 1024, 1024]);  clone_6 = None
        squeeze_dim_642: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_106, 0)
        squeeze_dim_643: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_107, 0)
        mm_default_321: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_642, squeeze_dim_643);  squeeze_dim_642 = squeeze_dim_643 = None
        unsqueeze_default_321: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_321, 0);  mm_default_321 = None
        view_108: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_321, [512, 16, 1, 1, 1024]);  unsqueeze_default_321 = None
        permute_126: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_108, [0, 1, 4, 2, 3]);  view_108 = None
        view_109: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_126, [512, 16, 1024]);  permute_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_11: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 11)
        inductor_random_default_87: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_11, 'rand');  inductor_lookup_seed_default_11 = None
        convert_element_type_default_137: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_87, torch.bfloat16);  inductor_random_default_87 = None
        gt_11: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_137, 0.1);  convert_element_type_default_137 = None
        mul_41: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_11, view_109);  view_109 = None
        mul_42: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_26: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_42, add_21);  mul_42 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_8: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_4[0]
        getitem_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_4[1];  var_mean_4 = None
        add_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_8, 1e-12);  getitem_8 = None
        rsqrt_4: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        sub_7: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_26, getitem_9);  add_26 = getitem_9 = None
        mul_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_4);  sub_7 = None
        mul_44: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_43, primals_40)
        add_28: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_44, primals_41);  mul_44 = primals_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_120: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_43, torch.bfloat16);  primals_43 = None
        convert_element_type_121: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_42, torch.bfloat16);  primals_42 = None
        convert_element_type_122: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16)
        view_110: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_122, [8192, 1024]);  convert_element_type_122 = None
        permute_127: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        addmm_4: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_120, view_110, permute_127);  convert_element_type_120 = None
        view_111: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_4, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_126: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_111, torch.float32);  view_111 = None
        mul_45: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.5)
        mul_46: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_126, 0.7071067811865476);  convert_element_type_126 = None
        erf_2: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_46);  mul_46 = None
        add_29: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_47: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_45, add_29);  mul_45 = add_29 = None
        convert_element_type_127: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_47, torch.bfloat16);  mul_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_12: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 12)
        inductor_random_default_86: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_12, 'rand');  inductor_lookup_seed_default_12 = None
        convert_element_type_default_136: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_86, torch.bfloat16);  inductor_random_default_86 = None
        gt_12: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_136, 0.1);  convert_element_type_default_136 = None
        mul_48: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_12, convert_element_type_127);  convert_element_type_127 = None
        mul_49: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_48, 1.1111111111111112);  mul_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_128: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_45, torch.bfloat16);  primals_45 = None
        convert_element_type_129: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_44, torch.bfloat16);  primals_44 = None
        view_112: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_49, [8192, 4096]);  mul_49 = None
        permute_128: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        addmm_5: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_128, view_112, permute_128);  convert_element_type_128 = None
        view_113: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_5, [512, 16, 1024]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_13: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 13)
        inductor_random_default_85: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_13, 'rand');  inductor_lookup_seed_default_13 = None
        convert_element_type_default_135: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_85, torch.bfloat16);  inductor_random_default_85 = None
        gt_13: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_135, 0.1);  convert_element_type_default_135 = None
        mul_50: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_13, view_113);  view_113 = None
        mul_51: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_50, 1.1111111111111112);  mul_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_30: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_51, add_28);  mul_51 = add_28 = None
        var_mean_5 = torch.ops.aten.var_mean.correction(add_30, [2], correction = 0, keepdim = True)
        getitem_10: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_5[0]
        getitem_11: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_5[1];  var_mean_5 = None
        add_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_10, 1e-12);  getitem_10 = None
        rsqrt_5: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_31);  add_31 = None
        sub_8: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_30, getitem_11);  add_30 = getitem_11 = None
        mul_52: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_5);  sub_8 = None
        mul_53: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_52, primals_46)
        add_32: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_53, primals_47);  mul_53 = primals_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_133: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16)
        convert_element_type_134: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_48, torch.bfloat16);  primals_48 = None
        unsqueeze_78: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_133, 3);  convert_element_type_133 = None
        unsqueeze_79: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_78, 4);  unsqueeze_78 = None
        unsqueeze_80: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_134, 3);  convert_element_type_134 = None
        unsqueeze_81: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_80, 4);  unsqueeze_80 = None
        view_114: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_79, [1, 8192, 1024]);  unsqueeze_79 = None
        view_115: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_81, [1, 1024, 1024]);  unsqueeze_81 = None
        squeeze_dim_640: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_114, 0)
        squeeze_dim_641: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_115, 0)
        mm_default_320: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_640, squeeze_dim_641);  squeeze_dim_641 = None
        unsqueeze_default_320: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_320, 0);  mm_default_320 = None
        view_116: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_320, [512, 16, 1, 16, 64]);  unsqueeze_default_320 = None
        permute_133: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_116, [0, 1, 3, 4, 2]);  view_116 = None
        view_117: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_133, [512, 16, 16, 64]);  permute_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_138: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_49, torch.bfloat16);  primals_49 = None
        unsqueeze_84: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_138, 3);  convert_element_type_138 = None
        unsqueeze_85: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_84, 4);  unsqueeze_84 = None
        view_119: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_85, [1, 1024, 1024]);  unsqueeze_85 = None
        squeeze_dim_639: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_119, 0)
        mm_default_319: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_640, squeeze_dim_639);  squeeze_dim_639 = None
        unsqueeze_default_319: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_319, 0);  mm_default_319 = None
        view_120: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_319, [512, 16, 1, 16, 64]);  unsqueeze_default_319 = None
        permute_138: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_120, [0, 1, 3, 4, 2]);  view_120 = None
        view_121: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_138, [512, 16, 16, 64]);  permute_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_142: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_50, torch.bfloat16);  primals_50 = None
        unsqueeze_88: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_142, 3);  convert_element_type_142 = None
        unsqueeze_89: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_88, 4);  unsqueeze_88 = None
        view_123: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_89, [1, 1024, 1024]);  unsqueeze_89 = None
        squeeze_dim_637: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_123, 0)
        mm_default_318: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_640, squeeze_dim_637);  squeeze_dim_640 = squeeze_dim_637 = None
        unsqueeze_default_318: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_318, 0);  mm_default_318 = None
        view_124: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_318, [512, 16, 1, 16, 64]);  unsqueeze_default_318 = None
        permute_143: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_124, [0, 1, 3, 4, 2]);  view_124 = None
        view_125: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_143, [512, 16, 16, 64]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_147: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_51, torch.bfloat16);  primals_51 = None
        unsqueeze_92: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_147, 3);  convert_element_type_147 = None
        unsqueeze_93: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_92, 4);  unsqueeze_92 = None
        view_127: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_93, [1, 1024, 1024]);  unsqueeze_93 = None
        squeeze_dim_635: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_127, 0);  view_127 = None
        mm_default_317: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_635);  squeeze_dim_635 = None
        unsqueeze_default_317: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_317, 0);  mm_default_317 = None
        view_128: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_317, [1024, 16, 1, 16, 64]);  unsqueeze_default_317 = None
        permute_148: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_128, [0, 1, 3, 4, 2]);  view_128 = None
        view_129: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_148, [1024, 16, 16, 64]);  permute_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_33: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_117, primals_52);  primals_52 = None
        convert_element_type_150: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_33, torch.bfloat16);  add_33 = None
        unsqueeze_94: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_150, 4);  convert_element_type_150 = None
        permute_149: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_94, [1, 2, 0, 4, 3]);  unsqueeze_94 = None
        unsqueeze_95: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_121, 4);  view_121 = None
        permute_150: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_95, [1, 2, 4, 0, 3]);  unsqueeze_95 = None
        permute_151: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_149, [0, 1, 2, 4, 3]);  permute_149 = None
        view_130: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_151, [256, 512, 64]);  permute_151 = None
        permute_152: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_150, [0, 1, 4, 3, 2]);  permute_150 = None
        view_131: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_152, [256, 64, 512]);  permute_152 = None
        bmm_28: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_130, view_131)
        view_132: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_28, [16, 16, 512, 1, 512]);  bmm_28 = None
        permute_153: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_132, [0, 1, 2, 4, 3]);  view_132 = None
        view_133: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_153, [16, 16, 512, 512]);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_34: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_117, primals_53);  view_117 = primals_53 = None
        convert_element_type_153: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16);  add_34 = None
        unsqueeze_96: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_153, 4);  convert_element_type_153 = None
        permute_154: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_96, [1, 2, 0, 4, 3]);  unsqueeze_96 = None
        unsqueeze_97: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_129, 4);  view_129 = None
        permute_155: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_97, [1, 2, 4, 0, 3]);  unsqueeze_97 = None
        permute_156: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_154, [0, 1, 2, 4, 3]);  permute_154 = None
        view_134: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_156, [256, 512, 64]);  permute_156 = None
        permute_157: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_155, [0, 1, 4, 3, 2]);  permute_155 = None
        view_135: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_157, [256, 64, 1024]);  permute_157 = None
        bmm_29: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_134, view_135)
        view_136: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_29, [16, 16, 512, 1, 1024]);  bmm_29 = None
        permute_158: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_136, [0, 1, 2, 4, 3]);  view_136 = None
        view_137: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_158, [16, 16, 512, 1024]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_138: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_137, [16, 16, 1024, 512]);  view_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_4: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_138, 2, 1, 9223372036854775807);  view_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_139: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_4, [16, 16, 512, 1023]);  slice_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_3: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_139, [None, None, None, iota_2]);  view_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_35: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_133, index_3);  view_133 = index_3 = None
        add_36: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_35, 0);  add_35 = None

        # No stacktrace found for following nodes
        mul_tensor_80: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_36, 0.125)
        convert_element_type_default_64: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_80, torch.float32);  mul_tensor_80 = None
        convert_element_type_default_65: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_36, torch.float32)
        mul_tensor_81: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_65, 1);  convert_element_type_default_65 = None
        amax_default_40: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_81, [3], True)
        sub_tensor_40: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_81, amax_default_40);  mul_tensor_81 = None
        mul_tensor_82: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_40, 0.125);  sub_tensor_40 = None
        amax_default_41: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_64, [3], True)
        sub_tensor_41: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_64, amax_default_41)
        abs_default_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_64)
        ne_scalar_20: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_20, inf);  abs_default_20 = None
        eq_tensor_21: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_64, convert_element_type_default_64);  convert_element_type_default_64 = None
        mul_tensor_83: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_21, ne_scalar_20);  eq_tensor_21 = ne_scalar_20 = None
        logical_not_default_40: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_83);  mul_tensor_83 = None
        any_dims_20: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_40, [3], True);  logical_not_default_40 = None
        logical_not_default_41: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_20);  any_dims_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_41, mul_tensor_82, sub_tensor_41);  mul_tensor_82 = sub_tensor_41 = None
        exp_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_21);  where_self_21 = None
        sum_4: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_3, [3], True)
        div_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_14: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 14)
        inductor_random_default_84: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_14, 'rand');  inductor_lookup_seed_default_14 = None
        gt_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_84, 0.1);  inductor_random_default_84 = None
        mul_55: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_14, div_4);  div_4 = None
        mul_56: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_55, 1.1111111111111112);  mul_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_157: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_56, torch.bfloat16);  mul_56 = None
        unsqueeze_98: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_157, 4);  convert_element_type_157 = None
        unsqueeze_99: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_125, 4);  view_125 = None
        permute_160: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_99, [4, 1, 2, 3, 0]);  unsqueeze_99 = None
        view_140: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_98, [256, 512, 512]);  unsqueeze_98 = None
        permute_162: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_160, [1, 2, 4, 3, 0]);  permute_160 = None
        view_141: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_162, [256, 512, 64]);  permute_162 = None
        bmm_30: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_140, view_141)
        view_142: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_30, [16, 16, 512, 1, 64]);  bmm_30 = None
        permute_163: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_142, [2, 0, 1, 4, 3]);  view_142 = None
        view_143: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_163, [512, 16, 16, 64]);  permute_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_160: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_54, torch.bfloat16);  primals_54 = None
        unsqueeze_100: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_143, 4);  view_143 = None
        permute_164: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_100, [0, 1, 4, 3, 2]);  unsqueeze_100 = None
        unsqueeze_101: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_160, 3);  convert_element_type_160 = None
        unsqueeze_102: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_101, 4);  unsqueeze_101 = None
        permute_165: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_102, [3, 4, 0, 2, 1]);  unsqueeze_102 = None
        permute_166: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_164, [0, 1, 3, 4, 2]);  permute_164 = None
        clone_7: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_166, memory_format = torch.contiguous_format);  permute_166 = None
        view_144: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_7, [1, 8192, 1024]);  clone_7 = None
        permute_167: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_165, [3, 4, 2, 0, 1]);  permute_165 = None
        clone_8: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_167, memory_format = torch.contiguous_format);  permute_167 = None
        view_145: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_8, [1, 1024, 1024]);  clone_8 = None
        squeeze_dim_632: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_144, 0)
        squeeze_dim_633: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_145, 0)
        mm_default_316: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_632, squeeze_dim_633);  squeeze_dim_632 = squeeze_dim_633 = None
        unsqueeze_default_316: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_316, 0);  mm_default_316 = None
        view_146: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_316, [512, 16, 1, 1, 1024]);  unsqueeze_default_316 = None
        permute_168: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_146, [0, 1, 4, 2, 3]);  view_146 = None
        view_147: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_168, [512, 16, 1024]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_15: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 15)
        inductor_random_default_83: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_15, 'rand');  inductor_lookup_seed_default_15 = None
        convert_element_type_default_134: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_83, torch.bfloat16);  inductor_random_default_83 = None
        gt_15: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_134, 0.1);  convert_element_type_default_134 = None
        mul_57: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_15, view_147);  view_147 = None
        mul_58: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_57, 1.1111111111111112);  mul_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_37: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_58, add_32);  mul_58 = add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_12: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_6[0]
        getitem_13: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_6[1];  var_mean_6 = None
        add_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_12, 1e-12);  getitem_12 = None
        rsqrt_6: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        sub_10: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_37, getitem_13);  add_37 = getitem_13 = None
        mul_59: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_6);  sub_10 = None
        mul_60: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_59, primals_55)
        add_39: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_60, primals_56);  mul_60 = primals_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_163: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_58, torch.bfloat16);  primals_58 = None
        convert_element_type_164: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_57, torch.bfloat16);  primals_57 = None
        convert_element_type_165: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_39, torch.bfloat16)
        view_148: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_165, [8192, 1024]);  convert_element_type_165 = None
        permute_169: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_6: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_163, view_148, permute_169);  convert_element_type_163 = None
        view_149: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_6, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_169: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_149, torch.float32);  view_149 = None
        mul_61: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.5)
        mul_62: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_169, 0.7071067811865476);  convert_element_type_169 = None
        erf_3: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_62);  mul_62 = None
        add_40: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_63: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_61, add_40);  mul_61 = add_40 = None
        convert_element_type_170: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_63, torch.bfloat16);  mul_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_16: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 16)
        inductor_random_default_82: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_16, 'rand');  inductor_lookup_seed_default_16 = None
        convert_element_type_default_133: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_82, torch.bfloat16);  inductor_random_default_82 = None
        gt_16: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_133, 0.1);  convert_element_type_default_133 = None
        mul_64: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_16, convert_element_type_170);  convert_element_type_170 = None
        mul_65: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_64, 1.1111111111111112);  mul_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_171: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_60, torch.bfloat16);  primals_60 = None
        convert_element_type_172: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_59, torch.bfloat16);  primals_59 = None
        view_150: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_65, [8192, 4096]);  mul_65 = None
        permute_170: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_7: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_171, view_150, permute_170);  convert_element_type_171 = None
        view_151: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_7, [512, 16, 1024]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_17: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 17)
        inductor_random_default_81: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_17, 'rand');  inductor_lookup_seed_default_17 = None
        convert_element_type_default_132: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_81, torch.bfloat16);  inductor_random_default_81 = None
        gt_17: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_132, 0.1);  convert_element_type_default_132 = None
        mul_66: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_17, view_151);  view_151 = None
        mul_67: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_66, 1.1111111111111112);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_41: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_67, add_39);  mul_67 = add_39 = None
        var_mean_7 = torch.ops.aten.var_mean.correction(add_41, [2], correction = 0, keepdim = True)
        getitem_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_7[0]
        getitem_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_7[1];  var_mean_7 = None
        add_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_14, 1e-12);  getitem_14 = None
        rsqrt_7: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_42);  add_42 = None
        sub_11: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_41, getitem_15);  add_41 = getitem_15 = None
        mul_68: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_7);  sub_11 = None
        mul_69: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_68, primals_61)
        add_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_69, primals_62);  mul_69 = primals_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_176: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_43, torch.bfloat16)
        convert_element_type_177: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_63, torch.bfloat16);  primals_63 = None
        unsqueeze_103: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_176, 3);  convert_element_type_176 = None
        unsqueeze_104: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_103, 4);  unsqueeze_103 = None
        unsqueeze_105: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_177, 3);  convert_element_type_177 = None
        unsqueeze_106: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_105, 4);  unsqueeze_105 = None
        view_152: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_104, [1, 8192, 1024]);  unsqueeze_104 = None
        view_153: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_106, [1, 1024, 1024]);  unsqueeze_106 = None
        squeeze_dim_630: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_152, 0)
        squeeze_dim_631: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_153, 0)
        mm_default_315: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_630, squeeze_dim_631);  squeeze_dim_631 = None
        unsqueeze_default_315: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_315, 0);  mm_default_315 = None
        view_154: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_315, [512, 16, 1, 16, 64]);  unsqueeze_default_315 = None
        permute_175: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_154, [0, 1, 3, 4, 2]);  view_154 = None
        view_155: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_175, [512, 16, 16, 64]);  permute_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_181: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_64, torch.bfloat16);  primals_64 = None
        unsqueeze_109: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_181, 3);  convert_element_type_181 = None
        unsqueeze_110: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_109, 4);  unsqueeze_109 = None
        view_157: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_110, [1, 1024, 1024]);  unsqueeze_110 = None
        squeeze_dim_629: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_157, 0)
        mm_default_314: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_630, squeeze_dim_629);  squeeze_dim_629 = None
        unsqueeze_default_314: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_314, 0);  mm_default_314 = None
        view_158: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_314, [512, 16, 1, 16, 64]);  unsqueeze_default_314 = None
        permute_180: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_158, [0, 1, 3, 4, 2]);  view_158 = None
        view_159: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_180, [512, 16, 16, 64]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_185: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_65, torch.bfloat16);  primals_65 = None
        unsqueeze_113: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_185, 3);  convert_element_type_185 = None
        unsqueeze_114: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_113, 4);  unsqueeze_113 = None
        view_161: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_114, [1, 1024, 1024]);  unsqueeze_114 = None
        squeeze_dim_627: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_161, 0)
        mm_default_313: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_630, squeeze_dim_627);  squeeze_dim_630 = squeeze_dim_627 = None
        unsqueeze_default_313: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_313, 0);  mm_default_313 = None
        view_162: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_313, [512, 16, 1, 16, 64]);  unsqueeze_default_313 = None
        permute_185: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_162, [0, 1, 3, 4, 2]);  view_162 = None
        view_163: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_185, [512, 16, 16, 64]);  permute_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_190: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_66, torch.bfloat16);  primals_66 = None
        unsqueeze_117: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_190, 3);  convert_element_type_190 = None
        unsqueeze_118: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_117, 4);  unsqueeze_117 = None
        view_165: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_118, [1, 1024, 1024]);  unsqueeze_118 = None
        squeeze_dim_625: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_165, 0);  view_165 = None
        mm_default_312: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_625);  squeeze_dim_625 = None
        unsqueeze_default_312: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_312, 0);  mm_default_312 = None
        view_166: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_312, [1024, 16, 1, 16, 64]);  unsqueeze_default_312 = None
        permute_190: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_166, [0, 1, 3, 4, 2]);  view_166 = None
        view_167: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_190, [1024, 16, 16, 64]);  permute_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_44: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_155, primals_67);  primals_67 = None
        convert_element_type_193: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16);  add_44 = None
        unsqueeze_119: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_193, 4);  convert_element_type_193 = None
        permute_191: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_119, [1, 2, 0, 4, 3]);  unsqueeze_119 = None
        unsqueeze_120: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_159, 4);  view_159 = None
        permute_192: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_120, [1, 2, 4, 0, 3]);  unsqueeze_120 = None
        permute_193: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_191, [0, 1, 2, 4, 3]);  permute_191 = None
        view_168: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_193, [256, 512, 64]);  permute_193 = None
        permute_194: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_192, [0, 1, 4, 3, 2]);  permute_192 = None
        view_169: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_194, [256, 64, 512]);  permute_194 = None
        bmm_36: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_168, view_169)
        view_170: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_36, [16, 16, 512, 1, 512]);  bmm_36 = None
        permute_195: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_170, [0, 1, 2, 4, 3]);  view_170 = None
        view_171: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_195, [16, 16, 512, 512]);  permute_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_45: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_155, primals_68);  view_155 = primals_68 = None
        convert_element_type_196: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_45, torch.bfloat16);  add_45 = None
        unsqueeze_121: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_196, 4);  convert_element_type_196 = None
        permute_196: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_121, [1, 2, 0, 4, 3]);  unsqueeze_121 = None
        unsqueeze_122: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_167, 4);  view_167 = None
        permute_197: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_122, [1, 2, 4, 0, 3]);  unsqueeze_122 = None
        permute_198: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_196, [0, 1, 2, 4, 3]);  permute_196 = None
        view_172: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_198, [256, 512, 64]);  permute_198 = None
        permute_199: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_197, [0, 1, 4, 3, 2]);  permute_197 = None
        view_173: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_199, [256, 64, 1024]);  permute_199 = None
        bmm_37: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_172, view_173)
        view_174: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_37, [16, 16, 512, 1, 1024]);  bmm_37 = None
        permute_200: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_174, [0, 1, 2, 4, 3]);  view_174 = None
        view_175: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_200, [16, 16, 512, 1024]);  permute_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_176: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_175, [16, 16, 1024, 512]);  view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_5: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_176, 2, 1, 9223372036854775807);  view_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_177: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_5, [16, 16, 512, 1023]);  slice_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_4: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_177, [None, None, None, iota_2]);  view_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_46: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_171, index_4);  view_171 = index_4 = None
        add_47: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_46, 0);  add_46 = None

        # No stacktrace found for following nodes
        mul_tensor_76: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_47, 0.125)
        convert_element_type_default_62: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_76, torch.float32);  mul_tensor_76 = None
        convert_element_type_default_63: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_47, torch.float32)
        mul_tensor_77: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_63, 1);  convert_element_type_default_63 = None
        amax_default_38: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_77, [3], True)
        sub_tensor_38: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_77, amax_default_38);  mul_tensor_77 = None
        mul_tensor_78: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_38, 0.125);  sub_tensor_38 = None
        amax_default_39: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_62, [3], True)
        sub_tensor_39: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_62, amax_default_39)
        abs_default_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_62)
        ne_scalar_19: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_19, inf);  abs_default_19 = None
        eq_tensor_20: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_62, convert_element_type_default_62);  convert_element_type_default_62 = None
        mul_tensor_79: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_20, ne_scalar_19);  eq_tensor_20 = ne_scalar_19 = None
        logical_not_default_38: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_79);  mul_tensor_79 = None
        any_dims_19: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_38, [3], True);  logical_not_default_38 = None
        logical_not_default_39: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_19);  any_dims_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_39, mul_tensor_78, sub_tensor_39);  mul_tensor_78 = sub_tensor_39 = None
        exp_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_20);  where_self_20 = None
        sum_5: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_4, [3], True)
        div_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_18: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 18)
        inductor_random_default_80: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_18, 'rand');  inductor_lookup_seed_default_18 = None
        gt_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_80, 0.1);  inductor_random_default_80 = None
        mul_71: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_18, div_5);  div_5 = None
        mul_72: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_71, 1.1111111111111112);  mul_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_200: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_72, torch.bfloat16);  mul_72 = None
        unsqueeze_123: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_200, 4);  convert_element_type_200 = None
        unsqueeze_124: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_163, 4);  view_163 = None
        permute_202: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_124, [4, 1, 2, 3, 0]);  unsqueeze_124 = None
        view_178: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_123, [256, 512, 512]);  unsqueeze_123 = None
        permute_204: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_202, [1, 2, 4, 3, 0]);  permute_202 = None
        view_179: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_204, [256, 512, 64]);  permute_204 = None
        bmm_38: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_178, view_179)
        view_180: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_38, [16, 16, 512, 1, 64]);  bmm_38 = None
        permute_205: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_180, [2, 0, 1, 4, 3]);  view_180 = None
        view_181: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_205, [512, 16, 16, 64]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_203: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_69, torch.bfloat16);  primals_69 = None
        unsqueeze_125: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_181, 4);  view_181 = None
        permute_206: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_125, [0, 1, 4, 3, 2]);  unsqueeze_125 = None
        unsqueeze_126: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_203, 3);  convert_element_type_203 = None
        unsqueeze_127: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_126, 4);  unsqueeze_126 = None
        permute_207: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_127, [3, 4, 0, 2, 1]);  unsqueeze_127 = None
        permute_208: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 4, 2]);  permute_206 = None
        clone_9: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None
        view_182: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_9, [1, 8192, 1024]);  clone_9 = None
        permute_209: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_207, [3, 4, 2, 0, 1]);  permute_207 = None
        clone_10: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_183: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_10, [1, 1024, 1024]);  clone_10 = None
        squeeze_dim_622: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_182, 0)
        squeeze_dim_623: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_183, 0)
        mm_default_311: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_622, squeeze_dim_623);  squeeze_dim_622 = squeeze_dim_623 = None
        unsqueeze_default_311: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_311, 0);  mm_default_311 = None
        view_184: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_311, [512, 16, 1, 1, 1024]);  unsqueeze_default_311 = None
        permute_210: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_184, [0, 1, 4, 2, 3]);  view_184 = None
        view_185: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_210, [512, 16, 1024]);  permute_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_19: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 19)
        inductor_random_default_79: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_19, 'rand');  inductor_lookup_seed_default_19 = None
        convert_element_type_default_131: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_79, torch.bfloat16);  inductor_random_default_79 = None
        gt_19: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_131, 0.1);  convert_element_type_default_131 = None
        mul_73: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_19, view_185);  view_185 = None
        mul_74: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_73, 1.1111111111111112);  mul_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_48: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_74, add_43);  mul_74 = add_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_8[0]
        getitem_17: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_8[1];  var_mean_8 = None
        add_49: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_16, 1e-12);  getitem_16 = None
        rsqrt_8: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        sub_13: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_48, getitem_17);  add_48 = getitem_17 = None
        mul_75: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_13, rsqrt_8);  sub_13 = None
        mul_76: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_75, primals_70)
        add_50: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_76, primals_71);  mul_76 = primals_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_206: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_73, torch.bfloat16);  primals_73 = None
        convert_element_type_207: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_72, torch.bfloat16);  primals_72 = None
        convert_element_type_208: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16)
        view_186: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_208, [8192, 1024]);  convert_element_type_208 = None
        permute_211: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        addmm_8: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_206, view_186, permute_211);  convert_element_type_206 = None
        view_187: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_8, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_212: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_187, torch.float32);  view_187 = None
        mul_77: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.5)
        mul_78: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_212, 0.7071067811865476);  convert_element_type_212 = None
        erf_4: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_78);  mul_78 = None
        add_51: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_79: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_77, add_51);  mul_77 = add_51 = None
        convert_element_type_213: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_79, torch.bfloat16);  mul_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_20: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 20)
        inductor_random_default_78: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_20, 'rand');  inductor_lookup_seed_default_20 = None
        convert_element_type_default_130: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_78, torch.bfloat16);  inductor_random_default_78 = None
        gt_20: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_130, 0.1);  convert_element_type_default_130 = None
        mul_80: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_20, convert_element_type_213);  convert_element_type_213 = None
        mul_81: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_80, 1.1111111111111112);  mul_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_214: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_75, torch.bfloat16);  primals_75 = None
        convert_element_type_215: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_74, torch.bfloat16);  primals_74 = None
        view_188: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_81, [8192, 4096]);  mul_81 = None
        permute_212: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_215, [1, 0]);  convert_element_type_215 = None
        addmm_9: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_214, view_188, permute_212);  convert_element_type_214 = None
        view_189: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_9, [512, 16, 1024]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_21: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 21)
        inductor_random_default_77: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_21, 'rand');  inductor_lookup_seed_default_21 = None
        convert_element_type_default_129: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_77, torch.bfloat16);  inductor_random_default_77 = None
        gt_21: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_129, 0.1);  convert_element_type_default_129 = None
        mul_82: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_21, view_189);  view_189 = None
        mul_83: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_82, 1.1111111111111112);  mul_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_52: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_83, add_50);  mul_83 = add_50 = None
        var_mean_9 = torch.ops.aten.var_mean.correction(add_52, [2], correction = 0, keepdim = True)
        getitem_18: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_9[0]
        getitem_19: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_9[1];  var_mean_9 = None
        add_53: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_18, 1e-12);  getitem_18 = None
        rsqrt_9: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_53);  add_53 = None
        sub_14: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_52, getitem_19);  add_52 = getitem_19 = None
        mul_84: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_9);  sub_14 = None
        mul_85: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_84, primals_76)
        add_54: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_85, primals_77);  mul_85 = primals_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_219: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        convert_element_type_220: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_78, torch.bfloat16);  primals_78 = None
        unsqueeze_128: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_219, 3);  convert_element_type_219 = None
        unsqueeze_129: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_128, 4);  unsqueeze_128 = None
        unsqueeze_130: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_220, 3);  convert_element_type_220 = None
        unsqueeze_131: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_130, 4);  unsqueeze_130 = None
        view_190: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_129, [1, 8192, 1024]);  unsqueeze_129 = None
        view_191: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_131, [1, 1024, 1024]);  unsqueeze_131 = None
        squeeze_dim_620: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_190, 0)
        squeeze_dim_621: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_191, 0)
        mm_default_310: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_620, squeeze_dim_621);  squeeze_dim_621 = None
        unsqueeze_default_310: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_310, 0);  mm_default_310 = None
        view_192: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_310, [512, 16, 1, 16, 64]);  unsqueeze_default_310 = None
        permute_217: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_192, [0, 1, 3, 4, 2]);  view_192 = None
        view_193: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_217, [512, 16, 16, 64]);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_224: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_79, torch.bfloat16);  primals_79 = None
        unsqueeze_134: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_224, 3);  convert_element_type_224 = None
        unsqueeze_135: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_134, 4);  unsqueeze_134 = None
        view_195: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_135, [1, 1024, 1024]);  unsqueeze_135 = None
        squeeze_dim_619: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_195, 0)
        mm_default_309: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_620, squeeze_dim_619);  squeeze_dim_619 = None
        unsqueeze_default_309: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_309, 0);  mm_default_309 = None
        view_196: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_309, [512, 16, 1, 16, 64]);  unsqueeze_default_309 = None
        permute_222: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_196, [0, 1, 3, 4, 2]);  view_196 = None
        view_197: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_222, [512, 16, 16, 64]);  permute_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_228: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_80, torch.bfloat16);  primals_80 = None
        unsqueeze_138: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_228, 3);  convert_element_type_228 = None
        unsqueeze_139: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_138, 4);  unsqueeze_138 = None
        view_199: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_139, [1, 1024, 1024]);  unsqueeze_139 = None
        squeeze_dim_617: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_199, 0)
        mm_default_308: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_620, squeeze_dim_617);  squeeze_dim_620 = squeeze_dim_617 = None
        unsqueeze_default_308: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_308, 0);  mm_default_308 = None
        view_200: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_308, [512, 16, 1, 16, 64]);  unsqueeze_default_308 = None
        permute_227: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_200, [0, 1, 3, 4, 2]);  view_200 = None
        view_201: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_227, [512, 16, 16, 64]);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_233: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_81, torch.bfloat16);  primals_81 = None
        unsqueeze_142: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_233, 3);  convert_element_type_233 = None
        unsqueeze_143: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_142, 4);  unsqueeze_142 = None
        view_203: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_143, [1, 1024, 1024]);  unsqueeze_143 = None
        squeeze_dim_615: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_203, 0);  view_203 = None
        mm_default_307: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_615);  squeeze_dim_615 = None
        unsqueeze_default_307: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_307, 0);  mm_default_307 = None
        view_204: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_307, [1024, 16, 1, 16, 64]);  unsqueeze_default_307 = None
        permute_232: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_204, [0, 1, 3, 4, 2]);  view_204 = None
        view_205: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_232, [1024, 16, 16, 64]);  permute_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_55: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, primals_82);  primals_82 = None
        convert_element_type_236: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_55, torch.bfloat16);  add_55 = None
        unsqueeze_144: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_236, 4);  convert_element_type_236 = None
        permute_233: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_144, [1, 2, 0, 4, 3]);  unsqueeze_144 = None
        unsqueeze_145: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_197, 4);  view_197 = None
        permute_234: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_145, [1, 2, 4, 0, 3]);  unsqueeze_145 = None
        permute_235: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_233, [0, 1, 2, 4, 3]);  permute_233 = None
        view_206: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_235, [256, 512, 64]);  permute_235 = None
        permute_236: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_234, [0, 1, 4, 3, 2]);  permute_234 = None
        view_207: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_236, [256, 64, 512]);  permute_236 = None
        bmm_44: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_206, view_207)
        view_208: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_44, [16, 16, 512, 1, 512]);  bmm_44 = None
        permute_237: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_208, [0, 1, 2, 4, 3]);  view_208 = None
        view_209: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_237, [16, 16, 512, 512]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_56: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_193, primals_83);  view_193 = primals_83 = None
        convert_element_type_239: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None
        unsqueeze_146: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_239, 4);  convert_element_type_239 = None
        permute_238: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_146, [1, 2, 0, 4, 3]);  unsqueeze_146 = None
        unsqueeze_147: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_205, 4);  view_205 = None
        permute_239: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_147, [1, 2, 4, 0, 3]);  unsqueeze_147 = None
        permute_240: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_238, [0, 1, 2, 4, 3]);  permute_238 = None
        view_210: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_240, [256, 512, 64]);  permute_240 = None
        permute_241: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_239, [0, 1, 4, 3, 2]);  permute_239 = None
        view_211: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_241, [256, 64, 1024]);  permute_241 = None
        bmm_45: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_210, view_211)
        view_212: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_45, [16, 16, 512, 1, 1024]);  bmm_45 = None
        permute_242: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_212, [0, 1, 2, 4, 3]);  view_212 = None
        view_213: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_242, [16, 16, 512, 1024]);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_214: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_213, [16, 16, 1024, 512]);  view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_6: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_214, 2, 1, 9223372036854775807);  view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_215: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_6, [16, 16, 512, 1023]);  slice_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_5: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_215, [None, None, None, iota_2]);  view_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_57: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_209, index_5);  view_209 = index_5 = None
        add_58: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_57, 0);  add_57 = None

        # No stacktrace found for following nodes
        mul_tensor_72: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_58, 0.125)
        convert_element_type_default_60: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_72, torch.float32);  mul_tensor_72 = None
        convert_element_type_default_61: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_58, torch.float32)
        mul_tensor_73: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_61, 1);  convert_element_type_default_61 = None
        amax_default_36: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_73, [3], True)
        sub_tensor_36: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_73, amax_default_36);  mul_tensor_73 = None
        mul_tensor_74: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_36, 0.125);  sub_tensor_36 = None
        amax_default_37: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_60, [3], True)
        sub_tensor_37: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_60, amax_default_37)
        abs_default_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_60)
        ne_scalar_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_18, inf);  abs_default_18 = None
        eq_tensor_19: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_60, convert_element_type_default_60);  convert_element_type_default_60 = None
        mul_tensor_75: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_19, ne_scalar_18);  eq_tensor_19 = ne_scalar_18 = None
        logical_not_default_36: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_75);  mul_tensor_75 = None
        any_dims_18: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_36, [3], True);  logical_not_default_36 = None
        logical_not_default_37: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_18);  any_dims_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_37, mul_tensor_74, sub_tensor_37);  mul_tensor_74 = sub_tensor_37 = None
        exp_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_19);  where_self_19 = None
        sum_6: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_5, [3], True)
        div_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_22: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 22)
        inductor_random_default_76: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_22, 'rand');  inductor_lookup_seed_default_22 = None
        gt_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_76, 0.1);  inductor_random_default_76 = None
        mul_87: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_22, div_6);  div_6 = None
        mul_88: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_87, 1.1111111111111112);  mul_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_243: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_88, torch.bfloat16);  mul_88 = None
        unsqueeze_148: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_243, 4);  convert_element_type_243 = None
        unsqueeze_149: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_201, 4);  view_201 = None
        permute_244: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_149, [4, 1, 2, 3, 0]);  unsqueeze_149 = None
        view_216: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_148, [256, 512, 512]);  unsqueeze_148 = None
        permute_246: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_244, [1, 2, 4, 3, 0]);  permute_244 = None
        view_217: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_246, [256, 512, 64]);  permute_246 = None
        bmm_46: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_216, view_217)
        view_218: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_46, [16, 16, 512, 1, 64]);  bmm_46 = None
        permute_247: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_218, [2, 0, 1, 4, 3]);  view_218 = None
        view_219: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_247, [512, 16, 16, 64]);  permute_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_246: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_84, torch.bfloat16);  primals_84 = None
        unsqueeze_150: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_219, 4);  view_219 = None
        permute_248: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_150, [0, 1, 4, 3, 2]);  unsqueeze_150 = None
        unsqueeze_151: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_246, 3);  convert_element_type_246 = None
        unsqueeze_152: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_151, 4);  unsqueeze_151 = None
        permute_249: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_152, [3, 4, 0, 2, 1]);  unsqueeze_152 = None
        permute_250: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_248, [0, 1, 3, 4, 2]);  permute_248 = None
        clone_11: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_220: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_11, [1, 8192, 1024]);  clone_11 = None
        permute_251: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_249, [3, 4, 2, 0, 1]);  permute_249 = None
        clone_12: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_251, memory_format = torch.contiguous_format);  permute_251 = None
        view_221: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_12, [1, 1024, 1024]);  clone_12 = None
        squeeze_dim_612: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_220, 0)
        squeeze_dim_613: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_221, 0)
        mm_default_306: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_612, squeeze_dim_613);  squeeze_dim_612 = squeeze_dim_613 = None
        unsqueeze_default_306: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_306, 0);  mm_default_306 = None
        view_222: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_306, [512, 16, 1, 1, 1024]);  unsqueeze_default_306 = None
        permute_252: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_222, [0, 1, 4, 2, 3]);  view_222 = None
        view_223: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_252, [512, 16, 1024]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_23: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 23)
        inductor_random_default_75: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_23, 'rand');  inductor_lookup_seed_default_23 = None
        convert_element_type_default_128: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_75, torch.bfloat16);  inductor_random_default_75 = None
        gt_23: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_128, 0.1);  convert_element_type_default_128 = None
        mul_89: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_23, view_223);  view_223 = None
        mul_90: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_89, 1.1111111111111112);  mul_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_59: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_90, add_54);  mul_90 = add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_10[0]
        getitem_21: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_10[1];  var_mean_10 = None
        add_60: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_20, 1e-12);  getitem_20 = None
        rsqrt_10: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        sub_16: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_59, getitem_21);  add_59 = getitem_21 = None
        mul_91: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_10);  sub_16 = None
        mul_92: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_91, primals_85)
        add_61: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_92, primals_86);  mul_92 = primals_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_249: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_88, torch.bfloat16);  primals_88 = None
        convert_element_type_250: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_87, torch.bfloat16);  primals_87 = None
        convert_element_type_251: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_61, torch.bfloat16)
        view_224: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_251, [8192, 1024]);  convert_element_type_251 = None
        permute_253: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_250, [1, 0]);  convert_element_type_250 = None
        addmm_10: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_249, view_224, permute_253);  convert_element_type_249 = None
        view_225: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_10, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_255: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_225, torch.float32);  view_225 = None
        mul_93: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_255, 0.5)
        mul_94: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_255, 0.7071067811865476);  convert_element_type_255 = None
        erf_5: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_94);  mul_94 = None
        add_62: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_95: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_93, add_62);  mul_93 = add_62 = None
        convert_element_type_256: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_95, torch.bfloat16);  mul_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_24: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 24)
        inductor_random_default_74: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_24, 'rand');  inductor_lookup_seed_default_24 = None
        convert_element_type_default_127: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_74, torch.bfloat16);  inductor_random_default_74 = None
        gt_24: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_127, 0.1);  convert_element_type_default_127 = None
        mul_96: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_24, convert_element_type_256);  convert_element_type_256 = None
        mul_97: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_96, 1.1111111111111112);  mul_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_257: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_90, torch.bfloat16);  primals_90 = None
        convert_element_type_258: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_89, torch.bfloat16);  primals_89 = None
        view_226: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_97, [8192, 4096]);  mul_97 = None
        permute_254: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_258, [1, 0]);  convert_element_type_258 = None
        addmm_11: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_257, view_226, permute_254);  convert_element_type_257 = None
        view_227: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_11, [512, 16, 1024]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_25: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 25)
        inductor_random_default_73: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_25, 'rand');  inductor_lookup_seed_default_25 = None
        convert_element_type_default_126: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_73, torch.bfloat16);  inductor_random_default_73 = None
        gt_25: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_126, 0.1);  convert_element_type_default_126 = None
        mul_98: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_25, view_227);  view_227 = None
        mul_99: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_98, 1.1111111111111112);  mul_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_63: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_99, add_61);  mul_99 = add_61 = None
        var_mean_11 = torch.ops.aten.var_mean.correction(add_63, [2], correction = 0, keepdim = True)
        getitem_22: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_11[0]
        getitem_23: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_11[1];  var_mean_11 = None
        add_64: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_22, 1e-12);  getitem_22 = None
        rsqrt_11: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_64);  add_64 = None
        sub_17: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_63, getitem_23);  add_63 = getitem_23 = None
        mul_100: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_17, rsqrt_11);  sub_17 = None
        mul_101: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_100, primals_91)
        add_65: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_101, primals_92);  mul_101 = primals_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_262: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_65, torch.bfloat16)
        convert_element_type_263: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_93, torch.bfloat16);  primals_93 = None
        unsqueeze_153: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_262, 3);  convert_element_type_262 = None
        unsqueeze_154: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_153, 4);  unsqueeze_153 = None
        unsqueeze_155: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_263, 3);  convert_element_type_263 = None
        unsqueeze_156: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_155, 4);  unsqueeze_155 = None
        view_228: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_154, [1, 8192, 1024]);  unsqueeze_154 = None
        view_229: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_156, [1, 1024, 1024]);  unsqueeze_156 = None
        squeeze_dim_610: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_228, 0)
        squeeze_dim_611: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_229, 0)
        mm_default_305: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_610, squeeze_dim_611);  squeeze_dim_611 = None
        unsqueeze_default_305: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_305, 0);  mm_default_305 = None
        view_230: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_305, [512, 16, 1, 16, 64]);  unsqueeze_default_305 = None
        permute_259: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_230, [0, 1, 3, 4, 2]);  view_230 = None
        view_231: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_259, [512, 16, 16, 64]);  permute_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_267: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_94, torch.bfloat16);  primals_94 = None
        unsqueeze_159: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_267, 3);  convert_element_type_267 = None
        unsqueeze_160: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_159, 4);  unsqueeze_159 = None
        view_233: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_160, [1, 1024, 1024]);  unsqueeze_160 = None
        squeeze_dim_609: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_233, 0)
        mm_default_304: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_610, squeeze_dim_609);  squeeze_dim_609 = None
        unsqueeze_default_304: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_304, 0);  mm_default_304 = None
        view_234: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_304, [512, 16, 1, 16, 64]);  unsqueeze_default_304 = None
        permute_264: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_234, [0, 1, 3, 4, 2]);  view_234 = None
        view_235: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_264, [512, 16, 16, 64]);  permute_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_271: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_95, torch.bfloat16);  primals_95 = None
        unsqueeze_163: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_271, 3);  convert_element_type_271 = None
        unsqueeze_164: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_163, 4);  unsqueeze_163 = None
        view_237: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_164, [1, 1024, 1024]);  unsqueeze_164 = None
        squeeze_dim_607: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_237, 0)
        mm_default_303: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_610, squeeze_dim_607);  squeeze_dim_610 = squeeze_dim_607 = None
        unsqueeze_default_303: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_303, 0);  mm_default_303 = None
        view_238: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_303, [512, 16, 1, 16, 64]);  unsqueeze_default_303 = None
        permute_269: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_238, [0, 1, 3, 4, 2]);  view_238 = None
        view_239: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_269, [512, 16, 16, 64]);  permute_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_276: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_96, torch.bfloat16);  primals_96 = None
        unsqueeze_167: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_276, 3);  convert_element_type_276 = None
        unsqueeze_168: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_167, 4);  unsqueeze_167 = None
        view_241: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_168, [1, 1024, 1024]);  unsqueeze_168 = None
        squeeze_dim_605: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_241, 0);  view_241 = None
        mm_default_302: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_605);  squeeze_dim_605 = None
        unsqueeze_default_302: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_302, 0);  mm_default_302 = None
        view_242: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_302, [1024, 16, 1, 16, 64]);  unsqueeze_default_302 = None
        permute_274: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_242, [0, 1, 3, 4, 2]);  view_242 = None
        view_243: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_274, [1024, 16, 16, 64]);  permute_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_66: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_231, primals_97);  primals_97 = None
        convert_element_type_279: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16);  add_66 = None
        unsqueeze_169: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_279, 4);  convert_element_type_279 = None
        permute_275: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_169, [1, 2, 0, 4, 3]);  unsqueeze_169 = None
        unsqueeze_170: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_235, 4);  view_235 = None
        permute_276: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_170, [1, 2, 4, 0, 3]);  unsqueeze_170 = None
        permute_277: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_275, [0, 1, 2, 4, 3]);  permute_275 = None
        view_244: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_277, [256, 512, 64]);  permute_277 = None
        permute_278: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_276, [0, 1, 4, 3, 2]);  permute_276 = None
        view_245: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_278, [256, 64, 512]);  permute_278 = None
        bmm_52: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_244, view_245)
        view_246: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_52, [16, 16, 512, 1, 512]);  bmm_52 = None
        permute_279: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_246, [0, 1, 2, 4, 3]);  view_246 = None
        view_247: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_279, [16, 16, 512, 512]);  permute_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_67: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_231, primals_98);  view_231 = primals_98 = None
        convert_element_type_282: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_67, torch.bfloat16);  add_67 = None
        unsqueeze_171: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_282, 4);  convert_element_type_282 = None
        permute_280: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_171, [1, 2, 0, 4, 3]);  unsqueeze_171 = None
        unsqueeze_172: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_243, 4);  view_243 = None
        permute_281: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_172, [1, 2, 4, 0, 3]);  unsqueeze_172 = None
        permute_282: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_280, [0, 1, 2, 4, 3]);  permute_280 = None
        view_248: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_282, [256, 512, 64]);  permute_282 = None
        permute_283: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_281, [0, 1, 4, 3, 2]);  permute_281 = None
        view_249: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_283, [256, 64, 1024]);  permute_283 = None
        bmm_53: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_248, view_249)
        view_250: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_53, [16, 16, 512, 1, 1024]);  bmm_53 = None
        permute_284: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_250, [0, 1, 2, 4, 3]);  view_250 = None
        view_251: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_284, [16, 16, 512, 1024]);  permute_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_252: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_251, [16, 16, 1024, 512]);  view_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_7: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_252, 2, 1, 9223372036854775807);  view_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_253: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_7, [16, 16, 512, 1023]);  slice_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_6: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_253, [None, None, None, iota_2]);  view_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_68: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_247, index_6);  view_247 = index_6 = None
        add_69: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_68, 0);  add_68 = None

        # No stacktrace found for following nodes
        mul_tensor_68: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_69, 0.125)
        convert_element_type_default_58: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_68, torch.float32);  mul_tensor_68 = None
        convert_element_type_default_59: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_69, torch.float32)
        mul_tensor_69: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_59, 1);  convert_element_type_default_59 = None
        amax_default_34: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_69, [3], True)
        sub_tensor_34: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_69, amax_default_34);  mul_tensor_69 = None
        mul_tensor_70: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_34, 0.125);  sub_tensor_34 = None
        amax_default_35: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_58, [3], True)
        sub_tensor_35: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_58, amax_default_35)
        abs_default_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_58)
        ne_scalar_17: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_17, inf);  abs_default_17 = None
        eq_tensor_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_58, convert_element_type_default_58);  convert_element_type_default_58 = None
        mul_tensor_71: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_18, ne_scalar_17);  eq_tensor_18 = ne_scalar_17 = None
        logical_not_default_34: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_71);  mul_tensor_71 = None
        any_dims_17: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_34, [3], True);  logical_not_default_34 = None
        logical_not_default_35: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_17);  any_dims_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_35, mul_tensor_70, sub_tensor_35);  mul_tensor_70 = sub_tensor_35 = None
        exp_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_18);  where_self_18 = None
        sum_7: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_6, [3], True)
        div_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_26: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 26)
        inductor_random_default_72: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_26, 'rand');  inductor_lookup_seed_default_26 = None
        gt_26: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_72, 0.1);  inductor_random_default_72 = None
        mul_103: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_26, div_7);  div_7 = None
        mul_104: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_103, 1.1111111111111112);  mul_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_286: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_104, torch.bfloat16);  mul_104 = None
        unsqueeze_173: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_286, 4);  convert_element_type_286 = None
        unsqueeze_174: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_239, 4);  view_239 = None
        permute_286: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_174, [4, 1, 2, 3, 0]);  unsqueeze_174 = None
        view_254: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_173, [256, 512, 512]);  unsqueeze_173 = None
        permute_288: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_286, [1, 2, 4, 3, 0]);  permute_286 = None
        view_255: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_288, [256, 512, 64]);  permute_288 = None
        bmm_54: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_254, view_255)
        view_256: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_54, [16, 16, 512, 1, 64]);  bmm_54 = None
        permute_289: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_256, [2, 0, 1, 4, 3]);  view_256 = None
        view_257: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_289, [512, 16, 16, 64]);  permute_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_289: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_99, torch.bfloat16);  primals_99 = None
        unsqueeze_175: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_257, 4);  view_257 = None
        permute_290: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_175, [0, 1, 4, 3, 2]);  unsqueeze_175 = None
        unsqueeze_176: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_289, 3);  convert_element_type_289 = None
        unsqueeze_177: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_176, 4);  unsqueeze_176 = None
        permute_291: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_177, [3, 4, 0, 2, 1]);  unsqueeze_177 = None
        permute_292: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_290, [0, 1, 3, 4, 2]);  permute_290 = None
        clone_13: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_292, memory_format = torch.contiguous_format);  permute_292 = None
        view_258: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_13, [1, 8192, 1024]);  clone_13 = None
        permute_293: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_291, [3, 4, 2, 0, 1]);  permute_291 = None
        clone_14: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_259: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_14, [1, 1024, 1024]);  clone_14 = None
        squeeze_dim_602: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_258, 0)
        squeeze_dim_603: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_259, 0)
        mm_default_301: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_602, squeeze_dim_603);  squeeze_dim_602 = squeeze_dim_603 = None
        unsqueeze_default_301: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_301, 0);  mm_default_301 = None
        view_260: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_301, [512, 16, 1, 1, 1024]);  unsqueeze_default_301 = None
        permute_294: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_260, [0, 1, 4, 2, 3]);  view_260 = None
        view_261: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_294, [512, 16, 1024]);  permute_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_27: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 27)
        inductor_random_default_71: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_27, 'rand');  inductor_lookup_seed_default_27 = None
        convert_element_type_default_125: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_71, torch.bfloat16);  inductor_random_default_71 = None
        gt_27: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_125, 0.1);  convert_element_type_default_125 = None
        mul_105: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_27, view_261);  view_261 = None
        mul_106: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_105, 1.1111111111111112);  mul_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_70: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_106, add_65);  mul_106 = add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_24: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_12[0]
        getitem_25: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_12[1];  var_mean_12 = None
        add_71: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_24, 1e-12);  getitem_24 = None
        rsqrt_12: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        sub_19: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_70, getitem_25);  add_70 = getitem_25 = None
        mul_107: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_12);  sub_19 = None
        mul_108: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_107, primals_100)
        add_72: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_108, primals_101);  mul_108 = primals_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_292: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_103, torch.bfloat16);  primals_103 = None
        convert_element_type_293: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_102, torch.bfloat16);  primals_102 = None
        convert_element_type_294: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        view_262: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_294, [8192, 1024]);  convert_element_type_294 = None
        permute_295: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_293, [1, 0]);  convert_element_type_293 = None
        addmm_12: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_292, view_262, permute_295);  convert_element_type_292 = None
        view_263: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_12, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_298: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_263, torch.float32);  view_263 = None
        mul_109: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, 0.5)
        mul_110: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_298, 0.7071067811865476);  convert_element_type_298 = None
        erf_6: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_110);  mul_110 = None
        add_73: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_111: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_109, add_73);  mul_109 = add_73 = None
        convert_element_type_299: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_111, torch.bfloat16);  mul_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_28: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 28)
        inductor_random_default_70: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_28, 'rand');  inductor_lookup_seed_default_28 = None
        convert_element_type_default_124: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_70, torch.bfloat16);  inductor_random_default_70 = None
        gt_28: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_124, 0.1);  convert_element_type_default_124 = None
        mul_112: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_28, convert_element_type_299);  convert_element_type_299 = None
        mul_113: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_112, 1.1111111111111112);  mul_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_300: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_105, torch.bfloat16);  primals_105 = None
        convert_element_type_301: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_104, torch.bfloat16);  primals_104 = None
        view_264: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_113, [8192, 4096]);  mul_113 = None
        permute_296: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        addmm_13: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_300, view_264, permute_296);  convert_element_type_300 = None
        view_265: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_13, [512, 16, 1024]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_29: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 29)
        inductor_random_default_69: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_29, 'rand');  inductor_lookup_seed_default_29 = None
        convert_element_type_default_123: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_69, torch.bfloat16);  inductor_random_default_69 = None
        gt_29: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_123, 0.1);  convert_element_type_default_123 = None
        mul_114: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_29, view_265);  view_265 = None
        mul_115: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_114, 1.1111111111111112);  mul_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_74: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_115, add_72);  mul_115 = add_72 = None
        var_mean_13 = torch.ops.aten.var_mean.correction(add_74, [2], correction = 0, keepdim = True)
        getitem_26: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_13[0]
        getitem_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_13[1];  var_mean_13 = None
        add_75: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_26, 1e-12);  getitem_26 = None
        rsqrt_13: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_75);  add_75 = None
        sub_20: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_74, getitem_27);  add_74 = getitem_27 = None
        mul_116: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_13);  sub_20 = None
        mul_117: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_116, primals_106)
        add_76: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_117, primals_107);  mul_117 = primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_305: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16)
        convert_element_type_306: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_108, torch.bfloat16);  primals_108 = None
        unsqueeze_178: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_305, 3);  convert_element_type_305 = None
        unsqueeze_179: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_178, 4);  unsqueeze_178 = None
        unsqueeze_180: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_306, 3);  convert_element_type_306 = None
        unsqueeze_181: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_180, 4);  unsqueeze_180 = None
        view_266: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_179, [1, 8192, 1024]);  unsqueeze_179 = None
        view_267: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_181, [1, 1024, 1024]);  unsqueeze_181 = None
        squeeze_dim_600: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_266, 0)
        squeeze_dim_601: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_267, 0)
        mm_default_300: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_600, squeeze_dim_601);  squeeze_dim_601 = None
        unsqueeze_default_300: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_300, 0);  mm_default_300 = None
        view_268: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_300, [512, 16, 1, 16, 64]);  unsqueeze_default_300 = None
        permute_301: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_268, [0, 1, 3, 4, 2]);  view_268 = None
        view_269: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_301, [512, 16, 16, 64]);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_310: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_109, torch.bfloat16);  primals_109 = None
        unsqueeze_184: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_310, 3);  convert_element_type_310 = None
        unsqueeze_185: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_184, 4);  unsqueeze_184 = None
        view_271: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_185, [1, 1024, 1024]);  unsqueeze_185 = None
        squeeze_dim_599: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_271, 0)
        mm_default_299: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_600, squeeze_dim_599);  squeeze_dim_599 = None
        unsqueeze_default_299: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_299, 0);  mm_default_299 = None
        view_272: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_299, [512, 16, 1, 16, 64]);  unsqueeze_default_299 = None
        permute_306: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_272, [0, 1, 3, 4, 2]);  view_272 = None
        view_273: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_306, [512, 16, 16, 64]);  permute_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_314: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_110, torch.bfloat16);  primals_110 = None
        unsqueeze_188: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_314, 3);  convert_element_type_314 = None
        unsqueeze_189: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_188, 4);  unsqueeze_188 = None
        view_275: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_189, [1, 1024, 1024]);  unsqueeze_189 = None
        squeeze_dim_597: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_275, 0)
        mm_default_298: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_600, squeeze_dim_597);  squeeze_dim_600 = squeeze_dim_597 = None
        unsqueeze_default_298: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_298, 0);  mm_default_298 = None
        view_276: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_298, [512, 16, 1, 16, 64]);  unsqueeze_default_298 = None
        permute_311: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_276, [0, 1, 3, 4, 2]);  view_276 = None
        view_277: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_311, [512, 16, 16, 64]);  permute_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_319: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_111, torch.bfloat16);  primals_111 = None
        unsqueeze_192: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_319, 3);  convert_element_type_319 = None
        unsqueeze_193: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_192, 4);  unsqueeze_192 = None
        view_279: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_193, [1, 1024, 1024]);  unsqueeze_193 = None
        squeeze_dim_595: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_279, 0);  view_279 = None
        mm_default_297: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_595);  squeeze_dim_595 = None
        unsqueeze_default_297: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_297, 0);  mm_default_297 = None
        view_280: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_297, [1024, 16, 1, 16, 64]);  unsqueeze_default_297 = None
        permute_316: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_280, [0, 1, 3, 4, 2]);  view_280 = None
        view_281: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_316, [1024, 16, 16, 64]);  permute_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_77: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_269, primals_112);  primals_112 = None
        convert_element_type_322: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_77, torch.bfloat16);  add_77 = None
        unsqueeze_194: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_322, 4);  convert_element_type_322 = None
        permute_317: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_194, [1, 2, 0, 4, 3]);  unsqueeze_194 = None
        unsqueeze_195: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_273, 4);  view_273 = None
        permute_318: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_195, [1, 2, 4, 0, 3]);  unsqueeze_195 = None
        permute_319: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_317, [0, 1, 2, 4, 3]);  permute_317 = None
        view_282: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_319, [256, 512, 64]);  permute_319 = None
        permute_320: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_318, [0, 1, 4, 3, 2]);  permute_318 = None
        view_283: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_320, [256, 64, 512]);  permute_320 = None
        bmm_60: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_282, view_283)
        view_284: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_60, [16, 16, 512, 1, 512]);  bmm_60 = None
        permute_321: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_284, [0, 1, 2, 4, 3]);  view_284 = None
        view_285: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_321, [16, 16, 512, 512]);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_78: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_269, primals_113);  view_269 = primals_113 = None
        convert_element_type_325: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16);  add_78 = None
        unsqueeze_196: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_325, 4);  convert_element_type_325 = None
        permute_322: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_196, [1, 2, 0, 4, 3]);  unsqueeze_196 = None
        unsqueeze_197: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_281, 4);  view_281 = None
        permute_323: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_197, [1, 2, 4, 0, 3]);  unsqueeze_197 = None
        permute_324: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_322, [0, 1, 2, 4, 3]);  permute_322 = None
        view_286: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_324, [256, 512, 64]);  permute_324 = None
        permute_325: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_323, [0, 1, 4, 3, 2]);  permute_323 = None
        view_287: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_325, [256, 64, 1024]);  permute_325 = None
        bmm_61: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_286, view_287)
        view_288: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_61, [16, 16, 512, 1, 1024]);  bmm_61 = None
        permute_326: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_288, [0, 1, 2, 4, 3]);  view_288 = None
        view_289: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_326, [16, 16, 512, 1024]);  permute_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_290: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_289, [16, 16, 1024, 512]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_8: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_290, 2, 1, 9223372036854775807);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_291: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_8, [16, 16, 512, 1023]);  slice_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_7: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_291, [None, None, None, iota_2]);  view_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_79: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_285, index_7);  view_285 = index_7 = None
        add_80: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_79, 0);  add_79 = None

        # No stacktrace found for following nodes
        mul_tensor_64: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_80, 0.125)
        convert_element_type_default_56: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_64, torch.float32);  mul_tensor_64 = None
        convert_element_type_default_57: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_80, torch.float32)
        mul_tensor_65: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_57, 1);  convert_element_type_default_57 = None
        amax_default_32: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_65, [3], True)
        sub_tensor_32: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_65, amax_default_32);  mul_tensor_65 = None
        mul_tensor_66: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_32, 0.125);  sub_tensor_32 = None
        amax_default_33: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_56, [3], True)
        sub_tensor_33: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_56, amax_default_33)
        abs_default_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_56)
        ne_scalar_16: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_16, inf);  abs_default_16 = None
        eq_tensor_17: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_56, convert_element_type_default_56);  convert_element_type_default_56 = None
        mul_tensor_67: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_17, ne_scalar_16);  eq_tensor_17 = ne_scalar_16 = None
        logical_not_default_32: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_67);  mul_tensor_67 = None
        any_dims_16: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_32, [3], True);  logical_not_default_32 = None
        logical_not_default_33: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_16);  any_dims_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_33, mul_tensor_66, sub_tensor_33);  mul_tensor_66 = sub_tensor_33 = None
        exp_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_17);  where_self_17 = None
        sum_8: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_7, [3], True)
        div_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_30: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 30)
        inductor_random_default_68: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_30, 'rand');  inductor_lookup_seed_default_30 = None
        gt_30: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_68, 0.1);  inductor_random_default_68 = None
        mul_119: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_30, div_8);  div_8 = None
        mul_120: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_119, 1.1111111111111112);  mul_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_329: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_120, torch.bfloat16);  mul_120 = None
        unsqueeze_198: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_329, 4);  convert_element_type_329 = None
        unsqueeze_199: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_277, 4);  view_277 = None
        permute_328: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_199, [4, 1, 2, 3, 0]);  unsqueeze_199 = None
        view_292: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_198, [256, 512, 512]);  unsqueeze_198 = None
        permute_330: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_328, [1, 2, 4, 3, 0]);  permute_328 = None
        view_293: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_330, [256, 512, 64]);  permute_330 = None
        bmm_62: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_292, view_293)
        view_294: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_62, [16, 16, 512, 1, 64]);  bmm_62 = None
        permute_331: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_294, [2, 0, 1, 4, 3]);  view_294 = None
        view_295: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_331, [512, 16, 16, 64]);  permute_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_332: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_114, torch.bfloat16);  primals_114 = None
        unsqueeze_200: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_295, 4);  view_295 = None
        permute_332: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_200, [0, 1, 4, 3, 2]);  unsqueeze_200 = None
        unsqueeze_201: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_332, 3);  convert_element_type_332 = None
        unsqueeze_202: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_201, 4);  unsqueeze_201 = None
        permute_333: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_202, [3, 4, 0, 2, 1]);  unsqueeze_202 = None
        permute_334: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_332, [0, 1, 3, 4, 2]);  permute_332 = None
        clone_15: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_334, memory_format = torch.contiguous_format);  permute_334 = None
        view_296: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_15, [1, 8192, 1024]);  clone_15 = None
        permute_335: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_333, [3, 4, 2, 0, 1]);  permute_333 = None
        clone_16: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_335, memory_format = torch.contiguous_format);  permute_335 = None
        view_297: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_16, [1, 1024, 1024]);  clone_16 = None
        squeeze_dim_592: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_296, 0)
        squeeze_dim_593: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_297, 0)
        mm_default_296: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_592, squeeze_dim_593);  squeeze_dim_592 = squeeze_dim_593 = None
        unsqueeze_default_296: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_296, 0);  mm_default_296 = None
        view_298: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_296, [512, 16, 1, 1, 1024]);  unsqueeze_default_296 = None
        permute_336: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_298, [0, 1, 4, 2, 3]);  view_298 = None
        view_299: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_336, [512, 16, 1024]);  permute_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_31: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 31)
        inductor_random_default_67: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_31, 'rand');  inductor_lookup_seed_default_31 = None
        convert_element_type_default_122: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_67, torch.bfloat16);  inductor_random_default_67 = None
        gt_31: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_122, 0.1);  convert_element_type_default_122 = None
        mul_121: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_31, view_299);  view_299 = None
        mul_122: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_121, 1.1111111111111112);  mul_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_81: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_122, add_76);  mul_122 = add_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_28: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_14[0]
        getitem_29: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_14[1];  var_mean_14 = None
        add_82: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_28, 1e-12);  getitem_28 = None
        rsqrt_14: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        sub_22: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_81, getitem_29);  add_81 = getitem_29 = None
        mul_123: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_14);  sub_22 = None
        mul_124: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_123, primals_115)
        add_83: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_124, primals_116);  mul_124 = primals_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_335: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_118, torch.bfloat16);  primals_118 = None
        convert_element_type_336: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_117, torch.bfloat16);  primals_117 = None
        convert_element_type_337: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_83, torch.bfloat16)
        view_300: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_337, [8192, 1024]);  convert_element_type_337 = None
        permute_337: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_336, [1, 0]);  convert_element_type_336 = None
        addmm_14: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_335, view_300, permute_337);  convert_element_type_335 = None
        view_301: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_14, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_341: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_301, torch.float32);  view_301 = None
        mul_125: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_341, 0.5)
        mul_126: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_341, 0.7071067811865476);  convert_element_type_341 = None
        erf_7: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_126);  mul_126 = None
        add_84: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_127: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_125, add_84);  mul_125 = add_84 = None
        convert_element_type_342: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_127, torch.bfloat16);  mul_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_32: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 32)
        inductor_random_default_66: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_32, 'rand');  inductor_lookup_seed_default_32 = None
        convert_element_type_default_121: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_66, torch.bfloat16);  inductor_random_default_66 = None
        gt_32: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_121, 0.1);  convert_element_type_default_121 = None
        mul_128: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_32, convert_element_type_342);  convert_element_type_342 = None
        mul_129: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_128, 1.1111111111111112);  mul_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_343: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_120, torch.bfloat16);  primals_120 = None
        convert_element_type_344: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_119, torch.bfloat16);  primals_119 = None
        view_302: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_129, [8192, 4096]);  mul_129 = None
        permute_338: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_344, [1, 0]);  convert_element_type_344 = None
        addmm_15: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_343, view_302, permute_338);  convert_element_type_343 = None
        view_303: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_15, [512, 16, 1024]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_33: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 33)
        inductor_random_default_65: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_33, 'rand');  inductor_lookup_seed_default_33 = None
        convert_element_type_default_120: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_65, torch.bfloat16);  inductor_random_default_65 = None
        gt_33: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_120, 0.1);  convert_element_type_default_120 = None
        mul_130: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_33, view_303);  view_303 = None
        mul_131: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_130, 1.1111111111111112);  mul_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_85: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_131, add_83);  mul_131 = add_83 = None
        var_mean_15 = torch.ops.aten.var_mean.correction(add_85, [2], correction = 0, keepdim = True)
        getitem_30: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_15[0]
        getitem_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_15[1];  var_mean_15 = None
        add_86: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_30, 1e-12);  getitem_30 = None
        rsqrt_15: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_86);  add_86 = None
        sub_23: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_85, getitem_31);  add_85 = getitem_31 = None
        mul_132: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_15);  sub_23 = None
        mul_133: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_132, primals_121)
        add_87: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_133, primals_122);  mul_133 = primals_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_348: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_87, torch.bfloat16)
        convert_element_type_349: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_123, torch.bfloat16);  primals_123 = None
        unsqueeze_203: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_348, 3);  convert_element_type_348 = None
        unsqueeze_204: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_203, 4);  unsqueeze_203 = None
        unsqueeze_205: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_349, 3);  convert_element_type_349 = None
        unsqueeze_206: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_205, 4);  unsqueeze_205 = None
        view_304: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_204, [1, 8192, 1024]);  unsqueeze_204 = None
        view_305: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_206, [1, 1024, 1024]);  unsqueeze_206 = None
        squeeze_dim_590: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_304, 0)
        squeeze_dim_591: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_305, 0)
        mm_default_295: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_590, squeeze_dim_591);  squeeze_dim_591 = None
        unsqueeze_default_295: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_295, 0);  mm_default_295 = None
        view_306: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_295, [512, 16, 1, 16, 64]);  unsqueeze_default_295 = None
        permute_343: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_306, [0, 1, 3, 4, 2]);  view_306 = None
        view_307: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_343, [512, 16, 16, 64]);  permute_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_353: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_124, torch.bfloat16);  primals_124 = None
        unsqueeze_209: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_353, 3);  convert_element_type_353 = None
        unsqueeze_210: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_209, 4);  unsqueeze_209 = None
        view_309: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_210, [1, 1024, 1024]);  unsqueeze_210 = None
        squeeze_dim_589: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_309, 0)
        mm_default_294: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_590, squeeze_dim_589);  squeeze_dim_589 = None
        unsqueeze_default_294: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_294, 0);  mm_default_294 = None
        view_310: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_294, [512, 16, 1, 16, 64]);  unsqueeze_default_294 = None
        permute_348: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_310, [0, 1, 3, 4, 2]);  view_310 = None
        view_311: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_348, [512, 16, 16, 64]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_357: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_125, torch.bfloat16);  primals_125 = None
        unsqueeze_213: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_357, 3);  convert_element_type_357 = None
        unsqueeze_214: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_213, 4);  unsqueeze_213 = None
        view_313: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_214, [1, 1024, 1024]);  unsqueeze_214 = None
        squeeze_dim_587: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_313, 0)
        mm_default_293: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_590, squeeze_dim_587);  squeeze_dim_590 = squeeze_dim_587 = None
        unsqueeze_default_293: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_293, 0);  mm_default_293 = None
        view_314: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_293, [512, 16, 1, 16, 64]);  unsqueeze_default_293 = None
        permute_353: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_314, [0, 1, 3, 4, 2]);  view_314 = None
        view_315: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_353, [512, 16, 16, 64]);  permute_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_362: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_126, torch.bfloat16);  primals_126 = None
        unsqueeze_217: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_362, 3);  convert_element_type_362 = None
        unsqueeze_218: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_217, 4);  unsqueeze_217 = None
        view_317: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_218, [1, 1024, 1024]);  unsqueeze_218 = None
        squeeze_dim_585: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_317, 0);  view_317 = None
        mm_default_292: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_585);  squeeze_dim_585 = None
        unsqueeze_default_292: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_292, 0);  mm_default_292 = None
        view_318: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_292, [1024, 16, 1, 16, 64]);  unsqueeze_default_292 = None
        permute_358: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_318, [0, 1, 3, 4, 2]);  view_318 = None
        view_319: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_358, [1024, 16, 16, 64]);  permute_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_88: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_307, primals_127);  primals_127 = None
        convert_element_type_365: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None
        unsqueeze_219: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_365, 4);  convert_element_type_365 = None
        permute_359: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_219, [1, 2, 0, 4, 3]);  unsqueeze_219 = None
        unsqueeze_220: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_311, 4);  view_311 = None
        permute_360: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_220, [1, 2, 4, 0, 3]);  unsqueeze_220 = None
        permute_361: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_359, [0, 1, 2, 4, 3]);  permute_359 = None
        view_320: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_361, [256, 512, 64]);  permute_361 = None
        permute_362: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_360, [0, 1, 4, 3, 2]);  permute_360 = None
        view_321: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_362, [256, 64, 512]);  permute_362 = None
        bmm_68: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_320, view_321)
        view_322: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_68, [16, 16, 512, 1, 512]);  bmm_68 = None
        permute_363: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_322, [0, 1, 2, 4, 3]);  view_322 = None
        view_323: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_363, [16, 16, 512, 512]);  permute_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_89: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_307, primals_128);  view_307 = primals_128 = None
        convert_element_type_368: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_89, torch.bfloat16);  add_89 = None
        unsqueeze_221: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_368, 4);  convert_element_type_368 = None
        permute_364: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_221, [1, 2, 0, 4, 3]);  unsqueeze_221 = None
        unsqueeze_222: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_319, 4);  view_319 = None
        permute_365: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_222, [1, 2, 4, 0, 3]);  unsqueeze_222 = None
        permute_366: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_364, [0, 1, 2, 4, 3]);  permute_364 = None
        view_324: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_366, [256, 512, 64]);  permute_366 = None
        permute_367: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_365, [0, 1, 4, 3, 2]);  permute_365 = None
        view_325: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_367, [256, 64, 1024]);  permute_367 = None
        bmm_69: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_324, view_325)
        view_326: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_69, [16, 16, 512, 1, 1024]);  bmm_69 = None
        permute_368: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_326, [0, 1, 2, 4, 3]);  view_326 = None
        view_327: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_368, [16, 16, 512, 1024]);  permute_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_328: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_327, [16, 16, 1024, 512]);  view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_9: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_328, 2, 1, 9223372036854775807);  view_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_329: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_9, [16, 16, 512, 1023]);  slice_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_8: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_329, [None, None, None, iota_2]);  view_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_90: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_323, index_8);  view_323 = index_8 = None
        add_91: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_90, 0);  add_90 = None

        # No stacktrace found for following nodes
        mul_tensor_60: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_91, 0.125)
        convert_element_type_default_54: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_60, torch.float32);  mul_tensor_60 = None
        convert_element_type_default_55: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_91, torch.float32)
        mul_tensor_61: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_55, 1);  convert_element_type_default_55 = None
        amax_default_30: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_61, [3], True)
        sub_tensor_30: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_61, amax_default_30);  mul_tensor_61 = None
        mul_tensor_62: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_30, 0.125);  sub_tensor_30 = None
        amax_default_31: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_54, [3], True)
        sub_tensor_31: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_54, amax_default_31)
        abs_default_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_54)
        ne_scalar_15: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_15, inf);  abs_default_15 = None
        eq_tensor_16: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_54, convert_element_type_default_54);  convert_element_type_default_54 = None
        mul_tensor_63: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_16, ne_scalar_15);  eq_tensor_16 = ne_scalar_15 = None
        logical_not_default_30: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_63);  mul_tensor_63 = None
        any_dims_15: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_30, [3], True);  logical_not_default_30 = None
        logical_not_default_31: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_15);  any_dims_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_31, mul_tensor_62, sub_tensor_31);  mul_tensor_62 = sub_tensor_31 = None
        exp_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_16);  where_self_16 = None
        sum_9: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_8, [3], True)
        div_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_34: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 34)
        inductor_random_default_64: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_34, 'rand');  inductor_lookup_seed_default_34 = None
        gt_34: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_64, 0.1);  inductor_random_default_64 = None
        mul_135: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_34, div_9);  div_9 = None
        mul_136: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_135, 1.1111111111111112);  mul_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_372: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_136, torch.bfloat16);  mul_136 = None
        unsqueeze_223: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_372, 4);  convert_element_type_372 = None
        unsqueeze_224: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_315, 4);  view_315 = None
        permute_370: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_224, [4, 1, 2, 3, 0]);  unsqueeze_224 = None
        view_330: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_223, [256, 512, 512]);  unsqueeze_223 = None
        permute_372: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_370, [1, 2, 4, 3, 0]);  permute_370 = None
        view_331: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_372, [256, 512, 64]);  permute_372 = None
        bmm_70: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_330, view_331)
        view_332: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_70, [16, 16, 512, 1, 64]);  bmm_70 = None
        permute_373: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_332, [2, 0, 1, 4, 3]);  view_332 = None
        view_333: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_373, [512, 16, 16, 64]);  permute_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_375: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_129, torch.bfloat16);  primals_129 = None
        unsqueeze_225: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_333, 4);  view_333 = None
        permute_374: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_225, [0, 1, 4, 3, 2]);  unsqueeze_225 = None
        unsqueeze_226: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_375, 3);  convert_element_type_375 = None
        unsqueeze_227: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_226, 4);  unsqueeze_226 = None
        permute_375: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_227, [3, 4, 0, 2, 1]);  unsqueeze_227 = None
        permute_376: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_374, [0, 1, 3, 4, 2]);  permute_374 = None
        clone_17: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_376, memory_format = torch.contiguous_format);  permute_376 = None
        view_334: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_17, [1, 8192, 1024]);  clone_17 = None
        permute_377: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_375, [3, 4, 2, 0, 1]);  permute_375 = None
        clone_18: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_377, memory_format = torch.contiguous_format);  permute_377 = None
        view_335: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_18, [1, 1024, 1024]);  clone_18 = None
        squeeze_dim_582: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_334, 0)
        squeeze_dim_583: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_335, 0)
        mm_default_291: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_582, squeeze_dim_583);  squeeze_dim_582 = squeeze_dim_583 = None
        unsqueeze_default_291: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_291, 0);  mm_default_291 = None
        view_336: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_291, [512, 16, 1, 1, 1024]);  unsqueeze_default_291 = None
        permute_378: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_336, [0, 1, 4, 2, 3]);  view_336 = None
        view_337: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_378, [512, 16, 1024]);  permute_378 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_35: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 35)
        inductor_random_default_63: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_35, 'rand');  inductor_lookup_seed_default_35 = None
        convert_element_type_default_119: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_63, torch.bfloat16);  inductor_random_default_63 = None
        gt_35: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_119, 0.1);  convert_element_type_default_119 = None
        mul_137: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_35, view_337);  view_337 = None
        mul_138: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_137, 1.1111111111111112);  mul_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_92: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_138, add_87);  mul_138 = add_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_16[0]
        getitem_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_16[1];  var_mean_16 = None
        add_93: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_32, 1e-12);  getitem_32 = None
        rsqrt_16: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        sub_25: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_92, getitem_33);  add_92 = getitem_33 = None
        mul_139: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_25, rsqrt_16);  sub_25 = None
        mul_140: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_139, primals_130)
        add_94: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_140, primals_131);  mul_140 = primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_378: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_133, torch.bfloat16);  primals_133 = None
        convert_element_type_379: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_132, torch.bfloat16);  primals_132 = None
        convert_element_type_380: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16)
        view_338: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_380, [8192, 1024]);  convert_element_type_380 = None
        permute_379: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_379, [1, 0]);  convert_element_type_379 = None
        addmm_16: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_378, view_338, permute_379);  convert_element_type_378 = None
        view_339: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_16, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_384: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_339, torch.float32);  view_339 = None
        mul_141: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_384, 0.5)
        mul_142: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_384, 0.7071067811865476);  convert_element_type_384 = None
        erf_8: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_95: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_143: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_141, add_95);  mul_141 = add_95 = None
        convert_element_type_385: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_143, torch.bfloat16);  mul_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_36: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 36)
        inductor_random_default_62: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_36, 'rand');  inductor_lookup_seed_default_36 = None
        convert_element_type_default_118: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_62, torch.bfloat16);  inductor_random_default_62 = None
        gt_36: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_118, 0.1);  convert_element_type_default_118 = None
        mul_144: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_36, convert_element_type_385);  convert_element_type_385 = None
        mul_145: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_386: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_135, torch.bfloat16);  primals_135 = None
        convert_element_type_387: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_134, torch.bfloat16);  primals_134 = None
        view_340: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_145, [8192, 4096]);  mul_145 = None
        permute_380: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_387, [1, 0]);  convert_element_type_387 = None
        addmm_17: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_386, view_340, permute_380);  convert_element_type_386 = None
        view_341: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_17, [512, 16, 1024]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_37: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 37)
        inductor_random_default_61: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_37, 'rand');  inductor_lookup_seed_default_37 = None
        convert_element_type_default_117: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_61, torch.bfloat16);  inductor_random_default_61 = None
        gt_37: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_117, 0.1);  convert_element_type_default_117 = None
        mul_146: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_37, view_341);  view_341 = None
        mul_147: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_146, 1.1111111111111112);  mul_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_96: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_147, add_94);  mul_147 = add_94 = None
        var_mean_17 = torch.ops.aten.var_mean.correction(add_96, [2], correction = 0, keepdim = True)
        getitem_34: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_17[0]
        getitem_35: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_17[1];  var_mean_17 = None
        add_97: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_34, 1e-12);  getitem_34 = None
        rsqrt_17: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_97);  add_97 = None
        sub_26: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_96, getitem_35);  add_96 = getitem_35 = None
        mul_148: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_17);  sub_26 = None
        mul_149: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_148, primals_136)
        add_98: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_149, primals_137);  mul_149 = primals_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_391: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16)
        convert_element_type_392: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_138, torch.bfloat16);  primals_138 = None
        unsqueeze_228: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_391, 3);  convert_element_type_391 = None
        unsqueeze_229: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_228, 4);  unsqueeze_228 = None
        unsqueeze_230: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_392, 3);  convert_element_type_392 = None
        unsqueeze_231: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_230, 4);  unsqueeze_230 = None
        view_342: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_229, [1, 8192, 1024]);  unsqueeze_229 = None
        view_343: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_231, [1, 1024, 1024]);  unsqueeze_231 = None
        squeeze_dim_580: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_342, 0)
        squeeze_dim_581: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_343, 0)
        mm_default_290: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_580, squeeze_dim_581);  squeeze_dim_581 = None
        unsqueeze_default_290: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_290, 0);  mm_default_290 = None
        view_344: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_290, [512, 16, 1, 16, 64]);  unsqueeze_default_290 = None
        permute_385: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_344, [0, 1, 3, 4, 2]);  view_344 = None
        view_345: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_385, [512, 16, 16, 64]);  permute_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_396: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_139, torch.bfloat16);  primals_139 = None
        unsqueeze_234: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_396, 3);  convert_element_type_396 = None
        unsqueeze_235: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_234, 4);  unsqueeze_234 = None
        view_347: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_235, [1, 1024, 1024]);  unsqueeze_235 = None
        squeeze_dim_579: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_347, 0)
        mm_default_289: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_580, squeeze_dim_579);  squeeze_dim_579 = None
        unsqueeze_default_289: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_289, 0);  mm_default_289 = None
        view_348: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_289, [512, 16, 1, 16, 64]);  unsqueeze_default_289 = None
        permute_390: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_348, [0, 1, 3, 4, 2]);  view_348 = None
        view_349: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_390, [512, 16, 16, 64]);  permute_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_400: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_140, torch.bfloat16);  primals_140 = None
        unsqueeze_238: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_400, 3);  convert_element_type_400 = None
        unsqueeze_239: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_238, 4);  unsqueeze_238 = None
        view_351: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_239, [1, 1024, 1024]);  unsqueeze_239 = None
        squeeze_dim_577: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_351, 0)
        mm_default_288: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_580, squeeze_dim_577);  squeeze_dim_580 = squeeze_dim_577 = None
        unsqueeze_default_288: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_288, 0);  mm_default_288 = None
        view_352: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_288, [512, 16, 1, 16, 64]);  unsqueeze_default_288 = None
        permute_395: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_352, [0, 1, 3, 4, 2]);  view_352 = None
        view_353: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_395, [512, 16, 16, 64]);  permute_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_405: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_141, torch.bfloat16);  primals_141 = None
        unsqueeze_242: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_405, 3);  convert_element_type_405 = None
        unsqueeze_243: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_242, 4);  unsqueeze_242 = None
        view_355: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_243, [1, 1024, 1024]);  unsqueeze_243 = None
        squeeze_dim_575: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_355, 0);  view_355 = None
        mm_default_287: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_575);  squeeze_dim_575 = None
        unsqueeze_default_287: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_287, 0);  mm_default_287 = None
        view_356: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_287, [1024, 16, 1, 16, 64]);  unsqueeze_default_287 = None
        permute_400: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_356, [0, 1, 3, 4, 2]);  view_356 = None
        view_357: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_400, [1024, 16, 16, 64]);  permute_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_99: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_345, primals_142);  primals_142 = None
        convert_element_type_408: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_99, torch.bfloat16);  add_99 = None
        unsqueeze_244: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_408, 4);  convert_element_type_408 = None
        permute_401: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_244, [1, 2, 0, 4, 3]);  unsqueeze_244 = None
        unsqueeze_245: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_349, 4);  view_349 = None
        permute_402: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_245, [1, 2, 4, 0, 3]);  unsqueeze_245 = None
        permute_403: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_401, [0, 1, 2, 4, 3]);  permute_401 = None
        view_358: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_403, [256, 512, 64]);  permute_403 = None
        permute_404: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_402, [0, 1, 4, 3, 2]);  permute_402 = None
        view_359: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_404, [256, 64, 512]);  permute_404 = None
        bmm_76: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_358, view_359)
        view_360: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_76, [16, 16, 512, 1, 512]);  bmm_76 = None
        permute_405: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_360, [0, 1, 2, 4, 3]);  view_360 = None
        view_361: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_405, [16, 16, 512, 512]);  permute_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_100: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_345, primals_143);  view_345 = primals_143 = None
        convert_element_type_411: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16);  add_100 = None
        unsqueeze_246: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_411, 4);  convert_element_type_411 = None
        permute_406: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_246, [1, 2, 0, 4, 3]);  unsqueeze_246 = None
        unsqueeze_247: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_357, 4);  view_357 = None
        permute_407: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_247, [1, 2, 4, 0, 3]);  unsqueeze_247 = None
        permute_408: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_406, [0, 1, 2, 4, 3]);  permute_406 = None
        view_362: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_408, [256, 512, 64]);  permute_408 = None
        permute_409: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_407, [0, 1, 4, 3, 2]);  permute_407 = None
        view_363: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_409, [256, 64, 1024]);  permute_409 = None
        bmm_77: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_362, view_363)
        view_364: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_77, [16, 16, 512, 1, 1024]);  bmm_77 = None
        permute_410: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_364, [0, 1, 2, 4, 3]);  view_364 = None
        view_365: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_410, [16, 16, 512, 1024]);  permute_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_366: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_365, [16, 16, 1024, 512]);  view_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_10: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_366, 2, 1, 9223372036854775807);  view_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_367: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_10, [16, 16, 512, 1023]);  slice_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_9: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_367, [None, None, None, iota_2]);  view_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_101: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_361, index_9);  view_361 = index_9 = None
        add_102: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_101, 0);  add_101 = None

        # No stacktrace found for following nodes
        mul_tensor_56: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_102, 0.125)
        convert_element_type_default_52: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_56, torch.float32);  mul_tensor_56 = None
        convert_element_type_default_53: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_102, torch.float32)
        mul_tensor_57: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_53, 1);  convert_element_type_default_53 = None
        amax_default_28: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_57, [3], True)
        sub_tensor_28: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_57, amax_default_28);  mul_tensor_57 = None
        mul_tensor_58: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_28, 0.125);  sub_tensor_28 = None
        amax_default_29: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_52, [3], True)
        sub_tensor_29: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_52, amax_default_29)
        abs_default_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_52)
        ne_scalar_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_14, inf);  abs_default_14 = None
        eq_tensor_15: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_52, convert_element_type_default_52);  convert_element_type_default_52 = None
        mul_tensor_59: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_15, ne_scalar_14);  eq_tensor_15 = ne_scalar_14 = None
        logical_not_default_28: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_59);  mul_tensor_59 = None
        any_dims_14: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_28, [3], True);  logical_not_default_28 = None
        logical_not_default_29: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_14);  any_dims_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_29, mul_tensor_58, sub_tensor_29);  mul_tensor_58 = sub_tensor_29 = None
        exp_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_15);  where_self_15 = None
        sum_10: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_9, [3], True)
        div_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_38: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 38)
        inductor_random_default_60: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_38, 'rand');  inductor_lookup_seed_default_38 = None
        gt_38: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_60, 0.1);  inductor_random_default_60 = None
        mul_151: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_38, div_10);  div_10 = None
        mul_152: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_151, 1.1111111111111112);  mul_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_415: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_152, torch.bfloat16);  mul_152 = None
        unsqueeze_248: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_415, 4);  convert_element_type_415 = None
        unsqueeze_249: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_353, 4);  view_353 = None
        permute_412: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_249, [4, 1, 2, 3, 0]);  unsqueeze_249 = None
        view_368: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_248, [256, 512, 512]);  unsqueeze_248 = None
        permute_414: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_412, [1, 2, 4, 3, 0]);  permute_412 = None
        view_369: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_414, [256, 512, 64]);  permute_414 = None
        bmm_78: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_368, view_369)
        view_370: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_78, [16, 16, 512, 1, 64]);  bmm_78 = None
        permute_415: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_370, [2, 0, 1, 4, 3]);  view_370 = None
        view_371: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_415, [512, 16, 16, 64]);  permute_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_418: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_144, torch.bfloat16);  primals_144 = None
        unsqueeze_250: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_371, 4);  view_371 = None
        permute_416: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_250, [0, 1, 4, 3, 2]);  unsqueeze_250 = None
        unsqueeze_251: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_418, 3);  convert_element_type_418 = None
        unsqueeze_252: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_251, 4);  unsqueeze_251 = None
        permute_417: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_252, [3, 4, 0, 2, 1]);  unsqueeze_252 = None
        permute_418: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_416, [0, 1, 3, 4, 2]);  permute_416 = None
        clone_19: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None
        view_372: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_19, [1, 8192, 1024]);  clone_19 = None
        permute_419: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_417, [3, 4, 2, 0, 1]);  permute_417 = None
        clone_20: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_419, memory_format = torch.contiguous_format);  permute_419 = None
        view_373: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_20, [1, 1024, 1024]);  clone_20 = None
        squeeze_dim_572: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_372, 0)
        squeeze_dim_573: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_373, 0)
        mm_default_286: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_572, squeeze_dim_573);  squeeze_dim_572 = squeeze_dim_573 = None
        unsqueeze_default_286: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_286, 0);  mm_default_286 = None
        view_374: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_286, [512, 16, 1, 1, 1024]);  unsqueeze_default_286 = None
        permute_420: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_374, [0, 1, 4, 2, 3]);  view_374 = None
        view_375: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_420, [512, 16, 1024]);  permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_39: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 39)
        inductor_random_default_59: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_39, 'rand');  inductor_lookup_seed_default_39 = None
        convert_element_type_default_116: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_59, torch.bfloat16);  inductor_random_default_59 = None
        gt_39: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_116, 0.1);  convert_element_type_default_116 = None
        mul_153: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_39, view_375);  view_375 = None
        mul_154: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_153, 1.1111111111111112);  mul_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_103: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_154, add_98);  mul_154 = add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_36: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_18[0]
        getitem_37: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_18[1];  var_mean_18 = None
        add_104: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_36, 1e-12);  getitem_36 = None
        rsqrt_18: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        sub_28: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_103, getitem_37);  add_103 = getitem_37 = None
        mul_155: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_18);  sub_28 = None
        mul_156: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_155, primals_145)
        add_105: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_156, primals_146);  mul_156 = primals_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_421: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_148, torch.bfloat16);  primals_148 = None
        convert_element_type_422: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_147, torch.bfloat16);  primals_147 = None
        convert_element_type_423: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_105, torch.bfloat16)
        view_376: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_423, [8192, 1024]);  convert_element_type_423 = None
        permute_421: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_422, [1, 0]);  convert_element_type_422 = None
        addmm_18: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_421, view_376, permute_421);  convert_element_type_421 = None
        view_377: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_18, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_427: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_377, torch.float32);  view_377 = None
        mul_157: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_427, 0.5)
        mul_158: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_427, 0.7071067811865476);  convert_element_type_427 = None
        erf_9: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_158);  mul_158 = None
        add_106: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_159: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_157, add_106);  mul_157 = add_106 = None
        convert_element_type_428: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_159, torch.bfloat16);  mul_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_40: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 40)
        inductor_random_default_58: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_40, 'rand');  inductor_lookup_seed_default_40 = None
        convert_element_type_default_115: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_58, torch.bfloat16);  inductor_random_default_58 = None
        gt_40: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_115, 0.1);  convert_element_type_default_115 = None
        mul_160: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_40, convert_element_type_428);  convert_element_type_428 = None
        mul_161: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_160, 1.1111111111111112);  mul_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_429: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_150, torch.bfloat16);  primals_150 = None
        convert_element_type_430: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_149, torch.bfloat16);  primals_149 = None
        view_378: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_161, [8192, 4096]);  mul_161 = None
        permute_422: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_430, [1, 0]);  convert_element_type_430 = None
        addmm_19: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_429, view_378, permute_422);  convert_element_type_429 = None
        view_379: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_19, [512, 16, 1024]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_41: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 41)
        inductor_random_default_57: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_41, 'rand');  inductor_lookup_seed_default_41 = None
        convert_element_type_default_114: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_57, torch.bfloat16);  inductor_random_default_57 = None
        gt_41: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_114, 0.1);  convert_element_type_default_114 = None
        mul_162: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_41, view_379);  view_379 = None
        mul_163: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_162, 1.1111111111111112);  mul_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_107: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_163, add_105);  mul_163 = add_105 = None
        var_mean_19 = torch.ops.aten.var_mean.correction(add_107, [2], correction = 0, keepdim = True)
        getitem_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_19[0]
        getitem_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_19[1];  var_mean_19 = None
        add_108: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_38, 1e-12);  getitem_38 = None
        rsqrt_19: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_108);  add_108 = None
        sub_29: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_107, getitem_39);  add_107 = getitem_39 = None
        mul_164: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_29, rsqrt_19);  sub_29 = None
        mul_165: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_164, primals_151)
        add_109: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_165, primals_152);  mul_165 = primals_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_434: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_109, torch.bfloat16)
        convert_element_type_435: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_153, torch.bfloat16);  primals_153 = None
        unsqueeze_253: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_434, 3);  convert_element_type_434 = None
        unsqueeze_254: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_253, 4);  unsqueeze_253 = None
        unsqueeze_255: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_435, 3);  convert_element_type_435 = None
        unsqueeze_256: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_255, 4);  unsqueeze_255 = None
        view_380: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_254, [1, 8192, 1024]);  unsqueeze_254 = None
        view_381: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_256, [1, 1024, 1024]);  unsqueeze_256 = None
        squeeze_dim_570: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_380, 0)
        squeeze_dim_571: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_381, 0)
        mm_default_285: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_570, squeeze_dim_571);  squeeze_dim_571 = None
        unsqueeze_default_285: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_285, 0);  mm_default_285 = None
        view_382: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_285, [512, 16, 1, 16, 64]);  unsqueeze_default_285 = None
        permute_427: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_382, [0, 1, 3, 4, 2]);  view_382 = None
        view_383: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_427, [512, 16, 16, 64]);  permute_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_439: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_154, torch.bfloat16);  primals_154 = None
        unsqueeze_259: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_439, 3);  convert_element_type_439 = None
        unsqueeze_260: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_259, 4);  unsqueeze_259 = None
        view_385: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_260, [1, 1024, 1024]);  unsqueeze_260 = None
        squeeze_dim_569: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_385, 0)
        mm_default_284: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_570, squeeze_dim_569);  squeeze_dim_569 = None
        unsqueeze_default_284: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_284, 0);  mm_default_284 = None
        view_386: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_284, [512, 16, 1, 16, 64]);  unsqueeze_default_284 = None
        permute_432: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_386, [0, 1, 3, 4, 2]);  view_386 = None
        view_387: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_432, [512, 16, 16, 64]);  permute_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_443: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_155, torch.bfloat16);  primals_155 = None
        unsqueeze_263: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_443, 3);  convert_element_type_443 = None
        unsqueeze_264: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_263, 4);  unsqueeze_263 = None
        view_389: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_264, [1, 1024, 1024]);  unsqueeze_264 = None
        squeeze_dim_567: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_389, 0)
        mm_default_283: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_570, squeeze_dim_567);  squeeze_dim_570 = squeeze_dim_567 = None
        unsqueeze_default_283: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_283, 0);  mm_default_283 = None
        view_390: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_283, [512, 16, 1, 16, 64]);  unsqueeze_default_283 = None
        permute_437: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_390, [0, 1, 3, 4, 2]);  view_390 = None
        view_391: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_437, [512, 16, 16, 64]);  permute_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_448: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_156, torch.bfloat16);  primals_156 = None
        unsqueeze_267: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_448, 3);  convert_element_type_448 = None
        unsqueeze_268: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_267, 4);  unsqueeze_267 = None
        view_393: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_268, [1, 1024, 1024]);  unsqueeze_268 = None
        squeeze_dim_565: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_393, 0);  view_393 = None
        mm_default_282: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_565);  squeeze_dim_565 = None
        unsqueeze_default_282: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_282, 0);  mm_default_282 = None
        view_394: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_282, [1024, 16, 1, 16, 64]);  unsqueeze_default_282 = None
        permute_442: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_394, [0, 1, 3, 4, 2]);  view_394 = None
        view_395: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_442, [1024, 16, 16, 64]);  permute_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_110: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_383, primals_157);  primals_157 = None
        convert_element_type_451: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16);  add_110 = None
        unsqueeze_269: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_451, 4);  convert_element_type_451 = None
        permute_443: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_269, [1, 2, 0, 4, 3]);  unsqueeze_269 = None
        unsqueeze_270: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_387, 4);  view_387 = None
        permute_444: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_270, [1, 2, 4, 0, 3]);  unsqueeze_270 = None
        permute_445: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_443, [0, 1, 2, 4, 3]);  permute_443 = None
        view_396: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_445, [256, 512, 64]);  permute_445 = None
        permute_446: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_444, [0, 1, 4, 3, 2]);  permute_444 = None
        view_397: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_446, [256, 64, 512]);  permute_446 = None
        bmm_84: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_396, view_397)
        view_398: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_84, [16, 16, 512, 1, 512]);  bmm_84 = None
        permute_447: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_398, [0, 1, 2, 4, 3]);  view_398 = None
        view_399: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_447, [16, 16, 512, 512]);  permute_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_111: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_383, primals_158);  view_383 = primals_158 = None
        convert_element_type_454: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_111, torch.bfloat16);  add_111 = None
        unsqueeze_271: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_454, 4);  convert_element_type_454 = None
        permute_448: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_271, [1, 2, 0, 4, 3]);  unsqueeze_271 = None
        unsqueeze_272: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_395, 4);  view_395 = None
        permute_449: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_272, [1, 2, 4, 0, 3]);  unsqueeze_272 = None
        permute_450: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_448, [0, 1, 2, 4, 3]);  permute_448 = None
        view_400: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_450, [256, 512, 64]);  permute_450 = None
        permute_451: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_449, [0, 1, 4, 3, 2]);  permute_449 = None
        view_401: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_451, [256, 64, 1024]);  permute_451 = None
        bmm_85: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_400, view_401)
        view_402: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_85, [16, 16, 512, 1, 1024]);  bmm_85 = None
        permute_452: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_402, [0, 1, 2, 4, 3]);  view_402 = None
        view_403: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_452, [16, 16, 512, 1024]);  permute_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_404: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_403, [16, 16, 1024, 512]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_11: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_404, 2, 1, 9223372036854775807);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_405: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_11, [16, 16, 512, 1023]);  slice_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_10: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_405, [None, None, None, iota_2]);  view_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_112: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_399, index_10);  view_399 = index_10 = None
        add_113: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_112, 0);  add_112 = None

        # No stacktrace found for following nodes
        mul_tensor_52: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_113, 0.125)
        convert_element_type_default_50: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_52, torch.float32);  mul_tensor_52 = None
        convert_element_type_default_51: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_113, torch.float32)
        mul_tensor_53: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_51, 1);  convert_element_type_default_51 = None
        amax_default_26: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_53, [3], True)
        sub_tensor_26: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_53, amax_default_26);  mul_tensor_53 = None
        mul_tensor_54: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_26, 0.125);  sub_tensor_26 = None
        amax_default_27: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_50, [3], True)
        sub_tensor_27: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_50, amax_default_27)
        abs_default_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_50)
        ne_scalar_13: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_13, inf);  abs_default_13 = None
        eq_tensor_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_50, convert_element_type_default_50);  convert_element_type_default_50 = None
        mul_tensor_55: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_14, ne_scalar_13);  eq_tensor_14 = ne_scalar_13 = None
        logical_not_default_26: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_55);  mul_tensor_55 = None
        any_dims_13: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_26, [3], True);  logical_not_default_26 = None
        logical_not_default_27: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_13);  any_dims_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_27, mul_tensor_54, sub_tensor_27);  mul_tensor_54 = sub_tensor_27 = None
        exp_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_14);  where_self_14 = None
        sum_11: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_10, [3], True)
        div_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_42: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 42)
        inductor_random_default_56: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_42, 'rand');  inductor_lookup_seed_default_42 = None
        gt_42: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_56, 0.1);  inductor_random_default_56 = None
        mul_167: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_42, div_11);  div_11 = None
        mul_168: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_167, 1.1111111111111112);  mul_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_458: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_168, torch.bfloat16);  mul_168 = None
        unsqueeze_273: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_458, 4);  convert_element_type_458 = None
        unsqueeze_274: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_391, 4);  view_391 = None
        permute_454: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_274, [4, 1, 2, 3, 0]);  unsqueeze_274 = None
        view_406: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_273, [256, 512, 512]);  unsqueeze_273 = None
        permute_456: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_454, [1, 2, 4, 3, 0]);  permute_454 = None
        view_407: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_456, [256, 512, 64]);  permute_456 = None
        bmm_86: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_406, view_407)
        view_408: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_86, [16, 16, 512, 1, 64]);  bmm_86 = None
        permute_457: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_408, [2, 0, 1, 4, 3]);  view_408 = None
        view_409: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_457, [512, 16, 16, 64]);  permute_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_461: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_159, torch.bfloat16);  primals_159 = None
        unsqueeze_275: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_409, 4);  view_409 = None
        permute_458: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_275, [0, 1, 4, 3, 2]);  unsqueeze_275 = None
        unsqueeze_276: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_461, 3);  convert_element_type_461 = None
        unsqueeze_277: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_276, 4);  unsqueeze_276 = None
        permute_459: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_277, [3, 4, 0, 2, 1]);  unsqueeze_277 = None
        permute_460: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_458, [0, 1, 3, 4, 2]);  permute_458 = None
        clone_21: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_460, memory_format = torch.contiguous_format);  permute_460 = None
        view_410: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_21, [1, 8192, 1024]);  clone_21 = None
        permute_461: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_459, [3, 4, 2, 0, 1]);  permute_459 = None
        clone_22: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_461, memory_format = torch.contiguous_format);  permute_461 = None
        view_411: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_22, [1, 1024, 1024]);  clone_22 = None
        squeeze_dim_562: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_410, 0)
        squeeze_dim_563: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_411, 0)
        mm_default_281: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_562, squeeze_dim_563);  squeeze_dim_562 = squeeze_dim_563 = None
        unsqueeze_default_281: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_281, 0);  mm_default_281 = None
        view_412: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_281, [512, 16, 1, 1, 1024]);  unsqueeze_default_281 = None
        permute_462: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_412, [0, 1, 4, 2, 3]);  view_412 = None
        view_413: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_462, [512, 16, 1024]);  permute_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_43: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 43)
        inductor_random_default_55: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_43, 'rand');  inductor_lookup_seed_default_43 = None
        convert_element_type_default_113: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_55, torch.bfloat16);  inductor_random_default_55 = None
        gt_43: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_113, 0.1);  convert_element_type_default_113 = None
        mul_169: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_43, view_413);  view_413 = None
        mul_170: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_169, 1.1111111111111112);  mul_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_114: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_170, add_109);  mul_170 = add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
        getitem_40: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_20[0]
        getitem_41: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_20[1];  var_mean_20 = None
        add_115: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_40, 1e-12);  getitem_40 = None
        rsqrt_20: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        sub_31: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_114, getitem_41);  add_114 = getitem_41 = None
        mul_171: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_20);  sub_31 = None
        mul_172: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_171, primals_160)
        add_116: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_172, primals_161);  mul_172 = primals_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_464: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_163, torch.bfloat16);  primals_163 = None
        convert_element_type_465: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_162, torch.bfloat16);  primals_162 = None
        convert_element_type_466: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16)
        view_414: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_466, [8192, 1024]);  convert_element_type_466 = None
        permute_463: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_465, [1, 0]);  convert_element_type_465 = None
        addmm_20: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_464, view_414, permute_463);  convert_element_type_464 = None
        view_415: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_20, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_470: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_415, torch.float32);  view_415 = None
        mul_173: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_470, 0.5)
        mul_174: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_470, 0.7071067811865476);  convert_element_type_470 = None
        erf_10: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_117: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_175: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_173, add_117);  mul_173 = add_117 = None
        convert_element_type_471: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_175, torch.bfloat16);  mul_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_44: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 44)
        inductor_random_default_54: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_44, 'rand');  inductor_lookup_seed_default_44 = None
        convert_element_type_default_112: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_54, torch.bfloat16);  inductor_random_default_54 = None
        gt_44: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_112, 0.1);  convert_element_type_default_112 = None
        mul_176: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_44, convert_element_type_471);  convert_element_type_471 = None
        mul_177: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_176, 1.1111111111111112);  mul_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_472: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_165, torch.bfloat16);  primals_165 = None
        convert_element_type_473: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_164, torch.bfloat16);  primals_164 = None
        view_416: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_177, [8192, 4096]);  mul_177 = None
        permute_464: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_473, [1, 0]);  convert_element_type_473 = None
        addmm_21: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_472, view_416, permute_464);  convert_element_type_472 = None
        view_417: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_21, [512, 16, 1024]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_45: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 45)
        inductor_random_default_53: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_45, 'rand');  inductor_lookup_seed_default_45 = None
        convert_element_type_default_111: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_53, torch.bfloat16);  inductor_random_default_53 = None
        gt_45: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_111, 0.1);  convert_element_type_default_111 = None
        mul_178: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_45, view_417);  view_417 = None
        mul_179: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_178, 1.1111111111111112);  mul_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_118: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_179, add_116);  mul_179 = add_116 = None
        var_mean_21 = torch.ops.aten.var_mean.correction(add_118, [2], correction = 0, keepdim = True)
        getitem_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_21[0]
        getitem_43: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_21[1];  var_mean_21 = None
        add_119: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_42, 1e-12);  getitem_42 = None
        rsqrt_21: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_119);  add_119 = None
        sub_32: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_118, getitem_43);  add_118 = getitem_43 = None
        mul_180: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_21);  sub_32 = None
        mul_181: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_180, primals_166)
        add_120: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_181, primals_167);  mul_181 = primals_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_477: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)
        convert_element_type_478: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_168, torch.bfloat16);  primals_168 = None
        unsqueeze_278: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_477, 3);  convert_element_type_477 = None
        unsqueeze_279: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_278, 4);  unsqueeze_278 = None
        unsqueeze_280: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_478, 3);  convert_element_type_478 = None
        unsqueeze_281: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_280, 4);  unsqueeze_280 = None
        view_418: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_279, [1, 8192, 1024]);  unsqueeze_279 = None
        view_419: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_281, [1, 1024, 1024]);  unsqueeze_281 = None
        squeeze_dim_560: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_418, 0)
        squeeze_dim_561: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_419, 0)
        mm_default_280: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_560, squeeze_dim_561);  squeeze_dim_561 = None
        unsqueeze_default_280: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_280, 0);  mm_default_280 = None
        view_420: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_280, [512, 16, 1, 16, 64]);  unsqueeze_default_280 = None
        permute_469: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_420, [0, 1, 3, 4, 2]);  view_420 = None
        view_421: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_469, [512, 16, 16, 64]);  permute_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_482: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_169, torch.bfloat16);  primals_169 = None
        unsqueeze_284: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_482, 3);  convert_element_type_482 = None
        unsqueeze_285: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_284, 4);  unsqueeze_284 = None
        view_423: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_285, [1, 1024, 1024]);  unsqueeze_285 = None
        squeeze_dim_559: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_423, 0)
        mm_default_279: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_560, squeeze_dim_559);  squeeze_dim_559 = None
        unsqueeze_default_279: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_279, 0);  mm_default_279 = None
        view_424: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_279, [512, 16, 1, 16, 64]);  unsqueeze_default_279 = None
        permute_474: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_424, [0, 1, 3, 4, 2]);  view_424 = None
        view_425: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_474, [512, 16, 16, 64]);  permute_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_486: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_170, torch.bfloat16);  primals_170 = None
        unsqueeze_288: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_486, 3);  convert_element_type_486 = None
        unsqueeze_289: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_288, 4);  unsqueeze_288 = None
        view_427: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_289, [1, 1024, 1024]);  unsqueeze_289 = None
        squeeze_dim_557: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_427, 0)
        mm_default_278: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_560, squeeze_dim_557);  squeeze_dim_560 = squeeze_dim_557 = None
        unsqueeze_default_278: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_278, 0);  mm_default_278 = None
        view_428: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_278, [512, 16, 1, 16, 64]);  unsqueeze_default_278 = None
        permute_479: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_428, [0, 1, 3, 4, 2]);  view_428 = None
        view_429: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_479, [512, 16, 16, 64]);  permute_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_491: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_171, torch.bfloat16);  primals_171 = None
        unsqueeze_292: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_491, 3);  convert_element_type_491 = None
        unsqueeze_293: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_292, 4);  unsqueeze_292 = None
        view_431: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_293, [1, 1024, 1024]);  unsqueeze_293 = None
        squeeze_dim_555: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_431, 0);  view_431 = None
        mm_default_277: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_555);  squeeze_dim_555 = None
        unsqueeze_default_277: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_277, 0);  mm_default_277 = None
        view_432: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_277, [1024, 16, 1, 16, 64]);  unsqueeze_default_277 = None
        permute_484: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_432, [0, 1, 3, 4, 2]);  view_432 = None
        view_433: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_484, [1024, 16, 16, 64]);  permute_484 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_121: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_421, primals_172);  primals_172 = None
        convert_element_type_494: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_121, torch.bfloat16);  add_121 = None
        unsqueeze_294: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_494, 4);  convert_element_type_494 = None
        permute_485: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_294, [1, 2, 0, 4, 3]);  unsqueeze_294 = None
        unsqueeze_295: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_425, 4);  view_425 = None
        permute_486: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_295, [1, 2, 4, 0, 3]);  unsqueeze_295 = None
        permute_487: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_485, [0, 1, 2, 4, 3]);  permute_485 = None
        view_434: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_487, [256, 512, 64]);  permute_487 = None
        permute_488: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_486, [0, 1, 4, 3, 2]);  permute_486 = None
        view_435: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_488, [256, 64, 512]);  permute_488 = None
        bmm_92: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_434, view_435)
        view_436: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_92, [16, 16, 512, 1, 512]);  bmm_92 = None
        permute_489: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_436, [0, 1, 2, 4, 3]);  view_436 = None
        view_437: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_489, [16, 16, 512, 512]);  permute_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_122: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_421, primals_173);  view_421 = primals_173 = None
        convert_element_type_497: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16);  add_122 = None
        unsqueeze_296: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_497, 4);  convert_element_type_497 = None
        permute_490: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_296, [1, 2, 0, 4, 3]);  unsqueeze_296 = None
        unsqueeze_297: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_433, 4);  view_433 = None
        permute_491: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_297, [1, 2, 4, 0, 3]);  unsqueeze_297 = None
        permute_492: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_490, [0, 1, 2, 4, 3]);  permute_490 = None
        view_438: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_492, [256, 512, 64]);  permute_492 = None
        permute_493: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_491, [0, 1, 4, 3, 2]);  permute_491 = None
        view_439: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_493, [256, 64, 1024]);  permute_493 = None
        bmm_93: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_438, view_439)
        view_440: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_93, [16, 16, 512, 1, 1024]);  bmm_93 = None
        permute_494: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_440, [0, 1, 2, 4, 3]);  view_440 = None
        view_441: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_494, [16, 16, 512, 1024]);  permute_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_442: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_441, [16, 16, 1024, 512]);  view_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_12: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_442, 2, 1, 9223372036854775807);  view_442 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_443: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_12, [16, 16, 512, 1023]);  slice_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_11: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_443, [None, None, None, iota_2]);  view_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_123: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_437, index_11);  view_437 = index_11 = None
        add_124: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_123, 0);  add_123 = None

        # No stacktrace found for following nodes
        mul_tensor_48: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_124, 0.125)
        convert_element_type_default_48: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_48, torch.float32);  mul_tensor_48 = None
        convert_element_type_default_49: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_124, torch.float32)
        mul_tensor_49: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_49, 1);  convert_element_type_default_49 = None
        amax_default_24: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_49, [3], True)
        sub_tensor_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_49, amax_default_24);  mul_tensor_49 = None
        mul_tensor_50: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_24, 0.125);  sub_tensor_24 = None
        amax_default_25: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_48, [3], True)
        sub_tensor_25: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_48, amax_default_25)
        abs_default_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_48)
        ne_scalar_12: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_12, inf);  abs_default_12 = None
        eq_tensor_13: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_48, convert_element_type_default_48);  convert_element_type_default_48 = None
        mul_tensor_51: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_13, ne_scalar_12);  eq_tensor_13 = ne_scalar_12 = None
        logical_not_default_24: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_51);  mul_tensor_51 = None
        any_dims_12: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_24, [3], True);  logical_not_default_24 = None
        logical_not_default_25: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_12);  any_dims_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_25, mul_tensor_50, sub_tensor_25);  mul_tensor_50 = sub_tensor_25 = None
        exp_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_13);  where_self_13 = None
        sum_12: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_11, [3], True)
        div_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_46: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 46)
        inductor_random_default_52: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_46, 'rand');  inductor_lookup_seed_default_46 = None
        gt_46: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_52, 0.1);  inductor_random_default_52 = None
        mul_183: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_46, div_12);  div_12 = None
        mul_184: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_183, 1.1111111111111112);  mul_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_501: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_184, torch.bfloat16);  mul_184 = None
        unsqueeze_298: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_501, 4);  convert_element_type_501 = None
        unsqueeze_299: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_429, 4);  view_429 = None
        permute_496: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_299, [4, 1, 2, 3, 0]);  unsqueeze_299 = None
        view_444: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_298, [256, 512, 512]);  unsqueeze_298 = None
        permute_498: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_496, [1, 2, 4, 3, 0]);  permute_496 = None
        view_445: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_498, [256, 512, 64]);  permute_498 = None
        bmm_94: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_444, view_445)
        view_446: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_94, [16, 16, 512, 1, 64]);  bmm_94 = None
        permute_499: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_446, [2, 0, 1, 4, 3]);  view_446 = None
        view_447: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_499, [512, 16, 16, 64]);  permute_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_504: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_174, torch.bfloat16);  primals_174 = None
        unsqueeze_300: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_447, 4);  view_447 = None
        permute_500: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_300, [0, 1, 4, 3, 2]);  unsqueeze_300 = None
        unsqueeze_301: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_504, 3);  convert_element_type_504 = None
        unsqueeze_302: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_301, 4);  unsqueeze_301 = None
        permute_501: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_302, [3, 4, 0, 2, 1]);  unsqueeze_302 = None
        permute_502: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_500, [0, 1, 3, 4, 2]);  permute_500 = None
        clone_23: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_502, memory_format = torch.contiguous_format);  permute_502 = None
        view_448: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_23, [1, 8192, 1024]);  clone_23 = None
        permute_503: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_501, [3, 4, 2, 0, 1]);  permute_501 = None
        clone_24: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_503, memory_format = torch.contiguous_format);  permute_503 = None
        view_449: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_24, [1, 1024, 1024]);  clone_24 = None
        squeeze_dim_552: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_448, 0)
        squeeze_dim_553: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_449, 0)
        mm_default_276: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_552, squeeze_dim_553);  squeeze_dim_552 = squeeze_dim_553 = None
        unsqueeze_default_276: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_276, 0);  mm_default_276 = None
        view_450: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_276, [512, 16, 1, 1, 1024]);  unsqueeze_default_276 = None
        permute_504: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_450, [0, 1, 4, 2, 3]);  view_450 = None
        view_451: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_504, [512, 16, 1024]);  permute_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_47: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 47)
        inductor_random_default_51: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_47, 'rand');  inductor_lookup_seed_default_47 = None
        convert_element_type_default_110: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_51, torch.bfloat16);  inductor_random_default_51 = None
        gt_47: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_110, 0.1);  convert_element_type_default_110 = None
        mul_185: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_47, view_451);  view_451 = None
        mul_186: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_185, 1.1111111111111112);  mul_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_125: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_186, add_120);  mul_186 = add_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_22[0]
        getitem_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_22[1];  var_mean_22 = None
        add_126: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_44, 1e-12);  getitem_44 = None
        rsqrt_22: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        sub_34: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_125, getitem_45);  add_125 = getitem_45 = None
        mul_187: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_22);  sub_34 = None
        mul_188: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_187, primals_175)
        add_127: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_188, primals_176);  mul_188 = primals_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_507: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_178, torch.bfloat16);  primals_178 = None
        convert_element_type_508: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_177, torch.bfloat16);  primals_177 = None
        convert_element_type_509: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_127, torch.bfloat16)
        view_452: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_509, [8192, 1024]);  convert_element_type_509 = None
        permute_505: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_508, [1, 0]);  convert_element_type_508 = None
        addmm_22: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_507, view_452, permute_505);  convert_element_type_507 = None
        view_453: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_22, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_513: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_453, torch.float32);  view_453 = None
        mul_189: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_513, 0.5)
        mul_190: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_513, 0.7071067811865476);  convert_element_type_513 = None
        erf_11: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_190);  mul_190 = None
        add_128: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_191: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_189, add_128);  mul_189 = add_128 = None
        convert_element_type_514: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_191, torch.bfloat16);  mul_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_48: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 48)
        inductor_random_default_50: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_48, 'rand');  inductor_lookup_seed_default_48 = None
        convert_element_type_default_109: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_50, torch.bfloat16);  inductor_random_default_50 = None
        gt_48: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_109, 0.1);  convert_element_type_default_109 = None
        mul_192: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_48, convert_element_type_514);  convert_element_type_514 = None
        mul_193: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_192, 1.1111111111111112);  mul_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_515: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_180, torch.bfloat16);  primals_180 = None
        convert_element_type_516: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_179, torch.bfloat16);  primals_179 = None
        view_454: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_193, [8192, 4096]);  mul_193 = None
        permute_506: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_516, [1, 0]);  convert_element_type_516 = None
        addmm_23: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_515, view_454, permute_506);  convert_element_type_515 = None
        view_455: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_23, [512, 16, 1024]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_49: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 49)
        inductor_random_default_49: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_49, 'rand');  inductor_lookup_seed_default_49 = None
        convert_element_type_default_108: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_49, torch.bfloat16);  inductor_random_default_49 = None
        gt_49: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_108, 0.1);  convert_element_type_default_108 = None
        mul_194: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_49, view_455);  view_455 = None
        mul_195: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_194, 1.1111111111111112);  mul_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_129: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_195, add_127);  mul_195 = add_127 = None
        var_mean_23 = torch.ops.aten.var_mean.correction(add_129, [2], correction = 0, keepdim = True)
        getitem_46: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_23[0]
        getitem_47: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_23[1];  var_mean_23 = None
        add_130: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_46, 1e-12);  getitem_46 = None
        rsqrt_23: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_130);  add_130 = None
        sub_35: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_129, getitem_47);  add_129 = getitem_47 = None
        mul_196: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_23);  sub_35 = None
        mul_197: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_196, primals_181)
        add_131: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_197, primals_182);  mul_197 = primals_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_520: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_131, torch.bfloat16)
        convert_element_type_521: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_183, torch.bfloat16);  primals_183 = None
        unsqueeze_303: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_520, 3);  convert_element_type_520 = None
        unsqueeze_304: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_303, 4);  unsqueeze_303 = None
        unsqueeze_305: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_521, 3);  convert_element_type_521 = None
        unsqueeze_306: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_305, 4);  unsqueeze_305 = None
        view_456: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_304, [1, 8192, 1024]);  unsqueeze_304 = None
        view_457: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_306, [1, 1024, 1024]);  unsqueeze_306 = None
        squeeze_dim_550: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_456, 0)
        squeeze_dim_551: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_457, 0)
        mm_default_275: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_550, squeeze_dim_551);  squeeze_dim_551 = None
        unsqueeze_default_275: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_275, 0);  mm_default_275 = None
        view_458: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_275, [512, 16, 1, 16, 64]);  unsqueeze_default_275 = None
        permute_511: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_458, [0, 1, 3, 4, 2]);  view_458 = None
        view_459: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_511, [512, 16, 16, 64]);  permute_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_525: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_184, torch.bfloat16);  primals_184 = None
        unsqueeze_309: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_525, 3);  convert_element_type_525 = None
        unsqueeze_310: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_309, 4);  unsqueeze_309 = None
        view_461: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_310, [1, 1024, 1024]);  unsqueeze_310 = None
        squeeze_dim_549: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_461, 0)
        mm_default_274: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_550, squeeze_dim_549);  squeeze_dim_549 = None
        unsqueeze_default_274: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_274, 0);  mm_default_274 = None
        view_462: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_274, [512, 16, 1, 16, 64]);  unsqueeze_default_274 = None
        permute_516: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_462, [0, 1, 3, 4, 2]);  view_462 = None
        view_463: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_516, [512, 16, 16, 64]);  permute_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_529: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_185, torch.bfloat16);  primals_185 = None
        unsqueeze_313: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_529, 3);  convert_element_type_529 = None
        unsqueeze_314: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_313, 4);  unsqueeze_313 = None
        view_465: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_314, [1, 1024, 1024]);  unsqueeze_314 = None
        squeeze_dim_547: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_465, 0)
        mm_default_273: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_550, squeeze_dim_547);  squeeze_dim_550 = squeeze_dim_547 = None
        unsqueeze_default_273: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_273, 0);  mm_default_273 = None
        view_466: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_273, [512, 16, 1, 16, 64]);  unsqueeze_default_273 = None
        permute_521: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_466, [0, 1, 3, 4, 2]);  view_466 = None
        view_467: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_521, [512, 16, 16, 64]);  permute_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_534: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_186, torch.bfloat16);  primals_186 = None
        unsqueeze_317: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_534, 3);  convert_element_type_534 = None
        unsqueeze_318: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_317, 4);  unsqueeze_317 = None
        view_469: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_318, [1, 1024, 1024]);  unsqueeze_318 = None
        squeeze_dim_545: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_469, 0);  view_469 = None
        mm_default_272: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_545);  squeeze_dim_545 = None
        unsqueeze_default_272: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_272, 0);  mm_default_272 = None
        view_470: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_272, [1024, 16, 1, 16, 64]);  unsqueeze_default_272 = None
        permute_526: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_470, [0, 1, 3, 4, 2]);  view_470 = None
        view_471: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_526, [1024, 16, 16, 64]);  permute_526 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_132: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, primals_187);  primals_187 = None
        convert_element_type_537: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16);  add_132 = None
        unsqueeze_319: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_537, 4);  convert_element_type_537 = None
        permute_527: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_319, [1, 2, 0, 4, 3]);  unsqueeze_319 = None
        unsqueeze_320: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_463, 4);  view_463 = None
        permute_528: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_320, [1, 2, 4, 0, 3]);  unsqueeze_320 = None
        permute_529: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_527, [0, 1, 2, 4, 3]);  permute_527 = None
        view_472: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_529, [256, 512, 64]);  permute_529 = None
        permute_530: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_528, [0, 1, 4, 3, 2]);  permute_528 = None
        view_473: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_530, [256, 64, 512]);  permute_530 = None
        bmm_100: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_472, view_473)
        view_474: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_100, [16, 16, 512, 1, 512]);  bmm_100 = None
        permute_531: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_474, [0, 1, 2, 4, 3]);  view_474 = None
        view_475: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_531, [16, 16, 512, 512]);  permute_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_133: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_459, primals_188);  view_459 = primals_188 = None
        convert_element_type_540: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_133, torch.bfloat16);  add_133 = None
        unsqueeze_321: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_540, 4);  convert_element_type_540 = None
        permute_532: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_321, [1, 2, 0, 4, 3]);  unsqueeze_321 = None
        unsqueeze_322: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_471, 4);  view_471 = None
        permute_533: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_322, [1, 2, 4, 0, 3]);  unsqueeze_322 = None
        permute_534: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_532, [0, 1, 2, 4, 3]);  permute_532 = None
        view_476: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_534, [256, 512, 64]);  permute_534 = None
        permute_535: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_533, [0, 1, 4, 3, 2]);  permute_533 = None
        view_477: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_535, [256, 64, 1024]);  permute_535 = None
        bmm_101: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_476, view_477)
        view_478: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_101, [16, 16, 512, 1, 1024]);  bmm_101 = None
        permute_536: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_478, [0, 1, 2, 4, 3]);  view_478 = None
        view_479: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_536, [16, 16, 512, 1024]);  permute_536 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_480: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_479, [16, 16, 1024, 512]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_13: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_480, 2, 1, 9223372036854775807);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_481: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_13, [16, 16, 512, 1023]);  slice_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_12: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_481, [None, None, None, iota_2]);  view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_134: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_475, index_12);  view_475 = index_12 = None
        add_135: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_134, 0);  add_134 = None

        # No stacktrace found for following nodes
        mul_tensor_44: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_135, 0.125)
        convert_element_type_default_46: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_44, torch.float32);  mul_tensor_44 = None
        convert_element_type_default_47: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_135, torch.float32)
        mul_tensor_45: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_47, 1);  convert_element_type_default_47 = None
        amax_default_22: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_45, [3], True)
        sub_tensor_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_45, amax_default_22);  mul_tensor_45 = None
        mul_tensor_46: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_22, 0.125);  sub_tensor_22 = None
        amax_default_23: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_46, [3], True)
        sub_tensor_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_46, amax_default_23)
        abs_default_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_46)
        ne_scalar_11: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_11, inf);  abs_default_11 = None
        eq_tensor_12: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_46, convert_element_type_default_46);  convert_element_type_default_46 = None
        mul_tensor_47: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_12, ne_scalar_11);  eq_tensor_12 = ne_scalar_11 = None
        logical_not_default_22: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_47);  mul_tensor_47 = None
        any_dims_11: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_22, [3], True);  logical_not_default_22 = None
        logical_not_default_23: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_11);  any_dims_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_23, mul_tensor_46, sub_tensor_23);  mul_tensor_46 = sub_tensor_23 = None
        exp_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_12);  where_self_12 = None
        sum_13: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_12, [3], True)
        div_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_50: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 50)
        inductor_random_default_48: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_50, 'rand');  inductor_lookup_seed_default_50 = None
        gt_50: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_48, 0.1);  inductor_random_default_48 = None
        mul_199: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_50, div_13);  div_13 = None
        mul_200: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_199, 1.1111111111111112);  mul_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_544: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_200, torch.bfloat16);  mul_200 = None
        unsqueeze_323: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_544, 4);  convert_element_type_544 = None
        unsqueeze_324: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_467, 4);  view_467 = None
        permute_538: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_324, [4, 1, 2, 3, 0]);  unsqueeze_324 = None
        view_482: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_323, [256, 512, 512]);  unsqueeze_323 = None
        permute_540: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_538, [1, 2, 4, 3, 0]);  permute_538 = None
        view_483: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_540, [256, 512, 64]);  permute_540 = None
        bmm_102: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_482, view_483)
        view_484: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_102, [16, 16, 512, 1, 64]);  bmm_102 = None
        permute_541: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_484, [2, 0, 1, 4, 3]);  view_484 = None
        view_485: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_541, [512, 16, 16, 64]);  permute_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_547: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_189, torch.bfloat16);  primals_189 = None
        unsqueeze_325: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_485, 4);  view_485 = None
        permute_542: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_325, [0, 1, 4, 3, 2]);  unsqueeze_325 = None
        unsqueeze_326: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_547, 3);  convert_element_type_547 = None
        unsqueeze_327: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_326, 4);  unsqueeze_326 = None
        permute_543: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_327, [3, 4, 0, 2, 1]);  unsqueeze_327 = None
        permute_544: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_542, [0, 1, 3, 4, 2]);  permute_542 = None
        clone_25: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_544, memory_format = torch.contiguous_format);  permute_544 = None
        view_486: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_25, [1, 8192, 1024]);  clone_25 = None
        permute_545: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_543, [3, 4, 2, 0, 1]);  permute_543 = None
        clone_26: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_545, memory_format = torch.contiguous_format);  permute_545 = None
        view_487: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_26, [1, 1024, 1024]);  clone_26 = None
        squeeze_dim_542: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_486, 0)
        squeeze_dim_543: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_487, 0)
        mm_default_271: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_542, squeeze_dim_543);  squeeze_dim_542 = squeeze_dim_543 = None
        unsqueeze_default_271: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_271, 0);  mm_default_271 = None
        view_488: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_271, [512, 16, 1, 1, 1024]);  unsqueeze_default_271 = None
        permute_546: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_488, [0, 1, 4, 2, 3]);  view_488 = None
        view_489: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_546, [512, 16, 1024]);  permute_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_51: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 51)
        inductor_random_default_47: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_51, 'rand');  inductor_lookup_seed_default_51 = None
        convert_element_type_default_107: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_47, torch.bfloat16);  inductor_random_default_47 = None
        gt_51: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_107, 0.1);  convert_element_type_default_107 = None
        mul_201: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_51, view_489);  view_489 = None
        mul_202: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_201, 1.1111111111111112);  mul_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_136: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_202, add_131);  mul_202 = add_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_48: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_24[0]
        getitem_49: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_24[1];  var_mean_24 = None
        add_137: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_48, 1e-12);  getitem_48 = None
        rsqrt_24: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        sub_37: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_136, getitem_49);  add_136 = getitem_49 = None
        mul_203: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_37, rsqrt_24);  sub_37 = None
        mul_204: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_203, primals_190)
        add_138: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_204, primals_191);  mul_204 = primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_550: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_193, torch.bfloat16);  primals_193 = None
        convert_element_type_551: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_192, torch.bfloat16);  primals_192 = None
        convert_element_type_552: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16)
        view_490: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_552, [8192, 1024]);  convert_element_type_552 = None
        permute_547: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_551, [1, 0]);  convert_element_type_551 = None
        addmm_24: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_550, view_490, permute_547);  convert_element_type_550 = None
        view_491: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_24, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_556: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_491, torch.float32);  view_491 = None
        mul_205: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_556, 0.5)
        mul_206: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_556, 0.7071067811865476);  convert_element_type_556 = None
        erf_12: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_206);  mul_206 = None
        add_139: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_207: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_205, add_139);  mul_205 = add_139 = None
        convert_element_type_557: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_207, torch.bfloat16);  mul_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_52: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 52)
        inductor_random_default_46: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_52, 'rand');  inductor_lookup_seed_default_52 = None
        convert_element_type_default_106: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_46, torch.bfloat16);  inductor_random_default_46 = None
        gt_52: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_106, 0.1);  convert_element_type_default_106 = None
        mul_208: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_52, convert_element_type_557);  convert_element_type_557 = None
        mul_209: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_208, 1.1111111111111112);  mul_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_558: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_195, torch.bfloat16);  primals_195 = None
        convert_element_type_559: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_194, torch.bfloat16);  primals_194 = None
        view_492: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_209, [8192, 4096]);  mul_209 = None
        permute_548: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_559, [1, 0]);  convert_element_type_559 = None
        addmm_25: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_558, view_492, permute_548);  convert_element_type_558 = None
        view_493: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_25, [512, 16, 1024]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_53: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 53)
        inductor_random_default_45: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_53, 'rand');  inductor_lookup_seed_default_53 = None
        convert_element_type_default_105: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_45, torch.bfloat16);  inductor_random_default_45 = None
        gt_53: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_105, 0.1);  convert_element_type_default_105 = None
        mul_210: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_53, view_493);  view_493 = None
        mul_211: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_210, 1.1111111111111112);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_140: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_211, add_138);  mul_211 = add_138 = None
        var_mean_25 = torch.ops.aten.var_mean.correction(add_140, [2], correction = 0, keepdim = True)
        getitem_50: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_25[0]
        getitem_51: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_25[1];  var_mean_25 = None
        add_141: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_50, 1e-12);  getitem_50 = None
        rsqrt_25: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_141);  add_141 = None
        sub_38: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_140, getitem_51);  add_140 = getitem_51 = None
        mul_212: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_25);  sub_38 = None
        mul_213: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_212, primals_196)
        add_142: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_213, primals_197);  mul_213 = primals_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_563: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16)
        convert_element_type_564: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_198, torch.bfloat16);  primals_198 = None
        unsqueeze_328: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_563, 3);  convert_element_type_563 = None
        unsqueeze_329: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_328, 4);  unsqueeze_328 = None
        unsqueeze_330: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_564, 3);  convert_element_type_564 = None
        unsqueeze_331: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_330, 4);  unsqueeze_330 = None
        view_494: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_329, [1, 8192, 1024]);  unsqueeze_329 = None
        view_495: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_331, [1, 1024, 1024]);  unsqueeze_331 = None
        squeeze_dim_540: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_494, 0)
        squeeze_dim_541: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_495, 0)
        mm_default_270: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_540, squeeze_dim_541);  squeeze_dim_541 = None
        unsqueeze_default_270: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_270, 0);  mm_default_270 = None
        view_496: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_270, [512, 16, 1, 16, 64]);  unsqueeze_default_270 = None
        permute_553: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_496, [0, 1, 3, 4, 2]);  view_496 = None
        view_497: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_553, [512, 16, 16, 64]);  permute_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_568: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_199, torch.bfloat16);  primals_199 = None
        unsqueeze_334: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_568, 3);  convert_element_type_568 = None
        unsqueeze_335: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_334, 4);  unsqueeze_334 = None
        view_499: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_335, [1, 1024, 1024]);  unsqueeze_335 = None
        squeeze_dim_539: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_499, 0)
        mm_default_269: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_540, squeeze_dim_539);  squeeze_dim_539 = None
        unsqueeze_default_269: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_269, 0);  mm_default_269 = None
        view_500: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_269, [512, 16, 1, 16, 64]);  unsqueeze_default_269 = None
        permute_558: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_500, [0, 1, 3, 4, 2]);  view_500 = None
        view_501: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_558, [512, 16, 16, 64]);  permute_558 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_572: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_200, torch.bfloat16);  primals_200 = None
        unsqueeze_338: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_572, 3);  convert_element_type_572 = None
        unsqueeze_339: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_338, 4);  unsqueeze_338 = None
        view_503: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_339, [1, 1024, 1024]);  unsqueeze_339 = None
        squeeze_dim_537: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_503, 0)
        mm_default_268: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_540, squeeze_dim_537);  squeeze_dim_540 = squeeze_dim_537 = None
        unsqueeze_default_268: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_268, 0);  mm_default_268 = None
        view_504: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_268, [512, 16, 1, 16, 64]);  unsqueeze_default_268 = None
        permute_563: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_504, [0, 1, 3, 4, 2]);  view_504 = None
        view_505: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_563, [512, 16, 16, 64]);  permute_563 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_577: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_201, torch.bfloat16);  primals_201 = None
        unsqueeze_342: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_577, 3);  convert_element_type_577 = None
        unsqueeze_343: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_342, 4);  unsqueeze_342 = None
        view_507: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_343, [1, 1024, 1024]);  unsqueeze_343 = None
        squeeze_dim_535: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_507, 0);  view_507 = None
        mm_default_267: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_535);  squeeze_dim_535 = None
        unsqueeze_default_267: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_267, 0);  mm_default_267 = None
        view_508: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_267, [1024, 16, 1, 16, 64]);  unsqueeze_default_267 = None
        permute_568: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_508, [0, 1, 3, 4, 2]);  view_508 = None
        view_509: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_568, [1024, 16, 16, 64]);  permute_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_143: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_497, primals_202);  primals_202 = None
        convert_element_type_580: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_143, torch.bfloat16);  add_143 = None
        unsqueeze_344: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_580, 4);  convert_element_type_580 = None
        permute_569: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_344, [1, 2, 0, 4, 3]);  unsqueeze_344 = None
        unsqueeze_345: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_501, 4);  view_501 = None
        permute_570: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_345, [1, 2, 4, 0, 3]);  unsqueeze_345 = None
        permute_571: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_569, [0, 1, 2, 4, 3]);  permute_569 = None
        view_510: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_571, [256, 512, 64]);  permute_571 = None
        permute_572: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_570, [0, 1, 4, 3, 2]);  permute_570 = None
        view_511: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_572, [256, 64, 512]);  permute_572 = None
        bmm_108: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_510, view_511)
        view_512: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_108, [16, 16, 512, 1, 512]);  bmm_108 = None
        permute_573: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_512, [0, 1, 2, 4, 3]);  view_512 = None
        view_513: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_573, [16, 16, 512, 512]);  permute_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_144: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_497, primals_203);  view_497 = primals_203 = None
        convert_element_type_583: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        unsqueeze_346: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_583, 4);  convert_element_type_583 = None
        permute_574: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_346, [1, 2, 0, 4, 3]);  unsqueeze_346 = None
        unsqueeze_347: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_509, 4);  view_509 = None
        permute_575: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_347, [1, 2, 4, 0, 3]);  unsqueeze_347 = None
        permute_576: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_574, [0, 1, 2, 4, 3]);  permute_574 = None
        view_514: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_576, [256, 512, 64]);  permute_576 = None
        permute_577: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_575, [0, 1, 4, 3, 2]);  permute_575 = None
        view_515: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_577, [256, 64, 1024]);  permute_577 = None
        bmm_109: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_514, view_515)
        view_516: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_109, [16, 16, 512, 1, 1024]);  bmm_109 = None
        permute_578: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_516, [0, 1, 2, 4, 3]);  view_516 = None
        view_517: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_578, [16, 16, 512, 1024]);  permute_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_518: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_517, [16, 16, 1024, 512]);  view_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_14: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_518, 2, 1, 9223372036854775807);  view_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_519: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_14, [16, 16, 512, 1023]);  slice_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_13: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_519, [None, None, None, iota_2]);  view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_145: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_513, index_13);  view_513 = index_13 = None
        add_146: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_145, 0);  add_145 = None

        # No stacktrace found for following nodes
        mul_tensor_40: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_146, 0.125)
        convert_element_type_default_44: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_40, torch.float32);  mul_tensor_40 = None
        convert_element_type_default_45: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_146, torch.float32)
        mul_tensor_41: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_45, 1);  convert_element_type_default_45 = None
        amax_default_20: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_41, [3], True)
        sub_tensor_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_41, amax_default_20);  mul_tensor_41 = None
        mul_tensor_42: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_20, 0.125);  sub_tensor_20 = None
        amax_default_21: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_44, [3], True)
        sub_tensor_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_44, amax_default_21)
        abs_default_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_44)
        ne_scalar_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_10, inf);  abs_default_10 = None
        eq_tensor_11: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_44, convert_element_type_default_44);  convert_element_type_default_44 = None
        mul_tensor_43: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_11, ne_scalar_10);  eq_tensor_11 = ne_scalar_10 = None
        logical_not_default_20: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_43);  mul_tensor_43 = None
        any_dims_10: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_20, [3], True);  logical_not_default_20 = None
        logical_not_default_21: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_10);  any_dims_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_21, mul_tensor_42, sub_tensor_21);  mul_tensor_42 = sub_tensor_21 = None
        exp_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_11);  where_self_11 = None
        sum_14: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_13, [3], True)
        div_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_54: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 54)
        inductor_random_default_44: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_54, 'rand');  inductor_lookup_seed_default_54 = None
        gt_54: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_44, 0.1);  inductor_random_default_44 = None
        mul_215: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_54, div_14);  div_14 = None
        mul_216: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_215, 1.1111111111111112);  mul_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_587: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_216, torch.bfloat16);  mul_216 = None
        unsqueeze_348: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_587, 4);  convert_element_type_587 = None
        unsqueeze_349: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_505, 4);  view_505 = None
        permute_580: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_349, [4, 1, 2, 3, 0]);  unsqueeze_349 = None
        view_520: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_348, [256, 512, 512]);  unsqueeze_348 = None
        permute_582: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_580, [1, 2, 4, 3, 0]);  permute_580 = None
        view_521: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_582, [256, 512, 64]);  permute_582 = None
        bmm_110: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_520, view_521)
        view_522: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_110, [16, 16, 512, 1, 64]);  bmm_110 = None
        permute_583: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_522, [2, 0, 1, 4, 3]);  view_522 = None
        view_523: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_583, [512, 16, 16, 64]);  permute_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_590: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_204, torch.bfloat16);  primals_204 = None
        unsqueeze_350: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_523, 4);  view_523 = None
        permute_584: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_350, [0, 1, 4, 3, 2]);  unsqueeze_350 = None
        unsqueeze_351: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_590, 3);  convert_element_type_590 = None
        unsqueeze_352: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_351, 4);  unsqueeze_351 = None
        permute_585: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_352, [3, 4, 0, 2, 1]);  unsqueeze_352 = None
        permute_586: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_584, [0, 1, 3, 4, 2]);  permute_584 = None
        clone_27: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_586, memory_format = torch.contiguous_format);  permute_586 = None
        view_524: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_27, [1, 8192, 1024]);  clone_27 = None
        permute_587: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_585, [3, 4, 2, 0, 1]);  permute_585 = None
        clone_28: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_587, memory_format = torch.contiguous_format);  permute_587 = None
        view_525: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_28, [1, 1024, 1024]);  clone_28 = None
        squeeze_dim_532: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_524, 0)
        squeeze_dim_533: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_525, 0)
        mm_default_266: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_532, squeeze_dim_533);  squeeze_dim_532 = squeeze_dim_533 = None
        unsqueeze_default_266: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_266, 0);  mm_default_266 = None
        view_526: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_266, [512, 16, 1, 1, 1024]);  unsqueeze_default_266 = None
        permute_588: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_526, [0, 1, 4, 2, 3]);  view_526 = None
        view_527: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_588, [512, 16, 1024]);  permute_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_55: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 55)
        inductor_random_default_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_55, 'rand');  inductor_lookup_seed_default_55 = None
        convert_element_type_default_104: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_43, torch.bfloat16);  inductor_random_default_43 = None
        gt_55: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_104, 0.1);  convert_element_type_default_104 = None
        mul_217: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_55, view_527);  view_527 = None
        mul_218: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_217, 1.1111111111111112);  mul_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_147: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_218, add_142);  mul_218 = add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_52: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_26[0]
        getitem_53: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_26[1];  var_mean_26 = None
        add_148: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_52, 1e-12);  getitem_52 = None
        rsqrt_26: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        sub_40: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_147, getitem_53);  add_147 = getitem_53 = None
        mul_219: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_26);  sub_40 = None
        mul_220: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_219, primals_205)
        add_149: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_220, primals_206);  mul_220 = primals_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_593: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_208, torch.bfloat16);  primals_208 = None
        convert_element_type_594: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_207, torch.bfloat16);  primals_207 = None
        convert_element_type_595: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_149, torch.bfloat16)
        view_528: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_595, [8192, 1024]);  convert_element_type_595 = None
        permute_589: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_594, [1, 0]);  convert_element_type_594 = None
        addmm_26: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_593, view_528, permute_589);  convert_element_type_593 = None
        view_529: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_26, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_599: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_529, torch.float32);  view_529 = None
        mul_221: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_599, 0.5)
        mul_222: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_599, 0.7071067811865476);  convert_element_type_599 = None
        erf_13: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_222);  mul_222 = None
        add_150: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_223: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_221, add_150);  mul_221 = add_150 = None
        convert_element_type_600: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_223, torch.bfloat16);  mul_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_56: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 56)
        inductor_random_default_42: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_56, 'rand');  inductor_lookup_seed_default_56 = None
        convert_element_type_default_103: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_42, torch.bfloat16);  inductor_random_default_42 = None
        gt_56: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_103, 0.1);  convert_element_type_default_103 = None
        mul_224: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_56, convert_element_type_600);  convert_element_type_600 = None
        mul_225: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_224, 1.1111111111111112);  mul_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_601: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_210, torch.bfloat16);  primals_210 = None
        convert_element_type_602: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_209, torch.bfloat16);  primals_209 = None
        view_530: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_225, [8192, 4096]);  mul_225 = None
        permute_590: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_602, [1, 0]);  convert_element_type_602 = None
        addmm_27: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_601, view_530, permute_590);  convert_element_type_601 = None
        view_531: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_27, [512, 16, 1024]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_57: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 57)
        inductor_random_default_41: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_57, 'rand');  inductor_lookup_seed_default_57 = None
        convert_element_type_default_102: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_41, torch.bfloat16);  inductor_random_default_41 = None
        gt_57: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_102, 0.1);  convert_element_type_default_102 = None
        mul_226: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_57, view_531);  view_531 = None
        mul_227: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_226, 1.1111111111111112);  mul_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_151: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_227, add_149);  mul_227 = add_149 = None
        var_mean_27 = torch.ops.aten.var_mean.correction(add_151, [2], correction = 0, keepdim = True)
        getitem_54: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_27[0]
        getitem_55: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_27[1];  var_mean_27 = None
        add_152: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_54, 1e-12);  getitem_54 = None
        rsqrt_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_152);  add_152 = None
        sub_41: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_151, getitem_55);  add_151 = getitem_55 = None
        mul_228: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_41, rsqrt_27);  sub_41 = None
        mul_229: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_228, primals_211)
        add_153: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_229, primals_212);  mul_229 = primals_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_606: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_153, torch.bfloat16)
        convert_element_type_607: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_213, torch.bfloat16);  primals_213 = None
        unsqueeze_353: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_606, 3);  convert_element_type_606 = None
        unsqueeze_354: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_353, 4);  unsqueeze_353 = None
        unsqueeze_355: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_607, 3);  convert_element_type_607 = None
        unsqueeze_356: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_355, 4);  unsqueeze_355 = None
        view_532: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_354, [1, 8192, 1024]);  unsqueeze_354 = None
        view_533: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_356, [1, 1024, 1024]);  unsqueeze_356 = None
        squeeze_dim_530: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_532, 0)
        squeeze_dim_531: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_533, 0)
        mm_default_265: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_530, squeeze_dim_531);  squeeze_dim_531 = None
        unsqueeze_default_265: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_265, 0);  mm_default_265 = None
        view_534: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_265, [512, 16, 1, 16, 64]);  unsqueeze_default_265 = None
        permute_595: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_534, [0, 1, 3, 4, 2]);  view_534 = None
        view_535: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_595, [512, 16, 16, 64]);  permute_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_611: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_214, torch.bfloat16);  primals_214 = None
        unsqueeze_359: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_611, 3);  convert_element_type_611 = None
        unsqueeze_360: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_359, 4);  unsqueeze_359 = None
        view_537: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_360, [1, 1024, 1024]);  unsqueeze_360 = None
        squeeze_dim_529: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_537, 0)
        mm_default_264: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_530, squeeze_dim_529);  squeeze_dim_529 = None
        unsqueeze_default_264: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_264, 0);  mm_default_264 = None
        view_538: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_264, [512, 16, 1, 16, 64]);  unsqueeze_default_264 = None
        permute_600: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_538, [0, 1, 3, 4, 2]);  view_538 = None
        view_539: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_600, [512, 16, 16, 64]);  permute_600 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_615: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_215, torch.bfloat16);  primals_215 = None
        unsqueeze_363: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_615, 3);  convert_element_type_615 = None
        unsqueeze_364: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_363, 4);  unsqueeze_363 = None
        view_541: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_364, [1, 1024, 1024]);  unsqueeze_364 = None
        squeeze_dim_527: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_541, 0)
        mm_default_263: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_530, squeeze_dim_527);  squeeze_dim_530 = squeeze_dim_527 = None
        unsqueeze_default_263: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_263, 0);  mm_default_263 = None
        view_542: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_263, [512, 16, 1, 16, 64]);  unsqueeze_default_263 = None
        permute_605: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_542, [0, 1, 3, 4, 2]);  view_542 = None
        view_543: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_605, [512, 16, 16, 64]);  permute_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_620: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_216, torch.bfloat16);  primals_216 = None
        unsqueeze_367: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_620, 3);  convert_element_type_620 = None
        unsqueeze_368: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_367, 4);  unsqueeze_367 = None
        view_545: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_368, [1, 1024, 1024]);  unsqueeze_368 = None
        squeeze_dim_525: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_545, 0);  view_545 = None
        mm_default_262: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_525);  squeeze_dim_525 = None
        unsqueeze_default_262: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_262, 0);  mm_default_262 = None
        view_546: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_262, [1024, 16, 1, 16, 64]);  unsqueeze_default_262 = None
        permute_610: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_546, [0, 1, 3, 4, 2]);  view_546 = None
        view_547: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_610, [1024, 16, 16, 64]);  permute_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_154: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_535, primals_217);  primals_217 = None
        convert_element_type_623: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        unsqueeze_369: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_623, 4);  convert_element_type_623 = None
        permute_611: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_369, [1, 2, 0, 4, 3]);  unsqueeze_369 = None
        unsqueeze_370: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_539, 4);  view_539 = None
        permute_612: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_370, [1, 2, 4, 0, 3]);  unsqueeze_370 = None
        permute_613: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_611, [0, 1, 2, 4, 3]);  permute_611 = None
        view_548: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_613, [256, 512, 64]);  permute_613 = None
        permute_614: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_612, [0, 1, 4, 3, 2]);  permute_612 = None
        view_549: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_614, [256, 64, 512]);  permute_614 = None
        bmm_116: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_548, view_549)
        view_550: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_116, [16, 16, 512, 1, 512]);  bmm_116 = None
        permute_615: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_550, [0, 1, 2, 4, 3]);  view_550 = None
        view_551: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_615, [16, 16, 512, 512]);  permute_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_155: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_535, primals_218);  view_535 = primals_218 = None
        convert_element_type_626: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_155, torch.bfloat16);  add_155 = None
        unsqueeze_371: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_626, 4);  convert_element_type_626 = None
        permute_616: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_371, [1, 2, 0, 4, 3]);  unsqueeze_371 = None
        unsqueeze_372: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_547, 4);  view_547 = None
        permute_617: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_372, [1, 2, 4, 0, 3]);  unsqueeze_372 = None
        permute_618: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_616, [0, 1, 2, 4, 3]);  permute_616 = None
        view_552: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_618, [256, 512, 64]);  permute_618 = None
        permute_619: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_617, [0, 1, 4, 3, 2]);  permute_617 = None
        view_553: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_619, [256, 64, 1024]);  permute_619 = None
        bmm_117: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_552, view_553)
        view_554: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_117, [16, 16, 512, 1, 1024]);  bmm_117 = None
        permute_620: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_554, [0, 1, 2, 4, 3]);  view_554 = None
        view_555: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_620, [16, 16, 512, 1024]);  permute_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_556: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_555, [16, 16, 1024, 512]);  view_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_15: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_556, 2, 1, 9223372036854775807);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_557: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_15, [16, 16, 512, 1023]);  slice_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_14: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_557, [None, None, None, iota_2]);  view_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_156: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_551, index_14);  view_551 = index_14 = None
        add_157: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_156, 0);  add_156 = None

        # No stacktrace found for following nodes
        mul_tensor_36: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_157, 0.125)
        convert_element_type_default_42: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_36, torch.float32);  mul_tensor_36 = None
        convert_element_type_default_43: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_157, torch.float32)
        mul_tensor_37: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_43, 1);  convert_element_type_default_43 = None
        amax_default_18: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_37, [3], True)
        sub_tensor_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_37, amax_default_18);  mul_tensor_37 = None
        mul_tensor_38: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_18, 0.125);  sub_tensor_18 = None
        amax_default_19: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_42, [3], True)
        sub_tensor_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_42, amax_default_19)
        abs_default_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_42)
        ne_scalar_9: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_9, inf);  abs_default_9 = None
        eq_tensor_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_42, convert_element_type_default_42);  convert_element_type_default_42 = None
        mul_tensor_39: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_10, ne_scalar_9);  eq_tensor_10 = ne_scalar_9 = None
        logical_not_default_18: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_39);  mul_tensor_39 = None
        any_dims_9: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_18, [3], True);  logical_not_default_18 = None
        logical_not_default_19: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_9);  any_dims_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_19, mul_tensor_38, sub_tensor_19);  mul_tensor_38 = sub_tensor_19 = None
        exp_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_10);  where_self_10 = None
        sum_15: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_14, [3], True)
        div_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_58: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 58)
        inductor_random_default_40: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_58, 'rand');  inductor_lookup_seed_default_58 = None
        gt_58: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_40, 0.1);  inductor_random_default_40 = None
        mul_231: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_58, div_15);  div_15 = None
        mul_232: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_231, 1.1111111111111112);  mul_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_630: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_232, torch.bfloat16);  mul_232 = None
        unsqueeze_373: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_630, 4);  convert_element_type_630 = None
        unsqueeze_374: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_543, 4);  view_543 = None
        permute_622: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_374, [4, 1, 2, 3, 0]);  unsqueeze_374 = None
        view_558: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_373, [256, 512, 512]);  unsqueeze_373 = None
        permute_624: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_622, [1, 2, 4, 3, 0]);  permute_622 = None
        view_559: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_624, [256, 512, 64]);  permute_624 = None
        bmm_118: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_558, view_559)
        view_560: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_118, [16, 16, 512, 1, 64]);  bmm_118 = None
        permute_625: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_560, [2, 0, 1, 4, 3]);  view_560 = None
        view_561: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_625, [512, 16, 16, 64]);  permute_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_633: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_219, torch.bfloat16);  primals_219 = None
        unsqueeze_375: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_561, 4);  view_561 = None
        permute_626: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_375, [0, 1, 4, 3, 2]);  unsqueeze_375 = None
        unsqueeze_376: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_633, 3);  convert_element_type_633 = None
        unsqueeze_377: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_376, 4);  unsqueeze_376 = None
        permute_627: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_377, [3, 4, 0, 2, 1]);  unsqueeze_377 = None
        permute_628: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_626, [0, 1, 3, 4, 2]);  permute_626 = None
        clone_29: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
        view_562: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_29, [1, 8192, 1024]);  clone_29 = None
        permute_629: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_627, [3, 4, 2, 0, 1]);  permute_627 = None
        clone_30: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_629, memory_format = torch.contiguous_format);  permute_629 = None
        view_563: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_30, [1, 1024, 1024]);  clone_30 = None
        squeeze_dim_522: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_562, 0)
        squeeze_dim_523: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_563, 0)
        mm_default_261: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_522, squeeze_dim_523);  squeeze_dim_522 = squeeze_dim_523 = None
        unsqueeze_default_261: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_261, 0);  mm_default_261 = None
        view_564: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_261, [512, 16, 1, 1, 1024]);  unsqueeze_default_261 = None
        permute_630: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_564, [0, 1, 4, 2, 3]);  view_564 = None
        view_565: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_630, [512, 16, 1024]);  permute_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_59: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 59)
        inductor_random_default_39: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_59, 'rand');  inductor_lookup_seed_default_59 = None
        convert_element_type_default_101: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_39, torch.bfloat16);  inductor_random_default_39 = None
        gt_59: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_101, 0.1);  convert_element_type_default_101 = None
        mul_233: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_59, view_565);  view_565 = None
        mul_234: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_233, 1.1111111111111112);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_158: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_234, add_153);  mul_234 = add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_56: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_28[0]
        getitem_57: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_28[1];  var_mean_28 = None
        add_159: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_56, 1e-12);  getitem_56 = None
        rsqrt_28: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        sub_43: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_158, getitem_57);  add_158 = getitem_57 = None
        mul_235: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_28);  sub_43 = None
        mul_236: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_235, primals_220)
        add_160: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_236, primals_221);  mul_236 = primals_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_636: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_223, torch.bfloat16);  primals_223 = None
        convert_element_type_637: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_222, torch.bfloat16);  primals_222 = None
        convert_element_type_638: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16)
        view_566: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_638, [8192, 1024]);  convert_element_type_638 = None
        permute_631: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_637, [1, 0]);  convert_element_type_637 = None
        addmm_28: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_636, view_566, permute_631);  convert_element_type_636 = None
        view_567: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_28, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_642: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_567, torch.float32);  view_567 = None
        mul_237: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, 0.5)
        mul_238: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_642, 0.7071067811865476);  convert_element_type_642 = None
        erf_14: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_238);  mul_238 = None
        add_161: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_239: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_237, add_161);  mul_237 = add_161 = None
        convert_element_type_643: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_239, torch.bfloat16);  mul_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_60: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 60)
        inductor_random_default_38: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_60, 'rand');  inductor_lookup_seed_default_60 = None
        convert_element_type_default_100: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_38, torch.bfloat16);  inductor_random_default_38 = None
        gt_60: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_100, 0.1);  convert_element_type_default_100 = None
        mul_240: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_60, convert_element_type_643);  convert_element_type_643 = None
        mul_241: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_240, 1.1111111111111112);  mul_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_644: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_225, torch.bfloat16);  primals_225 = None
        convert_element_type_645: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_224, torch.bfloat16);  primals_224 = None
        view_568: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_241, [8192, 4096]);  mul_241 = None
        permute_632: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_645, [1, 0]);  convert_element_type_645 = None
        addmm_29: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_644, view_568, permute_632);  convert_element_type_644 = None
        view_569: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_29, [512, 16, 1024]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_61: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 61)
        inductor_random_default_37: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_61, 'rand');  inductor_lookup_seed_default_61 = None
        convert_element_type_default_99: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_37, torch.bfloat16);  inductor_random_default_37 = None
        gt_61: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_99, 0.1);  convert_element_type_default_99 = None
        mul_242: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_61, view_569);  view_569 = None
        mul_243: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_242, 1.1111111111111112);  mul_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_162: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_243, add_160);  mul_243 = add_160 = None
        var_mean_29 = torch.ops.aten.var_mean.correction(add_162, [2], correction = 0, keepdim = True)
        getitem_58: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_29[0]
        getitem_59: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_29[1];  var_mean_29 = None
        add_163: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_58, 1e-12);  getitem_58 = None
        rsqrt_29: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_163);  add_163 = None
        sub_44: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_162, getitem_59);  add_162 = getitem_59 = None
        mul_244: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_29);  sub_44 = None
        mul_245: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_244, primals_226)
        add_164: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_245, primals_227);  mul_245 = primals_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_649: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16)
        convert_element_type_650: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_228, torch.bfloat16);  primals_228 = None
        unsqueeze_378: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_649, 3);  convert_element_type_649 = None
        unsqueeze_379: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_378, 4);  unsqueeze_378 = None
        unsqueeze_380: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_650, 3);  convert_element_type_650 = None
        unsqueeze_381: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_380, 4);  unsqueeze_380 = None
        view_570: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_379, [1, 8192, 1024]);  unsqueeze_379 = None
        view_571: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_381, [1, 1024, 1024]);  unsqueeze_381 = None
        squeeze_dim_520: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_570, 0)
        squeeze_dim_521: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_571, 0)
        mm_default_260: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_520, squeeze_dim_521);  squeeze_dim_521 = None
        unsqueeze_default_260: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_260, 0);  mm_default_260 = None
        view_572: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_260, [512, 16, 1, 16, 64]);  unsqueeze_default_260 = None
        permute_637: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_572, [0, 1, 3, 4, 2]);  view_572 = None
        view_573: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_637, [512, 16, 16, 64]);  permute_637 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_654: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_229, torch.bfloat16);  primals_229 = None
        unsqueeze_384: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_654, 3);  convert_element_type_654 = None
        unsqueeze_385: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_384, 4);  unsqueeze_384 = None
        view_575: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_385, [1, 1024, 1024]);  unsqueeze_385 = None
        squeeze_dim_519: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_575, 0)
        mm_default_259: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_520, squeeze_dim_519);  squeeze_dim_519 = None
        unsqueeze_default_259: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_259, 0);  mm_default_259 = None
        view_576: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_259, [512, 16, 1, 16, 64]);  unsqueeze_default_259 = None
        permute_642: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_576, [0, 1, 3, 4, 2]);  view_576 = None
        view_577: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_642, [512, 16, 16, 64]);  permute_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_658: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_230, torch.bfloat16);  primals_230 = None
        unsqueeze_388: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_658, 3);  convert_element_type_658 = None
        unsqueeze_389: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_388, 4);  unsqueeze_388 = None
        view_579: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_389, [1, 1024, 1024]);  unsqueeze_389 = None
        squeeze_dim_517: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_579, 0)
        mm_default_258: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_520, squeeze_dim_517);  squeeze_dim_520 = squeeze_dim_517 = None
        unsqueeze_default_258: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_258, 0);  mm_default_258 = None
        view_580: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_258, [512, 16, 1, 16, 64]);  unsqueeze_default_258 = None
        permute_647: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_580, [0, 1, 3, 4, 2]);  view_580 = None
        view_581: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_647, [512, 16, 16, 64]);  permute_647 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_663: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_231, torch.bfloat16);  primals_231 = None
        unsqueeze_392: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_663, 3);  convert_element_type_663 = None
        unsqueeze_393: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_392, 4);  unsqueeze_392 = None
        view_583: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_393, [1, 1024, 1024]);  unsqueeze_393 = None
        squeeze_dim_515: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_583, 0);  view_583 = None
        mm_default_257: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_515);  squeeze_dim_515 = None
        unsqueeze_default_257: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_257, 0);  mm_default_257 = None
        view_584: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_257, [1024, 16, 1, 16, 64]);  unsqueeze_default_257 = None
        permute_652: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_584, [0, 1, 3, 4, 2]);  view_584 = None
        view_585: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_652, [1024, 16, 16, 64]);  permute_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_165: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, primals_232);  primals_232 = None
        convert_element_type_666: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_165, torch.bfloat16);  add_165 = None
        unsqueeze_394: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_666, 4);  convert_element_type_666 = None
        permute_653: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_394, [1, 2, 0, 4, 3]);  unsqueeze_394 = None
        unsqueeze_395: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_577, 4);  view_577 = None
        permute_654: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_395, [1, 2, 4, 0, 3]);  unsqueeze_395 = None
        permute_655: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_653, [0, 1, 2, 4, 3]);  permute_653 = None
        view_586: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_655, [256, 512, 64]);  permute_655 = None
        permute_656: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_654, [0, 1, 4, 3, 2]);  permute_654 = None
        view_587: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_656, [256, 64, 512]);  permute_656 = None
        bmm_124: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_586, view_587)
        view_588: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_124, [16, 16, 512, 1, 512]);  bmm_124 = None
        permute_657: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_588, [0, 1, 2, 4, 3]);  view_588 = None
        view_589: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_657, [16, 16, 512, 512]);  permute_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_166: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_573, primals_233);  view_573 = primals_233 = None
        convert_element_type_669: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16);  add_166 = None
        unsqueeze_396: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_669, 4);  convert_element_type_669 = None
        permute_658: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_396, [1, 2, 0, 4, 3]);  unsqueeze_396 = None
        unsqueeze_397: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_585, 4);  view_585 = None
        permute_659: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_397, [1, 2, 4, 0, 3]);  unsqueeze_397 = None
        permute_660: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_658, [0, 1, 2, 4, 3]);  permute_658 = None
        view_590: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_660, [256, 512, 64]);  permute_660 = None
        permute_661: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_659, [0, 1, 4, 3, 2]);  permute_659 = None
        view_591: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_661, [256, 64, 1024]);  permute_661 = None
        bmm_125: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_590, view_591)
        view_592: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_125, [16, 16, 512, 1, 1024]);  bmm_125 = None
        permute_662: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_592, [0, 1, 2, 4, 3]);  view_592 = None
        view_593: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_662, [16, 16, 512, 1024]);  permute_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_594: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_593, [16, 16, 1024, 512]);  view_593 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_16: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_594, 2, 1, 9223372036854775807);  view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_595: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_16, [16, 16, 512, 1023]);  slice_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_15: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_595, [None, None, None, iota_2]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_167: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_589, index_15);  view_589 = index_15 = None
        add_168: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_167, 0);  add_167 = None

        # No stacktrace found for following nodes
        mul_tensor_32: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_168, 0.125)
        convert_element_type_default_40: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_32, torch.float32);  mul_tensor_32 = None
        convert_element_type_default_41: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_168, torch.float32)
        mul_tensor_33: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_41, 1);  convert_element_type_default_41 = None
        amax_default_16: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_33, [3], True)
        sub_tensor_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_33, amax_default_16);  mul_tensor_33 = None
        mul_tensor_34: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_16, 0.125);  sub_tensor_16 = None
        amax_default_17: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_40, [3], True)
        sub_tensor_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_40, amax_default_17)
        abs_default_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_40)
        ne_scalar_8: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_8, inf);  abs_default_8 = None
        eq_tensor_9: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_40, convert_element_type_default_40);  convert_element_type_default_40 = None
        mul_tensor_35: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_9, ne_scalar_8);  eq_tensor_9 = ne_scalar_8 = None
        logical_not_default_16: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_35);  mul_tensor_35 = None
        any_dims_8: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_16, [3], True);  logical_not_default_16 = None
        logical_not_default_17: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_8);  any_dims_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_17, mul_tensor_34, sub_tensor_17);  mul_tensor_34 = sub_tensor_17 = None
        exp_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_9);  where_self_9 = None
        sum_16: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_15, [3], True)
        div_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_62: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 62)
        inductor_random_default_36: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_62, 'rand');  inductor_lookup_seed_default_62 = None
        gt_62: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_36, 0.1);  inductor_random_default_36 = None
        mul_247: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_62, div_16);  div_16 = None
        mul_248: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_247, 1.1111111111111112);  mul_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_673: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_248, torch.bfloat16);  mul_248 = None
        unsqueeze_398: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_673, 4);  convert_element_type_673 = None
        unsqueeze_399: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_581, 4);  view_581 = None
        permute_664: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_399, [4, 1, 2, 3, 0]);  unsqueeze_399 = None
        view_596: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_398, [256, 512, 512]);  unsqueeze_398 = None
        permute_666: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_664, [1, 2, 4, 3, 0]);  permute_664 = None
        view_597: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_666, [256, 512, 64]);  permute_666 = None
        bmm_126: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_596, view_597)
        view_598: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_126, [16, 16, 512, 1, 64]);  bmm_126 = None
        permute_667: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_598, [2, 0, 1, 4, 3]);  view_598 = None
        view_599: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_667, [512, 16, 16, 64]);  permute_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_676: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_234, torch.bfloat16);  primals_234 = None
        unsqueeze_400: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_599, 4);  view_599 = None
        permute_668: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_400, [0, 1, 4, 3, 2]);  unsqueeze_400 = None
        unsqueeze_401: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_676, 3);  convert_element_type_676 = None
        unsqueeze_402: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_401, 4);  unsqueeze_401 = None
        permute_669: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_402, [3, 4, 0, 2, 1]);  unsqueeze_402 = None
        permute_670: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_668, [0, 1, 3, 4, 2]);  permute_668 = None
        clone_31: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_670, memory_format = torch.contiguous_format);  permute_670 = None
        view_600: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_31, [1, 8192, 1024]);  clone_31 = None
        permute_671: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_669, [3, 4, 2, 0, 1]);  permute_669 = None
        clone_32: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_671, memory_format = torch.contiguous_format);  permute_671 = None
        view_601: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_32, [1, 1024, 1024]);  clone_32 = None
        squeeze_dim_512: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_600, 0)
        squeeze_dim_513: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_601, 0)
        mm_default_256: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_512, squeeze_dim_513);  squeeze_dim_512 = squeeze_dim_513 = None
        unsqueeze_default_256: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_256, 0);  mm_default_256 = None
        view_602: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_256, [512, 16, 1, 1, 1024]);  unsqueeze_default_256 = None
        permute_672: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_602, [0, 1, 4, 2, 3]);  view_602 = None
        view_603: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_672, [512, 16, 1024]);  permute_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_63: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 63)
        inductor_random_default_35: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_63, 'rand');  inductor_lookup_seed_default_63 = None
        convert_element_type_default_98: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_35, torch.bfloat16);  inductor_random_default_35 = None
        gt_63: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_98, 0.1);  convert_element_type_default_98 = None
        mul_249: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_63, view_603);  view_603 = None
        mul_250: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_249, 1.1111111111111112);  mul_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_169: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_250, add_164);  mul_250 = add_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_60: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_30[0]
        getitem_61: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_30[1];  var_mean_30 = None
        add_170: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_60, 1e-12);  getitem_60 = None
        rsqrt_30: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        sub_46: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_169, getitem_61);  add_169 = getitem_61 = None
        mul_251: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_30);  sub_46 = None
        mul_252: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_251, primals_235)
        add_171: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_252, primals_236);  mul_252 = primals_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_679: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_238, torch.bfloat16);  primals_238 = None
        convert_element_type_680: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_237, torch.bfloat16);  primals_237 = None
        convert_element_type_681: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_171, torch.bfloat16)
        view_604: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_681, [8192, 1024]);  convert_element_type_681 = None
        permute_673: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_680, [1, 0]);  convert_element_type_680 = None
        addmm_30: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_679, view_604, permute_673);  convert_element_type_679 = None
        view_605: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_30, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_685: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_605, torch.float32);  view_605 = None
        mul_253: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, 0.5)
        mul_254: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_685, 0.7071067811865476);  convert_element_type_685 = None
        erf_15: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_254);  mul_254 = None
        add_172: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_255: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_253, add_172);  mul_253 = add_172 = None
        convert_element_type_686: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_255, torch.bfloat16);  mul_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_64: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 64)
        inductor_random_default_34: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_64, 'rand');  inductor_lookup_seed_default_64 = None
        convert_element_type_default_97: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_34, torch.bfloat16);  inductor_random_default_34 = None
        gt_64: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_97, 0.1);  convert_element_type_default_97 = None
        mul_256: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_64, convert_element_type_686);  convert_element_type_686 = None
        mul_257: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_256, 1.1111111111111112);  mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_687: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_240, torch.bfloat16);  primals_240 = None
        convert_element_type_688: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_239, torch.bfloat16);  primals_239 = None
        view_606: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_257, [8192, 4096]);  mul_257 = None
        permute_674: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_688, [1, 0]);  convert_element_type_688 = None
        addmm_31: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_687, view_606, permute_674);  convert_element_type_687 = None
        view_607: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_31, [512, 16, 1024]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_65: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 65)
        inductor_random_default_33: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_65, 'rand');  inductor_lookup_seed_default_65 = None
        convert_element_type_default_96: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_33, torch.bfloat16);  inductor_random_default_33 = None
        gt_65: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_96, 0.1);  convert_element_type_default_96 = None
        mul_258: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_65, view_607);  view_607 = None
        mul_259: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_258, 1.1111111111111112);  mul_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_173: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_259, add_171);  mul_259 = add_171 = None
        var_mean_31 = torch.ops.aten.var_mean.correction(add_173, [2], correction = 0, keepdim = True)
        getitem_62: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_31[0]
        getitem_63: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_31[1];  var_mean_31 = None
        add_174: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_62, 1e-12);  getitem_62 = None
        rsqrt_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_174);  add_174 = None
        sub_47: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_173, getitem_63);  add_173 = getitem_63 = None
        mul_260: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_31);  sub_47 = None
        mul_261: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_260, primals_241)
        add_175: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_261, primals_242);  mul_261 = primals_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_692: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_175, torch.bfloat16)
        convert_element_type_693: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_243, torch.bfloat16);  primals_243 = None
        unsqueeze_403: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_692, 3);  convert_element_type_692 = None
        unsqueeze_404: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_403, 4);  unsqueeze_403 = None
        unsqueeze_405: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_693, 3);  convert_element_type_693 = None
        unsqueeze_406: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_405, 4);  unsqueeze_405 = None
        view_608: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_404, [1, 8192, 1024]);  unsqueeze_404 = None
        view_609: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_406, [1, 1024, 1024]);  unsqueeze_406 = None
        squeeze_dim_510: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_608, 0)
        squeeze_dim_511: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_609, 0)
        mm_default_255: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_510, squeeze_dim_511);  squeeze_dim_511 = None
        unsqueeze_default_255: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_255, 0);  mm_default_255 = None
        view_610: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_255, [512, 16, 1, 16, 64]);  unsqueeze_default_255 = None
        permute_679: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_610, [0, 1, 3, 4, 2]);  view_610 = None
        view_611: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_679, [512, 16, 16, 64]);  permute_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_697: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_244, torch.bfloat16);  primals_244 = None
        unsqueeze_409: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_697, 3);  convert_element_type_697 = None
        unsqueeze_410: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_409, 4);  unsqueeze_409 = None
        view_613: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_410, [1, 1024, 1024]);  unsqueeze_410 = None
        squeeze_dim_509: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_613, 0)
        mm_default_254: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_510, squeeze_dim_509);  squeeze_dim_509 = None
        unsqueeze_default_254: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_254, 0);  mm_default_254 = None
        view_614: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_254, [512, 16, 1, 16, 64]);  unsqueeze_default_254 = None
        permute_684: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_614, [0, 1, 3, 4, 2]);  view_614 = None
        view_615: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_684, [512, 16, 16, 64]);  permute_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_701: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_245, torch.bfloat16);  primals_245 = None
        unsqueeze_413: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_701, 3);  convert_element_type_701 = None
        unsqueeze_414: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_413, 4);  unsqueeze_413 = None
        view_617: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_414, [1, 1024, 1024]);  unsqueeze_414 = None
        squeeze_dim_507: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_617, 0)
        mm_default_253: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_510, squeeze_dim_507);  squeeze_dim_510 = squeeze_dim_507 = None
        unsqueeze_default_253: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_253, 0);  mm_default_253 = None
        view_618: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_253, [512, 16, 1, 16, 64]);  unsqueeze_default_253 = None
        permute_689: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_618, [0, 1, 3, 4, 2]);  view_618 = None
        view_619: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_689, [512, 16, 16, 64]);  permute_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_706: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_246, torch.bfloat16);  primals_246 = None
        unsqueeze_417: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_706, 3);  convert_element_type_706 = None
        unsqueeze_418: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_417, 4);  unsqueeze_417 = None
        view_621: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_418, [1, 1024, 1024]);  unsqueeze_418 = None
        squeeze_dim_505: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_621, 0);  view_621 = None
        mm_default_252: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_505);  squeeze_dim_505 = None
        unsqueeze_default_252: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_252, 0);  mm_default_252 = None
        view_622: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_252, [1024, 16, 1, 16, 64]);  unsqueeze_default_252 = None
        permute_694: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_622, [0, 1, 3, 4, 2]);  view_622 = None
        view_623: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_694, [1024, 16, 16, 64]);  permute_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_176: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_611, primals_247);  primals_247 = None
        convert_element_type_709: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16);  add_176 = None
        unsqueeze_419: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_709, 4);  convert_element_type_709 = None
        permute_695: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_419, [1, 2, 0, 4, 3]);  unsqueeze_419 = None
        unsqueeze_420: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_615, 4);  view_615 = None
        permute_696: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_420, [1, 2, 4, 0, 3]);  unsqueeze_420 = None
        permute_697: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_695, [0, 1, 2, 4, 3]);  permute_695 = None
        view_624: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_697, [256, 512, 64]);  permute_697 = None
        permute_698: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_696, [0, 1, 4, 3, 2]);  permute_696 = None
        view_625: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_698, [256, 64, 512]);  permute_698 = None
        bmm_132: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_624, view_625)
        view_626: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_132, [16, 16, 512, 1, 512]);  bmm_132 = None
        permute_699: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_626, [0, 1, 2, 4, 3]);  view_626 = None
        view_627: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_699, [16, 16, 512, 512]);  permute_699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_177: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_611, primals_248);  view_611 = primals_248 = None
        convert_element_type_712: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_177, torch.bfloat16);  add_177 = None
        unsqueeze_421: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_712, 4);  convert_element_type_712 = None
        permute_700: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_421, [1, 2, 0, 4, 3]);  unsqueeze_421 = None
        unsqueeze_422: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_623, 4);  view_623 = None
        permute_701: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_422, [1, 2, 4, 0, 3]);  unsqueeze_422 = None
        permute_702: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_700, [0, 1, 2, 4, 3]);  permute_700 = None
        view_628: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_702, [256, 512, 64]);  permute_702 = None
        permute_703: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_701, [0, 1, 4, 3, 2]);  permute_701 = None
        view_629: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_703, [256, 64, 1024]);  permute_703 = None
        bmm_133: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_628, view_629)
        view_630: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_133, [16, 16, 512, 1, 1024]);  bmm_133 = None
        permute_704: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_630, [0, 1, 2, 4, 3]);  view_630 = None
        view_631: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_704, [16, 16, 512, 1024]);  permute_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_632: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_631, [16, 16, 1024, 512]);  view_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_17: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_632, 2, 1, 9223372036854775807);  view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_633: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_17, [16, 16, 512, 1023]);  slice_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_16: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_633, [None, None, None, iota_2]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_178: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_627, index_16);  view_627 = index_16 = None
        add_179: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_178, 0);  add_178 = None

        # No stacktrace found for following nodes
        mul_tensor_28: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_179, 0.125)
        convert_element_type_default_38: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_28, torch.float32);  mul_tensor_28 = None
        convert_element_type_default_39: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_179, torch.float32)
        mul_tensor_29: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_39, 1);  convert_element_type_default_39 = None
        amax_default_14: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_29, [3], True)
        sub_tensor_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_29, amax_default_14);  mul_tensor_29 = None
        mul_tensor_30: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_14, 0.125);  sub_tensor_14 = None
        amax_default_15: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_38, [3], True)
        sub_tensor_15: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_38, amax_default_15)
        abs_default_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_38)
        ne_scalar_7: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_7, inf);  abs_default_7 = None
        eq_tensor_8: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_38, convert_element_type_default_38);  convert_element_type_default_38 = None
        mul_tensor_31: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_8, ne_scalar_7);  eq_tensor_8 = ne_scalar_7 = None
        logical_not_default_14: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_31);  mul_tensor_31 = None
        any_dims_7: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_14, [3], True);  logical_not_default_14 = None
        logical_not_default_15: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_7);  any_dims_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_15, mul_tensor_30, sub_tensor_15);  mul_tensor_30 = sub_tensor_15 = None
        exp_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_8);  where_self_8 = None
        sum_17: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_16, [3], True)
        div_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_66: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 66)
        inductor_random_default_32: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_66, 'rand');  inductor_lookup_seed_default_66 = None
        gt_66: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_32, 0.1);  inductor_random_default_32 = None
        mul_263: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_66, div_17);  div_17 = None
        mul_264: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_263, 1.1111111111111112);  mul_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_716: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_264, torch.bfloat16);  mul_264 = None
        unsqueeze_423: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_716, 4);  convert_element_type_716 = None
        unsqueeze_424: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_619, 4);  view_619 = None
        permute_706: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_424, [4, 1, 2, 3, 0]);  unsqueeze_424 = None
        view_634: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_423, [256, 512, 512]);  unsqueeze_423 = None
        permute_708: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_706, [1, 2, 4, 3, 0]);  permute_706 = None
        view_635: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_708, [256, 512, 64]);  permute_708 = None
        bmm_134: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_634, view_635)
        view_636: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_134, [16, 16, 512, 1, 64]);  bmm_134 = None
        permute_709: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_636, [2, 0, 1, 4, 3]);  view_636 = None
        view_637: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_709, [512, 16, 16, 64]);  permute_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_719: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_249, torch.bfloat16);  primals_249 = None
        unsqueeze_425: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_637, 4);  view_637 = None
        permute_710: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_425, [0, 1, 4, 3, 2]);  unsqueeze_425 = None
        unsqueeze_426: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_719, 3);  convert_element_type_719 = None
        unsqueeze_427: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_426, 4);  unsqueeze_426 = None
        permute_711: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_427, [3, 4, 0, 2, 1]);  unsqueeze_427 = None
        permute_712: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_710, [0, 1, 3, 4, 2]);  permute_710 = None
        clone_33: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_712, memory_format = torch.contiguous_format);  permute_712 = None
        view_638: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_33, [1, 8192, 1024]);  clone_33 = None
        permute_713: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_711, [3, 4, 2, 0, 1]);  permute_711 = None
        clone_34: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_713, memory_format = torch.contiguous_format);  permute_713 = None
        view_639: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_34, [1, 1024, 1024]);  clone_34 = None
        squeeze_dim_502: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_638, 0)
        squeeze_dim_503: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_639, 0)
        mm_default_251: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_502, squeeze_dim_503);  squeeze_dim_502 = squeeze_dim_503 = None
        unsqueeze_default_251: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_251, 0);  mm_default_251 = None
        view_640: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_251, [512, 16, 1, 1, 1024]);  unsqueeze_default_251 = None
        permute_714: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_640, [0, 1, 4, 2, 3]);  view_640 = None
        view_641: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_714, [512, 16, 1024]);  permute_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_67: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 67)
        inductor_random_default_31: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_67, 'rand');  inductor_lookup_seed_default_67 = None
        convert_element_type_default_95: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_31, torch.bfloat16);  inductor_random_default_31 = None
        gt_67: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_95, 0.1);  convert_element_type_default_95 = None
        mul_265: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_67, view_641);  view_641 = None
        mul_266: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_265, 1.1111111111111112);  mul_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_180: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_266, add_175);  mul_266 = add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_64: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_32[0]
        getitem_65: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_32[1];  var_mean_32 = None
        add_181: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_64, 1e-12);  getitem_64 = None
        rsqrt_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        sub_49: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_180, getitem_65);  add_180 = getitem_65 = None
        mul_267: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_49, rsqrt_32);  sub_49 = None
        mul_268: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_267, primals_250)
        add_182: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_268, primals_251);  mul_268 = primals_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_722: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_253, torch.bfloat16);  primals_253 = None
        convert_element_type_723: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_252, torch.bfloat16);  primals_252 = None
        convert_element_type_724: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16)
        view_642: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_724, [8192, 1024]);  convert_element_type_724 = None
        permute_715: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_723, [1, 0]);  convert_element_type_723 = None
        addmm_32: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_722, view_642, permute_715);  convert_element_type_722 = None
        view_643: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_32, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_728: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_643, torch.float32);  view_643 = None
        mul_269: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_728, 0.5)
        mul_270: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_728, 0.7071067811865476);  convert_element_type_728 = None
        erf_16: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_270);  mul_270 = None
        add_183: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_271: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_269, add_183);  mul_269 = add_183 = None
        convert_element_type_729: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_271, torch.bfloat16);  mul_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_68: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 68)
        inductor_random_default_30: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_68, 'rand');  inductor_lookup_seed_default_68 = None
        convert_element_type_default_94: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_30, torch.bfloat16);  inductor_random_default_30 = None
        gt_68: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_94, 0.1);  convert_element_type_default_94 = None
        mul_272: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_68, convert_element_type_729);  convert_element_type_729 = None
        mul_273: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_272, 1.1111111111111112);  mul_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_730: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_255, torch.bfloat16);  primals_255 = None
        convert_element_type_731: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_254, torch.bfloat16);  primals_254 = None
        view_644: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_273, [8192, 4096]);  mul_273 = None
        permute_716: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_731, [1, 0]);  convert_element_type_731 = None
        addmm_33: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_730, view_644, permute_716);  convert_element_type_730 = None
        view_645: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_33, [512, 16, 1024]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_69: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 69)
        inductor_random_default_29: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_69, 'rand');  inductor_lookup_seed_default_69 = None
        convert_element_type_default_93: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_29, torch.bfloat16);  inductor_random_default_29 = None
        gt_69: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_93, 0.1);  convert_element_type_default_93 = None
        mul_274: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_69, view_645);  view_645 = None
        mul_275: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_274, 1.1111111111111112);  mul_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_184: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_275, add_182);  mul_275 = add_182 = None
        var_mean_33 = torch.ops.aten.var_mean.correction(add_184, [2], correction = 0, keepdim = True)
        getitem_66: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_33[0]
        getitem_67: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_33[1];  var_mean_33 = None
        add_185: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_66, 1e-12);  getitem_66 = None
        rsqrt_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_185);  add_185 = None
        sub_50: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_184, getitem_67);  add_184 = getitem_67 = None
        mul_276: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_33);  sub_50 = None
        mul_277: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_276, primals_256)
        add_186: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_277, primals_257);  mul_277 = primals_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_735: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16)
        convert_element_type_736: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_258, torch.bfloat16);  primals_258 = None
        unsqueeze_428: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_735, 3);  convert_element_type_735 = None
        unsqueeze_429: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_428, 4);  unsqueeze_428 = None
        unsqueeze_430: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_736, 3);  convert_element_type_736 = None
        unsqueeze_431: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_430, 4);  unsqueeze_430 = None
        view_646: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_429, [1, 8192, 1024]);  unsqueeze_429 = None
        view_647: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_431, [1, 1024, 1024]);  unsqueeze_431 = None
        squeeze_dim_500: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_646, 0)
        squeeze_dim_501: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_647, 0)
        mm_default_250: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_500, squeeze_dim_501);  squeeze_dim_501 = None
        unsqueeze_default_250: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_250, 0);  mm_default_250 = None
        view_648: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_250, [512, 16, 1, 16, 64]);  unsqueeze_default_250 = None
        permute_721: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_648, [0, 1, 3, 4, 2]);  view_648 = None
        view_649: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_721, [512, 16, 16, 64]);  permute_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_740: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_259, torch.bfloat16);  primals_259 = None
        unsqueeze_434: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_740, 3);  convert_element_type_740 = None
        unsqueeze_435: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_434, 4);  unsqueeze_434 = None
        view_651: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_435, [1, 1024, 1024]);  unsqueeze_435 = None
        squeeze_dim_499: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_651, 0)
        mm_default_249: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_500, squeeze_dim_499);  squeeze_dim_499 = None
        unsqueeze_default_249: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_249, 0);  mm_default_249 = None
        view_652: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_249, [512, 16, 1, 16, 64]);  unsqueeze_default_249 = None
        permute_726: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_652, [0, 1, 3, 4, 2]);  view_652 = None
        view_653: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_726, [512, 16, 16, 64]);  permute_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_744: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_260, torch.bfloat16);  primals_260 = None
        unsqueeze_438: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_744, 3);  convert_element_type_744 = None
        unsqueeze_439: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_438, 4);  unsqueeze_438 = None
        view_655: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_439, [1, 1024, 1024]);  unsqueeze_439 = None
        squeeze_dim_497: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_655, 0)
        mm_default_248: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_500, squeeze_dim_497);  squeeze_dim_500 = squeeze_dim_497 = None
        unsqueeze_default_248: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_248, 0);  mm_default_248 = None
        view_656: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_248, [512, 16, 1, 16, 64]);  unsqueeze_default_248 = None
        permute_731: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_656, [0, 1, 3, 4, 2]);  view_656 = None
        view_657: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_731, [512, 16, 16, 64]);  permute_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_749: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_261, torch.bfloat16);  primals_261 = None
        unsqueeze_442: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_749, 3);  convert_element_type_749 = None
        unsqueeze_443: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_442, 4);  unsqueeze_442 = None
        view_659: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_443, [1, 1024, 1024]);  unsqueeze_443 = None
        squeeze_dim_495: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_659, 0);  view_659 = None
        mm_default_247: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_495);  squeeze_dim_495 = None
        unsqueeze_default_247: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_247, 0);  mm_default_247 = None
        view_660: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_247, [1024, 16, 1, 16, 64]);  unsqueeze_default_247 = None
        permute_736: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_660, [0, 1, 3, 4, 2]);  view_660 = None
        view_661: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_736, [1024, 16, 16, 64]);  permute_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_187: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_649, primals_262);  primals_262 = None
        convert_element_type_752: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_187, torch.bfloat16);  add_187 = None
        unsqueeze_444: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_752, 4);  convert_element_type_752 = None
        permute_737: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_444, [1, 2, 0, 4, 3]);  unsqueeze_444 = None
        unsqueeze_445: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_653, 4);  view_653 = None
        permute_738: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_445, [1, 2, 4, 0, 3]);  unsqueeze_445 = None
        permute_739: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_737, [0, 1, 2, 4, 3]);  permute_737 = None
        view_662: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_739, [256, 512, 64]);  permute_739 = None
        permute_740: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_738, [0, 1, 4, 3, 2]);  permute_738 = None
        view_663: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_740, [256, 64, 512]);  permute_740 = None
        bmm_140: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_662, view_663)
        view_664: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_140, [16, 16, 512, 1, 512]);  bmm_140 = None
        permute_741: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_664, [0, 1, 2, 4, 3]);  view_664 = None
        view_665: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_741, [16, 16, 512, 512]);  permute_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_188: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_649, primals_263);  view_649 = primals_263 = None
        convert_element_type_755: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_188, torch.bfloat16);  add_188 = None
        unsqueeze_446: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_755, 4);  convert_element_type_755 = None
        permute_742: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_446, [1, 2, 0, 4, 3]);  unsqueeze_446 = None
        unsqueeze_447: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_661, 4);  view_661 = None
        permute_743: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_447, [1, 2, 4, 0, 3]);  unsqueeze_447 = None
        permute_744: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_742, [0, 1, 2, 4, 3]);  permute_742 = None
        view_666: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_744, [256, 512, 64]);  permute_744 = None
        permute_745: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_743, [0, 1, 4, 3, 2]);  permute_743 = None
        view_667: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_745, [256, 64, 1024]);  permute_745 = None
        bmm_141: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_666, view_667)
        view_668: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_141, [16, 16, 512, 1, 1024]);  bmm_141 = None
        permute_746: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_668, [0, 1, 2, 4, 3]);  view_668 = None
        view_669: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_746, [16, 16, 512, 1024]);  permute_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_670: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_669, [16, 16, 1024, 512]);  view_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_18: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_670, 2, 1, 9223372036854775807);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_671: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_18, [16, 16, 512, 1023]);  slice_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_17: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_671, [None, None, None, iota_2]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_189: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_665, index_17);  view_665 = index_17 = None
        add_190: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_189, 0);  add_189 = None

        # No stacktrace found for following nodes
        mul_tensor_24: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_190, 0.125)
        convert_element_type_default_36: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_24, torch.float32);  mul_tensor_24 = None
        convert_element_type_default_37: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_190, torch.float32)
        mul_tensor_25: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_37, 1);  convert_element_type_default_37 = None
        amax_default_12: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_25, [3], True)
        sub_tensor_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_25, amax_default_12);  mul_tensor_25 = None
        mul_tensor_26: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_12, 0.125);  sub_tensor_12 = None
        amax_default_13: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_36, [3], True)
        sub_tensor_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_36, amax_default_13)
        abs_default_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_36)
        ne_scalar_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_6, inf);  abs_default_6 = None
        eq_tensor_7: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_36, convert_element_type_default_36);  convert_element_type_default_36 = None
        mul_tensor_27: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_7, ne_scalar_6);  eq_tensor_7 = ne_scalar_6 = None
        logical_not_default_12: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_27);  mul_tensor_27 = None
        any_dims_6: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_12, [3], True);  logical_not_default_12 = None
        logical_not_default_13: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_6);  any_dims_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_13, mul_tensor_26, sub_tensor_13);  mul_tensor_26 = sub_tensor_13 = None
        exp_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_7);  where_self_7 = None
        sum_18: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_17, [3], True)
        div_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_70: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 70)
        inductor_random_default_28: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_70, 'rand');  inductor_lookup_seed_default_70 = None
        gt_70: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_28, 0.1);  inductor_random_default_28 = None
        mul_279: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_70, div_18);  div_18 = None
        mul_280: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_279, 1.1111111111111112);  mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_759: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_280, torch.bfloat16);  mul_280 = None
        unsqueeze_448: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_759, 4);  convert_element_type_759 = None
        unsqueeze_449: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_657, 4);  view_657 = None
        permute_748: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_449, [4, 1, 2, 3, 0]);  unsqueeze_449 = None
        view_672: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_448, [256, 512, 512]);  unsqueeze_448 = None
        permute_750: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_748, [1, 2, 4, 3, 0]);  permute_748 = None
        view_673: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_750, [256, 512, 64]);  permute_750 = None
        bmm_142: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_672, view_673)
        view_674: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_142, [16, 16, 512, 1, 64]);  bmm_142 = None
        permute_751: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_674, [2, 0, 1, 4, 3]);  view_674 = None
        view_675: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_751, [512, 16, 16, 64]);  permute_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_762: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_264, torch.bfloat16);  primals_264 = None
        unsqueeze_450: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_675, 4);  view_675 = None
        permute_752: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_450, [0, 1, 4, 3, 2]);  unsqueeze_450 = None
        unsqueeze_451: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_762, 3);  convert_element_type_762 = None
        unsqueeze_452: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_451, 4);  unsqueeze_451 = None
        permute_753: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_452, [3, 4, 0, 2, 1]);  unsqueeze_452 = None
        permute_754: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_752, [0, 1, 3, 4, 2]);  permute_752 = None
        clone_35: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_754, memory_format = torch.contiguous_format);  permute_754 = None
        view_676: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_35, [1, 8192, 1024]);  clone_35 = None
        permute_755: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_753, [3, 4, 2, 0, 1]);  permute_753 = None
        clone_36: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_755, memory_format = torch.contiguous_format);  permute_755 = None
        view_677: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_36, [1, 1024, 1024]);  clone_36 = None
        squeeze_dim_492: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_676, 0)
        squeeze_dim_493: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_677, 0)
        mm_default_246: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_492, squeeze_dim_493);  squeeze_dim_492 = squeeze_dim_493 = None
        unsqueeze_default_246: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_246, 0);  mm_default_246 = None
        view_678: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_246, [512, 16, 1, 1, 1024]);  unsqueeze_default_246 = None
        permute_756: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_678, [0, 1, 4, 2, 3]);  view_678 = None
        view_679: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_756, [512, 16, 1024]);  permute_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_71: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 71)
        inductor_random_default_27: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_71, 'rand');  inductor_lookup_seed_default_71 = None
        convert_element_type_default_92: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_27, torch.bfloat16);  inductor_random_default_27 = None
        gt_71: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_92, 0.1);  convert_element_type_default_92 = None
        mul_281: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_71, view_679);  view_679 = None
        mul_282: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_281, 1.1111111111111112);  mul_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_191: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_282, add_186);  mul_282 = add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_68: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_34[0]
        getitem_69: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_34[1];  var_mean_34 = None
        add_192: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_68, 1e-12);  getitem_68 = None
        rsqrt_34: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        sub_52: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_191, getitem_69);  add_191 = getitem_69 = None
        mul_283: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_34);  sub_52 = None
        mul_284: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_283, primals_265)
        add_193: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_284, primals_266);  mul_284 = primals_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_765: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_268, torch.bfloat16);  primals_268 = None
        convert_element_type_766: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_267, torch.bfloat16);  primals_267 = None
        convert_element_type_767: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_193, torch.bfloat16)
        view_680: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_767, [8192, 1024]);  convert_element_type_767 = None
        permute_757: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_766, [1, 0]);  convert_element_type_766 = None
        addmm_34: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_765, view_680, permute_757);  convert_element_type_765 = None
        view_681: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_34, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_771: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_681, torch.float32);  view_681 = None
        mul_285: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_771, 0.5)
        mul_286: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_771, 0.7071067811865476);  convert_element_type_771 = None
        erf_17: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_286);  mul_286 = None
        add_194: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_287: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_285, add_194);  mul_285 = add_194 = None
        convert_element_type_772: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_287, torch.bfloat16);  mul_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_72: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 72)
        inductor_random_default_26: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_72, 'rand');  inductor_lookup_seed_default_72 = None
        convert_element_type_default_91: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_26, torch.bfloat16);  inductor_random_default_26 = None
        gt_72: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_91, 0.1);  convert_element_type_default_91 = None
        mul_288: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_72, convert_element_type_772);  convert_element_type_772 = None
        mul_289: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_288, 1.1111111111111112);  mul_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_773: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_270, torch.bfloat16);  primals_270 = None
        convert_element_type_774: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_269, torch.bfloat16);  primals_269 = None
        view_682: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_289, [8192, 4096]);  mul_289 = None
        permute_758: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_774, [1, 0]);  convert_element_type_774 = None
        addmm_35: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_773, view_682, permute_758);  convert_element_type_773 = None
        view_683: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_35, [512, 16, 1024]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_73: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 73)
        inductor_random_default_25: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_73, 'rand');  inductor_lookup_seed_default_73 = None
        convert_element_type_default_90: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_25, torch.bfloat16);  inductor_random_default_25 = None
        gt_73: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_90, 0.1);  convert_element_type_default_90 = None
        mul_290: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_73, view_683);  view_683 = None
        mul_291: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_290, 1.1111111111111112);  mul_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_195: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_291, add_193);  mul_291 = add_193 = None
        var_mean_35 = torch.ops.aten.var_mean.correction(add_195, [2], correction = 0, keepdim = True)
        getitem_70: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_35[0]
        getitem_71: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_35[1];  var_mean_35 = None
        add_196: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_70, 1e-12);  getitem_70 = None
        rsqrt_35: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_196);  add_196 = None
        sub_53: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_195, getitem_71);  add_195 = getitem_71 = None
        mul_292: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_53, rsqrt_35);  sub_53 = None
        mul_293: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_292, primals_271)
        add_197: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_293, primals_272);  mul_293 = primals_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_778: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_197, torch.bfloat16)
        convert_element_type_779: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_273, torch.bfloat16);  primals_273 = None
        unsqueeze_453: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_778, 3);  convert_element_type_778 = None
        unsqueeze_454: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_453, 4);  unsqueeze_453 = None
        unsqueeze_455: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_779, 3);  convert_element_type_779 = None
        unsqueeze_456: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_455, 4);  unsqueeze_455 = None
        view_684: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_454, [1, 8192, 1024]);  unsqueeze_454 = None
        view_685: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_456, [1, 1024, 1024]);  unsqueeze_456 = None
        squeeze_dim_490: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_684, 0)
        squeeze_dim_491: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_685, 0)
        mm_default_245: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_490, squeeze_dim_491);  squeeze_dim_491 = None
        unsqueeze_default_245: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_245, 0);  mm_default_245 = None
        view_686: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_245, [512, 16, 1, 16, 64]);  unsqueeze_default_245 = None
        permute_763: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_686, [0, 1, 3, 4, 2]);  view_686 = None
        view_687: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_763, [512, 16, 16, 64]);  permute_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_783: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_274, torch.bfloat16);  primals_274 = None
        unsqueeze_459: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_783, 3);  convert_element_type_783 = None
        unsqueeze_460: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_459, 4);  unsqueeze_459 = None
        view_689: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_460, [1, 1024, 1024]);  unsqueeze_460 = None
        squeeze_dim_489: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_689, 0)
        mm_default_244: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_490, squeeze_dim_489);  squeeze_dim_489 = None
        unsqueeze_default_244: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_244, 0);  mm_default_244 = None
        view_690: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_244, [512, 16, 1, 16, 64]);  unsqueeze_default_244 = None
        permute_768: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_690, [0, 1, 3, 4, 2]);  view_690 = None
        view_691: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_768, [512, 16, 16, 64]);  permute_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_787: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_275, torch.bfloat16);  primals_275 = None
        unsqueeze_463: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_787, 3);  convert_element_type_787 = None
        unsqueeze_464: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_463, 4);  unsqueeze_463 = None
        view_693: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_464, [1, 1024, 1024]);  unsqueeze_464 = None
        squeeze_dim_487: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_693, 0)
        mm_default_243: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_490, squeeze_dim_487);  squeeze_dim_490 = squeeze_dim_487 = None
        unsqueeze_default_243: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_243, 0);  mm_default_243 = None
        view_694: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_243, [512, 16, 1, 16, 64]);  unsqueeze_default_243 = None
        permute_773: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_694, [0, 1, 3, 4, 2]);  view_694 = None
        view_695: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_773, [512, 16, 16, 64]);  permute_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_792: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_276, torch.bfloat16);  primals_276 = None
        unsqueeze_467: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_792, 3);  convert_element_type_792 = None
        unsqueeze_468: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_467, 4);  unsqueeze_467 = None
        view_697: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_468, [1, 1024, 1024]);  unsqueeze_468 = None
        squeeze_dim_485: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_697, 0);  view_697 = None
        mm_default_242: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_485);  squeeze_dim_485 = None
        unsqueeze_default_242: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_242, 0);  mm_default_242 = None
        view_698: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_242, [1024, 16, 1, 16, 64]);  unsqueeze_default_242 = None
        permute_778: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_698, [0, 1, 3, 4, 2]);  view_698 = None
        view_699: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_778, [1024, 16, 16, 64]);  permute_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_198: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_687, primals_277);  primals_277 = None
        convert_element_type_795: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16);  add_198 = None
        unsqueeze_469: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_795, 4);  convert_element_type_795 = None
        permute_779: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_469, [1, 2, 0, 4, 3]);  unsqueeze_469 = None
        unsqueeze_470: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_691, 4);  view_691 = None
        permute_780: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_470, [1, 2, 4, 0, 3]);  unsqueeze_470 = None
        permute_781: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_779, [0, 1, 2, 4, 3]);  permute_779 = None
        view_700: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_781, [256, 512, 64]);  permute_781 = None
        permute_782: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_780, [0, 1, 4, 3, 2]);  permute_780 = None
        view_701: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_782, [256, 64, 512]);  permute_782 = None
        bmm_148: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_700, view_701)
        view_702: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_148, [16, 16, 512, 1, 512]);  bmm_148 = None
        permute_783: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_702, [0, 1, 2, 4, 3]);  view_702 = None
        view_703: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_783, [16, 16, 512, 512]);  permute_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_199: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_687, primals_278);  view_687 = primals_278 = None
        convert_element_type_798: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_199, torch.bfloat16);  add_199 = None
        unsqueeze_471: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_798, 4);  convert_element_type_798 = None
        permute_784: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_471, [1, 2, 0, 4, 3]);  unsqueeze_471 = None
        unsqueeze_472: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_699, 4);  view_699 = None
        permute_785: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_472, [1, 2, 4, 0, 3]);  unsqueeze_472 = None
        permute_786: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_784, [0, 1, 2, 4, 3]);  permute_784 = None
        view_704: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_786, [256, 512, 64]);  permute_786 = None
        permute_787: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_785, [0, 1, 4, 3, 2]);  permute_785 = None
        view_705: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_787, [256, 64, 1024]);  permute_787 = None
        bmm_149: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_704, view_705)
        view_706: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_149, [16, 16, 512, 1, 1024]);  bmm_149 = None
        permute_788: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_706, [0, 1, 2, 4, 3]);  view_706 = None
        view_707: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_788, [16, 16, 512, 1024]);  permute_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_708: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_707, [16, 16, 1024, 512]);  view_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_19: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_708, 2, 1, 9223372036854775807);  view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_709: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_19, [16, 16, 512, 1023]);  slice_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_18: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_709, [None, None, None, iota_2]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_200: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_703, index_18);  view_703 = index_18 = None
        add_201: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_200, 0);  add_200 = None

        # No stacktrace found for following nodes
        mul_tensor_20: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_201, 0.125)
        convert_element_type_default_34: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_20, torch.float32);  mul_tensor_20 = None
        convert_element_type_default_35: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_201, torch.float32)
        mul_tensor_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_35, 1);  convert_element_type_default_35 = None
        amax_default_10: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_21, [3], True)
        sub_tensor_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_21, amax_default_10);  mul_tensor_21 = None
        mul_tensor_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_10, 0.125);  sub_tensor_10 = None
        amax_default_11: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_34, [3], True)
        sub_tensor_11: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_34, amax_default_11)
        abs_default_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_34)
        ne_scalar_5: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_5, inf);  abs_default_5 = None
        eq_tensor_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_34, convert_element_type_default_34);  convert_element_type_default_34 = None
        mul_tensor_23: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_6, ne_scalar_5);  eq_tensor_6 = ne_scalar_5 = None
        logical_not_default_10: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_23);  mul_tensor_23 = None
        any_dims_5: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_10, [3], True);  logical_not_default_10 = None
        logical_not_default_11: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_5);  any_dims_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_11, mul_tensor_22, sub_tensor_11);  mul_tensor_22 = sub_tensor_11 = None
        exp_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_6);  where_self_6 = None
        sum_19: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_18, [3], True)
        div_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_74: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 74)
        inductor_random_default_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_74, 'rand');  inductor_lookup_seed_default_74 = None
        gt_74: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_24, 0.1);  inductor_random_default_24 = None
        mul_295: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_74, div_19);  div_19 = None
        mul_296: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_295, 1.1111111111111112);  mul_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_802: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_296, torch.bfloat16);  mul_296 = None
        unsqueeze_473: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_802, 4);  convert_element_type_802 = None
        unsqueeze_474: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_695, 4);  view_695 = None
        permute_790: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_474, [4, 1, 2, 3, 0]);  unsqueeze_474 = None
        view_710: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_473, [256, 512, 512]);  unsqueeze_473 = None
        permute_792: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_790, [1, 2, 4, 3, 0]);  permute_790 = None
        view_711: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_792, [256, 512, 64]);  permute_792 = None
        bmm_150: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_710, view_711)
        view_712: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_150, [16, 16, 512, 1, 64]);  bmm_150 = None
        permute_793: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_712, [2, 0, 1, 4, 3]);  view_712 = None
        view_713: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_793, [512, 16, 16, 64]);  permute_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_805: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_279, torch.bfloat16);  primals_279 = None
        unsqueeze_475: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_713, 4);  view_713 = None
        permute_794: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_475, [0, 1, 4, 3, 2]);  unsqueeze_475 = None
        unsqueeze_476: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_805, 3);  convert_element_type_805 = None
        unsqueeze_477: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_476, 4);  unsqueeze_476 = None
        permute_795: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_477, [3, 4, 0, 2, 1]);  unsqueeze_477 = None
        permute_796: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_794, [0, 1, 3, 4, 2]);  permute_794 = None
        clone_37: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_796, memory_format = torch.contiguous_format);  permute_796 = None
        view_714: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_37, [1, 8192, 1024]);  clone_37 = None
        permute_797: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_795, [3, 4, 2, 0, 1]);  permute_795 = None
        clone_38: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_797, memory_format = torch.contiguous_format);  permute_797 = None
        view_715: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_38, [1, 1024, 1024]);  clone_38 = None
        squeeze_dim_482: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_714, 0)
        squeeze_dim_483: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_715, 0)
        mm_default_241: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_482, squeeze_dim_483);  squeeze_dim_482 = squeeze_dim_483 = None
        unsqueeze_default_241: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_241, 0);  mm_default_241 = None
        view_716: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_241, [512, 16, 1, 1, 1024]);  unsqueeze_default_241 = None
        permute_798: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_716, [0, 1, 4, 2, 3]);  view_716 = None
        view_717: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_798, [512, 16, 1024]);  permute_798 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_75: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 75)
        inductor_random_default_23: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_75, 'rand');  inductor_lookup_seed_default_75 = None
        convert_element_type_default_89: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_23, torch.bfloat16);  inductor_random_default_23 = None
        gt_75: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_89, 0.1);  convert_element_type_default_89 = None
        mul_297: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_75, view_717);  view_717 = None
        mul_298: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_297, 1.1111111111111112);  mul_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_202: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_298, add_197);  mul_298 = add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_72: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_36[0]
        getitem_73: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_36[1];  var_mean_36 = None
        add_203: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_72, 1e-12);  getitem_72 = None
        rsqrt_36: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        sub_55: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_202, getitem_73);  add_202 = getitem_73 = None
        mul_299: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_36);  sub_55 = None
        mul_300: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_299, primals_280)
        add_204: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_300, primals_281);  mul_300 = primals_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_808: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_283, torch.bfloat16);  primals_283 = None
        convert_element_type_809: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_282, torch.bfloat16);  primals_282 = None
        convert_element_type_810: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16)
        view_718: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_810, [8192, 1024]);  convert_element_type_810 = None
        permute_799: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_809, [1, 0]);  convert_element_type_809 = None
        addmm_36: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_808, view_718, permute_799);  convert_element_type_808 = None
        view_719: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_36, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_814: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_719, torch.float32);  view_719 = None
        mul_301: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_814, 0.5)
        mul_302: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_814, 0.7071067811865476);  convert_element_type_814 = None
        erf_18: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_302);  mul_302 = None
        add_205: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_303: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_301, add_205);  mul_301 = add_205 = None
        convert_element_type_815: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_303, torch.bfloat16);  mul_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_76: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 76)
        inductor_random_default_22: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_76, 'rand');  inductor_lookup_seed_default_76 = None
        convert_element_type_default_88: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_22, torch.bfloat16);  inductor_random_default_22 = None
        gt_76: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_88, 0.1);  convert_element_type_default_88 = None
        mul_304: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_76, convert_element_type_815);  convert_element_type_815 = None
        mul_305: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_304, 1.1111111111111112);  mul_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_816: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_285, torch.bfloat16);  primals_285 = None
        convert_element_type_817: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_284, torch.bfloat16);  primals_284 = None
        view_720: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_305, [8192, 4096]);  mul_305 = None
        permute_800: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_817, [1, 0]);  convert_element_type_817 = None
        addmm_37: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_816, view_720, permute_800);  convert_element_type_816 = None
        view_721: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_37, [512, 16, 1024]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_77: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 77)
        inductor_random_default_21: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_77, 'rand');  inductor_lookup_seed_default_77 = None
        convert_element_type_default_87: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_21, torch.bfloat16);  inductor_random_default_21 = None
        gt_77: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_87, 0.1);  convert_element_type_default_87 = None
        mul_306: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_77, view_721);  view_721 = None
        mul_307: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_306, 1.1111111111111112);  mul_306 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_206: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_307, add_204);  mul_307 = add_204 = None
        var_mean_37 = torch.ops.aten.var_mean.correction(add_206, [2], correction = 0, keepdim = True)
        getitem_74: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_37[0]
        getitem_75: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_37[1];  var_mean_37 = None
        add_207: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_74, 1e-12);  getitem_74 = None
        rsqrt_37: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_207);  add_207 = None
        sub_56: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_206, getitem_75);  add_206 = getitem_75 = None
        mul_308: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_37);  sub_56 = None
        mul_309: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_308, primals_286)
        add_208: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_309, primals_287);  mul_309 = primals_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_821: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16)
        convert_element_type_822: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_288, torch.bfloat16);  primals_288 = None
        unsqueeze_478: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_821, 3);  convert_element_type_821 = None
        unsqueeze_479: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_478, 4);  unsqueeze_478 = None
        unsqueeze_480: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_822, 3);  convert_element_type_822 = None
        unsqueeze_481: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_480, 4);  unsqueeze_480 = None
        view_722: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_479, [1, 8192, 1024]);  unsqueeze_479 = None
        view_723: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_481, [1, 1024, 1024]);  unsqueeze_481 = None
        squeeze_dim_480: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_722, 0)
        squeeze_dim_481: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_723, 0)
        mm_default_240: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_480, squeeze_dim_481);  squeeze_dim_481 = None
        unsqueeze_default_240: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_240, 0);  mm_default_240 = None
        view_724: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_240, [512, 16, 1, 16, 64]);  unsqueeze_default_240 = None
        permute_805: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_724, [0, 1, 3, 4, 2]);  view_724 = None
        view_725: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_805, [512, 16, 16, 64]);  permute_805 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_826: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_289, torch.bfloat16);  primals_289 = None
        unsqueeze_484: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_826, 3);  convert_element_type_826 = None
        unsqueeze_485: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_484, 4);  unsqueeze_484 = None
        view_727: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_485, [1, 1024, 1024]);  unsqueeze_485 = None
        squeeze_dim_479: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_727, 0)
        mm_default_239: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_480, squeeze_dim_479);  squeeze_dim_479 = None
        unsqueeze_default_239: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_239, 0);  mm_default_239 = None
        view_728: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_239, [512, 16, 1, 16, 64]);  unsqueeze_default_239 = None
        permute_810: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_728, [0, 1, 3, 4, 2]);  view_728 = None
        view_729: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_810, [512, 16, 16, 64]);  permute_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_830: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_290, torch.bfloat16);  primals_290 = None
        unsqueeze_488: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_830, 3);  convert_element_type_830 = None
        unsqueeze_489: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_488, 4);  unsqueeze_488 = None
        view_731: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_489, [1, 1024, 1024]);  unsqueeze_489 = None
        squeeze_dim_477: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_731, 0)
        mm_default_238: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_480, squeeze_dim_477);  squeeze_dim_480 = squeeze_dim_477 = None
        unsqueeze_default_238: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_238, 0);  mm_default_238 = None
        view_732: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_238, [512, 16, 1, 16, 64]);  unsqueeze_default_238 = None
        permute_815: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_732, [0, 1, 3, 4, 2]);  view_732 = None
        view_733: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_815, [512, 16, 16, 64]);  permute_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_835: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_291, torch.bfloat16);  primals_291 = None
        unsqueeze_492: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_835, 3);  convert_element_type_835 = None
        unsqueeze_493: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_492, 4);  unsqueeze_492 = None
        view_735: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_493, [1, 1024, 1024]);  unsqueeze_493 = None
        squeeze_dim_475: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_735, 0);  view_735 = None
        mm_default_237: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_475);  squeeze_dim_475 = None
        unsqueeze_default_237: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_237, 0);  mm_default_237 = None
        view_736: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_237, [1024, 16, 1, 16, 64]);  unsqueeze_default_237 = None
        permute_820: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_736, [0, 1, 3, 4, 2]);  view_736 = None
        view_737: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_820, [1024, 16, 16, 64]);  permute_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_209: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, primals_292);  primals_292 = None
        convert_element_type_838: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_209, torch.bfloat16);  add_209 = None
        unsqueeze_494: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_838, 4);  convert_element_type_838 = None
        permute_821: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_494, [1, 2, 0, 4, 3]);  unsqueeze_494 = None
        unsqueeze_495: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_729, 4);  view_729 = None
        permute_822: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_495, [1, 2, 4, 0, 3]);  unsqueeze_495 = None
        permute_823: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_821, [0, 1, 2, 4, 3]);  permute_821 = None
        view_738: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_823, [256, 512, 64]);  permute_823 = None
        permute_824: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_822, [0, 1, 4, 3, 2]);  permute_822 = None
        view_739: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_824, [256, 64, 512]);  permute_824 = None
        bmm_156: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_738, view_739)
        view_740: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_156, [16, 16, 512, 1, 512]);  bmm_156 = None
        permute_825: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_740, [0, 1, 2, 4, 3]);  view_740 = None
        view_741: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_825, [16, 16, 512, 512]);  permute_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_210: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_725, primals_293);  view_725 = primals_293 = None
        convert_element_type_841: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None
        unsqueeze_496: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_841, 4);  convert_element_type_841 = None
        permute_826: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_496, [1, 2, 0, 4, 3]);  unsqueeze_496 = None
        unsqueeze_497: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_737, 4);  view_737 = None
        permute_827: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_497, [1, 2, 4, 0, 3]);  unsqueeze_497 = None
        permute_828: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_826, [0, 1, 2, 4, 3]);  permute_826 = None
        view_742: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_828, [256, 512, 64]);  permute_828 = None
        permute_829: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_827, [0, 1, 4, 3, 2]);  permute_827 = None
        view_743: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_829, [256, 64, 1024]);  permute_829 = None
        bmm_157: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_742, view_743)
        view_744: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_157, [16, 16, 512, 1, 1024]);  bmm_157 = None
        permute_830: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_744, [0, 1, 2, 4, 3]);  view_744 = None
        view_745: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_830, [16, 16, 512, 1024]);  permute_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_746: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_745, [16, 16, 1024, 512]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_20: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_746, 2, 1, 9223372036854775807);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_747: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_20, [16, 16, 512, 1023]);  slice_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_19: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_747, [None, None, None, iota_2]);  view_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_211: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_741, index_19);  view_741 = index_19 = None
        add_212: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_211, 0);  add_211 = None

        # No stacktrace found for following nodes
        mul_tensor_16: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_212, 0.125)
        convert_element_type_default_32: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_16, torch.float32);  mul_tensor_16 = None
        convert_element_type_default_33: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_212, torch.float32)
        mul_tensor_17: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_33, 1);  convert_element_type_default_33 = None
        amax_default_8: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_17, [3], True)
        sub_tensor_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_17, amax_default_8);  mul_tensor_17 = None
        mul_tensor_18: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_8, 0.125);  sub_tensor_8 = None
        amax_default_9: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_32, [3], True)
        sub_tensor_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_32, amax_default_9)
        abs_default_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_32)
        ne_scalar_4: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_4, inf);  abs_default_4 = None
        eq_tensor_5: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_32, convert_element_type_default_32);  convert_element_type_default_32 = None
        mul_tensor_19: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_5, ne_scalar_4);  eq_tensor_5 = ne_scalar_4 = None
        logical_not_default_8: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_19);  mul_tensor_19 = None
        any_dims_4: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_8, [3], True);  logical_not_default_8 = None
        logical_not_default_9: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_4);  any_dims_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_9, mul_tensor_18, sub_tensor_9);  mul_tensor_18 = sub_tensor_9 = None
        exp_19: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_5);  where_self_5 = None
        sum_20: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_19, [3], True)
        div_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_78: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 78)
        inductor_random_default_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_78, 'rand');  inductor_lookup_seed_default_78 = None
        gt_78: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_20, 0.1);  inductor_random_default_20 = None
        mul_311: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_78, div_20);  div_20 = None
        mul_312: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_311, 1.1111111111111112);  mul_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_845: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_312, torch.bfloat16);  mul_312 = None
        unsqueeze_498: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_845, 4);  convert_element_type_845 = None
        unsqueeze_499: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_733, 4);  view_733 = None
        permute_832: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_499, [4, 1, 2, 3, 0]);  unsqueeze_499 = None
        view_748: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_498, [256, 512, 512]);  unsqueeze_498 = None
        permute_834: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_832, [1, 2, 4, 3, 0]);  permute_832 = None
        view_749: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_834, [256, 512, 64]);  permute_834 = None
        bmm_158: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_748, view_749)
        view_750: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_158, [16, 16, 512, 1, 64]);  bmm_158 = None
        permute_835: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_750, [2, 0, 1, 4, 3]);  view_750 = None
        view_751: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_835, [512, 16, 16, 64]);  permute_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_848: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_294, torch.bfloat16);  primals_294 = None
        unsqueeze_500: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_751, 4);  view_751 = None
        permute_836: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_500, [0, 1, 4, 3, 2]);  unsqueeze_500 = None
        unsqueeze_501: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_848, 3);  convert_element_type_848 = None
        unsqueeze_502: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_501, 4);  unsqueeze_501 = None
        permute_837: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_502, [3, 4, 0, 2, 1]);  unsqueeze_502 = None
        permute_838: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_836, [0, 1, 3, 4, 2]);  permute_836 = None
        clone_39: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_838, memory_format = torch.contiguous_format);  permute_838 = None
        view_752: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_39, [1, 8192, 1024]);  clone_39 = None
        permute_839: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_837, [3, 4, 2, 0, 1]);  permute_837 = None
        clone_40: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_839, memory_format = torch.contiguous_format);  permute_839 = None
        view_753: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_40, [1, 1024, 1024]);  clone_40 = None
        squeeze_dim_472: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_752, 0)
        squeeze_dim_473: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_753, 0)
        mm_default_236: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_472, squeeze_dim_473);  squeeze_dim_472 = squeeze_dim_473 = None
        unsqueeze_default_236: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_236, 0);  mm_default_236 = None
        view_754: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_236, [512, 16, 1, 1, 1024]);  unsqueeze_default_236 = None
        permute_840: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_754, [0, 1, 4, 2, 3]);  view_754 = None
        view_755: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_840, [512, 16, 1024]);  permute_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_79: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 79)
        inductor_random_default_19: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_79, 'rand');  inductor_lookup_seed_default_79 = None
        convert_element_type_default_86: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_19, torch.bfloat16);  inductor_random_default_19 = None
        gt_79: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_86, 0.1);  convert_element_type_default_86 = None
        mul_313: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_79, view_755);  view_755 = None
        mul_314: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_313, 1.1111111111111112);  mul_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_213: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_314, add_208);  mul_314 = add_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
        getitem_76: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_38[0]
        getitem_77: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_38[1];  var_mean_38 = None
        add_214: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_76, 1e-12);  getitem_76 = None
        rsqrt_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        sub_58: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_213, getitem_77);  add_213 = getitem_77 = None
        mul_315: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_38);  sub_58 = None
        mul_316: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_315, primals_295)
        add_215: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_316, primals_296);  mul_316 = primals_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_851: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_298, torch.bfloat16);  primals_298 = None
        convert_element_type_852: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_297, torch.bfloat16);  primals_297 = None
        convert_element_type_853: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_215, torch.bfloat16)
        view_756: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_853, [8192, 1024]);  convert_element_type_853 = None
        permute_841: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_852, [1, 0]);  convert_element_type_852 = None
        addmm_38: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_851, view_756, permute_841);  convert_element_type_851 = None
        view_757: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_38, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_857: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_757, torch.float32);  view_757 = None
        mul_317: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_857, 0.5)
        mul_318: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_857, 0.7071067811865476);  convert_element_type_857 = None
        erf_19: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_216: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_319: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_317, add_216);  mul_317 = add_216 = None
        convert_element_type_858: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_319, torch.bfloat16);  mul_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_80: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 80)
        inductor_random_default_18: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_80, 'rand');  inductor_lookup_seed_default_80 = None
        convert_element_type_default_85: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_18, torch.bfloat16);  inductor_random_default_18 = None
        gt_80: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_85, 0.1);  convert_element_type_default_85 = None
        mul_320: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_80, convert_element_type_858);  convert_element_type_858 = None
        mul_321: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_320, 1.1111111111111112);  mul_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_859: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_300, torch.bfloat16);  primals_300 = None
        convert_element_type_860: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_299, torch.bfloat16);  primals_299 = None
        view_758: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_321, [8192, 4096]);  mul_321 = None
        permute_842: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_860, [1, 0]);  convert_element_type_860 = None
        addmm_39: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_859, view_758, permute_842);  convert_element_type_859 = None
        view_759: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_39, [512, 16, 1024]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_81: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 81)
        inductor_random_default_17: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_81, 'rand');  inductor_lookup_seed_default_81 = None
        convert_element_type_default_84: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_17, torch.bfloat16);  inductor_random_default_17 = None
        gt_81: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_84, 0.1);  convert_element_type_default_84 = None
        mul_322: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_81, view_759);  view_759 = None
        mul_323: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_322, 1.1111111111111112);  mul_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_217: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_323, add_215);  mul_323 = add_215 = None
        var_mean_39 = torch.ops.aten.var_mean.correction(add_217, [2], correction = 0, keepdim = True)
        getitem_78: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_39[0]
        getitem_79: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_39[1];  var_mean_39 = None
        add_218: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_78, 1e-12);  getitem_78 = None
        rsqrt_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_218);  add_218 = None
        sub_59: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_217, getitem_79);  add_217 = getitem_79 = None
        mul_324: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_39);  sub_59 = None
        mul_325: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_324, primals_301)
        add_219: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_325, primals_302);  mul_325 = primals_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_864: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_219, torch.bfloat16)
        convert_element_type_865: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_303, torch.bfloat16);  primals_303 = None
        unsqueeze_503: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_864, 3);  convert_element_type_864 = None
        unsqueeze_504: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_503, 4);  unsqueeze_503 = None
        unsqueeze_505: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_865, 3);  convert_element_type_865 = None
        unsqueeze_506: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_505, 4);  unsqueeze_505 = None
        view_760: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_504, [1, 8192, 1024]);  unsqueeze_504 = None
        view_761: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_506, [1, 1024, 1024]);  unsqueeze_506 = None
        squeeze_dim_470: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_760, 0)
        squeeze_dim_471: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_761, 0)
        mm_default_235: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_470, squeeze_dim_471);  squeeze_dim_471 = None
        unsqueeze_default_235: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_235, 0);  mm_default_235 = None
        view_762: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_235, [512, 16, 1, 16, 64]);  unsqueeze_default_235 = None
        permute_847: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_762, [0, 1, 3, 4, 2]);  view_762 = None
        view_763: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_847, [512, 16, 16, 64]);  permute_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_869: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_304, torch.bfloat16);  primals_304 = None
        unsqueeze_509: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_869, 3);  convert_element_type_869 = None
        unsqueeze_510: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_509, 4);  unsqueeze_509 = None
        view_765: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_510, [1, 1024, 1024]);  unsqueeze_510 = None
        squeeze_dim_469: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_765, 0)
        mm_default_234: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_470, squeeze_dim_469);  squeeze_dim_469 = None
        unsqueeze_default_234: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_234, 0);  mm_default_234 = None
        view_766: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_234, [512, 16, 1, 16, 64]);  unsqueeze_default_234 = None
        permute_852: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_766, [0, 1, 3, 4, 2]);  view_766 = None
        view_767: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_852, [512, 16, 16, 64]);  permute_852 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_873: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_305, torch.bfloat16);  primals_305 = None
        unsqueeze_513: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_873, 3);  convert_element_type_873 = None
        unsqueeze_514: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_513, 4);  unsqueeze_513 = None
        view_769: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_514, [1, 1024, 1024]);  unsqueeze_514 = None
        squeeze_dim_467: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_769, 0)
        mm_default_233: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_470, squeeze_dim_467);  squeeze_dim_470 = squeeze_dim_467 = None
        unsqueeze_default_233: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_233, 0);  mm_default_233 = None
        view_770: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_233, [512, 16, 1, 16, 64]);  unsqueeze_default_233 = None
        permute_857: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_770, [0, 1, 3, 4, 2]);  view_770 = None
        view_771: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_857, [512, 16, 16, 64]);  permute_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_878: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_306, torch.bfloat16);  primals_306 = None
        unsqueeze_517: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_878, 3);  convert_element_type_878 = None
        unsqueeze_518: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_517, 4);  unsqueeze_517 = None
        view_773: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_518, [1, 1024, 1024]);  unsqueeze_518 = None
        squeeze_dim_465: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_773, 0);  view_773 = None
        mm_default_232: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_465);  squeeze_dim_465 = None
        unsqueeze_default_232: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_232, 0);  mm_default_232 = None
        view_774: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_232, [1024, 16, 1, 16, 64]);  unsqueeze_default_232 = None
        permute_862: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_774, [0, 1, 3, 4, 2]);  view_774 = None
        view_775: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_862, [1024, 16, 16, 64]);  permute_862 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_220: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_763, primals_307);  primals_307 = None
        convert_element_type_881: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16);  add_220 = None
        unsqueeze_519: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_881, 4);  convert_element_type_881 = None
        permute_863: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_519, [1, 2, 0, 4, 3]);  unsqueeze_519 = None
        unsqueeze_520: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_767, 4);  view_767 = None
        permute_864: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_520, [1, 2, 4, 0, 3]);  unsqueeze_520 = None
        permute_865: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_863, [0, 1, 2, 4, 3]);  permute_863 = None
        view_776: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_865, [256, 512, 64]);  permute_865 = None
        permute_866: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_864, [0, 1, 4, 3, 2]);  permute_864 = None
        view_777: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_866, [256, 64, 512]);  permute_866 = None
        bmm_164: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_776, view_777)
        view_778: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_164, [16, 16, 512, 1, 512]);  bmm_164 = None
        permute_867: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_778, [0, 1, 2, 4, 3]);  view_778 = None
        view_779: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_867, [16, 16, 512, 512]);  permute_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_221: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_763, primals_308);  view_763 = primals_308 = None
        convert_element_type_884: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_221, torch.bfloat16);  add_221 = None
        unsqueeze_521: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_884, 4);  convert_element_type_884 = None
        permute_868: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_521, [1, 2, 0, 4, 3]);  unsqueeze_521 = None
        unsqueeze_522: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_775, 4);  view_775 = None
        permute_869: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_522, [1, 2, 4, 0, 3]);  unsqueeze_522 = None
        permute_870: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_868, [0, 1, 2, 4, 3]);  permute_868 = None
        view_780: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_870, [256, 512, 64]);  permute_870 = None
        permute_871: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_869, [0, 1, 4, 3, 2]);  permute_869 = None
        view_781: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_871, [256, 64, 1024]);  permute_871 = None
        bmm_165: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_780, view_781)
        view_782: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_165, [16, 16, 512, 1, 1024]);  bmm_165 = None
        permute_872: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_782, [0, 1, 2, 4, 3]);  view_782 = None
        view_783: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_872, [16, 16, 512, 1024]);  permute_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_784: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_783, [16, 16, 1024, 512]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_21: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_784, 2, 1, 9223372036854775807);  view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_785: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_21, [16, 16, 512, 1023]);  slice_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_20: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_785, [None, None, None, iota_2]);  view_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_222: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_779, index_20);  view_779 = index_20 = None
        add_223: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_222, 0);  add_222 = None

        # No stacktrace found for following nodes
        mul_tensor_12: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_223, 0.125)
        convert_element_type_default_30: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_12, torch.float32);  mul_tensor_12 = None
        convert_element_type_default_31: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_223, torch.float32)
        mul_tensor_13: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_31, 1);  convert_element_type_default_31 = None
        amax_default_6: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_13, [3], True)
        sub_tensor_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_13, amax_default_6);  mul_tensor_13 = None
        mul_tensor_14: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_6, 0.125);  sub_tensor_6 = None
        amax_default_7: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_30, [3], True)
        sub_tensor_7: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_30, amax_default_7)
        abs_default_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_30)
        ne_scalar_3: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_3, inf);  abs_default_3 = None
        eq_tensor_4: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_30, convert_element_type_default_30);  convert_element_type_default_30 = None
        mul_tensor_15: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_4, ne_scalar_3);  eq_tensor_4 = ne_scalar_3 = None
        logical_not_default_6: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_15);  mul_tensor_15 = None
        any_dims_3: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_6, [3], True);  logical_not_default_6 = None
        logical_not_default_7: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_3);  any_dims_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_7, mul_tensor_14, sub_tensor_7);  mul_tensor_14 = sub_tensor_7 = None
        exp_20: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_4);  where_self_4 = None
        sum_21: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_20, [3], True)
        div_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_82: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 82)
        inductor_random_default_16: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_82, 'rand');  inductor_lookup_seed_default_82 = None
        gt_82: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_16, 0.1);  inductor_random_default_16 = None
        mul_327: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_82, div_21);  div_21 = None
        mul_328: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_327, 1.1111111111111112);  mul_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_888: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_328, torch.bfloat16);  mul_328 = None
        unsqueeze_523: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_888, 4);  convert_element_type_888 = None
        unsqueeze_524: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_771, 4);  view_771 = None
        permute_874: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_524, [4, 1, 2, 3, 0]);  unsqueeze_524 = None
        view_786: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_523, [256, 512, 512]);  unsqueeze_523 = None
        permute_876: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_874, [1, 2, 4, 3, 0]);  permute_874 = None
        view_787: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_876, [256, 512, 64]);  permute_876 = None
        bmm_166: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_786, view_787)
        view_788: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_166, [16, 16, 512, 1, 64]);  bmm_166 = None
        permute_877: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_788, [2, 0, 1, 4, 3]);  view_788 = None
        view_789: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_877, [512, 16, 16, 64]);  permute_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_891: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_309, torch.bfloat16);  primals_309 = None
        unsqueeze_525: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_789, 4);  view_789 = None
        permute_878: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_525, [0, 1, 4, 3, 2]);  unsqueeze_525 = None
        unsqueeze_526: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_891, 3);  convert_element_type_891 = None
        unsqueeze_527: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_526, 4);  unsqueeze_526 = None
        permute_879: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_527, [3, 4, 0, 2, 1]);  unsqueeze_527 = None
        permute_880: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_878, [0, 1, 3, 4, 2]);  permute_878 = None
        clone_41: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None
        view_790: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_41, [1, 8192, 1024]);  clone_41 = None
        permute_881: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_879, [3, 4, 2, 0, 1]);  permute_879 = None
        clone_42: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_881, memory_format = torch.contiguous_format);  permute_881 = None
        view_791: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_42, [1, 1024, 1024]);  clone_42 = None
        squeeze_dim_462: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_790, 0)
        squeeze_dim_463: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_791, 0)
        mm_default_231: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_462, squeeze_dim_463);  squeeze_dim_462 = squeeze_dim_463 = None
        unsqueeze_default_231: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_231, 0);  mm_default_231 = None
        view_792: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_231, [512, 16, 1, 1, 1024]);  unsqueeze_default_231 = None
        permute_882: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_792, [0, 1, 4, 2, 3]);  view_792 = None
        view_793: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_882, [512, 16, 1024]);  permute_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_83: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 83)
        inductor_random_default_15: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_83, 'rand');  inductor_lookup_seed_default_83 = None
        convert_element_type_default_83: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_15, torch.bfloat16);  inductor_random_default_15 = None
        gt_83: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_83, 0.1);  convert_element_type_default_83 = None
        mul_329: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_83, view_793);  view_793 = None
        mul_330: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_329, 1.1111111111111112);  mul_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_224: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_330, add_219);  mul_330 = add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
        getitem_80: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_40[0]
        getitem_81: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_40[1];  var_mean_40 = None
        add_225: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_80, 1e-12);  getitem_80 = None
        rsqrt_40: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        sub_61: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_224, getitem_81);  add_224 = getitem_81 = None
        mul_331: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_61, rsqrt_40);  sub_61 = None
        mul_332: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_331, primals_310)
        add_226: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_332, primals_311);  mul_332 = primals_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_894: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_313, torch.bfloat16);  primals_313 = None
        convert_element_type_895: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_312, torch.bfloat16);  primals_312 = None
        convert_element_type_896: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16)
        view_794: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_896, [8192, 1024]);  convert_element_type_896 = None
        permute_883: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_895, [1, 0]);  convert_element_type_895 = None
        addmm_40: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_894, view_794, permute_883);  convert_element_type_894 = None
        view_795: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_40, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_900: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_795, torch.float32);  view_795 = None
        mul_333: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_900, 0.5)
        mul_334: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_900, 0.7071067811865476);  convert_element_type_900 = None
        erf_20: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_334);  mul_334 = None
        add_227: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_335: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_333, add_227);  mul_333 = add_227 = None
        convert_element_type_901: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_335, torch.bfloat16);  mul_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_84: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 84)
        inductor_random_default_14: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_84, 'rand');  inductor_lookup_seed_default_84 = None
        convert_element_type_default_82: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_14, torch.bfloat16);  inductor_random_default_14 = None
        gt_84: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_82, 0.1);  convert_element_type_default_82 = None
        mul_336: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_84, convert_element_type_901);  convert_element_type_901 = None
        mul_337: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_336, 1.1111111111111112);  mul_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_902: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_315, torch.bfloat16);  primals_315 = None
        convert_element_type_903: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_314, torch.bfloat16);  primals_314 = None
        view_796: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_337, [8192, 4096]);  mul_337 = None
        permute_884: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_903, [1, 0]);  convert_element_type_903 = None
        addmm_41: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_902, view_796, permute_884);  convert_element_type_902 = None
        view_797: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_41, [512, 16, 1024]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_85: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 85)
        inductor_random_default_13: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_85, 'rand');  inductor_lookup_seed_default_85 = None
        convert_element_type_default_81: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_13, torch.bfloat16);  inductor_random_default_13 = None
        gt_85: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_81, 0.1);  convert_element_type_default_81 = None
        mul_338: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_85, view_797);  view_797 = None
        mul_339: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_338, 1.1111111111111112);  mul_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_228: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_339, add_226);  mul_339 = add_226 = None
        var_mean_41 = torch.ops.aten.var_mean.correction(add_228, [2], correction = 0, keepdim = True)
        getitem_82: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_41[0]
        getitem_83: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_41[1];  var_mean_41 = None
        add_229: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_82, 1e-12);  getitem_82 = None
        rsqrt_41: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_229);  add_229 = None
        sub_62: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_228, getitem_83);  add_228 = getitem_83 = None
        mul_340: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_41);  sub_62 = None
        mul_341: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_340, primals_316)
        add_230: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_341, primals_317);  mul_341 = primals_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_907: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_230, torch.bfloat16)
        convert_element_type_908: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_318, torch.bfloat16);  primals_318 = None
        unsqueeze_528: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_907, 3);  convert_element_type_907 = None
        unsqueeze_529: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_528, 4);  unsqueeze_528 = None
        unsqueeze_530: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_908, 3);  convert_element_type_908 = None
        unsqueeze_531: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_530, 4);  unsqueeze_530 = None
        view_798: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_529, [1, 8192, 1024]);  unsqueeze_529 = None
        view_799: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_531, [1, 1024, 1024]);  unsqueeze_531 = None
        squeeze_dim_460: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_798, 0)
        squeeze_dim_461: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_799, 0)
        mm_default_230: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_460, squeeze_dim_461);  squeeze_dim_461 = None
        unsqueeze_default_230: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_230, 0);  mm_default_230 = None
        view_800: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_230, [512, 16, 1, 16, 64]);  unsqueeze_default_230 = None
        permute_889: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_800, [0, 1, 3, 4, 2]);  view_800 = None
        view_801: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_889, [512, 16, 16, 64]);  permute_889 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_912: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_319, torch.bfloat16);  primals_319 = None
        unsqueeze_534: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_912, 3);  convert_element_type_912 = None
        unsqueeze_535: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_534, 4);  unsqueeze_534 = None
        view_803: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_535, [1, 1024, 1024]);  unsqueeze_535 = None
        squeeze_dim_459: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_803, 0)
        mm_default_229: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_460, squeeze_dim_459);  squeeze_dim_459 = None
        unsqueeze_default_229: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_229, 0);  mm_default_229 = None
        view_804: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_229, [512, 16, 1, 16, 64]);  unsqueeze_default_229 = None
        permute_894: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_804, [0, 1, 3, 4, 2]);  view_804 = None
        view_805: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_894, [512, 16, 16, 64]);  permute_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_916: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_320, torch.bfloat16);  primals_320 = None
        unsqueeze_538: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_916, 3);  convert_element_type_916 = None
        unsqueeze_539: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_538, 4);  unsqueeze_538 = None
        view_807: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_539, [1, 1024, 1024]);  unsqueeze_539 = None
        squeeze_dim_457: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_807, 0)
        mm_default_228: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_460, squeeze_dim_457);  squeeze_dim_460 = squeeze_dim_457 = None
        unsqueeze_default_228: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_228, 0);  mm_default_228 = None
        view_808: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_228, [512, 16, 1, 16, 64]);  unsqueeze_default_228 = None
        permute_899: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_808, [0, 1, 3, 4, 2]);  view_808 = None
        view_809: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_899, [512, 16, 16, 64]);  permute_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_921: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_321, torch.bfloat16);  primals_321 = None
        unsqueeze_542: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_921, 3);  convert_element_type_921 = None
        unsqueeze_543: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_542, 4);  unsqueeze_542 = None
        view_811: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_543, [1, 1024, 1024]);  unsqueeze_543 = None
        squeeze_dim_455: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_811, 0);  view_811 = None
        mm_default_227: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_455);  squeeze_dim_455 = None
        unsqueeze_default_227: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_227, 0);  mm_default_227 = None
        view_812: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_227, [1024, 16, 1, 16, 64]);  unsqueeze_default_227 = None
        permute_904: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_812, [0, 1, 3, 4, 2]);  view_812 = None
        view_813: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_904, [1024, 16, 16, 64]);  permute_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_231: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_801, primals_322);  primals_322 = None
        convert_element_type_924: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_231, torch.bfloat16);  add_231 = None
        unsqueeze_544: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_924, 4);  convert_element_type_924 = None
        permute_905: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_544, [1, 2, 0, 4, 3]);  unsqueeze_544 = None
        unsqueeze_545: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_805, 4);  view_805 = None
        permute_906: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_545, [1, 2, 4, 0, 3]);  unsqueeze_545 = None
        permute_907: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_905, [0, 1, 2, 4, 3]);  permute_905 = None
        view_814: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_907, [256, 512, 64]);  permute_907 = None
        permute_908: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_906, [0, 1, 4, 3, 2]);  permute_906 = None
        view_815: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_908, [256, 64, 512]);  permute_908 = None
        bmm_172: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_814, view_815)
        view_816: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_172, [16, 16, 512, 1, 512]);  bmm_172 = None
        permute_909: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_816, [0, 1, 2, 4, 3]);  view_816 = None
        view_817: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_909, [16, 16, 512, 512]);  permute_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_232: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_801, primals_323);  view_801 = primals_323 = None
        convert_element_type_927: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_232, torch.bfloat16);  add_232 = None
        unsqueeze_546: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_927, 4);  convert_element_type_927 = None
        permute_910: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_546, [1, 2, 0, 4, 3]);  unsqueeze_546 = None
        unsqueeze_547: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_813, 4);  view_813 = None
        permute_911: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_547, [1, 2, 4, 0, 3]);  unsqueeze_547 = None
        permute_912: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_910, [0, 1, 2, 4, 3]);  permute_910 = None
        view_818: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_912, [256, 512, 64]);  permute_912 = None
        permute_913: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_911, [0, 1, 4, 3, 2]);  permute_911 = None
        view_819: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_913, [256, 64, 1024]);  permute_913 = None
        bmm_173: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_818, view_819)
        view_820: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_173, [16, 16, 512, 1, 1024]);  bmm_173 = None
        permute_914: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_820, [0, 1, 2, 4, 3]);  view_820 = None
        view_821: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_914, [16, 16, 512, 1024]);  permute_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_822: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_821, [16, 16, 1024, 512]);  view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_22: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_822, 2, 1, 9223372036854775807);  view_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_823: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_22, [16, 16, 512, 1023]);  slice_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_21: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_823, [None, None, None, iota_2]);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_233: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_817, index_21);  view_817 = index_21 = None
        add_234: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_233, 0);  add_233 = None

        # No stacktrace found for following nodes
        mul_tensor_8: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_234, 0.125)
        convert_element_type_default_28: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_8, torch.float32);  mul_tensor_8 = None
        convert_element_type_default_29: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_234, torch.float32)
        mul_tensor_9: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_29, 1);  convert_element_type_default_29 = None
        amax_default_4: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_9, [3], True)
        sub_tensor_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_9, amax_default_4);  mul_tensor_9 = None
        mul_tensor_10: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_4, 0.125);  sub_tensor_4 = None
        amax_default_5: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_28, [3], True)
        sub_tensor_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_28, amax_default_5)
        abs_default_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_28)
        ne_scalar_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_2, inf);  abs_default_2 = None
        eq_tensor_3: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_28, convert_element_type_default_28);  convert_element_type_default_28 = None
        mul_tensor_11: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_3, ne_scalar_2);  eq_tensor_3 = ne_scalar_2 = None
        logical_not_default_4: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_11);  mul_tensor_11 = None
        any_dims_2: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_4, [3], True);  logical_not_default_4 = None
        logical_not_default_5: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_2);  any_dims_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_5, mul_tensor_10, sub_tensor_5);  mul_tensor_10 = sub_tensor_5 = None
        exp_21: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_3);  where_self_3 = None
        sum_22: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_21, [3], True)
        div_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_86: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 86)
        inductor_random_default_12: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_86, 'rand');  inductor_lookup_seed_default_86 = None
        gt_86: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_12, 0.1);  inductor_random_default_12 = None
        mul_343: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_86, div_22);  div_22 = None
        mul_344: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_343, 1.1111111111111112);  mul_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_931: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_344, torch.bfloat16);  mul_344 = None
        unsqueeze_548: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_931, 4);  convert_element_type_931 = None
        unsqueeze_549: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_809, 4);  view_809 = None
        permute_916: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_549, [4, 1, 2, 3, 0]);  unsqueeze_549 = None
        view_824: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_548, [256, 512, 512]);  unsqueeze_548 = None
        permute_918: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_916, [1, 2, 4, 3, 0]);  permute_916 = None
        view_825: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_918, [256, 512, 64]);  permute_918 = None
        bmm_174: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_824, view_825)
        view_826: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_174, [16, 16, 512, 1, 64]);  bmm_174 = None
        permute_919: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_826, [2, 0, 1, 4, 3]);  view_826 = None
        view_827: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_919, [512, 16, 16, 64]);  permute_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_934: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_324, torch.bfloat16);  primals_324 = None
        unsqueeze_550: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_827, 4);  view_827 = None
        permute_920: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_550, [0, 1, 4, 3, 2]);  unsqueeze_550 = None
        unsqueeze_551: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_934, 3);  convert_element_type_934 = None
        unsqueeze_552: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_551, 4);  unsqueeze_551 = None
        permute_921: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_552, [3, 4, 0, 2, 1]);  unsqueeze_552 = None
        permute_922: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_920, [0, 1, 3, 4, 2]);  permute_920 = None
        clone_43: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_922, memory_format = torch.contiguous_format);  permute_922 = None
        view_828: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_43, [1, 8192, 1024]);  clone_43 = None
        permute_923: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_921, [3, 4, 2, 0, 1]);  permute_921 = None
        clone_44: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_923, memory_format = torch.contiguous_format);  permute_923 = None
        view_829: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_44, [1, 1024, 1024]);  clone_44 = None
        squeeze_dim_452: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_828, 0)
        squeeze_dim_453: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_829, 0)
        mm_default_226: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_452, squeeze_dim_453);  squeeze_dim_452 = squeeze_dim_453 = None
        unsqueeze_default_226: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_226, 0);  mm_default_226 = None
        view_830: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_226, [512, 16, 1, 1, 1024]);  unsqueeze_default_226 = None
        permute_924: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_830, [0, 1, 4, 2, 3]);  view_830 = None
        view_831: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_924, [512, 16, 1024]);  permute_924 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_87: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 87)
        inductor_random_default_11: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_87, 'rand');  inductor_lookup_seed_default_87 = None
        convert_element_type_default_80: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_11, torch.bfloat16);  inductor_random_default_11 = None
        gt_87: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_80, 0.1);  convert_element_type_default_80 = None
        mul_345: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_87, view_831);  view_831 = None
        mul_346: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_345, 1.1111111111111112);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_235: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_346, add_230);  mul_346 = add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_235, [2], correction = 0, keepdim = True)
        getitem_84: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_42[0]
        getitem_85: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_42[1];  var_mean_42 = None
        add_236: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_84, 1e-12);  getitem_84 = None
        rsqrt_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        sub_64: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_235, getitem_85);  add_235 = getitem_85 = None
        mul_347: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_42);  sub_64 = None
        mul_348: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_347, primals_325)
        add_237: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_348, primals_326);  mul_348 = primals_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_937: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_328, torch.bfloat16);  primals_328 = None
        convert_element_type_938: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_327, torch.bfloat16);  primals_327 = None
        convert_element_type_939: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_237, torch.bfloat16)
        view_832: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_939, [8192, 1024]);  convert_element_type_939 = None
        permute_925: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_938, [1, 0]);  convert_element_type_938 = None
        addmm_42: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_937, view_832, permute_925);  convert_element_type_937 = None
        view_833: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_42, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_943: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_833, torch.float32);  view_833 = None
        mul_349: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_943, 0.5)
        mul_350: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_943, 0.7071067811865476);  convert_element_type_943 = None
        erf_21: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_350);  mul_350 = None
        add_238: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_351: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_349, add_238);  mul_349 = add_238 = None
        convert_element_type_944: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_351, torch.bfloat16);  mul_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_88: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 88)
        inductor_random_default_10: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_88, 'rand');  inductor_lookup_seed_default_88 = None
        convert_element_type_default_79: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_10, torch.bfloat16);  inductor_random_default_10 = None
        gt_88: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_79, 0.1);  convert_element_type_default_79 = None
        mul_352: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_88, convert_element_type_944);  convert_element_type_944 = None
        mul_353: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_352, 1.1111111111111112);  mul_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_945: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_330, torch.bfloat16);  primals_330 = None
        convert_element_type_946: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_329, torch.bfloat16);  primals_329 = None
        view_834: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_353, [8192, 4096]);  mul_353 = None
        permute_926: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_946, [1, 0]);  convert_element_type_946 = None
        addmm_43: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_945, view_834, permute_926);  convert_element_type_945 = None
        view_835: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_43, [512, 16, 1024]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_89: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 89)
        inductor_random_default_9: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_89, 'rand');  inductor_lookup_seed_default_89 = None
        convert_element_type_default_78: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_9, torch.bfloat16);  inductor_random_default_9 = None
        gt_89: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_78, 0.1);  convert_element_type_default_78 = None
        mul_354: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_89, view_835);  view_835 = None
        mul_355: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_354, 1.1111111111111112);  mul_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_239: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_355, add_237);  mul_355 = add_237 = None
        var_mean_43 = torch.ops.aten.var_mean.correction(add_239, [2], correction = 0, keepdim = True)
        getitem_86: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_43[0]
        getitem_87: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_43[1];  var_mean_43 = None
        add_240: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_86, 1e-12);  getitem_86 = None
        rsqrt_43: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_240);  add_240 = None
        sub_65: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_239, getitem_87);  add_239 = getitem_87 = None
        mul_356: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_65, rsqrt_43);  sub_65 = None
        mul_357: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_356, primals_331)
        add_241: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_357, primals_332);  mul_357 = primals_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_950: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_241, torch.bfloat16)
        convert_element_type_951: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_333, torch.bfloat16);  primals_333 = None
        unsqueeze_553: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_950, 3);  convert_element_type_950 = None
        unsqueeze_554: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_553, 4);  unsqueeze_553 = None
        unsqueeze_555: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_951, 3);  convert_element_type_951 = None
        unsqueeze_556: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_555, 4);  unsqueeze_555 = None
        view_836: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_554, [1, 8192, 1024]);  unsqueeze_554 = None
        view_837: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_556, [1, 1024, 1024]);  unsqueeze_556 = None
        squeeze_dim_450: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_836, 0)
        squeeze_dim_451: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_837, 0)
        mm_default_225: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_450, squeeze_dim_451);  squeeze_dim_451 = None
        unsqueeze_default_225: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_225, 0);  mm_default_225 = None
        view_838: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_225, [512, 16, 1, 16, 64]);  unsqueeze_default_225 = None
        permute_931: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_838, [0, 1, 3, 4, 2]);  view_838 = None
        view_839: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_931, [512, 16, 16, 64]);  permute_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_955: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_334, torch.bfloat16);  primals_334 = None
        unsqueeze_559: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_955, 3);  convert_element_type_955 = None
        unsqueeze_560: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_559, 4);  unsqueeze_559 = None
        view_841: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_560, [1, 1024, 1024]);  unsqueeze_560 = None
        squeeze_dim_449: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_841, 0)
        mm_default_224: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_450, squeeze_dim_449);  squeeze_dim_449 = None
        unsqueeze_default_224: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_224, 0);  mm_default_224 = None
        view_842: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_224, [512, 16, 1, 16, 64]);  unsqueeze_default_224 = None
        permute_936: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_842, [0, 1, 3, 4, 2]);  view_842 = None
        view_843: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_936, [512, 16, 16, 64]);  permute_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_959: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_335, torch.bfloat16);  primals_335 = None
        unsqueeze_563: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_959, 3);  convert_element_type_959 = None
        unsqueeze_564: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_563, 4);  unsqueeze_563 = None
        view_845: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_564, [1, 1024, 1024]);  unsqueeze_564 = None
        squeeze_dim_447: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_845, 0)
        mm_default_223: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_450, squeeze_dim_447);  squeeze_dim_450 = squeeze_dim_447 = None
        unsqueeze_default_223: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_223, 0);  mm_default_223 = None
        view_846: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_223, [512, 16, 1, 16, 64]);  unsqueeze_default_223 = None
        permute_941: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_846, [0, 1, 3, 4, 2]);  view_846 = None
        view_847: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_941, [512, 16, 16, 64]);  permute_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_964: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_336, torch.bfloat16);  primals_336 = None
        unsqueeze_567: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_964, 3);  convert_element_type_964 = None
        unsqueeze_568: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_567, 4);  unsqueeze_567 = None
        view_849: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_568, [1, 1024, 1024]);  unsqueeze_568 = None
        squeeze_dim_445: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_849, 0);  view_849 = None
        mm_default_222: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_445);  squeeze_dim_445 = None
        unsqueeze_default_222: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_222, 0);  mm_default_222 = None
        view_850: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_222, [1024, 16, 1, 16, 64]);  unsqueeze_default_222 = None
        permute_946: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_850, [0, 1, 3, 4, 2]);  view_850 = None
        view_851: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_946, [1024, 16, 16, 64]);  permute_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_242: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_839, primals_337);  primals_337 = None
        convert_element_type_967: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_242, torch.bfloat16);  add_242 = None
        unsqueeze_569: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_967, 4);  convert_element_type_967 = None
        permute_947: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_569, [1, 2, 0, 4, 3]);  unsqueeze_569 = None
        unsqueeze_570: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_843, 4);  view_843 = None
        permute_948: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_570, [1, 2, 4, 0, 3]);  unsqueeze_570 = None
        permute_949: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_947, [0, 1, 2, 4, 3]);  permute_947 = None
        view_852: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_949, [256, 512, 64]);  permute_949 = None
        permute_950: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_948, [0, 1, 4, 3, 2]);  permute_948 = None
        view_853: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_950, [256, 64, 512]);  permute_950 = None
        bmm_180: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_852, view_853)
        view_854: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_180, [16, 16, 512, 1, 512]);  bmm_180 = None
        permute_951: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_854, [0, 1, 2, 4, 3]);  view_854 = None
        view_855: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_951, [16, 16, 512, 512]);  permute_951 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_243: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_839, primals_338);  view_839 = primals_338 = None
        convert_element_type_970: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_243, torch.bfloat16);  add_243 = None
        unsqueeze_571: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_970, 4);  convert_element_type_970 = None
        permute_952: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_571, [1, 2, 0, 4, 3]);  unsqueeze_571 = None
        unsqueeze_572: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_851, 4);  view_851 = None
        permute_953: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_572, [1, 2, 4, 0, 3]);  unsqueeze_572 = None
        permute_954: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_952, [0, 1, 2, 4, 3]);  permute_952 = None
        view_856: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_954, [256, 512, 64]);  permute_954 = None
        permute_955: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_953, [0, 1, 4, 3, 2]);  permute_953 = None
        view_857: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_955, [256, 64, 1024]);  permute_955 = None
        bmm_181: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_856, view_857)
        view_858: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_181, [16, 16, 512, 1, 1024]);  bmm_181 = None
        permute_956: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_858, [0, 1, 2, 4, 3]);  view_858 = None
        view_859: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_956, [16, 16, 512, 1024]);  permute_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_860: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_859, [16, 16, 1024, 512]);  view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_23: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_860, 2, 1, 9223372036854775807);  view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_861: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_23, [16, 16, 512, 1023]);  slice_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_22: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_861, [None, None, None, iota_2]);  view_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_244: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_855, index_22);  view_855 = index_22 = None
        add_245: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_244, 0);  add_244 = None

        # No stacktrace found for following nodes
        mul_tensor_4: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_245, 0.125)
        convert_element_type_default_26: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor_4, torch.float32);  mul_tensor_4 = None
        convert_element_type_default_27: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_245, torch.float32)
        mul_tensor_5: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_27, 1);  convert_element_type_default_27 = None
        amax_default_2: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_5, [3], True)
        sub_tensor_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_5, amax_default_2);  mul_tensor_5 = None
        mul_tensor_6: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor_2, 0.125);  sub_tensor_2 = None
        amax_default_3: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_26, [3], True)
        sub_tensor_3: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_26, amax_default_3)
        abs_default_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_26)
        ne_scalar_1: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default_1, inf);  abs_default_1 = None
        eq_tensor_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_26, convert_element_type_default_26);  convert_element_type_default_26 = None
        mul_tensor_7: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_2, ne_scalar_1);  eq_tensor_2 = ne_scalar_1 = None
        logical_not_default_2: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_7);  mul_tensor_7 = None
        any_dims_1: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default_2, [3], True);  logical_not_default_2 = None
        logical_not_default_3: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims_1);  any_dims_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_3, mul_tensor_6, sub_tensor_3);  mul_tensor_6 = sub_tensor_3 = None
        exp_22: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_2);  where_self_2 = None
        sum_23: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_22, [3], True)
        div_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_90: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 90)
        inductor_random_default_8: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_90, 'rand');  inductor_lookup_seed_default_90 = None
        gt_90: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_8, 0.1);  inductor_random_default_8 = None
        mul_359: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_90, div_23);  div_23 = None
        mul_360: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_359, 1.1111111111111112);  mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_974: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_360, torch.bfloat16);  mul_360 = None
        unsqueeze_573: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_974, 4);  convert_element_type_974 = None
        unsqueeze_574: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_847, 4);  view_847 = None
        permute_958: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_574, [4, 1, 2, 3, 0]);  unsqueeze_574 = None
        view_862: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_573, [256, 512, 512]);  unsqueeze_573 = None
        permute_960: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_958, [1, 2, 4, 3, 0]);  permute_958 = None
        view_863: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_960, [256, 512, 64]);  permute_960 = None
        bmm_182: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_862, view_863)
        view_864: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_182, [16, 16, 512, 1, 64]);  bmm_182 = None
        permute_961: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_864, [2, 0, 1, 4, 3]);  view_864 = None
        view_865: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_961, [512, 16, 16, 64]);  permute_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_977: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_339, torch.bfloat16);  primals_339 = None
        unsqueeze_575: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_865, 4);  view_865 = None
        permute_962: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_575, [0, 1, 4, 3, 2]);  unsqueeze_575 = None
        unsqueeze_576: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_977, 3);  convert_element_type_977 = None
        unsqueeze_577: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_576, 4);  unsqueeze_576 = None
        permute_963: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_577, [3, 4, 0, 2, 1]);  unsqueeze_577 = None
        permute_964: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_962, [0, 1, 3, 4, 2]);  permute_962 = None
        clone_45: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_964, memory_format = torch.contiguous_format);  permute_964 = None
        view_866: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_45, [1, 8192, 1024]);  clone_45 = None
        permute_965: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_963, [3, 4, 2, 0, 1]);  permute_963 = None
        clone_46: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_965, memory_format = torch.contiguous_format);  permute_965 = None
        view_867: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_46, [1, 1024, 1024]);  clone_46 = None
        squeeze_dim_442: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_866, 0)
        squeeze_dim_443: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_867, 0)
        mm_default_221: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_442, squeeze_dim_443);  squeeze_dim_442 = squeeze_dim_443 = None
        unsqueeze_default_221: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_221, 0);  mm_default_221 = None
        view_868: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_221, [512, 16, 1, 1, 1024]);  unsqueeze_default_221 = None
        permute_966: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_868, [0, 1, 4, 2, 3]);  view_868 = None
        view_869: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_966, [512, 16, 1024]);  permute_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_91: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 91)
        inductor_random_default_7: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_91, 'rand');  inductor_lookup_seed_default_91 = None
        convert_element_type_default_77: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_7, torch.bfloat16);  inductor_random_default_7 = None
        gt_91: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_77, 0.1);  convert_element_type_default_77 = None
        mul_361: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_91, view_869);  view_869 = None
        mul_362: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_361, 1.1111111111111112);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_246: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_362, add_241);  mul_362 = add_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_246, [2], correction = 0, keepdim = True)
        getitem_88: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_44[0]
        getitem_89: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_44[1];  var_mean_44 = None
        add_247: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_88, 1e-12);  getitem_88 = None
        rsqrt_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        sub_67: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_246, getitem_89);  add_246 = getitem_89 = None
        mul_363: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_44);  sub_67 = None
        mul_364: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_363, primals_340)
        add_248: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_364, primals_341);  mul_364 = primals_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_980: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_343, torch.bfloat16);  primals_343 = None
        convert_element_type_981: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_342, torch.bfloat16);  primals_342 = None
        convert_element_type_982: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16)
        view_870: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_982, [8192, 1024]);  convert_element_type_982 = None
        permute_967: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_981, [1, 0]);  convert_element_type_981 = None
        addmm_44: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_980, view_870, permute_967);  convert_element_type_980 = None
        view_871: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_44, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_986: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_871, torch.float32);  view_871 = None
        mul_365: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_986, 0.5)
        mul_366: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_986, 0.7071067811865476);  convert_element_type_986 = None
        erf_22: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_366);  mul_366 = None
        add_249: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_367: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_365, add_249);  mul_365 = add_249 = None
        convert_element_type_987: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_367, torch.bfloat16);  mul_367 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_92: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 92)
        inductor_random_default_6: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_92, 'rand');  inductor_lookup_seed_default_92 = None
        convert_element_type_default_76: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_6, torch.bfloat16);  inductor_random_default_6 = None
        gt_92: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_76, 0.1);  convert_element_type_default_76 = None
        mul_368: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_92, convert_element_type_987);  convert_element_type_987 = None
        mul_369: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_368, 1.1111111111111112);  mul_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_988: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_345, torch.bfloat16);  primals_345 = None
        convert_element_type_989: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_344, torch.bfloat16);  primals_344 = None
        view_872: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_369, [8192, 4096]);  mul_369 = None
        permute_968: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_989, [1, 0]);  convert_element_type_989 = None
        addmm_45: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_988, view_872, permute_968);  convert_element_type_988 = None
        view_873: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_45, [512, 16, 1024]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_93: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 93)
        inductor_random_default_5: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_93, 'rand');  inductor_lookup_seed_default_93 = None
        convert_element_type_default_75: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_5, torch.bfloat16);  inductor_random_default_5 = None
        gt_93: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_75, 0.1);  convert_element_type_default_75 = None
        mul_370: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_93, view_873);  view_873 = None
        mul_371: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_370, 1.1111111111111112);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_250: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_371, add_248);  mul_371 = add_248 = None
        var_mean_45 = torch.ops.aten.var_mean.correction(add_250, [2], correction = 0, keepdim = True)
        getitem_90: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_45[0]
        getitem_91: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_45[1];  var_mean_45 = None
        add_251: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_90, 1e-12);  getitem_90 = None
        rsqrt_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_251);  add_251 = None
        sub_68: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_250, getitem_91);  add_250 = getitem_91 = None
        mul_372: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_45);  sub_68 = None
        mul_373: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_372, primals_346)
        add_252: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_373, primals_347);  mul_373 = primals_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        convert_element_type_993: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16)
        convert_element_type_994: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_348, torch.bfloat16);  primals_348 = None
        unsqueeze_578: "bf16[512, 16, 1024, 1][16384, 1024, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_993, 3);  convert_element_type_993 = None
        unsqueeze_579: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_578, 4);  unsqueeze_578 = None
        unsqueeze_580: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_994, 3);  convert_element_type_994 = None
        unsqueeze_581: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_580, 4);  unsqueeze_580 = None
        view_874: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_579, [1, 8192, 1024]);  unsqueeze_579 = None
        view_875: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_581, [1, 1024, 1024]);  unsqueeze_581 = None
        squeeze_dim_440: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_874, 0)
        squeeze_dim_441: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_875, 0)
        mm_default_220: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_440, squeeze_dim_441);  squeeze_dim_441 = None
        unsqueeze_default_220: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_220, 0);  mm_default_220 = None
        view_876: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_220, [512, 16, 1, 16, 64]);  unsqueeze_default_220 = None
        permute_973: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_876, [0, 1, 3, 4, 2]);  view_876 = None
        view_877: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_973, [512, 16, 16, 64]);  permute_973 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        convert_element_type_998: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_349, torch.bfloat16);  primals_349 = None
        unsqueeze_584: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_998, 3);  convert_element_type_998 = None
        unsqueeze_585: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_584, 4);  unsqueeze_584 = None
        view_879: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_585, [1, 1024, 1024]);  unsqueeze_585 = None
        squeeze_dim_439: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_879, 0)
        mm_default_219: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_440, squeeze_dim_439);  squeeze_dim_439 = None
        unsqueeze_default_219: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_219, 0);  mm_default_219 = None
        view_880: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_219, [512, 16, 1, 16, 64]);  unsqueeze_default_219 = None
        permute_978: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_880, [0, 1, 3, 4, 2]);  view_880 = None
        view_881: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_978, [512, 16, 16, 64]);  permute_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        convert_element_type_1002: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_350, torch.bfloat16);  primals_350 = None
        unsqueeze_588: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1002, 3);  convert_element_type_1002 = None
        unsqueeze_589: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_588, 4);  unsqueeze_588 = None
        view_883: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_589, [1, 1024, 1024]);  unsqueeze_589 = None
        squeeze_dim_437: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_883, 0)
        mm_default_218: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_440, squeeze_dim_437);  squeeze_dim_440 = squeeze_dim_437 = None
        unsqueeze_default_218: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_218, 0);  mm_default_218 = None
        view_884: "bf16[512, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_218, [512, 16, 1, 16, 64]);  unsqueeze_default_218 = None
        permute_983: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_884, [0, 1, 3, 4, 2]);  view_884 = None
        view_885: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_983, [512, 16, 16, 64]);  permute_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        convert_element_type_1007: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_351, torch.bfloat16);  primals_351 = None
        unsqueeze_592: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1007, 3);  convert_element_type_1007 = None
        unsqueeze_593: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_592, 4);  unsqueeze_592 = None
        view_887: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_593, [1, 1024, 1024]);  unsqueeze_593 = None
        squeeze_dim_435: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_887, 0);  view_887 = None
        mm_default_217: "bf16[16384, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_664, squeeze_dim_435);  squeeze_dim_664 = squeeze_dim_435 = None
        unsqueeze_default_217: "bf16[1, 16384, 1024][16777216, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_217, 0);  mm_default_217 = None
        view_888: "bf16[1024, 16, 1, 16, 64][16384, 1024, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_217, [1024, 16, 1, 16, 64]);  unsqueeze_default_217 = None
        permute_988: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_888, [0, 1, 3, 4, 2]);  view_888 = None
        view_889: "bf16[1024, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.reshape.default(permute_988, [1024, 16, 16, 64]);  permute_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        add_253: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_877, primals_352);  primals_352 = None
        convert_element_type_1010: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_253, torch.bfloat16);  add_253 = None
        unsqueeze_594: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1010, 4);  convert_element_type_1010 = None
        permute_989: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_594, [1, 2, 0, 4, 3]);  unsqueeze_594 = None
        unsqueeze_595: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_881, 4);  view_881 = None
        permute_990: "bf16[16, 16, 1, 512, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_595, [1, 2, 4, 0, 3]);  unsqueeze_595 = None
        permute_991: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_989, [0, 1, 2, 4, 3]);  permute_989 = None
        view_890: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_991, [256, 512, 64]);  permute_991 = None
        permute_992: "bf16[16, 16, 64, 512, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_990, [0, 1, 4, 3, 2]);  permute_990 = None
        view_891: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_992, [256, 64, 512]);  permute_992 = None
        bmm_188: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.bmm.default(view_890, view_891)
        view_892: "bf16[16, 16, 512, 1, 512][4194304, 262144, 512, 512, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_188, [16, 16, 512, 1, 512]);  bmm_188 = None
        permute_993: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_892, [0, 1, 2, 4, 3]);  view_892 = None
        view_893: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(permute_993, [16, 16, 512, 512]);  permute_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        add_254: "f32[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.aten.add.Tensor(view_877, primals_353);  view_877 = primals_353 = None
        convert_element_type_1013: "bf16[512, 16, 16, 64][16384, 1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16);  add_254 = None
        unsqueeze_596: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1013, 4);  convert_element_type_1013 = None
        permute_994: "bf16[16, 16, 512, 1, 64][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_596, [1, 2, 0, 4, 3]);  unsqueeze_596 = None
        unsqueeze_597: "bf16[1024, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_889, 4);  view_889 = None
        permute_995: "bf16[16, 16, 1, 1024, 64][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(unsqueeze_597, [1, 2, 4, 0, 3]);  unsqueeze_597 = None
        permute_996: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_994, [0, 1, 2, 4, 3]);  permute_994 = None
        view_894: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_996, [256, 512, 64]);  permute_996 = None
        permute_997: "bf16[16, 16, 64, 1024, 1][1024, 64, 1, 16384, 1]cuda:0" = torch.ops.aten.permute.default(permute_995, [0, 1, 4, 3, 2]);  permute_995 = None
        view_895: "bf16[256, 64, 1024][64, 1, 16384]cuda:0" = torch.ops.aten.reshape.default(permute_997, [256, 64, 1024]);  permute_997 = None
        bmm_189: "bf16[256, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.bmm.default(view_894, view_895)
        view_896: "bf16[16, 16, 512, 1, 1024][8388608, 524288, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_189, [16, 16, 512, 1, 1024]);  bmm_189 = None
        permute_998: "bf16[16, 16, 512, 1024, 1][8388608, 524288, 1024, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_896, [0, 1, 2, 4, 3]);  view_896 = None
        view_897: "bf16[16, 16, 512, 1024][8388608, 524288, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_998, [16, 16, 512, 1024]);  permute_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:84 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[3], x_size[2])
        view_898: "bf16[16, 16, 1024, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.reshape.default(view_897, [16, 16, 1024, 512]);  view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:85 in rel_shift_bnij, code: x = x[:, :, 1:, :]
        slice_24: "bf16[16, 16, 1023, 512][8388608, 524288, 512, 1]cuda:0" = torch.ops.aten.slice.Tensor(view_898, 2, 1, 9223372036854775807);  view_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:86 in rel_shift_bnij, code: x = x.reshape(x_size[0], x_size[1], x_size[2], x_size[3] - 1)
        view_899: "bf16[16, 16, 512, 1023][8388608, 524288, 1023, 1]cuda:0" = torch.ops.aten.reshape.default(slice_24, [16, 16, 512, 1023]);  slice_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:90 in rel_shift_bnij, code: x = torch.index_select(x, 3, torch.arange(klen, device=x.device, dtype=torch.long))
        index_23: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.index.Tensor(view_899, [None, None, None, iota_2]);  view_899 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:122 in rel_attn_core, code: attn_score = (ac + bd + ef) * self.scale
        add_255: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(view_893, index_23);  view_893 = index_23 = None
        add_256: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.add.Tensor(add_255, 0);  add_255 = None

        # No stacktrace found for following nodes
        mul_tensor: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(add_256, 0.125)
        convert_element_type_default_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_tensor, torch.float32);  mul_tensor = None
        convert_element_type_default_25: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_256, torch.float32)
        mul_tensor_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_default_25, 1);  convert_element_type_default_25 = None
        amax_default: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(mul_tensor_1, [3], True)
        sub_tensor: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(mul_tensor_1, amax_default);  mul_tensor_1 = None
        mul_tensor_2: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_tensor, 0.125);  sub_tensor = None
        amax_default_1: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_default_24, [3], True)
        sub_tensor_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_default_24, amax_default_1)
        abs_default: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.abs.default(convert_element_type_default_24)
        ne_scalar: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        eq_tensor_1: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.eq.Tensor(convert_element_type_default_24, convert_element_type_default_24);  convert_element_type_default_24 = None
        mul_tensor_3: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(eq_tensor_1, ne_scalar);  eq_tensor_1 = ne_scalar = None
        logical_not_default: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.logical_not.default(mul_tensor_3);  mul_tensor_3 = None
        any_dims: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.any.dims(logical_not_default, [3], True);  logical_not_default = None
        logical_not_default_1: "b8[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:131 in rel_attn_core, code: attn_prob = nn.functional.softmax(attn_score, dim=3)
        where_self_1: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.where.self(logical_not_default_1, mul_tensor_2, sub_tensor_1);  mul_tensor_2 = sub_tensor_1 = None
        exp_23: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.exp.default(where_self_1);  where_self_1 = None
        sum_24: "f32[16, 16, 512, 1][8192, 512, 1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_23, [3], True)
        div_24: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:132 in rel_attn_core, code: attn_prob = self.dropout(attn_prob)
        inductor_lookup_seed_default_94: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 94)
        inductor_random_default_4: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.inductor_random.default([16, 16, 512, 512], inductor_lookup_seed_default_94, 'rand');  inductor_lookup_seed_default_94 = None
        gt_94: "b8[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default_4, 0.1);  inductor_random_default_4 = None
        mul_375: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_94, div_24);  div_24 = None
        mul_376: "f32[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_375, 1.1111111111111112);  mul_375 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        convert_element_type_1017: "bf16[16, 16, 512, 512][4194304, 262144, 512, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_376, torch.bfloat16);  mul_376 = None
        unsqueeze_598: "bf16[16, 16, 512, 512, 1][4194304, 262144, 512, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1017, 4);  convert_element_type_1017 = None
        unsqueeze_599: "bf16[512, 16, 16, 64, 1][16384, 1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_885, 4);  view_885 = None
        permute_1000: "bf16[1, 16, 16, 64, 512][1, 1024, 64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(unsqueeze_599, [4, 1, 2, 3, 0]);  unsqueeze_599 = None
        view_900: "bf16[256, 512, 512][262144, 512, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_598, [256, 512, 512]);  unsqueeze_598 = None
        permute_1002: "bf16[16, 16, 512, 64, 1][1024, 64, 16384, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_1000, [1, 2, 4, 3, 0]);  permute_1000 = None
        view_901: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1002, [256, 512, 64]);  permute_1002 = None
        bmm_190: "bf16[256, 512, 64][32768, 64, 1]cuda:0" = torch.ops.aten.bmm.default(view_900, view_901)
        view_902: "bf16[16, 16, 512, 1, 64][524288, 32768, 64, 64, 1]cuda:0" = torch.ops.aten.reshape.default(bmm_190, [16, 16, 512, 1, 64]);  bmm_190 = None
        permute_1003: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 64]cuda:0" = torch.ops.aten.permute.default(view_902, [2, 0, 1, 4, 3]);  view_902 = None
        view_903: "bf16[512, 16, 16, 64][64, 524288, 32768, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1003, [512, 16, 16, 64]);  permute_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        convert_element_type_1020: "bf16[1024, 16, 64][1024, 64, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_354, torch.bfloat16);  primals_354 = None
        unsqueeze_600: "bf16[512, 16, 16, 64, 1][64, 524288, 32768, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(view_903, 4);  view_903 = None
        permute_1004: "bf16[512, 16, 1, 64, 16][64, 524288, 1, 1, 32768]cuda:0" = torch.ops.aten.permute.default(unsqueeze_600, [0, 1, 4, 3, 2]);  unsqueeze_600 = None
        unsqueeze_601: "bf16[1024, 16, 64, 1][1024, 64, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(convert_element_type_1020, 3);  convert_element_type_1020 = None
        unsqueeze_602: "bf16[1024, 16, 64, 1, 1][1024, 64, 1, 1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(unsqueeze_601, 4);  unsqueeze_601 = None
        permute_1005: "bf16[1, 1, 1024, 64, 16][1, 1, 1024, 1, 64]cuda:0" = torch.ops.aten.permute.default(unsqueeze_602, [3, 4, 0, 2, 1]);  unsqueeze_602 = None
        permute_1006: "bf16[512, 16, 64, 16, 1][64, 524288, 1, 32768, 1]cuda:0" = torch.ops.aten.permute.default(permute_1004, [0, 1, 3, 4, 2]);  permute_1004 = None
        clone_47: "bf16[512, 16, 64, 16, 1][16384, 1024, 16, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
        view_904: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_47, [1, 8192, 1024]);  clone_47 = None
        permute_1007: "bf16[64, 16, 1024, 1, 1][1, 64, 1024, 1, 1]cuda:0" = torch.ops.aten.permute.default(permute_1005, [3, 4, 2, 0, 1]);  permute_1005 = None
        clone_48: "bf16[64, 16, 1024, 1, 1][16384, 1024, 1, 1, 1]cuda:0" = torch.ops.aten.clone.default(permute_1007, memory_format = torch.contiguous_format);  permute_1007 = None
        view_905: "bf16[1, 1024, 1024][1048576, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(clone_48, [1, 1024, 1024]);  clone_48 = None
        squeeze_dim_432: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_904, 0)
        squeeze_dim_433: "bf16[1024, 1024][1024, 1]cuda:0" = torch.ops.aten.squeeze.dim(view_905, 0)
        mm_default_216: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.mm.default(squeeze_dim_432, squeeze_dim_433);  squeeze_dim_432 = squeeze_dim_433 = None
        unsqueeze_default_216: "bf16[1, 8192, 1024][8388608, 1024, 1]cuda:0" = torch.ops.aten.unsqueeze.default(mm_default_216, 0);  mm_default_216 = None
        view_906: "bf16[512, 16, 1, 1, 1024][16384, 1024, 1024, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(unsqueeze_default_216, [512, 16, 1, 1, 1024]);  unsqueeze_default_216 = None
        permute_1008: "bf16[512, 16, 1024, 1, 1][16384, 1024, 1, 1024, 1024]cuda:0" = torch.ops.aten.permute.default(view_906, [0, 1, 4, 2, 3]);  view_906 = None
        view_907: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(permute_1008, [512, 16, 1024]);  permute_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:147 in post_attention, code: attn_out = self.dropout(attn_out)
        inductor_lookup_seed_default_95: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 95)
        inductor_random_default_3: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_95, 'rand');  inductor_lookup_seed_default_95 = None
        convert_element_type_default_74: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_3, torch.bfloat16);  inductor_random_default_3 = None
        gt_95: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_74, 0.1);  convert_element_type_default_74 = None
        mul_377: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_95, view_907);  view_907 = None
        mul_378: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_377, 1.1111111111111112);  mul_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:149 in post_attention, code: attn_out = attn_out + h
        add_257: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_378, add_252);  mul_378 = add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_257, [2], correction = 0, keepdim = True)
        getitem_92: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_46[0]
        getitem_93: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_46[1];  var_mean_46 = None
        add_258: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_92, 1e-12);  getitem_92 = None
        rsqrt_46: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        sub_70: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_257, getitem_93);  add_257 = getitem_93 = None
        mul_379: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_46);  sub_70 = None
        mul_380: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_379, primals_355)
        add_259: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_380, primals_356);  mul_380 = primals_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        convert_element_type_1023: "bf16[4096][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_358, torch.bfloat16);  primals_358 = None
        convert_element_type_1024: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_357, torch.bfloat16);  primals_357 = None
        convert_element_type_1025: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(add_259, torch.bfloat16)
        view_908: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1025, [8192, 1024]);  convert_element_type_1025 = None
        permute_1009: "bf16[1024, 4096][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1024, [1, 0]);  convert_element_type_1024 = None
        addmm_46: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1023, view_908, permute_1009);  convert_element_type_1023 = None
        view_909: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_46, [512, 16, 4096])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        convert_element_type_1029: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_909, torch.float32);  view_909 = None
        mul_381: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1029, 0.5)
        mul_382: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(convert_element_type_1029, 0.7071067811865476);  convert_element_type_1029 = None
        erf_23: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.erf.default(mul_382);  mul_382 = None
        add_260: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_383: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_381, add_260);  mul_381 = add_260 = None
        convert_element_type_1030: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(mul_383, torch.bfloat16);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:301 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_96: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 96)
        inductor_random_default_2: "f32[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 4096], inductor_lookup_seed_default_96, 'rand');  inductor_lookup_seed_default_96 = None
        convert_element_type_default_73: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_2, torch.bfloat16);  inductor_random_default_2 = None
        gt_96: "b8[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_73, 0.1);  convert_element_type_default_73 = None
        mul_384: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_96, convert_element_type_1030);  convert_element_type_1030 = None
        mul_385: "bf16[512, 16, 4096][65536, 4096, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_384, 1.1111111111111112);  mul_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        convert_element_type_1031: "bf16[1024][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_360, torch.bfloat16);  primals_360 = None
        convert_element_type_1032: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_359, torch.bfloat16);  primals_359 = None
        view_910: "bf16[8192, 4096][4096, 1]cuda:0" = torch.ops.aten.reshape.default(mul_385, [8192, 4096]);  mul_385 = None
        permute_1010: "bf16[4096, 1024][1, 4096]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1032, [1, 0]);  convert_element_type_1032 = None
        addmm_47: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1031, view_910, permute_1010);  convert_element_type_1031 = None
        view_911: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_47, [512, 16, 1024]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:303 in forward, code: output = self.dropout(output)
        inductor_lookup_seed_default_97: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 97)
        inductor_random_default_1: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_97, 'rand');  inductor_lookup_seed_default_97 = None
        convert_element_type_default_72: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(inductor_random_default_1, torch.bfloat16);  inductor_random_default_1 = None
        gt_97: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(convert_element_type_default_72, 0.1);  convert_element_type_default_72 = None
        mul_386: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_97, view_911);  view_911 = None
        mul_387: "bf16[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_386, 1.1111111111111112);  mul_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        add_261: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_387, add_259);  mul_387 = add_259 = None
        var_mean_47 = torch.ops.aten.var_mean.correction(add_261, [2], correction = 0, keepdim = True)
        getitem_94: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_47[0]
        getitem_95: "f32[512, 16, 1][16, 1, 1]cuda:0" = var_mean_47[1];  var_mean_47 = None
        add_262: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.add.Tensor(getitem_94, 1e-12);  getitem_94 = None
        rsqrt_47: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.rsqrt.default(add_262);  add_262 = None
        sub_71: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.sub.Tensor(add_261, getitem_95);  add_261 = getitem_95 = None
        mul_388: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_47);  sub_71 = None
        mul_389: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_388, primals_361)
        add_263: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.add.Tensor(mul_389, primals_362);  mul_389 = primals_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1177 in forward, code: output = self.dropout(output_g if output_g is not None else output_h)
        inductor_lookup_seed_default_98: "i64[][]cuda:0" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds_default, 98);  inductor_seeds_default = None
        inductor_random_default: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.prims.inductor_random.default([512, 16, 1024], inductor_lookup_seed_default_98, 'rand');  inductor_lookup_seed_default_98 = None
        gt_98: "b8[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.gt.Scalar(inductor_random_default, 0.1);  inductor_random_default = None
        mul_390: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(gt_98, add_263);  add_263 = None
        mul_391: "f32[512, 16, 1024][16384, 1024, 1]cuda:0" = torch.ops.aten.mul.Tensor(mul_390, 1.1111111111111112);  mul_390 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1180 in forward, code: output = output.permute(1, 0, 2).contiguous()
        permute_1011: "f32[16, 512, 1024][1024, 16384, 1]cuda:0" = torch.ops.aten.permute.default(mul_391, [1, 0, 2]);  mul_391 = None
        clone_49: "f32[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.aten.clone.default(permute_1011, memory_format = torch.contiguous_format);  permute_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        convert_element_type_1036: "bf16[32000][1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_363, torch.bfloat16);  primals_363 = None
        convert_element_type_1037: "bf16[32000, 1024][1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(primals_2, torch.bfloat16);  primals_2 = None
        convert_element_type_1038: "bf16[16, 512, 1024][524288, 1024, 1]cuda:0" = torch.ops.prims.convert_element_type.default(clone_49, torch.bfloat16);  clone_49 = None
        view_912: "bf16[8192, 1024][1024, 1]cuda:0" = torch.ops.aten.reshape.default(convert_element_type_1038, [8192, 1024]);  convert_element_type_1038 = None
        permute_1012: "bf16[1024, 32000][1, 1024]cuda:0" = torch.ops.aten.permute.default(convert_element_type_1037, [1, 0]);  convert_element_type_1037 = None
        addmm_48: "bf16[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.addmm.default(convert_element_type_1036, view_912, permute_1012);  convert_element_type_1036 = None
        view_913: "bf16[16, 512, 32000][16384000, 32000, 1]cuda:0" = torch.ops.aten.reshape.default(addmm_48, [16, 512, 32000]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1432 in forward, code: loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        view_914: "bf16[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.reshape.default(view_913, [-1, 32000])
        view_915: "i64[8192][1]cuda:0" = torch.ops.aten.reshape.default(primals_364, [-1])
        convert_element_type_1042: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(view_914, torch.float32);  view_914 = None
        amax_24: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.amax.default(convert_element_type_1042, [1], True)
        sub_72: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.sub.Tensor(convert_element_type_1042, amax_24);  convert_element_type_1042 = None
        exp_24: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.exp.default(sub_72)
        sum_25: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_73: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.aten.sub.Tensor(sub_72, log);  sub_72 = None
        convert_element_type_1043: "bf16[8192, 32000][32000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(sub_73, torch.bfloat16);  sub_73 = None
        convert_element_type_1044: "f32[8192, 32000][32000, 1]cuda:0" = torch.ops.prims.convert_element_type.default(convert_element_type_1043, torch.float32);  convert_element_type_1043 = None
        ne: "b8[8192][1]cuda:0" = torch.ops.aten.ne.Scalar(view_915, -100)
        full_default: "i64[][]cuda:0" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "i64[8192][1]cuda:0" = torch.ops.aten.where.self(ne, view_915, full_default);  view_915 = full_default = None
        unsqueeze_603: "i64[8192, 1][1, 1]cuda:0" = torch.ops.aten.unsqueeze.default(where, 1);  where = None
        gather: "f32[8192, 1][1, 1]cuda:0" = torch.ops.aten.gather.default(convert_element_type_1044, 1, unsqueeze_603);  convert_element_type_1044 = unsqueeze_603 = None
        squeeze: "f32[8192][1]cuda:0" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[8192][1]cuda:0" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_1: "f32[][]cuda:0" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_1: "f32[8192][1]cuda:0" = torch.ops.aten.where.self(ne, neg, full_default_1);  neg = full_default_1 = None
        sum_26: "i64[][]cuda:0" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_1045: "f32[][]cuda:0" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[][]cuda:0" = torch.ops.aten.sum.default(where_1);  where_1 = None
        div_25: "f32[][]cuda:0" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_1045);  sum_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:1426 in forward, code: logits = self.lm_loss(hidden_states[:, slice_indices, :])
        permute_1013: "bf16[32000, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_1012, [1, 0]);  permute_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_27: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_47, 1024);  rsqrt_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1018: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_1010, [1, 0]);  permute_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1022: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_1009, [1, 0]);  permute_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_28: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_46, 1024);  rsqrt_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1027: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_904, [0, 2, 1]);  view_904 = None
        squeeze_dim_430: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1027, 0);  permute_1027 = None
        permute_1028: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_905, [0, 2, 1]);  view_905 = None
        squeeze_dim_429: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1028, 0);  permute_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1034: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_900, [0, 2, 1]);  view_900 = None
        permute_1035: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_901, [0, 2, 1]);  view_901 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1041: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_894, [0, 2, 1]);  view_894 = None
        permute_1042: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_895, [0, 2, 1]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1048: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_890, [0, 2, 1]);  view_890 = None
        permute_1049: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_891, [0, 2, 1]);  view_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:259 in forward, code: k_head_r = torch.einsum("ibh,hnd->ibnd", r.type(self.r.dtype), self.r)
        permute_1055: "bf16[1, 1024, 16384][16777216, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        squeeze_dim_426: "bf16[1024, 16384][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1055, 0);  permute_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1059: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_874, [0, 2, 1]);  view_874 = None
        squeeze_dim_424: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1059, 0);  permute_1059 = None
        permute_1060: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_883, [0, 2, 1]);  view_883 = None
        squeeze_dim_423: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1060, 0);  permute_1060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1067: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_879, [0, 2, 1]);  view_879 = None
        squeeze_dim_419: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1067, 0);  permute_1067 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1074: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_875, [0, 2, 1]);  view_875 = None
        squeeze_dim_415: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1074, 0);  permute_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_29: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_45, 1024);  rsqrt_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1079: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_968, [1, 0]);  permute_968 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1083: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_967, [1, 0]);  permute_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_30: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_44, 1024);  rsqrt_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1088: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_866, [0, 2, 1]);  view_866 = None
        squeeze_dim_412: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1088, 0);  permute_1088 = None
        permute_1089: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_867, [0, 2, 1]);  view_867 = None
        squeeze_dim_411: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1089, 0);  permute_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1095: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_862, [0, 2, 1]);  view_862 = None
        permute_1096: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_863, [0, 2, 1]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1102: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_856, [0, 2, 1]);  view_856 = None
        permute_1103: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_857, [0, 2, 1]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1109: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_852, [0, 2, 1]);  view_852 = None
        permute_1110: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_853, [0, 2, 1]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1120: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_836, [0, 2, 1]);  view_836 = None
        squeeze_dim_406: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1120, 0);  permute_1120 = None
        permute_1121: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_845, [0, 2, 1]);  view_845 = None
        squeeze_dim_405: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1121, 0);  permute_1121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1128: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_841, [0, 2, 1]);  view_841 = None
        squeeze_dim_401: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1128, 0);  permute_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1135: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_837, [0, 2, 1]);  view_837 = None
        squeeze_dim_397: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1135, 0);  permute_1135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_31: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_43, 1024);  rsqrt_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1140: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_926, [1, 0]);  permute_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1144: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_925, [1, 0]);  permute_925 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_32: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_42, 1024);  rsqrt_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1149: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_828, [0, 2, 1]);  view_828 = None
        squeeze_dim_394: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1149, 0);  permute_1149 = None
        permute_1150: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_829, [0, 2, 1]);  view_829 = None
        squeeze_dim_393: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1150, 0);  permute_1150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1156: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_824, [0, 2, 1]);  view_824 = None
        permute_1157: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_825, [0, 2, 1]);  view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1163: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_818, [0, 2, 1]);  view_818 = None
        permute_1164: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_819, [0, 2, 1]);  view_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1170: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_814, [0, 2, 1]);  view_814 = None
        permute_1171: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_815, [0, 2, 1]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1181: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_798, [0, 2, 1]);  view_798 = None
        squeeze_dim_388: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1181, 0);  permute_1181 = None
        permute_1182: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_807, [0, 2, 1]);  view_807 = None
        squeeze_dim_387: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1182, 0);  permute_1182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1189: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_803, [0, 2, 1]);  view_803 = None
        squeeze_dim_383: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1189, 0);  permute_1189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1196: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_799, [0, 2, 1]);  view_799 = None
        squeeze_dim_379: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1196, 0);  permute_1196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_33: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_41, 1024);  rsqrt_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1201: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_884, [1, 0]);  permute_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1205: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_883, [1, 0]);  permute_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_34: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_40, 1024);  rsqrt_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1210: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_790, [0, 2, 1]);  view_790 = None
        squeeze_dim_376: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1210, 0);  permute_1210 = None
        permute_1211: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_791, [0, 2, 1]);  view_791 = None
        squeeze_dim_375: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1211, 0);  permute_1211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1217: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_786, [0, 2, 1]);  view_786 = None
        permute_1218: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_787, [0, 2, 1]);  view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1224: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_780, [0, 2, 1]);  view_780 = None
        permute_1225: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_781, [0, 2, 1]);  view_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1231: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_776, [0, 2, 1]);  view_776 = None
        permute_1232: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_777, [0, 2, 1]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1242: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_760, [0, 2, 1]);  view_760 = None
        squeeze_dim_370: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1242, 0);  permute_1242 = None
        permute_1243: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_769, [0, 2, 1]);  view_769 = None
        squeeze_dim_369: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1243, 0);  permute_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1250: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_765, [0, 2, 1]);  view_765 = None
        squeeze_dim_365: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1250, 0);  permute_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1257: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_761, [0, 2, 1]);  view_761 = None
        squeeze_dim_361: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1257, 0);  permute_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_35: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_39, 1024);  rsqrt_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1262: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_842, [1, 0]);  permute_842 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1266: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_841, [1, 0]);  permute_841 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_36: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_38, 1024);  rsqrt_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1271: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_752, [0, 2, 1]);  view_752 = None
        squeeze_dim_358: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1271, 0);  permute_1271 = None
        permute_1272: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_753, [0, 2, 1]);  view_753 = None
        squeeze_dim_357: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1272, 0);  permute_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1278: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_748, [0, 2, 1]);  view_748 = None
        permute_1279: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_749, [0, 2, 1]);  view_749 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1285: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_742, [0, 2, 1]);  view_742 = None
        permute_1286: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_743, [0, 2, 1]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1292: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_738, [0, 2, 1]);  view_738 = None
        permute_1293: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_739, [0, 2, 1]);  view_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1303: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_722, [0, 2, 1]);  view_722 = None
        squeeze_dim_352: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1303, 0);  permute_1303 = None
        permute_1304: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_731, [0, 2, 1]);  view_731 = None
        squeeze_dim_351: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1304, 0);  permute_1304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1311: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_727, [0, 2, 1]);  view_727 = None
        squeeze_dim_347: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1311, 0);  permute_1311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1318: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_723, [0, 2, 1]);  view_723 = None
        squeeze_dim_343: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1318, 0);  permute_1318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_37: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_37, 1024);  rsqrt_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1323: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_800, [1, 0]);  permute_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1327: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_799, [1, 0]);  permute_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_38: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_36, 1024);  rsqrt_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1332: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_714, [0, 2, 1]);  view_714 = None
        squeeze_dim_340: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1332, 0);  permute_1332 = None
        permute_1333: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        squeeze_dim_339: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1333, 0);  permute_1333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1339: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_710, [0, 2, 1]);  view_710 = None
        permute_1340: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_711, [0, 2, 1]);  view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1346: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_704, [0, 2, 1]);  view_704 = None
        permute_1347: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_705, [0, 2, 1]);  view_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1353: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_700, [0, 2, 1]);  view_700 = None
        permute_1354: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_701, [0, 2, 1]);  view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1364: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_684, [0, 2, 1]);  view_684 = None
        squeeze_dim_334: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1364, 0);  permute_1364 = None
        permute_1365: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_693, [0, 2, 1]);  view_693 = None
        squeeze_dim_333: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1365, 0);  permute_1365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1372: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_689, [0, 2, 1]);  view_689 = None
        squeeze_dim_329: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1372, 0);  permute_1372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1379: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_685, [0, 2, 1]);  view_685 = None
        squeeze_dim_325: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1379, 0);  permute_1379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_39: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_35, 1024);  rsqrt_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1384: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_758, [1, 0]);  permute_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1388: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_757, [1, 0]);  permute_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_40: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_34, 1024);  rsqrt_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1393: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_676, [0, 2, 1]);  view_676 = None
        squeeze_dim_322: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1393, 0);  permute_1393 = None
        permute_1394: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_677, [0, 2, 1]);  view_677 = None
        squeeze_dim_321: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1394, 0);  permute_1394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1400: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_672, [0, 2, 1]);  view_672 = None
        permute_1401: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_673, [0, 2, 1]);  view_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1407: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_666, [0, 2, 1]);  view_666 = None
        permute_1408: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_667, [0, 2, 1]);  view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1414: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_662, [0, 2, 1]);  view_662 = None
        permute_1415: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_663, [0, 2, 1]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1425: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_646, [0, 2, 1]);  view_646 = None
        squeeze_dim_316: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1425, 0);  permute_1425 = None
        permute_1426: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_655, [0, 2, 1]);  view_655 = None
        squeeze_dim_315: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1426, 0);  permute_1426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1433: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_651, [0, 2, 1]);  view_651 = None
        squeeze_dim_311: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1433, 0);  permute_1433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1440: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_647, [0, 2, 1]);  view_647 = None
        squeeze_dim_307: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1440, 0);  permute_1440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_41: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_33, 1024);  rsqrt_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1445: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_716, [1, 0]);  permute_716 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1449: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_715, [1, 0]);  permute_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_42: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_32, 1024);  rsqrt_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1454: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_638, [0, 2, 1]);  view_638 = None
        squeeze_dim_304: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1454, 0);  permute_1454 = None
        permute_1455: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_639, [0, 2, 1]);  view_639 = None
        squeeze_dim_303: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1455, 0);  permute_1455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1461: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_634, [0, 2, 1]);  view_634 = None
        permute_1462: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_635, [0, 2, 1]);  view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1468: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_628, [0, 2, 1]);  view_628 = None
        permute_1469: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_629, [0, 2, 1]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1475: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_624, [0, 2, 1]);  view_624 = None
        permute_1476: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_625, [0, 2, 1]);  view_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1486: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None
        squeeze_dim_298: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1486, 0);  permute_1486 = None
        permute_1487: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_617, [0, 2, 1]);  view_617 = None
        squeeze_dim_297: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1487, 0);  permute_1487 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1494: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_613, [0, 2, 1]);  view_613 = None
        squeeze_dim_293: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1494, 0);  permute_1494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1501: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_609, [0, 2, 1]);  view_609 = None
        squeeze_dim_289: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1501, 0);  permute_1501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_43: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_31, 1024);  rsqrt_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1506: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_674, [1, 0]);  permute_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1510: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_673, [1, 0]);  permute_673 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_44: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_30, 1024);  rsqrt_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1515: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_600, [0, 2, 1]);  view_600 = None
        squeeze_dim_286: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1515, 0);  permute_1515 = None
        permute_1516: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_601, [0, 2, 1]);  view_601 = None
        squeeze_dim_285: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1516, 0);  permute_1516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1522: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_596, [0, 2, 1]);  view_596 = None
        permute_1523: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_597, [0, 2, 1]);  view_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1529: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_590, [0, 2, 1]);  view_590 = None
        permute_1530: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_591, [0, 2, 1]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1536: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_586, [0, 2, 1]);  view_586 = None
        permute_1537: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_587, [0, 2, 1]);  view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1547: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None
        squeeze_dim_280: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1547, 0);  permute_1547 = None
        permute_1548: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_579, [0, 2, 1]);  view_579 = None
        squeeze_dim_279: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1548, 0);  permute_1548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1555: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_575, [0, 2, 1]);  view_575 = None
        squeeze_dim_275: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1555, 0);  permute_1555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1562: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
        squeeze_dim_271: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1562, 0);  permute_1562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_45: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_29, 1024);  rsqrt_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1567: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_632, [1, 0]);  permute_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1571: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_631, [1, 0]);  permute_631 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_46: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_28, 1024);  rsqrt_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1576: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_562, [0, 2, 1]);  view_562 = None
        squeeze_dim_268: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1576, 0);  permute_1576 = None
        permute_1577: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_563, [0, 2, 1]);  view_563 = None
        squeeze_dim_267: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1577, 0);  permute_1577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1583: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_558, [0, 2, 1]);  view_558 = None
        permute_1584: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_559, [0, 2, 1]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1590: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_552, [0, 2, 1]);  view_552 = None
        permute_1591: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_553, [0, 2, 1]);  view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1597: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_548, [0, 2, 1]);  view_548 = None
        permute_1598: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_549, [0, 2, 1]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1608: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_532, [0, 2, 1]);  view_532 = None
        squeeze_dim_262: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1608, 0);  permute_1608 = None
        permute_1609: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_541, [0, 2, 1]);  view_541 = None
        squeeze_dim_261: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1609, 0);  permute_1609 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1616: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_537, [0, 2, 1]);  view_537 = None
        squeeze_dim_257: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1616, 0);  permute_1616 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1623: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
        squeeze_dim_253: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1623, 0);  permute_1623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_47: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_27, 1024);  rsqrt_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1628: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_590, [1, 0]);  permute_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1632: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_589, [1, 0]);  permute_589 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_48: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_26, 1024);  rsqrt_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1637: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_524, [0, 2, 1]);  view_524 = None
        squeeze_dim_250: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1637, 0);  permute_1637 = None
        permute_1638: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_525, [0, 2, 1]);  view_525 = None
        squeeze_dim_249: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1638, 0);  permute_1638 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1644: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_520, [0, 2, 1]);  view_520 = None
        permute_1645: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_521, [0, 2, 1]);  view_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1651: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_514, [0, 2, 1]);  view_514 = None
        permute_1652: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_515, [0, 2, 1]);  view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1658: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_510, [0, 2, 1]);  view_510 = None
        permute_1659: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_511, [0, 2, 1]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1669: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        squeeze_dim_244: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1669, 0);  permute_1669 = None
        permute_1670: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_503, [0, 2, 1]);  view_503 = None
        squeeze_dim_243: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1670, 0);  permute_1670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1677: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_499, [0, 2, 1]);  view_499 = None
        squeeze_dim_239: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1677, 0);  permute_1677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1684: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_495, [0, 2, 1]);  view_495 = None
        squeeze_dim_235: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1684, 0);  permute_1684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_49: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_25, 1024);  rsqrt_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1689: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_548, [1, 0]);  permute_548 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1693: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_547, [1, 0]);  permute_547 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_50: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_24, 1024);  rsqrt_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1698: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_486, [0, 2, 1]);  view_486 = None
        squeeze_dim_232: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1698, 0);  permute_1698 = None
        permute_1699: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_487, [0, 2, 1]);  view_487 = None
        squeeze_dim_231: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1699, 0);  permute_1699 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1705: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_482, [0, 2, 1]);  view_482 = None
        permute_1706: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_483, [0, 2, 1]);  view_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1712: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_476, [0, 2, 1]);  view_476 = None
        permute_1713: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_477, [0, 2, 1]);  view_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1719: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_472, [0, 2, 1]);  view_472 = None
        permute_1720: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_473, [0, 2, 1]);  view_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1730: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        squeeze_dim_226: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1730, 0);  permute_1730 = None
        permute_1731: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_465, [0, 2, 1]);  view_465 = None
        squeeze_dim_225: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1731, 0);  permute_1731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1738: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_461, [0, 2, 1]);  view_461 = None
        squeeze_dim_221: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1738, 0);  permute_1738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1745: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_457, [0, 2, 1]);  view_457 = None
        squeeze_dim_217: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1745, 0);  permute_1745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_51: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_23, 1024);  rsqrt_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1750: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_506, [1, 0]);  permute_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1754: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_505, [1, 0]);  permute_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_52: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_22, 1024);  rsqrt_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1759: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_448, [0, 2, 1]);  view_448 = None
        squeeze_dim_214: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1759, 0);  permute_1759 = None
        permute_1760: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_449, [0, 2, 1]);  view_449 = None
        squeeze_dim_213: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1760, 0);  permute_1760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1766: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_444, [0, 2, 1]);  view_444 = None
        permute_1767: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_445, [0, 2, 1]);  view_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1773: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_438, [0, 2, 1]);  view_438 = None
        permute_1774: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_439, [0, 2, 1]);  view_439 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1780: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_434, [0, 2, 1]);  view_434 = None
        permute_1781: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_435, [0, 2, 1]);  view_435 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1791: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_418, [0, 2, 1]);  view_418 = None
        squeeze_dim_208: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1791, 0);  permute_1791 = None
        permute_1792: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_427, [0, 2, 1]);  view_427 = None
        squeeze_dim_207: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1792, 0);  permute_1792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1799: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
        squeeze_dim_203: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1799, 0);  permute_1799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1806: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        squeeze_dim_199: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1806, 0);  permute_1806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_53: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_21, 1024);  rsqrt_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1811: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_464, [1, 0]);  permute_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1815: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_54: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_20, 1024);  rsqrt_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1820: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_410, [0, 2, 1]);  view_410 = None
        squeeze_dim_196: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1820, 0);  permute_1820 = None
        permute_1821: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_411, [0, 2, 1]);  view_411 = None
        squeeze_dim_195: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1821, 0);  permute_1821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1827: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_406, [0, 2, 1]);  view_406 = None
        permute_1828: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_407, [0, 2, 1]);  view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1834: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_400, [0, 2, 1]);  view_400 = None
        permute_1835: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_401, [0, 2, 1]);  view_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1841: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_396, [0, 2, 1]);  view_396 = None
        permute_1842: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_397, [0, 2, 1]);  view_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1852: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_380, [0, 2, 1]);  view_380 = None
        squeeze_dim_190: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1852, 0);  permute_1852 = None
        permute_1853: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_389, [0, 2, 1]);  view_389 = None
        squeeze_dim_189: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1853, 0);  permute_1853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1860: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
        squeeze_dim_185: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1860, 0);  permute_1860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1867: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_381, [0, 2, 1]);  view_381 = None
        squeeze_dim_181: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1867, 0);  permute_1867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_55: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_19, 1024);  rsqrt_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1872: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1876: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_56: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_18, 1024);  rsqrt_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1881: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_372, [0, 2, 1]);  view_372 = None
        squeeze_dim_178: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1881, 0);  permute_1881 = None
        permute_1882: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_373, [0, 2, 1]);  view_373 = None
        squeeze_dim_177: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1882, 0);  permute_1882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1888: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_368, [0, 2, 1]);  view_368 = None
        permute_1889: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_369, [0, 2, 1]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1895: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_362, [0, 2, 1]);  view_362 = None
        permute_1896: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_363, [0, 2, 1]);  view_363 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1902: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_358, [0, 2, 1]);  view_358 = None
        permute_1903: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_359, [0, 2, 1]);  view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1913: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_342, [0, 2, 1]);  view_342 = None
        squeeze_dim_172: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1913, 0);  permute_1913 = None
        permute_1914: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_351, [0, 2, 1]);  view_351 = None
        squeeze_dim_171: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1914, 0);  permute_1914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1921: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_347, [0, 2, 1]);  view_347 = None
        squeeze_dim_167: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1921, 0);  permute_1921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1928: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_343, [0, 2, 1]);  view_343 = None
        squeeze_dim_163: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1928, 0);  permute_1928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_57: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_17, 1024);  rsqrt_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1933: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1937: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_58: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_16, 1024);  rsqrt_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_1942: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_334, [0, 2, 1]);  view_334 = None
        squeeze_dim_160: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1942, 0);  permute_1942 = None
        permute_1943: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_335, [0, 2, 1]);  view_335 = None
        squeeze_dim_159: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1943, 0);  permute_1943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_1949: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_330, [0, 2, 1]);  view_330 = None
        permute_1950: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_331, [0, 2, 1]);  view_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_1956: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_324, [0, 2, 1]);  view_324 = None
        permute_1957: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_325, [0, 2, 1]);  view_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_1963: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_320, [0, 2, 1]);  view_320 = None
        permute_1964: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_321, [0, 2, 1]);  view_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_1974: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_304, [0, 2, 1]);  view_304 = None
        squeeze_dim_154: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1974, 0);  permute_1974 = None
        permute_1975: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_313, [0, 2, 1]);  view_313 = None
        squeeze_dim_153: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1975, 0);  permute_1975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_1982: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        squeeze_dim_149: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1982, 0);  permute_1982 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_1989: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_305, [0, 2, 1]);  view_305 = None
        squeeze_dim_145: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_1989, 0);  permute_1989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_59: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_15, 1024);  rsqrt_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_1994: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_1998: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_60: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_14, 1024);  rsqrt_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2003: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_296, [0, 2, 1]);  view_296 = None
        squeeze_dim_142: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2003, 0);  permute_2003 = None
        permute_2004: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_297, [0, 2, 1]);  view_297 = None
        squeeze_dim_141: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2004, 0);  permute_2004 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2010: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_292, [0, 2, 1]);  view_292 = None
        permute_2011: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_293, [0, 2, 1]);  view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2017: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_286, [0, 2, 1]);  view_286 = None
        permute_2018: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_287, [0, 2, 1]);  view_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2024: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_282, [0, 2, 1]);  view_282 = None
        permute_2025: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_283, [0, 2, 1]);  view_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2035: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_266, [0, 2, 1]);  view_266 = None
        squeeze_dim_136: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2035, 0);  permute_2035 = None
        permute_2036: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        squeeze_dim_135: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2036, 0);  permute_2036 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2043: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
        squeeze_dim_131: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2043, 0);  permute_2043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2050: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_267, [0, 2, 1]);  view_267 = None
        squeeze_dim_127: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2050, 0);  permute_2050 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_61: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_13, 1024);  rsqrt_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2055: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2059: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_62: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_12, 1024);  rsqrt_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2064: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_258, [0, 2, 1]);  view_258 = None
        squeeze_dim_124: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2064, 0);  permute_2064 = None
        permute_2065: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_259, [0, 2, 1]);  view_259 = None
        squeeze_dim_123: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2065, 0);  permute_2065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2071: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_254, [0, 2, 1]);  view_254 = None
        permute_2072: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_255, [0, 2, 1]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2078: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_248, [0, 2, 1]);  view_248 = None
        permute_2079: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_249, [0, 2, 1]);  view_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2085: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_244, [0, 2, 1]);  view_244 = None
        permute_2086: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_245, [0, 2, 1]);  view_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2096: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_228, [0, 2, 1]);  view_228 = None
        squeeze_dim_118: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2096, 0);  permute_2096 = None
        permute_2097: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        squeeze_dim_117: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2097, 0);  permute_2097 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2104: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_233, [0, 2, 1]);  view_233 = None
        squeeze_dim_113: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2104, 0);  permute_2104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2111: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_229, [0, 2, 1]);  view_229 = None
        squeeze_dim_109: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2111, 0);  permute_2111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_63: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_11, 1024);  rsqrt_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2116: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2120: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_64: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_10, 1024);  rsqrt_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2125: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_220, [0, 2, 1]);  view_220 = None
        squeeze_dim_106: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2125, 0);  permute_2125 = None
        permute_2126: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_221, [0, 2, 1]);  view_221 = None
        squeeze_dim_105: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2126, 0);  permute_2126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2132: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_216, [0, 2, 1]);  view_216 = None
        permute_2133: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_217, [0, 2, 1]);  view_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2139: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_210, [0, 2, 1]);  view_210 = None
        permute_2140: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_211, [0, 2, 1]);  view_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2146: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_206, [0, 2, 1]);  view_206 = None
        permute_2147: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_207, [0, 2, 1]);  view_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2157: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_190, [0, 2, 1]);  view_190 = None
        squeeze_dim_100: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2157, 0);  permute_2157 = None
        permute_2158: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_199, [0, 2, 1]);  view_199 = None
        squeeze_dim_99: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2158, 0);  permute_2158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2165: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_195, [0, 2, 1]);  view_195 = None
        squeeze_dim_95: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2165, 0);  permute_2165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2172: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_191, [0, 2, 1]);  view_191 = None
        squeeze_dim_91: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2172, 0);  permute_2172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_65: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_9, 1024);  rsqrt_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2177: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2181: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_66: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_8, 1024);  rsqrt_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2186: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_182, [0, 2, 1]);  view_182 = None
        squeeze_dim_88: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2186, 0);  permute_2186 = None
        permute_2187: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_183, [0, 2, 1]);  view_183 = None
        squeeze_dim_87: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2187, 0);  permute_2187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2193: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_178, [0, 2, 1]);  view_178 = None
        permute_2194: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_179, [0, 2, 1]);  view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2200: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_172, [0, 2, 1]);  view_172 = None
        permute_2201: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_173, [0, 2, 1]);  view_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2207: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_168, [0, 2, 1]);  view_168 = None
        permute_2208: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_169, [0, 2, 1]);  view_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2218: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_152, [0, 2, 1]);  view_152 = None
        squeeze_dim_82: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2218, 0);  permute_2218 = None
        permute_2219: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        squeeze_dim_81: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2219, 0);  permute_2219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2226: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_157, [0, 2, 1]);  view_157 = None
        squeeze_dim_77: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2226, 0);  permute_2226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2233: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_153, [0, 2, 1]);  view_153 = None
        squeeze_dim_73: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2233, 0);  permute_2233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_67: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_7, 1024);  rsqrt_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2238: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2242: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_68: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_6, 1024);  rsqrt_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2247: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_144, [0, 2, 1]);  view_144 = None
        squeeze_dim_70: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2247, 0);  permute_2247 = None
        permute_2248: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_145, [0, 2, 1]);  view_145 = None
        squeeze_dim_69: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2248, 0);  permute_2248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2254: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_140, [0, 2, 1]);  view_140 = None
        permute_2255: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_141, [0, 2, 1]);  view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2261: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_134, [0, 2, 1]);  view_134 = None
        permute_2262: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_135, [0, 2, 1]);  view_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2268: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_130, [0, 2, 1]);  view_130 = None
        permute_2269: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_131, [0, 2, 1]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2279: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_114, [0, 2, 1]);  view_114 = None
        squeeze_dim_64: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2279, 0);  permute_2279 = None
        permute_2280: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        squeeze_dim_63: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2280, 0);  permute_2280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2287: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_119, [0, 2, 1]);  view_119 = None
        squeeze_dim_59: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2287, 0);  permute_2287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2294: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_115, [0, 2, 1]);  view_115 = None
        squeeze_dim_55: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2294, 0);  permute_2294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_69: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_5, 1024);  rsqrt_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2299: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2303: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_70: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_4, 1024);  rsqrt_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2308: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_106, [0, 2, 1]);  view_106 = None
        squeeze_dim_52: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2308, 0);  permute_2308 = None
        permute_2309: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_107, [0, 2, 1]);  view_107 = None
        squeeze_dim_51: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2309, 0);  permute_2309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2315: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_102, [0, 2, 1]);  view_102 = None
        permute_2316: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_103, [0, 2, 1]);  view_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2322: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_96, [0, 2, 1]);  view_96 = None
        permute_2323: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_97, [0, 2, 1]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2329: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_92, [0, 2, 1]);  view_92 = None
        permute_2330: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_93, [0, 2, 1]);  view_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2340: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_76, [0, 2, 1]);  view_76 = None
        squeeze_dim_46: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2340, 0);  permute_2340 = None
        permute_2341: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_85, [0, 2, 1]);  view_85 = None
        squeeze_dim_45: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2341, 0);  permute_2341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2348: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_81, [0, 2, 1]);  view_81 = None
        squeeze_dim_41: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2348, 0);  permute_2348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2355: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_77, [0, 2, 1]);  view_77 = None
        squeeze_dim_37: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2355, 0);  permute_2355 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_71: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_3, 1024);  rsqrt_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2360: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2364: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_72: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_2, 1024);  rsqrt_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2369: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_68, [0, 2, 1]);  view_68 = None
        squeeze_dim_34: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2369, 0);  permute_2369 = None
        permute_2370: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_69, [0, 2, 1]);  view_69 = None
        squeeze_dim_33: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2370, 0);  permute_2370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2376: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_64, [0, 2, 1]);  view_64 = None
        permute_2377: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_65, [0, 2, 1]);  view_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2383: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_58, [0, 2, 1]);  view_58 = None
        permute_2384: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_59, [0, 2, 1]);  view_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2390: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_54, [0, 2, 1]);  view_54 = None
        permute_2391: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_55, [0, 2, 1]);  view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2401: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_38, [0, 2, 1]);  view_38 = None
        squeeze_dim_28: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2401, 0);  permute_2401 = None
        permute_2402: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_47, [0, 2, 1]);  view_47 = None
        squeeze_dim_27: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2402, 0);  permute_2402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2409: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_43, [0, 2, 1]);  view_43 = None
        squeeze_dim_23: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2409, 0);  permute_2409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2416: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_39, [0, 2, 1]);  view_39 = None
        squeeze_dim_19: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2416, 0);  permute_2416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:304 in forward, code: output = self.layer_norm(output + inp)
        div_73: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt_1, 1024);  rsqrt_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:302 in forward, code: output = self.layer_2(output)
        permute_2421: "bf16[1024, 4096][4096, 1]cuda:0" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:299 in forward, code: output = self.layer_1(output)
        permute_2425: "bf16[4096, 1024][1024, 1]cuda:0" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:150 in post_attention, code: output = self.layer_norm(attn_out)
        div_74: "f32[512, 16, 1][16, 1, 1]cuda:0" = torch.ops.aten.div.Tensor(rsqrt, 1024);  rsqrt = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:145 in post_attention, code: attn_out = torch.einsum("ibnd,hnd->ibh", attn_vec, self.o)
        permute_2430: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_30, [0, 2, 1]);  view_30 = None
        squeeze_dim_16: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2430, 0);  permute_2430 = None
        permute_2431: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_31, [0, 2, 1]);  view_31 = None
        squeeze_dim_15: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2431, 0);  permute_2431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:135 in rel_attn_core, code: attn_vec = torch.einsum("bnij,jbnd->ibnd", attn_prob, v_head_h)
        permute_2437: "bf16[256, 512, 512][262144, 1, 512]cuda:0" = torch.ops.aten.permute.default(view_26, [0, 2, 1]);  view_26 = None
        permute_2438: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_27, [0, 2, 1]);  view_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:111 in rel_attn_core, code: bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
        permute_2444: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_20, [0, 2, 1]);  view_20 = None
        permute_2445: "bf16[256, 1024, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_21, [0, 2, 1]);  view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:108 in rel_attn_core, code: ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
        permute_2451: "bf16[256, 64, 512][64, 1, 16384]cuda:0" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_2452: "bf16[256, 512, 64][64, 16384, 1]cuda:0" = torch.ops.aten.permute.default(view_17, [0, 2, 1]);  view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:255 in forward, code: v_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.v)
        permute_2462: "bf16[1, 1024, 8192][8388608, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view, [0, 2, 1]);  view = None
        squeeze_dim_10: "bf16[1024, 8192][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2462, 0);  permute_2462 = None
        permute_2463: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_9, [0, 2, 1]);  view_9 = None
        squeeze_dim_9: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2463, 0);  permute_2463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:254 in forward, code: k_head_h = torch.einsum("ibh,hnd->ibnd", cat, self.k)
        permute_2470: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_5, [0, 2, 1]);  view_5 = None
        squeeze_dim_5: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2470, 0);  permute_2470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/xlnet/modeling_xlnet.py:253 in forward, code: q_head_h = torch.einsum("ibh,hnd->ibnd", h, self.q)
        permute_2477: "bf16[1, 1024, 1024][1048576, 1, 1024]cuda:0" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        squeeze_dim_1: "bf16[1024, 1024][1, 1024]cuda:0" = torch.ops.aten.squeeze.dim(permute_2477, 0);  permute_2477 = None
        return (div_25, view_913, primals_10, primals_16, primals_25, primals_31, primals_40, primals_46, primals_55, primals_61, primals_70, primals_76, primals_85, primals_91, primals_100, primals_106, primals_115, primals_121, primals_130, primals_136, primals_145, primals_151, primals_160, primals_166, primals_175, primals_181, primals_190, primals_196, primals_205, primals_211, primals_220, primals_226, primals_235, primals_241, primals_250, primals_256, primals_265, primals_271, primals_280, primals_286, primals_295, primals_301, primals_310, primals_316, primals_325, primals_331, primals_340, primals_346, primals_355, primals_361, primals_364, clone, gt, iota_2, add_3, amax_default_46, amax_default_47, logical_not_default_47, sum_1, gt_2, gt_3, mul_11, view_34, addmm, gt_4, view_36, gt_5, mul_20, add_14, amax_default_44, amax_default_45, logical_not_default_45, sum_2, gt_6, gt_7, mul_27, view_72, addmm_2, gt_8, view_74, gt_9, mul_36, add_25, amax_default_42, amax_default_43, logical_not_default_43, sum_3, gt_10, gt_11, mul_43, view_110, addmm_4, gt_12, view_112, gt_13, mul_52, add_36, amax_default_40, amax_default_41, logical_not_default_41, sum_4, gt_14, gt_15, mul_59, view_148, addmm_6, gt_16, view_150, gt_17, mul_68, add_47, amax_default_38, amax_default_39, logical_not_default_39, sum_5, gt_18, gt_19, mul_75, view_186, addmm_8, gt_20, view_188, gt_21, mul_84, add_58, amax_default_36, amax_default_37, logical_not_default_37, sum_6, gt_22, gt_23, mul_91, view_224, addmm_10, gt_24, view_226, gt_25, mul_100, add_69, amax_default_34, amax_default_35, logical_not_default_35, sum_7, gt_26, gt_27, mul_107, view_262, addmm_12, gt_28, view_264, gt_29, mul_116, add_80, amax_default_32, amax_default_33, logical_not_default_33, sum_8, gt_30, gt_31, mul_123, view_300, addmm_14, gt_32, view_302, gt_33, mul_132, add_91, amax_default_30, amax_default_31, logical_not_default_31, sum_9, gt_34, gt_35, mul_139, view_338, addmm_16, gt_36, view_340, gt_37, mul_148, add_102, amax_default_28, amax_default_29, logical_not_default_29, sum_10, gt_38, gt_39, mul_155, view_376, addmm_18, gt_40, view_378, gt_41, mul_164, add_113, amax_default_26, amax_default_27, logical_not_default_27, sum_11, gt_42, gt_43, mul_171, view_414, addmm_20, gt_44, view_416, gt_45, mul_180, add_124, amax_default_24, amax_default_25, logical_not_default_25, sum_12, gt_46, gt_47, mul_187, view_452, addmm_22, gt_48, view_454, gt_49, mul_196, add_135, amax_default_22, amax_default_23, logical_not_default_23, sum_13, gt_50, gt_51, mul_203, view_490, addmm_24, gt_52, view_492, gt_53, mul_212, add_146, amax_default_20, amax_default_21, logical_not_default_21, sum_14, gt_54, gt_55, mul_219, view_528, addmm_26, gt_56, view_530, gt_57, mul_228, add_157, amax_default_18, amax_default_19, logical_not_default_19, sum_15, gt_58, gt_59, mul_235, view_566, addmm_28, gt_60, view_568, gt_61, mul_244, add_168, amax_default_16, amax_default_17, logical_not_default_17, sum_16, gt_62, gt_63, mul_251, view_604, addmm_30, gt_64, view_606, gt_65, mul_260, add_179, amax_default_14, amax_default_15, logical_not_default_15, sum_17, gt_66, gt_67, mul_267, view_642, addmm_32, gt_68, view_644, gt_69, mul_276, add_190, amax_default_12, amax_default_13, logical_not_default_13, sum_18, gt_70, gt_71, mul_283, view_680, addmm_34, gt_72, view_682, gt_73, mul_292, add_201, amax_default_10, amax_default_11, logical_not_default_11, sum_19, gt_74, gt_75, mul_299, view_718, addmm_36, gt_76, view_720, gt_77, mul_308, add_212, amax_default_8, amax_default_9, logical_not_default_9, sum_20, gt_78, gt_79, mul_315, view_756, addmm_38, gt_80, view_758, gt_81, mul_324, add_223, amax_default_6, amax_default_7, logical_not_default_7, sum_21, gt_82, gt_83, mul_331, view_794, addmm_40, gt_84, view_796, gt_85, mul_340, add_234, amax_default_4, amax_default_5, logical_not_default_5, sum_22, gt_86, gt_87, mul_347, view_832, addmm_42, gt_88, view_834, gt_89, mul_356, add_245, amax_default_2, amax_default_3, logical_not_default_3, sum_23, gt_90, gt_91, mul_363, view_870, addmm_44, gt_92, view_872, gt_93, mul_372, add_256, amax_default, amax_default_1, logical_not_default_1, sum_24, gt_94, gt_95, mul_379, view_908, addmm_46, gt_96, view_910, gt_97, mul_388, gt_98, view_912, view_913, amax_24, log, convert_element_type_1045, permute_1013, div_27, permute_1018, permute_1022, div_28, squeeze_dim_430, squeeze_dim_429, permute_1034, permute_1035, permute_1041, permute_1042, permute_1048, permute_1049, squeeze_dim_426, squeeze_dim_424, squeeze_dim_423, squeeze_dim_419, squeeze_dim_415, div_29, permute_1079, permute_1083, div_30, squeeze_dim_412, squeeze_dim_411, permute_1095, permute_1096, permute_1102, permute_1103, permute_1109, permute_1110, squeeze_dim_406, squeeze_dim_405, squeeze_dim_401, squeeze_dim_397, div_31, permute_1140, permute_1144, div_32, squeeze_dim_394, squeeze_dim_393, permute_1156, permute_1157, permute_1163, permute_1164, permute_1170, permute_1171, squeeze_dim_388, squeeze_dim_387, squeeze_dim_383, squeeze_dim_379, div_33, permute_1201, permute_1205, div_34, squeeze_dim_376, squeeze_dim_375, permute_1217, permute_1218, permute_1224, permute_1225, permute_1231, permute_1232, squeeze_dim_370, squeeze_dim_369, squeeze_dim_365, squeeze_dim_361, div_35, permute_1262, permute_1266, div_36, squeeze_dim_358, squeeze_dim_357, permute_1278, permute_1279, permute_1285, permute_1286, permute_1292, permute_1293, squeeze_dim_352, squeeze_dim_351, squeeze_dim_347, squeeze_dim_343, div_37, permute_1323, permute_1327, div_38, squeeze_dim_340, squeeze_dim_339, permute_1339, permute_1340, permute_1346, permute_1347, permute_1353, permute_1354, squeeze_dim_334, squeeze_dim_333, squeeze_dim_329, squeeze_dim_325, div_39, permute_1384, permute_1388, div_40, squeeze_dim_322, squeeze_dim_321, permute_1400, permute_1401, permute_1407, permute_1408, permute_1414, permute_1415, squeeze_dim_316, squeeze_dim_315, squeeze_dim_311, squeeze_dim_307, div_41, permute_1445, permute_1449, div_42, squeeze_dim_304, squeeze_dim_303, permute_1461, permute_1462, permute_1468, permute_1469, permute_1475, permute_1476, squeeze_dim_298, squeeze_dim_297, squeeze_dim_293, squeeze_dim_289, div_43, permute_1506, permute_1510, div_44, squeeze_dim_286, squeeze_dim_285, permute_1522, permute_1523, permute_1529, permute_1530, permute_1536, permute_1537, squeeze_dim_280, squeeze_dim_279, squeeze_dim_275, squeeze_dim_271, div_45, permute_1567, permute_1571, div_46, squeeze_dim_268, squeeze_dim_267, permute_1583, permute_1584, permute_1590, permute_1591, permute_1597, permute_1598, squeeze_dim_262, squeeze_dim_261, squeeze_dim_257, squeeze_dim_253, div_47, permute_1628, permute_1632, div_48, squeeze_dim_250, squeeze_dim_249, permute_1644, permute_1645, permute_1651, permute_1652, permute_1658, permute_1659, squeeze_dim_244, squeeze_dim_243, squeeze_dim_239, squeeze_dim_235, div_49, permute_1689, permute_1693, div_50, squeeze_dim_232, squeeze_dim_231, permute_1705, permute_1706, permute_1712, permute_1713, permute_1719, permute_1720, squeeze_dim_226, squeeze_dim_225, squeeze_dim_221, squeeze_dim_217, div_51, permute_1750, permute_1754, div_52, squeeze_dim_214, squeeze_dim_213, permute_1766, permute_1767, permute_1773, permute_1774, permute_1780, permute_1781, squeeze_dim_208, squeeze_dim_207, squeeze_dim_203, squeeze_dim_199, div_53, permute_1811, permute_1815, div_54, squeeze_dim_196, squeeze_dim_195, permute_1827, permute_1828, permute_1834, permute_1835, permute_1841, permute_1842, squeeze_dim_190, squeeze_dim_189, squeeze_dim_185, squeeze_dim_181, div_55, permute_1872, permute_1876, div_56, squeeze_dim_178, squeeze_dim_177, permute_1888, permute_1889, permute_1895, permute_1896, permute_1902, permute_1903, squeeze_dim_172, squeeze_dim_171, squeeze_dim_167, squeeze_dim_163, div_57, permute_1933, permute_1937, div_58, squeeze_dim_160, squeeze_dim_159, permute_1949, permute_1950, permute_1956, permute_1957, permute_1963, permute_1964, squeeze_dim_154, squeeze_dim_153, squeeze_dim_149, squeeze_dim_145, div_59, permute_1994, permute_1998, div_60, squeeze_dim_142, squeeze_dim_141, permute_2010, permute_2011, permute_2017, permute_2018, permute_2024, permute_2025, squeeze_dim_136, squeeze_dim_135, squeeze_dim_131, squeeze_dim_127, div_61, permute_2055, permute_2059, div_62, squeeze_dim_124, squeeze_dim_123, permute_2071, permute_2072, permute_2078, permute_2079, permute_2085, permute_2086, squeeze_dim_118, squeeze_dim_117, squeeze_dim_113, squeeze_dim_109, div_63, permute_2116, permute_2120, div_64, squeeze_dim_106, squeeze_dim_105, permute_2132, permute_2133, permute_2139, permute_2140, permute_2146, permute_2147, squeeze_dim_100, squeeze_dim_99, squeeze_dim_95, squeeze_dim_91, div_65, permute_2177, permute_2181, div_66, squeeze_dim_88, squeeze_dim_87, permute_2193, permute_2194, permute_2200, permute_2201, permute_2207, permute_2208, squeeze_dim_82, squeeze_dim_81, squeeze_dim_77, squeeze_dim_73, div_67, permute_2238, permute_2242, div_68, squeeze_dim_70, squeeze_dim_69, permute_2254, permute_2255, permute_2261, permute_2262, permute_2268, permute_2269, squeeze_dim_64, squeeze_dim_63, squeeze_dim_59, squeeze_dim_55, div_69, permute_2299, permute_2303, div_70, squeeze_dim_52, squeeze_dim_51, permute_2315, permute_2316, permute_2322, permute_2323, permute_2329, permute_2330, squeeze_dim_46, squeeze_dim_45, squeeze_dim_41, squeeze_dim_37, div_71, permute_2360, permute_2364, div_72, squeeze_dim_34, squeeze_dim_33, permute_2376, permute_2377, permute_2383, permute_2384, permute_2390, permute_2391, squeeze_dim_28, squeeze_dim_27, squeeze_dim_23, squeeze_dim_19, div_73, permute_2421, permute_2425, div_74, squeeze_dim_16, squeeze_dim_15, permute_2437, permute_2438, permute_2444, permute_2445, permute_2451, permute_2452, squeeze_dim_10, squeeze_dim_9, squeeze_dim_5, squeeze_dim_1)

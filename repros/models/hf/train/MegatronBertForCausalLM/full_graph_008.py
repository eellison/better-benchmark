class GraphModule(torch.nn.Module):
    def forward(self, primals_2: "i64[16, 512]", primals_3: "f32[29056, 1024]", primals_4: "i64[1, 512]", primals_7: "f32[1024]", primals_9: "f32[1024, 1024]", primals_11: "f32[1024, 1024]", primals_13: "f32[1024, 1024]", primals_15: "f32[1024, 1024]", primals_17: "f32[1024]", primals_19: "f32[4096, 1024]", primals_21: "f32[1024, 4096]", primals_23: "f32[1024]", primals_25: "f32[1024, 1024]", primals_27: "f32[1024, 1024]", primals_29: "f32[1024, 1024]", primals_31: "f32[1024, 1024]", primals_33: "f32[1024]", primals_35: "f32[4096, 1024]", primals_37: "f32[1024, 4096]", primals_39: "f32[1024]", primals_41: "f32[1024, 1024]", primals_43: "f32[1024, 1024]", primals_45: "f32[1024, 1024]", primals_47: "f32[1024, 1024]", primals_49: "f32[1024]", primals_51: "f32[4096, 1024]", primals_53: "f32[1024, 4096]", primals_55: "f32[1024]", primals_57: "f32[1024, 1024]", primals_59: "f32[1024, 1024]", primals_61: "f32[1024, 1024]", primals_63: "f32[1024, 1024]", primals_65: "f32[1024]", primals_67: "f32[4096, 1024]", primals_69: "f32[1024, 4096]", primals_71: "f32[1024]", primals_73: "f32[1024, 1024]", primals_75: "f32[1024, 1024]", primals_77: "f32[1024, 1024]", primals_79: "f32[1024, 1024]", primals_81: "f32[1024]", primals_83: "f32[4096, 1024]", primals_85: "f32[1024, 4096]", primals_87: "f32[1024]", primals_89: "f32[1024, 1024]", primals_91: "f32[1024, 1024]", primals_93: "f32[1024, 1024]", primals_95: "f32[1024, 1024]", primals_97: "f32[1024]", primals_99: "f32[4096, 1024]", primals_101: "f32[1024, 4096]", primals_103: "f32[1024]", primals_105: "f32[1024, 1024]", primals_107: "f32[1024, 1024]", primals_109: "f32[1024, 1024]", primals_111: "f32[1024, 1024]", primals_113: "f32[1024]", primals_115: "f32[4096, 1024]", primals_117: "f32[1024, 4096]", primals_119: "f32[1024]", primals_121: "f32[1024, 1024]", primals_123: "f32[1024, 1024]", primals_125: "f32[1024, 1024]", primals_127: "f32[1024, 1024]", primals_129: "f32[1024]", primals_131: "f32[4096, 1024]", primals_133: "f32[1024, 4096]", primals_135: "f32[1024]", primals_137: "f32[1024, 1024]", primals_139: "f32[1024, 1024]", primals_141: "f32[1024, 1024]", primals_143: "f32[1024, 1024]", primals_145: "f32[1024]", primals_147: "f32[4096, 1024]", primals_149: "f32[1024, 4096]", primals_151: "f32[1024]", primals_153: "f32[1024, 1024]", primals_155: "f32[1024, 1024]", primals_157: "f32[1024, 1024]", primals_159: "f32[1024, 1024]", primals_161: "f32[1024]", primals_163: "f32[4096, 1024]", primals_165: "f32[1024, 4096]", primals_167: "f32[1024]", primals_169: "f32[1024, 1024]", primals_171: "f32[1024, 1024]", primals_173: "f32[1024, 1024]", primals_175: "f32[1024, 1024]", primals_177: "f32[1024]", primals_179: "f32[4096, 1024]", primals_181: "f32[1024, 4096]", primals_183: "f32[1024]", primals_185: "f32[1024, 1024]", primals_187: "f32[1024, 1024]", primals_189: "f32[1024, 1024]", primals_191: "f32[1024, 1024]", primals_193: "f32[1024]", primals_195: "f32[4096, 1024]", primals_197: "f32[1024, 4096]", primals_199: "f32[1024]", primals_201: "f32[1024, 1024]", primals_203: "f32[1024, 1024]", primals_205: "f32[1024, 1024]", primals_207: "f32[1024, 1024]", primals_209: "f32[1024]", primals_211: "f32[4096, 1024]", primals_213: "f32[1024, 4096]", primals_215: "f32[1024]", primals_217: "f32[1024, 1024]", primals_219: "f32[1024, 1024]", primals_221: "f32[1024, 1024]", primals_223: "f32[1024, 1024]", primals_225: "f32[1024]", primals_227: "f32[4096, 1024]", primals_229: "f32[1024, 4096]", primals_231: "f32[1024]", primals_233: "f32[1024, 1024]", primals_235: "f32[1024, 1024]", primals_237: "f32[1024, 1024]", primals_239: "f32[1024, 1024]", primals_241: "f32[1024]", primals_243: "f32[4096, 1024]", primals_245: "f32[1024, 4096]", primals_247: "f32[1024]", primals_249: "f32[1024, 1024]", primals_251: "f32[1024, 1024]", primals_253: "f32[1024, 1024]", primals_255: "f32[1024, 1024]", primals_257: "f32[1024]", primals_259: "f32[4096, 1024]", primals_261: "f32[1024, 4096]", primals_263: "f32[1024]", primals_265: "f32[1024, 1024]", primals_267: "f32[1024, 1024]", primals_269: "f32[1024, 1024]", primals_271: "f32[1024, 1024]", primals_273: "f32[1024]", primals_275: "f32[4096, 1024]", primals_277: "f32[1024, 4096]", primals_279: "f32[1024]", primals_281: "f32[1024, 1024]", primals_283: "f32[1024, 1024]", primals_285: "f32[1024, 1024]", primals_287: "f32[1024, 1024]", primals_289: "f32[1024]", primals_291: "f32[4096, 1024]", primals_293: "f32[1024, 4096]", primals_295: "f32[1024]", primals_297: "f32[1024, 1024]", primals_299: "f32[1024, 1024]", primals_301: "f32[1024, 1024]", primals_303: "f32[1024, 1024]", primals_305: "f32[1024]", primals_307: "f32[4096, 1024]", primals_309: "f32[1024, 4096]", primals_311: "f32[1024]", primals_313: "f32[1024, 1024]", primals_315: "f32[1024, 1024]", primals_317: "f32[1024, 1024]", primals_319: "f32[1024, 1024]", primals_321: "f32[1024]", primals_323: "f32[4096, 1024]", primals_325: "f32[1024, 4096]", primals_327: "f32[1024]", primals_329: "f32[1024, 1024]", primals_331: "f32[1024, 1024]", primals_333: "f32[1024, 1024]", primals_335: "f32[1024, 1024]", primals_337: "f32[1024]", primals_339: "f32[4096, 1024]", primals_341: "f32[1024, 4096]", primals_343: "f32[1024]", primals_345: "f32[1024, 1024]", primals_347: "f32[1024, 1024]", primals_349: "f32[1024, 1024]", primals_351: "f32[1024, 1024]", primals_353: "f32[1024]", primals_355: "f32[4096, 1024]", primals_357: "f32[1024, 4096]", primals_359: "f32[1024]", primals_361: "f32[1024, 1024]", primals_363: "f32[1024, 1024]", primals_365: "f32[1024, 1024]", primals_367: "f32[1024, 1024]", primals_369: "f32[1024]", primals_371: "f32[4096, 1024]", primals_373: "f32[1024, 4096]", primals_375: "f32[1024]", primals_377: "f32[1024, 1024]", primals_379: "f32[1024, 1024]", primals_381: "f32[1024, 1024]", primals_383: "f32[1024, 1024]", primals_385: "f32[1024]", primals_387: "f32[4096, 1024]", primals_389: "f32[1024, 4096]", primals_391: "f32[1024]", primals_393: "f32[1024, 1024]", primals_395: "f32[1024]", full_default: "i64[16, 512]", gt: "b8[16, 512, 1024]", mul_3: "f32[16, 512, 1024]", view: "f32[8192, 1024]", permute_default_138: "f32[16, 16, 512, 64]", permute_default_139: "f32[16, 16, 512, 64]", permute_default_140: "f32[16, 16, 512, 64]", getitem_261: "f32[16, 16, 512, 64]", getitem_262: "f32[16, 16, 512]", getitem_263: "i64[]", getitem_264: "i64[]", view_16: "f32[8192, 1024]", gt_2: "b8[16, 512, 1024]", mul_9: "f32[16, 512, 1024]", view_18: "f32[8192, 1024]", addmm_4: "f32[8192, 4096]", view_20: "f32[8192, 4096]", gt_3: "b8[16, 512, 1024]", mul_16: "f32[16, 512, 1024]", view_22: "f32[8192, 1024]", permute_default_132: "f32[16, 16, 512, 64]", permute_default_133: "f32[16, 16, 512, 64]", permute_default_134: "f32[16, 16, 512, 64]", getitem_254: "f32[16, 16, 512, 64]", getitem_255: "f32[16, 16, 512]", getitem_256: "i64[]", getitem_257: "i64[]", view_38: "f32[8192, 1024]", gt_5: "b8[16, 512, 1024]", mul_22: "f32[16, 512, 1024]", view_40: "f32[8192, 1024]", addmm_10: "f32[8192, 4096]", view_42: "f32[8192, 4096]", gt_6: "b8[16, 512, 1024]", mul_29: "f32[16, 512, 1024]", view_44: "f32[8192, 1024]", permute_default_126: "f32[16, 16, 512, 64]", permute_default_127: "f32[16, 16, 512, 64]", permute_default_128: "f32[16, 16, 512, 64]", getitem_247: "f32[16, 16, 512, 64]", getitem_248: "f32[16, 16, 512]", getitem_249: "i64[]", getitem_250: "i64[]", view_60: "f32[8192, 1024]", gt_8: "b8[16, 512, 1024]", mul_35: "f32[16, 512, 1024]", view_62: "f32[8192, 1024]", addmm_16: "f32[8192, 4096]", view_64: "f32[8192, 4096]", gt_9: "b8[16, 512, 1024]", mul_42: "f32[16, 512, 1024]", view_66: "f32[8192, 1024]", permute_default_120: "f32[16, 16, 512, 64]", permute_default_121: "f32[16, 16, 512, 64]", permute_default_122: "f32[16, 16, 512, 64]", getitem_240: "f32[16, 16, 512, 64]", getitem_241: "f32[16, 16, 512]", getitem_242: "i64[]", getitem_243: "i64[]", view_82: "f32[8192, 1024]", gt_11: "b8[16, 512, 1024]", mul_48: "f32[16, 512, 1024]", view_84: "f32[8192, 1024]", addmm_22: "f32[8192, 4096]", view_86: "f32[8192, 4096]", gt_12: "b8[16, 512, 1024]", mul_55: "f32[16, 512, 1024]", view_88: "f32[8192, 1024]", permute_default_114: "f32[16, 16, 512, 64]", permute_default_115: "f32[16, 16, 512, 64]", permute_default_116: "f32[16, 16, 512, 64]", getitem_233: "f32[16, 16, 512, 64]", getitem_234: "f32[16, 16, 512]", getitem_235: "i64[]", getitem_236: "i64[]", view_104: "f32[8192, 1024]", gt_14: "b8[16, 512, 1024]", mul_61: "f32[16, 512, 1024]", view_106: "f32[8192, 1024]", addmm_28: "f32[8192, 4096]", view_108: "f32[8192, 4096]", gt_15: "b8[16, 512, 1024]", mul_68: "f32[16, 512, 1024]", view_110: "f32[8192, 1024]", permute_default_108: "f32[16, 16, 512, 64]", permute_default_109: "f32[16, 16, 512, 64]", permute_default_110: "f32[16, 16, 512, 64]", getitem_226: "f32[16, 16, 512, 64]", getitem_227: "f32[16, 16, 512]", getitem_228: "i64[]", getitem_229: "i64[]", view_126: "f32[8192, 1024]", gt_17: "b8[16, 512, 1024]", mul_74: "f32[16, 512, 1024]", view_128: "f32[8192, 1024]", addmm_34: "f32[8192, 4096]", view_130: "f32[8192, 4096]", gt_18: "b8[16, 512, 1024]", mul_81: "f32[16, 512, 1024]", view_132: "f32[8192, 1024]", permute_default_102: "f32[16, 16, 512, 64]", permute_default_103: "f32[16, 16, 512, 64]", permute_default_104: "f32[16, 16, 512, 64]", getitem_219: "f32[16, 16, 512, 64]", getitem_220: "f32[16, 16, 512]", getitem_221: "i64[]", getitem_222: "i64[]", view_148: "f32[8192, 1024]", gt_20: "b8[16, 512, 1024]", mul_87: "f32[16, 512, 1024]", view_150: "f32[8192, 1024]", addmm_40: "f32[8192, 4096]", view_152: "f32[8192, 4096]", gt_21: "b8[16, 512, 1024]", mul_94: "f32[16, 512, 1024]", view_154: "f32[8192, 1024]", permute_default_96: "f32[16, 16, 512, 64]", permute_default_97: "f32[16, 16, 512, 64]", permute_default_98: "f32[16, 16, 512, 64]", getitem_212: "f32[16, 16, 512, 64]", getitem_213: "f32[16, 16, 512]", getitem_214: "i64[]", getitem_215: "i64[]", view_170: "f32[8192, 1024]", gt_23: "b8[16, 512, 1024]", mul_100: "f32[16, 512, 1024]", view_172: "f32[8192, 1024]", addmm_46: "f32[8192, 4096]", view_174: "f32[8192, 4096]", gt_24: "b8[16, 512, 1024]", mul_107: "f32[16, 512, 1024]", view_176: "f32[8192, 1024]", permute_default_90: "f32[16, 16, 512, 64]", permute_default_91: "f32[16, 16, 512, 64]", permute_default_92: "f32[16, 16, 512, 64]", getitem_205: "f32[16, 16, 512, 64]", getitem_206: "f32[16, 16, 512]", getitem_207: "i64[]", getitem_208: "i64[]", view_192: "f32[8192, 1024]", gt_26: "b8[16, 512, 1024]", mul_113: "f32[16, 512, 1024]", view_194: "f32[8192, 1024]", addmm_52: "f32[8192, 4096]", view_196: "f32[8192, 4096]", gt_27: "b8[16, 512, 1024]", mul_120: "f32[16, 512, 1024]", view_198: "f32[8192, 1024]", permute_default_84: "f32[16, 16, 512, 64]", permute_default_85: "f32[16, 16, 512, 64]", permute_default_86: "f32[16, 16, 512, 64]", getitem_198: "f32[16, 16, 512, 64]", getitem_199: "f32[16, 16, 512]", getitem_200: "i64[]", getitem_201: "i64[]", view_214: "f32[8192, 1024]", gt_29: "b8[16, 512, 1024]", mul_126: "f32[16, 512, 1024]", view_216: "f32[8192, 1024]", addmm_58: "f32[8192, 4096]", view_218: "f32[8192, 4096]", gt_30: "b8[16, 512, 1024]", mul_133: "f32[16, 512, 1024]", view_220: "f32[8192, 1024]", permute_default_78: "f32[16, 16, 512, 64]", permute_default_79: "f32[16, 16, 512, 64]", permute_default_80: "f32[16, 16, 512, 64]", getitem_191: "f32[16, 16, 512, 64]", getitem_192: "f32[16, 16, 512]", getitem_193: "i64[]", getitem_194: "i64[]", view_236: "f32[8192, 1024]", gt_32: "b8[16, 512, 1024]", mul_139: "f32[16, 512, 1024]", view_238: "f32[8192, 1024]", addmm_64: "f32[8192, 4096]", view_240: "f32[8192, 4096]", gt_33: "b8[16, 512, 1024]", mul_146: "f32[16, 512, 1024]", view_242: "f32[8192, 1024]", permute_default_72: "f32[16, 16, 512, 64]", permute_default_73: "f32[16, 16, 512, 64]", permute_default_74: "f32[16, 16, 512, 64]", getitem_184: "f32[16, 16, 512, 64]", getitem_185: "f32[16, 16, 512]", getitem_186: "i64[]", getitem_187: "i64[]", view_258: "f32[8192, 1024]", gt_35: "b8[16, 512, 1024]", mul_152: "f32[16, 512, 1024]", view_260: "f32[8192, 1024]", addmm_70: "f32[8192, 4096]", view_262: "f32[8192, 4096]", gt_36: "b8[16, 512, 1024]", mul_159: "f32[16, 512, 1024]", view_264: "f32[8192, 1024]", permute_default_66: "f32[16, 16, 512, 64]", permute_default_67: "f32[16, 16, 512, 64]", permute_default_68: "f32[16, 16, 512, 64]", getitem_177: "f32[16, 16, 512, 64]", getitem_178: "f32[16, 16, 512]", getitem_179: "i64[]", getitem_180: "i64[]", view_280: "f32[8192, 1024]", gt_38: "b8[16, 512, 1024]", mul_165: "f32[16, 512, 1024]", view_282: "f32[8192, 1024]", addmm_76: "f32[8192, 4096]", view_284: "f32[8192, 4096]", gt_39: "b8[16, 512, 1024]", mul_172: "f32[16, 512, 1024]", view_286: "f32[8192, 1024]", permute_default_60: "f32[16, 16, 512, 64]", permute_default_61: "f32[16, 16, 512, 64]", permute_default_62: "f32[16, 16, 512, 64]", getitem_170: "f32[16, 16, 512, 64]", getitem_171: "f32[16, 16, 512]", getitem_172: "i64[]", getitem_173: "i64[]", view_302: "f32[8192, 1024]", gt_41: "b8[16, 512, 1024]", mul_178: "f32[16, 512, 1024]", view_304: "f32[8192, 1024]", addmm_82: "f32[8192, 4096]", view_306: "f32[8192, 4096]", gt_42: "b8[16, 512, 1024]", mul_185: "f32[16, 512, 1024]", view_308: "f32[8192, 1024]", permute_default_54: "f32[16, 16, 512, 64]", permute_default_55: "f32[16, 16, 512, 64]", permute_default_56: "f32[16, 16, 512, 64]", getitem_163: "f32[16, 16, 512, 64]", getitem_164: "f32[16, 16, 512]", getitem_165: "i64[]", getitem_166: "i64[]", view_324: "f32[8192, 1024]", gt_44: "b8[16, 512, 1024]", mul_191: "f32[16, 512, 1024]", view_326: "f32[8192, 1024]", addmm_88: "f32[8192, 4096]", view_328: "f32[8192, 4096]", gt_45: "b8[16, 512, 1024]", mul_198: "f32[16, 512, 1024]", view_330: "f32[8192, 1024]", permute_default_48: "f32[16, 16, 512, 64]", permute_default_49: "f32[16, 16, 512, 64]", permute_default_50: "f32[16, 16, 512, 64]", getitem_156: "f32[16, 16, 512, 64]", getitem_157: "f32[16, 16, 512]", getitem_158: "i64[]", getitem_159: "i64[]", view_346: "f32[8192, 1024]", gt_47: "b8[16, 512, 1024]", mul_204: "f32[16, 512, 1024]", view_348: "f32[8192, 1024]", addmm_94: "f32[8192, 4096]", view_350: "f32[8192, 4096]", gt_48: "b8[16, 512, 1024]", mul_211: "f32[16, 512, 1024]", view_352: "f32[8192, 1024]", permute_default_42: "f32[16, 16, 512, 64]", permute_default_43: "f32[16, 16, 512, 64]", permute_default_44: "f32[16, 16, 512, 64]", getitem_149: "f32[16, 16, 512, 64]", getitem_150: "f32[16, 16, 512]", getitem_151: "i64[]", getitem_152: "i64[]", view_368: "f32[8192, 1024]", gt_50: "b8[16, 512, 1024]", mul_217: "f32[16, 512, 1024]", view_370: "f32[8192, 1024]", addmm_100: "f32[8192, 4096]", view_372: "f32[8192, 4096]", gt_51: "b8[16, 512, 1024]", mul_224: "f32[16, 512, 1024]", view_374: "f32[8192, 1024]", permute_default_36: "f32[16, 16, 512, 64]", permute_default_37: "f32[16, 16, 512, 64]", permute_default_38: "f32[16, 16, 512, 64]", getitem_142: "f32[16, 16, 512, 64]", getitem_143: "f32[16, 16, 512]", getitem_144: "i64[]", getitem_145: "i64[]", view_390: "f32[8192, 1024]", gt_53: "b8[16, 512, 1024]", mul_230: "f32[16, 512, 1024]", view_392: "f32[8192, 1024]", addmm_106: "f32[8192, 4096]", view_394: "f32[8192, 4096]", gt_54: "b8[16, 512, 1024]", mul_237: "f32[16, 512, 1024]", view_396: "f32[8192, 1024]", permute_default_30: "f32[16, 16, 512, 64]", permute_default_31: "f32[16, 16, 512, 64]", permute_default_32: "f32[16, 16, 512, 64]", getitem_135: "f32[16, 16, 512, 64]", getitem_136: "f32[16, 16, 512]", getitem_137: "i64[]", getitem_138: "i64[]", view_412: "f32[8192, 1024]", gt_56: "b8[16, 512, 1024]", mul_243: "f32[16, 512, 1024]", view_414: "f32[8192, 1024]", addmm_112: "f32[8192, 4096]", view_416: "f32[8192, 4096]", gt_57: "b8[16, 512, 1024]", mul_250: "f32[16, 512, 1024]", view_418: "f32[8192, 1024]", permute_default_24: "f32[16, 16, 512, 64]", permute_default_25: "f32[16, 16, 512, 64]", permute_default_26: "f32[16, 16, 512, 64]", getitem_128: "f32[16, 16, 512, 64]", getitem_129: "f32[16, 16, 512]", getitem_130: "i64[]", getitem_131: "i64[]", view_434: "f32[8192, 1024]", gt_59: "b8[16, 512, 1024]", mul_256: "f32[16, 512, 1024]", view_436: "f32[8192, 1024]", addmm_118: "f32[8192, 4096]", view_438: "f32[8192, 4096]", gt_60: "b8[16, 512, 1024]", mul_263: "f32[16, 512, 1024]", view_440: "f32[8192, 1024]", permute_default_18: "f32[16, 16, 512, 64]", permute_default_19: "f32[16, 16, 512, 64]", permute_default_20: "f32[16, 16, 512, 64]", getitem_121: "f32[16, 16, 512, 64]", getitem_122: "f32[16, 16, 512]", getitem_123: "i64[]", getitem_124: "i64[]", view_456: "f32[8192, 1024]", gt_62: "b8[16, 512, 1024]", mul_269: "f32[16, 512, 1024]", view_458: "f32[8192, 1024]", addmm_124: "f32[8192, 4096]", view_460: "f32[8192, 4096]", gt_63: "b8[16, 512, 1024]", mul_276: "f32[16, 512, 1024]", view_462: "f32[8192, 1024]", permute_default_12: "f32[16, 16, 512, 64]", permute_default_13: "f32[16, 16, 512, 64]", permute_default_14: "f32[16, 16, 512, 64]", getitem_114: "f32[16, 16, 512, 64]", getitem_115: "f32[16, 16, 512]", getitem_116: "i64[]", getitem_117: "i64[]", view_478: "f32[8192, 1024]", gt_65: "b8[16, 512, 1024]", mul_282: "f32[16, 512, 1024]", view_480: "f32[8192, 1024]", addmm_130: "f32[8192, 4096]", view_482: "f32[8192, 4096]", gt_66: "b8[16, 512, 1024]", mul_289: "f32[16, 512, 1024]", view_484: "f32[8192, 1024]", permute_default_6: "f32[16, 16, 512, 64]", permute_default_7: "f32[16, 16, 512, 64]", permute_default_8: "f32[16, 16, 512, 64]", getitem_107: "f32[16, 16, 512, 64]", getitem_108: "f32[16, 16, 512]", getitem_109: "i64[]", getitem_110: "i64[]", view_500: "f32[8192, 1024]", gt_68: "b8[16, 512, 1024]", mul_295: "f32[16, 512, 1024]", view_502: "f32[8192, 1024]", addmm_136: "f32[8192, 4096]", view_504: "f32[8192, 4096]", gt_69: "b8[16, 512, 1024]", mul_302: "f32[16, 512, 1024]", view_506: "f32[8192, 1024]", permute_default: "f32[16, 16, 512, 64]", permute_default_1: "f32[16, 16, 512, 64]", permute_default_2: "f32[16, 16, 512, 64]", getitem_100: "f32[16, 16, 512, 64]", getitem_101: "f32[16, 16, 512]", getitem_102: "i64[]", getitem_103: "i64[]", view_522: "f32[8192, 1024]", gt_71: "b8[16, 512, 1024]", mul_308: "f32[16, 512, 1024]", view_524: "f32[8192, 1024]", addmm_142: "f32[8192, 4096]", view_526: "f32[8192, 4096]", gt_72: "b8[16, 512, 1024]", mul_315: "f32[16, 512, 1024]", view_528: "f32[8192, 1024]", addmm_144: "f32[8192, 1024]", getitem_99: "f32[16, 512, 1]", rsqrt_49: "f32[16, 512, 1]", view_530: "f32[8192, 1024]", view_531: "f32[16, 512, 29056]", constant_pad_nd: "i64[16, 513]", amax_24: "f32[8192, 1]", log: "f32[8192, 1]", convert_element_type: "f32[]", div_51: "f32[16, 512, 1]", div_52: "f32[16, 512, 1]", div_54: "f32[16, 512, 1]", div_55: "f32[16, 512, 1]", div_57: "f32[16, 512, 1]", div_58: "f32[16, 512, 1]", div_60: "f32[16, 512, 1]", div_61: "f32[16, 512, 1]", div_63: "f32[16, 512, 1]", div_64: "f32[16, 512, 1]", div_66: "f32[16, 512, 1]", div_67: "f32[16, 512, 1]", div_69: "f32[16, 512, 1]", div_70: "f32[16, 512, 1]", div_72: "f32[16, 512, 1]", div_73: "f32[16, 512, 1]", div_75: "f32[16, 512, 1]", div_76: "f32[16, 512, 1]", div_78: "f32[16, 512, 1]", div_79: "f32[16, 512, 1]", div_81: "f32[16, 512, 1]", div_82: "f32[16, 512, 1]", div_84: "f32[16, 512, 1]", div_85: "f32[16, 512, 1]", div_87: "f32[16, 512, 1]", div_88: "f32[16, 512, 1]", div_90: "f32[16, 512, 1]", div_91: "f32[16, 512, 1]", div_93: "f32[16, 512, 1]", div_94: "f32[16, 512, 1]", div_96: "f32[16, 512, 1]", div_97: "f32[16, 512, 1]", div_99: "f32[16, 512, 1]", div_100: "f32[16, 512, 1]", div_102: "f32[16, 512, 1]", div_103: "f32[16, 512, 1]", div_105: "f32[16, 512, 1]", div_106: "f32[16, 512, 1]", div_108: "f32[16, 512, 1]", div_109: "f32[16, 512, 1]", div_111: "f32[16, 512, 1]", div_112: "f32[16, 512, 1]", div_114: "f32[16, 512, 1]", div_115: "f32[16, 512, 1]", div_117: "f32[16, 512, 1]", div_118: "f32[16, 512, 1]", div_120: "f32[16, 512, 1]", div_121: "f32[16, 512, 1]", div_123: "f32[16, 512, 1]", tangents_1: "f32[]", tangents_2: "f32[16, 512, 29056]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        div_49: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type);  tangents_1 = convert_element_type = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:61 in ForCausalLMLoss, code: shift_labels = labels[..., 1:].contiguous()
        slice_1: "i64[16, 512]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 1, 1, 9223372036854775807);  constant_pad_nd = None
        clone_96: "i64[16, 512]" = torch.ops.aten.clone.default(slice_1, memory_format = torch.contiguous_format);  slice_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:65 in ForCausalLMLoss, code: shift_labels = shift_labels.view(-1)
        view_533: "i64[8192]" = torch.ops.aten.reshape.default(clone_96, [-1]);  clone_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        unsqueeze_3: "i64[8192, 1]" = torch.ops.aten.unsqueeze.default(view_533, 1);  view_533 = None
        ne_3: "b8[8192, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_3, -100)
        full_default_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "i64[8192, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_3, full_default_2);  unsqueeze_3 = full_default_2 = None

        # No stacktrace found for following nodes
        iota_default: "i64[29056]" = torch.ops.prims.iota.default(29056, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 29056]" = torch.ops.aten.reshape.default(iota_default, [1, 29056]);  iota_default = None
        expand_default: "i64[8192, 29056]" = torch.ops.aten.expand.default(where_2, [8192, 29056]);  where_2 = None
        eq_tensor: "b8[8192, 29056]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        where_self: "f32[8192, 29056]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None
        full_default_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_3: "f32[8192, 1]" = torch.ops.aten.where.self(ne_3, div_49, full_default_3);  ne_3 = div_49 = None
        mul_322: "f32[8192, 29056]" = torch.ops.aten.mul.Tensor(where_self, where_3);  where_self = where_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_532: "f32[8192, 29056]" = torch.ops.aten.reshape.default(view_531, [-1, 29056]);  view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:37 in fixed_cross_entropy, code: loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
        sub_75: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(view_532, amax_24);  view_532 = amax_24 = None
        sub_76: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(sub_75, log);  sub_75 = log = None
        exp_25: "f32[8192, 29056]" = torch.ops.aten.exp.default(sub_76);  sub_76 = None
        sum_28: "f32[8192, 1]" = torch.ops.aten.sum.dim_IntList(mul_322, [1], True)
        mul_323: "f32[8192, 29056]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_77: "f32[8192, 29056]" = torch.ops.aten.sub.Tensor(mul_322, mul_323);  mul_322 = mul_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/loss/loss_utils.py:64 in ForCausalLMLoss, code: logits = logits.view(-1, vocab_size)
        view_534: "f32[16, 512, 29056]" = torch.ops.aten.reshape.default(sub_77, [16, 512, 29056]);  sub_77 = None
        add_199: "f32[16, 512, 29056]" = torch.ops.aten.add.Tensor(tangents_2, view_534);  tangents_2 = view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:465 in forward, code: hidden_states = self.decoder(hidden_states)
        view_535: "f32[8192, 29056]" = torch.ops.aten.reshape.default(add_199, [8192, 29056]);  add_199 = None
        permute_265: "f32[1024, 29056]" = torch.ops.aten.permute.default(primals_3, [1, 0]);  primals_3 = None
        permute_266: "f32[29056, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_535, permute_266);  permute_266 = None
        permute_267: "f32[29056, 8192]" = torch.ops.aten.permute.default(view_535, [1, 0])
        mm_1: "f32[29056, 1024]" = torch.ops.aten.mm.default(permute_267, view_530);  permute_267 = view_530 = None
        sum_29: "f32[1, 29056]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[29056]" = torch.ops.aten.reshape.default(sum_29, [29056]);  sum_29 = None
        view_537: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm, [16, 512, 1024]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_325: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, primals_395);  primals_395 = None
        mul_326: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, 1024)
        sum_30: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_325, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        view_529: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(addmm_144, [16, 512, 1024]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_317: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.5)
        mul_318: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, 0.7071067811865476)
        erf_24: "f32[16, 512, 1024]" = torch.ops.aten.erf.default(mul_318);  mul_318 = None
        add_196: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_319: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_317, add_196);  mul_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:448 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        sub_74: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_319, getitem_99);  mul_319 = getitem_99 = None
        mul_320: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_49);  sub_74 = None
        mul_327: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_325, mul_320);  mul_325 = None
        sum_31: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_327, [2], True);  mul_327 = None
        mul_328: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_320, sum_31);  sum_31 = None
        sub_79: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_326, sum_30);  mul_326 = sum_30 = None
        sub_80: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_79, mul_328);  sub_79 = mul_328 = None
        div_50: "f32[16, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt_49, 1024);  rsqrt_49 = None
        mul_329: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_50, sub_80);  div_50 = sub_80 = None
        mul_330: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_537, mul_320);  mul_320 = None
        sum_32: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_330, [0, 1]);  mul_330 = None
        sum_33: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_332: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_196, 0.5);  add_196 = None
        mul_333: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, view_529)
        mul_334: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_333, -0.5);  mul_333 = None
        exp_26: "f32[16, 512, 1024]" = torch.ops.aten.exp.default(mul_334);  mul_334 = None
        mul_335: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(exp_26, 0.3989422804014327);  exp_26 = None
        mul_336: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_529, mul_335);  view_529 = mul_335 = None
        add_201: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_332, mul_336);  mul_332 = mul_336 = None
        mul_337: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_329, add_201);  mul_329 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:446 in forward, code: hidden_states = self.dense(hidden_states)
        view_538: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_337, [8192, 1024]);  mul_337 = None
        permute_264: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
        permute_270: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        mm_2: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_538, permute_270);  permute_270 = None
        permute_271: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_538, [1, 0])
        mm_3: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_271, view_528);  permute_271 = view_528 = None
        sum_34: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        view_539: "f32[1024]" = torch.ops.aten.reshape.default(sum_34, [1024]);  sum_34 = None
        view_540: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_2, [16, 512, 1024]);  mm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:392 in forward, code: hidden_states = self.ln(hidden_states)
        mul_339: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, primals_391);  primals_391 = None
        mul_340: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_339, 1024)
        sum_35: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_339, [2], True)
        mul_341: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_339, mul_315);  mul_339 = None
        sum_36: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
        mul_342: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_315, sum_36);  sum_36 = None
        sub_82: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_340, sum_35);  mul_340 = sum_35 = None
        sub_83: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_82, mul_342);  sub_82 = mul_342 = None
        mul_343: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_51, sub_83);  div_51 = sub_83 = None
        mul_344: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_540, mul_315);  mul_315 = None
        sum_37: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_344, [0, 1]);  mul_344 = None
        sum_38: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_1: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_345: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_346: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_343, mul_345);  mul_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_541: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_346, [8192, 1024]);  mul_346 = None
        permute_263: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_389, [1, 0]);  primals_389 = None
        permute_274: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_4: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_541, permute_274);  permute_274 = None
        permute_275: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_541, [1, 0])
        mm_5: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_275, view_526);  permute_275 = view_526 = None
        sum_39: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        view_542: "f32[1024]" = torch.ops.aten.reshape.default(sum_39, [1024]);  sum_39 = None
        view_543: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_4, [16, 512, 4096]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_525: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_142, [16, 512, 4096]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_311: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, 0.7071067811865476)
        erf_23: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_311);  mul_311 = None
        add_192: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_348: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_192, 0.5);  add_192 = None
        mul_349: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, view_525)
        mul_350: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_349, -0.5);  mul_349 = None
        exp_27: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_350);  mul_350 = None
        mul_351: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_27, 0.3989422804014327);  exp_27 = None
        mul_352: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_525, mul_351);  view_525 = mul_351 = None
        add_203: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_348, mul_352);  mul_348 = mul_352 = None
        mul_353: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_543, add_203);  view_543 = add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_544: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_353, [8192, 4096]);  mul_353 = None
        permute_262: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
        permute_278: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_6: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_544, permute_278);  permute_278 = None
        permute_279: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_544, [1, 0])
        mm_7: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_279, view_524);  permute_279 = view_524 = None
        sum_40: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
        view_545: "f32[4096]" = torch.ops.aten.reshape.default(sum_40, [4096]);  sum_40 = None
        view_546: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_6, [16, 512, 1024]);  mm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_355: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, primals_385);  primals_385 = None
        mul_356: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_355, 1024)
        sum_41: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_355, [2], True)
        mul_357: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_355, mul_308);  mul_355 = None
        sum_42: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_357, [2], True);  mul_357 = None
        mul_358: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_308, sum_42);  sum_42 = None
        sub_85: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_356, sum_41);  mul_356 = sum_41 = None
        sub_86: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_85, mul_358);  sub_85 = mul_358 = None
        mul_359: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_52, sub_86);  div_52 = sub_86 = None
        mul_360: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_546, mul_308);  mul_308 = None
        sum_43: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_360, [0, 1]);  mul_360 = None
        sum_44: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_546, [0, 1]);  view_546 = None
        add_204: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(mul_343, mul_359);  mul_343 = mul_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_71, torch.float32);  gt_71 = None
        mul_361: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_362: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_204, mul_361);  mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_547: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_362, [8192, 1024]);  mul_362 = None
        permute_261: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_383, [1, 0]);  primals_383 = None
        permute_282: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        mm_8: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_547, permute_282);  permute_282 = None
        permute_283: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_547, [1, 0])
        mm_9: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_283, view_522);  permute_283 = view_522 = None
        sum_45: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
        view_548: "f32[1024]" = torch.ops.aten.reshape.default(sum_45, [1024]);  sum_45 = None
        view_549: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_8, [16, 512, 1024]);  mm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_550: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_549, [16, 512, 16, 64]);  view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_286: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_550, [0, 2, 1, 3]);  view_550 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_99: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_286, memory_format = torch.contiguous_format);  permute_286 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_99, permute_default, permute_default_1, permute_default_2, None, getitem_100, getitem_101, getitem_102, getitem_103, 0.1, [True, True, True, False], scale = 0.125);  clone_99 = permute_default = permute_default_1 = permute_default_2 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = None
        getitem_104: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[0]
        getitem_105: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[1]
        getitem_106: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default[2];  _scaled_dot_product_efficient_attention_backward_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_5: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3]);  getitem_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_4: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_105, [0, 2, 1, 3]);  getitem_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_3: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_106, [0, 2, 1, 3]);  getitem_106 = None
        clone_101: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_3, memory_format = torch.contiguous_format);  permute_default_3 = None
        view_557: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_101, [16, 512, 1024]);  clone_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_558: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_557, [8192, 1024]);  view_557 = None
        permute_257: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
        permute_293: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
        mm_10: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_558, permute_293);  permute_293 = None
        permute_294: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_558, [1, 0])
        mm_11: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_294, view_506);  permute_294 = None
        sum_47: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_558, [0], True);  view_558 = None
        view_559: "f32[1024]" = torch.ops.aten.reshape.default(sum_47, [1024]);  sum_47 = None
        view_560: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_10, [16, 512, 1024]);  mm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_561: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_4, [16, 512, 1024]);  permute_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_102: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_561, memory_format = torch.contiguous_format);  view_561 = None
        view_562: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_102, [8192, 1024]);  clone_102 = None
        permute_255: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_379, [1, 0]);  primals_379 = None
        permute_298: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        mm_12: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_562, permute_298);  permute_298 = None
        permute_299: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_562, [1, 0])
        mm_13: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_299, view_506);  permute_299 = None
        sum_48: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_562, [0], True);  view_562 = None
        view_563: "f32[1024]" = torch.ops.aten.reshape.default(sum_48, [1024]);  sum_48 = None
        view_564: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_12, [16, 512, 1024]);  mm_12 = None
        add_205: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_560, view_564);  view_560 = view_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_103: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_5, memory_format = torch.contiguous_format);  permute_default_5 = None
        view_565: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_103, [16, 512, 1024]);  clone_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_566: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_565, [8192, 1024]);  view_565 = None
        permute_253: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_377, [1, 0]);  primals_377 = None
        permute_303: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_14: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_566, permute_303);  permute_303 = None
        permute_304: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_566, [1, 0])
        mm_15: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_304, view_506);  permute_304 = view_506 = None
        sum_49: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        view_567: "f32[1024]" = torch.ops.aten.reshape.default(sum_49, [1024]);  sum_49 = None
        view_568: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_14, [16, 512, 1024]);  mm_14 = None
        add_206: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_205, view_568);  add_205 = view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_367: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, primals_375);  primals_375 = None
        mul_368: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, 1024)
        sum_50: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_367, [2], True)
        mul_369: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_367, mul_302);  mul_367 = None
        sum_51: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_369, [2], True);  mul_369 = None
        mul_370: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_302, sum_51);  sum_51 = None
        sub_88: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_368, sum_50);  mul_368 = sum_50 = None
        sub_89: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_88, mul_370);  sub_88 = mul_370 = None
        mul_371: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_54, sub_89);  div_54 = sub_89 = None
        mul_372: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_206, mul_302);  mul_302 = None
        sum_52: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_372, [0, 1]);  mul_372 = None
        sum_53: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None
        add_207: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_204, mul_371);  add_204 = mul_371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_69, torch.float32);  gt_69 = None
        mul_373: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_374: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_207, mul_373);  mul_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_569: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_374, [8192, 1024]);  mul_374 = None
        permute_252: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_373, [1, 0]);  primals_373 = None
        permute_307: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        mm_16: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_569, permute_307);  permute_307 = None
        permute_308: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_569, [1, 0])
        mm_17: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_308, view_504);  permute_308 = view_504 = None
        sum_54: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
        view_570: "f32[1024]" = torch.ops.aten.reshape.default(sum_54, [1024]);  sum_54 = None
        view_571: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_16, [16, 512, 4096]);  mm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_503: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_136, [16, 512, 4096]);  addmm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_298: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, 0.7071067811865476)
        erf_22: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_298);  mul_298 = None
        add_184: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_376: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_184, 0.5);  add_184 = None
        mul_377: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, view_503)
        mul_378: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_377, -0.5);  mul_377 = None
        exp_28: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_378);  mul_378 = None
        mul_379: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_28, 0.3989422804014327);  exp_28 = None
        mul_380: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_503, mul_379);  view_503 = mul_379 = None
        add_209: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_376, mul_380);  mul_376 = mul_380 = None
        mul_381: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_571, add_209);  view_571 = add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_572: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_381, [8192, 4096]);  mul_381 = None
        permute_251: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_371, [1, 0]);  primals_371 = None
        permute_311: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        mm_18: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_572, permute_311);  permute_311 = None
        permute_312: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_572, [1, 0])
        mm_19: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_312, view_502);  permute_312 = view_502 = None
        sum_55: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
        view_573: "f32[4096]" = torch.ops.aten.reshape.default(sum_55, [4096]);  sum_55 = None
        view_574: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_18, [16, 512, 1024]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_383: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, primals_369);  primals_369 = None
        mul_384: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_383, 1024)
        sum_56: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_383, [2], True)
        mul_385: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_383, mul_295);  mul_383 = None
        sum_57: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_385, [2], True);  mul_385 = None
        mul_386: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_295, sum_57);  sum_57 = None
        sub_91: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_384, sum_56);  mul_384 = sum_56 = None
        sub_92: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_91, mul_386);  sub_91 = mul_386 = None
        mul_387: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_55, sub_92);  div_55 = sub_92 = None
        mul_388: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_574, mul_295);  mul_295 = None
        sum_58: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_388, [0, 1]);  mul_388 = None
        sum_59: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_574, [0, 1]);  view_574 = None
        add_210: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_207, mul_387);  add_207 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_5: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_68, torch.float32);  gt_68 = None
        mul_389: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_390: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_210, mul_389);  mul_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_575: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_390, [8192, 1024]);  mul_390 = None
        permute_250: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
        permute_315: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_250, [1, 0]);  permute_250 = None
        mm_20: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_575, permute_315);  permute_315 = None
        permute_316: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_575, [1, 0])
        mm_21: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_316, view_500);  permute_316 = view_500 = None
        sum_60: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
        view_576: "f32[1024]" = torch.ops.aten.reshape.default(sum_60, [1024]);  sum_60 = None
        view_577: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_20, [16, 512, 1024]);  mm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_578: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_577, [16, 512, 16, 64]);  view_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_319: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_106: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_319, memory_format = torch.contiguous_format);  permute_319 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_106, permute_default_6, permute_default_7, permute_default_8, None, getitem_107, getitem_108, getitem_109, getitem_110, 0.1, [True, True, True, False], scale = 0.125);  clone_106 = permute_default_6 = permute_default_7 = permute_default_8 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = None
        getitem_111: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[0]
        getitem_112: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[1]
        getitem_113: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_1[2];  _scaled_dot_product_efficient_attention_backward_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_11: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_111, [0, 2, 1, 3]);  getitem_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_10: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_9: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_113, [0, 2, 1, 3]);  getitem_113 = None
        clone_108: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_9, memory_format = torch.contiguous_format);  permute_default_9 = None
        view_585: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_108, [16, 512, 1024]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_586: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_585, [8192, 1024]);  view_585 = None
        permute_246: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_365, [1, 0]);  primals_365 = None
        permute_326: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_246, [1, 0]);  permute_246 = None
        mm_22: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_586, permute_326);  permute_326 = None
        permute_327: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_586, [1, 0])
        mm_23: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_327, view_484);  permute_327 = None
        sum_62: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
        view_587: "f32[1024]" = torch.ops.aten.reshape.default(sum_62, [1024]);  sum_62 = None
        view_588: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_22, [16, 512, 1024]);  mm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_589: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_10, [16, 512, 1024]);  permute_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_109: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_589, memory_format = torch.contiguous_format);  view_589 = None
        view_590: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_109, [8192, 1024]);  clone_109 = None
        permute_244: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
        permute_331: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_24: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_590, permute_331);  permute_331 = None
        permute_332: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_590, [1, 0])
        mm_25: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_332, view_484);  permute_332 = None
        sum_63: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_590, [0], True);  view_590 = None
        view_591: "f32[1024]" = torch.ops.aten.reshape.default(sum_63, [1024]);  sum_63 = None
        view_592: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_24, [16, 512, 1024]);  mm_24 = None
        add_211: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_588, view_592);  view_588 = view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_110: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_11, memory_format = torch.contiguous_format);  permute_default_11 = None
        view_593: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_110, [16, 512, 1024]);  clone_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_594: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_593, [8192, 1024]);  view_593 = None
        permute_242: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
        permute_336: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_26: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_594, permute_336);  permute_336 = None
        permute_337: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_594, [1, 0])
        mm_27: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_337, view_484);  permute_337 = view_484 = None
        sum_64: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
        view_595: "f32[1024]" = torch.ops.aten.reshape.default(sum_64, [1024]);  sum_64 = None
        view_596: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_26, [16, 512, 1024]);  mm_26 = None
        add_212: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_211, view_596);  add_211 = view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_395: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, primals_359);  primals_359 = None
        mul_396: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_395, 1024)
        sum_65: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_395, [2], True)
        mul_397: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_395, mul_289);  mul_395 = None
        sum_66: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_397, [2], True);  mul_397 = None
        mul_398: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_289, sum_66);  sum_66 = None
        sub_94: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_396, sum_65);  mul_396 = sum_65 = None
        sub_95: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_94, mul_398);  sub_94 = mul_398 = None
        mul_399: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_57, sub_95);  div_57 = sub_95 = None
        mul_400: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_212, mul_289);  mul_289 = None
        sum_67: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_400, [0, 1]);  mul_400 = None
        sum_68: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None
        add_213: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_210, mul_399);  add_210 = mul_399 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_7: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_66, torch.float32);  gt_66 = None
        mul_401: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_402: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_213, mul_401);  mul_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_597: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_402, [8192, 1024]);  mul_402 = None
        permute_241: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
        permute_340: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_28: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_597, permute_340);  permute_340 = None
        permute_341: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_597, [1, 0])
        mm_29: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_341, view_482);  permute_341 = view_482 = None
        sum_69: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
        view_598: "f32[1024]" = torch.ops.aten.reshape.default(sum_69, [1024]);  sum_69 = None
        view_599: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_28, [16, 512, 4096]);  mm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_481: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_130, [16, 512, 4096]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_285: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, 0.7071067811865476)
        erf_21: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_285);  mul_285 = None
        add_176: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_404: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_176, 0.5);  add_176 = None
        mul_405: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, view_481)
        mul_406: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_405, -0.5);  mul_405 = None
        exp_29: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_406);  mul_406 = None
        mul_407: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_29, 0.3989422804014327);  exp_29 = None
        mul_408: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_481, mul_407);  view_481 = mul_407 = None
        add_215: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_404, mul_408);  mul_404 = mul_408 = None
        mul_409: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_599, add_215);  view_599 = add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_600: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_409, [8192, 4096]);  mul_409 = None
        permute_240: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
        permute_344: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        mm_30: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_600, permute_344);  permute_344 = None
        permute_345: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_600, [1, 0])
        mm_31: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_345, view_480);  permute_345 = view_480 = None
        sum_70: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
        view_601: "f32[4096]" = torch.ops.aten.reshape.default(sum_70, [4096]);  sum_70 = None
        view_602: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_30, [16, 512, 1024]);  mm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_411: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, primals_353);  primals_353 = None
        mul_412: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_411, 1024)
        sum_71: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_411, [2], True)
        mul_413: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_411, mul_282);  mul_411 = None
        sum_72: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        mul_414: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_282, sum_72);  sum_72 = None
        sub_97: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_412, sum_71);  mul_412 = sum_71 = None
        sub_98: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_97, mul_414);  sub_97 = mul_414 = None
        mul_415: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_58, sub_98);  div_58 = sub_98 = None
        mul_416: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_602, mul_282);  mul_282 = None
        sum_73: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_416, [0, 1]);  mul_416 = None
        sum_74: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_602, [0, 1]);  view_602 = None
        add_216: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_213, mul_415);  add_213 = mul_415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_417: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_418: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_216, mul_417);  mul_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_603: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_418, [8192, 1024]);  mul_418 = None
        permute_239: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
        permute_348: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        mm_32: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_603, permute_348);  permute_348 = None
        permute_349: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_603, [1, 0])
        mm_33: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_349, view_478);  permute_349 = view_478 = None
        sum_75: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
        view_604: "f32[1024]" = torch.ops.aten.reshape.default(sum_75, [1024]);  sum_75 = None
        view_605: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_32, [16, 512, 1024]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_606: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_605, [16, 512, 16, 64]);  view_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_352: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_113: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_352, memory_format = torch.contiguous_format);  permute_352 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_113, permute_default_12, permute_default_13, permute_default_14, None, getitem_114, getitem_115, getitem_116, getitem_117, 0.1, [True, True, True, False], scale = 0.125);  clone_113 = permute_default_12 = permute_default_13 = permute_default_14 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = None
        getitem_118: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[0]
        getitem_119: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[1]
        getitem_120: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_2[2];  _scaled_dot_product_efficient_attention_backward_default_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_17: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_118, [0, 2, 1, 3]);  getitem_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_16: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_119, [0, 2, 1, 3]);  getitem_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_15: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_120, [0, 2, 1, 3]);  getitem_120 = None
        clone_115: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_15, memory_format = torch.contiguous_format);  permute_default_15 = None
        view_613: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_115, [16, 512, 1024]);  clone_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_614: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_613, [8192, 1024]);  view_613 = None
        permute_235: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
        permute_359: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        mm_34: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_614, permute_359);  permute_359 = None
        permute_360: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_35: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_360, view_462);  permute_360 = None
        sum_77: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_614, [0], True);  view_614 = None
        view_615: "f32[1024]" = torch.ops.aten.reshape.default(sum_77, [1024]);  sum_77 = None
        view_616: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_34, [16, 512, 1024]);  mm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_617: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_16, [16, 512, 1024]);  permute_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_116: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_617, memory_format = torch.contiguous_format);  view_617 = None
        view_618: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_116, [8192, 1024]);  clone_116 = None
        permute_233: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_347, [1, 0]);  primals_347 = None
        permute_364: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_36: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_618, permute_364);  permute_364 = None
        permute_365: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_618, [1, 0])
        mm_37: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_365, view_462);  permute_365 = None
        sum_78: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
        view_619: "f32[1024]" = torch.ops.aten.reshape.default(sum_78, [1024]);  sum_78 = None
        view_620: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_36, [16, 512, 1024]);  mm_36 = None
        add_217: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_616, view_620);  view_616 = view_620 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_117: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_17, memory_format = torch.contiguous_format);  permute_default_17 = None
        view_621: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_117, [16, 512, 1024]);  clone_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_622: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_621, [8192, 1024]);  view_621 = None
        permute_231: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
        permute_369: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        mm_38: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_622, permute_369);  permute_369 = None
        permute_370: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_622, [1, 0])
        mm_39: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_370, view_462);  permute_370 = view_462 = None
        sum_79: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        view_623: "f32[1024]" = torch.ops.aten.reshape.default(sum_79, [1024]);  sum_79 = None
        view_624: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_38, [16, 512, 1024]);  mm_38 = None
        add_218: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_217, view_624);  add_217 = view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_423: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, primals_343);  primals_343 = None
        mul_424: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_423, 1024)
        sum_80: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [2], True)
        mul_425: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_423, mul_276);  mul_423 = None
        sum_81: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_425, [2], True);  mul_425 = None
        mul_426: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_276, sum_81);  sum_81 = None
        sub_100: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_424, sum_80);  mul_424 = sum_80 = None
        sub_101: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_100, mul_426);  sub_100 = mul_426 = None
        mul_427: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_60, sub_101);  div_60 = sub_101 = None
        mul_428: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_218, mul_276);  mul_276 = None
        sum_82: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_428, [0, 1]);  mul_428 = None
        sum_83: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None
        add_219: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_216, mul_427);  add_216 = mul_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_429: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_430: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_219, mul_429);  mul_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_625: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_430, [8192, 1024]);  mul_430 = None
        permute_230: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_341, [1, 0]);  primals_341 = None
        permute_373: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_230, [1, 0]);  permute_230 = None
        mm_40: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_625, permute_373);  permute_373 = None
        permute_374: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_625, [1, 0])
        mm_41: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_374, view_460);  permute_374 = view_460 = None
        sum_84: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        view_626: "f32[1024]" = torch.ops.aten.reshape.default(sum_84, [1024]);  sum_84 = None
        view_627: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_40, [16, 512, 4096]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_459: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_124, [16, 512, 4096]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_272: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, 0.7071067811865476)
        erf_20: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_272);  mul_272 = None
        add_168: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_432: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_168, 0.5);  add_168 = None
        mul_433: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, view_459)
        mul_434: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_433, -0.5);  mul_433 = None
        exp_30: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_434);  mul_434 = None
        mul_435: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_30, 0.3989422804014327);  exp_30 = None
        mul_436: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_459, mul_435);  view_459 = mul_435 = None
        add_221: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_432, mul_436);  mul_432 = mul_436 = None
        mul_437: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_627, add_221);  view_627 = add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_628: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_437, [8192, 4096]);  mul_437 = None
        permute_229: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
        permute_377: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_229, [1, 0]);  permute_229 = None
        mm_42: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_628, permute_377);  permute_377 = None
        permute_378: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_628, [1, 0])
        mm_43: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_378, view_458);  permute_378 = view_458 = None
        sum_85: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        view_629: "f32[4096]" = torch.ops.aten.reshape.default(sum_85, [4096]);  sum_85 = None
        view_630: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_42, [16, 512, 1024]);  mm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_439: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, primals_337);  primals_337 = None
        mul_440: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_439, 1024)
        sum_86: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True)
        mul_441: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_439, mul_269);  mul_439 = None
        sum_87: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_441, [2], True);  mul_441 = None
        mul_442: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_269, sum_87);  sum_87 = None
        sub_103: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_440, sum_86);  mul_440 = sum_86 = None
        sub_104: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_103, mul_442);  sub_103 = mul_442 = None
        mul_443: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_61, sub_104);  div_61 = sub_104 = None
        mul_444: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_630, mul_269);  mul_269 = None
        sum_88: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_444, [0, 1]);  mul_444 = None
        sum_89: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_630, [0, 1]);  view_630 = None
        add_222: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_219, mul_443);  add_219 = mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_11: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_445: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_446: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_222, mul_445);  mul_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_631: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_446, [8192, 1024]);  mul_446 = None
        permute_228: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_335, [1, 0]);  primals_335 = None
        permute_381: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_44: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_631, permute_381);  permute_381 = None
        permute_382: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_631, [1, 0])
        mm_45: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_382, view_456);  permute_382 = view_456 = None
        sum_90: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        view_632: "f32[1024]" = torch.ops.aten.reshape.default(sum_90, [1024]);  sum_90 = None
        view_633: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_44, [16, 512, 1024]);  mm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_634: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_633, [16, 512, 16, 64]);  view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_385: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_634, [0, 2, 1, 3]);  view_634 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_120: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_385, memory_format = torch.contiguous_format);  permute_385 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_120, permute_default_18, permute_default_19, permute_default_20, None, getitem_121, getitem_122, getitem_123, getitem_124, 0.1, [True, True, True, False], scale = 0.125);  clone_120 = permute_default_18 = permute_default_19 = permute_default_20 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = None
        getitem_125: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[0]
        getitem_126: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[1]
        getitem_127: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_3[2];  _scaled_dot_product_efficient_attention_backward_default_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_23: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_125, [0, 2, 1, 3]);  getitem_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_22: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3]);  getitem_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_21: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_127, [0, 2, 1, 3]);  getitem_127 = None
        clone_122: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_21, memory_format = torch.contiguous_format);  permute_default_21 = None
        view_641: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_122, [16, 512, 1024]);  clone_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_642: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_641, [8192, 1024]);  view_641 = None
        permute_224: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
        permute_392: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        mm_46: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_642, permute_392);  permute_392 = None
        permute_393: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_642, [1, 0])
        mm_47: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_393, view_440);  permute_393 = None
        sum_92: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_642, [0], True);  view_642 = None
        view_643: "f32[1024]" = torch.ops.aten.reshape.default(sum_92, [1024]);  sum_92 = None
        view_644: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_46, [16, 512, 1024]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_645: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_22, [16, 512, 1024]);  permute_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_123: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_645, memory_format = torch.contiguous_format);  view_645 = None
        view_646: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_123, [8192, 1024]);  clone_123 = None
        permute_222: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
        permute_397: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_48: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_646, permute_397);  permute_397 = None
        permute_398: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_646, [1, 0])
        mm_49: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_398, view_440);  permute_398 = None
        sum_93: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_646, [0], True);  view_646 = None
        view_647: "f32[1024]" = torch.ops.aten.reshape.default(sum_93, [1024]);  sum_93 = None
        view_648: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_48, [16, 512, 1024]);  mm_48 = None
        add_223: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_644, view_648);  view_644 = view_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_124: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_23, memory_format = torch.contiguous_format);  permute_default_23 = None
        view_649: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_124, [16, 512, 1024]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_650: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_649, [8192, 1024]);  view_649 = None
        permute_220: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_329, [1, 0]);  primals_329 = None
        permute_402: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_50: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_650, permute_402);  permute_402 = None
        permute_403: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_650, [1, 0])
        mm_51: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_403, view_440);  permute_403 = view_440 = None
        sum_94: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        view_651: "f32[1024]" = torch.ops.aten.reshape.default(sum_94, [1024]);  sum_94 = None
        view_652: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_50, [16, 512, 1024]);  mm_50 = None
        add_224: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_223, view_652);  add_223 = view_652 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_451: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, primals_327);  primals_327 = None
        mul_452: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_451, 1024)
        sum_95: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_451, [2], True)
        mul_453: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_451, mul_263);  mul_451 = None
        sum_96: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_453, [2], True);  mul_453 = None
        mul_454: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_263, sum_96);  sum_96 = None
        sub_106: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_452, sum_95);  mul_452 = sum_95 = None
        sub_107: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_106, mul_454);  sub_106 = mul_454 = None
        mul_455: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_63, sub_107);  div_63 = sub_107 = None
        mul_456: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_224, mul_263);  mul_263 = None
        sum_97: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_456, [0, 1]);  mul_456 = None
        sum_98: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None
        add_225: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_222, mul_455);  add_222 = mul_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_13: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_457: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_458: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_225, mul_457);  mul_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_653: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_458, [8192, 1024]);  mul_458 = None
        permute_219: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_325, [1, 0]);  primals_325 = None
        permute_406: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        mm_52: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_653, permute_406);  permute_406 = None
        permute_407: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_653, [1, 0])
        mm_53: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_407, view_438);  permute_407 = view_438 = None
        sum_99: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        view_654: "f32[1024]" = torch.ops.aten.reshape.default(sum_99, [1024]);  sum_99 = None
        view_655: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_52, [16, 512, 4096]);  mm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_437: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_118, [16, 512, 4096]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_259: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, 0.7071067811865476)
        erf_19: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_259);  mul_259 = None
        add_160: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_460: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_160, 0.5);  add_160 = None
        mul_461: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, view_437)
        mul_462: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_461, -0.5);  mul_461 = None
        exp_31: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_462);  mul_462 = None
        mul_463: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_31, 0.3989422804014327);  exp_31 = None
        mul_464: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_437, mul_463);  view_437 = mul_463 = None
        add_227: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_460, mul_464);  mul_460 = mul_464 = None
        mul_465: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_655, add_227);  view_655 = add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_656: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_465, [8192, 4096]);  mul_465 = None
        permute_218: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_323, [1, 0]);  primals_323 = None
        permute_410: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        mm_54: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_656, permute_410);  permute_410 = None
        permute_411: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_656, [1, 0])
        mm_55: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_411, view_436);  permute_411 = view_436 = None
        sum_100: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
        view_657: "f32[4096]" = torch.ops.aten.reshape.default(sum_100, [4096]);  sum_100 = None
        view_658: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_54, [16, 512, 1024]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_467: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, primals_321);  primals_321 = None
        mul_468: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_467, 1024)
        sum_101: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_467, [2], True)
        mul_469: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_467, mul_256);  mul_467 = None
        sum_102: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_469, [2], True);  mul_469 = None
        mul_470: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_256, sum_102);  sum_102 = None
        sub_109: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_468, sum_101);  mul_468 = sum_101 = None
        sub_110: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_109, mul_470);  sub_109 = mul_470 = None
        mul_471: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_64, sub_110);  div_64 = sub_110 = None
        mul_472: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_658, mul_256);  mul_256 = None
        sum_103: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_472, [0, 1]);  mul_472 = None
        sum_104: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_658, [0, 1]);  view_658 = None
        add_228: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_225, mul_471);  add_225 = mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_473: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_474: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_228, mul_473);  mul_473 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_659: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_474, [8192, 1024]);  mul_474 = None
        permute_217: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
        permute_414: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        mm_56: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_659, permute_414);  permute_414 = None
        permute_415: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_659, [1, 0])
        mm_57: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_415, view_434);  permute_415 = view_434 = None
        sum_105: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
        view_660: "f32[1024]" = torch.ops.aten.reshape.default(sum_105, [1024]);  sum_105 = None
        view_661: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_56, [16, 512, 1024]);  mm_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_662: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_661, [16, 512, 16, 64]);  view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_418: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_662, [0, 2, 1, 3]);  view_662 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_127: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_418, memory_format = torch.contiguous_format);  permute_418 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_127, permute_default_24, permute_default_25, permute_default_26, None, getitem_128, getitem_129, getitem_130, getitem_131, 0.1, [True, True, True, False], scale = 0.125);  clone_127 = permute_default_24 = permute_default_25 = permute_default_26 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = None
        getitem_132: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[0]
        getitem_133: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[1]
        getitem_134: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_4[2];  _scaled_dot_product_efficient_attention_backward_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_29: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_28: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_133, [0, 2, 1, 3]);  getitem_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_27: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_134, [0, 2, 1, 3]);  getitem_134 = None
        clone_129: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_27, memory_format = torch.contiguous_format);  permute_default_27 = None
        view_669: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_129, [16, 512, 1024]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_670: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_669, [8192, 1024]);  view_669 = None
        permute_213: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_317, [1, 0]);  primals_317 = None
        permute_425: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        mm_58: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_670, permute_425);  permute_425 = None
        permute_426: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_59: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_426, view_418);  permute_426 = None
        sum_107: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
        view_671: "f32[1024]" = torch.ops.aten.reshape.default(sum_107, [1024]);  sum_107 = None
        view_672: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_58, [16, 512, 1024]);  mm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_673: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_28, [16, 512, 1024]);  permute_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_130: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_673, memory_format = torch.contiguous_format);  view_673 = None
        view_674: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_130, [8192, 1024]);  clone_130 = None
        permute_211: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
        permute_430: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_60: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_674, permute_430);  permute_430 = None
        permute_431: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_674, [1, 0])
        mm_61: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_431, view_418);  permute_431 = None
        sum_108: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_674, [0], True);  view_674 = None
        view_675: "f32[1024]" = torch.ops.aten.reshape.default(sum_108, [1024]);  sum_108 = None
        view_676: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_60, [16, 512, 1024]);  mm_60 = None
        add_229: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_672, view_676);  view_672 = view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_131: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_29, memory_format = torch.contiguous_format);  permute_default_29 = None
        view_677: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_131, [16, 512, 1024]);  clone_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_678: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_677, [8192, 1024]);  view_677 = None
        permute_209: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        permute_435: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_209, [1, 0]);  permute_209 = None
        mm_62: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_678, permute_435);  permute_435 = None
        permute_436: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_678, [1, 0])
        mm_63: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_436, view_418);  permute_436 = view_418 = None
        sum_109: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
        view_679: "f32[1024]" = torch.ops.aten.reshape.default(sum_109, [1024]);  sum_109 = None
        view_680: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_62, [16, 512, 1024]);  mm_62 = None
        add_230: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_229, view_680);  add_229 = view_680 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_479: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, primals_311);  primals_311 = None
        mul_480: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_479, 1024)
        sum_110: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_479, [2], True)
        mul_481: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_479, mul_250);  mul_479 = None
        sum_111: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_481, [2], True);  mul_481 = None
        mul_482: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_250, sum_111);  sum_111 = None
        sub_112: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_480, sum_110);  mul_480 = sum_110 = None
        sub_113: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_112, mul_482);  sub_112 = mul_482 = None
        mul_483: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_66, sub_113);  div_66 = sub_113 = None
        mul_484: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_230, mul_250);  mul_250 = None
        sum_112: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_484, [0, 1]);  mul_484 = None
        sum_113: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None
        add_231: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_228, mul_483);  add_228 = mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_485: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_486: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_231, mul_485);  mul_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_681: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_486, [8192, 1024]);  mul_486 = None
        permute_208: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
        permute_439: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_208, [1, 0]);  permute_208 = None
        mm_64: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_681, permute_439);  permute_439 = None
        permute_440: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_681, [1, 0])
        mm_65: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_440, view_416);  permute_440 = view_416 = None
        sum_114: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
        view_682: "f32[1024]" = torch.ops.aten.reshape.default(sum_114, [1024]);  sum_114 = None
        view_683: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_64, [16, 512, 4096]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_415: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_112, [16, 512, 4096]);  addmm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_246: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476)
        erf_18: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_246);  mul_246 = None
        add_152: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_488: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_152, 0.5);  add_152 = None
        mul_489: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, view_415)
        mul_490: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_489, -0.5);  mul_489 = None
        exp_32: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_490);  mul_490 = None
        mul_491: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_32, 0.3989422804014327);  exp_32 = None
        mul_492: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_415, mul_491);  view_415 = mul_491 = None
        add_233: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_488, mul_492);  mul_488 = mul_492 = None
        mul_493: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_683, add_233);  view_683 = add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_684: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_493, [8192, 4096]);  mul_493 = None
        permute_207: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        permute_443: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_66: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_684, permute_443);  permute_443 = None
        permute_444: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_684, [1, 0])
        mm_67: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_444, view_414);  permute_444 = view_414 = None
        sum_115: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
        view_685: "f32[4096]" = torch.ops.aten.reshape.default(sum_115, [4096]);  sum_115 = None
        view_686: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_66, [16, 512, 1024]);  mm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_495: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, primals_305);  primals_305 = None
        mul_496: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_495, 1024)
        sum_116: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_495, [2], True)
        mul_497: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_495, mul_243);  mul_495 = None
        sum_117: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_497, [2], True);  mul_497 = None
        mul_498: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_243, sum_117);  sum_117 = None
        sub_115: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_496, sum_116);  mul_496 = sum_116 = None
        sub_116: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_115, mul_498);  sub_115 = mul_498 = None
        mul_499: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_67, sub_116);  div_67 = sub_116 = None
        mul_500: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_686, mul_243);  mul_243 = None
        sum_118: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_500, [0, 1]);  mul_500 = None
        sum_119: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_686, [0, 1]);  view_686 = None
        add_234: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_231, mul_499);  add_231 = mul_499 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_17: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_501: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_502: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_234, mul_501);  mul_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_687: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_502, [8192, 1024]);  mul_502 = None
        permute_206: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
        permute_447: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_206, [1, 0]);  permute_206 = None
        mm_68: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_687, permute_447);  permute_447 = None
        permute_448: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_687, [1, 0])
        mm_69: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_448, view_412);  permute_448 = view_412 = None
        sum_120: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        view_688: "f32[1024]" = torch.ops.aten.reshape.default(sum_120, [1024]);  sum_120 = None
        view_689: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_68, [16, 512, 1024]);  mm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_690: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_689, [16, 512, 16, 64]);  view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_451: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_134: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_451, memory_format = torch.contiguous_format);  permute_451 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_134, permute_default_30, permute_default_31, permute_default_32, None, getitem_135, getitem_136, getitem_137, getitem_138, 0.1, [True, True, True, False], scale = 0.125);  clone_134 = permute_default_30 = permute_default_31 = permute_default_32 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = None
        getitem_139: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[0]
        getitem_140: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[1]
        getitem_141: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_5[2];  _scaled_dot_product_efficient_attention_backward_default_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_35: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_139, [0, 2, 1, 3]);  getitem_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_34: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_140, [0, 2, 1, 3]);  getitem_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_33: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_141, [0, 2, 1, 3]);  getitem_141 = None
        clone_136: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_33, memory_format = torch.contiguous_format);  permute_default_33 = None
        view_697: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_136, [16, 512, 1024]);  clone_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_698: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_697, [8192, 1024]);  view_697 = None
        permute_202: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
        permute_458: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        mm_70: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_698, permute_458);  permute_458 = None
        permute_459: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_698, [1, 0])
        mm_71: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_459, view_396);  permute_459 = None
        sum_122: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
        view_699: "f32[1024]" = torch.ops.aten.reshape.default(sum_122, [1024]);  sum_122 = None
        view_700: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_70, [16, 512, 1024]);  mm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_701: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_34, [16, 512, 1024]);  permute_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_137: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_701, memory_format = torch.contiguous_format);  view_701 = None
        view_702: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_137, [8192, 1024]);  clone_137 = None
        permute_200: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_299, [1, 0]);  primals_299 = None
        permute_463: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_72: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_702, permute_463);  permute_463 = None
        permute_464: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_702, [1, 0])
        mm_73: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_464, view_396);  permute_464 = None
        sum_123: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_702, [0], True);  view_702 = None
        view_703: "f32[1024]" = torch.ops.aten.reshape.default(sum_123, [1024]);  sum_123 = None
        view_704: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_72, [16, 512, 1024]);  mm_72 = None
        add_235: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_700, view_704);  view_700 = view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_138: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_35, memory_format = torch.contiguous_format);  permute_default_35 = None
        view_705: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_138, [16, 512, 1024]);  clone_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_706: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_705, [8192, 1024]);  view_705 = None
        permute_198: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
        permute_468: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        mm_74: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_706, permute_468);  permute_468 = None
        permute_469: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_706, [1, 0])
        mm_75: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_469, view_396);  permute_469 = view_396 = None
        sum_124: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
        view_707: "f32[1024]" = torch.ops.aten.reshape.default(sum_124, [1024]);  sum_124 = None
        view_708: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_74, [16, 512, 1024]);  mm_74 = None
        add_236: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_235, view_708);  add_235 = view_708 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_507: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, primals_295);  primals_295 = None
        mul_508: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_507, 1024)
        sum_125: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_507, [2], True)
        mul_509: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_507, mul_237);  mul_507 = None
        sum_126: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True);  mul_509 = None
        mul_510: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_237, sum_126);  sum_126 = None
        sub_118: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_508, sum_125);  mul_508 = sum_125 = None
        sub_119: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_118, mul_510);  sub_118 = mul_510 = None
        mul_511: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_69, sub_119);  div_69 = sub_119 = None
        mul_512: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_236, mul_237);  mul_237 = None
        sum_127: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_512, [0, 1]);  mul_512 = None
        sum_128: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None
        add_237: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_234, mul_511);  add_234 = mul_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_19: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_513: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_514: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_237, mul_513);  mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_709: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_514, [8192, 1024]);  mul_514 = None
        permute_197: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_293, [1, 0]);  primals_293 = None
        permute_472: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        mm_76: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_709, permute_472);  permute_472 = None
        permute_473: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_709, [1, 0])
        mm_77: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_473, view_394);  permute_473 = view_394 = None
        sum_129: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        view_710: "f32[1024]" = torch.ops.aten.reshape.default(sum_129, [1024]);  sum_129 = None
        view_711: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_76, [16, 512, 4096]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_393: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_106, [16, 512, 4096]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_233: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, 0.7071067811865476)
        erf_17: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_233);  mul_233 = None
        add_144: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_516: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_144, 0.5);  add_144 = None
        mul_517: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, view_393)
        mul_518: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_517, -0.5);  mul_517 = None
        exp_33: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_518);  mul_518 = None
        mul_519: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_33, 0.3989422804014327);  exp_33 = None
        mul_520: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_393, mul_519);  view_393 = mul_519 = None
        add_239: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_516, mul_520);  mul_516 = mul_520 = None
        mul_521: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_711, add_239);  view_711 = add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_712: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_521, [8192, 4096]);  mul_521 = None
        permute_196: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        permute_476: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        mm_78: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_712, permute_476);  permute_476 = None
        permute_477: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_712, [1, 0])
        mm_79: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_477, view_392);  permute_477 = view_392 = None
        sum_130: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        view_713: "f32[4096]" = torch.ops.aten.reshape.default(sum_130, [4096]);  sum_130 = None
        view_714: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_78, [16, 512, 1024]);  mm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_523: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, primals_289);  primals_289 = None
        mul_524: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_523, 1024)
        sum_131: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_523, [2], True)
        mul_525: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_523, mul_230);  mul_523 = None
        sum_132: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_525, [2], True);  mul_525 = None
        mul_526: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_230, sum_132);  sum_132 = None
        sub_121: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_524, sum_131);  mul_524 = sum_131 = None
        sub_122: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_121, mul_526);  sub_121 = mul_526 = None
        mul_527: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_70, sub_122);  div_70 = sub_122 = None
        mul_528: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_714, mul_230);  mul_230 = None
        sum_133: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_528, [0, 1]);  mul_528 = None
        sum_134: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_714, [0, 1]);  view_714 = None
        add_240: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_237, mul_527);  add_237 = mul_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_529: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_530: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_240, mul_529);  mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_715: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_530, [8192, 1024]);  mul_530 = None
        permute_195: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_287, [1, 0]);  primals_287 = None
        permute_480: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_80: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_715, permute_480);  permute_480 = None
        permute_481: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_715, [1, 0])
        mm_81: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_481, view_390);  permute_481 = view_390 = None
        sum_135: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        view_716: "f32[1024]" = torch.ops.aten.reshape.default(sum_135, [1024]);  sum_135 = None
        view_717: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_80, [16, 512, 1024]);  mm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_718: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_717, [16, 512, 16, 64]);  view_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_484: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_718, [0, 2, 1, 3]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_141: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_484, memory_format = torch.contiguous_format);  permute_484 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_141, permute_default_36, permute_default_37, permute_default_38, None, getitem_142, getitem_143, getitem_144, getitem_145, 0.1, [True, True, True, False], scale = 0.125);  clone_141 = permute_default_36 = permute_default_37 = permute_default_38 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = None
        getitem_146: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[0]
        getitem_147: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[1]
        getitem_148: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_6[2];  _scaled_dot_product_efficient_attention_backward_default_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_41: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_146, [0, 2, 1, 3]);  getitem_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_40: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_147, [0, 2, 1, 3]);  getitem_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_39: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3]);  getitem_148 = None
        clone_143: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_39, memory_format = torch.contiguous_format);  permute_default_39 = None
        view_725: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_143, [16, 512, 1024]);  clone_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_726: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_725, [8192, 1024]);  view_725 = None
        permute_191: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        permute_491: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        mm_82: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_726, permute_491);  permute_491 = None
        permute_492: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_726, [1, 0])
        mm_83: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_492, view_374);  permute_492 = None
        sum_137: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_726, [0], True);  view_726 = None
        view_727: "f32[1024]" = torch.ops.aten.reshape.default(sum_137, [1024]);  sum_137 = None
        view_728: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_82, [16, 512, 1024]);  mm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_729: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_40, [16, 512, 1024]);  permute_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_144: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_729, memory_format = torch.contiguous_format);  view_729 = None
        view_730: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_144, [8192, 1024]);  clone_144 = None
        permute_189: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        permute_496: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_189, [1, 0]);  permute_189 = None
        mm_84: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_730, permute_496);  permute_496 = None
        permute_497: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_730, [1, 0])
        mm_85: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_497, view_374);  permute_497 = None
        sum_138: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_730, [0], True);  view_730 = None
        view_731: "f32[1024]" = torch.ops.aten.reshape.default(sum_138, [1024]);  sum_138 = None
        view_732: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_84, [16, 512, 1024]);  mm_84 = None
        add_241: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_728, view_732);  view_728 = view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_145: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_41, memory_format = torch.contiguous_format);  permute_default_41 = None
        view_733: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_145, [16, 512, 1024]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_734: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_733, [8192, 1024]);  view_733 = None
        permute_187: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_281, [1, 0]);  primals_281 = None
        permute_501: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        mm_86: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_734, permute_501);  permute_501 = None
        permute_502: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_734, [1, 0])
        mm_87: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_502, view_374);  permute_502 = view_374 = None
        sum_139: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        view_735: "f32[1024]" = torch.ops.aten.reshape.default(sum_139, [1024]);  sum_139 = None
        view_736: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_86, [16, 512, 1024]);  mm_86 = None
        add_242: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_241, view_736);  add_241 = view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_535: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, primals_279);  primals_279 = None
        mul_536: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_535, 1024)
        sum_140: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_535, [2], True)
        mul_537: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_535, mul_224);  mul_535 = None
        sum_141: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True);  mul_537 = None
        mul_538: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_224, sum_141);  sum_141 = None
        sub_124: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_536, sum_140);  mul_536 = sum_140 = None
        sub_125: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_124, mul_538);  sub_124 = mul_538 = None
        mul_539: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_72, sub_125);  div_72 = sub_125 = None
        mul_540: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_242, mul_224);  mul_224 = None
        sum_142: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_540, [0, 1]);  mul_540 = None
        sum_143: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None
        add_243: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_240, mul_539);  add_240 = mul_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_541: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_542: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_243, mul_541);  mul_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_737: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_542, [8192, 1024]);  mul_542 = None
        permute_186: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
        permute_505: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_88: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_737, permute_505);  permute_505 = None
        permute_506: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_737, [1, 0])
        mm_89: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_506, view_372);  permute_506 = view_372 = None
        sum_144: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        view_738: "f32[1024]" = torch.ops.aten.reshape.default(sum_144, [1024]);  sum_144 = None
        view_739: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_88, [16, 512, 4096]);  mm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_371: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_100, [16, 512, 4096]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_220: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, 0.7071067811865476)
        erf_16: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_220);  mul_220 = None
        add_136: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_544: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_136, 0.5);  add_136 = None
        mul_545: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, view_371)
        mul_546: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_545, -0.5);  mul_545 = None
        exp_34: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_546);  mul_546 = None
        mul_547: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_34, 0.3989422804014327);  exp_34 = None
        mul_548: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_371, mul_547);  view_371 = mul_547 = None
        add_245: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_544, mul_548);  mul_544 = mul_548 = None
        mul_549: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_739, add_245);  view_739 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_740: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_549, [8192, 4096]);  mul_549 = None
        permute_185: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_275, [1, 0]);  primals_275 = None
        permute_509: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        mm_90: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_740, permute_509);  permute_509 = None
        permute_510: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_740, [1, 0])
        mm_91: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_510, view_370);  permute_510 = view_370 = None
        sum_145: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        view_741: "f32[4096]" = torch.ops.aten.reshape.default(sum_145, [4096]);  sum_145 = None
        view_742: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_90, [16, 512, 1024]);  mm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_551: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, primals_273);  primals_273 = None
        mul_552: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_551, 1024)
        sum_146: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_551, [2], True)
        mul_553: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_551, mul_217);  mul_551 = None
        sum_147: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_553, [2], True);  mul_553 = None
        mul_554: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_217, sum_147);  sum_147 = None
        sub_127: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_552, sum_146);  mul_552 = sum_146 = None
        sub_128: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_127, mul_554);  sub_127 = mul_554 = None
        mul_555: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_73, sub_128);  div_73 = sub_128 = None
        mul_556: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_742, mul_217);  mul_217 = None
        sum_148: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_556, [0, 1]);  mul_556 = None
        sum_149: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None
        add_246: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_243, mul_555);  add_243 = mul_555 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_23: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_557: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_558: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_246, mul_557);  mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_743: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_558, [8192, 1024]);  mul_558 = None
        permute_184: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        permute_513: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_92: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_743, permute_513);  permute_513 = None
        permute_514: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_93: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_514, view_368);  permute_514 = view_368 = None
        sum_150: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        view_744: "f32[1024]" = torch.ops.aten.reshape.default(sum_150, [1024]);  sum_150 = None
        view_745: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_92, [16, 512, 1024]);  mm_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_746: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_745, [16, 512, 16, 64]);  view_745 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_517: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_746, [0, 2, 1, 3]);  view_746 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_148: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_517, memory_format = torch.contiguous_format);  permute_517 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_148, permute_default_42, permute_default_43, permute_default_44, None, getitem_149, getitem_150, getitem_151, getitem_152, 0.1, [True, True, True, False], scale = 0.125);  clone_148 = permute_default_42 = permute_default_43 = permute_default_44 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = None
        getitem_153: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[0]
        getitem_154: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[1]
        getitem_155: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_7[2];  _scaled_dot_product_efficient_attention_backward_default_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_47: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_153, [0, 2, 1, 3]);  getitem_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_46: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_154, [0, 2, 1, 3]);  getitem_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_45: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_155, [0, 2, 1, 3]);  getitem_155 = None
        clone_150: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_45, memory_format = torch.contiguous_format);  permute_default_45 = None
        view_753: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_150, [16, 512, 1024]);  clone_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_754: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_753, [8192, 1024]);  view_753 = None
        permute_180: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_269, [1, 0]);  primals_269 = None
        permute_524: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        mm_94: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_754, permute_524);  permute_524 = None
        permute_525: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_754, [1, 0])
        mm_95: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_525, view_352);  permute_525 = None
        sum_152: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_754, [0], True);  view_754 = None
        view_755: "f32[1024]" = torch.ops.aten.reshape.default(sum_152, [1024]);  sum_152 = None
        view_756: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_94, [16, 512, 1024]);  mm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_757: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_46, [16, 512, 1024]);  permute_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_151: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_757, memory_format = torch.contiguous_format);  view_757 = None
        view_758: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_151, [8192, 1024]);  clone_151 = None
        permute_178: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
        permute_529: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_96: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_758, permute_529);  permute_529 = None
        permute_530: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_758, [1, 0])
        mm_97: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_530, view_352);  permute_530 = None
        sum_153: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_758, [0], True);  view_758 = None
        view_759: "f32[1024]" = torch.ops.aten.reshape.default(sum_153, [1024]);  sum_153 = None
        view_760: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_96, [16, 512, 1024]);  mm_96 = None
        add_247: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_756, view_760);  view_756 = view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_152: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_47, memory_format = torch.contiguous_format);  permute_default_47 = None
        view_761: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_152, [16, 512, 1024]);  clone_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_762: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_761, [8192, 1024]);  view_761 = None
        permute_176: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        permute_534: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        mm_98: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_762, permute_534);  permute_534 = None
        permute_535: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_762, [1, 0])
        mm_99: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_535, view_352);  permute_535 = view_352 = None
        sum_154: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        view_763: "f32[1024]" = torch.ops.aten.reshape.default(sum_154, [1024]);  sum_154 = None
        view_764: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_98, [16, 512, 1024]);  mm_98 = None
        add_248: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_247, view_764);  add_247 = view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_563: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, primals_263);  primals_263 = None
        mul_564: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_563, 1024)
        sum_155: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True)
        mul_565: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_563, mul_211);  mul_563 = None
        sum_156: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True);  mul_565 = None
        mul_566: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_211, sum_156);  sum_156 = None
        sub_130: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_564, sum_155);  mul_564 = sum_155 = None
        sub_131: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_130, mul_566);  sub_130 = mul_566 = None
        mul_567: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_75, sub_131);  div_75 = sub_131 = None
        mul_568: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_248, mul_211);  mul_211 = None
        sum_157: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 1]);  mul_568 = None
        sum_158: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None
        add_249: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_246, mul_567);  add_246 = mul_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_25: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_569: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_570: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_249, mul_569);  mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_765: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_570, [8192, 1024]);  mul_570 = None
        permute_175: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
        permute_538: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        mm_100: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_765, permute_538);  permute_538 = None
        permute_539: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_765, [1, 0])
        mm_101: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_539, view_350);  permute_539 = view_350 = None
        sum_159: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        view_766: "f32[1024]" = torch.ops.aten.reshape.default(sum_159, [1024]);  sum_159 = None
        view_767: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_100, [16, 512, 4096]);  mm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_349: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_94, [16, 512, 4096]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_207: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, 0.7071067811865476)
        erf_15: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_128: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_572: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_128, 0.5);  add_128 = None
        mul_573: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, view_349)
        mul_574: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_573, -0.5);  mul_573 = None
        exp_35: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_574);  mul_574 = None
        mul_575: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_35, 0.3989422804014327);  exp_35 = None
        mul_576: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_349, mul_575);  view_349 = mul_575 = None
        add_251: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_572, mul_576);  mul_572 = mul_576 = None
        mul_577: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_767, add_251);  view_767 = add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_768: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_577, [8192, 4096]);  mul_577 = None
        permute_174: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        permute_542: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_102: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_768, permute_542);  permute_542 = None
        permute_543: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_768, [1, 0])
        mm_103: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_543, view_348);  permute_543 = view_348 = None
        sum_160: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        view_769: "f32[4096]" = torch.ops.aten.reshape.default(sum_160, [4096]);  sum_160 = None
        view_770: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_102, [16, 512, 1024]);  mm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_579: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, primals_257);  primals_257 = None
        mul_580: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_579, 1024)
        sum_161: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_579, [2], True)
        mul_581: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_579, mul_204);  mul_579 = None
        sum_162: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_581, [2], True);  mul_581 = None
        mul_582: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_204, sum_162);  sum_162 = None
        sub_133: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_580, sum_161);  mul_580 = sum_161 = None
        sub_134: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_133, mul_582);  sub_133 = mul_582 = None
        mul_583: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_76, sub_134);  div_76 = sub_134 = None
        mul_584: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_770, mul_204);  mul_204 = None
        sum_163: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_584, [0, 1]);  mul_584 = None
        sum_164: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_770, [0, 1]);  view_770 = None
        add_252: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_249, mul_583);  add_249 = mul_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_585: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_586: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_252, mul_585);  mul_585 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_771: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_586, [8192, 1024]);  mul_586 = None
        permute_173: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
        permute_546: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        mm_104: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_771, permute_546);  permute_546 = None
        permute_547: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_105: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_547, view_346);  permute_547 = view_346 = None
        sum_165: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
        view_772: "f32[1024]" = torch.ops.aten.reshape.default(sum_165, [1024]);  sum_165 = None
        view_773: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_104, [16, 512, 1024]);  mm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_774: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_773, [16, 512, 16, 64]);  view_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_550: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_774, [0, 2, 1, 3]);  view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_155: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_550, memory_format = torch.contiguous_format);  permute_550 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_155, permute_default_48, permute_default_49, permute_default_50, None, getitem_156, getitem_157, getitem_158, getitem_159, 0.1, [True, True, True, False], scale = 0.125);  clone_155 = permute_default_48 = permute_default_49 = permute_default_50 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = None
        getitem_160: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[0]
        getitem_161: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[1]
        getitem_162: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_8[2];  _scaled_dot_product_efficient_attention_backward_default_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_53: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_160, [0, 2, 1, 3]);  getitem_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_52: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_161, [0, 2, 1, 3]);  getitem_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_51: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None
        clone_157: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_51, memory_format = torch.contiguous_format);  permute_default_51 = None
        view_781: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_157, [16, 512, 1024]);  clone_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_782: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_781, [8192, 1024]);  view_781 = None
        permute_169: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
        permute_557: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        mm_106: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_782, permute_557);  permute_557 = None
        permute_558: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_782, [1, 0])
        mm_107: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_558, view_330);  permute_558 = None
        sum_167: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_782, [0], True);  view_782 = None
        view_783: "f32[1024]" = torch.ops.aten.reshape.default(sum_167, [1024]);  sum_167 = None
        view_784: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_106, [16, 512, 1024]);  mm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_785: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_52, [16, 512, 1024]);  permute_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_158: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_785, memory_format = torch.contiguous_format);  view_785 = None
        view_786: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_158, [8192, 1024]);  clone_158 = None
        permute_167: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_251, [1, 0]);  primals_251 = None
        permute_562: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        mm_108: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_786, permute_562);  permute_562 = None
        permute_563: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_786, [1, 0])
        mm_109: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_563, view_330);  permute_563 = None
        sum_168: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
        view_787: "f32[1024]" = torch.ops.aten.reshape.default(sum_168, [1024]);  sum_168 = None
        view_788: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_108, [16, 512, 1024]);  mm_108 = None
        add_253: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_784, view_788);  view_784 = view_788 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_159: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_53, memory_format = torch.contiguous_format);  permute_default_53 = None
        view_789: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_159, [16, 512, 1024]);  clone_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_790: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_789, [8192, 1024]);  view_789 = None
        permute_165: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
        permute_567: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_110: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_790, permute_567);  permute_567 = None
        permute_568: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_790, [1, 0])
        mm_111: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_568, view_330);  permute_568 = view_330 = None
        sum_169: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        view_791: "f32[1024]" = torch.ops.aten.reshape.default(sum_169, [1024]);  sum_169 = None
        view_792: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_110, [16, 512, 1024]);  mm_110 = None
        add_254: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_253, view_792);  add_253 = view_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_591: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, primals_247);  primals_247 = None
        mul_592: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_591, 1024)
        sum_170: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_591, [2], True)
        mul_593: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_591, mul_198);  mul_591 = None
        sum_171: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_593, [2], True);  mul_593 = None
        mul_594: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_198, sum_171);  sum_171 = None
        sub_136: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_592, sum_170);  mul_592 = sum_170 = None
        sub_137: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_136, mul_594);  sub_136 = mul_594 = None
        mul_595: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_78, sub_137);  div_78 = sub_137 = None
        mul_596: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_254, mul_198);  mul_198 = None
        sum_172: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_596, [0, 1]);  mul_596 = None
        sum_173: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None
        add_255: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_252, mul_595);  add_252 = mul_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_597: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_598: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_255, mul_597);  mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_793: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_598, [8192, 1024]);  mul_598 = None
        permute_164: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_245, [1, 0]);  primals_245 = None
        permute_571: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_164, [1, 0]);  permute_164 = None
        mm_112: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_793, permute_571);  permute_571 = None
        permute_572: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_793, [1, 0])
        mm_113: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_572, view_328);  permute_572 = view_328 = None
        sum_174: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        view_794: "f32[1024]" = torch.ops.aten.reshape.default(sum_174, [1024]);  sum_174 = None
        view_795: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_112, [16, 512, 4096]);  mm_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_327: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_88, [16, 512, 4096]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_194: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, 0.7071067811865476)
        erf_14: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_194);  mul_194 = None
        add_120: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_600: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_120, 0.5);  add_120 = None
        mul_601: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, view_327)
        mul_602: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_601, -0.5);  mul_601 = None
        exp_36: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_602);  mul_602 = None
        mul_603: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_36, 0.3989422804014327);  exp_36 = None
        mul_604: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_327, mul_603);  view_327 = mul_603 = None
        add_257: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_600, mul_604);  mul_600 = mul_604 = None
        mul_605: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_795, add_257);  view_795 = add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_796: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_605, [8192, 4096]);  mul_605 = None
        permute_163: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
        permute_575: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_114: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_796, permute_575);  permute_575 = None
        permute_576: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_115: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_576, view_326);  permute_576 = view_326 = None
        sum_175: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        view_797: "f32[4096]" = torch.ops.aten.reshape.default(sum_175, [4096]);  sum_175 = None
        view_798: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_114, [16, 512, 1024]);  mm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_607: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, primals_241);  primals_241 = None
        mul_608: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_607, 1024)
        sum_176: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_607, [2], True)
        mul_609: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_607, mul_191);  mul_607 = None
        sum_177: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True);  mul_609 = None
        mul_610: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_191, sum_177);  sum_177 = None
        sub_139: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_608, sum_176);  mul_608 = sum_176 = None
        sub_140: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_139, mul_610);  sub_139 = mul_610 = None
        mul_611: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_79, sub_140);  div_79 = sub_140 = None
        mul_612: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_798, mul_191);  mul_191 = None
        sum_178: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_612, [0, 1]);  mul_612 = None
        sum_179: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_798, [0, 1]);  view_798 = None
        add_258: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_255, mul_611);  add_255 = mul_611 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_29: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_613: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_614: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_258, mul_613);  mul_613 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_799: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_614, [8192, 1024]);  mul_614 = None
        permute_162: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_239, [1, 0]);  primals_239 = None
        permute_579: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        mm_116: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_799, permute_579);  permute_579 = None
        permute_580: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_117: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_580, view_324);  permute_580 = view_324 = None
        sum_180: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        view_800: "f32[1024]" = torch.ops.aten.reshape.default(sum_180, [1024]);  sum_180 = None
        view_801: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_116, [16, 512, 1024]);  mm_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_802: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_801, [16, 512, 16, 64]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_583: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_802, [0, 2, 1, 3]);  view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_162: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_583, memory_format = torch.contiguous_format);  permute_583 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_162, permute_default_54, permute_default_55, permute_default_56, None, getitem_163, getitem_164, getitem_165, getitem_166, 0.1, [True, True, True, False], scale = 0.125);  clone_162 = permute_default_54 = permute_default_55 = permute_default_56 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = None
        getitem_167: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[0]
        getitem_168: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[1]
        getitem_169: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_9[2];  _scaled_dot_product_efficient_attention_backward_default_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_59: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_167, [0, 2, 1, 3]);  getitem_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_58: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_168, [0, 2, 1, 3]);  getitem_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_57: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_169, [0, 2, 1, 3]);  getitem_169 = None
        clone_164: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_57, memory_format = torch.contiguous_format);  permute_default_57 = None
        view_809: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_164, [16, 512, 1024]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_810: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_809, [8192, 1024]);  view_809 = None
        permute_158: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        permute_590: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        mm_118: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_810, permute_590);  permute_590 = None
        permute_591: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_810, [1, 0])
        mm_119: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_591, view_308);  permute_591 = None
        sum_182: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
        view_811: "f32[1024]" = torch.ops.aten.reshape.default(sum_182, [1024]);  sum_182 = None
        view_812: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_118, [16, 512, 1024]);  mm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_813: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_58, [16, 512, 1024]);  permute_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_165: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_813, memory_format = torch.contiguous_format);  view_813 = None
        view_814: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_165, [8192, 1024]);  clone_165 = None
        permute_156: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
        permute_595: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        mm_120: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_814, permute_595);  permute_595 = None
        permute_596: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_814, [1, 0])
        mm_121: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_596, view_308);  permute_596 = None
        sum_183: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_814, [0], True);  view_814 = None
        view_815: "f32[1024]" = torch.ops.aten.reshape.default(sum_183, [1024]);  sum_183 = None
        view_816: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_120, [16, 512, 1024]);  mm_120 = None
        add_259: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_812, view_816);  view_812 = view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_166: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_59, memory_format = torch.contiguous_format);  permute_default_59 = None
        view_817: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_166, [16, 512, 1024]);  clone_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_818: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_817, [8192, 1024]);  view_817 = None
        permute_154: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_233, [1, 0]);  primals_233 = None
        permute_600: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        mm_122: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_818, permute_600);  permute_600 = None
        permute_601: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_818, [1, 0])
        mm_123: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_601, view_308);  permute_601 = view_308 = None
        sum_184: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        view_819: "f32[1024]" = torch.ops.aten.reshape.default(sum_184, [1024]);  sum_184 = None
        view_820: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_122, [16, 512, 1024]);  mm_122 = None
        add_260: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_259, view_820);  add_259 = view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_619: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, primals_231);  primals_231 = None
        mul_620: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_619, 1024)
        sum_185: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_619, [2], True)
        mul_621: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_619, mul_185);  mul_619 = None
        sum_186: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_621, [2], True);  mul_621 = None
        mul_622: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_185, sum_186);  sum_186 = None
        sub_142: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_620, sum_185);  mul_620 = sum_185 = None
        sub_143: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_142, mul_622);  sub_142 = mul_622 = None
        mul_623: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_81, sub_143);  div_81 = sub_143 = None
        mul_624: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_260, mul_185);  mul_185 = None
        sum_187: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_624, [0, 1]);  mul_624 = None
        sum_188: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None
        add_261: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_258, mul_623);  add_258 = mul_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_31: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_625: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_626: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_261, mul_625);  mul_625 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_821: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_626, [8192, 1024]);  mul_626 = None
        permute_153: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
        permute_604: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_124: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_821, permute_604);  permute_604 = None
        permute_605: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_821, [1, 0])
        mm_125: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_605, view_306);  permute_605 = view_306 = None
        sum_189: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        view_822: "f32[1024]" = torch.ops.aten.reshape.default(sum_189, [1024]);  sum_189 = None
        view_823: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_124, [16, 512, 4096]);  mm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_305: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_82, [16, 512, 4096]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_181: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, 0.7071067811865476)
        erf_13: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_181);  mul_181 = None
        add_112: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_628: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_112, 0.5);  add_112 = None
        mul_629: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, view_305)
        mul_630: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_629, -0.5);  mul_629 = None
        exp_37: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_630);  mul_630 = None
        mul_631: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_37, 0.3989422804014327);  exp_37 = None
        mul_632: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_305, mul_631);  view_305 = mul_631 = None
        add_263: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_628, mul_632);  mul_628 = mul_632 = None
        mul_633: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_823, add_263);  view_823 = add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_824: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_633, [8192, 4096]);  mul_633 = None
        permute_152: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_227, [1, 0]);  primals_227 = None
        permute_608: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        mm_126: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_824, permute_608);  permute_608 = None
        permute_609: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_127: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_609, view_304);  permute_609 = view_304 = None
        sum_190: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
        view_825: "f32[4096]" = torch.ops.aten.reshape.default(sum_190, [4096]);  sum_190 = None
        view_826: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_126, [16, 512, 1024]);  mm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_635: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, primals_225);  primals_225 = None
        mul_636: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_635, 1024)
        sum_191: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True)
        mul_637: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_635, mul_178);  mul_635 = None
        sum_192: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_637, [2], True);  mul_637 = None
        mul_638: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_178, sum_192);  sum_192 = None
        sub_145: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_636, sum_191);  mul_636 = sum_191 = None
        sub_146: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_145, mul_638);  sub_145 = mul_638 = None
        mul_639: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_82, sub_146);  div_82 = sub_146 = None
        mul_640: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_826, mul_178);  mul_178 = None
        sum_193: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_640, [0, 1]);  mul_640 = None
        sum_194: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_826, [0, 1]);  view_826 = None
        add_264: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_261, mul_639);  add_261 = mul_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_641: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_642: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_264, mul_641);  mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_827: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_642, [8192, 1024]);  mul_642 = None
        permute_151: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
        permute_612: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_128: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_827, permute_612);  permute_612 = None
        permute_613: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_827, [1, 0])
        mm_129: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_613, view_302);  permute_613 = view_302 = None
        sum_195: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
        view_828: "f32[1024]" = torch.ops.aten.reshape.default(sum_195, [1024]);  sum_195 = None
        view_829: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_128, [16, 512, 1024]);  mm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_830: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_829, [16, 512, 16, 64]);  view_829 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_616: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_830, [0, 2, 1, 3]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_169: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_616, memory_format = torch.contiguous_format);  permute_616 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_169, permute_default_60, permute_default_61, permute_default_62, None, getitem_170, getitem_171, getitem_172, getitem_173, 0.1, [True, True, True, False], scale = 0.125);  clone_169 = permute_default_60 = permute_default_61 = permute_default_62 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = None
        getitem_174: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[0]
        getitem_175: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[1]
        getitem_176: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_10[2];  _scaled_dot_product_efficient_attention_backward_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_65: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_174, [0, 2, 1, 3]);  getitem_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_64: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_175, [0, 2, 1, 3]);  getitem_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_63: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_176, [0, 2, 1, 3]);  getitem_176 = None
        clone_171: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_63, memory_format = torch.contiguous_format);  permute_default_63 = None
        view_837: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_171, [16, 512, 1024]);  clone_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_838: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_837, [8192, 1024]);  view_837 = None
        permute_147: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_221, [1, 0]);  primals_221 = None
        permute_623: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        mm_130: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_838, permute_623);  permute_623 = None
        permute_624: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_838, [1, 0])
        mm_131: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_624, view_286);  permute_624 = None
        sum_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_838, [0], True);  view_838 = None
        view_839: "f32[1024]" = torch.ops.aten.reshape.default(sum_197, [1024]);  sum_197 = None
        view_840: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_130, [16, 512, 1024]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_841: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_64, [16, 512, 1024]);  permute_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_172: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_841, memory_format = torch.contiguous_format);  view_841 = None
        view_842: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_172, [8192, 1024]);  clone_172 = None
        permute_145: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        permute_628: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        mm_132: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_842, permute_628);  permute_628 = None
        permute_629: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_842, [1, 0])
        mm_133: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_629, view_286);  permute_629 = None
        sum_198: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
        view_843: "f32[1024]" = torch.ops.aten.reshape.default(sum_198, [1024]);  sum_198 = None
        view_844: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_132, [16, 512, 1024]);  mm_132 = None
        add_265: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_840, view_844);  view_840 = view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_173: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_65, memory_format = torch.contiguous_format);  permute_default_65 = None
        view_845: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_173, [16, 512, 1024]);  clone_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_846: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_845, [8192, 1024]);  view_845 = None
        permute_143: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        permute_633: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        mm_134: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_846, permute_633);  permute_633 = None
        permute_634: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_846, [1, 0])
        mm_135: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_634, view_286);  permute_634 = view_286 = None
        sum_199: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        view_847: "f32[1024]" = torch.ops.aten.reshape.default(sum_199, [1024]);  sum_199 = None
        view_848: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_134, [16, 512, 1024]);  mm_134 = None
        add_266: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_265, view_848);  add_265 = view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_647: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, primals_215);  primals_215 = None
        mul_648: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_647, 1024)
        sum_200: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_647, [2], True)
        mul_649: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_647, mul_172);  mul_647 = None
        sum_201: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_649, [2], True);  mul_649 = None
        mul_650: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_172, sum_201);  sum_201 = None
        sub_148: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_648, sum_200);  mul_648 = sum_200 = None
        sub_149: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_148, mul_650);  sub_148 = mul_650 = None
        mul_651: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_84, sub_149);  div_84 = sub_149 = None
        mul_652: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_266, mul_172);  mul_172 = None
        sum_202: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_652, [0, 1]);  mul_652 = None
        sum_203: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None
        add_267: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_264, mul_651);  add_264 = mul_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_653: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_654: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_267, mul_653);  mul_653 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_849: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_654, [8192, 1024]);  mul_654 = None
        permute_142: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
        permute_637: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_136: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_849, permute_637);  permute_637 = None
        permute_638: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_137: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_638, view_284);  permute_638 = view_284 = None
        sum_204: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        view_850: "f32[1024]" = torch.ops.aten.reshape.default(sum_204, [1024]);  sum_204 = None
        view_851: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_136, [16, 512, 4096]);  mm_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_283: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_76, [16, 512, 4096]);  addmm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_168: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, 0.7071067811865476)
        erf_12: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_168);  mul_168 = None
        add_104: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_656: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_104, 0.5);  add_104 = None
        mul_657: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, view_283)
        mul_658: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_657, -0.5);  mul_657 = None
        exp_38: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_658);  mul_658 = None
        mul_659: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_38, 0.3989422804014327);  exp_38 = None
        mul_660: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_283, mul_659);  view_283 = mul_659 = None
        add_269: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_656, mul_660);  mul_656 = mul_660 = None
        mul_661: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_851, add_269);  view_851 = add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_852: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_661, [8192, 4096]);  mul_661 = None
        permute_141: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        permute_641: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        mm_138: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_852, permute_641);  permute_641 = None
        permute_642: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_139: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_642, view_282);  permute_642 = view_282 = None
        sum_205: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
        view_853: "f32[4096]" = torch.ops.aten.reshape.default(sum_205, [4096]);  sum_205 = None
        view_854: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_138, [16, 512, 1024]);  mm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_663: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, primals_209);  primals_209 = None
        mul_664: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_663, 1024)
        sum_206: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_663, [2], True)
        mul_665: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_663, mul_165);  mul_663 = None
        sum_207: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_665, [2], True);  mul_665 = None
        mul_666: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_165, sum_207);  sum_207 = None
        sub_151: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_664, sum_206);  mul_664 = sum_206 = None
        sub_152: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_151, mul_666);  sub_151 = mul_666 = None
        mul_667: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_85, sub_152);  div_85 = sub_152 = None
        mul_668: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_854, mul_165);  mul_165 = None
        sum_208: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_668, [0, 1]);  mul_668 = None
        sum_209: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_854, [0, 1]);  view_854 = None
        add_270: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_267, mul_667);  add_267 = mul_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_35: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_669: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_670: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_270, mul_669);  mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_855: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_670, [8192, 1024]);  mul_670 = None
        permute_140: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
        permute_645: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_140: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_855, permute_645);  permute_645 = None
        permute_646: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_855, [1, 0])
        mm_141: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_646, view_280);  permute_646 = view_280 = None
        sum_210: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        view_856: "f32[1024]" = torch.ops.aten.reshape.default(sum_210, [1024]);  sum_210 = None
        view_857: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_140, [16, 512, 1024]);  mm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_858: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_857, [16, 512, 16, 64]);  view_857 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_649: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_176: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_649, memory_format = torch.contiguous_format);  permute_649 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_176, permute_default_66, permute_default_67, permute_default_68, None, getitem_177, getitem_178, getitem_179, getitem_180, 0.1, [True, True, True, False], scale = 0.125);  clone_176 = permute_default_66 = permute_default_67 = permute_default_68 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = None
        getitem_181: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[0]
        getitem_182: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[1]
        getitem_183: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_11[2];  _scaled_dot_product_efficient_attention_backward_default_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_71: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3]);  getitem_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_70: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_69: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_183, [0, 2, 1, 3]);  getitem_183 = None
        clone_178: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_69, memory_format = torch.contiguous_format);  permute_default_69 = None
        view_865: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_178, [16, 512, 1024]);  clone_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_866: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_865, [8192, 1024]);  view_865 = None
        permute_136: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        permute_656: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mm_142: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_866, permute_656);  permute_656 = None
        permute_657: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_866, [1, 0])
        mm_143: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_657, view_264);  permute_657 = None
        sum_212: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_866, [0], True);  view_866 = None
        view_867: "f32[1024]" = torch.ops.aten.reshape.default(sum_212, [1024]);  sum_212 = None
        view_868: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_142, [16, 512, 1024]);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_869: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_70, [16, 512, 1024]);  permute_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_179: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_869, memory_format = torch.contiguous_format);  view_869 = None
        view_870: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_179, [8192, 1024]);  clone_179 = None
        permute_134: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_203, [1, 0]);  primals_203 = None
        permute_661: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        mm_144: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_870, permute_661);  permute_661 = None
        permute_662: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_870, [1, 0])
        mm_145: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_662, view_264);  permute_662 = None
        sum_213: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_870, [0], True);  view_870 = None
        view_871: "f32[1024]" = torch.ops.aten.reshape.default(sum_213, [1024]);  sum_213 = None
        view_872: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_144, [16, 512, 1024]);  mm_144 = None
        add_271: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_868, view_872);  view_868 = view_872 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_180: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_71, memory_format = torch.contiguous_format);  permute_default_71 = None
        view_873: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_180, [16, 512, 1024]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_874: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_873, [8192, 1024]);  view_873 = None
        permute_132: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        permute_666: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_146: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_874, permute_666);  permute_666 = None
        permute_667: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_874, [1, 0])
        mm_147: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_667, view_264);  permute_667 = view_264 = None
        sum_214: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
        view_875: "f32[1024]" = torch.ops.aten.reshape.default(sum_214, [1024]);  sum_214 = None
        view_876: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_146, [16, 512, 1024]);  mm_146 = None
        add_272: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_271, view_876);  add_271 = view_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_675: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, primals_199);  primals_199 = None
        mul_676: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_675, 1024)
        sum_215: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_675, [2], True)
        mul_677: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_675, mul_159);  mul_675 = None
        sum_216: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_677, [2], True);  mul_677 = None
        mul_678: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_159, sum_216);  sum_216 = None
        sub_154: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_676, sum_215);  mul_676 = sum_215 = None
        sub_155: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_154, mul_678);  sub_154 = mul_678 = None
        mul_679: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_87, sub_155);  div_87 = sub_155 = None
        mul_680: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_272, mul_159);  mul_159 = None
        sum_217: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_680, [0, 1]);  mul_680 = None
        sum_218: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None
        add_273: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_270, mul_679);  add_270 = mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_37: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_681: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_682: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_273, mul_681);  mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_877: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_682, [8192, 1024]);  mul_682 = None
        permute_131: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_197, [1, 0]);  primals_197 = None
        permute_670: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        mm_148: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_877, permute_670);  permute_670 = None
        permute_671: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_149: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_671, view_262);  permute_671 = view_262 = None
        sum_219: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
        view_878: "f32[1024]" = torch.ops.aten.reshape.default(sum_219, [1024]);  sum_219 = None
        view_879: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_148, [16, 512, 4096]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_261: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_70, [16, 512, 4096]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_155: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, 0.7071067811865476)
        erf_11: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_155);  mul_155 = None
        add_96: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_684: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_96, 0.5);  add_96 = None
        mul_685: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, view_261)
        mul_686: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_685, -0.5);  mul_685 = None
        exp_39: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_686);  mul_686 = None
        mul_687: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_39, 0.3989422804014327);  exp_39 = None
        mul_688: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_261, mul_687);  view_261 = mul_687 = None
        add_275: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_684, mul_688);  mul_684 = mul_688 = None
        mul_689: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_879, add_275);  view_879 = add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_880: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_689, [8192, 4096]);  mul_689 = None
        permute_130: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        permute_674: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_150: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_880, permute_674);  permute_674 = None
        permute_675: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_880, [1, 0])
        mm_151: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_675, view_260);  permute_675 = view_260 = None
        sum_220: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        view_881: "f32[4096]" = torch.ops.aten.reshape.default(sum_220, [4096]);  sum_220 = None
        view_882: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_150, [16, 512, 1024]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_691: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, primals_193);  primals_193 = None
        mul_692: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_691, 1024)
        sum_221: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_691, [2], True)
        mul_693: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_691, mul_152);  mul_691 = None
        sum_222: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_693, [2], True);  mul_693 = None
        mul_694: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_152, sum_222);  sum_222 = None
        sub_157: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_692, sum_221);  mul_692 = sum_221 = None
        sub_158: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_157, mul_694);  sub_157 = mul_694 = None
        mul_695: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_88, sub_158);  div_88 = sub_158 = None
        mul_696: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_882, mul_152);  mul_152 = None
        sum_223: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_696, [0, 1]);  mul_696 = None
        sum_224: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_882, [0, 1]);  view_882 = None
        add_276: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_273, mul_695);  add_273 = mul_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_697: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_698: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_276, mul_697);  mul_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_883: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_698, [8192, 1024]);  mul_698 = None
        permute_129: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_191, [1, 0]);  primals_191 = None
        permute_678: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_129, [1, 0]);  permute_129 = None
        mm_152: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_883, permute_678);  permute_678 = None
        permute_679: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_883, [1, 0])
        mm_153: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_679, view_258);  permute_679 = view_258 = None
        sum_225: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        view_884: "f32[1024]" = torch.ops.aten.reshape.default(sum_225, [1024]);  sum_225 = None
        view_885: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_152, [16, 512, 1024]);  mm_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_886: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_885, [16, 512, 16, 64]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_682: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_183: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_682, memory_format = torch.contiguous_format);  permute_682 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_12 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_183, permute_default_72, permute_default_73, permute_default_74, None, getitem_184, getitem_185, getitem_186, getitem_187, 0.1, [True, True, True, False], scale = 0.125);  clone_183 = permute_default_72 = permute_default_73 = permute_default_74 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = None
        getitem_188: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_12[0]
        getitem_189: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_12[1]
        getitem_190: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_12[2];  _scaled_dot_product_efficient_attention_backward_default_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_77: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_188, [0, 2, 1, 3]);  getitem_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_76: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_189, [0, 2, 1, 3]);  getitem_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_75: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_190, [0, 2, 1, 3]);  getitem_190 = None
        clone_185: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_75, memory_format = torch.contiguous_format);  permute_default_75 = None
        view_893: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_185, [16, 512, 1024]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_894: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_893, [8192, 1024]);  view_893 = None
        permute_125: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_689: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        mm_154: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_894, permute_689);  permute_689 = None
        permute_690: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_894, [1, 0])
        mm_155: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_690, view_242);  permute_690 = None
        sum_227: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_894, [0], True);  view_894 = None
        view_895: "f32[1024]" = torch.ops.aten.reshape.default(sum_227, [1024]);  sum_227 = None
        view_896: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_154, [16, 512, 1024]);  mm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_897: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_76, [16, 512, 1024]);  permute_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_186: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_897, memory_format = torch.contiguous_format);  view_897 = None
        view_898: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_186, [8192, 1024]);  clone_186 = None
        permute_123: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        permute_694: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_156: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_898, permute_694);  permute_694 = None
        permute_695: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_898, [1, 0])
        mm_157: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_695, view_242);  permute_695 = None
        sum_228: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_898, [0], True);  view_898 = None
        view_899: "f32[1024]" = torch.ops.aten.reshape.default(sum_228, [1024]);  sum_228 = None
        view_900: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_156, [16, 512, 1024]);  mm_156 = None
        add_277: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_896, view_900);  view_896 = view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_187: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_77, memory_format = torch.contiguous_format);  permute_default_77 = None
        view_901: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_187, [16, 512, 1024]);  clone_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_902: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_901, [8192, 1024]);  view_901 = None
        permute_121: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_699: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_158: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_902, permute_699);  permute_699 = None
        permute_700: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_159: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_700, view_242);  permute_700 = view_242 = None
        sum_229: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
        view_903: "f32[1024]" = torch.ops.aten.reshape.default(sum_229, [1024]);  sum_229 = None
        view_904: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_158, [16, 512, 1024]);  mm_158 = None
        add_278: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_277, view_904);  add_277 = view_904 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_703: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, primals_183);  primals_183 = None
        mul_704: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_703, 1024)
        sum_230: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_703, [2], True)
        mul_705: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_703, mul_146);  mul_703 = None
        sum_231: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_705, [2], True);  mul_705 = None
        mul_706: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_146, sum_231);  sum_231 = None
        sub_160: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_704, sum_230);  mul_704 = sum_230 = None
        sub_161: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_160, mul_706);  sub_160 = mul_706 = None
        mul_707: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_90, sub_161);  div_90 = sub_161 = None
        mul_708: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_278, mul_146);  mul_146 = None
        sum_232: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_708, [0, 1]);  mul_708 = None
        sum_233: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None
        add_279: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_276, mul_707);  add_276 = mul_707 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_40: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_709: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_710: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_279, mul_709);  mul_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_905: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_710, [8192, 1024]);  mul_710 = None
        permute_120: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        permute_703: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        mm_160: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_905, permute_703);  permute_703 = None
        permute_704: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_161: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_704, view_240);  permute_704 = view_240 = None
        sum_234: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
        view_906: "f32[1024]" = torch.ops.aten.reshape.default(sum_234, [1024]);  sum_234 = None
        view_907: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_160, [16, 512, 4096]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_239: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_64, [16, 512, 4096]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_142: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, 0.7071067811865476)
        erf_10: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_142);  mul_142 = None
        add_88: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_712: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_88, 0.5);  add_88 = None
        mul_713: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, view_239)
        mul_714: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_713, -0.5);  mul_713 = None
        exp_40: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_714);  mul_714 = None
        mul_715: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_40, 0.3989422804014327);  exp_40 = None
        mul_716: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_239, mul_715);  view_239 = mul_715 = None
        add_281: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_712, mul_716);  mul_712 = mul_716 = None
        mul_717: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_907, add_281);  view_907 = add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_908: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_717, [8192, 4096]);  mul_717 = None
        permute_119: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        permute_707: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_162: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_908, permute_707);  permute_707 = None
        permute_708: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_908, [1, 0])
        mm_163: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_708, view_238);  permute_708 = view_238 = None
        sum_235: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
        view_909: "f32[4096]" = torch.ops.aten.reshape.default(sum_235, [4096]);  sum_235 = None
        view_910: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_162, [16, 512, 1024]);  mm_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_719: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, primals_177);  primals_177 = None
        mul_720: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_719, 1024)
        sum_236: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_719, [2], True)
        mul_721: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_719, mul_139);  mul_719 = None
        sum_237: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_721, [2], True);  mul_721 = None
        mul_722: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_139, sum_237);  sum_237 = None
        sub_163: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_720, sum_236);  mul_720 = sum_236 = None
        sub_164: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_163, mul_722);  sub_163 = mul_722 = None
        mul_723: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_91, sub_164);  div_91 = sub_164 = None
        mul_724: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_910, mul_139);  mul_139 = None
        sum_238: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_724, [0, 1]);  mul_724 = None
        sum_239: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_910, [0, 1]);  view_910 = None
        add_282: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_279, mul_723);  add_279 = mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_41: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_725: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_726: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_282, mul_725);  mul_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_911: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_726, [8192, 1024]);  mul_726 = None
        permute_118: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_711: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_164: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_911, permute_711);  permute_711 = None
        permute_712: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_911, [1, 0])
        mm_165: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_712, view_236);  permute_712 = view_236 = None
        sum_240: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
        view_912: "f32[1024]" = torch.ops.aten.reshape.default(sum_240, [1024]);  sum_240 = None
        view_913: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_164, [16, 512, 1024]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_914: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_913, [16, 512, 16, 64]);  view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_715: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_914, [0, 2, 1, 3]);  view_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_190: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_715, memory_format = torch.contiguous_format);  permute_715 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_13 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_190, permute_default_78, permute_default_79, permute_default_80, None, getitem_191, getitem_192, getitem_193, getitem_194, 0.1, [True, True, True, False], scale = 0.125);  clone_190 = permute_default_78 = permute_default_79 = permute_default_80 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = None
        getitem_195: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_13[0]
        getitem_196: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_13[1]
        getitem_197: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_13[2];  _scaled_dot_product_efficient_attention_backward_default_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_83: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_195, [0, 2, 1, 3]);  getitem_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_82: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_196, [0, 2, 1, 3]);  getitem_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_81: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_197, [0, 2, 1, 3]);  getitem_197 = None
        clone_192: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_81, memory_format = torch.contiguous_format);  permute_default_81 = None
        view_921: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_192, [16, 512, 1024]);  clone_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_922: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_921, [8192, 1024]);  view_921 = None
        permute_114: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_173, [1, 0]);  primals_173 = None
        permute_722: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        mm_166: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_922, permute_722);  permute_722 = None
        permute_723: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_922, [1, 0])
        mm_167: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_723, view_220);  permute_723 = None
        sum_242: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
        view_923: "f32[1024]" = torch.ops.aten.reshape.default(sum_242, [1024]);  sum_242 = None
        view_924: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_166, [16, 512, 1024]);  mm_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_925: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_82, [16, 512, 1024]);  permute_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_193: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_925, memory_format = torch.contiguous_format);  view_925 = None
        view_926: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_193, [8192, 1024]);  clone_193 = None
        permute_112: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_727: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        mm_168: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_926, permute_727);  permute_727 = None
        permute_728: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_926, [1, 0])
        mm_169: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_728, view_220);  permute_728 = None
        sum_243: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_926, [0], True);  view_926 = None
        view_927: "f32[1024]" = torch.ops.aten.reshape.default(sum_243, [1024]);  sum_243 = None
        view_928: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_168, [16, 512, 1024]);  mm_168 = None
        add_283: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_924, view_928);  view_924 = view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_194: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_83, memory_format = torch.contiguous_format);  permute_default_83 = None
        view_929: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_194, [16, 512, 1024]);  clone_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_930: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_929, [8192, 1024]);  view_929 = None
        permute_110: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_732: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_110, [1, 0]);  permute_110 = None
        mm_170: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_930, permute_732);  permute_732 = None
        permute_733: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_171: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_733, view_220);  permute_733 = view_220 = None
        sum_244: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        view_931: "f32[1024]" = torch.ops.aten.reshape.default(sum_244, [1024]);  sum_244 = None
        view_932: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_170, [16, 512, 1024]);  mm_170 = None
        add_284: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_283, view_932);  add_283 = view_932 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_731: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, primals_167);  primals_167 = None
        mul_732: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_731, 1024)
        sum_245: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_731, [2], True)
        mul_733: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_731, mul_133);  mul_731 = None
        sum_246: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
        mul_734: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_133, sum_246);  sum_246 = None
        sub_166: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_732, sum_245);  mul_732 = sum_245 = None
        sub_167: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_166, mul_734);  sub_166 = mul_734 = None
        mul_735: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_93, sub_167);  div_93 = sub_167 = None
        mul_736: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_284, mul_133);  mul_133 = None
        sum_247: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_736, [0, 1]);  mul_736 = None
        sum_248: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None
        add_285: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_282, mul_735);  add_282 = mul_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_43: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_737: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_738: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_285, mul_737);  mul_737 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_933: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_738, [8192, 1024]);  mul_738 = None
        permute_109: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        permute_736: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_172: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_933, permute_736);  permute_736 = None
        permute_737: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_933, [1, 0])
        mm_173: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_737, view_218);  permute_737 = view_218 = None
        sum_249: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
        view_934: "f32[1024]" = torch.ops.aten.reshape.default(sum_249, [1024]);  sum_249 = None
        view_935: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_172, [16, 512, 4096]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_217: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_58, [16, 512, 4096]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_129: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, 0.7071067811865476)
        erf_9: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_129);  mul_129 = None
        add_80: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_740: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_80, 0.5);  add_80 = None
        mul_741: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, view_217)
        mul_742: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_741, -0.5);  mul_741 = None
        exp_41: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_742);  mul_742 = None
        mul_743: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_41, 0.3989422804014327);  exp_41 = None
        mul_744: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_217, mul_743);  view_217 = mul_743 = None
        add_287: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_740, mul_744);  mul_740 = mul_744 = None
        mul_745: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_935, add_287);  view_935 = add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_936: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_745, [8192, 4096]);  mul_745 = None
        permute_108: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        permute_740: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_108, [1, 0]);  permute_108 = None
        mm_174: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_936, permute_740);  permute_740 = None
        permute_741: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_936, [1, 0])
        mm_175: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_741, view_216);  permute_741 = view_216 = None
        sum_250: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
        view_937: "f32[4096]" = torch.ops.aten.reshape.default(sum_250, [4096]);  sum_250 = None
        view_938: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_174, [16, 512, 1024]);  mm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_747: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, primals_161);  primals_161 = None
        mul_748: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_747, 1024)
        sum_251: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_747, [2], True)
        mul_749: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_747, mul_126);  mul_747 = None
        sum_252: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_749, [2], True);  mul_749 = None
        mul_750: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_126, sum_252);  sum_252 = None
        sub_169: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_748, sum_251);  mul_748 = sum_251 = None
        sub_170: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_169, mul_750);  sub_169 = mul_750 = None
        mul_751: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_94, sub_170);  div_94 = sub_170 = None
        mul_752: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_938, mul_126);  mul_126 = None
        sum_253: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_752, [0, 1]);  mul_752 = None
        sum_254: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_938, [0, 1]);  view_938 = None
        add_288: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_285, mul_751);  add_285 = mul_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_753: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_754: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_288, mul_753);  mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_939: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_754, [8192, 1024]);  mul_754 = None
        permute_107: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        permute_744: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_176: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_939, permute_744);  permute_744 = None
        permute_745: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_939, [1, 0])
        mm_177: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_745, view_214);  permute_745 = view_214 = None
        sum_255: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
        view_940: "f32[1024]" = torch.ops.aten.reshape.default(sum_255, [1024]);  sum_255 = None
        view_941: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_176, [16, 512, 1024]);  mm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_942: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_941, [16, 512, 16, 64]);  view_941 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_748: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_942, [0, 2, 1, 3]);  view_942 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_197: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_748, memory_format = torch.contiguous_format);  permute_748 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_14 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_197, permute_default_84, permute_default_85, permute_default_86, None, getitem_198, getitem_199, getitem_200, getitem_201, 0.1, [True, True, True, False], scale = 0.125);  clone_197 = permute_default_84 = permute_default_85 = permute_default_86 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = None
        getitem_202: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_14[0]
        getitem_203: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_14[1]
        getitem_204: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_14[2];  _scaled_dot_product_efficient_attention_backward_default_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_89: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_88: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3]);  getitem_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_87: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_204, [0, 2, 1, 3]);  getitem_204 = None
        clone_199: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_87, memory_format = torch.contiguous_format);  permute_default_87 = None
        view_949: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_199, [16, 512, 1024]);  clone_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_950: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_949, [8192, 1024]);  view_949 = None
        permute_103: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_755: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        mm_178: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_950, permute_755);  permute_755 = None
        permute_756: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_950, [1, 0])
        mm_179: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_756, view_198);  permute_756 = None
        sum_257: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_950, [0], True);  view_950 = None
        view_951: "f32[1024]" = torch.ops.aten.reshape.default(sum_257, [1024]);  sum_257 = None
        view_952: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_178, [16, 512, 1024]);  mm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_953: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_88, [16, 512, 1024]);  permute_default_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_200: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_953, memory_format = torch.contiguous_format);  view_953 = None
        view_954: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_200, [8192, 1024]);  clone_200 = None
        permute_101: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_760: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_180: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_954, permute_760);  permute_760 = None
        permute_761: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_954, [1, 0])
        mm_181: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_761, view_198);  permute_761 = None
        sum_258: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_954, [0], True);  view_954 = None
        view_955: "f32[1024]" = torch.ops.aten.reshape.default(sum_258, [1024]);  sum_258 = None
        view_956: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_180, [16, 512, 1024]);  mm_180 = None
        add_289: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_952, view_956);  view_952 = view_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_201: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_89, memory_format = torch.contiguous_format);  permute_default_89 = None
        view_957: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_201, [16, 512, 1024]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_958: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_957, [8192, 1024]);  view_957 = None
        permute_99: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_765: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_182: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_958, permute_765);  permute_765 = None
        permute_766: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_183: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_766, view_198);  permute_766 = view_198 = None
        sum_259: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
        view_959: "f32[1024]" = torch.ops.aten.reshape.default(sum_259, [1024]);  sum_259 = None
        view_960: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_182, [16, 512, 1024]);  mm_182 = None
        add_290: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_289, view_960);  add_289 = view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_759: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, primals_151);  primals_151 = None
        mul_760: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_759, 1024)
        sum_260: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_759, [2], True)
        mul_761: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_759, mul_120);  mul_759 = None
        sum_261: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True);  mul_761 = None
        mul_762: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_120, sum_261);  sum_261 = None
        sub_172: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_760, sum_260);  mul_760 = sum_260 = None
        sub_173: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_172, mul_762);  sub_172 = mul_762 = None
        mul_763: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_96, sub_173);  div_96 = sub_173 = None
        mul_764: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_290, mul_120);  mul_120 = None
        sum_262: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_764, [0, 1]);  mul_764 = None
        sum_263: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None
        add_291: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_288, mul_763);  add_288 = mul_763 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_46: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_765: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_766: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_291, mul_765);  mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_961: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_766, [8192, 1024]);  mul_766 = None
        permute_98: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_149, [1, 0]);  primals_149 = None
        permute_769: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        mm_184: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_961, permute_769);  permute_769 = None
        permute_770: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_961, [1, 0])
        mm_185: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_770, view_196);  permute_770 = view_196 = None
        sum_264: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
        view_962: "f32[1024]" = torch.ops.aten.reshape.default(sum_264, [1024]);  sum_264 = None
        view_963: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_184, [16, 512, 4096]);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_195: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_52, [16, 512, 4096]);  addmm_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_116: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, 0.7071067811865476)
        erf_8: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_116);  mul_116 = None
        add_72: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_768: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_72, 0.5);  add_72 = None
        mul_769: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, view_195)
        mul_770: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_769, -0.5);  mul_769 = None
        exp_42: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_770);  mul_770 = None
        mul_771: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_42, 0.3989422804014327);  exp_42 = None
        mul_772: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_195, mul_771);  view_195 = mul_771 = None
        add_293: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_768, mul_772);  mul_768 = mul_772 = None
        mul_773: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_963, add_293);  view_963 = add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_964: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_773, [8192, 4096]);  mul_773 = None
        permute_97: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_773: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_186: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_964, permute_773);  permute_773 = None
        permute_774: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_964, [1, 0])
        mm_187: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_774, view_194);  permute_774 = view_194 = None
        sum_265: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        view_965: "f32[4096]" = torch.ops.aten.reshape.default(sum_265, [4096]);  sum_265 = None
        view_966: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_186, [16, 512, 1024]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_775: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, primals_145);  primals_145 = None
        mul_776: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_775, 1024)
        sum_266: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_775, [2], True)
        mul_777: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_775, mul_113);  mul_775 = None
        sum_267: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_777, [2], True);  mul_777 = None
        mul_778: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_113, sum_267);  sum_267 = None
        sub_175: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_776, sum_266);  mul_776 = sum_266 = None
        sub_176: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_175, mul_778);  sub_175 = mul_778 = None
        mul_779: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_97, sub_176);  div_97 = sub_176 = None
        mul_780: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_966, mul_113);  mul_113 = None
        sum_268: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 1]);  mul_780 = None
        sum_269: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_966, [0, 1]);  view_966 = None
        add_294: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_291, mul_779);  add_291 = mul_779 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_47: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_781: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_782: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_294, mul_781);  mul_781 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_967: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_782, [8192, 1024]);  mul_782 = None
        permute_96: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_777: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_188: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_967, permute_777);  permute_777 = None
        permute_778: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_967, [1, 0])
        mm_189: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_778, view_192);  permute_778 = view_192 = None
        sum_270: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
        view_968: "f32[1024]" = torch.ops.aten.reshape.default(sum_270, [1024]);  sum_270 = None
        view_969: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_188, [16, 512, 1024]);  mm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_970: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_969, [16, 512, 16, 64]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_781: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_204: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_781, memory_format = torch.contiguous_format);  permute_781 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_15 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_204, permute_default_90, permute_default_91, permute_default_92, None, getitem_205, getitem_206, getitem_207, getitem_208, 0.1, [True, True, True, False], scale = 0.125);  clone_204 = permute_default_90 = permute_default_91 = permute_default_92 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = None
        getitem_209: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_15[0]
        getitem_210: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_15[1]
        getitem_211: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_15[2];  _scaled_dot_product_efficient_attention_backward_default_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_95: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_209, [0, 2, 1, 3]);  getitem_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_94: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_210, [0, 2, 1, 3]);  getitem_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_93: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_211, [0, 2, 1, 3]);  getitem_211 = None
        clone_206: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_93, memory_format = torch.contiguous_format);  permute_default_93 = None
        view_977: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_206, [16, 512, 1024]);  clone_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_978: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_977, [8192, 1024]);  view_977 = None
        permute_92: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_788: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        mm_190: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_978, permute_788);  permute_788 = None
        permute_789: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_978, [1, 0])
        mm_191: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_789, view_176);  permute_789 = None
        sum_272: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_978, [0], True);  view_978 = None
        view_979: "f32[1024]" = torch.ops.aten.reshape.default(sum_272, [1024]);  sum_272 = None
        view_980: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_190, [16, 512, 1024]);  mm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_981: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_94, [16, 512, 1024]);  permute_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_207: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_981, memory_format = torch.contiguous_format);  view_981 = None
        view_982: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_207, [8192, 1024]);  clone_207 = None
        permute_90: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_793: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_90, [1, 0]);  permute_90 = None
        mm_192: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_982, permute_793);  permute_793 = None
        permute_794: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_982, [1, 0])
        mm_193: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_794, view_176);  permute_794 = None
        sum_273: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
        view_983: "f32[1024]" = torch.ops.aten.reshape.default(sum_273, [1024]);  sum_273 = None
        view_984: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_192, [16, 512, 1024]);  mm_192 = None
        add_295: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_980, view_984);  view_980 = view_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_208: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_95, memory_format = torch.contiguous_format);  permute_default_95 = None
        view_985: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_208, [16, 512, 1024]);  clone_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_986: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_985, [8192, 1024]);  view_985 = None
        permute_88: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_798: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_88, [1, 0]);  permute_88 = None
        mm_194: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_986, permute_798);  permute_798 = None
        permute_799: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_986, [1, 0])
        mm_195: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_799, view_176);  permute_799 = view_176 = None
        sum_274: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
        view_987: "f32[1024]" = torch.ops.aten.reshape.default(sum_274, [1024]);  sum_274 = None
        view_988: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_194, [16, 512, 1024]);  mm_194 = None
        add_296: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_295, view_988);  add_295 = view_988 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_787: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, primals_135);  primals_135 = None
        mul_788: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_787, 1024)
        sum_275: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_787, [2], True)
        mul_789: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_787, mul_107);  mul_787 = None
        sum_276: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_789, [2], True);  mul_789 = None
        mul_790: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_107, sum_276);  sum_276 = None
        sub_178: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_788, sum_275);  mul_788 = sum_275 = None
        sub_179: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_178, mul_790);  sub_178 = mul_790 = None
        mul_791: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_99, sub_179);  div_99 = sub_179 = None
        mul_792: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_296, mul_107);  mul_107 = None
        sum_277: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_792, [0, 1]);  mul_792 = None
        sum_278: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None
        add_297: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_294, mul_791);  add_294 = mul_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_49: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_793: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_794: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_297, mul_793);  mul_793 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_989: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_794, [8192, 1024]);  mul_794 = None
        permute_87: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_802: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_196: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_989, permute_802);  permute_802 = None
        permute_803: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_989, [1, 0])
        mm_197: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_803, view_174);  permute_803 = view_174 = None
        sum_279: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
        view_990: "f32[1024]" = torch.ops.aten.reshape.default(sum_279, [1024]);  sum_279 = None
        view_991: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_196, [16, 512, 4096]);  mm_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_173: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_46, [16, 512, 4096]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_103: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, 0.7071067811865476)
        erf_7: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_103);  mul_103 = None
        add_64: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_796: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_64, 0.5);  add_64 = None
        mul_797: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, view_173)
        mul_798: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_797, -0.5);  mul_797 = None
        exp_43: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_798);  mul_798 = None
        mul_799: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_43, 0.3989422804014327);  exp_43 = None
        mul_800: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_173, mul_799);  view_173 = mul_799 = None
        add_299: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_796, mul_800);  mul_796 = mul_800 = None
        mul_801: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_991, add_299);  view_991 = add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_992: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_801, [8192, 4096]);  mul_801 = None
        permute_86: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_131, [1, 0]);  primals_131 = None
        permute_806: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_86, [1, 0]);  permute_86 = None
        mm_198: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_992, permute_806);  permute_806 = None
        permute_807: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_992, [1, 0])
        mm_199: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_807, view_172);  permute_807 = view_172 = None
        sum_280: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
        view_993: "f32[4096]" = torch.ops.aten.reshape.default(sum_280, [4096]);  sum_280 = None
        view_994: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_198, [16, 512, 1024]);  mm_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_803: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, primals_129);  primals_129 = None
        mul_804: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_803, 1024)
        sum_281: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_803, [2], True)
        mul_805: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_803, mul_100);  mul_803 = None
        sum_282: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_805, [2], True);  mul_805 = None
        mul_806: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_100, sum_282);  sum_282 = None
        sub_181: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_804, sum_281);  mul_804 = sum_281 = None
        sub_182: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_181, mul_806);  sub_181 = mul_806 = None
        mul_807: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_100, sub_182);  div_100 = sub_182 = None
        mul_808: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_994, mul_100);  mul_100 = None
        sum_283: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_808, [0, 1]);  mul_808 = None
        sum_284: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_994, [0, 1]);  view_994 = None
        add_300: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_297, mul_807);  add_297 = mul_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_809: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_810: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_300, mul_809);  mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_995: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_810, [8192, 1024]);  mul_810 = None
        permute_85: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_810: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_200: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_995, permute_810);  permute_810 = None
        permute_811: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_995, [1, 0])
        mm_201: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_811, view_170);  permute_811 = view_170 = None
        sum_285: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
        view_996: "f32[1024]" = torch.ops.aten.reshape.default(sum_285, [1024]);  sum_285 = None
        view_997: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_200, [16, 512, 1024]);  mm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_998: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_997, [16, 512, 16, 64]);  view_997 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_814: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_998, [0, 2, 1, 3]);  view_998 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_211: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_814, memory_format = torch.contiguous_format);  permute_814 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_16 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_211, permute_default_96, permute_default_97, permute_default_98, None, getitem_212, getitem_213, getitem_214, getitem_215, 0.1, [True, True, True, False], scale = 0.125);  clone_211 = permute_default_96 = permute_default_97 = permute_default_98 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = None
        getitem_216: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_16[0]
        getitem_217: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_16[1]
        getitem_218: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_16[2];  _scaled_dot_product_efficient_attention_backward_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_101: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_216, [0, 2, 1, 3]);  getitem_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_100: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_217, [0, 2, 1, 3]);  getitem_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_99: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_218, [0, 2, 1, 3]);  getitem_218 = None
        clone_213: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_99, memory_format = torch.contiguous_format);  permute_default_99 = None
        view_1005: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_213, [16, 512, 1024]);  clone_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1006: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1005, [8192, 1024]);  view_1005 = None
        permute_81: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_821: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_202: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1006, permute_821);  permute_821 = None
        permute_822: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1006, [1, 0])
        mm_203: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_822, view_154);  permute_822 = None
        sum_287: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1006, [0], True);  view_1006 = None
        view_1007: "f32[1024]" = torch.ops.aten.reshape.default(sum_287, [1024]);  sum_287 = None
        view_1008: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_202, [16, 512, 1024]);  mm_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1009: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_100, [16, 512, 1024]);  permute_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_214: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1009, memory_format = torch.contiguous_format);  view_1009 = None
        view_1010: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_214, [8192, 1024]);  clone_214 = None
        permute_79: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_826: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        mm_204: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1010, permute_826);  permute_826 = None
        permute_827: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1010, [1, 0])
        mm_205: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_827, view_154);  permute_827 = None
        sum_288: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1010, [0], True);  view_1010 = None
        view_1011: "f32[1024]" = torch.ops.aten.reshape.default(sum_288, [1024]);  sum_288 = None
        view_1012: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_204, [16, 512, 1024]);  mm_204 = None
        add_301: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1008, view_1012);  view_1008 = view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_215: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_101, memory_format = torch.contiguous_format);  permute_default_101 = None
        view_1013: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_215, [16, 512, 1024]);  clone_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1014: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1013, [8192, 1024]);  view_1013 = None
        permute_77: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        permute_831: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_206: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1014, permute_831);  permute_831 = None
        permute_832: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        mm_207: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_832, view_154);  permute_832 = view_154 = None
        sum_289: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        view_1015: "f32[1024]" = torch.ops.aten.reshape.default(sum_289, [1024]);  sum_289 = None
        view_1016: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_206, [16, 512, 1024]);  mm_206 = None
        add_302: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_301, view_1016);  add_301 = view_1016 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_815: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, primals_119);  primals_119 = None
        mul_816: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_815, 1024)
        sum_290: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [2], True)
        mul_817: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_815, mul_94);  mul_815 = None
        sum_291: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True);  mul_817 = None
        mul_818: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_94, sum_291);  sum_291 = None
        sub_184: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_816, sum_290);  mul_816 = sum_290 = None
        sub_185: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_184, mul_818);  sub_184 = mul_818 = None
        mul_819: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_102, sub_185);  div_102 = sub_185 = None
        mul_820: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_302, mul_94);  mul_94 = None
        sum_292: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 1]);  mul_820 = None
        sum_293: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None
        add_303: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_300, mul_819);  add_300 = mul_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_52: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_821: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_822: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_303, mul_821);  mul_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1017: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_822, [8192, 1024]);  mul_822 = None
        permute_76: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        permute_835: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        mm_208: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1017, permute_835);  permute_835 = None
        permute_836: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1017, [1, 0])
        mm_209: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_836, view_152);  permute_836 = view_152 = None
        sum_294: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
        view_1018: "f32[1024]" = torch.ops.aten.reshape.default(sum_294, [1024]);  sum_294 = None
        view_1019: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_208, [16, 512, 4096]);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_151: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_40, [16, 512, 4096]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_90: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, 0.7071067811865476)
        erf_6: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_90);  mul_90 = None
        add_56: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_824: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_56, 0.5);  add_56 = None
        mul_825: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, view_151)
        mul_826: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_825, -0.5);  mul_825 = None
        exp_44: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_826);  mul_826 = None
        mul_827: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_44, 0.3989422804014327);  exp_44 = None
        mul_828: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_151, mul_827);  view_151 = mul_827 = None
        add_305: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_824, mul_828);  mul_824 = mul_828 = None
        mul_829: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1019, add_305);  view_1019 = add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1020: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_829, [8192, 4096]);  mul_829 = None
        permute_75: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_839: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_210: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1020, permute_839);  permute_839 = None
        permute_840: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1020, [1, 0])
        mm_211: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_840, view_150);  permute_840 = view_150 = None
        sum_295: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
        view_1021: "f32[4096]" = torch.ops.aten.reshape.default(sum_295, [4096]);  sum_295 = None
        view_1022: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_210, [16, 512, 1024]);  mm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_831: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, primals_113);  primals_113 = None
        mul_832: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_831, 1024)
        sum_296: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_831, [2], True)
        mul_833: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_831, mul_87);  mul_831 = None
        sum_297: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True);  mul_833 = None
        mul_834: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_87, sum_297);  sum_297 = None
        sub_187: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_832, sum_296);  mul_832 = sum_296 = None
        sub_188: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_187, mul_834);  sub_187 = mul_834 = None
        mul_835: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_103, sub_188);  div_103 = sub_188 = None
        mul_836: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1022, mul_87);  mul_87 = None
        sum_298: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_836, [0, 1]);  mul_836 = None
        sum_299: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1022, [0, 1]);  view_1022 = None
        add_306: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_303, mul_835);  add_303 = mul_835 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_53: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_837: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_838: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_306, mul_837);  mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1023: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_838, [8192, 1024]);  mul_838 = None
        permute_74: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_843: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        mm_212: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1023, permute_843);  permute_843 = None
        permute_844: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1023, [1, 0])
        mm_213: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_844, view_148);  permute_844 = view_148 = None
        sum_300: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
        view_1024: "f32[1024]" = torch.ops.aten.reshape.default(sum_300, [1024]);  sum_300 = None
        view_1025: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_212, [16, 512, 1024]);  mm_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1026: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1025, [16, 512, 16, 64]);  view_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_847: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1026, [0, 2, 1, 3]);  view_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_218: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_847, memory_format = torch.contiguous_format);  permute_847 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_17 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_218, permute_default_102, permute_default_103, permute_default_104, None, getitem_219, getitem_220, getitem_221, getitem_222, 0.1, [True, True, True, False], scale = 0.125);  clone_218 = permute_default_102 = permute_default_103 = permute_default_104 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = None
        getitem_223: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_17[0]
        getitem_224: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_17[1]
        getitem_225: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_17[2];  _scaled_dot_product_efficient_attention_backward_default_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_107: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_223, [0, 2, 1, 3]);  getitem_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_106: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_224, [0, 2, 1, 3]);  getitem_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_105: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3]);  getitem_225 = None
        clone_220: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_105, memory_format = torch.contiguous_format);  permute_default_105 = None
        view_1033: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_220, [16, 512, 1024]);  clone_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1034: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1033, [8192, 1024]);  view_1033 = None
        permute_70: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_854: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_214: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1034, permute_854);  permute_854 = None
        permute_855: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1034, [1, 0])
        mm_215: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_855, view_132);  permute_855 = None
        sum_302: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
        view_1035: "f32[1024]" = torch.ops.aten.reshape.default(sum_302, [1024]);  sum_302 = None
        view_1036: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_214, [16, 512, 1024]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1037: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_106, [16, 512, 1024]);  permute_default_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_221: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1037, memory_format = torch.contiguous_format);  view_1037 = None
        view_1038: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_221, [8192, 1024]);  clone_221 = None
        permute_68: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_107, [1, 0]);  primals_107 = None
        permute_859: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_68, [1, 0]);  permute_68 = None
        mm_216: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1038, permute_859);  permute_859 = None
        permute_860: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1038, [1, 0])
        mm_217: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_860, view_132);  permute_860 = None
        sum_303: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1038, [0], True);  view_1038 = None
        view_1039: "f32[1024]" = torch.ops.aten.reshape.default(sum_303, [1024]);  sum_303 = None
        view_1040: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_216, [16, 512, 1024]);  mm_216 = None
        add_307: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1036, view_1040);  view_1036 = view_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_222: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_107, memory_format = torch.contiguous_format);  permute_default_107 = None
        view_1041: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_222, [16, 512, 1024]);  clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1042: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1041, [8192, 1024]);  view_1041 = None
        permute_66: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_864: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_66, [1, 0]);  permute_66 = None
        mm_218: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1042, permute_864);  permute_864 = None
        permute_865: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1042, [1, 0])
        mm_219: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_865, view_132);  permute_865 = view_132 = None
        sum_304: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
        view_1043: "f32[1024]" = torch.ops.aten.reshape.default(sum_304, [1024]);  sum_304 = None
        view_1044: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_218, [16, 512, 1024]);  mm_218 = None
        add_308: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_307, view_1044);  add_307 = view_1044 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_843: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, primals_103);  primals_103 = None
        mul_844: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_843, 1024)
        sum_305: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_843, [2], True)
        mul_845: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_843, mul_81);  mul_843 = None
        sum_306: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_845, [2], True);  mul_845 = None
        mul_846: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_81, sum_306);  sum_306 = None
        sub_190: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_844, sum_305);  mul_844 = sum_305 = None
        sub_191: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_190, mul_846);  sub_190 = mul_846 = None
        mul_847: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_105, sub_191);  div_105 = sub_191 = None
        mul_848: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_308, mul_81);  mul_81 = None
        sum_307: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_848, [0, 1]);  mul_848 = None
        sum_308: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None
        add_309: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_306, mul_847);  add_306 = mul_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_55: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_849: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_850: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_309, mul_849);  mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1045: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_850, [8192, 1024]);  mul_850 = None
        permute_65: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_868: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_220: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1045, permute_868);  permute_868 = None
        permute_869: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1045, [1, 0])
        mm_221: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_869, view_130);  permute_869 = view_130 = None
        sum_309: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
        view_1046: "f32[1024]" = torch.ops.aten.reshape.default(sum_309, [1024]);  sum_309 = None
        view_1047: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_220, [16, 512, 4096]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_129: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_34, [16, 512, 4096]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_77: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, 0.7071067811865476)
        erf_5: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_77);  mul_77 = None
        add_48: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_852: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_48, 0.5);  add_48 = None
        mul_853: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, view_129)
        mul_854: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_853, -0.5);  mul_853 = None
        exp_45: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_854);  mul_854 = None
        mul_855: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_45, 0.3989422804014327);  exp_45 = None
        mul_856: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_129, mul_855);  view_129 = mul_855 = None
        add_311: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_852, mul_856);  mul_852 = mul_856 = None
        mul_857: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1047, add_311);  view_1047 = add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1048: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_857, [8192, 4096]);  mul_857 = None
        permute_64: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_872: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_64, [1, 0]);  permute_64 = None
        mm_222: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1048, permute_872);  permute_872 = None
        permute_873: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1048, [1, 0])
        mm_223: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_873, view_128);  permute_873 = view_128 = None
        sum_310: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
        view_1049: "f32[4096]" = torch.ops.aten.reshape.default(sum_310, [4096]);  sum_310 = None
        view_1050: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_222, [16, 512, 1024]);  mm_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_859: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, primals_97);  primals_97 = None
        mul_860: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_859, 1024)
        sum_311: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_859, [2], True)
        mul_861: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_859, mul_74);  mul_859 = None
        sum_312: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_861, [2], True);  mul_861 = None
        mul_862: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_74, sum_312);  sum_312 = None
        sub_193: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_860, sum_311);  mul_860 = sum_311 = None
        sub_194: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_193, mul_862);  sub_193 = mul_862 = None
        mul_863: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_106, sub_194);  div_106 = sub_194 = None
        mul_864: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1050, mul_74);  mul_74 = None
        sum_313: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_864, [0, 1]);  mul_864 = None
        sum_314: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1050, [0, 1]);  view_1050 = None
        add_312: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_309, mul_863);  add_309 = mul_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_56: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_865: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_866: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_312, mul_865);  mul_865 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1051: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_866, [8192, 1024]);  mul_866 = None
        permute_63: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_876: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_224: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1051, permute_876);  permute_876 = None
        permute_877: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1051, [1, 0])
        mm_225: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_877, view_126);  permute_877 = view_126 = None
        sum_315: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
        view_1052: "f32[1024]" = torch.ops.aten.reshape.default(sum_315, [1024]);  sum_315 = None
        view_1053: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_224, [16, 512, 1024]);  mm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1054: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1053, [16, 512, 16, 64]);  view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_880: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_225: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_880, memory_format = torch.contiguous_format);  permute_880 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_18 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_225, permute_default_108, permute_default_109, permute_default_110, None, getitem_226, getitem_227, getitem_228, getitem_229, 0.1, [True, True, True, False], scale = 0.125);  clone_225 = permute_default_108 = permute_default_109 = permute_default_110 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = None
        getitem_230: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_18[0]
        getitem_231: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_18[1]
        getitem_232: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_18[2];  _scaled_dot_product_efficient_attention_backward_default_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_113: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_230, [0, 2, 1, 3]);  getitem_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_112: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_231, [0, 2, 1, 3]);  getitem_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_111: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_232, [0, 2, 1, 3]);  getitem_232 = None
        clone_227: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_111, memory_format = torch.contiguous_format);  permute_default_111 = None
        view_1061: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_227, [16, 512, 1024]);  clone_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1062: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1061, [8192, 1024]);  view_1061 = None
        permute_59: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        permute_887: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_226: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1062, permute_887);  permute_887 = None
        permute_888: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_227: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_888, view_110);  permute_888 = None
        sum_317: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1062, [0], True);  view_1062 = None
        view_1063: "f32[1024]" = torch.ops.aten.reshape.default(sum_317, [1024]);  sum_317 = None
        view_1064: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_226, [16, 512, 1024]);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1065: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_112, [16, 512, 1024]);  permute_default_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_228: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1065, memory_format = torch.contiguous_format);  view_1065 = None
        view_1066: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_228, [8192, 1024]);  clone_228 = None
        permute_57: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_892: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_228: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1066, permute_892);  permute_892 = None
        permute_893: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1066, [1, 0])
        mm_229: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_893, view_110);  permute_893 = None
        sum_318: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
        view_1067: "f32[1024]" = torch.ops.aten.reshape.default(sum_318, [1024]);  sum_318 = None
        view_1068: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_228, [16, 512, 1024]);  mm_228 = None
        add_313: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1064, view_1068);  view_1064 = view_1068 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_229: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_113, memory_format = torch.contiguous_format);  permute_default_113 = None
        view_1069: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_229, [16, 512, 1024]);  clone_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1070: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1069, [8192, 1024]);  view_1069 = None
        permute_55: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_89, [1, 0]);  primals_89 = None
        permute_897: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        mm_230: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1070, permute_897);  permute_897 = None
        permute_898: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1070, [1, 0])
        mm_231: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_898, view_110);  permute_898 = view_110 = None
        sum_319: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
        view_1071: "f32[1024]" = torch.ops.aten.reshape.default(sum_319, [1024]);  sum_319 = None
        view_1072: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_230, [16, 512, 1024]);  mm_230 = None
        add_314: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_313, view_1072);  add_313 = view_1072 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_871: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, primals_87);  primals_87 = None
        mul_872: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_871, 1024)
        sum_320: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_871, [2], True)
        mul_873: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_871, mul_68);  mul_871 = None
        sum_321: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_873, [2], True);  mul_873 = None
        mul_874: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_68, sum_321);  sum_321 = None
        sub_196: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_872, sum_320);  mul_872 = sum_320 = None
        sub_197: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_196, mul_874);  sub_196 = mul_874 = None
        mul_875: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_108, sub_197);  div_108 = sub_197 = None
        mul_876: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_314, mul_68);  mul_68 = None
        sum_322: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_876, [0, 1]);  mul_876 = None
        sum_323: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None
        add_315: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_312, mul_875);  add_312 = mul_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_58: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_877: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_878: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_315, mul_877);  mul_877 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1073: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_878, [8192, 1024]);  mul_878 = None
        permute_54: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_901: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        mm_232: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1073, permute_901);  permute_901 = None
        permute_902: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1073, [1, 0])
        mm_233: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_902, view_108);  permute_902 = view_108 = None
        sum_324: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
        view_1074: "f32[1024]" = torch.ops.aten.reshape.default(sum_324, [1024]);  sum_324 = None
        view_1075: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_232, [16, 512, 4096]);  mm_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_107: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_28, [16, 512, 4096]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_64: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, 0.7071067811865476)
        erf_4: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_40: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_880: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_40, 0.5);  add_40 = None
        mul_881: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, view_107)
        mul_882: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_881, -0.5);  mul_881 = None
        exp_46: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_882);  mul_882 = None
        mul_883: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_46, 0.3989422804014327);  exp_46 = None
        mul_884: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_107, mul_883);  view_107 = mul_883 = None
        add_317: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_880, mul_884);  mul_880 = mul_884 = None
        mul_885: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1075, add_317);  view_1075 = add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1076: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_885, [8192, 4096]);  mul_885 = None
        permute_53: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_905: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_234: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1076, permute_905);  permute_905 = None
        permute_906: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        mm_235: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_906, view_106);  permute_906 = view_106 = None
        sum_325: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        view_1077: "f32[4096]" = torch.ops.aten.reshape.default(sum_325, [4096]);  sum_325 = None
        view_1078: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_234, [16, 512, 1024]);  mm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_887: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, primals_81);  primals_81 = None
        mul_888: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_887, 1024)
        sum_326: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
        mul_889: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_887, mul_61);  mul_887 = None
        sum_327: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
        mul_890: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_61, sum_327);  sum_327 = None
        sub_199: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_888, sum_326);  mul_888 = sum_326 = None
        sub_200: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_199, mul_890);  sub_199 = mul_890 = None
        mul_891: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_109, sub_200);  div_109 = sub_200 = None
        mul_892: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1078, mul_61);  mul_61 = None
        sum_328: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
        sum_329: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1078, [0, 1]);  view_1078 = None
        add_318: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_315, mul_891);  add_315 = mul_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_59: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_893: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_894: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_318, mul_893);  mul_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1079: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_894, [8192, 1024]);  mul_894 = None
        permute_52: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_909: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        mm_236: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1079, permute_909);  permute_909 = None
        permute_910: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1079, [1, 0])
        mm_237: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_910, view_104);  permute_910 = view_104 = None
        sum_330: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
        view_1080: "f32[1024]" = torch.ops.aten.reshape.default(sum_330, [1024]);  sum_330 = None
        view_1081: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_236, [16, 512, 1024]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1082: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1081, [16, 512, 16, 64]);  view_1081 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_913: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1082, [0, 2, 1, 3]);  view_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_232: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_913, memory_format = torch.contiguous_format);  permute_913 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_19 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_232, permute_default_114, permute_default_115, permute_default_116, None, getitem_233, getitem_234, getitem_235, getitem_236, 0.1, [True, True, True, False], scale = 0.125);  clone_232 = permute_default_114 = permute_default_115 = permute_default_116 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = None
        getitem_237: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_19[0]
        getitem_238: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_19[1]
        getitem_239: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_19[2];  _scaled_dot_product_efficient_attention_backward_default_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_119: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_237, [0, 2, 1, 3]);  getitem_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_118: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_238, [0, 2, 1, 3]);  getitem_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_117: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_239, [0, 2, 1, 3]);  getitem_239 = None
        clone_234: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_117, memory_format = torch.contiguous_format);  permute_default_117 = None
        view_1089: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_234, [16, 512, 1024]);  clone_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1090: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1089, [8192, 1024]);  view_1089 = None
        permute_48: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_77, [1, 0]);  primals_77 = None
        permute_920: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_238: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1090, permute_920);  permute_920 = None
        permute_921: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1090, [1, 0])
        mm_239: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_921, view_88);  permute_921 = None
        sum_332: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1090, [0], True);  view_1090 = None
        view_1091: "f32[1024]" = torch.ops.aten.reshape.default(sum_332, [1024]);  sum_332 = None
        view_1092: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_238, [16, 512, 1024]);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1093: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_118, [16, 512, 1024]);  permute_default_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_235: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1093, memory_format = torch.contiguous_format);  view_1093 = None
        view_1094: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_235, [8192, 1024]);  clone_235 = None
        permute_46: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_925: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_240: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1094, permute_925);  permute_925 = None
        permute_926: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1094, [1, 0])
        mm_241: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_926, view_88);  permute_926 = None
        sum_333: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1094, [0], True);  view_1094 = None
        view_1095: "f32[1024]" = torch.ops.aten.reshape.default(sum_333, [1024]);  sum_333 = None
        view_1096: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_240, [16, 512, 1024]);  mm_240 = None
        add_319: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1092, view_1096);  view_1092 = view_1096 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_236: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_119, memory_format = torch.contiguous_format);  permute_default_119 = None
        view_1097: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_236, [16, 512, 1024]);  clone_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1098: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1097, [8192, 1024]);  view_1097 = None
        permute_44: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_930: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_44, [1, 0]);  permute_44 = None
        mm_242: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1098, permute_930);  permute_930 = None
        permute_931: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1098, [1, 0])
        mm_243: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_931, view_88);  permute_931 = view_88 = None
        sum_334: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
        view_1099: "f32[1024]" = torch.ops.aten.reshape.default(sum_334, [1024]);  sum_334 = None
        view_1100: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_242, [16, 512, 1024]);  mm_242 = None
        add_320: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_319, view_1100);  add_319 = view_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_899: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, primals_71);  primals_71 = None
        mul_900: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_899, 1024)
        sum_335: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_899, [2], True)
        mul_901: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_899, mul_55);  mul_899 = None
        sum_336: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_901, [2], True);  mul_901 = None
        mul_902: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_55, sum_336);  sum_336 = None
        sub_202: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_900, sum_335);  mul_900 = sum_335 = None
        sub_203: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_202, mul_902);  sub_202 = mul_902 = None
        mul_903: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_111, sub_203);  div_111 = sub_203 = None
        mul_904: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_320, mul_55);  mul_55 = None
        sum_337: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_904, [0, 1]);  mul_904 = None
        sum_338: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_320, [0, 1]);  add_320 = None
        add_321: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_318, mul_903);  add_318 = mul_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_61: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_905: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_906: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_321, mul_905);  mul_905 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1101: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_906, [8192, 1024]);  mul_906 = None
        permute_43: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_934: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        mm_244: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1101, permute_934);  permute_934 = None
        permute_935: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1101, [1, 0])
        mm_245: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_935, view_86);  permute_935 = view_86 = None
        sum_339: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
        view_1102: "f32[1024]" = torch.ops.aten.reshape.default(sum_339, [1024]);  sum_339 = None
        view_1103: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_244, [16, 512, 4096]);  mm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_85: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_22, [16, 512, 4096]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_51: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, 0.7071067811865476)
        erf_3: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_51);  mul_51 = None
        add_32: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_908: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_32, 0.5);  add_32 = None
        mul_909: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, view_85)
        mul_910: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_909, -0.5);  mul_909 = None
        exp_47: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_910);  mul_910 = None
        mul_911: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_47, 0.3989422804014327);  exp_47 = None
        mul_912: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_85, mul_911);  view_85 = mul_911 = None
        add_323: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_908, mul_912);  mul_908 = mul_912 = None
        mul_913: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1103, add_323);  view_1103 = add_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1104: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_913, [8192, 4096]);  mul_913 = None
        permute_42: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        permute_938: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        mm_246: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1104, permute_938);  permute_938 = None
        permute_939: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1104, [1, 0])
        mm_247: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_939, view_84);  permute_939 = view_84 = None
        sum_340: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
        view_1105: "f32[4096]" = torch.ops.aten.reshape.default(sum_340, [4096]);  sum_340 = None
        view_1106: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_246, [16, 512, 1024]);  mm_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_915: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, primals_65);  primals_65 = None
        mul_916: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_915, 1024)
        sum_341: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_915, [2], True)
        mul_917: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_915, mul_48);  mul_915 = None
        sum_342: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_917, [2], True);  mul_917 = None
        mul_918: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_48, sum_342);  sum_342 = None
        sub_205: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_916, sum_341);  mul_916 = sum_341 = None
        sub_206: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_205, mul_918);  sub_205 = mul_918 = None
        mul_919: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_112, sub_206);  div_112 = sub_206 = None
        mul_920: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1106, mul_48);  mul_48 = None
        sum_343: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_920, [0, 1]);  mul_920 = None
        sum_344: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1106, [0, 1]);  view_1106 = None
        add_324: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_321, mul_919);  add_321 = mul_919 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_921: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_922: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_324, mul_921);  mul_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1107: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_922, [8192, 1024]);  mul_922 = None
        permute_41: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        permute_942: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_248: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1107, permute_942);  permute_942 = None
        permute_943: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1107, [1, 0])
        mm_249: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_943, view_82);  permute_943 = view_82 = None
        sum_345: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
        view_1108: "f32[1024]" = torch.ops.aten.reshape.default(sum_345, [1024]);  sum_345 = None
        view_1109: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_248, [16, 512, 1024]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1110: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1109, [16, 512, 16, 64]);  view_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_946: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1110, [0, 2, 1, 3]);  view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_239: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_946, memory_format = torch.contiguous_format);  permute_946 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_20 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_239, permute_default_120, permute_default_121, permute_default_122, None, getitem_240, getitem_241, getitem_242, getitem_243, 0.1, [True, True, True, False], scale = 0.125);  clone_239 = permute_default_120 = permute_default_121 = permute_default_122 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = None
        getitem_244: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_20[0]
        getitem_245: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_20[1]
        getitem_246: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_20[2];  _scaled_dot_product_efficient_attention_backward_default_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_125: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_244, [0, 2, 1, 3]);  getitem_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_124: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_245, [0, 2, 1, 3]);  getitem_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_123: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_246, [0, 2, 1, 3]);  getitem_246 = None
        clone_241: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_123, memory_format = torch.contiguous_format);  permute_default_123 = None
        view_1117: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_241, [16, 512, 1024]);  clone_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1118: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1117, [8192, 1024]);  view_1117 = None
        permute_37: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_953: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_250: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1118, permute_953);  permute_953 = None
        permute_954: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1118, [1, 0])
        mm_251: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_954, view_66);  permute_954 = None
        sum_347: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1118, [0], True);  view_1118 = None
        view_1119: "f32[1024]" = torch.ops.aten.reshape.default(sum_347, [1024]);  sum_347 = None
        view_1120: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_250, [16, 512, 1024]);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1121: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_124, [16, 512, 1024]);  permute_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_242: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1121, memory_format = torch.contiguous_format);  view_1121 = None
        view_1122: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_242, [8192, 1024]);  clone_242 = None
        permute_35: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_958: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_252: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1122, permute_958);  permute_958 = None
        permute_959: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1122, [1, 0])
        mm_253: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_959, view_66);  permute_959 = None
        sum_348: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1122, [0], True);  view_1122 = None
        view_1123: "f32[1024]" = torch.ops.aten.reshape.default(sum_348, [1024]);  sum_348 = None
        view_1124: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_252, [16, 512, 1024]);  mm_252 = None
        add_325: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1120, view_1124);  view_1120 = view_1124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_243: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_125, memory_format = torch.contiguous_format);  permute_default_125 = None
        view_1125: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_243, [16, 512, 1024]);  clone_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1126: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1125, [8192, 1024]);  view_1125 = None
        permute_33: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_963: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_254: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1126, permute_963);  permute_963 = None
        permute_964: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1126, [1, 0])
        mm_255: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_964, view_66);  permute_964 = view_66 = None
        sum_349: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
        view_1127: "f32[1024]" = torch.ops.aten.reshape.default(sum_349, [1024]);  sum_349 = None
        view_1128: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_254, [16, 512, 1024]);  mm_254 = None
        add_326: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_325, view_1128);  add_325 = view_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_927: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, primals_55);  primals_55 = None
        mul_928: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_927, 1024)
        sum_350: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_927, [2], True)
        mul_929: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_927, mul_42);  mul_927 = None
        sum_351: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_929, [2], True);  mul_929 = None
        mul_930: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_42, sum_351);  sum_351 = None
        sub_208: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_928, sum_350);  mul_928 = sum_350 = None
        sub_209: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_208, mul_930);  sub_208 = mul_930 = None
        mul_931: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_114, sub_209);  div_114 = sub_209 = None
        mul_932: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_326, mul_42);  mul_42 = None
        sum_352: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_932, [0, 1]);  mul_932 = None
        sum_353: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_326, [0, 1]);  add_326 = None
        add_327: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_324, mul_931);  add_324 = mul_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_64: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_933: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_934: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_327, mul_933);  mul_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1129: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_934, [8192, 1024]);  mul_934 = None
        permute_32: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_967: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        mm_256: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1129, permute_967);  permute_967 = None
        permute_968: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_257: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_968, view_64);  permute_968 = view_64 = None
        sum_354: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
        view_1130: "f32[1024]" = torch.ops.aten.reshape.default(sum_354, [1024]);  sum_354 = None
        view_1131: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_256, [16, 512, 4096]);  mm_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_63: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_16, [16, 512, 4096]);  addmm_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_38: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, 0.7071067811865476)
        erf_2: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_38);  mul_38 = None
        add_24: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_936: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_24, 0.5);  add_24 = None
        mul_937: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, view_63)
        mul_938: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_937, -0.5);  mul_937 = None
        exp_48: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_938);  mul_938 = None
        mul_939: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_48, 0.3989422804014327);  exp_48 = None
        mul_940: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_63, mul_939);  view_63 = mul_939 = None
        add_329: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_936, mul_940);  mul_936 = mul_940 = None
        mul_941: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1131, add_329);  view_1131 = add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1132: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_941, [8192, 4096]);  mul_941 = None
        permute_31: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_971: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        mm_258: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1132, permute_971);  permute_971 = None
        permute_972: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_259: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_972, view_62);  permute_972 = view_62 = None
        sum_355: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        view_1133: "f32[4096]" = torch.ops.aten.reshape.default(sum_355, [4096]);  sum_355 = None
        view_1134: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_258, [16, 512, 1024]);  mm_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_943: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, primals_49);  primals_49 = None
        mul_944: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_943, 1024)
        sum_356: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_943, [2], True)
        mul_945: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_943, mul_35);  mul_943 = None
        sum_357: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_945, [2], True);  mul_945 = None
        mul_946: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_35, sum_357);  sum_357 = None
        sub_211: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_944, sum_356);  mul_944 = sum_356 = None
        sub_212: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_211, mul_946);  sub_211 = mul_946 = None
        mul_947: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_115, sub_212);  div_115 = sub_212 = None
        mul_948: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1134, mul_35);  mul_35 = None
        sum_358: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_948, [0, 1]);  mul_948 = None
        sum_359: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1134, [0, 1]);  view_1134 = None
        add_330: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_327, mul_947);  add_327 = mul_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_65: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_949: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_950: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_330, mul_949);  mul_949 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1135: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_950, [8192, 1024]);  mul_950 = None
        permute_30: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_975: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_30, [1, 0]);  permute_30 = None
        mm_260: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1135, permute_975);  permute_975 = None
        permute_976: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        mm_261: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_976, view_60);  permute_976 = view_60 = None
        sum_360: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        view_1136: "f32[1024]" = torch.ops.aten.reshape.default(sum_360, [1024]);  sum_360 = None
        view_1137: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_260, [16, 512, 1024]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1138: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1137, [16, 512, 16, 64]);  view_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_979: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1138, [0, 2, 1, 3]);  view_1138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_246: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_979, memory_format = torch.contiguous_format);  permute_979 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_21 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_246, permute_default_126, permute_default_127, permute_default_128, None, getitem_247, getitem_248, getitem_249, getitem_250, 0.1, [True, True, True, False], scale = 0.125);  clone_246 = permute_default_126 = permute_default_127 = permute_default_128 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = None
        getitem_251: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_21[0]
        getitem_252: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_21[1]
        getitem_253: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_21[2];  _scaled_dot_product_efficient_attention_backward_default_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_131: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_251, [0, 2, 1, 3]);  getitem_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_130: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_252, [0, 2, 1, 3]);  getitem_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_129: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_253, [0, 2, 1, 3]);  getitem_253 = None
        clone_248: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_129, memory_format = torch.contiguous_format);  permute_default_129 = None
        view_1145: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_248, [16, 512, 1024]);  clone_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1146: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1145, [8192, 1024]);  view_1145 = None
        permute_26: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        permute_986: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_26, [1, 0]);  permute_26 = None
        mm_262: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1146, permute_986);  permute_986 = None
        permute_987: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1146, [1, 0])
        mm_263: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_987, view_44);  permute_987 = None
        sum_362: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1146, [0], True);  view_1146 = None
        view_1147: "f32[1024]" = torch.ops.aten.reshape.default(sum_362, [1024]);  sum_362 = None
        view_1148: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_262, [16, 512, 1024]);  mm_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1149: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_130, [16, 512, 1024]);  permute_default_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_249: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1149, memory_format = torch.contiguous_format);  view_1149 = None
        view_1150: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_249, [8192, 1024]);  clone_249 = None
        permute_24: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_991: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_264: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1150, permute_991);  permute_991 = None
        permute_992: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1150, [1, 0])
        mm_265: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_992, view_44);  permute_992 = None
        sum_363: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1150, [0], True);  view_1150 = None
        view_1151: "f32[1024]" = torch.ops.aten.reshape.default(sum_363, [1024]);  sum_363 = None
        view_1152: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_264, [16, 512, 1024]);  mm_264 = None
        add_331: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1148, view_1152);  view_1148 = view_1152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_250: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_131, memory_format = torch.contiguous_format);  permute_default_131 = None
        view_1153: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_250, [16, 512, 1024]);  clone_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1154: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1153, [8192, 1024]);  view_1153 = None
        permute_22: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_996: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_266: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1154, permute_996);  permute_996 = None
        permute_997: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1154, [1, 0])
        mm_267: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_997, view_44);  permute_997 = view_44 = None
        sum_364: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
        view_1155: "f32[1024]" = torch.ops.aten.reshape.default(sum_364, [1024]);  sum_364 = None
        view_1156: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_266, [16, 512, 1024]);  mm_266 = None
        add_332: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_331, view_1156);  add_331 = view_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_955: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, primals_39);  primals_39 = None
        mul_956: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_955, 1024)
        sum_365: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_955, [2], True)
        mul_957: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_955, mul_29);  mul_955 = None
        sum_366: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_957, [2], True);  mul_957 = None
        mul_958: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_29, sum_366);  sum_366 = None
        sub_214: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_956, sum_365);  mul_956 = sum_365 = None
        sub_215: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_214, mul_958);  sub_214 = mul_958 = None
        mul_959: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_117, sub_215);  div_117 = sub_215 = None
        mul_960: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_332, mul_29);  mul_29 = None
        sum_367: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 1]);  mul_960 = None
        sum_368: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_332, [0, 1]);  add_332 = None
        add_333: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_330, mul_959);  add_330 = mul_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_67: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_961: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_962: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_333, mul_961);  mul_961 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1157: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_962, [8192, 1024]);  mul_962 = None
        permute_21: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_1000: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_268: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1157, permute_1000);  permute_1000 = None
        permute_1001: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1157, [1, 0])
        mm_269: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1001, view_42);  permute_1001 = view_42 = None
        sum_369: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
        view_1158: "f32[1024]" = torch.ops.aten.reshape.default(sum_369, [1024]);  sum_369 = None
        view_1159: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_268, [16, 512, 4096]);  mm_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_41: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_10, [16, 512, 4096]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_25: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476)
        erf_1: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_25);  mul_25 = None
        add_16: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_964: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_16, 0.5);  add_16 = None
        mul_965: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, view_41)
        mul_966: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_965, -0.5);  mul_965 = None
        exp_49: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_966);  mul_966 = None
        mul_967: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_49, 0.3989422804014327);  exp_49 = None
        mul_968: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_41, mul_967);  view_41 = mul_967 = None
        add_335: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_964, mul_968);  mul_964 = mul_968 = None
        mul_969: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1159, add_335);  view_1159 = add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1160: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_969, [8192, 4096]);  mul_969 = None
        permute_20: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_1004: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        mm_270: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1160, permute_1004);  permute_1004 = None
        permute_1005: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1160, [1, 0])
        mm_271: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1005, view_40);  permute_1005 = view_40 = None
        sum_370: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
        view_1161: "f32[4096]" = torch.ops.aten.reshape.default(sum_370, [4096]);  sum_370 = None
        view_1162: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_270, [16, 512, 1024]);  mm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_971: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, primals_33);  primals_33 = None
        mul_972: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_971, 1024)
        sum_371: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_971, [2], True)
        mul_973: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_971, mul_22);  mul_971 = None
        sum_372: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_973, [2], True);  mul_973 = None
        mul_974: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_22, sum_372);  sum_372 = None
        sub_217: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_972, sum_371);  mul_972 = sum_371 = None
        sub_218: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_217, mul_974);  sub_217 = mul_974 = None
        mul_975: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_118, sub_218);  div_118 = sub_218 = None
        mul_976: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1162, mul_22);  mul_22 = None
        sum_373: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_976, [0, 1]);  mul_976 = None
        sum_374: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1162, [0, 1]);  view_1162 = None
        add_336: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_333, mul_975);  add_333 = mul_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_68: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_977: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_978: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_336, mul_977);  mul_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1163: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_978, [8192, 1024]);  mul_978 = None
        permute_19: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        permute_1008: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        mm_272: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1163, permute_1008);  permute_1008 = None
        permute_1009: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1163, [1, 0])
        mm_273: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1009, view_38);  permute_1009 = view_38 = None
        sum_375: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
        view_1164: "f32[1024]" = torch.ops.aten.reshape.default(sum_375, [1024]);  sum_375 = None
        view_1165: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_272, [16, 512, 1024]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1166: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1165, [16, 512, 16, 64]);  view_1165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_1012: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_253: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_22 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_253, permute_default_132, permute_default_133, permute_default_134, None, getitem_254, getitem_255, getitem_256, getitem_257, 0.1, [True, True, True, False], scale = 0.125);  clone_253 = permute_default_132 = permute_default_133 = permute_default_134 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = None
        getitem_258: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_22[0]
        getitem_259: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_22[1]
        getitem_260: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_22[2];  _scaled_dot_product_efficient_attention_backward_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_137: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3]);  getitem_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_136: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_259, [0, 2, 1, 3]);  getitem_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_135: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_260, [0, 2, 1, 3]);  getitem_260 = None
        clone_255: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_135, memory_format = torch.contiguous_format);  permute_default_135 = None
        view_1173: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_255, [16, 512, 1024]);  clone_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1174: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1173, [8192, 1024]);  view_1173 = None
        permute_15: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_1019: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_274: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1174, permute_1019);  permute_1019 = None
        permute_1020: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1174, [1, 0])
        mm_275: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1020, view_22);  permute_1020 = None
        sum_377: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1174, [0], True);  view_1174 = None
        view_1175: "f32[1024]" = torch.ops.aten.reshape.default(sum_377, [1024]);  sum_377 = None
        view_1176: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_274, [16, 512, 1024]);  mm_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1177: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_136, [16, 512, 1024]);  permute_default_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_256: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1177, memory_format = torch.contiguous_format);  view_1177 = None
        view_1178: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_256, [8192, 1024]);  clone_256 = None
        permute_13: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        permute_1024: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_276: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1178, permute_1024);  permute_1024 = None
        permute_1025: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1178, [1, 0])
        mm_277: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1025, view_22);  permute_1025 = None
        sum_378: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1178, [0], True);  view_1178 = None
        view_1179: "f32[1024]" = torch.ops.aten.reshape.default(sum_378, [1024]);  sum_378 = None
        view_1180: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_276, [16, 512, 1024]);  mm_276 = None
        add_337: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1176, view_1180);  view_1176 = view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_257: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_137, memory_format = torch.contiguous_format);  permute_default_137 = None
        view_1181: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_257, [16, 512, 1024]);  clone_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1182: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1181, [8192, 1024]);  view_1181 = None
        permute_11: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_1029: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_278: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1182, permute_1029);  permute_1029 = None
        permute_1030: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_279: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1030, view_22);  permute_1030 = view_22 = None
        sum_379: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
        view_1183: "f32[1024]" = torch.ops.aten.reshape.default(sum_379, [1024]);  sum_379 = None
        view_1184: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_278, [16, 512, 1024]);  mm_278 = None
        add_338: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_337, view_1184);  add_337 = view_1184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_983: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, primals_23);  primals_23 = None
        mul_984: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_983, 1024)
        sum_380: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_983, [2], True)
        mul_985: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_983, mul_16);  mul_983 = None
        sum_381: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_985, [2], True);  mul_985 = None
        mul_986: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_16, sum_381);  sum_381 = None
        sub_220: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_984, sum_380);  mul_984 = sum_380 = None
        sub_221: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_220, mul_986);  sub_220 = mul_986 = None
        mul_987: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_120, sub_221);  div_120 = sub_221 = None
        mul_988: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_338, mul_16);  mul_16 = None
        sum_382: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_988, [0, 1]);  mul_988 = None
        sum_383: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_338, [0, 1]);  add_338 = None
        add_339: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_336, mul_987);  add_336 = mul_987 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:262 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_70: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_989: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_990: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_339, mul_989);  mul_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:261 in forward, code: hidden_states = self.dense(hidden_states)
        view_1185: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_990, [8192, 1024]);  mul_990 = None
        permute_10: "f32[4096, 1024]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_1033: "f32[1024, 4096]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_280: "f32[8192, 4096]" = torch.ops.aten.mm.default(view_1185, permute_1033);  permute_1033 = None
        permute_1034: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_281: "f32[1024, 4096]" = torch.ops.aten.mm.default(permute_1034, view_20);  permute_1034 = view_20 = None
        sum_384: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
        view_1186: "f32[1024]" = torch.ops.aten.reshape.default(sum_384, [1024]);  sum_384 = None
        view_1187: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(mm_280, [16, 512, 4096]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_19: "f32[16, 512, 4096]" = torch.ops.aten.reshape.default(addmm_4, [16, 512, 4096]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_12: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476)
        erf: "f32[16, 512, 4096]" = torch.ops.aten.erf.default(mul_12);  mul_12 = None
        add_8: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_992: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(add_8, 0.5);  add_8 = None
        mul_993: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, view_19)
        mul_994: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(mul_993, -0.5);  mul_993 = None
        exp_50: "f32[16, 512, 4096]" = torch.ops.aten.exp.default(mul_994);  mul_994 = None
        mul_995: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(exp_50, 0.3989422804014327);  exp_50 = None
        mul_996: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_19, mul_995);  view_19 = mul_995 = None
        add_341: "f32[16, 512, 4096]" = torch.ops.aten.add.Tensor(mul_992, mul_996);  mul_992 = mul_996 = None
        mul_997: "f32[16, 512, 4096]" = torch.ops.aten.mul.Tensor(view_1187, add_341);  view_1187 = add_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:248 in forward, code: hidden_states = self.dense(hidden_states)
        view_1188: "f32[8192, 4096]" = torch.ops.aten.reshape.default(mul_997, [8192, 4096]);  mul_997 = None
        permute_9: "f32[1024, 4096]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_1037: "f32[4096, 1024]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_282: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1188, permute_1037);  permute_1037 = None
        permute_1038: "f32[4096, 8192]" = torch.ops.aten.permute.default(view_1188, [1, 0])
        mm_283: "f32[4096, 1024]" = torch.ops.aten.mm.default(permute_1038, view_18);  permute_1038 = view_18 = None
        sum_385: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
        view_1189: "f32[4096]" = torch.ops.aten.reshape.default(sum_385, [4096]);  sum_385 = None
        view_1190: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_282, [16, 512, 1024]);  mm_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:326 in feed_forward_chunk, code: ln_output = self.ln(attention_output)
        mul_999: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, primals_17);  primals_17 = None
        mul_1000: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_999, 1024)
        sum_386: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_999, [2], True)
        mul_1001: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_999, mul_9);  mul_999 = None
        sum_387: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1001, [2], True);  mul_1001 = None
        mul_1002: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_9, sum_387);  sum_387 = None
        sub_223: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1000, sum_386);  mul_1000 = sum_386 = None
        sub_224: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_223, mul_1002);  sub_223 = mul_1002 = None
        mul_1003: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_121, sub_224);  div_121 = sub_224 = None
        mul_1004: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(view_1190, mul_9);  mul_9 = None
        sum_388: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1004, [0, 1]);  mul_1004 = None
        sum_389: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1190, [0, 1]);  view_1190 = None
        add_342: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_339, mul_1003);  add_339 = mul_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:203 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_71: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_1005: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_1006: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_342, mul_1005);  mul_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:202 in forward, code: hidden_states = self.dense(hidden_states)
        view_1191: "f32[8192, 1024]" = torch.ops.aten.reshape.default(mul_1006, [8192, 1024]);  mul_1006 = None
        permute_8: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_1041: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_8, [1, 0]);  permute_8 = None
        mm_284: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1191, permute_1041);  permute_1041 = None
        permute_1042: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1191, [1, 0])
        mm_285: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1042, view_16);  permute_1042 = view_16 = None
        sum_390: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
        view_1192: "f32[1024]" = torch.ops.aten.reshape.default(sum_390, [1024]);  sum_390 = None
        view_1193: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_284, [16, 512, 1024]);  mm_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:189 in forward, code: context_layer = context_layer.view(new_context_layer_shape)
        view_1194: "f32[16, 512, 16, 64]" = torch.ops.aten.reshape.default(view_1193, [16, 512, 16, 64]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:187 in forward, code: context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        permute_1045: "f32[16, 16, 512, 64]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:185 in forward, code: context_layer = torch.matmul(attention_probs, value_layer)
        clone_260: "f32[16, 16, 512, 64]" = torch.ops.aten.clone.default(permute_1045, memory_format = torch.contiguous_format);  permute_1045 = None

        # No stacktrace found for following nodes
        _scaled_dot_product_efficient_attention_backward_default_23 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(clone_260, permute_default_138, permute_default_139, permute_default_140, None, getitem_261, getitem_262, getitem_263, getitem_264, 0.1, [True, True, True, False], scale = 0.125);  clone_260 = permute_default_138 = permute_default_139 = permute_default_140 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = None
        getitem_265: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_23[0]
        getitem_266: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_23[1]
        getitem_267: "f32[16, 16, 512, 64]" = _scaled_dot_product_efficient_attention_backward_default_23[2];  _scaled_dot_product_efficient_attention_backward_default_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        permute_default_143: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_265, [0, 2, 1, 3]);  getitem_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        permute_default_142: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_266, [0, 2, 1, 3]);  getitem_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:161 in forward, code: value_layer = value_layer.view(hidden_shape).transpose(1, 2)
        permute_default_141: "f32[16, 512, 16, 64]" = torch.ops.aten.permute.default(getitem_267, [0, 2, 1, 3]);  getitem_267 = None
        clone_262: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_141, memory_format = torch.contiguous_format);  permute_default_141 = None
        view_1201: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_262, [16, 512, 1024]);  clone_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:160 in forward, code: value_layer = self.value(current_states)
        view_1202: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1201, [8192, 1024]);  view_1201 = None
        permute_4: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        permute_1052: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_286: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1202, permute_1052);  permute_1052 = None
        permute_1053: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1202, [1, 0])
        mm_287: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1053, view);  permute_1053 = None
        sum_392: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1202, [0], True);  view_1202 = None
        view_1203: "f32[1024]" = torch.ops.aten.reshape.default(sum_392, [1024]);  sum_392 = None
        view_1204: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_286, [16, 512, 1024]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:159 in forward, code: key_layer = key_layer.view(hidden_shape).transpose(1, 2)
        view_1205: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(permute_default_142, [16, 512, 1024]);  permute_default_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:158 in forward, code: key_layer = self.key(current_states)
        clone_263: "f32[16, 512, 1024]" = torch.ops.aten.clone.default(view_1205, memory_format = torch.contiguous_format);  view_1205 = None
        view_1206: "f32[8192, 1024]" = torch.ops.aten.reshape.default(clone_263, [8192, 1024]);  clone_263 = None
        permute_2: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_1057: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_288: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1206, permute_1057);  permute_1057 = None
        permute_1058: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1206, [1, 0])
        mm_289: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1058, view);  permute_1058 = None
        sum_393: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1206, [0], True);  view_1206 = None
        view_1207: "f32[1024]" = torch.ops.aten.reshape.default(sum_393, [1024]);  sum_393 = None
        view_1208: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_288, [16, 512, 1024]);  mm_288 = None
        add_343: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(view_1204, view_1208);  view_1204 = view_1208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:137 in forward, code: query_layer = query_layer.view(hidden_shape).transpose(1, 2)
        clone_264: "f32[16, 512, 16, 64]" = torch.ops.aten.clone.default(permute_default_143, memory_format = torch.contiguous_format);  permute_default_143 = None
        view_1209: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(clone_264, [16, 512, 1024]);  clone_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:136 in forward, code: query_layer = self.query(hidden_states)
        view_1210: "f32[8192, 1024]" = torch.ops.aten.reshape.default(view_1209, [8192, 1024]);  view_1209 = None
        permute: "f32[1024, 1024]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        permute_1062: "f32[1024, 1024]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_290: "f32[8192, 1024]" = torch.ops.aten.mm.default(view_1210, permute_1062);  permute_1062 = None
        permute_1063: "f32[1024, 8192]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        mm_291: "f32[1024, 1024]" = torch.ops.aten.mm.default(permute_1063, view);  permute_1063 = view = None
        sum_394: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        view_1211: "f32[1024]" = torch.ops.aten.reshape.default(sum_394, [1024]);  sum_394 = None
        view_1212: "f32[16, 512, 1024]" = torch.ops.aten.reshape.default(mm_290, [16, 512, 1024]);  mm_290 = None
        add_344: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_343, view_1212);  add_343 = view_1212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:224 in forward, code: ln_outputs = self.ln(hidden_states)
        mul_1011: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, primals_7);  primals_7 = None
        mul_1012: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1011, 1024)
        sum_395: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1011, [2], True)
        mul_1013: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_1011, mul_3);  mul_1011 = None
        sum_396: "f32[16, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1013, [2], True);  mul_1013 = None
        mul_1014: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(mul_3, sum_396);  sum_396 = None
        sub_226: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(mul_1012, sum_395);  mul_1012 = sum_395 = None
        sub_227: "f32[16, 512, 1024]" = torch.ops.aten.sub.Tensor(sub_226, mul_1014);  sub_226 = mul_1014 = None
        mul_1015: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(div_123, sub_227);  div_123 = sub_227 = None
        mul_1016: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_344, mul_3);  mul_3 = None
        sum_397: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_1016, [0, 1]);  mul_1016 = None
        sum_398: "f32[1024]" = torch.ops.aten.sum.dim_IntList(add_344, [0, 1]);  add_344 = None
        add_345: "f32[16, 512, 1024]" = torch.ops.aten.add.Tensor(add_342, mul_1015);  add_342 = mul_1015 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:98 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_73: "f32[16, 512, 1024]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1017: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
        mul_1018: "f32[16, 512, 1024]" = torch.ops.aten.mul.Tensor(add_345, mul_1017);  add_345 = mul_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:94 in forward, code: embeddings += position_embeddings
        sum_399: "f32[1, 512, 1024]" = torch.ops.aten.sum.dim_IntList(mul_1018, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:93 in forward, code: position_embeddings = self.position_embeddings(position_ids)
        eq: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_4, -1)
        unsqueeze_4: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_4: "f32[1, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_4, full_default_3, sum_399);  unsqueeze_4 = sum_399 = None
        full_default_8: "f32[512, 1024]" = torch.ops.aten.full.default([512, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[512, 1024]" = torch.ops.aten.index_put.default(full_default_8, [primals_4], where_4, True);  full_default_8 = primals_4 = where_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:90 in forward, code: token_type_embeddings = self.token_type_embeddings(token_type_ids)
        full_default_9: "b8[16, 512, 1]" = torch.ops.aten.full.default([16, 512, 1], False, dtype = torch.bool, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[16, 512, 1024]" = torch.ops.aten.where.self(full_default_9, full_default_3, mul_1018);  full_default_9 = None
        full_default_11: "f32[2, 1024]" = torch.ops.aten.full.default([2, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[2, 1024]" = torch.ops.aten.index_put.default(full_default_11, [full_default], where_5, True);  full_default_11 = full_default = where_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/megatron_bert/modeling_megatron_bert.py:89 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_2: "b8[16, 512]" = torch.ops.aten.eq.Scalar(primals_2, 0)
        unsqueeze_6: "b8[16, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_6: "f32[16, 512, 1024]" = torch.ops.aten.where.self(unsqueeze_6, full_default_3, mul_1018);  unsqueeze_6 = full_default_3 = mul_1018 = None
        full_default_13: "f32[29056, 1024]" = torch.ops.aten.full.default([29056, 1024], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_2: "f32[29056, 1024]" = torch.ops.aten.index_put.default(full_default_13, [primals_2], where_6, True);  full_default_13 = primals_2 = where_6 = None
        add_346: "f32[29056, 1024]" = torch.ops.aten.add.Tensor(mm_1, index_put_2);  mm_1 = index_put_2 = None
        return (None, None, add_346, None, index_put_1, index_put, sum_397, sum_398, mm_291, view_1211, mm_289, view_1207, mm_287, view_1203, mm_285, view_1192, sum_388, sum_389, mm_283, view_1189, mm_281, view_1186, sum_382, sum_383, mm_279, view_1183, mm_277, view_1179, mm_275, view_1175, mm_273, view_1164, sum_373, sum_374, mm_271, view_1161, mm_269, view_1158, sum_367, sum_368, mm_267, view_1155, mm_265, view_1151, mm_263, view_1147, mm_261, view_1136, sum_358, sum_359, mm_259, view_1133, mm_257, view_1130, sum_352, sum_353, mm_255, view_1127, mm_253, view_1123, mm_251, view_1119, mm_249, view_1108, sum_343, sum_344, mm_247, view_1105, mm_245, view_1102, sum_337, sum_338, mm_243, view_1099, mm_241, view_1095, mm_239, view_1091, mm_237, view_1080, sum_328, sum_329, mm_235, view_1077, mm_233, view_1074, sum_322, sum_323, mm_231, view_1071, mm_229, view_1067, mm_227, view_1063, mm_225, view_1052, sum_313, sum_314, mm_223, view_1049, mm_221, view_1046, sum_307, sum_308, mm_219, view_1043, mm_217, view_1039, mm_215, view_1035, mm_213, view_1024, sum_298, sum_299, mm_211, view_1021, mm_209, view_1018, sum_292, sum_293, mm_207, view_1015, mm_205, view_1011, mm_203, view_1007, mm_201, view_996, sum_283, sum_284, mm_199, view_993, mm_197, view_990, sum_277, sum_278, mm_195, view_987, mm_193, view_983, mm_191, view_979, mm_189, view_968, sum_268, sum_269, mm_187, view_965, mm_185, view_962, sum_262, sum_263, mm_183, view_959, mm_181, view_955, mm_179, view_951, mm_177, view_940, sum_253, sum_254, mm_175, view_937, mm_173, view_934, sum_247, sum_248, mm_171, view_931, mm_169, view_927, mm_167, view_923, mm_165, view_912, sum_238, sum_239, mm_163, view_909, mm_161, view_906, sum_232, sum_233, mm_159, view_903, mm_157, view_899, mm_155, view_895, mm_153, view_884, sum_223, sum_224, mm_151, view_881, mm_149, view_878, sum_217, sum_218, mm_147, view_875, mm_145, view_871, mm_143, view_867, mm_141, view_856, sum_208, sum_209, mm_139, view_853, mm_137, view_850, sum_202, sum_203, mm_135, view_847, mm_133, view_843, mm_131, view_839, mm_129, view_828, sum_193, sum_194, mm_127, view_825, mm_125, view_822, sum_187, sum_188, mm_123, view_819, mm_121, view_815, mm_119, view_811, mm_117, view_800, sum_178, sum_179, mm_115, view_797, mm_113, view_794, sum_172, sum_173, mm_111, view_791, mm_109, view_787, mm_107, view_783, mm_105, view_772, sum_163, sum_164, mm_103, view_769, mm_101, view_766, sum_157, sum_158, mm_99, view_763, mm_97, view_759, mm_95, view_755, mm_93, view_744, sum_148, sum_149, mm_91, view_741, mm_89, view_738, sum_142, sum_143, mm_87, view_735, mm_85, view_731, mm_83, view_727, mm_81, view_716, sum_133, sum_134, mm_79, view_713, mm_77, view_710, sum_127, sum_128, mm_75, view_707, mm_73, view_703, mm_71, view_699, mm_69, view_688, sum_118, sum_119, mm_67, view_685, mm_65, view_682, sum_112, sum_113, mm_63, view_679, mm_61, view_675, mm_59, view_671, mm_57, view_660, sum_103, sum_104, mm_55, view_657, mm_53, view_654, sum_97, sum_98, mm_51, view_651, mm_49, view_647, mm_47, view_643, mm_45, view_632, sum_88, sum_89, mm_43, view_629, mm_41, view_626, sum_82, sum_83, mm_39, view_623, mm_37, view_619, mm_35, view_615, mm_33, view_604, sum_73, sum_74, mm_31, view_601, mm_29, view_598, sum_67, sum_68, mm_27, view_595, mm_25, view_591, mm_23, view_587, mm_21, view_576, sum_58, sum_59, mm_19, view_573, mm_17, view_570, sum_52, sum_53, mm_15, view_567, mm_13, view_563, mm_11, view_559, mm_9, view_548, sum_43, sum_44, mm_7, view_545, mm_5, view_542, sum_37, sum_38, mm_3, view_539, sum_32, sum_33, view_536)

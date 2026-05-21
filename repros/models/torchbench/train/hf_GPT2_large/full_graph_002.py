class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[4, 512]", primals_2: "f32[50257, 1280]", primals_4: "f32[1280]", primals_7: "f32[1280, 3840]", primals_9: "f32[1280, 1280]", primals_10: "f32[1280]", primals_13: "f32[1280, 5120]", primals_15: "f32[5120, 1280]", primals_16: "f32[1280]", primals_19: "f32[1280, 3840]", primals_21: "f32[1280, 1280]", primals_22: "f32[1280]", primals_25: "f32[1280, 5120]", primals_27: "f32[5120, 1280]", primals_28: "f32[1280]", primals_31: "f32[1280, 3840]", primals_33: "f32[1280, 1280]", primals_34: "f32[1280]", primals_37: "f32[1280, 5120]", primals_39: "f32[5120, 1280]", primals_40: "f32[1280]", primals_43: "f32[1280, 3840]", primals_45: "f32[1280, 1280]", primals_46: "f32[1280]", primals_49: "f32[1280, 5120]", primals_51: "f32[5120, 1280]", primals_52: "f32[1280]", primals_55: "f32[1280, 3840]", primals_57: "f32[1280, 1280]", primals_58: "f32[1280]", primals_61: "f32[1280, 5120]", primals_63: "f32[5120, 1280]", primals_64: "f32[1280]", primals_67: "f32[1280, 3840]", primals_69: "f32[1280, 1280]", primals_70: "f32[1280]", primals_73: "f32[1280, 5120]", primals_75: "f32[5120, 1280]", primals_76: "f32[1280]", primals_79: "f32[1280, 3840]", primals_81: "f32[1280, 1280]", primals_82: "f32[1280]", primals_85: "f32[1280, 5120]", primals_87: "f32[5120, 1280]", primals_88: "f32[1280]", primals_91: "f32[1280, 3840]", primals_93: "f32[1280, 1280]", primals_94: "f32[1280]", primals_97: "f32[1280, 5120]", primals_99: "f32[5120, 1280]", primals_100: "f32[1280]", primals_103: "f32[1280, 3840]", primals_105: "f32[1280, 1280]", primals_106: "f32[1280]", primals_109: "f32[1280, 5120]", primals_111: "f32[5120, 1280]", primals_112: "f32[1280]", primals_115: "f32[1280, 3840]", primals_117: "f32[1280, 1280]", primals_118: "f32[1280]", primals_121: "f32[1280, 5120]", primals_123: "f32[5120, 1280]", primals_124: "f32[1280]", primals_127: "f32[1280, 3840]", primals_129: "f32[1280, 1280]", primals_130: "f32[1280]", primals_133: "f32[1280, 5120]", primals_135: "f32[5120, 1280]", primals_136: "f32[1280]", primals_139: "f32[1280, 3840]", primals_141: "f32[1280, 1280]", primals_142: "f32[1280]", primals_145: "f32[1280, 5120]", primals_147: "f32[5120, 1280]", primals_148: "f32[1280]", primals_151: "f32[1280, 3840]", primals_153: "f32[1280, 1280]", primals_154: "f32[1280]", primals_157: "f32[1280, 5120]", primals_159: "f32[5120, 1280]", primals_160: "f32[1280]", primals_163: "f32[1280, 3840]", primals_165: "f32[1280, 1280]", primals_166: "f32[1280]", primals_169: "f32[1280, 5120]", primals_171: "f32[5120, 1280]", primals_172: "f32[1280]", primals_175: "f32[1280, 3840]", primals_177: "f32[1280, 1280]", primals_178: "f32[1280]", primals_181: "f32[1280, 5120]", primals_183: "f32[5120, 1280]", primals_184: "f32[1280]", primals_187: "f32[1280, 3840]", primals_189: "f32[1280, 1280]", primals_190: "f32[1280]", primals_193: "f32[1280, 5120]", primals_195: "f32[5120, 1280]", primals_196: "f32[1280]", primals_199: "f32[1280, 3840]", primals_201: "f32[1280, 1280]", primals_202: "f32[1280]", primals_205: "f32[1280, 5120]", primals_207: "f32[5120, 1280]", primals_208: "f32[1280]", primals_211: "f32[1280, 3840]", primals_213: "f32[1280, 1280]", primals_214: "f32[1280]", primals_217: "f32[1280, 5120]", primals_219: "f32[5120, 1280]", primals_220: "f32[1280]", primals_223: "f32[1280, 3840]", primals_225: "f32[1280, 1280]", primals_226: "f32[1280]", primals_229: "f32[1280, 5120]", primals_231: "f32[5120, 1280]", primals_232: "f32[1280]", primals_235: "f32[1280, 3840]", primals_237: "f32[1280, 1280]", primals_238: "f32[1280]", primals_241: "f32[1280, 5120]", primals_243: "f32[5120, 1280]", primals_244: "f32[1280]", primals_247: "f32[1280, 3840]", primals_249: "f32[1280, 1280]", primals_250: "f32[1280]", primals_253: "f32[1280, 5120]", primals_255: "f32[5120, 1280]", primals_256: "f32[1280]", primals_259: "f32[1280, 3840]", primals_261: "f32[1280, 1280]", primals_262: "f32[1280]", primals_265: "f32[1280, 5120]", primals_267: "f32[5120, 1280]", primals_268: "f32[1280]", primals_271: "f32[1280, 3840]", primals_273: "f32[1280, 1280]", primals_274: "f32[1280]", primals_277: "f32[1280, 5120]", primals_279: "f32[5120, 1280]", primals_280: "f32[1280]", primals_283: "f32[1280, 3840]", primals_285: "f32[1280, 1280]", primals_286: "f32[1280]", primals_289: "f32[1280, 5120]", primals_291: "f32[5120, 1280]", primals_292: "f32[1280]", primals_295: "f32[1280, 3840]", primals_297: "f32[1280, 1280]", primals_298: "f32[1280]", primals_301: "f32[1280, 5120]", primals_303: "f32[5120, 1280]", primals_304: "f32[1280]", primals_307: "f32[1280, 3840]", primals_309: "f32[1280, 1280]", primals_310: "f32[1280]", primals_313: "f32[1280, 5120]", primals_315: "f32[5120, 1280]", primals_316: "f32[1280]", primals_319: "f32[1280, 3840]", primals_321: "f32[1280, 1280]", primals_322: "f32[1280]", primals_325: "f32[1280, 5120]", primals_327: "f32[5120, 1280]", primals_328: "f32[1280]", primals_331: "f32[1280, 3840]", primals_333: "f32[1280, 1280]", primals_334: "f32[1280]", primals_337: "f32[1280, 5120]", primals_339: "f32[5120, 1280]", primals_340: "f32[1280]", primals_343: "f32[1280, 3840]", primals_345: "f32[1280, 1280]", primals_346: "f32[1280]", primals_349: "f32[1280, 5120]", primals_351: "f32[5120, 1280]", primals_352: "f32[1280]", primals_355: "f32[1280, 3840]", primals_357: "f32[1280, 1280]", primals_358: "f32[1280]", primals_361: "f32[1280, 5120]", primals_363: "f32[5120, 1280]", primals_364: "f32[1280]", primals_367: "f32[1280, 3840]", primals_369: "f32[1280, 1280]", primals_370: "f32[1280]", primals_373: "f32[1280, 5120]", primals_375: "f32[5120, 1280]", primals_376: "f32[1280]", primals_379: "f32[1280, 3840]", primals_381: "f32[1280, 1280]", primals_382: "f32[1280]", primals_385: "f32[1280, 5120]", primals_387: "f32[5120, 1280]", primals_388: "f32[1280]", primals_391: "f32[1280, 3840]", primals_393: "f32[1280, 1280]", primals_394: "f32[1280]", primals_397: "f32[1280, 5120]", primals_399: "f32[5120, 1280]", primals_400: "f32[1280]", primals_403: "f32[1280, 3840]", primals_405: "f32[1280, 1280]", primals_406: "f32[1280]", primals_409: "f32[1280, 5120]", primals_411: "f32[5120, 1280]", primals_412: "f32[1280]", primals_415: "f32[1280, 3840]", primals_417: "f32[1280, 1280]", primals_418: "f32[1280]", primals_421: "f32[1280, 5120]", primals_423: "f32[5120, 1280]", primals_424: "f32[1280]", primals_427: "f32[1280, 3840]", primals_429: "f32[1280, 1280]", primals_430: "f32[1280]", primals_433: "f32[1280, 5120]", primals_435: "f32[5120, 1280]", primals_436: "f32[1280]", unsqueeze: "i64[1, 512]", gt: "b8[4, 512, 1280]", mul_2: "f32[4, 512, 1280]", permute: "f32[4, 20, 512, 64]", permute_1: "f32[4, 20, 512, 64]", permute_2: "f32[4, 20, 512, 64]", where: "f32[4, 1, 512, 512]", getitem_5: "f32[4, 20, 512, 64]", getitem_6: "f32[4, 20, 512]", getitem_7: "i64[]", getitem_8: "i64[]", gt_1: "b8[4, 512, 1280]", mul_6: "f32[4, 512, 1280]", addmm_2: "f32[2048, 5120]", gt_2: "b8[4, 512, 1280]", mul_14: "f32[4, 512, 1280]", permute_4: "f32[4, 20, 512, 64]", permute_5: "f32[4, 20, 512, 64]", permute_6: "f32[4, 20, 512, 64]", getitem_16: "f32[4, 20, 512, 64]", getitem_17: "f32[4, 20, 512]", getitem_18: "i64[]", getitem_19: "i64[]", gt_3: "b8[4, 512, 1280]", mul_18: "f32[4, 512, 1280]", addmm_6: "f32[2048, 5120]", gt_4: "b8[4, 512, 1280]", mul_26: "f32[4, 512, 1280]", permute_8: "f32[4, 20, 512, 64]", permute_9: "f32[4, 20, 512, 64]", permute_10: "f32[4, 20, 512, 64]", getitem_27: "f32[4, 20, 512, 64]", getitem_28: "f32[4, 20, 512]", getitem_29: "i64[]", getitem_30: "i64[]", gt_5: "b8[4, 512, 1280]", mul_30: "f32[4, 512, 1280]", addmm_10: "f32[2048, 5120]", gt_6: "b8[4, 512, 1280]", mul_38: "f32[4, 512, 1280]", permute_12: "f32[4, 20, 512, 64]", permute_13: "f32[4, 20, 512, 64]", permute_14: "f32[4, 20, 512, 64]", getitem_38: "f32[4, 20, 512, 64]", getitem_39: "f32[4, 20, 512]", getitem_40: "i64[]", getitem_41: "i64[]", gt_7: "b8[4, 512, 1280]", mul_42: "f32[4, 512, 1280]", addmm_14: "f32[2048, 5120]", gt_8: "b8[4, 512, 1280]", mul_50: "f32[4, 512, 1280]", permute_16: "f32[4, 20, 512, 64]", permute_17: "f32[4, 20, 512, 64]", permute_18: "f32[4, 20, 512, 64]", getitem_49: "f32[4, 20, 512, 64]", getitem_50: "f32[4, 20, 512]", getitem_51: "i64[]", getitem_52: "i64[]", gt_9: "b8[4, 512, 1280]", mul_54: "f32[4, 512, 1280]", addmm_18: "f32[2048, 5120]", gt_10: "b8[4, 512, 1280]", mul_62: "f32[4, 512, 1280]", permute_20: "f32[4, 20, 512, 64]", permute_21: "f32[4, 20, 512, 64]", permute_22: "f32[4, 20, 512, 64]", getitem_60: "f32[4, 20, 512, 64]", getitem_61: "f32[4, 20, 512]", getitem_62: "i64[]", getitem_63: "i64[]", gt_11: "b8[4, 512, 1280]", mul_66: "f32[4, 512, 1280]", addmm_22: "f32[2048, 5120]", gt_12: "b8[4, 512, 1280]", mul_74: "f32[4, 512, 1280]", permute_24: "f32[4, 20, 512, 64]", permute_25: "f32[4, 20, 512, 64]", permute_26: "f32[4, 20, 512, 64]", getitem_71: "f32[4, 20, 512, 64]", getitem_72: "f32[4, 20, 512]", getitem_73: "i64[]", getitem_74: "i64[]", gt_13: "b8[4, 512, 1280]", mul_78: "f32[4, 512, 1280]", addmm_26: "f32[2048, 5120]", gt_14: "b8[4, 512, 1280]", mul_86: "f32[4, 512, 1280]", permute_28: "f32[4, 20, 512, 64]", permute_29: "f32[4, 20, 512, 64]", permute_30: "f32[4, 20, 512, 64]", getitem_82: "f32[4, 20, 512, 64]", getitem_83: "f32[4, 20, 512]", getitem_84: "i64[]", getitem_85: "i64[]", gt_15: "b8[4, 512, 1280]", mul_90: "f32[4, 512, 1280]", addmm_30: "f32[2048, 5120]", gt_16: "b8[4, 512, 1280]", mul_98: "f32[4, 512, 1280]", permute_32: "f32[4, 20, 512, 64]", permute_33: "f32[4, 20, 512, 64]", permute_34: "f32[4, 20, 512, 64]", getitem_93: "f32[4, 20, 512, 64]", getitem_94: "f32[4, 20, 512]", getitem_95: "i64[]", getitem_96: "i64[]", gt_17: "b8[4, 512, 1280]", mul_102: "f32[4, 512, 1280]", addmm_34: "f32[2048, 5120]", gt_18: "b8[4, 512, 1280]", mul_110: "f32[4, 512, 1280]", permute_36: "f32[4, 20, 512, 64]", permute_37: "f32[4, 20, 512, 64]", permute_38: "f32[4, 20, 512, 64]", getitem_104: "f32[4, 20, 512, 64]", getitem_105: "f32[4, 20, 512]", getitem_106: "i64[]", getitem_107: "i64[]", gt_19: "b8[4, 512, 1280]", mul_114: "f32[4, 512, 1280]", addmm_38: "f32[2048, 5120]", gt_20: "b8[4, 512, 1280]", mul_122: "f32[4, 512, 1280]", permute_40: "f32[4, 20, 512, 64]", permute_41: "f32[4, 20, 512, 64]", permute_42: "f32[4, 20, 512, 64]", getitem_115: "f32[4, 20, 512, 64]", getitem_116: "f32[4, 20, 512]", getitem_117: "i64[]", getitem_118: "i64[]", gt_21: "b8[4, 512, 1280]", mul_126: "f32[4, 512, 1280]", addmm_42: "f32[2048, 5120]", gt_22: "b8[4, 512, 1280]", mul_134: "f32[4, 512, 1280]", permute_44: "f32[4, 20, 512, 64]", permute_45: "f32[4, 20, 512, 64]", permute_46: "f32[4, 20, 512, 64]", getitem_126: "f32[4, 20, 512, 64]", getitem_127: "f32[4, 20, 512]", getitem_128: "i64[]", getitem_129: "i64[]", gt_23: "b8[4, 512, 1280]", mul_138: "f32[4, 512, 1280]", addmm_46: "f32[2048, 5120]", gt_24: "b8[4, 512, 1280]", mul_146: "f32[4, 512, 1280]", permute_48: "f32[4, 20, 512, 64]", permute_49: "f32[4, 20, 512, 64]", permute_50: "f32[4, 20, 512, 64]", getitem_137: "f32[4, 20, 512, 64]", getitem_138: "f32[4, 20, 512]", getitem_139: "i64[]", getitem_140: "i64[]", gt_25: "b8[4, 512, 1280]", mul_150: "f32[4, 512, 1280]", addmm_50: "f32[2048, 5120]", gt_26: "b8[4, 512, 1280]", mul_158: "f32[4, 512, 1280]", permute_52: "f32[4, 20, 512, 64]", permute_53: "f32[4, 20, 512, 64]", permute_54: "f32[4, 20, 512, 64]", getitem_148: "f32[4, 20, 512, 64]", getitem_149: "f32[4, 20, 512]", getitem_150: "i64[]", getitem_151: "i64[]", gt_27: "b8[4, 512, 1280]", mul_162: "f32[4, 512, 1280]", addmm_54: "f32[2048, 5120]", gt_28: "b8[4, 512, 1280]", mul_170: "f32[4, 512, 1280]", permute_56: "f32[4, 20, 512, 64]", permute_57: "f32[4, 20, 512, 64]", permute_58: "f32[4, 20, 512, 64]", getitem_159: "f32[4, 20, 512, 64]", getitem_160: "f32[4, 20, 512]", getitem_161: "i64[]", getitem_162: "i64[]", gt_29: "b8[4, 512, 1280]", mul_174: "f32[4, 512, 1280]", addmm_58: "f32[2048, 5120]", gt_30: "b8[4, 512, 1280]", mul_182: "f32[4, 512, 1280]", permute_60: "f32[4, 20, 512, 64]", permute_61: "f32[4, 20, 512, 64]", permute_62: "f32[4, 20, 512, 64]", getitem_170: "f32[4, 20, 512, 64]", getitem_171: "f32[4, 20, 512]", getitem_172: "i64[]", getitem_173: "i64[]", gt_31: "b8[4, 512, 1280]", mul_186: "f32[4, 512, 1280]", addmm_62: "f32[2048, 5120]", gt_32: "b8[4, 512, 1280]", mul_194: "f32[4, 512, 1280]", permute_64: "f32[4, 20, 512, 64]", permute_65: "f32[4, 20, 512, 64]", permute_66: "f32[4, 20, 512, 64]", getitem_181: "f32[4, 20, 512, 64]", getitem_182: "f32[4, 20, 512]", getitem_183: "i64[]", getitem_184: "i64[]", gt_33: "b8[4, 512, 1280]", mul_198: "f32[4, 512, 1280]", addmm_66: "f32[2048, 5120]", gt_34: "b8[4, 512, 1280]", mul_206: "f32[4, 512, 1280]", permute_68: "f32[4, 20, 512, 64]", permute_69: "f32[4, 20, 512, 64]", permute_70: "f32[4, 20, 512, 64]", getitem_192: "f32[4, 20, 512, 64]", getitem_193: "f32[4, 20, 512]", getitem_194: "i64[]", getitem_195: "i64[]", gt_35: "b8[4, 512, 1280]", mul_210: "f32[4, 512, 1280]", addmm_70: "f32[2048, 5120]", gt_36: "b8[4, 512, 1280]", mul_218: "f32[4, 512, 1280]", permute_72: "f32[4, 20, 512, 64]", permute_73: "f32[4, 20, 512, 64]", permute_74: "f32[4, 20, 512, 64]", getitem_203: "f32[4, 20, 512, 64]", getitem_204: "f32[4, 20, 512]", getitem_205: "i64[]", getitem_206: "i64[]", gt_37: "b8[4, 512, 1280]", mul_222: "f32[4, 512, 1280]", addmm_74: "f32[2048, 5120]", gt_38: "b8[4, 512, 1280]", mul_230: "f32[4, 512, 1280]", permute_76: "f32[4, 20, 512, 64]", permute_77: "f32[4, 20, 512, 64]", permute_78: "f32[4, 20, 512, 64]", getitem_214: "f32[4, 20, 512, 64]", getitem_215: "f32[4, 20, 512]", getitem_216: "i64[]", getitem_217: "i64[]", gt_39: "b8[4, 512, 1280]", mul_234: "f32[4, 512, 1280]", addmm_78: "f32[2048, 5120]", gt_40: "b8[4, 512, 1280]", mul_242: "f32[4, 512, 1280]", permute_80: "f32[4, 20, 512, 64]", permute_81: "f32[4, 20, 512, 64]", permute_82: "f32[4, 20, 512, 64]", getitem_225: "f32[4, 20, 512, 64]", getitem_226: "f32[4, 20, 512]", getitem_227: "i64[]", getitem_228: "i64[]", gt_41: "b8[4, 512, 1280]", mul_246: "f32[4, 512, 1280]", addmm_82: "f32[2048, 5120]", gt_42: "b8[4, 512, 1280]", mul_254: "f32[4, 512, 1280]", permute_84: "f32[4, 20, 512, 64]", permute_85: "f32[4, 20, 512, 64]", permute_86: "f32[4, 20, 512, 64]", getitem_236: "f32[4, 20, 512, 64]", getitem_237: "f32[4, 20, 512]", getitem_238: "i64[]", getitem_239: "i64[]", gt_43: "b8[4, 512, 1280]", mul_258: "f32[4, 512, 1280]", addmm_86: "f32[2048, 5120]", gt_44: "b8[4, 512, 1280]", mul_266: "f32[4, 512, 1280]", permute_88: "f32[4, 20, 512, 64]", permute_89: "f32[4, 20, 512, 64]", permute_90: "f32[4, 20, 512, 64]", getitem_247: "f32[4, 20, 512, 64]", getitem_248: "f32[4, 20, 512]", getitem_249: "i64[]", getitem_250: "i64[]", gt_45: "b8[4, 512, 1280]", mul_270: "f32[4, 512, 1280]", addmm_90: "f32[2048, 5120]", gt_46: "b8[4, 512, 1280]", mul_278: "f32[4, 512, 1280]", permute_92: "f32[4, 20, 512, 64]", permute_93: "f32[4, 20, 512, 64]", permute_94: "f32[4, 20, 512, 64]", getitem_258: "f32[4, 20, 512, 64]", getitem_259: "f32[4, 20, 512]", getitem_260: "i64[]", getitem_261: "i64[]", gt_47: "b8[4, 512, 1280]", mul_282: "f32[4, 512, 1280]", addmm_94: "f32[2048, 5120]", gt_48: "b8[4, 512, 1280]", mul_290: "f32[4, 512, 1280]", permute_96: "f32[4, 20, 512, 64]", permute_97: "f32[4, 20, 512, 64]", permute_98: "f32[4, 20, 512, 64]", getitem_269: "f32[4, 20, 512, 64]", getitem_270: "f32[4, 20, 512]", getitem_271: "i64[]", getitem_272: "i64[]", gt_49: "b8[4, 512, 1280]", mul_294: "f32[4, 512, 1280]", addmm_98: "f32[2048, 5120]", gt_50: "b8[4, 512, 1280]", mul_302: "f32[4, 512, 1280]", permute_100: "f32[4, 20, 512, 64]", permute_101: "f32[4, 20, 512, 64]", permute_102: "f32[4, 20, 512, 64]", getitem_280: "f32[4, 20, 512, 64]", getitem_281: "f32[4, 20, 512]", getitem_282: "i64[]", getitem_283: "i64[]", gt_51: "b8[4, 512, 1280]", mul_306: "f32[4, 512, 1280]", addmm_102: "f32[2048, 5120]", gt_52: "b8[4, 512, 1280]", mul_314: "f32[4, 512, 1280]", permute_104: "f32[4, 20, 512, 64]", permute_105: "f32[4, 20, 512, 64]", permute_106: "f32[4, 20, 512, 64]", getitem_291: "f32[4, 20, 512, 64]", getitem_292: "f32[4, 20, 512]", getitem_293: "i64[]", getitem_294: "i64[]", gt_53: "b8[4, 512, 1280]", mul_318: "f32[4, 512, 1280]", addmm_106: "f32[2048, 5120]", gt_54: "b8[4, 512, 1280]", mul_326: "f32[4, 512, 1280]", permute_108: "f32[4, 20, 512, 64]", permute_109: "f32[4, 20, 512, 64]", permute_110: "f32[4, 20, 512, 64]", getitem_302: "f32[4, 20, 512, 64]", getitem_303: "f32[4, 20, 512]", getitem_304: "i64[]", getitem_305: "i64[]", gt_55: "b8[4, 512, 1280]", mul_330: "f32[4, 512, 1280]", addmm_110: "f32[2048, 5120]", gt_56: "b8[4, 512, 1280]", mul_338: "f32[4, 512, 1280]", permute_112: "f32[4, 20, 512, 64]", permute_113: "f32[4, 20, 512, 64]", permute_114: "f32[4, 20, 512, 64]", getitem_313: "f32[4, 20, 512, 64]", getitem_314: "f32[4, 20, 512]", getitem_315: "i64[]", getitem_316: "i64[]", gt_57: "b8[4, 512, 1280]", mul_342: "f32[4, 512, 1280]", addmm_114: "f32[2048, 5120]", gt_58: "b8[4, 512, 1280]", mul_350: "f32[4, 512, 1280]", permute_116: "f32[4, 20, 512, 64]", permute_117: "f32[4, 20, 512, 64]", permute_118: "f32[4, 20, 512, 64]", getitem_324: "f32[4, 20, 512, 64]", getitem_325: "f32[4, 20, 512]", getitem_326: "i64[]", getitem_327: "i64[]", gt_59: "b8[4, 512, 1280]", mul_354: "f32[4, 512, 1280]", addmm_118: "f32[2048, 5120]", gt_60: "b8[4, 512, 1280]", mul_362: "f32[4, 512, 1280]", permute_120: "f32[4, 20, 512, 64]", permute_121: "f32[4, 20, 512, 64]", permute_122: "f32[4, 20, 512, 64]", getitem_335: "f32[4, 20, 512, 64]", getitem_336: "f32[4, 20, 512]", getitem_337: "i64[]", getitem_338: "i64[]", gt_61: "b8[4, 512, 1280]", mul_366: "f32[4, 512, 1280]", addmm_122: "f32[2048, 5120]", gt_62: "b8[4, 512, 1280]", mul_374: "f32[4, 512, 1280]", permute_124: "f32[4, 20, 512, 64]", permute_125: "f32[4, 20, 512, 64]", permute_126: "f32[4, 20, 512, 64]", getitem_346: "f32[4, 20, 512, 64]", getitem_347: "f32[4, 20, 512]", getitem_348: "i64[]", getitem_349: "i64[]", gt_63: "b8[4, 512, 1280]", mul_378: "f32[4, 512, 1280]", addmm_126: "f32[2048, 5120]", gt_64: "b8[4, 512, 1280]", mul_386: "f32[4, 512, 1280]", permute_128: "f32[4, 20, 512, 64]", permute_129: "f32[4, 20, 512, 64]", permute_130: "f32[4, 20, 512, 64]", getitem_357: "f32[4, 20, 512, 64]", getitem_358: "f32[4, 20, 512]", getitem_359: "i64[]", getitem_360: "i64[]", gt_65: "b8[4, 512, 1280]", mul_390: "f32[4, 512, 1280]", addmm_130: "f32[2048, 5120]", gt_66: "b8[4, 512, 1280]", mul_398: "f32[4, 512, 1280]", permute_132: "f32[4, 20, 512, 64]", permute_133: "f32[4, 20, 512, 64]", permute_134: "f32[4, 20, 512, 64]", getitem_368: "f32[4, 20, 512, 64]", getitem_369: "f32[4, 20, 512]", getitem_370: "i64[]", getitem_371: "i64[]", gt_67: "b8[4, 512, 1280]", mul_402: "f32[4, 512, 1280]", addmm_134: "f32[2048, 5120]", gt_68: "b8[4, 512, 1280]", mul_410: "f32[4, 512, 1280]", permute_136: "f32[4, 20, 512, 64]", permute_137: "f32[4, 20, 512, 64]", permute_138: "f32[4, 20, 512, 64]", getitem_379: "f32[4, 20, 512, 64]", getitem_380: "f32[4, 20, 512]", getitem_381: "i64[]", getitem_382: "i64[]", gt_69: "b8[4, 512, 1280]", mul_414: "f32[4, 512, 1280]", addmm_138: "f32[2048, 5120]", gt_70: "b8[4, 512, 1280]", mul_422: "f32[4, 512, 1280]", permute_140: "f32[4, 20, 512, 64]", permute_141: "f32[4, 20, 512, 64]", permute_142: "f32[4, 20, 512, 64]", getitem_390: "f32[4, 20, 512, 64]", getitem_391: "f32[4, 20, 512]", getitem_392: "i64[]", getitem_393: "i64[]", gt_71: "b8[4, 512, 1280]", mul_426: "f32[4, 512, 1280]", addmm_142: "f32[2048, 5120]", gt_72: "b8[4, 512, 1280]", mul_434: "f32[4, 512, 1280]", view_434: "f32[2048, 1280]", div: "f32[4, 512, 1]", permute_150: "f32[5120, 2048]", permute_152: "f32[1280, 2048]", div_1: "f32[4, 512, 1]", permute_160: "f32[1280, 2048]", div_2: "f32[4, 512, 1]", permute_162: "f32[5120, 2048]", permute_164: "f32[1280, 2048]", div_3: "f32[4, 512, 1]", permute_172: "f32[1280, 2048]", div_4: "f32[4, 512, 1]", permute_174: "f32[5120, 2048]", permute_176: "f32[1280, 2048]", div_5: "f32[4, 512, 1]", permute_184: "f32[1280, 2048]", div_6: "f32[4, 512, 1]", permute_186: "f32[5120, 2048]", permute_188: "f32[1280, 2048]", div_7: "f32[4, 512, 1]", permute_196: "f32[1280, 2048]", div_8: "f32[4, 512, 1]", permute_198: "f32[5120, 2048]", permute_200: "f32[1280, 2048]", div_9: "f32[4, 512, 1]", permute_208: "f32[1280, 2048]", div_10: "f32[4, 512, 1]", permute_210: "f32[5120, 2048]", permute_212: "f32[1280, 2048]", div_11: "f32[4, 512, 1]", permute_220: "f32[1280, 2048]", div_12: "f32[4, 512, 1]", permute_222: "f32[5120, 2048]", permute_224: "f32[1280, 2048]", div_13: "f32[4, 512, 1]", permute_232: "f32[1280, 2048]", div_14: "f32[4, 512, 1]", permute_234: "f32[5120, 2048]", permute_236: "f32[1280, 2048]", div_15: "f32[4, 512, 1]", permute_244: "f32[1280, 2048]", div_16: "f32[4, 512, 1]", permute_246: "f32[5120, 2048]", permute_248: "f32[1280, 2048]", div_17: "f32[4, 512, 1]", permute_256: "f32[1280, 2048]", div_18: "f32[4, 512, 1]", permute_258: "f32[5120, 2048]", permute_260: "f32[1280, 2048]", div_19: "f32[4, 512, 1]", permute_268: "f32[1280, 2048]", div_20: "f32[4, 512, 1]", permute_270: "f32[5120, 2048]", permute_272: "f32[1280, 2048]", div_21: "f32[4, 512, 1]", permute_280: "f32[1280, 2048]", div_22: "f32[4, 512, 1]", permute_282: "f32[5120, 2048]", permute_284: "f32[1280, 2048]", div_23: "f32[4, 512, 1]", permute_292: "f32[1280, 2048]", div_24: "f32[4, 512, 1]", permute_294: "f32[5120, 2048]", permute_296: "f32[1280, 2048]", div_25: "f32[4, 512, 1]", permute_304: "f32[1280, 2048]", div_26: "f32[4, 512, 1]", permute_306: "f32[5120, 2048]", permute_308: "f32[1280, 2048]", div_27: "f32[4, 512, 1]", permute_316: "f32[1280, 2048]", div_28: "f32[4, 512, 1]", permute_318: "f32[5120, 2048]", permute_320: "f32[1280, 2048]", div_29: "f32[4, 512, 1]", permute_328: "f32[1280, 2048]", div_30: "f32[4, 512, 1]", permute_330: "f32[5120, 2048]", permute_332: "f32[1280, 2048]", div_31: "f32[4, 512, 1]", permute_340: "f32[1280, 2048]", div_32: "f32[4, 512, 1]", permute_342: "f32[5120, 2048]", permute_344: "f32[1280, 2048]", div_33: "f32[4, 512, 1]", permute_352: "f32[1280, 2048]", div_34: "f32[4, 512, 1]", permute_354: "f32[5120, 2048]", permute_356: "f32[1280, 2048]", div_35: "f32[4, 512, 1]", permute_364: "f32[1280, 2048]", div_36: "f32[4, 512, 1]", permute_366: "f32[5120, 2048]", permute_368: "f32[1280, 2048]", div_37: "f32[4, 512, 1]", permute_376: "f32[1280, 2048]", div_38: "f32[4, 512, 1]", permute_378: "f32[5120, 2048]", permute_380: "f32[1280, 2048]", div_39: "f32[4, 512, 1]", permute_388: "f32[1280, 2048]", div_40: "f32[4, 512, 1]", permute_390: "f32[5120, 2048]", permute_392: "f32[1280, 2048]", div_41: "f32[4, 512, 1]", permute_400: "f32[1280, 2048]", div_42: "f32[4, 512, 1]", permute_402: "f32[5120, 2048]", permute_404: "f32[1280, 2048]", div_43: "f32[4, 512, 1]", permute_412: "f32[1280, 2048]", div_44: "f32[4, 512, 1]", permute_414: "f32[5120, 2048]", permute_416: "f32[1280, 2048]", div_45: "f32[4, 512, 1]", permute_424: "f32[1280, 2048]", div_46: "f32[4, 512, 1]", permute_426: "f32[5120, 2048]", permute_428: "f32[1280, 2048]", div_47: "f32[4, 512, 1]", permute_436: "f32[1280, 2048]", div_48: "f32[4, 512, 1]", permute_438: "f32[5120, 2048]", permute_440: "f32[1280, 2048]", div_49: "f32[4, 512, 1]", permute_448: "f32[1280, 2048]", div_50: "f32[4, 512, 1]", permute_450: "f32[5120, 2048]", permute_452: "f32[1280, 2048]", div_51: "f32[4, 512, 1]", permute_460: "f32[1280, 2048]", div_52: "f32[4, 512, 1]", permute_462: "f32[5120, 2048]", permute_464: "f32[1280, 2048]", div_53: "f32[4, 512, 1]", permute_472: "f32[1280, 2048]", div_54: "f32[4, 512, 1]", permute_474: "f32[5120, 2048]", permute_476: "f32[1280, 2048]", div_55: "f32[4, 512, 1]", permute_484: "f32[1280, 2048]", div_56: "f32[4, 512, 1]", permute_486: "f32[5120, 2048]", permute_488: "f32[1280, 2048]", div_57: "f32[4, 512, 1]", permute_496: "f32[1280, 2048]", div_58: "f32[4, 512, 1]", permute_498: "f32[5120, 2048]", permute_500: "f32[1280, 2048]", div_59: "f32[4, 512, 1]", permute_508: "f32[1280, 2048]", div_60: "f32[4, 512, 1]", permute_510: "f32[5120, 2048]", permute_512: "f32[1280, 2048]", div_61: "f32[4, 512, 1]", permute_520: "f32[1280, 2048]", div_62: "f32[4, 512, 1]", permute_522: "f32[5120, 2048]", permute_524: "f32[1280, 2048]", div_63: "f32[4, 512, 1]", permute_532: "f32[1280, 2048]", div_64: "f32[4, 512, 1]", permute_534: "f32[5120, 2048]", permute_536: "f32[1280, 2048]", div_65: "f32[4, 512, 1]", permute_544: "f32[1280, 2048]", div_66: "f32[4, 512, 1]", permute_546: "f32[5120, 2048]", permute_548: "f32[1280, 2048]", div_67: "f32[4, 512, 1]", permute_556: "f32[1280, 2048]", div_68: "f32[4, 512, 1]", permute_558: "f32[5120, 2048]", permute_560: "f32[1280, 2048]", div_69: "f32[4, 512, 1]", permute_568: "f32[1280, 2048]", div_70: "f32[4, 512, 1]", permute_570: "f32[5120, 2048]", permute_572: "f32[1280, 2048]", div_71: "f32[4, 512, 1]", permute_580: "f32[1280, 2048]", div_72: "f32[4, 512, 1]", tangents_1: "f32[4, 512, 50257]", tangents_2: "f32[4, 512, 1280]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:706 in forward, code: logits = self.lm_head(hidden_states[:, slice_indices, :])
        view_436: "f32[2048, 50257]" = torch.ops.aten.reshape.default(tangents_1, [2048, 50257]);  tangents_1 = None
        permute_145: "f32[50257, 2048]" = torch.ops.aten.permute.default(view_436, [1, 0])
        mm_1: "f32[50257, 1280]" = torch.ops.aten.mm.default(permute_145, view_434);  permute_145 = view_434 = None
        permute_144: "f32[1280, 50257]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_147: "f32[50257, 1280]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_2: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_436, permute_147);  view_436 = permute_147 = None
        view_437: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_2, [4, 512, 1280]);  mm_2 = None
        add_294: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(tangents_2, view_437);  tangents_2 = view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:628 in forward, code: hidden_states = self.ln_f(hidden_states)
        mul_437: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_294, primals_436);  primals_436 = None
        mul_438: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_437, 1280)
        sum_1: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_437, [2], True)
        mul_439: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_437, mul_434);  mul_437 = None
        sum_2: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        mul_440: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_434, sum_2);  sum_2 = None
        sub_76: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_438, sum_1);  mul_438 = sum_1 = None
        sub_77: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_76, mul_440);  sub_76 = mul_440 = None
        mul_441: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div, sub_77);  div = sub_77 = None
        mul_442: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_294, mul_434);  mul_434 = None
        sum_3: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_442, [0, 1]);  mul_442 = None
        sum_4: "f32[1280]" = torch.ops.aten.sum.dim_IntList(add_294, [0, 1]);  add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_443: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type, 1.1111111111111112);  convert_element_type = None
        mul_444: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_441, mul_443);  mul_443 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_439: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_444, [2048, 1280]);  mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_149: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_435, [1, 0]);  primals_435 = None
        mm_3: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_439, permute_149);  permute_149 = None
        mm_4: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_150, view_439);  permute_150 = None
        sum_5: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_439, [0], True);  view_439 = None
        view_440: "f32[1280]" = torch.ops.aten.reshape.default(sum_5, [1280]);  sum_5 = None
        view_441: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_3, [4, 512, 5120]);  mm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_430: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_142, [4, 512, 5120]);  addmm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_428: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_430, 0.5)
        mul_445: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_441, mul_428);  mul_428 = None
        pow_36: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_430, 3.0)
        mul_429: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_36, 0.044715);  pow_36 = None
        add_289: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_430, mul_429);  mul_429 = None
        mul_430: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_289, 0.7978845608028654);  add_289 = None
        tanh_35: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_430);  mul_430 = None
        add_290: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_35, 1.0)
        mul_446: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_441, add_290);  view_441 = add_290 = None
        mul_447: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_35, tanh_35);  tanh_35 = None
        sub_78: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_447);  mul_447 = None
        mul_448: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_445, sub_78);  mul_445 = sub_78 = None
        mul_449: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_448, 0.7978845608028654);  mul_448 = None
        mul_450: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_449, 0.044715)
        pow_37: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_430, 2.0);  view_430 = None
        mul_451: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_37, 3.0);  pow_37 = None
        mul_452: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_450, mul_451);  mul_450 = mul_451 = None
        add_295: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_449, mul_452);  mul_449 = mul_452 = None
        mul_453: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_446, 0.5);  mul_446 = None
        add_296: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_295, mul_453);  add_295 = mul_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_442: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_296, [2048, 5120]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_151: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_433, [1, 0]);  primals_433 = None
        mm_5: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_442, permute_151);  permute_151 = None
        mm_6: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_152, view_442);  permute_152 = None
        sum_6: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_442, [0], True);  view_442 = None
        view_443: "f32[5120]" = torch.ops.aten.reshape.default(sum_6, [5120]);  sum_6 = None
        view_444: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_5, [4, 512, 1280]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_455: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_444, primals_430);  primals_430 = None
        mul_456: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_455, 1280)
        sum_7: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_455, [2], True)
        mul_457: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_455, mul_426);  mul_455 = None
        sum_8: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_457, [2], True);  mul_457 = None
        mul_458: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_426, sum_8);  sum_8 = None
        sub_80: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_456, sum_7);  mul_456 = sum_7 = None
        sub_81: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_80, mul_458);  sub_80 = mul_458 = None
        mul_459: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_1, sub_81);  div_1 = sub_81 = None
        mul_460: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_444, mul_426);  mul_426 = None
        sum_9: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_460, [0, 1]);  mul_460 = None
        sum_10: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_444, [0, 1]);  view_444 = None
        add_297: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(mul_441, mul_459);  mul_441 = mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_1: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_71, torch.float32);  gt_71 = None
        mul_461: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_1, 1.1111111111111112);  convert_element_type_1 = None
        mul_462: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_297, mul_461);  mul_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_445: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_462, [2048, 1280]);  mul_462 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_153: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_429, [1, 0]);  primals_429 = None
        mm_7: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_445, permute_153);  permute_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_143: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_390, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_426: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_143, [4, 512, -1]);  permute_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_427: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_426, [-1, 1280]);  view_426 = None
        permute_154: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_427, [1, 0]);  view_427 = None
        mm_8: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_154, view_445);  permute_154 = None
        sum_11: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_445, [0], True);  view_445 = None
        view_446: "f32[1280]" = torch.ops.aten.reshape.default(sum_11, [1280]);  sum_11 = None
        view_447: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_7, [4, 512, 1280]);  mm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_448: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_447, [4, 512, 20, 64]);  view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_155: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_448, [0, 2, 1, 3]);  view_448 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_2: "f32[4, 20, 512, 512]" = torch.ops.aten.expand.default(where, [4, 20, 512, 512]);  where = None
        _scaled_dot_product_efficient_attention_backward = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_155, permute_142, permute_140, permute_141, expand_2, getitem_390, getitem_391, getitem_392, getitem_393, 0.1, [True, True, True, False], scale = 0.125);  permute_155 = permute_142 = permute_140 = permute_141 = getitem_390 = getitem_391 = getitem_392 = getitem_393 = None
        getitem_398: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward[0]
        getitem_399: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward[1]
        getitem_400: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward[2];  _scaled_dot_product_efficient_attention_backward = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_156: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_398, [0, 2, 1, 3]);  getitem_398 = None
        view_449: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_156, [4, 512, 1280]);  permute_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_157: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_400, [0, 2, 1, 3]);  getitem_400 = None
        view_450: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_157, [4, 512, 1280]);  permute_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_158: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_399, [0, 2, 1, 3]);  getitem_399 = None
        view_451: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_158, [4, 512, 1280]);  permute_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_1: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_449, view_451, view_450], 2);  view_449 = view_451 = view_450 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_452: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_1, [2048, 3840]);  cat_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_159: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_427, [1, 0]);  primals_427 = None
        mm_9: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_452, permute_159);  permute_159 = None
        mm_10: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_160, view_452);  permute_160 = None
        sum_12: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_452, [0], True);  view_452 = None
        view_453: "f32[3840]" = torch.ops.aten.reshape.default(sum_12, [3840]);  sum_12 = None
        view_454: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_9, [4, 512, 1280]);  mm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_464: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_454, primals_424);  primals_424 = None
        mul_465: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_464, 1280)
        sum_13: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_464, [2], True)
        mul_466: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_464, mul_422);  mul_464 = None
        sum_14: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_466, [2], True);  mul_466 = None
        mul_467: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_422, sum_14);  sum_14 = None
        sub_83: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_465, sum_13);  mul_465 = sum_13 = None
        sub_84: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_83, mul_467);  sub_83 = mul_467 = None
        mul_468: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_2, sub_84);  div_2 = sub_84 = None
        mul_469: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_454, mul_422);  mul_422 = None
        sum_15: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_469, [0, 1]);  mul_469 = None
        sum_16: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_454, [0, 1]);  view_454 = None
        add_298: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_297, mul_468);  add_297 = mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_2: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_470: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_2, 1.1111111111111112);  convert_element_type_2 = None
        mul_471: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_298, mul_470);  mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_455: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_471, [2048, 1280]);  mul_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_161: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_423, [1, 0]);  primals_423 = None
        mm_11: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_455, permute_161);  permute_161 = None
        mm_12: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_162, view_455);  permute_162 = None
        sum_17: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_455, [0], True);  view_455 = None
        view_456: "f32[1280]" = torch.ops.aten.reshape.default(sum_17, [1280]);  sum_17 = None
        view_457: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_11, [4, 512, 5120]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_418: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_138, [4, 512, 5120]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_416: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_418, 0.5)
        mul_472: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_457, mul_416);  mul_416 = None
        pow_35: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_418, 3.0)
        mul_417: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_35, 0.044715);  pow_35 = None
        add_281: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_418, mul_417);  mul_417 = None
        mul_418: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_281, 0.7978845608028654);  add_281 = None
        tanh_34: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_418);  mul_418 = None
        add_282: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_34, 1.0)
        mul_473: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_457, add_282);  view_457 = add_282 = None
        mul_474: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_34, tanh_34);  tanh_34 = None
        sub_85: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_474);  mul_474 = None
        mul_475: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_472, sub_85);  mul_472 = sub_85 = None
        mul_476: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_475, 0.7978845608028654);  mul_475 = None
        mul_477: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_476, 0.044715)
        pow_38: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_418, 2.0);  view_418 = None
        mul_478: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_38, 3.0);  pow_38 = None
        mul_479: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_477, mul_478);  mul_477 = mul_478 = None
        add_299: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_476, mul_479);  mul_476 = mul_479 = None
        mul_480: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_473, 0.5);  mul_473 = None
        add_300: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_299, mul_480);  add_299 = mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_458: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_300, [2048, 5120]);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_163: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_421, [1, 0]);  primals_421 = None
        mm_13: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_458, permute_163);  permute_163 = None
        mm_14: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_164, view_458);  permute_164 = None
        sum_18: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_458, [0], True);  view_458 = None
        view_459: "f32[5120]" = torch.ops.aten.reshape.default(sum_18, [5120]);  sum_18 = None
        view_460: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_13, [4, 512, 1280]);  mm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_482: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_460, primals_418);  primals_418 = None
        mul_483: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_482, 1280)
        sum_19: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_482, [2], True)
        mul_484: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_482, mul_414);  mul_482 = None
        sum_20: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_484, [2], True);  mul_484 = None
        mul_485: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_414, sum_20);  sum_20 = None
        sub_87: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_483, sum_19);  mul_483 = sum_19 = None
        sub_88: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_87, mul_485);  sub_87 = mul_485 = None
        mul_486: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_3, sub_88);  div_3 = sub_88 = None
        mul_487: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_460, mul_414);  mul_414 = None
        sum_21: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1]);  mul_487 = None
        sum_22: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_460, [0, 1]);  view_460 = None
        add_301: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_298, mul_486);  add_298 = mul_486 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_3: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_69, torch.float32);  gt_69 = None
        mul_488: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_3, 1.1111111111111112);  convert_element_type_3 = None
        mul_489: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_301, mul_488);  mul_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_461: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_489, [2048, 1280]);  mul_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_165: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_417, [1, 0]);  primals_417 = None
        mm_15: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_461, permute_165);  permute_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_139: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_379, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_414: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_139, [4, 512, -1]);  permute_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_415: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_414, [-1, 1280]);  view_414 = None
        permute_166: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_415, [1, 0]);  view_415 = None
        mm_16: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_166, view_461);  permute_166 = None
        sum_23: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_461, [0], True);  view_461 = None
        view_462: "f32[1280]" = torch.ops.aten.reshape.default(sum_23, [1280]);  sum_23 = None
        view_463: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_15, [4, 512, 1280]);  mm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_464: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_463, [4, 512, 20, 64]);  view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_167: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_464, [0, 2, 1, 3]);  view_464 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_1 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_167, permute_138, permute_136, permute_137, expand_2, getitem_379, getitem_380, getitem_381, getitem_382, 0.1, [True, True, True, False], scale = 0.125);  permute_167 = permute_138 = permute_136 = permute_137 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = None
        getitem_402: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_1[0]
        getitem_403: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_1[1]
        getitem_404: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_1[2];  _scaled_dot_product_efficient_attention_backward_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_168: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_402, [0, 2, 1, 3]);  getitem_402 = None
        view_465: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_168, [4, 512, 1280]);  permute_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_169: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_404, [0, 2, 1, 3]);  getitem_404 = None
        view_466: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_169, [4, 512, 1280]);  permute_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_170: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_403, [0, 2, 1, 3]);  getitem_403 = None
        view_467: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_170, [4, 512, 1280]);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_2: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_465, view_467, view_466], 2);  view_465 = view_467 = view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_468: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_2, [2048, 3840]);  cat_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_171: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_415, [1, 0]);  primals_415 = None
        mm_17: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_468, permute_171);  permute_171 = None
        mm_18: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_172, view_468);  permute_172 = None
        sum_24: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_468, [0], True);  view_468 = None
        view_469: "f32[3840]" = torch.ops.aten.reshape.default(sum_24, [3840]);  sum_24 = None
        view_470: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_17, [4, 512, 1280]);  mm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_491: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_470, primals_412);  primals_412 = None
        mul_492: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_491, 1280)
        sum_25: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_491, [2], True)
        mul_493: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_491, mul_410);  mul_491 = None
        sum_26: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_493, [2], True);  mul_493 = None
        mul_494: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_410, sum_26);  sum_26 = None
        sub_90: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_492, sum_25);  mul_492 = sum_25 = None
        sub_91: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_90, mul_494);  sub_90 = mul_494 = None
        mul_495: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_4, sub_91);  div_4 = sub_91 = None
        mul_496: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_470, mul_410);  mul_410 = None
        sum_27: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_496, [0, 1]);  mul_496 = None
        sum_28: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_470, [0, 1]);  view_470 = None
        add_302: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_301, mul_495);  add_301 = mul_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_4: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_68, torch.float32);  gt_68 = None
        mul_497: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_4, 1.1111111111111112);  convert_element_type_4 = None
        mul_498: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_302, mul_497);  mul_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_471: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_498, [2048, 1280]);  mul_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_173: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_411, [1, 0]);  primals_411 = None
        mm_19: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_471, permute_173);  permute_173 = None
        mm_20: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_174, view_471);  permute_174 = None
        sum_29: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_471, [0], True);  view_471 = None
        view_472: "f32[1280]" = torch.ops.aten.reshape.default(sum_29, [1280]);  sum_29 = None
        view_473: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_19, [4, 512, 5120]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_406: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_134, [4, 512, 5120]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_404: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_406, 0.5)
        mul_499: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_473, mul_404);  mul_404 = None
        pow_34: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_406, 3.0)
        mul_405: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_34, 0.044715);  pow_34 = None
        add_273: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_406, mul_405);  mul_405 = None
        mul_406: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_273, 0.7978845608028654);  add_273 = None
        tanh_33: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_406);  mul_406 = None
        add_274: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_33, 1.0)
        mul_500: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_473, add_274);  view_473 = add_274 = None
        mul_501: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_33, tanh_33);  tanh_33 = None
        sub_92: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_501);  mul_501 = None
        mul_502: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_499, sub_92);  mul_499 = sub_92 = None
        mul_503: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_502, 0.7978845608028654);  mul_502 = None
        mul_504: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_503, 0.044715)
        pow_39: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_406, 2.0);  view_406 = None
        mul_505: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_39, 3.0);  pow_39 = None
        mul_506: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_504, mul_505);  mul_504 = mul_505 = None
        add_303: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_503, mul_506);  mul_503 = mul_506 = None
        mul_507: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_500, 0.5);  mul_500 = None
        add_304: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_303, mul_507);  add_303 = mul_507 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_474: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_304, [2048, 5120]);  add_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_175: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_409, [1, 0]);  primals_409 = None
        mm_21: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_474, permute_175);  permute_175 = None
        mm_22: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_176, view_474);  permute_176 = None
        sum_30: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_474, [0], True);  view_474 = None
        view_475: "f32[5120]" = torch.ops.aten.reshape.default(sum_30, [5120]);  sum_30 = None
        view_476: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_21, [4, 512, 1280]);  mm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_509: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_476, primals_406);  primals_406 = None
        mul_510: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_509, 1280)
        sum_31: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_509, [2], True)
        mul_511: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_509, mul_402);  mul_509 = None
        sum_32: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True);  mul_511 = None
        mul_512: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_402, sum_32);  sum_32 = None
        sub_94: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_510, sum_31);  mul_510 = sum_31 = None
        sub_95: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_94, mul_512);  sub_94 = mul_512 = None
        mul_513: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_5, sub_95);  div_5 = sub_95 = None
        mul_514: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_476, mul_402);  mul_402 = None
        sum_33: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_514, [0, 1]);  mul_514 = None
        sum_34: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_476, [0, 1]);  view_476 = None
        add_305: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_302, mul_513);  add_302 = mul_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_5: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_67, torch.float32);  gt_67 = None
        mul_515: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_5, 1.1111111111111112);  convert_element_type_5 = None
        mul_516: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_305, mul_515);  mul_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_477: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_516, [2048, 1280]);  mul_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_177: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_405, [1, 0]);  primals_405 = None
        mm_23: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_477, permute_177);  permute_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_135: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_368, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_402: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_135, [4, 512, -1]);  permute_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_403: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_402, [-1, 1280]);  view_402 = None
        permute_178: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_403, [1, 0]);  view_403 = None
        mm_24: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_178, view_477);  permute_178 = None
        sum_35: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_477, [0], True);  view_477 = None
        view_478: "f32[1280]" = torch.ops.aten.reshape.default(sum_35, [1280]);  sum_35 = None
        view_479: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_23, [4, 512, 1280]);  mm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_480: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_479, [4, 512, 20, 64]);  view_479 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_179: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_480, [0, 2, 1, 3]);  view_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_2 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_179, permute_134, permute_132, permute_133, expand_2, getitem_368, getitem_369, getitem_370, getitem_371, 0.1, [True, True, True, False], scale = 0.125);  permute_179 = permute_134 = permute_132 = permute_133 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = None
        getitem_406: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_2[0]
        getitem_407: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_2[1]
        getitem_408: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_2[2];  _scaled_dot_product_efficient_attention_backward_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_180: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_406, [0, 2, 1, 3]);  getitem_406 = None
        view_481: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_180, [4, 512, 1280]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_181: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_408, [0, 2, 1, 3]);  getitem_408 = None
        view_482: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_181, [4, 512, 1280]);  permute_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_182: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_407, [0, 2, 1, 3]);  getitem_407 = None
        view_483: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_182, [4, 512, 1280]);  permute_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_3: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_481, view_483, view_482], 2);  view_481 = view_483 = view_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_484: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_3, [2048, 3840]);  cat_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_183: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_403, [1, 0]);  primals_403 = None
        mm_25: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_484, permute_183);  permute_183 = None
        mm_26: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_184, view_484);  permute_184 = None
        sum_36: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_484, [0], True);  view_484 = None
        view_485: "f32[3840]" = torch.ops.aten.reshape.default(sum_36, [3840]);  sum_36 = None
        view_486: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_25, [4, 512, 1280]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_518: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_486, primals_400);  primals_400 = None
        mul_519: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_518, 1280)
        sum_37: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_518, [2], True)
        mul_520: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_518, mul_398);  mul_518 = None
        sum_38: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_520, [2], True);  mul_520 = None
        mul_521: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_398, sum_38);  sum_38 = None
        sub_97: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_519, sum_37);  mul_519 = sum_37 = None
        sub_98: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_97, mul_521);  sub_97 = mul_521 = None
        mul_522: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_6, sub_98);  div_6 = sub_98 = None
        mul_523: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_486, mul_398);  mul_398 = None
        sum_39: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 1]);  mul_523 = None
        sum_40: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_486, [0, 1]);  view_486 = None
        add_306: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_305, mul_522);  add_305 = mul_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_6: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_66, torch.float32);  gt_66 = None
        mul_524: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_525: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_306, mul_524);  mul_524 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_487: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_525, [2048, 1280]);  mul_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_185: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_399, [1, 0]);  primals_399 = None
        mm_27: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_487, permute_185);  permute_185 = None
        mm_28: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_186, view_487);  permute_186 = None
        sum_41: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_487, [0], True);  view_487 = None
        view_488: "f32[1280]" = torch.ops.aten.reshape.default(sum_41, [1280]);  sum_41 = None
        view_489: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_27, [4, 512, 5120]);  mm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_394: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_130, [4, 512, 5120]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_392: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_394, 0.5)
        mul_526: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_489, mul_392);  mul_392 = None
        pow_33: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_394, 3.0)
        mul_393: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_265: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_394, mul_393);  mul_393 = None
        mul_394: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_265, 0.7978845608028654);  add_265 = None
        tanh_32: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_394);  mul_394 = None
        add_266: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_32, 1.0)
        mul_527: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_489, add_266);  view_489 = add_266 = None
        mul_528: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_32, tanh_32);  tanh_32 = None
        sub_99: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_528);  mul_528 = None
        mul_529: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_526, sub_99);  mul_526 = sub_99 = None
        mul_530: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_529, 0.7978845608028654);  mul_529 = None
        mul_531: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_530, 0.044715)
        pow_40: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_394, 2.0);  view_394 = None
        mul_532: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_40, 3.0);  pow_40 = None
        mul_533: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_531, mul_532);  mul_531 = mul_532 = None
        add_307: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_530, mul_533);  mul_530 = mul_533 = None
        mul_534: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_527, 0.5);  mul_527 = None
        add_308: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_307, mul_534);  add_307 = mul_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_490: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_308, [2048, 5120]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_187: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_397, [1, 0]);  primals_397 = None
        mm_29: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_490, permute_187);  permute_187 = None
        mm_30: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_188, view_490);  permute_188 = None
        sum_42: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_490, [0], True);  view_490 = None
        view_491: "f32[5120]" = torch.ops.aten.reshape.default(sum_42, [5120]);  sum_42 = None
        view_492: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_29, [4, 512, 1280]);  mm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_536: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_492, primals_394);  primals_394 = None
        mul_537: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_536, 1280)
        sum_43: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_536, [2], True)
        mul_538: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_536, mul_390);  mul_536 = None
        sum_44: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_538, [2], True);  mul_538 = None
        mul_539: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_390, sum_44);  sum_44 = None
        sub_101: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_537, sum_43);  mul_537 = sum_43 = None
        sub_102: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_101, mul_539);  sub_101 = mul_539 = None
        mul_540: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_7, sub_102);  div_7 = sub_102 = None
        mul_541: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_492, mul_390);  mul_390 = None
        sum_45: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_541, [0, 1]);  mul_541 = None
        sum_46: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_492, [0, 1]);  view_492 = None
        add_309: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_306, mul_540);  add_306 = mul_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_7: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_542: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_543: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_309, mul_542);  mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_493: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_543, [2048, 1280]);  mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_189: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_393, [1, 0]);  primals_393 = None
        mm_31: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_493, permute_189);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_131: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_357, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_390: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_131, [4, 512, -1]);  permute_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_391: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_390, [-1, 1280]);  view_390 = None
        permute_190: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_391, [1, 0]);  view_391 = None
        mm_32: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_190, view_493);  permute_190 = None
        sum_47: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_493, [0], True);  view_493 = None
        view_494: "f32[1280]" = torch.ops.aten.reshape.default(sum_47, [1280]);  sum_47 = None
        view_495: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_31, [4, 512, 1280]);  mm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_496: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_495, [4, 512, 20, 64]);  view_495 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_191: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_3 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_191, permute_130, permute_128, permute_129, expand_2, getitem_357, getitem_358, getitem_359, getitem_360, 0.1, [True, True, True, False], scale = 0.125);  permute_191 = permute_130 = permute_128 = permute_129 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = None
        getitem_410: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_3[0]
        getitem_411: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_3[1]
        getitem_412: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_3[2];  _scaled_dot_product_efficient_attention_backward_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_192: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_410, [0, 2, 1, 3]);  getitem_410 = None
        view_497: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_192, [4, 512, 1280]);  permute_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_193: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_412, [0, 2, 1, 3]);  getitem_412 = None
        view_498: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_193, [4, 512, 1280]);  permute_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_194: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_411, [0, 2, 1, 3]);  getitem_411 = None
        view_499: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_194, [4, 512, 1280]);  permute_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_4: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_497, view_499, view_498], 2);  view_497 = view_499 = view_498 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_500: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_4, [2048, 3840]);  cat_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_195: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_391, [1, 0]);  primals_391 = None
        mm_33: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_500, permute_195);  permute_195 = None
        mm_34: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_196, view_500);  permute_196 = None
        sum_48: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_500, [0], True);  view_500 = None
        view_501: "f32[3840]" = torch.ops.aten.reshape.default(sum_48, [3840]);  sum_48 = None
        view_502: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_33, [4, 512, 1280]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_545: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_502, primals_388);  primals_388 = None
        mul_546: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_545, 1280)
        sum_49: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_545, [2], True)
        mul_547: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_545, mul_386);  mul_545 = None
        sum_50: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_547, [2], True);  mul_547 = None
        mul_548: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_386, sum_50);  sum_50 = None
        sub_104: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_546, sum_49);  mul_546 = sum_49 = None
        sub_105: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_104, mul_548);  sub_104 = mul_548 = None
        mul_549: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_8, sub_105);  div_8 = sub_105 = None
        mul_550: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_502, mul_386);  mul_386 = None
        sum_51: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_550, [0, 1]);  mul_550 = None
        sum_52: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_502, [0, 1]);  view_502 = None
        add_310: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_309, mul_549);  add_309 = mul_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_551: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_552: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_310, mul_551);  mul_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_503: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_552, [2048, 1280]);  mul_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_197: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_387, [1, 0]);  primals_387 = None
        mm_35: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_503, permute_197);  permute_197 = None
        mm_36: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_198, view_503);  permute_198 = None
        sum_53: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_503, [0], True);  view_503 = None
        view_504: "f32[1280]" = torch.ops.aten.reshape.default(sum_53, [1280]);  sum_53 = None
        view_505: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_35, [4, 512, 5120]);  mm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_382: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_126, [4, 512, 5120]);  addmm_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_380: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_382, 0.5)
        mul_553: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_505, mul_380);  mul_380 = None
        pow_32: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_382, 3.0)
        mul_381: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_32, 0.044715);  pow_32 = None
        add_257: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_382, mul_381);  mul_381 = None
        mul_382: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_257, 0.7978845608028654);  add_257 = None
        tanh_31: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_382);  mul_382 = None
        add_258: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_31, 1.0)
        mul_554: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_505, add_258);  view_505 = add_258 = None
        mul_555: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_31, tanh_31);  tanh_31 = None
        sub_106: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_555);  mul_555 = None
        mul_556: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_553, sub_106);  mul_553 = sub_106 = None
        mul_557: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_556, 0.7978845608028654);  mul_556 = None
        mul_558: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_557, 0.044715)
        pow_41: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_382, 2.0);  view_382 = None
        mul_559: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_41, 3.0);  pow_41 = None
        mul_560: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_558, mul_559);  mul_558 = mul_559 = None
        add_311: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_557, mul_560);  mul_557 = mul_560 = None
        mul_561: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_554, 0.5);  mul_554 = None
        add_312: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_311, mul_561);  add_311 = mul_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_506: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_312, [2048, 5120]);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_199: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_385, [1, 0]);  primals_385 = None
        mm_37: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_506, permute_199);  permute_199 = None
        mm_38: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_200, view_506);  permute_200 = None
        sum_54: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_506, [0], True);  view_506 = None
        view_507: "f32[5120]" = torch.ops.aten.reshape.default(sum_54, [5120]);  sum_54 = None
        view_508: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_37, [4, 512, 1280]);  mm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_563: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_508, primals_382);  primals_382 = None
        mul_564: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_563, 1280)
        sum_55: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_563, [2], True)
        mul_565: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_563, mul_378);  mul_563 = None
        sum_56: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_565, [2], True);  mul_565 = None
        mul_566: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_378, sum_56);  sum_56 = None
        sub_108: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_564, sum_55);  mul_564 = sum_55 = None
        sub_109: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_108, mul_566);  sub_108 = mul_566 = None
        mul_567: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_9, sub_109);  div_9 = sub_109 = None
        mul_568: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_508, mul_378);  mul_378 = None
        sum_57: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_568, [0, 1]);  mul_568 = None
        sum_58: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_508, [0, 1]);  view_508 = None
        add_313: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_310, mul_567);  add_310 = mul_567 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_9: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_569: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_570: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_313, mul_569);  mul_569 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_509: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_570, [2048, 1280]);  mul_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_201: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_381, [1, 0]);  primals_381 = None
        mm_39: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_509, permute_201);  permute_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_127: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_346, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_378: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_127, [4, 512, -1]);  permute_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_379: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_378, [-1, 1280]);  view_378 = None
        permute_202: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_379, [1, 0]);  view_379 = None
        mm_40: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_202, view_509);  permute_202 = None
        sum_59: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_509, [0], True);  view_509 = None
        view_510: "f32[1280]" = torch.ops.aten.reshape.default(sum_59, [1280]);  sum_59 = None
        view_511: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_39, [4, 512, 1280]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_512: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_511, [4, 512, 20, 64]);  view_511 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_203: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_4 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_203, permute_126, permute_124, permute_125, expand_2, getitem_346, getitem_347, getitem_348, getitem_349, 0.1, [True, True, True, False], scale = 0.125);  permute_203 = permute_126 = permute_124 = permute_125 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = None
        getitem_414: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_4[0]
        getitem_415: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_4[1]
        getitem_416: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_4[2];  _scaled_dot_product_efficient_attention_backward_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_204: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_414, [0, 2, 1, 3]);  getitem_414 = None
        view_513: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_204, [4, 512, 1280]);  permute_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_205: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_416, [0, 2, 1, 3]);  getitem_416 = None
        view_514: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_205, [4, 512, 1280]);  permute_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_206: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_415, [0, 2, 1, 3]);  getitem_415 = None
        view_515: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_206, [4, 512, 1280]);  permute_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_5: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_513, view_515, view_514], 2);  view_513 = view_515 = view_514 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_516: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_5, [2048, 3840]);  cat_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_207: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_379, [1, 0]);  primals_379 = None
        mm_41: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_516, permute_207);  permute_207 = None
        mm_42: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_208, view_516);  permute_208 = None
        sum_60: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_516, [0], True);  view_516 = None
        view_517: "f32[3840]" = torch.ops.aten.reshape.default(sum_60, [3840]);  sum_60 = None
        view_518: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_41, [4, 512, 1280]);  mm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_572: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_518, primals_376);  primals_376 = None
        mul_573: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_572, 1280)
        sum_61: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_572, [2], True)
        mul_574: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_572, mul_374);  mul_572 = None
        sum_62: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_574, [2], True);  mul_574 = None
        mul_575: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_374, sum_62);  sum_62 = None
        sub_111: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_573, sum_61);  mul_573 = sum_61 = None
        sub_112: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_111, mul_575);  sub_111 = mul_575 = None
        mul_576: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_10, sub_112);  div_10 = sub_112 = None
        mul_577: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_518, mul_374);  mul_374 = None
        sum_63: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_577, [0, 1]);  mul_577 = None
        sum_64: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_518, [0, 1]);  view_518 = None
        add_314: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_313, mul_576);  add_313 = mul_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_10: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_578: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_579: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_314, mul_578);  mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_519: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_579, [2048, 1280]);  mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_209: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_375, [1, 0]);  primals_375 = None
        mm_43: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_519, permute_209);  permute_209 = None
        mm_44: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_210, view_519);  permute_210 = None
        sum_65: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_519, [0], True);  view_519 = None
        view_520: "f32[1280]" = torch.ops.aten.reshape.default(sum_65, [1280]);  sum_65 = None
        view_521: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_43, [4, 512, 5120]);  mm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_370: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_122, [4, 512, 5120]);  addmm_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_368: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_370, 0.5)
        mul_580: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_521, mul_368);  mul_368 = None
        pow_31: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_370, 3.0)
        mul_369: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_31, 0.044715);  pow_31 = None
        add_249: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_370, mul_369);  mul_369 = None
        mul_370: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_249, 0.7978845608028654);  add_249 = None
        tanh_30: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_370);  mul_370 = None
        add_250: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_30, 1.0)
        mul_581: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_521, add_250);  view_521 = add_250 = None
        mul_582: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_30, tanh_30);  tanh_30 = None
        sub_113: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_582);  mul_582 = None
        mul_583: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_580, sub_113);  mul_580 = sub_113 = None
        mul_584: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_583, 0.7978845608028654);  mul_583 = None
        mul_585: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_584, 0.044715)
        pow_42: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_370, 2.0);  view_370 = None
        mul_586: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_42, 3.0);  pow_42 = None
        mul_587: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_585, mul_586);  mul_585 = mul_586 = None
        add_315: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_584, mul_587);  mul_584 = mul_587 = None
        mul_588: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_581, 0.5);  mul_581 = None
        add_316: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_315, mul_588);  add_315 = mul_588 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_522: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_316, [2048, 5120]);  add_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_211: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_373, [1, 0]);  primals_373 = None
        mm_45: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_522, permute_211);  permute_211 = None
        mm_46: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_212, view_522);  permute_212 = None
        sum_66: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_522, [0], True);  view_522 = None
        view_523: "f32[5120]" = torch.ops.aten.reshape.default(sum_66, [5120]);  sum_66 = None
        view_524: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_45, [4, 512, 1280]);  mm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_590: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_524, primals_370);  primals_370 = None
        mul_591: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_590, 1280)
        sum_67: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_590, [2], True)
        mul_592: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_590, mul_366);  mul_590 = None
        sum_68: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_592, [2], True);  mul_592 = None
        mul_593: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_366, sum_68);  sum_68 = None
        sub_115: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_591, sum_67);  mul_591 = sum_67 = None
        sub_116: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_115, mul_593);  sub_115 = mul_593 = None
        mul_594: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_11, sub_116);  div_11 = sub_116 = None
        mul_595: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_524, mul_366);  mul_366 = None
        sum_69: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_595, [0, 1]);  mul_595 = None
        sum_70: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_524, [0, 1]);  view_524 = None
        add_317: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_314, mul_594);  add_314 = mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_11: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_596: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_597: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_317, mul_596);  mul_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_525: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_597, [2048, 1280]);  mul_597 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_213: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_369, [1, 0]);  primals_369 = None
        mm_47: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_525, permute_213);  permute_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_123: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_335, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_366: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_123, [4, 512, -1]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_367: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_366, [-1, 1280]);  view_366 = None
        permute_214: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_367, [1, 0]);  view_367 = None
        mm_48: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_214, view_525);  permute_214 = None
        sum_71: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_525, [0], True);  view_525 = None
        view_526: "f32[1280]" = torch.ops.aten.reshape.default(sum_71, [1280]);  sum_71 = None
        view_527: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_47, [4, 512, 1280]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_528: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_527, [4, 512, 20, 64]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_215: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_5 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_215, permute_122, permute_120, permute_121, expand_2, getitem_335, getitem_336, getitem_337, getitem_338, 0.1, [True, True, True, False], scale = 0.125);  permute_215 = permute_122 = permute_120 = permute_121 = getitem_335 = getitem_336 = getitem_337 = getitem_338 = None
        getitem_418: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_5[0]
        getitem_419: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_5[1]
        getitem_420: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_5[2];  _scaled_dot_product_efficient_attention_backward_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_216: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_418, [0, 2, 1, 3]);  getitem_418 = None
        view_529: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_216, [4, 512, 1280]);  permute_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_217: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_420, [0, 2, 1, 3]);  getitem_420 = None
        view_530: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_217, [4, 512, 1280]);  permute_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_218: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_419, [0, 2, 1, 3]);  getitem_419 = None
        view_531: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_218, [4, 512, 1280]);  permute_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_6: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_529, view_531, view_530], 2);  view_529 = view_531 = view_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_532: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_6, [2048, 3840]);  cat_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_219: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_367, [1, 0]);  primals_367 = None
        mm_49: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_532, permute_219);  permute_219 = None
        mm_50: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_220, view_532);  permute_220 = None
        sum_72: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_532, [0], True);  view_532 = None
        view_533: "f32[3840]" = torch.ops.aten.reshape.default(sum_72, [3840]);  sum_72 = None
        view_534: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_49, [4, 512, 1280]);  mm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_599: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_534, primals_364);  primals_364 = None
        mul_600: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_599, 1280)
        sum_73: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_599, [2], True)
        mul_601: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_599, mul_362);  mul_599 = None
        sum_74: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_601, [2], True);  mul_601 = None
        mul_602: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_362, sum_74);  sum_74 = None
        sub_118: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_600, sum_73);  mul_600 = sum_73 = None
        sub_119: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_118, mul_602);  sub_118 = mul_602 = None
        mul_603: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_12, sub_119);  div_12 = sub_119 = None
        mul_604: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_534, mul_362);  mul_362 = None
        sum_75: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_604, [0, 1]);  mul_604 = None
        sum_76: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_534, [0, 1]);  view_534 = None
        add_318: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_317, mul_603);  add_317 = mul_603 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_12: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_605: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_606: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_318, mul_605);  mul_605 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_535: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_606, [2048, 1280]);  mul_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_221: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_363, [1, 0]);  primals_363 = None
        mm_51: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_535, permute_221);  permute_221 = None
        mm_52: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_222, view_535);  permute_222 = None
        sum_77: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        view_536: "f32[1280]" = torch.ops.aten.reshape.default(sum_77, [1280]);  sum_77 = None
        view_537: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_51, [4, 512, 5120]);  mm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_358: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_118, [4, 512, 5120]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_356: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_358, 0.5)
        mul_607: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_537, mul_356);  mul_356 = None
        pow_30: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_358, 3.0)
        mul_357: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_30, 0.044715);  pow_30 = None
        add_241: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_358, mul_357);  mul_357 = None
        mul_358: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_241, 0.7978845608028654);  add_241 = None
        tanh_29: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_358);  mul_358 = None
        add_242: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_29, 1.0)
        mul_608: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_537, add_242);  view_537 = add_242 = None
        mul_609: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_29, tanh_29);  tanh_29 = None
        sub_120: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_609);  mul_609 = None
        mul_610: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_607, sub_120);  mul_607 = sub_120 = None
        mul_611: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_610, 0.7978845608028654);  mul_610 = None
        mul_612: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_611, 0.044715)
        pow_43: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_358, 2.0);  view_358 = None
        mul_613: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_43, 3.0);  pow_43 = None
        mul_614: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_612, mul_613);  mul_612 = mul_613 = None
        add_319: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_611, mul_614);  mul_611 = mul_614 = None
        mul_615: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_608, 0.5);  mul_608 = None
        add_320: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_319, mul_615);  add_319 = mul_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_538: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_320, [2048, 5120]);  add_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_223: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_361, [1, 0]);  primals_361 = None
        mm_53: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_538, permute_223);  permute_223 = None
        mm_54: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_224, view_538);  permute_224 = None
        sum_78: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        view_539: "f32[5120]" = torch.ops.aten.reshape.default(sum_78, [5120]);  sum_78 = None
        view_540: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_53, [4, 512, 1280]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_617: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_540, primals_358);  primals_358 = None
        mul_618: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_617, 1280)
        sum_79: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_617, [2], True)
        mul_619: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_617, mul_354);  mul_617 = None
        sum_80: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_619, [2], True);  mul_619 = None
        mul_620: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_354, sum_80);  sum_80 = None
        sub_122: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_618, sum_79);  mul_618 = sum_79 = None
        sub_123: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_122, mul_620);  sub_122 = mul_620 = None
        mul_621: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_13, sub_123);  div_13 = sub_123 = None
        mul_622: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_540, mul_354);  mul_354 = None
        sum_81: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_622, [0, 1]);  mul_622 = None
        sum_82: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None
        add_321: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_318, mul_621);  add_318 = mul_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_13: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_623: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_624: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_321, mul_623);  mul_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_541: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_624, [2048, 1280]);  mul_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_225: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_357, [1, 0]);  primals_357 = None
        mm_55: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_541, permute_225);  permute_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_119: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_324, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_354: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_119, [4, 512, -1]);  permute_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_355: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_354, [-1, 1280]);  view_354 = None
        permute_226: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_355, [1, 0]);  view_355 = None
        mm_56: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_226, view_541);  permute_226 = None
        sum_83: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        view_542: "f32[1280]" = torch.ops.aten.reshape.default(sum_83, [1280]);  sum_83 = None
        view_543: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_55, [4, 512, 1280]);  mm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_544: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_543, [4, 512, 20, 64]);  view_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_544, [0, 2, 1, 3]);  view_544 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_6 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_227, permute_118, permute_116, permute_117, expand_2, getitem_324, getitem_325, getitem_326, getitem_327, 0.1, [True, True, True, False], scale = 0.125);  permute_227 = permute_118 = permute_116 = permute_117 = getitem_324 = getitem_325 = getitem_326 = getitem_327 = None
        getitem_422: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_6[0]
        getitem_423: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_6[1]
        getitem_424: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_6[2];  _scaled_dot_product_efficient_attention_backward_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_228: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_422, [0, 2, 1, 3]);  getitem_422 = None
        view_545: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_228, [4, 512, 1280]);  permute_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_229: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_424, [0, 2, 1, 3]);  getitem_424 = None
        view_546: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_229, [4, 512, 1280]);  permute_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_230: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_423, [0, 2, 1, 3]);  getitem_423 = None
        view_547: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_230, [4, 512, 1280]);  permute_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_7: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_545, view_547, view_546], 2);  view_545 = view_547 = view_546 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_548: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_7, [2048, 3840]);  cat_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_231: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_355, [1, 0]);  primals_355 = None
        mm_57: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_548, permute_231);  permute_231 = None
        mm_58: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_232, view_548);  permute_232 = None
        sum_84: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_548, [0], True);  view_548 = None
        view_549: "f32[3840]" = torch.ops.aten.reshape.default(sum_84, [3840]);  sum_84 = None
        view_550: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_57, [4, 512, 1280]);  mm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_626: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_550, primals_352);  primals_352 = None
        mul_627: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_626, 1280)
        sum_85: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_626, [2], True)
        mul_628: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_626, mul_350);  mul_626 = None
        sum_86: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_628, [2], True);  mul_628 = None
        mul_629: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_350, sum_86);  sum_86 = None
        sub_125: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_627, sum_85);  mul_627 = sum_85 = None
        sub_126: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_125, mul_629);  sub_125 = mul_629 = None
        mul_630: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_14, sub_126);  div_14 = sub_126 = None
        mul_631: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_550, mul_350);  mul_350 = None
        sum_87: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_631, [0, 1]);  mul_631 = None
        sum_88: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_550, [0, 1]);  view_550 = None
        add_322: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_321, mul_630);  add_321 = mul_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_632: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_633: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_322, mul_632);  mul_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_551: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_633, [2048, 1280]);  mul_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_233: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_351, [1, 0]);  primals_351 = None
        mm_59: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_551, permute_233);  permute_233 = None
        mm_60: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_234, view_551);  permute_234 = None
        sum_89: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_551, [0], True);  view_551 = None
        view_552: "f32[1280]" = torch.ops.aten.reshape.default(sum_89, [1280]);  sum_89 = None
        view_553: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_59, [4, 512, 5120]);  mm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_346: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_114, [4, 512, 5120]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_344: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_346, 0.5)
        mul_634: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_553, mul_344);  mul_344 = None
        pow_29: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_346, 3.0)
        mul_345: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_233: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_346, mul_345);  mul_345 = None
        mul_346: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_233, 0.7978845608028654);  add_233 = None
        tanh_28: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_346);  mul_346 = None
        add_234: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_28, 1.0)
        mul_635: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_553, add_234);  view_553 = add_234 = None
        mul_636: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_28, tanh_28);  tanh_28 = None
        sub_127: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_636);  mul_636 = None
        mul_637: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_634, sub_127);  mul_634 = sub_127 = None
        mul_638: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_637, 0.7978845608028654);  mul_637 = None
        mul_639: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_638, 0.044715)
        pow_44: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_346, 2.0);  view_346 = None
        mul_640: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_44, 3.0);  pow_44 = None
        mul_641: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_639, mul_640);  mul_639 = mul_640 = None
        add_323: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_638, mul_641);  mul_638 = mul_641 = None
        mul_642: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_635, 0.5);  mul_635 = None
        add_324: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_323, mul_642);  add_323 = mul_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_554: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_324, [2048, 5120]);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_235: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_349, [1, 0]);  primals_349 = None
        mm_61: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_554, permute_235);  permute_235 = None
        mm_62: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_236, view_554);  permute_236 = None
        sum_90: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_554, [0], True);  view_554 = None
        view_555: "f32[5120]" = torch.ops.aten.reshape.default(sum_90, [5120]);  sum_90 = None
        view_556: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_61, [4, 512, 1280]);  mm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_644: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_556, primals_346);  primals_346 = None
        mul_645: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_644, 1280)
        sum_91: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_644, [2], True)
        mul_646: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_644, mul_342);  mul_644 = None
        sum_92: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_646, [2], True);  mul_646 = None
        mul_647: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_342, sum_92);  sum_92 = None
        sub_129: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_645, sum_91);  mul_645 = sum_91 = None
        sub_130: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_129, mul_647);  sub_129 = mul_647 = None
        mul_648: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_15, sub_130);  div_15 = sub_130 = None
        mul_649: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_556, mul_342);  mul_342 = None
        sum_93: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_649, [0, 1]);  mul_649 = None
        sum_94: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_556, [0, 1]);  view_556 = None
        add_325: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_322, mul_648);  add_322 = mul_648 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_15: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_650: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_651: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_325, mul_650);  mul_650 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_557: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_651, [2048, 1280]);  mul_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_237: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_345, [1, 0]);  primals_345 = None
        mm_63: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_557, permute_237);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_115: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_313, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_342: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_115, [4, 512, -1]);  permute_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_343: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_342, [-1, 1280]);  view_342 = None
        permute_238: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_343, [1, 0]);  view_343 = None
        mm_64: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_238, view_557);  permute_238 = None
        sum_95: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_557, [0], True);  view_557 = None
        view_558: "f32[1280]" = torch.ops.aten.reshape.default(sum_95, [1280]);  sum_95 = None
        view_559: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_63, [4, 512, 1280]);  mm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_560: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_559, [4, 512, 20, 64]);  view_559 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_239: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_560, [0, 2, 1, 3]);  view_560 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_7 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_239, permute_114, permute_112, permute_113, expand_2, getitem_313, getitem_314, getitem_315, getitem_316, 0.1, [True, True, True, False], scale = 0.125);  permute_239 = permute_114 = permute_112 = permute_113 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = None
        getitem_426: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_7[0]
        getitem_427: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_7[1]
        getitem_428: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_7[2];  _scaled_dot_product_efficient_attention_backward_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_240: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_426, [0, 2, 1, 3]);  getitem_426 = None
        view_561: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_240, [4, 512, 1280]);  permute_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_241: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_428, [0, 2, 1, 3]);  getitem_428 = None
        view_562: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_241, [4, 512, 1280]);  permute_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_242: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_427, [0, 2, 1, 3]);  getitem_427 = None
        view_563: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_242, [4, 512, 1280]);  permute_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_8: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_561, view_563, view_562], 2);  view_561 = view_563 = view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_564: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_8, [2048, 3840]);  cat_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_243: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_343, [1, 0]);  primals_343 = None
        mm_65: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_564, permute_243);  permute_243 = None
        mm_66: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_244, view_564);  permute_244 = None
        sum_96: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_564, [0], True);  view_564 = None
        view_565: "f32[3840]" = torch.ops.aten.reshape.default(sum_96, [3840]);  sum_96 = None
        view_566: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_65, [4, 512, 1280]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_653: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_566, primals_340);  primals_340 = None
        mul_654: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_653, 1280)
        sum_97: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_653, [2], True)
        mul_655: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_653, mul_338);  mul_653 = None
        sum_98: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_655, [2], True);  mul_655 = None
        mul_656: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_338, sum_98);  sum_98 = None
        sub_132: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_654, sum_97);  mul_654 = sum_97 = None
        sub_133: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_132, mul_656);  sub_132 = mul_656 = None
        mul_657: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_16, sub_133);  div_16 = sub_133 = None
        mul_658: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_566, mul_338);  mul_338 = None
        sum_99: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_658, [0, 1]);  mul_658 = None
        sum_100: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_566, [0, 1]);  view_566 = None
        add_326: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_325, mul_657);  add_325 = mul_657 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_16: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_659: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_660: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_326, mul_659);  mul_659 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_567: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_660, [2048, 1280]);  mul_660 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_245: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_339, [1, 0]);  primals_339 = None
        mm_67: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_567, permute_245);  permute_245 = None
        mm_68: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_246, view_567);  permute_246 = None
        sum_101: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_567, [0], True);  view_567 = None
        view_568: "f32[1280]" = torch.ops.aten.reshape.default(sum_101, [1280]);  sum_101 = None
        view_569: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_67, [4, 512, 5120]);  mm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_334: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_110, [4, 512, 5120]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_332: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_334, 0.5)
        mul_661: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_569, mul_332);  mul_332 = None
        pow_28: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_334, 3.0)
        mul_333: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_28, 0.044715);  pow_28 = None
        add_225: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_334, mul_333);  mul_333 = None
        mul_334: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_225, 0.7978845608028654);  add_225 = None
        tanh_27: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_334);  mul_334 = None
        add_226: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_27, 1.0)
        mul_662: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_569, add_226);  view_569 = add_226 = None
        mul_663: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_27, tanh_27);  tanh_27 = None
        sub_134: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_663);  mul_663 = None
        mul_664: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_661, sub_134);  mul_661 = sub_134 = None
        mul_665: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_664, 0.7978845608028654);  mul_664 = None
        mul_666: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_665, 0.044715)
        pow_45: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_334, 2.0);  view_334 = None
        mul_667: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_45, 3.0);  pow_45 = None
        mul_668: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_666, mul_667);  mul_666 = mul_667 = None
        add_327: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_665, mul_668);  mul_665 = mul_668 = None
        mul_669: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_662, 0.5);  mul_662 = None
        add_328: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_327, mul_669);  add_327 = mul_669 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_570: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_328, [2048, 5120]);  add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_247: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_337, [1, 0]);  primals_337 = None
        mm_69: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_570, permute_247);  permute_247 = None
        mm_70: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_248, view_570);  permute_248 = None
        sum_102: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_570, [0], True);  view_570 = None
        view_571: "f32[5120]" = torch.ops.aten.reshape.default(sum_102, [5120]);  sum_102 = None
        view_572: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_69, [4, 512, 1280]);  mm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_671: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_572, primals_334);  primals_334 = None
        mul_672: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_671, 1280)
        sum_103: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True)
        mul_673: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_671, mul_330);  mul_671 = None
        sum_104: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_673, [2], True);  mul_673 = None
        mul_674: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_330, sum_104);  sum_104 = None
        sub_136: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_672, sum_103);  mul_672 = sum_103 = None
        sub_137: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_136, mul_674);  sub_136 = mul_674 = None
        mul_675: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_17, sub_137);  div_17 = sub_137 = None
        mul_676: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_572, mul_330);  mul_330 = None
        sum_105: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_676, [0, 1]);  mul_676 = None
        sum_106: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_572, [0, 1]);  view_572 = None
        add_329: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_326, mul_675);  add_326 = mul_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_17: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_677: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_678: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_329, mul_677);  mul_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_573: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_678, [2048, 1280]);  mul_678 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_249: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_333, [1, 0]);  primals_333 = None
        mm_71: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_573, permute_249);  permute_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_111: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_302, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_330: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_111, [4, 512, -1]);  permute_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_331: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_330, [-1, 1280]);  view_330 = None
        permute_250: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_331, [1, 0]);  view_331 = None
        mm_72: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_250, view_573);  permute_250 = None
        sum_107: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_573, [0], True);  view_573 = None
        view_574: "f32[1280]" = torch.ops.aten.reshape.default(sum_107, [1280]);  sum_107 = None
        view_575: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_71, [4, 512, 1280]);  mm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_576: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_575, [4, 512, 20, 64]);  view_575 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_251: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_8 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_251, permute_110, permute_108, permute_109, expand_2, getitem_302, getitem_303, getitem_304, getitem_305, 0.1, [True, True, True, False], scale = 0.125);  permute_251 = permute_110 = permute_108 = permute_109 = getitem_302 = getitem_303 = getitem_304 = getitem_305 = None
        getitem_430: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_8[0]
        getitem_431: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_8[1]
        getitem_432: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_8[2];  _scaled_dot_product_efficient_attention_backward_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_252: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_430, [0, 2, 1, 3]);  getitem_430 = None
        view_577: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_252, [4, 512, 1280]);  permute_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_253: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_432, [0, 2, 1, 3]);  getitem_432 = None
        view_578: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_253, [4, 512, 1280]);  permute_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_254: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_431, [0, 2, 1, 3]);  getitem_431 = None
        view_579: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_254, [4, 512, 1280]);  permute_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_9: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_577, view_579, view_578], 2);  view_577 = view_579 = view_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_580: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_9, [2048, 3840]);  cat_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_255: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_331, [1, 0]);  primals_331 = None
        mm_73: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_580, permute_255);  permute_255 = None
        mm_74: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_256, view_580);  permute_256 = None
        sum_108: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_580, [0], True);  view_580 = None
        view_581: "f32[3840]" = torch.ops.aten.reshape.default(sum_108, [3840]);  sum_108 = None
        view_582: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_73, [4, 512, 1280]);  mm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_680: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_582, primals_328);  primals_328 = None
        mul_681: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_680, 1280)
        sum_109: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_680, [2], True)
        mul_682: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_680, mul_326);  mul_680 = None
        sum_110: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_682, [2], True);  mul_682 = None
        mul_683: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_326, sum_110);  sum_110 = None
        sub_139: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_681, sum_109);  mul_681 = sum_109 = None
        sub_140: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_139, mul_683);  sub_139 = mul_683 = None
        mul_684: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_18, sub_140);  div_18 = sub_140 = None
        mul_685: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_582, mul_326);  mul_326 = None
        sum_111: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_685, [0, 1]);  mul_685 = None
        sum_112: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_582, [0, 1]);  view_582 = None
        add_330: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_329, mul_684);  add_329 = mul_684 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_18: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_686: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_687: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_330, mul_686);  mul_686 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_583: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_687, [2048, 1280]);  mul_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_257: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_327, [1, 0]);  primals_327 = None
        mm_75: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_583, permute_257);  permute_257 = None
        mm_76: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_258, view_583);  permute_258 = None
        sum_113: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_583, [0], True);  view_583 = None
        view_584: "f32[1280]" = torch.ops.aten.reshape.default(sum_113, [1280]);  sum_113 = None
        view_585: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_75, [4, 512, 5120]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_322: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_106, [4, 512, 5120]);  addmm_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_320: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_322, 0.5)
        mul_688: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_585, mul_320);  mul_320 = None
        pow_27: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_322, 3.0)
        mul_321: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_27, 0.044715);  pow_27 = None
        add_217: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_322, mul_321);  mul_321 = None
        mul_322: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_217, 0.7978845608028654);  add_217 = None
        tanh_26: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_322);  mul_322 = None
        add_218: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_26, 1.0)
        mul_689: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_585, add_218);  view_585 = add_218 = None
        mul_690: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_26, tanh_26);  tanh_26 = None
        sub_141: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_690);  mul_690 = None
        mul_691: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_688, sub_141);  mul_688 = sub_141 = None
        mul_692: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_691, 0.7978845608028654);  mul_691 = None
        mul_693: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_692, 0.044715)
        pow_46: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_322, 2.0);  view_322 = None
        mul_694: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_46, 3.0);  pow_46 = None
        mul_695: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_693, mul_694);  mul_693 = mul_694 = None
        add_331: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_692, mul_695);  mul_692 = mul_695 = None
        mul_696: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_689, 0.5);  mul_689 = None
        add_332: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_331, mul_696);  add_331 = mul_696 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_586: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_332, [2048, 5120]);  add_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_259: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_325, [1, 0]);  primals_325 = None
        mm_77: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_586, permute_259);  permute_259 = None
        mm_78: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_260, view_586);  permute_260 = None
        sum_114: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_586, [0], True);  view_586 = None
        view_587: "f32[5120]" = torch.ops.aten.reshape.default(sum_114, [5120]);  sum_114 = None
        view_588: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_77, [4, 512, 1280]);  mm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_698: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_588, primals_322);  primals_322 = None
        mul_699: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_698, 1280)
        sum_115: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_698, [2], True)
        mul_700: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_698, mul_318);  mul_698 = None
        sum_116: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_700, [2], True);  mul_700 = None
        mul_701: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_318, sum_116);  sum_116 = None
        sub_143: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_699, sum_115);  mul_699 = sum_115 = None
        sub_144: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_143, mul_701);  sub_143 = mul_701 = None
        mul_702: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_19, sub_144);  div_19 = sub_144 = None
        mul_703: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_588, mul_318);  mul_318 = None
        sum_117: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_703, [0, 1]);  mul_703 = None
        sum_118: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_588, [0, 1]);  view_588 = None
        add_333: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_330, mul_702);  add_330 = mul_702 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_19: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_704: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_705: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_333, mul_704);  mul_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_589: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_705, [2048, 1280]);  mul_705 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_261: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_321, [1, 0]);  primals_321 = None
        mm_79: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_589, permute_261);  permute_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_107: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_291, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_318: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_107, [4, 512, -1]);  permute_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_319: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_318, [-1, 1280]);  view_318 = None
        permute_262: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_319, [1, 0]);  view_319 = None
        mm_80: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_262, view_589);  permute_262 = None
        sum_119: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
        view_590: "f32[1280]" = torch.ops.aten.reshape.default(sum_119, [1280]);  sum_119 = None
        view_591: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_79, [4, 512, 1280]);  mm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_592: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_591, [4, 512, 20, 64]);  view_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_263: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_592, [0, 2, 1, 3]);  view_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_9 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_263, permute_106, permute_104, permute_105, expand_2, getitem_291, getitem_292, getitem_293, getitem_294, 0.1, [True, True, True, False], scale = 0.125);  permute_263 = permute_106 = permute_104 = permute_105 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = None
        getitem_434: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_9[0]
        getitem_435: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_9[1]
        getitem_436: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_9[2];  _scaled_dot_product_efficient_attention_backward_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_264: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_434, [0, 2, 1, 3]);  getitem_434 = None
        view_593: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_264, [4, 512, 1280]);  permute_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_265: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_436, [0, 2, 1, 3]);  getitem_436 = None
        view_594: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_265, [4, 512, 1280]);  permute_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_266: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_435, [0, 2, 1, 3]);  getitem_435 = None
        view_595: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_266, [4, 512, 1280]);  permute_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_10: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_593, view_595, view_594], 2);  view_593 = view_595 = view_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_596: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_10, [2048, 3840]);  cat_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_267: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_319, [1, 0]);  primals_319 = None
        mm_81: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_596, permute_267);  permute_267 = None
        mm_82: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_268, view_596);  permute_268 = None
        sum_120: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_596, [0], True);  view_596 = None
        view_597: "f32[3840]" = torch.ops.aten.reshape.default(sum_120, [3840]);  sum_120 = None
        view_598: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_81, [4, 512, 1280]);  mm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_707: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_598, primals_316);  primals_316 = None
        mul_708: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_707, 1280)
        sum_121: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True)
        mul_709: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_707, mul_314);  mul_707 = None
        sum_122: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_709, [2], True);  mul_709 = None
        mul_710: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_314, sum_122);  sum_122 = None
        sub_146: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_708, sum_121);  mul_708 = sum_121 = None
        sub_147: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_146, mul_710);  sub_146 = mul_710 = None
        mul_711: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_20, sub_147);  div_20 = sub_147 = None
        mul_712: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_598, mul_314);  mul_314 = None
        sum_123: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_712, [0, 1]);  mul_712 = None
        sum_124: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_598, [0, 1]);  view_598 = None
        add_334: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_333, mul_711);  add_333 = mul_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_713: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_714: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_334, mul_713);  mul_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_599: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_714, [2048, 1280]);  mul_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_269: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_315, [1, 0]);  primals_315 = None
        mm_83: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_599, permute_269);  permute_269 = None
        mm_84: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_270, view_599);  permute_270 = None
        sum_125: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_599, [0], True);  view_599 = None
        view_600: "f32[1280]" = torch.ops.aten.reshape.default(sum_125, [1280]);  sum_125 = None
        view_601: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_83, [4, 512, 5120]);  mm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_310: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_102, [4, 512, 5120]);  addmm_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_308: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_310, 0.5)
        mul_715: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_601, mul_308);  mul_308 = None
        pow_26: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_310, 3.0)
        mul_309: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_26, 0.044715);  pow_26 = None
        add_209: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_310, mul_309);  mul_309 = None
        mul_310: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_209, 0.7978845608028654);  add_209 = None
        tanh_25: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_310);  mul_310 = None
        add_210: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_25, 1.0)
        mul_716: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_601, add_210);  view_601 = add_210 = None
        mul_717: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_25, tanh_25);  tanh_25 = None
        sub_148: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_717);  mul_717 = None
        mul_718: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_715, sub_148);  mul_715 = sub_148 = None
        mul_719: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_718, 0.7978845608028654);  mul_718 = None
        mul_720: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_719, 0.044715)
        pow_47: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_310, 2.0);  view_310 = None
        mul_721: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_47, 3.0);  pow_47 = None
        mul_722: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_720, mul_721);  mul_720 = mul_721 = None
        add_335: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_719, mul_722);  mul_719 = mul_722 = None
        mul_723: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_716, 0.5);  mul_716 = None
        add_336: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_335, mul_723);  add_335 = mul_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_602: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_336, [2048, 5120]);  add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_271: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_313, [1, 0]);  primals_313 = None
        mm_85: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_602, permute_271);  permute_271 = None
        mm_86: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_272, view_602);  permute_272 = None
        sum_126: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_602, [0], True);  view_602 = None
        view_603: "f32[5120]" = torch.ops.aten.reshape.default(sum_126, [5120]);  sum_126 = None
        view_604: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_85, [4, 512, 1280]);  mm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_725: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_604, primals_310);  primals_310 = None
        mul_726: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_725, 1280)
        sum_127: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_725, [2], True)
        mul_727: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_725, mul_306);  mul_725 = None
        sum_128: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_727, [2], True);  mul_727 = None
        mul_728: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_306, sum_128);  sum_128 = None
        sub_150: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_726, sum_127);  mul_726 = sum_127 = None
        sub_151: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_150, mul_728);  sub_150 = mul_728 = None
        mul_729: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_21, sub_151);  div_21 = sub_151 = None
        mul_730: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_604, mul_306);  mul_306 = None
        sum_129: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_730, [0, 1]);  mul_730 = None
        sum_130: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_604, [0, 1]);  view_604 = None
        add_337: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_334, mul_729);  add_334 = mul_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_21: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_731: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_732: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_337, mul_731);  mul_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_605: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_732, [2048, 1280]);  mul_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_273: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_309, [1, 0]);  primals_309 = None
        mm_87: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_605, permute_273);  permute_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_103: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_280, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_306: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_103, [4, 512, -1]);  permute_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_307: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_306, [-1, 1280]);  view_306 = None
        permute_274: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_307, [1, 0]);  view_307 = None
        mm_88: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_274, view_605);  permute_274 = None
        sum_131: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_605, [0], True);  view_605 = None
        view_606: "f32[1280]" = torch.ops.aten.reshape.default(sum_131, [1280]);  sum_131 = None
        view_607: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_87, [4, 512, 1280]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_608: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_607, [4, 512, 20, 64]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_275: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_10 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_275, permute_102, permute_100, permute_101, expand_2, getitem_280, getitem_281, getitem_282, getitem_283, 0.1, [True, True, True, False], scale = 0.125);  permute_275 = permute_102 = permute_100 = permute_101 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = None
        getitem_438: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_10[0]
        getitem_439: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_10[1]
        getitem_440: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_10[2];  _scaled_dot_product_efficient_attention_backward_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_276: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_438, [0, 2, 1, 3]);  getitem_438 = None
        view_609: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_276, [4, 512, 1280]);  permute_276 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_277: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_440, [0, 2, 1, 3]);  getitem_440 = None
        view_610: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_277, [4, 512, 1280]);  permute_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_278: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_439, [0, 2, 1, 3]);  getitem_439 = None
        view_611: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_278, [4, 512, 1280]);  permute_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_11: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_609, view_611, view_610], 2);  view_609 = view_611 = view_610 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_612: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_11, [2048, 3840]);  cat_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_279: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_307, [1, 0]);  primals_307 = None
        mm_89: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_612, permute_279);  permute_279 = None
        mm_90: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_280, view_612);  permute_280 = None
        sum_132: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        view_613: "f32[3840]" = torch.ops.aten.reshape.default(sum_132, [3840]);  sum_132 = None
        view_614: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_89, [4, 512, 1280]);  mm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_734: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_614, primals_304);  primals_304 = None
        mul_735: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_734, 1280)
        sum_133: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_734, [2], True)
        mul_736: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_734, mul_302);  mul_734 = None
        sum_134: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_736, [2], True);  mul_736 = None
        mul_737: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_302, sum_134);  sum_134 = None
        sub_153: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_735, sum_133);  mul_735 = sum_133 = None
        sub_154: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_153, mul_737);  sub_153 = mul_737 = None
        mul_738: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_22, sub_154);  div_22 = sub_154 = None
        mul_739: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_614, mul_302);  mul_302 = None
        sum_135: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_739, [0, 1]);  mul_739 = None
        sum_136: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_614, [0, 1]);  view_614 = None
        add_338: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_337, mul_738);  add_337 = mul_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_22: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_740: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_741: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_338, mul_740);  mul_740 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_615: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_741, [2048, 1280]);  mul_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_281: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_303, [1, 0]);  primals_303 = None
        mm_91: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_615, permute_281);  permute_281 = None
        mm_92: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_282, view_615);  permute_282 = None
        sum_137: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_615, [0], True);  view_615 = None
        view_616: "f32[1280]" = torch.ops.aten.reshape.default(sum_137, [1280]);  sum_137 = None
        view_617: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_91, [4, 512, 5120]);  mm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_298: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_98, [4, 512, 5120]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_296: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_298, 0.5)
        mul_742: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_617, mul_296);  mul_296 = None
        pow_25: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_298, 3.0)
        mul_297: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_25, 0.044715);  pow_25 = None
        add_201: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_298, mul_297);  mul_297 = None
        mul_298: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_201, 0.7978845608028654);  add_201 = None
        tanh_24: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_298);  mul_298 = None
        add_202: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_24, 1.0)
        mul_743: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_617, add_202);  view_617 = add_202 = None
        mul_744: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_24, tanh_24);  tanh_24 = None
        sub_155: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_744);  mul_744 = None
        mul_745: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_742, sub_155);  mul_742 = sub_155 = None
        mul_746: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_745, 0.7978845608028654);  mul_745 = None
        mul_747: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_746, 0.044715)
        pow_48: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_298, 2.0);  view_298 = None
        mul_748: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_48, 3.0);  pow_48 = None
        mul_749: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_747, mul_748);  mul_747 = mul_748 = None
        add_339: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_746, mul_749);  mul_746 = mul_749 = None
        mul_750: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_743, 0.5);  mul_743 = None
        add_340: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_339, mul_750);  add_339 = mul_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_618: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_340, [2048, 5120]);  add_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_283: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_301, [1, 0]);  primals_301 = None
        mm_93: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_618, permute_283);  permute_283 = None
        mm_94: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_284, view_618);  permute_284 = None
        sum_138: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_618, [0], True);  view_618 = None
        view_619: "f32[5120]" = torch.ops.aten.reshape.default(sum_138, [5120]);  sum_138 = None
        view_620: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_93, [4, 512, 1280]);  mm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_752: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_620, primals_298);  primals_298 = None
        mul_753: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_752, 1280)
        sum_139: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_752, [2], True)
        mul_754: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_752, mul_294);  mul_752 = None
        sum_140: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_754, [2], True);  mul_754 = None
        mul_755: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_294, sum_140);  sum_140 = None
        sub_157: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_753, sum_139);  mul_753 = sum_139 = None
        sub_158: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_157, mul_755);  sub_157 = mul_755 = None
        mul_756: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_23, sub_158);  div_23 = sub_158 = None
        mul_757: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_620, mul_294);  mul_294 = None
        sum_141: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_757, [0, 1]);  mul_757 = None
        sum_142: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_620, [0, 1]);  view_620 = None
        add_341: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_338, mul_756);  add_338 = mul_756 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_23: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_758: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_759: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_341, mul_758);  mul_758 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_621: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_759, [2048, 1280]);  mul_759 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_285: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_297, [1, 0]);  primals_297 = None
        mm_95: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_621, permute_285);  permute_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_99: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_269, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_294: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_99, [4, 512, -1]);  permute_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_295: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_294, [-1, 1280]);  view_294 = None
        permute_286: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_295, [1, 0]);  view_295 = None
        mm_96: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_286, view_621);  permute_286 = None
        sum_143: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_621, [0], True);  view_621 = None
        view_622: "f32[1280]" = torch.ops.aten.reshape.default(sum_143, [1280]);  sum_143 = None
        view_623: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_95, [4, 512, 1280]);  mm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_624: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_623, [4, 512, 20, 64]);  view_623 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_11 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_287, permute_98, permute_96, permute_97, expand_2, getitem_269, getitem_270, getitem_271, getitem_272, 0.1, [True, True, True, False], scale = 0.125);  permute_287 = permute_98 = permute_96 = permute_97 = getitem_269 = getitem_270 = getitem_271 = getitem_272 = None
        getitem_442: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_11[0]
        getitem_443: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_11[1]
        getitem_444: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_11[2];  _scaled_dot_product_efficient_attention_backward_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_288: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_442, [0, 2, 1, 3]);  getitem_442 = None
        view_625: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_288, [4, 512, 1280]);  permute_288 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_289: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_444, [0, 2, 1, 3]);  getitem_444 = None
        view_626: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_289, [4, 512, 1280]);  permute_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_290: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_443, [0, 2, 1, 3]);  getitem_443 = None
        view_627: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_290, [4, 512, 1280]);  permute_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_12: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_625, view_627, view_626], 2);  view_625 = view_627 = view_626 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_628: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_12, [2048, 3840]);  cat_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_291: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_295, [1, 0]);  primals_295 = None
        mm_97: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_628, permute_291);  permute_291 = None
        mm_98: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_292, view_628);  permute_292 = None
        sum_144: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        view_629: "f32[3840]" = torch.ops.aten.reshape.default(sum_144, [3840]);  sum_144 = None
        view_630: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_97, [4, 512, 1280]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_761: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_630, primals_292);  primals_292 = None
        mul_762: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_761, 1280)
        sum_145: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_761, [2], True)
        mul_763: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_761, mul_290);  mul_761 = None
        sum_146: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_763, [2], True);  mul_763 = None
        mul_764: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_290, sum_146);  sum_146 = None
        sub_160: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_762, sum_145);  mul_762 = sum_145 = None
        sub_161: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_160, mul_764);  sub_160 = mul_764 = None
        mul_765: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_24, sub_161);  div_24 = sub_161 = None
        mul_766: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_630, mul_290);  mul_290 = None
        sum_147: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_766, [0, 1]);  mul_766 = None
        sum_148: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_630, [0, 1]);  view_630 = None
        add_342: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_341, mul_765);  add_341 = mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_24: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_767: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_768: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_342, mul_767);  mul_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_631: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_768, [2048, 1280]);  mul_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_293: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_291, [1, 0]);  primals_291 = None
        mm_99: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_631, permute_293);  permute_293 = None
        mm_100: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_294, view_631);  permute_294 = None
        sum_149: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        view_632: "f32[1280]" = torch.ops.aten.reshape.default(sum_149, [1280]);  sum_149 = None
        view_633: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_99, [4, 512, 5120]);  mm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_286: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_94, [4, 512, 5120]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_284: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_286, 0.5)
        mul_769: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_633, mul_284);  mul_284 = None
        pow_24: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_286, 3.0)
        mul_285: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_193: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_286, mul_285);  mul_285 = None
        mul_286: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_193, 0.7978845608028654);  add_193 = None
        tanh_23: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_286);  mul_286 = None
        add_194: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_23, 1.0)
        mul_770: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_633, add_194);  view_633 = add_194 = None
        mul_771: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_23, tanh_23);  tanh_23 = None
        sub_162: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_771);  mul_771 = None
        mul_772: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_769, sub_162);  mul_769 = sub_162 = None
        mul_773: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_772, 0.7978845608028654);  mul_772 = None
        mul_774: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_773, 0.044715)
        pow_49: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_286, 2.0);  view_286 = None
        mul_775: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_49, 3.0);  pow_49 = None
        mul_776: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_774, mul_775);  mul_774 = mul_775 = None
        add_343: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_773, mul_776);  mul_773 = mul_776 = None
        mul_777: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_770, 0.5);  mul_770 = None
        add_344: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_343, mul_777);  add_343 = mul_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_634: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_344, [2048, 5120]);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_295: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_289, [1, 0]);  primals_289 = None
        mm_101: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_634, permute_295);  permute_295 = None
        mm_102: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_296, view_634);  permute_296 = None
        sum_150: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_634, [0], True);  view_634 = None
        view_635: "f32[5120]" = torch.ops.aten.reshape.default(sum_150, [5120]);  sum_150 = None
        view_636: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_101, [4, 512, 1280]);  mm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_779: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_636, primals_286);  primals_286 = None
        mul_780: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_779, 1280)
        sum_151: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_779, [2], True)
        mul_781: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_779, mul_282);  mul_779 = None
        sum_152: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_781, [2], True);  mul_781 = None
        mul_782: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_282, sum_152);  sum_152 = None
        sub_164: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_780, sum_151);  mul_780 = sum_151 = None
        sub_165: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_164, mul_782);  sub_164 = mul_782 = None
        mul_783: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_25, sub_165);  div_25 = sub_165 = None
        mul_784: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_636, mul_282);  mul_282 = None
        sum_153: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_784, [0, 1]);  mul_784 = None
        sum_154: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_636, [0, 1]);  view_636 = None
        add_345: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_342, mul_783);  add_342 = mul_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_25: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_785: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_786: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_345, mul_785);  mul_785 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_637: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_786, [2048, 1280]);  mul_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_297: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_285, [1, 0]);  primals_285 = None
        mm_103: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_637, permute_297);  permute_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_95: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_258, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_282: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_95, [4, 512, -1]);  permute_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_283: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_282, [-1, 1280]);  view_282 = None
        permute_298: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_283, [1, 0]);  view_283 = None
        mm_104: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_298, view_637);  permute_298 = None
        sum_155: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_637, [0], True);  view_637 = None
        view_638: "f32[1280]" = torch.ops.aten.reshape.default(sum_155, [1280]);  sum_155 = None
        view_639: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_103, [4, 512, 1280]);  mm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_640: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_639, [4, 512, 20, 64]);  view_639 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_299: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_12 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_299, permute_94, permute_92, permute_93, expand_2, getitem_258, getitem_259, getitem_260, getitem_261, 0.1, [True, True, True, False], scale = 0.125);  permute_299 = permute_94 = permute_92 = permute_93 = getitem_258 = getitem_259 = getitem_260 = getitem_261 = None
        getitem_446: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_12[0]
        getitem_447: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_12[1]
        getitem_448: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_12[2];  _scaled_dot_product_efficient_attention_backward_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_300: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_446, [0, 2, 1, 3]);  getitem_446 = None
        view_641: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_300, [4, 512, 1280]);  permute_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_301: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_448, [0, 2, 1, 3]);  getitem_448 = None
        view_642: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_301, [4, 512, 1280]);  permute_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_302: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_447, [0, 2, 1, 3]);  getitem_447 = None
        view_643: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_302, [4, 512, 1280]);  permute_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_13: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_641, view_643, view_642], 2);  view_641 = view_643 = view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_644: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_13, [2048, 3840]);  cat_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_303: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_283, [1, 0]);  primals_283 = None
        mm_105: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_644, permute_303);  permute_303 = None
        mm_106: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_304, view_644);  permute_304 = None
        sum_156: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_644, [0], True);  view_644 = None
        view_645: "f32[3840]" = torch.ops.aten.reshape.default(sum_156, [3840]);  sum_156 = None
        view_646: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_105, [4, 512, 1280]);  mm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_788: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_646, primals_280);  primals_280 = None
        mul_789: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_788, 1280)
        sum_157: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_788, [2], True)
        mul_790: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_788, mul_278);  mul_788 = None
        sum_158: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_790, [2], True);  mul_790 = None
        mul_791: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_278, sum_158);  sum_158 = None
        sub_167: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_789, sum_157);  mul_789 = sum_157 = None
        sub_168: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_167, mul_791);  sub_167 = mul_791 = None
        mul_792: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_26, sub_168);  div_26 = sub_168 = None
        mul_793: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_646, mul_278);  mul_278 = None
        sum_159: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_793, [0, 1]);  mul_793 = None
        sum_160: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_646, [0, 1]);  view_646 = None
        add_346: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_345, mul_792);  add_345 = mul_792 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_794: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_795: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_346, mul_794);  mul_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_647: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_795, [2048, 1280]);  mul_795 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_305: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_279, [1, 0]);  primals_279 = None
        mm_107: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_647, permute_305);  permute_305 = None
        mm_108: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_306, view_647);  permute_306 = None
        sum_161: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_647, [0], True);  view_647 = None
        view_648: "f32[1280]" = torch.ops.aten.reshape.default(sum_161, [1280]);  sum_161 = None
        view_649: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_107, [4, 512, 5120]);  mm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_274: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_90, [4, 512, 5120]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_272: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_274, 0.5)
        mul_796: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_649, mul_272);  mul_272 = None
        pow_23: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_274, 3.0)
        mul_273: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_23, 0.044715);  pow_23 = None
        add_185: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_274, mul_273);  mul_273 = None
        mul_274: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_185, 0.7978845608028654);  add_185 = None
        tanh_22: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_274);  mul_274 = None
        add_186: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_22, 1.0)
        mul_797: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_649, add_186);  view_649 = add_186 = None
        mul_798: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_22, tanh_22);  tanh_22 = None
        sub_169: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_798);  mul_798 = None
        mul_799: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_796, sub_169);  mul_796 = sub_169 = None
        mul_800: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_799, 0.7978845608028654);  mul_799 = None
        mul_801: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_800, 0.044715)
        pow_50: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_274, 2.0);  view_274 = None
        mul_802: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_50, 3.0);  pow_50 = None
        mul_803: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_801, mul_802);  mul_801 = mul_802 = None
        add_347: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_800, mul_803);  mul_800 = mul_803 = None
        mul_804: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_797, 0.5);  mul_797 = None
        add_348: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_347, mul_804);  add_347 = mul_804 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_650: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_348, [2048, 5120]);  add_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_307: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_277, [1, 0]);  primals_277 = None
        mm_109: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_650, permute_307);  permute_307 = None
        mm_110: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_308, view_650);  permute_308 = None
        sum_162: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        view_651: "f32[5120]" = torch.ops.aten.reshape.default(sum_162, [5120]);  sum_162 = None
        view_652: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_109, [4, 512, 1280]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_806: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_652, primals_274);  primals_274 = None
        mul_807: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_806, 1280)
        sum_163: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_806, [2], True)
        mul_808: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_806, mul_270);  mul_806 = None
        sum_164: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_808, [2], True);  mul_808 = None
        mul_809: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_270, sum_164);  sum_164 = None
        sub_171: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_807, sum_163);  mul_807 = sum_163 = None
        sub_172: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_171, mul_809);  sub_171 = mul_809 = None
        mul_810: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_27, sub_172);  div_27 = sub_172 = None
        mul_811: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_652, mul_270);  mul_270 = None
        sum_165: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_811, [0, 1]);  mul_811 = None
        sum_166: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_652, [0, 1]);  view_652 = None
        add_349: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_346, mul_810);  add_346 = mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_27: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_812: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_813: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_349, mul_812);  mul_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_653: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_813, [2048, 1280]);  mul_813 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_309: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_273, [1, 0]);  primals_273 = None
        mm_111: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_653, permute_309);  permute_309 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_91: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_247, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_270: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_91, [4, 512, -1]);  permute_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_271: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_270, [-1, 1280]);  view_270 = None
        permute_310: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_271, [1, 0]);  view_271 = None
        mm_112: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_310, view_653);  permute_310 = None
        sum_167: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        view_654: "f32[1280]" = torch.ops.aten.reshape.default(sum_167, [1280]);  sum_167 = None
        view_655: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_111, [4, 512, 1280]);  mm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_656: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_655, [4, 512, 20, 64]);  view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_311: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_13 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_311, permute_90, permute_88, permute_89, expand_2, getitem_247, getitem_248, getitem_249, getitem_250, 0.1, [True, True, True, False], scale = 0.125);  permute_311 = permute_90 = permute_88 = permute_89 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = None
        getitem_450: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_13[0]
        getitem_451: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_13[1]
        getitem_452: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_13[2];  _scaled_dot_product_efficient_attention_backward_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_312: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_450, [0, 2, 1, 3]);  getitem_450 = None
        view_657: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_312, [4, 512, 1280]);  permute_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_313: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_452, [0, 2, 1, 3]);  getitem_452 = None
        view_658: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_313, [4, 512, 1280]);  permute_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_314: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_451, [0, 2, 1, 3]);  getitem_451 = None
        view_659: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_314, [4, 512, 1280]);  permute_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_14: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_657, view_659, view_658], 2);  view_657 = view_659 = view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_660: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_14, [2048, 3840]);  cat_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_315: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_271, [1, 0]);  primals_271 = None
        mm_113: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_660, permute_315);  permute_315 = None
        mm_114: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_316, view_660);  permute_316 = None
        sum_168: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
        view_661: "f32[3840]" = torch.ops.aten.reshape.default(sum_168, [3840]);  sum_168 = None
        view_662: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_113, [4, 512, 1280]);  mm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_815: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_662, primals_268);  primals_268 = None
        mul_816: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_815, 1280)
        sum_169: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_815, [2], True)
        mul_817: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_815, mul_266);  mul_815 = None
        sum_170: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True);  mul_817 = None
        mul_818: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_266, sum_170);  sum_170 = None
        sub_174: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_816, sum_169);  mul_816 = sum_169 = None
        sub_175: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_174, mul_818);  sub_174 = mul_818 = None
        mul_819: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_28, sub_175);  div_28 = sub_175 = None
        mul_820: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_662, mul_266);  mul_266 = None
        sum_171: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_820, [0, 1]);  mul_820 = None
        sum_172: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_662, [0, 1]);  view_662 = None
        add_350: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_349, mul_819);  add_349 = mul_819 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_28: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_821: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_822: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_350, mul_821);  mul_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_663: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_822, [2048, 1280]);  mul_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_317: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_267, [1, 0]);  primals_267 = None
        mm_115: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_663, permute_317);  permute_317 = None
        mm_116: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_318, view_663);  permute_318 = None
        sum_173: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
        view_664: "f32[1280]" = torch.ops.aten.reshape.default(sum_173, [1280]);  sum_173 = None
        view_665: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_115, [4, 512, 5120]);  mm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_262: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_86, [4, 512, 5120]);  addmm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_260: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_262, 0.5)
        mul_823: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_665, mul_260);  mul_260 = None
        pow_22: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_262, 3.0)
        mul_261: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_22, 0.044715);  pow_22 = None
        add_177: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_262, mul_261);  mul_261 = None
        mul_262: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_177, 0.7978845608028654);  add_177 = None
        tanh_21: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_262);  mul_262 = None
        add_178: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_21, 1.0)
        mul_824: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_665, add_178);  view_665 = add_178 = None
        mul_825: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_21, tanh_21);  tanh_21 = None
        sub_176: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_825);  mul_825 = None
        mul_826: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_823, sub_176);  mul_823 = sub_176 = None
        mul_827: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_826, 0.7978845608028654);  mul_826 = None
        mul_828: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_827, 0.044715)
        pow_51: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_262, 2.0);  view_262 = None
        mul_829: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_51, 3.0);  pow_51 = None
        mul_830: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_828, mul_829);  mul_828 = mul_829 = None
        add_351: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_827, mul_830);  mul_827 = mul_830 = None
        mul_831: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_824, 0.5);  mul_824 = None
        add_352: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_351, mul_831);  add_351 = mul_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_666: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_352, [2048, 5120]);  add_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_319: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_265, [1, 0]);  primals_265 = None
        mm_117: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_666, permute_319);  permute_319 = None
        mm_118: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_320, view_666);  permute_320 = None
        sum_174: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_666, [0], True);  view_666 = None
        view_667: "f32[5120]" = torch.ops.aten.reshape.default(sum_174, [5120]);  sum_174 = None
        view_668: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_117, [4, 512, 1280]);  mm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_833: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_668, primals_262);  primals_262 = None
        mul_834: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_833, 1280)
        sum_175: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_833, [2], True)
        mul_835: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_833, mul_258);  mul_833 = None
        sum_176: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_835, [2], True);  mul_835 = None
        mul_836: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_258, sum_176);  sum_176 = None
        sub_178: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_834, sum_175);  mul_834 = sum_175 = None
        sub_179: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_178, mul_836);  sub_178 = mul_836 = None
        mul_837: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_29, sub_179);  div_29 = sub_179 = None
        mul_838: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_668, mul_258);  mul_258 = None
        sum_177: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_838, [0, 1]);  mul_838 = None
        sum_178: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_668, [0, 1]);  view_668 = None
        add_353: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_350, mul_837);  add_350 = mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_29: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_839: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_840: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_353, mul_839);  mul_839 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_669: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_840, [2048, 1280]);  mul_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_321: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_261, [1, 0]);  primals_261 = None
        mm_119: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_669, permute_321);  permute_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_87: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_236, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_258: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_87, [4, 512, -1]);  permute_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_259: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_258, [-1, 1280]);  view_258 = None
        permute_322: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_259, [1, 0]);  view_259 = None
        mm_120: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_322, view_669);  permute_322 = None
        sum_179: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_669, [0], True);  view_669 = None
        view_670: "f32[1280]" = torch.ops.aten.reshape.default(sum_179, [1280]);  sum_179 = None
        view_671: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_119, [4, 512, 1280]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_672: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_671, [4, 512, 20, 64]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_323: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_14 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_323, permute_86, permute_84, permute_85, expand_2, getitem_236, getitem_237, getitem_238, getitem_239, 0.1, [True, True, True, False], scale = 0.125);  permute_323 = permute_86 = permute_84 = permute_85 = getitem_236 = getitem_237 = getitem_238 = getitem_239 = None
        getitem_454: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_14[0]
        getitem_455: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_14[1]
        getitem_456: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_14[2];  _scaled_dot_product_efficient_attention_backward_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_324: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_454, [0, 2, 1, 3]);  getitem_454 = None
        view_673: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_324, [4, 512, 1280]);  permute_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_325: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_456, [0, 2, 1, 3]);  getitem_456 = None
        view_674: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_325, [4, 512, 1280]);  permute_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_326: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_455, [0, 2, 1, 3]);  getitem_455 = None
        view_675: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_326, [4, 512, 1280]);  permute_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_15: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_673, view_675, view_674], 2);  view_673 = view_675 = view_674 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_676: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_15, [2048, 3840]);  cat_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_327: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_259, [1, 0]);  primals_259 = None
        mm_121: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_676, permute_327);  permute_327 = None
        mm_122: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_328, view_676);  permute_328 = None
        sum_180: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_676, [0], True);  view_676 = None
        view_677: "f32[3840]" = torch.ops.aten.reshape.default(sum_180, [3840]);  sum_180 = None
        view_678: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_121, [4, 512, 1280]);  mm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_842: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_678, primals_256);  primals_256 = None
        mul_843: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_842, 1280)
        sum_181: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_842, [2], True)
        mul_844: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_842, mul_254);  mul_842 = None
        sum_182: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_844, [2], True);  mul_844 = None
        mul_845: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_254, sum_182);  sum_182 = None
        sub_181: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_843, sum_181);  mul_843 = sum_181 = None
        sub_182: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_181, mul_845);  sub_181 = mul_845 = None
        mul_846: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_30, sub_182);  div_30 = sub_182 = None
        mul_847: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_678, mul_254);  mul_254 = None
        sum_183: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_847, [0, 1]);  mul_847 = None
        sum_184: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_678, [0, 1]);  view_678 = None
        add_354: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_353, mul_846);  add_353 = mul_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_30: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_848: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_849: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_354, mul_848);  mul_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_679: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_849, [2048, 1280]);  mul_849 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_329: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_255, [1, 0]);  primals_255 = None
        mm_123: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_679, permute_329);  permute_329 = None
        mm_124: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_330, view_679);  permute_330 = None
        sum_185: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_679, [0], True);  view_679 = None
        view_680: "f32[1280]" = torch.ops.aten.reshape.default(sum_185, [1280]);  sum_185 = None
        view_681: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_123, [4, 512, 5120]);  mm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_250: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_82, [4, 512, 5120]);  addmm_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_248: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_250, 0.5)
        mul_850: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_681, mul_248);  mul_248 = None
        pow_21: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_250, 3.0)
        mul_249: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_169: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_250, mul_249);  mul_249 = None
        mul_250: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_169, 0.7978845608028654);  add_169 = None
        tanh_20: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_250);  mul_250 = None
        add_170: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_20, 1.0)
        mul_851: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_681, add_170);  view_681 = add_170 = None
        mul_852: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_20, tanh_20);  tanh_20 = None
        sub_183: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_852);  mul_852 = None
        mul_853: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_850, sub_183);  mul_850 = sub_183 = None
        mul_854: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_853, 0.7978845608028654);  mul_853 = None
        mul_855: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_854, 0.044715)
        pow_52: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_250, 2.0);  view_250 = None
        mul_856: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_52, 3.0);  pow_52 = None
        mul_857: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_855, mul_856);  mul_855 = mul_856 = None
        add_355: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_854, mul_857);  mul_854 = mul_857 = None
        mul_858: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_851, 0.5);  mul_851 = None
        add_356: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_355, mul_858);  add_355 = mul_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_682: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_356, [2048, 5120]);  add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_331: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_253, [1, 0]);  primals_253 = None
        mm_125: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_682, permute_331);  permute_331 = None
        mm_126: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_332, view_682);  permute_332 = None
        sum_186: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_682, [0], True);  view_682 = None
        view_683: "f32[5120]" = torch.ops.aten.reshape.default(sum_186, [5120]);  sum_186 = None
        view_684: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_125, [4, 512, 1280]);  mm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_860: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_684, primals_250);  primals_250 = None
        mul_861: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_860, 1280)
        sum_187: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_860, [2], True)
        mul_862: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_860, mul_246);  mul_860 = None
        sum_188: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_862, [2], True);  mul_862 = None
        mul_863: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_246, sum_188);  sum_188 = None
        sub_185: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_861, sum_187);  mul_861 = sum_187 = None
        sub_186: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_185, mul_863);  sub_185 = mul_863 = None
        mul_864: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_31, sub_186);  div_31 = sub_186 = None
        mul_865: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_684, mul_246);  mul_246 = None
        sum_189: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_865, [0, 1]);  mul_865 = None
        sum_190: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_684, [0, 1]);  view_684 = None
        add_357: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_354, mul_864);  add_354 = mul_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_31: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_866: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_867: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_357, mul_866);  mul_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_685: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_867, [2048, 1280]);  mul_867 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_333: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_249, [1, 0]);  primals_249 = None
        mm_127: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_685, permute_333);  permute_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_83: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_225, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_246: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_83, [4, 512, -1]);  permute_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_247: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_246, [-1, 1280]);  view_246 = None
        permute_334: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_247, [1, 0]);  view_247 = None
        mm_128: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_334, view_685);  permute_334 = None
        sum_191: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_685, [0], True);  view_685 = None
        view_686: "f32[1280]" = torch.ops.aten.reshape.default(sum_191, [1280]);  sum_191 = None
        view_687: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_127, [4, 512, 1280]);  mm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_688: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_687, [4, 512, 20, 64]);  view_687 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_335: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_688, [0, 2, 1, 3]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_15 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_335, permute_82, permute_80, permute_81, expand_2, getitem_225, getitem_226, getitem_227, getitem_228, 0.1, [True, True, True, False], scale = 0.125);  permute_335 = permute_82 = permute_80 = permute_81 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = None
        getitem_458: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_15[0]
        getitem_459: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_15[1]
        getitem_460: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_15[2];  _scaled_dot_product_efficient_attention_backward_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_336: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_458, [0, 2, 1, 3]);  getitem_458 = None
        view_689: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_336, [4, 512, 1280]);  permute_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_337: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_460, [0, 2, 1, 3]);  getitem_460 = None
        view_690: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_337, [4, 512, 1280]);  permute_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_338: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_459, [0, 2, 1, 3]);  getitem_459 = None
        view_691: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_338, [4, 512, 1280]);  permute_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_16: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_689, view_691, view_690], 2);  view_689 = view_691 = view_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_692: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_16, [2048, 3840]);  cat_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_339: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_247, [1, 0]);  primals_247 = None
        mm_129: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_692, permute_339);  permute_339 = None
        mm_130: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_340, view_692);  permute_340 = None
        sum_192: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_692, [0], True);  view_692 = None
        view_693: "f32[3840]" = torch.ops.aten.reshape.default(sum_192, [3840]);  sum_192 = None
        view_694: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_129, [4, 512, 1280]);  mm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_869: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_694, primals_244);  primals_244 = None
        mul_870: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_869, 1280)
        sum_193: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_869, [2], True)
        mul_871: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_869, mul_242);  mul_869 = None
        sum_194: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_871, [2], True);  mul_871 = None
        mul_872: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_242, sum_194);  sum_194 = None
        sub_188: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_870, sum_193);  mul_870 = sum_193 = None
        sub_189: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_188, mul_872);  sub_188 = mul_872 = None
        mul_873: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_32, sub_189);  div_32 = sub_189 = None
        mul_874: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_694, mul_242);  mul_242 = None
        sum_195: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_874, [0, 1]);  mul_874 = None
        sum_196: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_694, [0, 1]);  view_694 = None
        add_358: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_357, mul_873);  add_357 = mul_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_875: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_876: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_358, mul_875);  mul_875 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_695: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_876, [2048, 1280]);  mul_876 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_341: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_243, [1, 0]);  primals_243 = None
        mm_131: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_695, permute_341);  permute_341 = None
        mm_132: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_342, view_695);  permute_342 = None
        sum_197: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_695, [0], True);  view_695 = None
        view_696: "f32[1280]" = torch.ops.aten.reshape.default(sum_197, [1280]);  sum_197 = None
        view_697: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_131, [4, 512, 5120]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_238: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_78, [4, 512, 5120]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_236: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_238, 0.5)
        mul_877: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_697, mul_236);  mul_236 = None
        pow_20: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 3.0)
        mul_237: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_20, 0.044715);  pow_20 = None
        add_161: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_238, mul_237);  mul_237 = None
        mul_238: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_161, 0.7978845608028654);  add_161 = None
        tanh_19: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_238);  mul_238 = None
        add_162: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_19, 1.0)
        mul_878: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_697, add_162);  view_697 = add_162 = None
        mul_879: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_19, tanh_19);  tanh_19 = None
        sub_190: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_879);  mul_879 = None
        mul_880: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_877, sub_190);  mul_877 = sub_190 = None
        mul_881: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_880, 0.7978845608028654);  mul_880 = None
        mul_882: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_881, 0.044715)
        pow_53: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_238, 2.0);  view_238 = None
        mul_883: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_53, 3.0);  pow_53 = None
        mul_884: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_882, mul_883);  mul_882 = mul_883 = None
        add_359: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_881, mul_884);  mul_881 = mul_884 = None
        mul_885: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_878, 0.5);  mul_878 = None
        add_360: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_359, mul_885);  add_359 = mul_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_698: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_360, [2048, 5120]);  add_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_343: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_241, [1, 0]);  primals_241 = None
        mm_133: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_698, permute_343);  permute_343 = None
        mm_134: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_344, view_698);  permute_344 = None
        sum_198: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_698, [0], True);  view_698 = None
        view_699: "f32[5120]" = torch.ops.aten.reshape.default(sum_198, [5120]);  sum_198 = None
        view_700: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_133, [4, 512, 1280]);  mm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_887: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_700, primals_238);  primals_238 = None
        mul_888: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_887, 1280)
        sum_199: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_887, [2], True)
        mul_889: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_887, mul_234);  mul_887 = None
        sum_200: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
        mul_890: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_234, sum_200);  sum_200 = None
        sub_192: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_888, sum_199);  mul_888 = sum_199 = None
        sub_193: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_192, mul_890);  sub_192 = mul_890 = None
        mul_891: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_33, sub_193);  div_33 = sub_193 = None
        mul_892: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_700, mul_234);  mul_234 = None
        sum_201: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_892, [0, 1]);  mul_892 = None
        sum_202: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_700, [0, 1]);  view_700 = None
        add_361: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_358, mul_891);  add_358 = mul_891 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_33: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_893: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_894: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_361, mul_893);  mul_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_701: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_894, [2048, 1280]);  mul_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_345: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_237, [1, 0]);  primals_237 = None
        mm_135: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_701, permute_345);  permute_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_79: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_214, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_234: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_79, [4, 512, -1]);  permute_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_235: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_234, [-1, 1280]);  view_234 = None
        permute_346: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_235, [1, 0]);  view_235 = None
        mm_136: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_346, view_701);  permute_346 = None
        sum_203: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
        view_702: "f32[1280]" = torch.ops.aten.reshape.default(sum_203, [1280]);  sum_203 = None
        view_703: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_135, [4, 512, 1280]);  mm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_704: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_703, [4, 512, 20, 64]);  view_703 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_347: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_16 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_347, permute_78, permute_76, permute_77, expand_2, getitem_214, getitem_215, getitem_216, getitem_217, 0.1, [True, True, True, False], scale = 0.125);  permute_347 = permute_78 = permute_76 = permute_77 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = None
        getitem_462: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_16[0]
        getitem_463: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_16[1]
        getitem_464: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_16[2];  _scaled_dot_product_efficient_attention_backward_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_348: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_462, [0, 2, 1, 3]);  getitem_462 = None
        view_705: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_348, [4, 512, 1280]);  permute_348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_349: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_464, [0, 2, 1, 3]);  getitem_464 = None
        view_706: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_349, [4, 512, 1280]);  permute_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_350: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_463, [0, 2, 1, 3]);  getitem_463 = None
        view_707: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_350, [4, 512, 1280]);  permute_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_17: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_705, view_707, view_706], 2);  view_705 = view_707 = view_706 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_708: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_17, [2048, 3840]);  cat_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_351: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_235, [1, 0]);  primals_235 = None
        mm_137: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_708, permute_351);  permute_351 = None
        mm_138: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_352, view_708);  permute_352 = None
        sum_204: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_708, [0], True);  view_708 = None
        view_709: "f32[3840]" = torch.ops.aten.reshape.default(sum_204, [3840]);  sum_204 = None
        view_710: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_137, [4, 512, 1280]);  mm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_896: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_710, primals_232);  primals_232 = None
        mul_897: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_896, 1280)
        sum_205: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_896, [2], True)
        mul_898: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_896, mul_230);  mul_896 = None
        sum_206: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_898, [2], True);  mul_898 = None
        mul_899: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_230, sum_206);  sum_206 = None
        sub_195: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_897, sum_205);  mul_897 = sum_205 = None
        sub_196: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_195, mul_899);  sub_195 = mul_899 = None
        mul_900: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_34, sub_196);  div_34 = sub_196 = None
        mul_901: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_710, mul_230);  mul_230 = None
        sum_207: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_901, [0, 1]);  mul_901 = None
        sum_208: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_710, [0, 1]);  view_710 = None
        add_362: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_361, mul_900);  add_361 = mul_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_34: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_902: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_903: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_362, mul_902);  mul_902 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_711: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_903, [2048, 1280]);  mul_903 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_353: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_231, [1, 0]);  primals_231 = None
        mm_139: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_711, permute_353);  permute_353 = None
        mm_140: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_354, view_711);  permute_354 = None
        sum_209: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_711, [0], True);  view_711 = None
        view_712: "f32[1280]" = torch.ops.aten.reshape.default(sum_209, [1280]);  sum_209 = None
        view_713: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_139, [4, 512, 5120]);  mm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_226: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_74, [4, 512, 5120]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_224: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_226, 0.5)
        mul_904: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_713, mul_224);  mul_224 = None
        pow_19: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_226, 3.0)
        mul_225: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_19, 0.044715);  pow_19 = None
        add_153: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_226, mul_225);  mul_225 = None
        mul_226: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_153, 0.7978845608028654);  add_153 = None
        tanh_18: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_226);  mul_226 = None
        add_154: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_18, 1.0)
        mul_905: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_713, add_154);  view_713 = add_154 = None
        mul_906: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_18, tanh_18);  tanh_18 = None
        sub_197: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_906);  mul_906 = None
        mul_907: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_904, sub_197);  mul_904 = sub_197 = None
        mul_908: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_907, 0.7978845608028654);  mul_907 = None
        mul_909: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_908, 0.044715)
        pow_54: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_226, 2.0);  view_226 = None
        mul_910: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_54, 3.0);  pow_54 = None
        mul_911: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_909, mul_910);  mul_909 = mul_910 = None
        add_363: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_908, mul_911);  mul_908 = mul_911 = None
        mul_912: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_905, 0.5);  mul_905 = None
        add_364: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_363, mul_912);  add_363 = mul_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_714: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_364, [2048, 5120]);  add_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_355: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_229, [1, 0]);  primals_229 = None
        mm_141: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_714, permute_355);  permute_355 = None
        mm_142: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_356, view_714);  permute_356 = None
        sum_210: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_714, [0], True);  view_714 = None
        view_715: "f32[5120]" = torch.ops.aten.reshape.default(sum_210, [5120]);  sum_210 = None
        view_716: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_141, [4, 512, 1280]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_914: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_716, primals_226);  primals_226 = None
        mul_915: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_914, 1280)
        sum_211: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_914, [2], True)
        mul_916: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_914, mul_222);  mul_914 = None
        sum_212: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_916, [2], True);  mul_916 = None
        mul_917: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_222, sum_212);  sum_212 = None
        sub_199: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_915, sum_211);  mul_915 = sum_211 = None
        sub_200: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_199, mul_917);  sub_199 = mul_917 = None
        mul_918: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_35, sub_200);  div_35 = sub_200 = None
        mul_919: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_716, mul_222);  mul_222 = None
        sum_213: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_919, [0, 1]);  mul_919 = None
        sum_214: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_716, [0, 1]);  view_716 = None
        add_365: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_362, mul_918);  add_362 = mul_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_35: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_920: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_921: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_365, mul_920);  mul_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_717: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_921, [2048, 1280]);  mul_921 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_357: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_225, [1, 0]);  primals_225 = None
        mm_143: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_717, permute_357);  permute_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_203, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_222: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_75, [4, 512, -1]);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_223: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_222, [-1, 1280]);  view_222 = None
        permute_358: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_223, [1, 0]);  view_223 = None
        mm_144: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_358, view_717);  permute_358 = None
        sum_215: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_717, [0], True);  view_717 = None
        view_718: "f32[1280]" = torch.ops.aten.reshape.default(sum_215, [1280]);  sum_215 = None
        view_719: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_143, [4, 512, 1280]);  mm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_720: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_719, [4, 512, 20, 64]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_359: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_17 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_359, permute_74, permute_72, permute_73, expand_2, getitem_203, getitem_204, getitem_205, getitem_206, 0.1, [True, True, True, False], scale = 0.125);  permute_359 = permute_74 = permute_72 = permute_73 = getitem_203 = getitem_204 = getitem_205 = getitem_206 = None
        getitem_466: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_17[0]
        getitem_467: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_17[1]
        getitem_468: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_17[2];  _scaled_dot_product_efficient_attention_backward_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_360: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_466, [0, 2, 1, 3]);  getitem_466 = None
        view_721: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_360, [4, 512, 1280]);  permute_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_361: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_468, [0, 2, 1, 3]);  getitem_468 = None
        view_722: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_361, [4, 512, 1280]);  permute_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_362: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_467, [0, 2, 1, 3]);  getitem_467 = None
        view_723: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_362, [4, 512, 1280]);  permute_362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_18: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_721, view_723, view_722], 2);  view_721 = view_723 = view_722 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_724: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_18, [2048, 3840]);  cat_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_363: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_223, [1, 0]);  primals_223 = None
        mm_145: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_724, permute_363);  permute_363 = None
        mm_146: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_364, view_724);  permute_364 = None
        sum_216: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
        view_725: "f32[3840]" = torch.ops.aten.reshape.default(sum_216, [3840]);  sum_216 = None
        view_726: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_145, [4, 512, 1280]);  mm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_923: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_726, primals_220);  primals_220 = None
        mul_924: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_923, 1280)
        sum_217: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_923, [2], True)
        mul_925: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_923, mul_218);  mul_923 = None
        sum_218: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_925, [2], True);  mul_925 = None
        mul_926: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_218, sum_218);  sum_218 = None
        sub_202: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_924, sum_217);  mul_924 = sum_217 = None
        sub_203: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_202, mul_926);  sub_202 = mul_926 = None
        mul_927: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_36, sub_203);  div_36 = sub_203 = None
        mul_928: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_726, mul_218);  mul_218 = None
        sum_219: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_928, [0, 1]);  mul_928 = None
        sum_220: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_726, [0, 1]);  view_726 = None
        add_366: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_365, mul_927);  add_365 = mul_927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_36: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_929: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_930: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_366, mul_929);  mul_929 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_727: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_930, [2048, 1280]);  mul_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_365: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_219, [1, 0]);  primals_219 = None
        mm_147: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_727, permute_365);  permute_365 = None
        mm_148: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_366, view_727);  permute_366 = None
        sum_221: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
        view_728: "f32[1280]" = torch.ops.aten.reshape.default(sum_221, [1280]);  sum_221 = None
        view_729: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_147, [4, 512, 5120]);  mm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_214: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_70, [4, 512, 5120]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_212: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_214, 0.5)
        mul_931: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_729, mul_212);  mul_212 = None
        pow_18: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 3.0)
        mul_213: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_145: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_214, mul_213);  mul_213 = None
        mul_214: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_145, 0.7978845608028654);  add_145 = None
        tanh_17: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_214);  mul_214 = None
        add_146: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_17, 1.0)
        mul_932: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_729, add_146);  view_729 = add_146 = None
        mul_933: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_17, tanh_17);  tanh_17 = None
        sub_204: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_933);  mul_933 = None
        mul_934: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_931, sub_204);  mul_931 = sub_204 = None
        mul_935: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_934, 0.7978845608028654);  mul_934 = None
        mul_936: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_935, 0.044715)
        pow_55: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_214, 2.0);  view_214 = None
        mul_937: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_55, 3.0);  pow_55 = None
        mul_938: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_936, mul_937);  mul_936 = mul_937 = None
        add_367: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_935, mul_938);  mul_935 = mul_938 = None
        mul_939: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_932, 0.5);  mul_932 = None
        add_368: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_367, mul_939);  add_367 = mul_939 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_730: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_368, [2048, 5120]);  add_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_367: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_217, [1, 0]);  primals_217 = None
        mm_149: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_730, permute_367);  permute_367 = None
        mm_150: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_368, view_730);  permute_368 = None
        sum_222: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_730, [0], True);  view_730 = None
        view_731: "f32[5120]" = torch.ops.aten.reshape.default(sum_222, [5120]);  sum_222 = None
        view_732: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_149, [4, 512, 1280]);  mm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_941: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_732, primals_214);  primals_214 = None
        mul_942: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_941, 1280)
        sum_223: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_941, [2], True)
        mul_943: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_941, mul_210);  mul_941 = None
        sum_224: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_943, [2], True);  mul_943 = None
        mul_944: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_210, sum_224);  sum_224 = None
        sub_206: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_942, sum_223);  mul_942 = sum_223 = None
        sub_207: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_206, mul_944);  sub_206 = mul_944 = None
        mul_945: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_37, sub_207);  div_37 = sub_207 = None
        mul_946: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_732, mul_210);  mul_210 = None
        sum_225: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_946, [0, 1]);  mul_946 = None
        sum_226: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_732, [0, 1]);  view_732 = None
        add_369: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_366, mul_945);  add_366 = mul_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_37: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_947: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_948: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_369, mul_947);  mul_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_733: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_948, [2048, 1280]);  mul_948 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_369: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_213, [1, 0]);  primals_213 = None
        mm_151: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_733, permute_369);  permute_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_71: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_210: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_71, [4, 512, -1]);  permute_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_211: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_210, [-1, 1280]);  view_210 = None
        permute_370: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_211, [1, 0]);  view_211 = None
        mm_152: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_370, view_733);  permute_370 = None
        sum_227: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_733, [0], True);  view_733 = None
        view_734: "f32[1280]" = torch.ops.aten.reshape.default(sum_227, [1280]);  sum_227 = None
        view_735: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_151, [4, 512, 1280]);  mm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_736: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_735, [4, 512, 20, 64]);  view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_371: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_736, [0, 2, 1, 3]);  view_736 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_18 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_371, permute_70, permute_68, permute_69, expand_2, getitem_192, getitem_193, getitem_194, getitem_195, 0.1, [True, True, True, False], scale = 0.125);  permute_371 = permute_70 = permute_68 = permute_69 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = None
        getitem_470: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_18[0]
        getitem_471: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_18[1]
        getitem_472: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_18[2];  _scaled_dot_product_efficient_attention_backward_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_372: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_470, [0, 2, 1, 3]);  getitem_470 = None
        view_737: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_372, [4, 512, 1280]);  permute_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_373: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_472, [0, 2, 1, 3]);  getitem_472 = None
        view_738: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_373, [4, 512, 1280]);  permute_373 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_374: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_471, [0, 2, 1, 3]);  getitem_471 = None
        view_739: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_374, [4, 512, 1280]);  permute_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_19: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_737, view_739, view_738], 2);  view_737 = view_739 = view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_740: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_19, [2048, 3840]);  cat_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_375: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_211, [1, 0]);  primals_211 = None
        mm_153: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_740, permute_375);  permute_375 = None
        mm_154: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_376, view_740);  permute_376 = None
        sum_228: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        view_741: "f32[3840]" = torch.ops.aten.reshape.default(sum_228, [3840]);  sum_228 = None
        view_742: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_153, [4, 512, 1280]);  mm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_950: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_742, primals_208);  primals_208 = None
        mul_951: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_950, 1280)
        sum_229: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_950, [2], True)
        mul_952: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_950, mul_206);  mul_950 = None
        sum_230: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_952, [2], True);  mul_952 = None
        mul_953: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_206, sum_230);  sum_230 = None
        sub_209: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_951, sum_229);  mul_951 = sum_229 = None
        sub_210: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_209, mul_953);  sub_209 = mul_953 = None
        mul_954: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_38, sub_210);  div_38 = sub_210 = None
        mul_955: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_742, mul_206);  mul_206 = None
        sum_231: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_955, [0, 1]);  mul_955 = None
        sum_232: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_742, [0, 1]);  view_742 = None
        add_370: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_369, mul_954);  add_369 = mul_954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_956: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_957: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_370, mul_956);  mul_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_743: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_957, [2048, 1280]);  mul_957 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_377: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_207, [1, 0]);  primals_207 = None
        mm_155: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_743, permute_377);  permute_377 = None
        mm_156: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_378, view_743);  permute_378 = None
        sum_233: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        view_744: "f32[1280]" = torch.ops.aten.reshape.default(sum_233, [1280]);  sum_233 = None
        view_745: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_155, [4, 512, 5120]);  mm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_202: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_66, [4, 512, 5120]);  addmm_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_200: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_202, 0.5)
        mul_958: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_745, mul_200);  mul_200 = None
        pow_17: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_202, 3.0)
        mul_201: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_17, 0.044715);  pow_17 = None
        add_137: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_202, mul_201);  mul_201 = None
        mul_202: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_137, 0.7978845608028654);  add_137 = None
        tanh_16: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_202);  mul_202 = None
        add_138: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_16, 1.0)
        mul_959: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_745, add_138);  view_745 = add_138 = None
        mul_960: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_16, tanh_16);  tanh_16 = None
        sub_211: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_960);  mul_960 = None
        mul_961: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_958, sub_211);  mul_958 = sub_211 = None
        mul_962: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_961, 0.7978845608028654);  mul_961 = None
        mul_963: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_962, 0.044715)
        pow_56: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_202, 2.0);  view_202 = None
        mul_964: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_56, 3.0);  pow_56 = None
        mul_965: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_963, mul_964);  mul_963 = mul_964 = None
        add_371: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_962, mul_965);  mul_962 = mul_965 = None
        mul_966: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_959, 0.5);  mul_959 = None
        add_372: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_371, mul_966);  add_371 = mul_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_746: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_372, [2048, 5120]);  add_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_379: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_205, [1, 0]);  primals_205 = None
        mm_157: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_746, permute_379);  permute_379 = None
        mm_158: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_380, view_746);  permute_380 = None
        sum_234: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_746, [0], True);  view_746 = None
        view_747: "f32[5120]" = torch.ops.aten.reshape.default(sum_234, [5120]);  sum_234 = None
        view_748: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_157, [4, 512, 1280]);  mm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_968: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_748, primals_202);  primals_202 = None
        mul_969: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_968, 1280)
        sum_235: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_968, [2], True)
        mul_970: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_968, mul_198);  mul_968 = None
        sum_236: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_970, [2], True);  mul_970 = None
        mul_971: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_198, sum_236);  sum_236 = None
        sub_213: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_969, sum_235);  mul_969 = sum_235 = None
        sub_214: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_213, mul_971);  sub_213 = mul_971 = None
        mul_972: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_39, sub_214);  div_39 = sub_214 = None
        mul_973: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_748, mul_198);  mul_198 = None
        sum_237: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_973, [0, 1]);  mul_973 = None
        sum_238: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_748, [0, 1]);  view_748 = None
        add_373: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_370, mul_972);  add_370 = mul_972 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_39: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_974: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_975: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_373, mul_974);  mul_974 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_749: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_975, [2048, 1280]);  mul_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_381: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_201, [1, 0]);  primals_201 = None
        mm_159: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_749, permute_381);  permute_381 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_67: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_181, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_198: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_67, [4, 512, -1]);  permute_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_199: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_198, [-1, 1280]);  view_198 = None
        permute_382: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_199, [1, 0]);  view_199 = None
        mm_160: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_382, view_749);  permute_382 = None
        sum_239: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
        view_750: "f32[1280]" = torch.ops.aten.reshape.default(sum_239, [1280]);  sum_239 = None
        view_751: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_159, [4, 512, 1280]);  mm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_752: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_751, [4, 512, 20, 64]);  view_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_383: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_752, [0, 2, 1, 3]);  view_752 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_19 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_383, permute_66, permute_64, permute_65, expand_2, getitem_181, getitem_182, getitem_183, getitem_184, 0.1, [True, True, True, False], scale = 0.125);  permute_383 = permute_66 = permute_64 = permute_65 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = None
        getitem_474: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_19[0]
        getitem_475: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_19[1]
        getitem_476: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_19[2];  _scaled_dot_product_efficient_attention_backward_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_384: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_474, [0, 2, 1, 3]);  getitem_474 = None
        view_753: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_384, [4, 512, 1280]);  permute_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_385: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_476, [0, 2, 1, 3]);  getitem_476 = None
        view_754: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_385, [4, 512, 1280]);  permute_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_386: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_475, [0, 2, 1, 3]);  getitem_475 = None
        view_755: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_386, [4, 512, 1280]);  permute_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_20: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_753, view_755, view_754], 2);  view_753 = view_755 = view_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_756: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_20, [2048, 3840]);  cat_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_387: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_199, [1, 0]);  primals_199 = None
        mm_161: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_756, permute_387);  permute_387 = None
        mm_162: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_388, view_756);  permute_388 = None
        sum_240: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_756, [0], True);  view_756 = None
        view_757: "f32[3840]" = torch.ops.aten.reshape.default(sum_240, [3840]);  sum_240 = None
        view_758: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_161, [4, 512, 1280]);  mm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_977: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_758, primals_196);  primals_196 = None
        mul_978: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_977, 1280)
        sum_241: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_977, [2], True)
        mul_979: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_977, mul_194);  mul_977 = None
        sum_242: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_979, [2], True);  mul_979 = None
        mul_980: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_194, sum_242);  sum_242 = None
        sub_216: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_978, sum_241);  mul_978 = sum_241 = None
        sub_217: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_216, mul_980);  sub_216 = mul_980 = None
        mul_981: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_40, sub_217);  div_40 = sub_217 = None
        mul_982: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_758, mul_194);  mul_194 = None
        sum_243: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_982, [0, 1]);  mul_982 = None
        sum_244: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_758, [0, 1]);  view_758 = None
        add_374: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_373, mul_981);  add_373 = mul_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_40: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_983: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_984: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_374, mul_983);  mul_983 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_759: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_984, [2048, 1280]);  mul_984 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_389: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_195, [1, 0]);  primals_195 = None
        mm_163: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_759, permute_389);  permute_389 = None
        mm_164: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_390, view_759);  permute_390 = None
        sum_245: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
        view_760: "f32[1280]" = torch.ops.aten.reshape.default(sum_245, [1280]);  sum_245 = None
        view_761: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_163, [4, 512, 5120]);  mm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_190: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_62, [4, 512, 5120]);  addmm_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_188: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_190, 0.5)
        mul_985: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_761, mul_188);  mul_188 = None
        pow_16: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_190, 3.0)
        mul_189: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_16, 0.044715);  pow_16 = None
        add_129: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_190, mul_189);  mul_189 = None
        mul_190: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_129, 0.7978845608028654);  add_129 = None
        tanh_15: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_190);  mul_190 = None
        add_130: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_986: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_761, add_130);  view_761 = add_130 = None
        mul_987: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_218: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_987);  mul_987 = None
        mul_988: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_985, sub_218);  mul_985 = sub_218 = None
        mul_989: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_988, 0.7978845608028654);  mul_988 = None
        mul_990: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_989, 0.044715)
        pow_57: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_190, 2.0);  view_190 = None
        mul_991: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_57, 3.0);  pow_57 = None
        mul_992: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_990, mul_991);  mul_990 = mul_991 = None
        add_375: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_989, mul_992);  mul_989 = mul_992 = None
        mul_993: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_986, 0.5);  mul_986 = None
        add_376: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_375, mul_993);  add_375 = mul_993 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_762: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_376, [2048, 5120]);  add_376 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_391: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_193, [1, 0]);  primals_193 = None
        mm_165: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_762, permute_391);  permute_391 = None
        mm_166: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_392, view_762);  permute_392 = None
        sum_246: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        view_763: "f32[5120]" = torch.ops.aten.reshape.default(sum_246, [5120]);  sum_246 = None
        view_764: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_165, [4, 512, 1280]);  mm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_995: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_764, primals_190);  primals_190 = None
        mul_996: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_995, 1280)
        sum_247: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_995, [2], True)
        mul_997: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_995, mul_186);  mul_995 = None
        sum_248: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_997, [2], True);  mul_997 = None
        mul_998: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_186, sum_248);  sum_248 = None
        sub_220: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_996, sum_247);  mul_996 = sum_247 = None
        sub_221: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_220, mul_998);  sub_220 = mul_998 = None
        mul_999: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_41, sub_221);  div_41 = sub_221 = None
        mul_1000: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_764, mul_186);  mul_186 = None
        sum_249: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1000, [0, 1]);  mul_1000 = None
        sum_250: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_764, [0, 1]);  view_764 = None
        add_377: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_374, mul_999);  add_374 = mul_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_41: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_1001: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_1002: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_377, mul_1001);  mul_1001 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_765: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1002, [2048, 1280]);  mul_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_393: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        mm_167: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_765, permute_393);  permute_393 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_63: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_170, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_186: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_63, [4, 512, -1]);  permute_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_187: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_186, [-1, 1280]);  view_186 = None
        permute_394: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_187, [1, 0]);  view_187 = None
        mm_168: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_394, view_765);  permute_394 = None
        sum_251: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        view_766: "f32[1280]" = torch.ops.aten.reshape.default(sum_251, [1280]);  sum_251 = None
        view_767: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_167, [4, 512, 1280]);  mm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_768: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_767, [4, 512, 20, 64]);  view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_395: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_768, [0, 2, 1, 3]);  view_768 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_20 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_395, permute_62, permute_60, permute_61, expand_2, getitem_170, getitem_171, getitem_172, getitem_173, 0.1, [True, True, True, False], scale = 0.125);  permute_395 = permute_62 = permute_60 = permute_61 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = None
        getitem_478: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_20[0]
        getitem_479: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_20[1]
        getitem_480: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_20[2];  _scaled_dot_product_efficient_attention_backward_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_396: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_478, [0, 2, 1, 3]);  getitem_478 = None
        view_769: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_396, [4, 512, 1280]);  permute_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_397: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_480, [0, 2, 1, 3]);  getitem_480 = None
        view_770: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_397, [4, 512, 1280]);  permute_397 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_398: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_479, [0, 2, 1, 3]);  getitem_479 = None
        view_771: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_398, [4, 512, 1280]);  permute_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_21: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_769, view_771, view_770], 2);  view_769 = view_771 = view_770 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_772: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_21, [2048, 3840]);  cat_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_399: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_187, [1, 0]);  primals_187 = None
        mm_169: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_772, permute_399);  permute_399 = None
        mm_170: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_400, view_772);  permute_400 = None
        sum_252: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_772, [0], True);  view_772 = None
        view_773: "f32[3840]" = torch.ops.aten.reshape.default(sum_252, [3840]);  sum_252 = None
        view_774: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_169, [4, 512, 1280]);  mm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1004: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_774, primals_184);  primals_184 = None
        mul_1005: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1004, 1280)
        sum_253: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1004, [2], True)
        mul_1006: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1004, mul_182);  mul_1004 = None
        sum_254: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1006, [2], True);  mul_1006 = None
        mul_1007: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_182, sum_254);  sum_254 = None
        sub_223: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1005, sum_253);  mul_1005 = sum_253 = None
        sub_224: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_223, mul_1007);  sub_223 = mul_1007 = None
        mul_1008: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_42, sub_224);  div_42 = sub_224 = None
        mul_1009: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_774, mul_182);  mul_182 = None
        sum_255: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1009, [0, 1]);  mul_1009 = None
        sum_256: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_774, [0, 1]);  view_774 = None
        add_378: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_377, mul_1008);  add_377 = mul_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_42: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_1010: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_1011: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_378, mul_1010);  mul_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_775: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1011, [2048, 1280]);  mul_1011 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_401: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        mm_171: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_775, permute_401);  permute_401 = None
        mm_172: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_402, view_775);  permute_402 = None
        sum_257: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_775, [0], True);  view_775 = None
        view_776: "f32[1280]" = torch.ops.aten.reshape.default(sum_257, [1280]);  sum_257 = None
        view_777: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_171, [4, 512, 5120]);  mm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_178: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_58, [4, 512, 5120]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_176: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_178, 0.5)
        mul_1012: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_777, mul_176);  mul_176 = None
        pow_15: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 3.0)
        mul_177: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_121: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_178, mul_177);  mul_177 = None
        mul_178: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_121, 0.7978845608028654);  add_121 = None
        tanh_14: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_178);  mul_178 = None
        add_122: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_1013: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_777, add_122);  view_777 = add_122 = None
        mul_1014: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_225: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1014);  mul_1014 = None
        mul_1015: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1012, sub_225);  mul_1012 = sub_225 = None
        mul_1016: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1015, 0.7978845608028654);  mul_1015 = None
        mul_1017: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1016, 0.044715)
        pow_58: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_178, 2.0);  view_178 = None
        mul_1018: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_58, 3.0);  pow_58 = None
        mul_1019: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1017, mul_1018);  mul_1017 = mul_1018 = None
        add_379: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1016, mul_1019);  mul_1016 = mul_1019 = None
        mul_1020: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1013, 0.5);  mul_1013 = None
        add_380: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_379, mul_1020);  add_379 = mul_1020 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_778: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_380, [2048, 5120]);  add_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_403: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        mm_173: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_778, permute_403);  permute_403 = None
        mm_174: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_404, view_778);  permute_404 = None
        sum_258: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_778, [0], True);  view_778 = None
        view_779: "f32[5120]" = torch.ops.aten.reshape.default(sum_258, [5120]);  sum_258 = None
        view_780: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_173, [4, 512, 1280]);  mm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1022: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_780, primals_178);  primals_178 = None
        mul_1023: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1022, 1280)
        sum_259: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1022, [2], True)
        mul_1024: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1022, mul_174);  mul_1022 = None
        sum_260: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1024, [2], True);  mul_1024 = None
        mul_1025: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_174, sum_260);  sum_260 = None
        sub_227: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1023, sum_259);  mul_1023 = sum_259 = None
        sub_228: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_227, mul_1025);  sub_227 = mul_1025 = None
        mul_1026: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_43, sub_228);  div_43 = sub_228 = None
        mul_1027: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_780, mul_174);  mul_174 = None
        sum_261: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1027, [0, 1]);  mul_1027 = None
        sum_262: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_780, [0, 1]);  view_780 = None
        add_381: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_378, mul_1026);  add_378 = mul_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_43: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_1028: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_1029: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_381, mul_1028);  mul_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_781: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1029, [2048, 1280]);  mul_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_405: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_177, [1, 0]);  primals_177 = None
        mm_175: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_781, permute_405);  permute_405 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_59: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_159, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_174: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_59, [4, 512, -1]);  permute_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_175: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_174, [-1, 1280]);  view_174 = None
        permute_406: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_175, [1, 0]);  view_175 = None
        mm_176: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_406, view_781);  permute_406 = None
        sum_263: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_781, [0], True);  view_781 = None
        view_782: "f32[1280]" = torch.ops.aten.reshape.default(sum_263, [1280]);  sum_263 = None
        view_783: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_175, [4, 512, 1280]);  mm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_784: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_783, [4, 512, 20, 64]);  view_783 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_407: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_21 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_407, permute_58, permute_56, permute_57, expand_2, getitem_159, getitem_160, getitem_161, getitem_162, 0.1, [True, True, True, False], scale = 0.125);  permute_407 = permute_58 = permute_56 = permute_57 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = None
        getitem_482: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_21[0]
        getitem_483: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_21[1]
        getitem_484: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_21[2];  _scaled_dot_product_efficient_attention_backward_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_408: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_482, [0, 2, 1, 3]);  getitem_482 = None
        view_785: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_408, [4, 512, 1280]);  permute_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_409: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_484, [0, 2, 1, 3]);  getitem_484 = None
        view_786: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_409, [4, 512, 1280]);  permute_409 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_410: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_483, [0, 2, 1, 3]);  getitem_483 = None
        view_787: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_410, [4, 512, 1280]);  permute_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_22: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_785, view_787, view_786], 2);  view_785 = view_787 = view_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_788: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_22, [2048, 3840]);  cat_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_411: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        mm_177: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_788, permute_411);  permute_411 = None
        mm_178: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_412, view_788);  permute_412 = None
        sum_264: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_788, [0], True);  view_788 = None
        view_789: "f32[3840]" = torch.ops.aten.reshape.default(sum_264, [3840]);  sum_264 = None
        view_790: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_177, [4, 512, 1280]);  mm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1031: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_790, primals_172);  primals_172 = None
        mul_1032: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1031, 1280)
        sum_265: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1031, [2], True)
        mul_1033: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1031, mul_170);  mul_1031 = None
        sum_266: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1033, [2], True);  mul_1033 = None
        mul_1034: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_170, sum_266);  sum_266 = None
        sub_230: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1032, sum_265);  mul_1032 = sum_265 = None
        sub_231: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_230, mul_1034);  sub_230 = mul_1034 = None
        mul_1035: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_44, sub_231);  div_44 = sub_231 = None
        mul_1036: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_790, mul_170);  mul_170 = None
        sum_267: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1036, [0, 1]);  mul_1036 = None
        sum_268: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_790, [0, 1]);  view_790 = None
        add_382: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_381, mul_1035);  add_381 = mul_1035 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_1037: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_1038: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_382, mul_1037);  mul_1037 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_791: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1038, [2048, 1280]);  mul_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_413: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        mm_179: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_791, permute_413);  permute_413 = None
        mm_180: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_414, view_791);  permute_414 = None
        sum_269: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_791, [0], True);  view_791 = None
        view_792: "f32[1280]" = torch.ops.aten.reshape.default(sum_269, [1280]);  sum_269 = None
        view_793: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_179, [4, 512, 5120]);  mm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_166: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_54, [4, 512, 5120]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_164: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_166, 0.5)
        mul_1039: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_793, mul_164);  mul_164 = None
        pow_14: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_166, 3.0)
        mul_165: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_14, 0.044715);  pow_14 = None
        add_113: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_166, mul_165);  mul_165 = None
        mul_166: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_113, 0.7978845608028654);  add_113 = None
        tanh_13: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_166);  mul_166 = None
        add_114: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_1040: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_793, add_114);  view_793 = add_114 = None
        mul_1041: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_232: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1041);  mul_1041 = None
        mul_1042: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1039, sub_232);  mul_1039 = sub_232 = None
        mul_1043: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1042, 0.7978845608028654);  mul_1042 = None
        mul_1044: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1043, 0.044715)
        pow_59: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_166, 2.0);  view_166 = None
        mul_1045: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_59, 3.0);  pow_59 = None
        mul_1046: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1044, mul_1045);  mul_1044 = mul_1045 = None
        add_383: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1043, mul_1046);  mul_1043 = mul_1046 = None
        mul_1047: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1040, 0.5);  mul_1040 = None
        add_384: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_383, mul_1047);  add_383 = mul_1047 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_794: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_384, [2048, 5120]);  add_384 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_415: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        mm_181: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_794, permute_415);  permute_415 = None
        mm_182: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_416, view_794);  permute_416 = None
        sum_270: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_794, [0], True);  view_794 = None
        view_795: "f32[5120]" = torch.ops.aten.reshape.default(sum_270, [5120]);  sum_270 = None
        view_796: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_181, [4, 512, 1280]);  mm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1049: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_796, primals_166);  primals_166 = None
        mul_1050: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1049, 1280)
        sum_271: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1049, [2], True)
        mul_1051: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1049, mul_162);  mul_1049 = None
        sum_272: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1051, [2], True);  mul_1051 = None
        mul_1052: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_162, sum_272);  sum_272 = None
        sub_234: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1050, sum_271);  mul_1050 = sum_271 = None
        sub_235: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_234, mul_1052);  sub_234 = mul_1052 = None
        mul_1053: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_45, sub_235);  div_45 = sub_235 = None
        mul_1054: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_796, mul_162);  mul_162 = None
        sum_273: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1054, [0, 1]);  mul_1054 = None
        sum_274: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_796, [0, 1]);  view_796 = None
        add_385: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_382, mul_1053);  add_382 = mul_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_45: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_1055: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_1056: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_385, mul_1055);  mul_1055 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_797: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1056, [2048, 1280]);  mul_1056 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_417: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        mm_183: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_797, permute_417);  permute_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_55: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_148, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_162: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_55, [4, 512, -1]);  permute_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_163: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_162, [-1, 1280]);  view_162 = None
        permute_418: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_163, [1, 0]);  view_163 = None
        mm_184: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_418, view_797);  permute_418 = None
        sum_275: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_797, [0], True);  view_797 = None
        view_798: "f32[1280]" = torch.ops.aten.reshape.default(sum_275, [1280]);  sum_275 = None
        view_799: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_183, [4, 512, 1280]);  mm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_800: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_799, [4, 512, 20, 64]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_419: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_800, [0, 2, 1, 3]);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_22 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_419, permute_54, permute_52, permute_53, expand_2, getitem_148, getitem_149, getitem_150, getitem_151, 0.1, [True, True, True, False], scale = 0.125);  permute_419 = permute_54 = permute_52 = permute_53 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = None
        getitem_486: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_22[0]
        getitem_487: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_22[1]
        getitem_488: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_22[2];  _scaled_dot_product_efficient_attention_backward_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_420: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_486, [0, 2, 1, 3]);  getitem_486 = None
        view_801: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_420, [4, 512, 1280]);  permute_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_421: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_488, [0, 2, 1, 3]);  getitem_488 = None
        view_802: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_421, [4, 512, 1280]);  permute_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_422: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_487, [0, 2, 1, 3]);  getitem_487 = None
        view_803: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_422, [4, 512, 1280]);  permute_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_23: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_801, view_803, view_802], 2);  view_801 = view_803 = view_802 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_804: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_23, [2048, 3840]);  cat_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_423: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_163, [1, 0]);  primals_163 = None
        mm_185: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_804, permute_423);  permute_423 = None
        mm_186: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_424, view_804);  permute_424 = None
        sum_276: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_804, [0], True);  view_804 = None
        view_805: "f32[3840]" = torch.ops.aten.reshape.default(sum_276, [3840]);  sum_276 = None
        view_806: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_185, [4, 512, 1280]);  mm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1058: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_806, primals_160);  primals_160 = None
        mul_1059: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1058, 1280)
        sum_277: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1058, [2], True)
        mul_1060: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1058, mul_158);  mul_1058 = None
        sum_278: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1060, [2], True);  mul_1060 = None
        mul_1061: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_158, sum_278);  sum_278 = None
        sub_237: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1059, sum_277);  mul_1059 = sum_277 = None
        sub_238: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_237, mul_1061);  sub_237 = mul_1061 = None
        mul_1062: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_46, sub_238);  div_46 = sub_238 = None
        mul_1063: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_806, mul_158);  mul_158 = None
        sum_279: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1063, [0, 1]);  mul_1063 = None
        sum_280: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_806, [0, 1]);  view_806 = None
        add_386: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_385, mul_1062);  add_385 = mul_1062 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_46: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_1064: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_1065: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_386, mul_1064);  mul_1064 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_807: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1065, [2048, 1280]);  mul_1065 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_425: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_159, [1, 0]);  primals_159 = None
        mm_187: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_807, permute_425);  permute_425 = None
        mm_188: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_426, view_807);  permute_426 = None
        sum_281: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_807, [0], True);  view_807 = None
        view_808: "f32[1280]" = torch.ops.aten.reshape.default(sum_281, [1280]);  sum_281 = None
        view_809: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_187, [4, 512, 5120]);  mm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_154: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_50, [4, 512, 5120]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_152: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_154, 0.5)
        mul_1066: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_809, mul_152);  mul_152 = None
        pow_13: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_154, 3.0)
        mul_153: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_13, 0.044715);  pow_13 = None
        add_105: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_154, mul_153);  mul_153 = None
        mul_154: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_105, 0.7978845608028654);  add_105 = None
        tanh_12: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_154);  mul_154 = None
        add_106: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_1067: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_809, add_106);  view_809 = add_106 = None
        mul_1068: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_239: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1068);  mul_1068 = None
        mul_1069: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1066, sub_239);  mul_1066 = sub_239 = None
        mul_1070: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1069, 0.7978845608028654);  mul_1069 = None
        mul_1071: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1070, 0.044715)
        pow_60: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_154, 2.0);  view_154 = None
        mul_1072: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_60, 3.0);  pow_60 = None
        mul_1073: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1071, mul_1072);  mul_1071 = mul_1072 = None
        add_387: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1070, mul_1073);  mul_1070 = mul_1073 = None
        mul_1074: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1067, 0.5);  mul_1067 = None
        add_388: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_387, mul_1074);  add_387 = mul_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_810: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_388, [2048, 5120]);  add_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_427: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        mm_189: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_810, permute_427);  permute_427 = None
        mm_190: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_428, view_810);  permute_428 = None
        sum_282: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_810, [0], True);  view_810 = None
        view_811: "f32[5120]" = torch.ops.aten.reshape.default(sum_282, [5120]);  sum_282 = None
        view_812: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_189, [4, 512, 1280]);  mm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1076: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_812, primals_154);  primals_154 = None
        mul_1077: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1076, 1280)
        sum_283: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1076, [2], True)
        mul_1078: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1076, mul_150);  mul_1076 = None
        sum_284: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1078, [2], True);  mul_1078 = None
        mul_1079: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_150, sum_284);  sum_284 = None
        sub_241: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1077, sum_283);  mul_1077 = sum_283 = None
        sub_242: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_241, mul_1079);  sub_241 = mul_1079 = None
        mul_1080: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_47, sub_242);  div_47 = sub_242 = None
        mul_1081: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_812, mul_150);  mul_150 = None
        sum_285: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1081, [0, 1]);  mul_1081 = None
        sum_286: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_812, [0, 1]);  view_812 = None
        add_389: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_386, mul_1080);  add_386 = mul_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_47: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_1082: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_1083: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_389, mul_1082);  mul_1082 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_813: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1083, [2048, 1280]);  mul_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_429: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        mm_191: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_813, permute_429);  permute_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_51: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_137, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_150: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_51, [4, 512, -1]);  permute_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_151: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_150, [-1, 1280]);  view_150 = None
        permute_430: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_151, [1, 0]);  view_151 = None
        mm_192: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_430, view_813);  permute_430 = None
        sum_287: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
        view_814: "f32[1280]" = torch.ops.aten.reshape.default(sum_287, [1280]);  sum_287 = None
        view_815: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_191, [4, 512, 1280]);  mm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_816: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_815, [4, 512, 20, 64]);  view_815 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_431: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_23 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_431, permute_50, permute_48, permute_49, expand_2, getitem_137, getitem_138, getitem_139, getitem_140, 0.1, [True, True, True, False], scale = 0.125);  permute_431 = permute_50 = permute_48 = permute_49 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = None
        getitem_490: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_23[0]
        getitem_491: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_23[1]
        getitem_492: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_23[2];  _scaled_dot_product_efficient_attention_backward_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_432: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_490, [0, 2, 1, 3]);  getitem_490 = None
        view_817: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_432, [4, 512, 1280]);  permute_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_433: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_492, [0, 2, 1, 3]);  getitem_492 = None
        view_818: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_433, [4, 512, 1280]);  permute_433 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_434: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_491, [0, 2, 1, 3]);  getitem_491 = None
        view_819: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_434, [4, 512, 1280]);  permute_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_24: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_817, view_819, view_818], 2);  view_817 = view_819 = view_818 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_820: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_24, [2048, 3840]);  cat_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_435: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        mm_193: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_820, permute_435);  permute_435 = None
        mm_194: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_436, view_820);  permute_436 = None
        sum_288: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_820, [0], True);  view_820 = None
        view_821: "f32[3840]" = torch.ops.aten.reshape.default(sum_288, [3840]);  sum_288 = None
        view_822: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_193, [4, 512, 1280]);  mm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1085: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_822, primals_148);  primals_148 = None
        mul_1086: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1085, 1280)
        sum_289: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1085, [2], True)
        mul_1087: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1085, mul_146);  mul_1085 = None
        sum_290: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1087, [2], True);  mul_1087 = None
        mul_1088: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_146, sum_290);  sum_290 = None
        sub_244: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1086, sum_289);  mul_1086 = sum_289 = None
        sub_245: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_244, mul_1088);  sub_244 = mul_1088 = None
        mul_1089: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_48, sub_245);  div_48 = sub_245 = None
        mul_1090: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_822, mul_146);  mul_146 = None
        sum_291: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1090, [0, 1]);  mul_1090 = None
        sum_292: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_822, [0, 1]);  view_822 = None
        add_390: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_389, mul_1089);  add_389 = mul_1089 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_48: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_1091: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_1092: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_390, mul_1091);  mul_1091 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_823: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1092, [2048, 1280]);  mul_1092 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_437: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        mm_195: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_823, permute_437);  permute_437 = None
        mm_196: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_438, view_823);  permute_438 = None
        sum_293: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_823, [0], True);  view_823 = None
        view_824: "f32[1280]" = torch.ops.aten.reshape.default(sum_293, [1280]);  sum_293 = None
        view_825: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_195, [4, 512, 5120]);  mm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_142: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_46, [4, 512, 5120]);  addmm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_140: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_142, 0.5)
        mul_1093: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_825, mul_140);  mul_140 = None
        pow_12: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 3.0)
        mul_141: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_97: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_142, mul_141);  mul_141 = None
        mul_142: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_97, 0.7978845608028654);  add_97 = None
        tanh_11: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_142);  mul_142 = None
        add_98: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_1094: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_825, add_98);  view_825 = add_98 = None
        mul_1095: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_246: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1095);  mul_1095 = None
        mul_1096: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1093, sub_246);  mul_1093 = sub_246 = None
        mul_1097: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1096, 0.7978845608028654);  mul_1096 = None
        mul_1098: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1097, 0.044715)
        pow_61: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_142, 2.0);  view_142 = None
        mul_1099: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_61, 3.0);  pow_61 = None
        mul_1100: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1098, mul_1099);  mul_1098 = mul_1099 = None
        add_391: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1097, mul_1100);  mul_1097 = mul_1100 = None
        mul_1101: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1094, 0.5);  mul_1094 = None
        add_392: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_391, mul_1101);  add_391 = mul_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_826: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_392, [2048, 5120]);  add_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_439: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_145, [1, 0]);  primals_145 = None
        mm_197: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_826, permute_439);  permute_439 = None
        mm_198: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_440, view_826);  permute_440 = None
        sum_294: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_826, [0], True);  view_826 = None
        view_827: "f32[5120]" = torch.ops.aten.reshape.default(sum_294, [5120]);  sum_294 = None
        view_828: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_197, [4, 512, 1280]);  mm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1103: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_828, primals_142);  primals_142 = None
        mul_1104: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1103, 1280)
        sum_295: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1103, [2], True)
        mul_1105: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1103, mul_138);  mul_1103 = None
        sum_296: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1105, [2], True);  mul_1105 = None
        mul_1106: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_138, sum_296);  sum_296 = None
        sub_248: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1104, sum_295);  mul_1104 = sum_295 = None
        sub_249: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_248, mul_1106);  sub_248 = mul_1106 = None
        mul_1107: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_49, sub_249);  div_49 = sub_249 = None
        mul_1108: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_828, mul_138);  mul_138 = None
        sum_297: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1108, [0, 1]);  mul_1108 = None
        sum_298: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_828, [0, 1]);  view_828 = None
        add_393: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_390, mul_1107);  add_390 = mul_1107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_49: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_1109: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_1110: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_393, mul_1109);  mul_1109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_829: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1110, [2048, 1280]);  mul_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_441: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        mm_199: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_829, permute_441);  permute_441 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_126, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_138: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_47, [4, 512, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_139: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_138, [-1, 1280]);  view_138 = None
        permute_442: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_139, [1, 0]);  view_139 = None
        mm_200: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_442, view_829);  permute_442 = None
        sum_299: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_829, [0], True);  view_829 = None
        view_830: "f32[1280]" = torch.ops.aten.reshape.default(sum_299, [1280]);  sum_299 = None
        view_831: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_199, [4, 512, 1280]);  mm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_832: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_831, [4, 512, 20, 64]);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_443: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_832, [0, 2, 1, 3]);  view_832 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_24 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_443, permute_46, permute_44, permute_45, expand_2, getitem_126, getitem_127, getitem_128, getitem_129, 0.1, [True, True, True, False], scale = 0.125);  permute_443 = permute_46 = permute_44 = permute_45 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = None
        getitem_494: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_24[0]
        getitem_495: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_24[1]
        getitem_496: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_24[2];  _scaled_dot_product_efficient_attention_backward_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_444: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_494, [0, 2, 1, 3]);  getitem_494 = None
        view_833: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_444, [4, 512, 1280]);  permute_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_445: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_496, [0, 2, 1, 3]);  getitem_496 = None
        view_834: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_445, [4, 512, 1280]);  permute_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_446: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_495, [0, 2, 1, 3]);  getitem_495 = None
        view_835: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_446, [4, 512, 1280]);  permute_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_25: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_833, view_835, view_834], 2);  view_833 = view_835 = view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_836: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_25, [2048, 3840]);  cat_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_447: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        mm_201: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_836, permute_447);  permute_447 = None
        mm_202: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_448, view_836);  permute_448 = None
        sum_300: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_836, [0], True);  view_836 = None
        view_837: "f32[3840]" = torch.ops.aten.reshape.default(sum_300, [3840]);  sum_300 = None
        view_838: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_201, [4, 512, 1280]);  mm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1112: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_838, primals_136);  primals_136 = None
        mul_1113: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1112, 1280)
        sum_301: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1112, [2], True)
        mul_1114: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1112, mul_134);  mul_1112 = None
        sum_302: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1114, [2], True);  mul_1114 = None
        mul_1115: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_134, sum_302);  sum_302 = None
        sub_251: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1113, sum_301);  mul_1113 = sum_301 = None
        sub_252: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_251, mul_1115);  sub_251 = mul_1115 = None
        mul_1116: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_50, sub_252);  div_50 = sub_252 = None
        mul_1117: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_838, mul_134);  mul_134 = None
        sum_303: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1117, [0, 1]);  mul_1117 = None
        sum_304: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_838, [0, 1]);  view_838 = None
        add_394: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_393, mul_1116);  add_393 = mul_1116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_1118: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_1119: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_394, mul_1118);  mul_1118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_839: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1119, [2048, 1280]);  mul_1119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_449: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_135, [1, 0]);  primals_135 = None
        mm_203: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_839, permute_449);  permute_449 = None
        mm_204: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_450, view_839);  permute_450 = None
        sum_305: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_839, [0], True);  view_839 = None
        view_840: "f32[1280]" = torch.ops.aten.reshape.default(sum_305, [1280]);  sum_305 = None
        view_841: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_203, [4, 512, 5120]);  mm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_130: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_42, [4, 512, 5120]);  addmm_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_128: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_130, 0.5)
        mul_1120: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_841, mul_128);  mul_128 = None
        pow_11: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 3.0)
        mul_129: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_11, 0.044715);  pow_11 = None
        add_89: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_130, mul_129);  mul_129 = None
        mul_130: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_89, 0.7978845608028654);  add_89 = None
        tanh_10: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_130);  mul_130 = None
        add_90: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_1121: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_841, add_90);  view_841 = add_90 = None
        mul_1122: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_253: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1122);  mul_1122 = None
        mul_1123: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1120, sub_253);  mul_1120 = sub_253 = None
        mul_1124: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1123, 0.7978845608028654);  mul_1123 = None
        mul_1125: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1124, 0.044715)
        pow_62: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_130, 2.0);  view_130 = None
        mul_1126: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_62, 3.0);  pow_62 = None
        mul_1127: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1125, mul_1126);  mul_1125 = mul_1126 = None
        add_395: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1124, mul_1127);  mul_1124 = mul_1127 = None
        mul_1128: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1121, 0.5);  mul_1121 = None
        add_396: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_395, mul_1128);  add_395 = mul_1128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_842: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_396, [2048, 5120]);  add_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_451: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        mm_205: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_842, permute_451);  permute_451 = None
        mm_206: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_452, view_842);  permute_452 = None
        sum_306: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_842, [0], True);  view_842 = None
        view_843: "f32[5120]" = torch.ops.aten.reshape.default(sum_306, [5120]);  sum_306 = None
        view_844: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_205, [4, 512, 1280]);  mm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1130: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_844, primals_130);  primals_130 = None
        mul_1131: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1130, 1280)
        sum_307: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1130, [2], True)
        mul_1132: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1130, mul_126);  mul_1130 = None
        sum_308: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1132, [2], True);  mul_1132 = None
        mul_1133: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_126, sum_308);  sum_308 = None
        sub_255: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1131, sum_307);  mul_1131 = sum_307 = None
        sub_256: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_255, mul_1133);  sub_255 = mul_1133 = None
        mul_1134: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_51, sub_256);  div_51 = sub_256 = None
        mul_1135: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_844, mul_126);  mul_126 = None
        sum_309: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1135, [0, 1]);  mul_1135 = None
        sum_310: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_844, [0, 1]);  view_844 = None
        add_397: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_394, mul_1134);  add_394 = mul_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_51: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_1136: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_1137: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_397, mul_1136);  mul_1136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_845: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1137, [2048, 1280]);  mul_1137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_453: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        mm_207: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_845, permute_453);  permute_453 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_43: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_115, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_126: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_43, [4, 512, -1]);  permute_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_127: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_126, [-1, 1280]);  view_126 = None
        permute_454: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_127, [1, 0]);  view_127 = None
        mm_208: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_454, view_845);  permute_454 = None
        sum_311: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_845, [0], True);  view_845 = None
        view_846: "f32[1280]" = torch.ops.aten.reshape.default(sum_311, [1280]);  sum_311 = None
        view_847: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_207, [4, 512, 1280]);  mm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_848: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_847, [4, 512, 20, 64]);  view_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_455: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_848, [0, 2, 1, 3]);  view_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_25 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_455, permute_42, permute_40, permute_41, expand_2, getitem_115, getitem_116, getitem_117, getitem_118, 0.1, [True, True, True, False], scale = 0.125);  permute_455 = permute_42 = permute_40 = permute_41 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = None
        getitem_498: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_25[0]
        getitem_499: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_25[1]
        getitem_500: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_25[2];  _scaled_dot_product_efficient_attention_backward_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_456: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_498, [0, 2, 1, 3]);  getitem_498 = None
        view_849: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_456, [4, 512, 1280]);  permute_456 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_457: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_500, [0, 2, 1, 3]);  getitem_500 = None
        view_850: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_457, [4, 512, 1280]);  permute_457 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_458: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_499, [0, 2, 1, 3]);  getitem_499 = None
        view_851: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_458, [4, 512, 1280]);  permute_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_26: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_849, view_851, view_850], 2);  view_849 = view_851 = view_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_852: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_26, [2048, 3840]);  cat_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_459: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        mm_209: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_852, permute_459);  permute_459 = None
        mm_210: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_460, view_852);  permute_460 = None
        sum_312: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
        view_853: "f32[3840]" = torch.ops.aten.reshape.default(sum_312, [3840]);  sum_312 = None
        view_854: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_209, [4, 512, 1280]);  mm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1139: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_854, primals_124);  primals_124 = None
        mul_1140: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1139, 1280)
        sum_313: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1139, [2], True)
        mul_1141: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1139, mul_122);  mul_1139 = None
        sum_314: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1141, [2], True);  mul_1141 = None
        mul_1142: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_122, sum_314);  sum_314 = None
        sub_258: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1140, sum_313);  mul_1140 = sum_313 = None
        sub_259: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_258, mul_1142);  sub_258 = mul_1142 = None
        mul_1143: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_52, sub_259);  div_52 = sub_259 = None
        mul_1144: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_854, mul_122);  mul_122 = None
        sum_315: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1144, [0, 1]);  mul_1144 = None
        sum_316: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_854, [0, 1]);  view_854 = None
        add_398: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_397, mul_1143);  add_397 = mul_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_52: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_1145: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_1146: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_398, mul_1145);  mul_1145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_855: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1146, [2048, 1280]);  mul_1146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_461: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        mm_211: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_855, permute_461);  permute_461 = None
        mm_212: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_462, view_855);  permute_462 = None
        sum_317: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        view_856: "f32[1280]" = torch.ops.aten.reshape.default(sum_317, [1280]);  sum_317 = None
        view_857: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_211, [4, 512, 5120]);  mm_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_118: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_38, [4, 512, 5120]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_116: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_118, 0.5)
        mul_1147: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_857, mul_116);  mul_116 = None
        pow_10: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 3.0)
        mul_117: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_10, 0.044715);  pow_10 = None
        add_81: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_118, mul_117);  mul_117 = None
        mul_118: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_81, 0.7978845608028654);  add_81 = None
        tanh_9: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_118);  mul_118 = None
        add_82: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_1148: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_857, add_82);  view_857 = add_82 = None
        mul_1149: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_260: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1149);  mul_1149 = None
        mul_1150: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1147, sub_260);  mul_1147 = sub_260 = None
        mul_1151: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1150, 0.7978845608028654);  mul_1150 = None
        mul_1152: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1151, 0.044715)
        pow_63: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_118, 2.0);  view_118 = None
        mul_1153: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_63, 3.0);  pow_63 = None
        mul_1154: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1152, mul_1153);  mul_1152 = mul_1153 = None
        add_399: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1151, mul_1154);  mul_1151 = mul_1154 = None
        mul_1155: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1148, 0.5);  mul_1148 = None
        add_400: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_399, mul_1155);  add_399 = mul_1155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_858: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_400, [2048, 5120]);  add_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_463: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_121, [1, 0]);  primals_121 = None
        mm_213: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_858, permute_463);  permute_463 = None
        mm_214: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_464, view_858);  permute_464 = None
        sum_318: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_858, [0], True);  view_858 = None
        view_859: "f32[5120]" = torch.ops.aten.reshape.default(sum_318, [5120]);  sum_318 = None
        view_860: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_213, [4, 512, 1280]);  mm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1157: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_860, primals_118);  primals_118 = None
        mul_1158: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1157, 1280)
        sum_319: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1157, [2], True)
        mul_1159: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1157, mul_114);  mul_1157 = None
        sum_320: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1159, [2], True);  mul_1159 = None
        mul_1160: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_114, sum_320);  sum_320 = None
        sub_262: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1158, sum_319);  mul_1158 = sum_319 = None
        sub_263: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_262, mul_1160);  sub_262 = mul_1160 = None
        mul_1161: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_53, sub_263);  div_53 = sub_263 = None
        mul_1162: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_860, mul_114);  mul_114 = None
        sum_321: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1162, [0, 1]);  mul_1162 = None
        sum_322: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_860, [0, 1]);  view_860 = None
        add_401: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_398, mul_1161);  add_398 = mul_1161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_53: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_1163: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_1164: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_401, mul_1163);  mul_1163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_861: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1164, [2048, 1280]);  mul_1164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_465: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_117, [1, 0]);  primals_117 = None
        mm_215: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_861, permute_465);  permute_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_39: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_104, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_114: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_39, [4, 512, -1]);  permute_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_115: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_114, [-1, 1280]);  view_114 = None
        permute_466: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_115, [1, 0]);  view_115 = None
        mm_216: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_466, view_861);  permute_466 = None
        sum_323: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_861, [0], True);  view_861 = None
        view_862: "f32[1280]" = torch.ops.aten.reshape.default(sum_323, [1280]);  sum_323 = None
        view_863: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_215, [4, 512, 1280]);  mm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_864: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_863, [4, 512, 20, 64]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_467: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_864, [0, 2, 1, 3]);  view_864 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_26 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_467, permute_38, permute_36, permute_37, expand_2, getitem_104, getitem_105, getitem_106, getitem_107, 0.1, [True, True, True, False], scale = 0.125);  permute_467 = permute_38 = permute_36 = permute_37 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = None
        getitem_502: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_26[0]
        getitem_503: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_26[1]
        getitem_504: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_26[2];  _scaled_dot_product_efficient_attention_backward_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_468: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_502, [0, 2, 1, 3]);  getitem_502 = None
        view_865: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_468, [4, 512, 1280]);  permute_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_469: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_504, [0, 2, 1, 3]);  getitem_504 = None
        view_866: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_469, [4, 512, 1280]);  permute_469 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_470: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_503, [0, 2, 1, 3]);  getitem_503 = None
        view_867: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_470, [4, 512, 1280]);  permute_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_27: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_865, view_867, view_866], 2);  view_865 = view_867 = view_866 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_868: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_27, [2048, 3840]);  cat_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_471: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        mm_217: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_868, permute_471);  permute_471 = None
        mm_218: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_472, view_868);  permute_472 = None
        sum_324: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_868, [0], True);  view_868 = None
        view_869: "f32[3840]" = torch.ops.aten.reshape.default(sum_324, [3840]);  sum_324 = None
        view_870: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_217, [4, 512, 1280]);  mm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1166: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_870, primals_112);  primals_112 = None
        mul_1167: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1166, 1280)
        sum_325: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1166, [2], True)
        mul_1168: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1166, mul_110);  mul_1166 = None
        sum_326: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1168, [2], True);  mul_1168 = None
        mul_1169: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_110, sum_326);  sum_326 = None
        sub_265: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1167, sum_325);  mul_1167 = sum_325 = None
        sub_266: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_265, mul_1169);  sub_265 = mul_1169 = None
        mul_1170: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_54, sub_266);  div_54 = sub_266 = None
        mul_1171: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_870, mul_110);  mul_110 = None
        sum_327: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1171, [0, 1]);  mul_1171 = None
        sum_328: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_870, [0, 1]);  view_870 = None
        add_402: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_401, mul_1170);  add_401 = mul_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_54: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_1172: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_1173: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_402, mul_1172);  mul_1172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_871: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1173, [2048, 1280]);  mul_1173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_473: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        mm_219: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_871, permute_473);  permute_473 = None
        mm_220: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_474, view_871);  permute_474 = None
        sum_329: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_871, [0], True);  view_871 = None
        view_872: "f32[1280]" = torch.ops.aten.reshape.default(sum_329, [1280]);  sum_329 = None
        view_873: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_219, [4, 512, 5120]);  mm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_106: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_34, [4, 512, 5120]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_104: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_106, 0.5)
        mul_1174: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_873, mul_104);  mul_104 = None
        pow_9: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 3.0)
        mul_105: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_73: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_106, mul_105);  mul_105 = None
        mul_106: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_73, 0.7978845608028654);  add_73 = None
        tanh_8: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_106);  mul_106 = None
        add_74: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_1175: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_873, add_74);  view_873 = add_74 = None
        mul_1176: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_267: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1176);  mul_1176 = None
        mul_1177: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1174, sub_267);  mul_1174 = sub_267 = None
        mul_1178: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1177, 0.7978845608028654);  mul_1177 = None
        mul_1179: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1178, 0.044715)
        pow_64: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_106, 2.0);  view_106 = None
        mul_1180: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_64, 3.0);  pow_64 = None
        mul_1181: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1179, mul_1180);  mul_1179 = mul_1180 = None
        add_403: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1178, mul_1181);  mul_1178 = mul_1181 = None
        mul_1182: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1175, 0.5);  mul_1175 = None
        add_404: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_403, mul_1182);  add_403 = mul_1182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_874: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_404, [2048, 5120]);  add_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_475: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        mm_221: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_874, permute_475);  permute_475 = None
        mm_222: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_476, view_874);  permute_476 = None
        sum_330: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
        view_875: "f32[5120]" = torch.ops.aten.reshape.default(sum_330, [5120]);  sum_330 = None
        view_876: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_221, [4, 512, 1280]);  mm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1184: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_876, primals_106);  primals_106 = None
        mul_1185: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1184, 1280)
        sum_331: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1184, [2], True)
        mul_1186: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1184, mul_102);  mul_1184 = None
        sum_332: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1186, [2], True);  mul_1186 = None
        mul_1187: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_102, sum_332);  sum_332 = None
        sub_269: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1185, sum_331);  mul_1185 = sum_331 = None
        sub_270: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_269, mul_1187);  sub_269 = mul_1187 = None
        mul_1188: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_55, sub_270);  div_55 = sub_270 = None
        mul_1189: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_876, mul_102);  mul_102 = None
        sum_333: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1189, [0, 1]);  mul_1189 = None
        sum_334: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_876, [0, 1]);  view_876 = None
        add_405: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_402, mul_1188);  add_402 = mul_1188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_55: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_1190: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_1191: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_405, mul_1190);  mul_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_877: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1191, [2048, 1280]);  mul_1191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_477: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        mm_223: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_877, permute_477);  permute_477 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_35: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_93, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_102: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_35, [4, 512, -1]);  permute_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_103: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_102, [-1, 1280]);  view_102 = None
        permute_478: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_103, [1, 0]);  view_103 = None
        mm_224: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_478, view_877);  permute_478 = None
        sum_335: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
        view_878: "f32[1280]" = torch.ops.aten.reshape.default(sum_335, [1280]);  sum_335 = None
        view_879: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_223, [4, 512, 1280]);  mm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_880: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_879, [4, 512, 20, 64]);  view_879 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_479: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_880, [0, 2, 1, 3]);  view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_27 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_479, permute_34, permute_32, permute_33, expand_2, getitem_93, getitem_94, getitem_95, getitem_96, 0.1, [True, True, True, False], scale = 0.125);  permute_479 = permute_34 = permute_32 = permute_33 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = None
        getitem_506: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_27[0]
        getitem_507: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_27[1]
        getitem_508: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_27[2];  _scaled_dot_product_efficient_attention_backward_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_480: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_506, [0, 2, 1, 3]);  getitem_506 = None
        view_881: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_480, [4, 512, 1280]);  permute_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_481: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_508, [0, 2, 1, 3]);  getitem_508 = None
        view_882: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_481, [4, 512, 1280]);  permute_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_482: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_507, [0, 2, 1, 3]);  getitem_507 = None
        view_883: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_482, [4, 512, 1280]);  permute_482 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_28: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_881, view_883, view_882], 2);  view_881 = view_883 = view_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_884: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_28, [2048, 3840]);  cat_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_483: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_103, [1, 0]);  primals_103 = None
        mm_225: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_884, permute_483);  permute_483 = None
        mm_226: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_484, view_884);  permute_484 = None
        sum_336: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_884, [0], True);  view_884 = None
        view_885: "f32[3840]" = torch.ops.aten.reshape.default(sum_336, [3840]);  sum_336 = None
        view_886: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_225, [4, 512, 1280]);  mm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1193: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_886, primals_100);  primals_100 = None
        mul_1194: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1193, 1280)
        sum_337: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1193, [2], True)
        mul_1195: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1193, mul_98);  mul_1193 = None
        sum_338: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1195, [2], True);  mul_1195 = None
        mul_1196: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_98, sum_338);  sum_338 = None
        sub_272: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1194, sum_337);  mul_1194 = sum_337 = None
        sub_273: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_272, mul_1196);  sub_272 = mul_1196 = None
        mul_1197: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_56, sub_273);  div_56 = sub_273 = None
        mul_1198: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_886, mul_98);  mul_98 = None
        sum_339: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1198, [0, 1]);  mul_1198 = None
        sum_340: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_886, [0, 1]);  view_886 = None
        add_406: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_405, mul_1197);  add_405 = mul_1197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_56: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_1199: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_1200: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_406, mul_1199);  mul_1199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_887: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1200, [2048, 1280]);  mul_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_485: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        mm_227: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_887, permute_485);  permute_485 = None
        mm_228: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_486, view_887);  permute_486 = None
        sum_341: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_887, [0], True);  view_887 = None
        view_888: "f32[1280]" = torch.ops.aten.reshape.default(sum_341, [1280]);  sum_341 = None
        view_889: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_227, [4, 512, 5120]);  mm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_94: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_30, [4, 512, 5120]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_92: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_94, 0.5)
        mul_1201: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_889, mul_92);  mul_92 = None
        pow_8: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 3.0)
        mul_93: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_8, 0.044715);  pow_8 = None
        add_65: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_94, mul_93);  mul_93 = None
        mul_94: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_65, 0.7978845608028654);  add_65 = None
        tanh_7: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_94);  mul_94 = None
        add_66: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_1202: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_889, add_66);  view_889 = add_66 = None
        mul_1203: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_274: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1203);  mul_1203 = None
        mul_1204: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1201, sub_274);  mul_1201 = sub_274 = None
        mul_1205: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1204, 0.7978845608028654);  mul_1204 = None
        mul_1206: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1205, 0.044715)
        pow_65: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_94, 2.0);  view_94 = None
        mul_1207: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_65, 3.0);  pow_65 = None
        mul_1208: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1206, mul_1207);  mul_1206 = mul_1207 = None
        add_407: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1205, mul_1208);  mul_1205 = mul_1208 = None
        mul_1209: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1202, 0.5);  mul_1202 = None
        add_408: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_407, mul_1209);  add_407 = mul_1209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_890: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_408, [2048, 5120]);  add_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_487: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        mm_229: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_890, permute_487);  permute_487 = None
        mm_230: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_488, view_890);  permute_488 = None
        sum_342: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
        view_891: "f32[5120]" = torch.ops.aten.reshape.default(sum_342, [5120]);  sum_342 = None
        view_892: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_229, [4, 512, 1280]);  mm_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1211: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_892, primals_94);  primals_94 = None
        mul_1212: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1211, 1280)
        sum_343: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1211, [2], True)
        mul_1213: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1211, mul_90);  mul_1211 = None
        sum_344: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1213, [2], True);  mul_1213 = None
        mul_1214: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_90, sum_344);  sum_344 = None
        sub_276: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1212, sum_343);  mul_1212 = sum_343 = None
        sub_277: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_276, mul_1214);  sub_276 = mul_1214 = None
        mul_1215: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_57, sub_277);  div_57 = sub_277 = None
        mul_1216: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_892, mul_90);  mul_90 = None
        sum_345: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1216, [0, 1]);  mul_1216 = None
        sum_346: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_892, [0, 1]);  view_892 = None
        add_409: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_406, mul_1215);  add_406 = mul_1215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_57: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_1217: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_1218: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_409, mul_1217);  mul_1217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_893: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1218, [2048, 1280]);  mul_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_489: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_93, [1, 0]);  primals_93 = None
        mm_231: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_893, permute_489);  permute_489 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_31: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_90: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_31, [4, 512, -1]);  permute_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_91: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_90, [-1, 1280]);  view_90 = None
        permute_490: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_91, [1, 0]);  view_91 = None
        mm_232: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_490, view_893);  permute_490 = None
        sum_347: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_893, [0], True);  view_893 = None
        view_894: "f32[1280]" = torch.ops.aten.reshape.default(sum_347, [1280]);  sum_347 = None
        view_895: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_231, [4, 512, 1280]);  mm_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_896: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_895, [4, 512, 20, 64]);  view_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_491: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_896, [0, 2, 1, 3]);  view_896 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_28 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_491, permute_30, permute_28, permute_29, expand_2, getitem_82, getitem_83, getitem_84, getitem_85, 0.1, [True, True, True, False], scale = 0.125);  permute_491 = permute_30 = permute_28 = permute_29 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = None
        getitem_510: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_28[0]
        getitem_511: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_28[1]
        getitem_512: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_28[2];  _scaled_dot_product_efficient_attention_backward_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_492: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_510, [0, 2, 1, 3]);  getitem_510 = None
        view_897: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_492, [4, 512, 1280]);  permute_492 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_493: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_512, [0, 2, 1, 3]);  getitem_512 = None
        view_898: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_493, [4, 512, 1280]);  permute_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_494: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_511, [0, 2, 1, 3]);  getitem_511 = None
        view_899: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_494, [4, 512, 1280]);  permute_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_29: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_897, view_899, view_898], 2);  view_897 = view_899 = view_898 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_900: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_29, [2048, 3840]);  cat_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_495: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        mm_233: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_900, permute_495);  permute_495 = None
        mm_234: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_496, view_900);  permute_496 = None
        sum_348: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_900, [0], True);  view_900 = None
        view_901: "f32[3840]" = torch.ops.aten.reshape.default(sum_348, [3840]);  sum_348 = None
        view_902: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_233, [4, 512, 1280]);  mm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1220: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_902, primals_88);  primals_88 = None
        mul_1221: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1220, 1280)
        sum_349: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1220, [2], True)
        mul_1222: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1220, mul_86);  mul_1220 = None
        sum_350: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1222, [2], True);  mul_1222 = None
        mul_1223: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_86, sum_350);  sum_350 = None
        sub_279: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1221, sum_349);  mul_1221 = sum_349 = None
        sub_280: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_279, mul_1223);  sub_279 = mul_1223 = None
        mul_1224: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_58, sub_280);  div_58 = sub_280 = None
        mul_1225: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_902, mul_86);  mul_86 = None
        sum_351: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1225, [0, 1]);  mul_1225 = None
        sum_352: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_902, [0, 1]);  view_902 = None
        add_410: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_409, mul_1224);  add_409 = mul_1224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_58: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_1226: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_1227: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_410, mul_1226);  mul_1226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_903: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1227, [2048, 1280]);  mul_1227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_497: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        mm_235: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_903, permute_497);  permute_497 = None
        mm_236: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_498, view_903);  permute_498 = None
        sum_353: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
        view_904: "f32[1280]" = torch.ops.aten.reshape.default(sum_353, [1280]);  sum_353 = None
        view_905: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_235, [4, 512, 5120]);  mm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_82: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_26, [4, 512, 5120]);  addmm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_82, 0.5)
        mul_1228: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_905, mul_80);  mul_80 = None
        pow_7: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 3.0)
        mul_81: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_7, 0.044715);  pow_7 = None
        add_57: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_82, mul_81);  mul_81 = None
        mul_82: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_57, 0.7978845608028654);  add_57 = None
        tanh_6: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_58: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_1229: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_905, add_58);  view_905 = add_58 = None
        mul_1230: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_281: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1230);  mul_1230 = None
        mul_1231: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1228, sub_281);  mul_1228 = sub_281 = None
        mul_1232: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1231, 0.7978845608028654);  mul_1231 = None
        mul_1233: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1232, 0.044715)
        pow_66: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_82, 2.0);  view_82 = None
        mul_1234: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_66, 3.0);  pow_66 = None
        mul_1235: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1233, mul_1234);  mul_1233 = mul_1234 = None
        add_411: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1232, mul_1235);  mul_1232 = mul_1235 = None
        mul_1236: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1229, 0.5);  mul_1229 = None
        add_412: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_411, mul_1236);  add_411 = mul_1236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_906: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_412, [2048, 5120]);  add_412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_499: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        mm_237: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_906, permute_499);  permute_499 = None
        mm_238: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_500, view_906);  permute_500 = None
        sum_354: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_906, [0], True);  view_906 = None
        view_907: "f32[5120]" = torch.ops.aten.reshape.default(sum_354, [5120]);  sum_354 = None
        view_908: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_237, [4, 512, 1280]);  mm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1238: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_908, primals_82);  primals_82 = None
        mul_1239: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1238, 1280)
        sum_355: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1238, [2], True)
        mul_1240: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1238, mul_78);  mul_1238 = None
        sum_356: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1240, [2], True);  mul_1240 = None
        mul_1241: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_78, sum_356);  sum_356 = None
        sub_283: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1239, sum_355);  mul_1239 = sum_355 = None
        sub_284: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_283, mul_1241);  sub_283 = mul_1241 = None
        mul_1242: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_59, sub_284);  div_59 = sub_284 = None
        mul_1243: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_908, mul_78);  mul_78 = None
        sum_357: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1243, [0, 1]);  mul_1243 = None
        sum_358: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_908, [0, 1]);  view_908 = None
        add_413: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_410, mul_1242);  add_410 = mul_1242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_59: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_1244: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_1245: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_413, mul_1244);  mul_1244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_909: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1245, [2048, 1280]);  mul_1245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_501: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        mm_239: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_909, permute_501);  permute_501 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_27: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_71, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_78: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_27, [4, 512, -1]);  permute_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_79: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_78, [-1, 1280]);  view_78 = None
        permute_502: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_79, [1, 0]);  view_79 = None
        mm_240: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_502, view_909);  permute_502 = None
        sum_359: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_909, [0], True);  view_909 = None
        view_910: "f32[1280]" = torch.ops.aten.reshape.default(sum_359, [1280]);  sum_359 = None
        view_911: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_239, [4, 512, 1280]);  mm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_912: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_911, [4, 512, 20, 64]);  view_911 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_503: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_912, [0, 2, 1, 3]);  view_912 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_29 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_503, permute_26, permute_24, permute_25, expand_2, getitem_71, getitem_72, getitem_73, getitem_74, 0.1, [True, True, True, False], scale = 0.125);  permute_503 = permute_26 = permute_24 = permute_25 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = None
        getitem_514: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_29[0]
        getitem_515: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_29[1]
        getitem_516: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_29[2];  _scaled_dot_product_efficient_attention_backward_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_504: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_514, [0, 2, 1, 3]);  getitem_514 = None
        view_913: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_504, [4, 512, 1280]);  permute_504 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_505: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_516, [0, 2, 1, 3]);  getitem_516 = None
        view_914: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_505, [4, 512, 1280]);  permute_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_506: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_515, [0, 2, 1, 3]);  getitem_515 = None
        view_915: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_506, [4, 512, 1280]);  permute_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_30: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_913, view_915, view_914], 2);  view_913 = view_915 = view_914 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_916: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_30, [2048, 3840]);  cat_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_507: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        mm_241: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_916, permute_507);  permute_507 = None
        mm_242: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_508, view_916);  permute_508 = None
        sum_360: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_916, [0], True);  view_916 = None
        view_917: "f32[3840]" = torch.ops.aten.reshape.default(sum_360, [3840]);  sum_360 = None
        view_918: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_241, [4, 512, 1280]);  mm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1247: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_918, primals_76);  primals_76 = None
        mul_1248: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1247, 1280)
        sum_361: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1247, [2], True)
        mul_1249: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1247, mul_74);  mul_1247 = None
        sum_362: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1249, [2], True);  mul_1249 = None
        mul_1250: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_74, sum_362);  sum_362 = None
        sub_286: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1248, sum_361);  mul_1248 = sum_361 = None
        sub_287: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_286, mul_1250);  sub_286 = mul_1250 = None
        mul_1251: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_60, sub_287);  div_60 = sub_287 = None
        mul_1252: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_918, mul_74);  mul_74 = None
        sum_363: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1252, [0, 1]);  mul_1252 = None
        sum_364: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_918, [0, 1]);  view_918 = None
        add_414: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_413, mul_1251);  add_413 = mul_1251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_60: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_1253: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_1254: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_414, mul_1253);  mul_1253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_919: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1254, [2048, 1280]);  mul_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_509: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        mm_243: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_919, permute_509);  permute_509 = None
        mm_244: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_510, view_919);  permute_510 = None
        sum_365: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_919, [0], True);  view_919 = None
        view_920: "f32[1280]" = torch.ops.aten.reshape.default(sum_365, [1280]);  sum_365 = None
        view_921: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_243, [4, 512, 5120]);  mm_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_70: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_22, [4, 512, 5120]);  addmm_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_68: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_70, 0.5)
        mul_1255: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_921, mul_68);  mul_68 = None
        pow_6: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 3.0)
        mul_69: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_49: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_70, mul_69);  mul_69 = None
        mul_70: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_49, 0.7978845608028654);  add_49 = None
        tanh_5: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_70);  mul_70 = None
        add_50: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_1256: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_921, add_50);  view_921 = add_50 = None
        mul_1257: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_288: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1257);  mul_1257 = None
        mul_1258: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1255, sub_288);  mul_1255 = sub_288 = None
        mul_1259: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1258, 0.7978845608028654);  mul_1258 = None
        mul_1260: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1259, 0.044715)
        pow_67: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_70, 2.0);  view_70 = None
        mul_1261: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_67, 3.0);  pow_67 = None
        mul_1262: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1260, mul_1261);  mul_1260 = mul_1261 = None
        add_415: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1259, mul_1262);  mul_1259 = mul_1262 = None
        mul_1263: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1256, 0.5);  mul_1256 = None
        add_416: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_415, mul_1263);  add_415 = mul_1263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_922: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_416, [2048, 5120]);  add_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_511: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        mm_245: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_922, permute_511);  permute_511 = None
        mm_246: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_512, view_922);  permute_512 = None
        sum_366: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_922, [0], True);  view_922 = None
        view_923: "f32[5120]" = torch.ops.aten.reshape.default(sum_366, [5120]);  sum_366 = None
        view_924: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_245, [4, 512, 1280]);  mm_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1265: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_924, primals_70);  primals_70 = None
        mul_1266: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1265, 1280)
        sum_367: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1265, [2], True)
        mul_1267: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1265, mul_66);  mul_1265 = None
        sum_368: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1267, [2], True);  mul_1267 = None
        mul_1268: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_66, sum_368);  sum_368 = None
        sub_290: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1266, sum_367);  mul_1266 = sum_367 = None
        sub_291: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_290, mul_1268);  sub_290 = mul_1268 = None
        mul_1269: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_61, sub_291);  div_61 = sub_291 = None
        mul_1270: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_924, mul_66);  mul_66 = None
        sum_369: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1270, [0, 1]);  mul_1270 = None
        sum_370: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_924, [0, 1]);  view_924 = None
        add_417: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_414, mul_1269);  add_414 = mul_1269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_61: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_1271: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_1272: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_417, mul_1271);  mul_1271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_925: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1272, [2048, 1280]);  mul_1272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_513: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        mm_247: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_925, permute_513);  permute_513 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_23: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_60, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_66: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_23, [4, 512, -1]);  permute_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_67: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_66, [-1, 1280]);  view_66 = None
        permute_514: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_67, [1, 0]);  view_67 = None
        mm_248: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_514, view_925);  permute_514 = None
        sum_371: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
        view_926: "f32[1280]" = torch.ops.aten.reshape.default(sum_371, [1280]);  sum_371 = None
        view_927: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_247, [4, 512, 1280]);  mm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_928: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_927, [4, 512, 20, 64]);  view_927 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_515: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_928, [0, 2, 1, 3]);  view_928 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_30 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_515, permute_22, permute_20, permute_21, expand_2, getitem_60, getitem_61, getitem_62, getitem_63, 0.1, [True, True, True, False], scale = 0.125);  permute_515 = permute_22 = permute_20 = permute_21 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = None
        getitem_518: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_30[0]
        getitem_519: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_30[1]
        getitem_520: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_30[2];  _scaled_dot_product_efficient_attention_backward_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_516: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_518, [0, 2, 1, 3]);  getitem_518 = None
        view_929: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_516, [4, 512, 1280]);  permute_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_517: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_520, [0, 2, 1, 3]);  getitem_520 = None
        view_930: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_517, [4, 512, 1280]);  permute_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_518: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_519, [0, 2, 1, 3]);  getitem_519 = None
        view_931: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_518, [4, 512, 1280]);  permute_518 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_31: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_929, view_931, view_930], 2);  view_929 = view_931 = view_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_932: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_31, [2048, 3840]);  cat_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_519: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_67, [1, 0]);  primals_67 = None
        mm_249: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_932, permute_519);  permute_519 = None
        mm_250: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_520, view_932);  permute_520 = None
        sum_372: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_932, [0], True);  view_932 = None
        view_933: "f32[3840]" = torch.ops.aten.reshape.default(sum_372, [3840]);  sum_372 = None
        view_934: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_249, [4, 512, 1280]);  mm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1274: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_934, primals_64);  primals_64 = None
        mul_1275: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1274, 1280)
        sum_373: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1274, [2], True)
        mul_1276: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1274, mul_62);  mul_1274 = None
        sum_374: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1276, [2], True);  mul_1276 = None
        mul_1277: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_62, sum_374);  sum_374 = None
        sub_293: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1275, sum_373);  mul_1275 = sum_373 = None
        sub_294: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_293, mul_1277);  sub_293 = mul_1277 = None
        mul_1278: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_62, sub_294);  div_62 = sub_294 = None
        mul_1279: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_934, mul_62);  mul_62 = None
        sum_375: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1279, [0, 1]);  mul_1279 = None
        sum_376: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_934, [0, 1]);  view_934 = None
        add_418: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_417, mul_1278);  add_417 = mul_1278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_1280: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_1281: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_418, mul_1280);  mul_1280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_935: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1281, [2048, 1280]);  mul_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_521: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_63, [1, 0]);  primals_63 = None
        mm_251: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_935, permute_521);  permute_521 = None
        mm_252: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_522, view_935);  permute_522 = None
        sum_377: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_935, [0], True);  view_935 = None
        view_936: "f32[1280]" = torch.ops.aten.reshape.default(sum_377, [1280]);  sum_377 = None
        view_937: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_251, [4, 512, 5120]);  mm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_58: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_18, [4, 512, 5120]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_56: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_58, 0.5)
        mul_1282: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_937, mul_56);  mul_56 = None
        pow_5: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 3.0)
        mul_57: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_5, 0.044715);  pow_5 = None
        add_41: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_58, mul_57);  mul_57 = None
        mul_58: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_41, 0.7978845608028654);  add_41 = None
        tanh_4: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_58);  mul_58 = None
        add_42: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_1283: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_937, add_42);  view_937 = add_42 = None
        mul_1284: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_295: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1284);  mul_1284 = None
        mul_1285: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1282, sub_295);  mul_1282 = sub_295 = None
        mul_1286: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1285, 0.7978845608028654);  mul_1285 = None
        mul_1287: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1286, 0.044715)
        pow_68: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_58, 2.0);  view_58 = None
        mul_1288: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_68, 3.0);  pow_68 = None
        mul_1289: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1287, mul_1288);  mul_1287 = mul_1288 = None
        add_419: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1286, mul_1289);  mul_1286 = mul_1289 = None
        mul_1290: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1283, 0.5);  mul_1283 = None
        add_420: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_419, mul_1290);  add_419 = mul_1290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_938: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_420, [2048, 5120]);  add_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_523: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        mm_253: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_938, permute_523);  permute_523 = None
        mm_254: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_524, view_938);  permute_524 = None
        sum_378: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_938, [0], True);  view_938 = None
        view_939: "f32[5120]" = torch.ops.aten.reshape.default(sum_378, [5120]);  sum_378 = None
        view_940: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_253, [4, 512, 1280]);  mm_253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1292: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_940, primals_58);  primals_58 = None
        mul_1293: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1292, 1280)
        sum_379: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1292, [2], True)
        mul_1294: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1292, mul_54);  mul_1292 = None
        sum_380: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1294, [2], True);  mul_1294 = None
        mul_1295: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_54, sum_380);  sum_380 = None
        sub_297: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1293, sum_379);  mul_1293 = sum_379 = None
        sub_298: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_297, mul_1295);  sub_297 = mul_1295 = None
        mul_1296: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_63, sub_298);  div_63 = sub_298 = None
        mul_1297: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_940, mul_54);  mul_54 = None
        sum_381: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1297, [0, 1]);  mul_1297 = None
        sum_382: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_940, [0, 1]);  view_940 = None
        add_421: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_418, mul_1296);  add_418 = mul_1296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_63: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_1298: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_1299: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_421, mul_1298);  mul_1298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_941: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1299, [2048, 1280]);  mul_1299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_525: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        mm_255: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_941, permute_525);  permute_525 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_19: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_49, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_54: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_19, [4, 512, -1]);  permute_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_55: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_54, [-1, 1280]);  view_54 = None
        permute_526: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_55, [1, 0]);  view_55 = None
        mm_256: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_526, view_941);  permute_526 = None
        sum_383: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_941, [0], True);  view_941 = None
        view_942: "f32[1280]" = torch.ops.aten.reshape.default(sum_383, [1280]);  sum_383 = None
        view_943: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_255, [4, 512, 1280]);  mm_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_944: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_943, [4, 512, 20, 64]);  view_943 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_527: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_944, [0, 2, 1, 3]);  view_944 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_31 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_527, permute_18, permute_16, permute_17, expand_2, getitem_49, getitem_50, getitem_51, getitem_52, 0.1, [True, True, True, False], scale = 0.125);  permute_527 = permute_18 = permute_16 = permute_17 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = None
        getitem_522: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_31[0]
        getitem_523: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_31[1]
        getitem_524: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_31[2];  _scaled_dot_product_efficient_attention_backward_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_528: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_522, [0, 2, 1, 3]);  getitem_522 = None
        view_945: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_528, [4, 512, 1280]);  permute_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_529: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_524, [0, 2, 1, 3]);  getitem_524 = None
        view_946: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_529, [4, 512, 1280]);  permute_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_530: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_523, [0, 2, 1, 3]);  getitem_523 = None
        view_947: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_530, [4, 512, 1280]);  permute_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_32: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_945, view_947, view_946], 2);  view_945 = view_947 = view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_948: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_32, [2048, 3840]);  cat_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_531: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        mm_257: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_948, permute_531);  permute_531 = None
        mm_258: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_532, view_948);  permute_532 = None
        sum_384: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_948, [0], True);  view_948 = None
        view_949: "f32[3840]" = torch.ops.aten.reshape.default(sum_384, [3840]);  sum_384 = None
        view_950: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_257, [4, 512, 1280]);  mm_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1301: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_950, primals_52);  primals_52 = None
        mul_1302: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1301, 1280)
        sum_385: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1301, [2], True)
        mul_1303: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1301, mul_50);  mul_1301 = None
        sum_386: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1303, [2], True);  mul_1303 = None
        mul_1304: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_50, sum_386);  sum_386 = None
        sub_300: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1302, sum_385);  mul_1302 = sum_385 = None
        sub_301: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_300, mul_1304);  sub_300 = mul_1304 = None
        mul_1305: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_64, sub_301);  div_64 = sub_301 = None
        mul_1306: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_950, mul_50);  mul_50 = None
        sum_387: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1306, [0, 1]);  mul_1306 = None
        sum_388: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_950, [0, 1]);  view_950 = None
        add_422: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_421, mul_1305);  add_421 = mul_1305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_64: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_1307: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_1308: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_422, mul_1307);  mul_1307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_951: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1308, [2048, 1280]);  mul_1308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_533: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        mm_259: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_951, permute_533);  permute_533 = None
        mm_260: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_534, view_951);  permute_534 = None
        sum_389: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_951, [0], True);  view_951 = None
        view_952: "f32[1280]" = torch.ops.aten.reshape.default(sum_389, [1280]);  sum_389 = None
        view_953: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_259, [4, 512, 5120]);  mm_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_46: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_14, [4, 512, 5120]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_44: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_46, 0.5)
        mul_1309: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_953, mul_44);  mul_44 = None
        pow_4: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 3.0)
        mul_45: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_4, 0.044715);  pow_4 = None
        add_33: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_46, mul_45);  mul_45 = None
        mul_46: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_33, 0.7978845608028654);  add_33 = None
        tanh_3: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_46);  mul_46 = None
        add_34: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_1310: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_953, add_34);  view_953 = add_34 = None
        mul_1311: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_302: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1311);  mul_1311 = None
        mul_1312: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1309, sub_302);  mul_1309 = sub_302 = None
        mul_1313: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1312, 0.7978845608028654);  mul_1312 = None
        mul_1314: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1313, 0.044715)
        pow_69: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_46, 2.0);  view_46 = None
        mul_1315: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_69, 3.0);  pow_69 = None
        mul_1316: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1314, mul_1315);  mul_1314 = mul_1315 = None
        add_423: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1313, mul_1316);  mul_1313 = mul_1316 = None
        mul_1317: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1310, 0.5);  mul_1310 = None
        add_424: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_423, mul_1317);  add_423 = mul_1317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_954: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_424, [2048, 5120]);  add_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_535: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_49, [1, 0]);  primals_49 = None
        mm_261: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_954, permute_535);  permute_535 = None
        mm_262: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_536, view_954);  permute_536 = None
        sum_390: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_954, [0], True);  view_954 = None
        view_955: "f32[5120]" = torch.ops.aten.reshape.default(sum_390, [5120]);  sum_390 = None
        view_956: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_261, [4, 512, 1280]);  mm_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1319: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_956, primals_46);  primals_46 = None
        mul_1320: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1319, 1280)
        sum_391: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1319, [2], True)
        mul_1321: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1319, mul_42);  mul_1319 = None
        sum_392: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1321, [2], True);  mul_1321 = None
        mul_1322: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_42, sum_392);  sum_392 = None
        sub_304: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1320, sum_391);  mul_1320 = sum_391 = None
        sub_305: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_304, mul_1322);  sub_304 = mul_1322 = None
        mul_1323: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_65, sub_305);  div_65 = sub_305 = None
        mul_1324: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_956, mul_42);  mul_42 = None
        sum_393: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1324, [0, 1]);  mul_1324 = None
        sum_394: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_956, [0, 1]);  view_956 = None
        add_425: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_422, mul_1323);  add_422 = mul_1323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_65: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_1325: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_1326: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_425, mul_1325);  mul_1325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_957: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1326, [2048, 1280]);  mul_1326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_537: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_45, [1, 0]);  primals_45 = None
        mm_263: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_957, permute_537);  permute_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_15: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_38, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_42: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_15, [4, 512, -1]);  permute_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_43: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_42, [-1, 1280]);  view_42 = None
        permute_538: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_43, [1, 0]);  view_43 = None
        mm_264: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_538, view_957);  permute_538 = None
        sum_395: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_957, [0], True);  view_957 = None
        view_958: "f32[1280]" = torch.ops.aten.reshape.default(sum_395, [1280]);  sum_395 = None
        view_959: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_263, [4, 512, 1280]);  mm_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_960: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_959, [4, 512, 20, 64]);  view_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_539: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_32 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_539, permute_14, permute_12, permute_13, expand_2, getitem_38, getitem_39, getitem_40, getitem_41, 0.1, [True, True, True, False], scale = 0.125);  permute_539 = permute_14 = permute_12 = permute_13 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = None
        getitem_526: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_32[0]
        getitem_527: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_32[1]
        getitem_528: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_32[2];  _scaled_dot_product_efficient_attention_backward_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_540: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_526, [0, 2, 1, 3]);  getitem_526 = None
        view_961: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_540, [4, 512, 1280]);  permute_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_541: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_528, [0, 2, 1, 3]);  getitem_528 = None
        view_962: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_541, [4, 512, 1280]);  permute_541 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_542: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_527, [0, 2, 1, 3]);  getitem_527 = None
        view_963: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_542, [4, 512, 1280]);  permute_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_33: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_961, view_963, view_962], 2);  view_961 = view_963 = view_962 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_964: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_33, [2048, 3840]);  cat_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_543: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        mm_265: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_964, permute_543);  permute_543 = None
        mm_266: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_544, view_964);  permute_544 = None
        sum_396: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        view_965: "f32[3840]" = torch.ops.aten.reshape.default(sum_396, [3840]);  sum_396 = None
        view_966: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_265, [4, 512, 1280]);  mm_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1328: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_966, primals_40);  primals_40 = None
        mul_1329: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1328, 1280)
        sum_397: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1328, [2], True)
        mul_1330: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1328, mul_38);  mul_1328 = None
        sum_398: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1330, [2], True);  mul_1330 = None
        mul_1331: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_38, sum_398);  sum_398 = None
        sub_307: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1329, sum_397);  mul_1329 = sum_397 = None
        sub_308: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_307, mul_1331);  sub_307 = mul_1331 = None
        mul_1332: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_66, sub_308);  div_66 = sub_308 = None
        mul_1333: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_966, mul_38);  mul_38 = None
        sum_399: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1333, [0, 1]);  mul_1333 = None
        sum_400: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_966, [0, 1]);  view_966 = None
        add_426: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_425, mul_1332);  add_425 = mul_1332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_66: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_1334: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_1335: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_426, mul_1334);  mul_1334 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_967: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1335, [2048, 1280]);  mul_1335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_545: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        mm_267: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_967, permute_545);  permute_545 = None
        mm_268: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_546, view_967);  permute_546 = None
        sum_401: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
        view_968: "f32[1280]" = torch.ops.aten.reshape.default(sum_401, [1280]);  sum_401 = None
        view_969: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_267, [4, 512, 5120]);  mm_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_34: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_10, [4, 512, 5120]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_32: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_34, 0.5)
        mul_1336: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_969, mul_32);  mul_32 = None
        pow_3: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 3.0)
        mul_33: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_25: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_34, mul_33);  mul_33 = None
        mul_34: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_34);  mul_34 = None
        add_26: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_1337: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_969, add_26);  view_969 = add_26 = None
        mul_1338: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_309: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1338);  mul_1338 = None
        mul_1339: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1336, sub_309);  mul_1336 = sub_309 = None
        mul_1340: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1339, 0.7978845608028654);  mul_1339 = None
        mul_1341: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1340, 0.044715)
        pow_70: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_34, 2.0);  view_34 = None
        mul_1342: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_70, 3.0);  pow_70 = None
        mul_1343: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1341, mul_1342);  mul_1341 = mul_1342 = None
        add_427: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1340, mul_1343);  mul_1340 = mul_1343 = None
        mul_1344: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1337, 0.5);  mul_1337 = None
        add_428: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_427, mul_1344);  add_427 = mul_1344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_970: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_428, [2048, 5120]);  add_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_547: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        mm_269: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_970, permute_547);  permute_547 = None
        mm_270: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_548, view_970);  permute_548 = None
        sum_402: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_970, [0], True);  view_970 = None
        view_971: "f32[5120]" = torch.ops.aten.reshape.default(sum_402, [5120]);  sum_402 = None
        view_972: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_269, [4, 512, 1280]);  mm_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1346: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_972, primals_34);  primals_34 = None
        mul_1347: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1346, 1280)
        sum_403: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1346, [2], True)
        mul_1348: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1346, mul_30);  mul_1346 = None
        sum_404: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1348, [2], True);  mul_1348 = None
        mul_1349: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_30, sum_404);  sum_404 = None
        sub_311: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1347, sum_403);  mul_1347 = sum_403 = None
        sub_312: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_311, mul_1349);  sub_311 = mul_1349 = None
        mul_1350: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_67, sub_312);  div_67 = sub_312 = None
        mul_1351: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_972, mul_30);  mul_30 = None
        sum_405: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1351, [0, 1]);  mul_1351 = None
        sum_406: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_972, [0, 1]);  view_972 = None
        add_429: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_426, mul_1350);  add_426 = mul_1350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_67: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_1352: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_1353: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_429, mul_1352);  mul_1352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_973: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1353, [2048, 1280]);  mul_1353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_549: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        mm_271: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_973, permute_549);  permute_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_11: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_27, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_30: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_11, [4, 512, -1]);  permute_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_31: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_30, [-1, 1280]);  view_30 = None
        permute_550: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_31, [1, 0]);  view_31 = None
        mm_272: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_550, view_973);  permute_550 = None
        sum_407: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_973, [0], True);  view_973 = None
        view_974: "f32[1280]" = torch.ops.aten.reshape.default(sum_407, [1280]);  sum_407 = None
        view_975: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_271, [4, 512, 1280]);  mm_271 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_976: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_975, [4, 512, 20, 64]);  view_975 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_551: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_976, [0, 2, 1, 3]);  view_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_33 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_551, permute_10, permute_8, permute_9, expand_2, getitem_27, getitem_28, getitem_29, getitem_30, 0.1, [True, True, True, False], scale = 0.125);  permute_551 = permute_10 = permute_8 = permute_9 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = None
        getitem_530: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_33[0]
        getitem_531: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_33[1]
        getitem_532: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_33[2];  _scaled_dot_product_efficient_attention_backward_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_552: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_530, [0, 2, 1, 3]);  getitem_530 = None
        view_977: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_552, [4, 512, 1280]);  permute_552 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_553: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_532, [0, 2, 1, 3]);  getitem_532 = None
        view_978: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_553, [4, 512, 1280]);  permute_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_554: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_531, [0, 2, 1, 3]);  getitem_531 = None
        view_979: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_554, [4, 512, 1280]);  permute_554 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_34: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_977, view_979, view_978], 2);  view_977 = view_979 = view_978 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_980: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_34, [2048, 3840]);  cat_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_555: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_31, [1, 0]);  primals_31 = None
        mm_273: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_980, permute_555);  permute_555 = None
        mm_274: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_556, view_980);  permute_556 = None
        sum_408: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_980, [0], True);  view_980 = None
        view_981: "f32[3840]" = torch.ops.aten.reshape.default(sum_408, [3840]);  sum_408 = None
        view_982: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_273, [4, 512, 1280]);  mm_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1355: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_982, primals_28);  primals_28 = None
        mul_1356: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1355, 1280)
        sum_409: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1355, [2], True)
        mul_1357: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1355, mul_26);  mul_1355 = None
        sum_410: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1357, [2], True);  mul_1357 = None
        mul_1358: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_26, sum_410);  sum_410 = None
        sub_314: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1356, sum_409);  mul_1356 = sum_409 = None
        sub_315: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_314, mul_1358);  sub_314 = mul_1358 = None
        mul_1359: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_68, sub_315);  div_68 = sub_315 = None
        mul_1360: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_982, mul_26);  mul_26 = None
        sum_411: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1360, [0, 1]);  mul_1360 = None
        sum_412: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_982, [0, 1]);  view_982 = None
        add_430: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_429, mul_1359);  add_429 = mul_1359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_68: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_1361: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_1362: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_430, mul_1361);  mul_1361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_983: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1362, [2048, 1280]);  mul_1362 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_557: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_27, [1, 0]);  primals_27 = None
        mm_275: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_983, permute_557);  permute_557 = None
        mm_276: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_558, view_983);  permute_558 = None
        sum_413: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_983, [0], True);  view_983 = None
        view_984: "f32[1280]" = torch.ops.aten.reshape.default(sum_413, [1280]);  sum_413 = None
        view_985: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_275, [4, 512, 5120]);  mm_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_22: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_6, [4, 512, 5120]);  addmm_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_20: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_22, 0.5)
        mul_1363: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_985, mul_20);  mul_20 = None
        pow_2: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 3.0)
        mul_21: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_2, 0.044715);  pow_2 = None
        add_17: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_22, mul_21);  mul_21 = None
        mul_22: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_17, 0.7978845608028654);  add_17 = None
        tanh_1: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_22);  mul_22 = None
        add_18: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_1364: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_985, add_18);  view_985 = add_18 = None
        mul_1365: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_316: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1365);  mul_1365 = None
        mul_1366: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1363, sub_316);  mul_1363 = sub_316 = None
        mul_1367: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1366, 0.7978845608028654);  mul_1366 = None
        mul_1368: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1367, 0.044715)
        pow_71: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_22, 2.0);  view_22 = None
        mul_1369: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_71, 3.0);  pow_71 = None
        mul_1370: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1368, mul_1369);  mul_1368 = mul_1369 = None
        add_431: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1367, mul_1370);  mul_1367 = mul_1370 = None
        mul_1371: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1364, 0.5);  mul_1364 = None
        add_432: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_431, mul_1371);  add_431 = mul_1371 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_986: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_432, [2048, 5120]);  add_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_559: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        mm_277: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_986, permute_559);  permute_559 = None
        mm_278: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_560, view_986);  permute_560 = None
        sum_414: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
        view_987: "f32[5120]" = torch.ops.aten.reshape.default(sum_414, [5120]);  sum_414 = None
        view_988: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_277, [4, 512, 1280]);  mm_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1373: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_988, primals_22);  primals_22 = None
        mul_1374: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1373, 1280)
        sum_415: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1373, [2], True)
        mul_1375: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1373, mul_18);  mul_1373 = None
        sum_416: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1375, [2], True);  mul_1375 = None
        mul_1376: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_18, sum_416);  sum_416 = None
        sub_318: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1374, sum_415);  mul_1374 = sum_415 = None
        sub_319: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_318, mul_1376);  sub_318 = mul_1376 = None
        mul_1377: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_69, sub_319);  div_69 = sub_319 = None
        mul_1378: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_988, mul_18);  mul_18 = None
        sum_417: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1378, [0, 1]);  mul_1378 = None
        sum_418: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_988, [0, 1]);  view_988 = None
        add_433: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_430, mul_1377);  add_430 = mul_1377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_69: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_1379: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_1380: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_433, mul_1379);  mul_1379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_989: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1380, [2048, 1280]);  mul_1380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_561: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        mm_279: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_989, permute_561);  permute_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_16, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_18: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_7, [4, 512, -1]);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_19: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_18, [-1, 1280]);  view_18 = None
        permute_562: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_19, [1, 0]);  view_19 = None
        mm_280: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_562, view_989);  permute_562 = None
        sum_419: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
        view_990: "f32[1280]" = torch.ops.aten.reshape.default(sum_419, [1280]);  sum_419 = None
        view_991: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_279, [4, 512, 1280]);  mm_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_992: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_991, [4, 512, 20, 64]);  view_991 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_563: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_992, [0, 2, 1, 3]);  view_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_34 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_563, permute_6, permute_4, permute_5, expand_2, getitem_16, getitem_17, getitem_18, getitem_19, 0.1, [True, True, True, False], scale = 0.125);  permute_563 = permute_6 = permute_4 = permute_5 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = None
        getitem_534: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_34[0]
        getitem_535: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_34[1]
        getitem_536: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_34[2];  _scaled_dot_product_efficient_attention_backward_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_564: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_534, [0, 2, 1, 3]);  getitem_534 = None
        view_993: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_564, [4, 512, 1280]);  permute_564 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_565: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_536, [0, 2, 1, 3]);  getitem_536 = None
        view_994: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_565, [4, 512, 1280]);  permute_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_566: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_535, [0, 2, 1, 3]);  getitem_535 = None
        view_995: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_566, [4, 512, 1280]);  permute_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_35: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_993, view_995, view_994], 2);  view_993 = view_995 = view_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_996: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_35, [2048, 3840]);  cat_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_567: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        mm_281: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_996, permute_567);  permute_567 = None
        mm_282: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_568, view_996);  permute_568 = None
        sum_420: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_996, [0], True);  view_996 = None
        view_997: "f32[3840]" = torch.ops.aten.reshape.default(sum_420, [3840]);  sum_420 = None
        view_998: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_281, [4, 512, 1280]);  mm_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1382: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_998, primals_16);  primals_16 = None
        mul_1383: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1382, 1280)
        sum_421: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1382, [2], True)
        mul_1384: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1382, mul_14);  mul_1382 = None
        sum_422: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1384, [2], True);  mul_1384 = None
        mul_1385: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_14, sum_422);  sum_422 = None
        sub_321: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1383, sum_421);  mul_1383 = sum_421 = None
        sub_322: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_321, mul_1385);  sub_321 = mul_1385 = None
        mul_1386: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_70, sub_322);  div_70 = sub_322 = None
        mul_1387: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_998, mul_14);  mul_14 = None
        sum_423: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1387, [0, 1]);  mul_1387 = None
        sum_424: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_998, [0, 1]);  view_998 = None
        add_434: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_433, mul_1386);  add_433 = mul_1386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:242 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_70: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_1388: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_1389: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_434, mul_1388);  mul_1388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_999: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1389, [2048, 1280]);  mul_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_569: "f32[1280, 5120]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        mm_283: "f32[2048, 5120]" = torch.ops.aten.mm.default(view_999, permute_569);  permute_569 = None
        mm_284: "f32[5120, 1280]" = torch.ops.aten.mm.default(permute_570, view_999);  permute_570 = None
        sum_425: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_999, [0], True);  view_999 = None
        view_1000: "f32[1280]" = torch.ops.aten.reshape.default(sum_425, [1280]);  sum_425 = None
        view_1001: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(mm_283, [4, 512, 5120]);  mm_283 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_10: "f32[4, 512, 5120]" = torch.ops.aten.reshape.default(addmm_2, [4, 512, 5120]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_8: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_10, 0.5)
        mul_1390: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_1001, mul_8);  mul_8 = None
        pow_1: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 3.0)
        mul_9: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(pow_1, 0.044715);  pow_1 = None
        add_9: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(view_10, mul_9);  mul_9 = None
        mul_10: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(add_9, 0.7978845608028654);  add_9 = None
        tanh: "f32[4, 512, 5120]" = torch.ops.aten.tanh.default(mul_10);  mul_10 = None
        add_10: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_1391: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(view_1001, add_10);  view_1001 = add_10 = None
        mul_1392: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_323: "f32[4, 512, 5120]" = torch.ops.aten.sub.Tensor(1, mul_1392);  mul_1392 = None
        mul_1393: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1390, sub_323);  mul_1390 = sub_323 = None
        mul_1394: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1393, 0.7978845608028654);  mul_1393 = None
        mul_1395: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1394, 0.044715)
        pow_72: "f32[4, 512, 5120]" = torch.ops.aten.pow.Tensor_Scalar(view_10, 2.0);  view_10 = None
        mul_1396: "f32[4, 512, 5120]" = torch.ops.aten.mul.Scalar(pow_72, 3.0);  pow_72 = None
        mul_1397: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1395, mul_1396);  mul_1395 = mul_1396 = None
        add_435: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(mul_1394, mul_1397);  mul_1394 = mul_1397 = None
        mul_1398: "f32[4, 512, 5120]" = torch.ops.aten.mul.Tensor(mul_1391, 0.5);  mul_1391 = None
        add_436: "f32[4, 512, 5120]" = torch.ops.aten.add.Tensor(add_435, mul_1398);  add_435 = mul_1398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_1002: "f32[2048, 5120]" = torch.ops.aten.reshape.default(add_436, [2048, 5120]);  add_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_571: "f32[5120, 1280]" = torch.ops.aten.permute.default(primals_13, [1, 0]);  primals_13 = None
        mm_285: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_1002, permute_571);  permute_571 = None
        mm_286: "f32[1280, 5120]" = torch.ops.aten.mm.default(permute_572, view_1002);  permute_572 = None
        sum_426: "f32[1, 5120]" = torch.ops.aten.sum.dim_IntList(view_1002, [0], True);  view_1002 = None
        view_1003: "f32[5120]" = torch.ops.aten.reshape.default(sum_426, [5120]);  sum_426 = None
        view_1004: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_285, [4, 512, 1280]);  mm_285 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:304 in forward, code: hidden_states = self.ln_2(hidden_states)
        mul_1400: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_1004, primals_10);  primals_10 = None
        mul_1401: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1400, 1280)
        sum_427: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1400, [2], True)
        mul_1402: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1400, mul_6);  mul_1400 = None
        sum_428: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1402, [2], True);  mul_1402 = None
        mul_1403: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_6, sum_428);  sum_428 = None
        sub_325: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1401, sum_427);  mul_1401 = sum_427 = None
        sub_326: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_325, mul_1403);  sub_325 = mul_1403 = None
        mul_1404: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_71, sub_326);  div_71 = sub_326 = None
        mul_1405: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_1004, mul_6);  mul_6 = None
        sum_429: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1405, [0, 1]);  mul_1405 = None
        sum_430: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_1004, [0, 1]);  view_1004 = None
        add_437: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_434, mul_1404);  add_434 = mul_1404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:224 in forward, code: attn_output = self.resid_dropout(attn_output)
        convert_element_type_71: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt_1, torch.float32);  gt_1 = None
        mul_1406: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_1407: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_437, mul_1406);  mul_1406 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_1005: "f32[2048, 1280]" = torch.ops.aten.reshape.default(mul_1407, [2048, 1280]);  mul_1407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_573: "f32[1280, 1280]" = torch.ops.aten.permute.default(primals_9, [1, 0]);  primals_9 = None
        mm_287: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_1005, permute_573);  permute_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_3: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_5, [0, 2, 1, 3])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_6: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_3, [4, 512, -1]);  permute_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        view_7: "f32[2048, 1280]" = torch.ops.aten.reshape.default(view_6, [-1, 1280]);  view_6 = None
        permute_574: "f32[1280, 2048]" = torch.ops.aten.permute.default(view_7, [1, 0]);  view_7 = None
        mm_288: "f32[1280, 1280]" = torch.ops.aten.mm.default(permute_574, view_1005);  permute_574 = None
        sum_431: "f32[1, 1280]" = torch.ops.aten.sum.dim_IntList(view_1005, [0], True);  view_1005 = None
        view_1006: "f32[1280]" = torch.ops.aten.reshape.default(sum_431, [1280]);  sum_431 = None
        view_1007: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_287, [4, 512, 1280]);  mm_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:222 in forward, code: attn_output = attn_output.reshape(*attn_output.shape[:-2], -1).contiguous()
        view_1008: "f32[4, 512, 20, 64]" = torch.ops.aten.reshape.default(view_1007, [4, 512, 20, 64]);  view_1007 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_575: "f32[4, 20, 512, 64]" = torch.ops.aten.permute.default(view_1008, [0, 2, 1, 3]);  view_1008 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        _scaled_dot_product_efficient_attention_backward_35 = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default(permute_575, permute_2, permute, permute_1, expand_2, getitem_5, getitem_6, getitem_7, getitem_8, 0.1, [True, True, True, False], scale = 0.125);  permute_575 = permute_2 = permute = permute_1 = expand_2 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = None
        getitem_538: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_35[0]
        getitem_539: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_35[1]
        getitem_540: "f32[4, 20, 512, 64]" = _scaled_dot_product_efficient_attention_backward_35[2];  _scaled_dot_product_efficient_attention_backward_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:191 in forward, code: query_states = query_states.view(shape_q).transpose(1, 2)
        permute_576: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_538, [0, 2, 1, 3]);  getitem_538 = None
        view_1009: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_576, [4, 512, 1280]);  permute_576 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:188 in forward, code: value_states = value_states.view(shape_kv).transpose(1, 2)
        permute_577: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_540, [0, 2, 1, 3]);  getitem_540 = None
        view_1010: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_577, [4, 512, 1280]);  permute_577 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:187 in forward, code: key_states = key_states.view(shape_kv).transpose(1, 2)
        permute_578: "f32[4, 512, 20, 64]" = torch.ops.aten.permute.default(getitem_539, [0, 2, 1, 3]);  getitem_539 = None
        view_1011: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(permute_578, [4, 512, 1280]);  permute_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:185 in forward, code: query_states, key_states, value_states = self.c_attn(hidden_states).split(self.split_size, dim=2)
        cat_36: "f32[4, 512, 3840]" = torch.ops.aten.cat.default([view_1009, view_1011, view_1010], 2);  view_1009 = view_1011 = view_1010 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:122 in forward, code: x = x.view(size_out)
        view_1012: "f32[2048, 3840]" = torch.ops.aten.reshape.default(cat_36, [2048, 3840]);  cat_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/pytorch_utils.py:121 in forward, code: x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        permute_579: "f32[3840, 1280]" = torch.ops.aten.permute.default(primals_7, [1, 0]);  primals_7 = None
        mm_289: "f32[2048, 1280]" = torch.ops.aten.mm.default(view_1012, permute_579);  permute_579 = None
        mm_290: "f32[1280, 3840]" = torch.ops.aten.mm.default(permute_580, view_1012);  permute_580 = None
        sum_432: "f32[1, 3840]" = torch.ops.aten.sum.dim_IntList(view_1012, [0], True);  view_1012 = None
        view_1013: "f32[3840]" = torch.ops.aten.reshape.default(sum_432, [3840]);  sum_432 = None
        view_1014: "f32[4, 512, 1280]" = torch.ops.aten.reshape.default(mm_289, [4, 512, 1280]);  mm_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:273 in forward, code: hidden_states = self.ln_1(hidden_states)
        mul_1409: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_1014, primals_4);  primals_4 = None
        mul_1410: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1409, 1280)
        sum_433: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1409, [2], True)
        mul_1411: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_1409, mul_2);  mul_1409 = None
        sum_434: "f32[4, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_1411, [2], True);  mul_1411 = None
        mul_1412: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(mul_2, sum_434);  sum_434 = None
        sub_328: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(mul_1410, sum_433);  mul_1410 = sum_433 = None
        sub_329: "f32[4, 512, 1280]" = torch.ops.aten.sub.Tensor(sub_328, mul_1412);  sub_328 = mul_1412 = None
        mul_1413: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(div_72, sub_329);  div_72 = sub_329 = None
        mul_1414: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(view_1014, mul_2);  mul_2 = None
        sum_435: "f32[1280]" = torch.ops.aten.sum.dim_IntList(mul_1414, [0, 1]);  mul_1414 = None
        sum_436: "f32[1280]" = torch.ops.aten.sum.dim_IntList(view_1014, [0, 1]);  view_1014 = None
        add_438: "f32[4, 512, 1280]" = torch.ops.aten.add.Tensor(add_437, mul_1413);  add_437 = mul_1413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:612 in forward, code: hidden_states = self.drop(hidden_states)
        convert_element_type_72: "f32[4, 512, 1280]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1415: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
        mul_1416: "f32[4, 512, 1280]" = torch.ops.aten.mul.Tensor(add_438, mul_1415);  add_438 = mul_1415 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:585 in forward, code: hidden_states = inputs_embeds + position_embeds.to(inputs_embeds.device)
        sum_437: "f32[1, 512, 1280]" = torch.ops.aten.sum.dim_IntList(mul_1416, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        eq_1: "b8[1, 512]" = torch.ops.aten.eq.Scalar(unsqueeze, -1)
        unsqueeze_10: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_2: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:584 in forward, code: position_embeds = self.wpe(position_ids)
        where_36: "f32[1, 512, 1280]" = torch.ops.aten.where.self(unsqueeze_10, full_default_2, sum_437);  unsqueeze_10 = sum_437 = None
        full_default_74: "f32[1024, 1280]" = torch.ops.aten.full.default([1024, 1280], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[1024, 1280]" = torch.ops.aten.index_put.default(full_default_74, [unsqueeze], where_36, True);  full_default_74 = unsqueeze = where_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/gpt2/modeling_gpt2.py:577 in forward, code: inputs_embeds = self.wte(input_ids)
        eq_2: "b8[4, 512]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_11: "b8[4, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_37: "f32[4, 512, 1280]" = torch.ops.aten.where.self(unsqueeze_11, full_default_2, mul_1416);  unsqueeze_11 = full_default_2 = mul_1416 = None
        full_default_76: "f32[50257, 1280]" = torch.ops.aten.full.default([50257, 1280], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[50257, 1280]" = torch.ops.aten.index_put.default(full_default_76, [primals_1], where_37, True);  full_default_76 = primals_1 = where_37 = None
        add_439: "f32[50257, 1280]" = torch.ops.aten.add.Tensor(mm_1, index_put_1);  mm_1 = index_put_1 = None
        return (None, add_439, index_put, sum_435, sum_436, view_1013, mm_290, view_1006, mm_288, sum_429, sum_430, view_1003, mm_286, view_1000, mm_284, sum_423, sum_424, view_997, mm_282, view_990, mm_280, sum_417, sum_418, view_987, mm_278, view_984, mm_276, sum_411, sum_412, view_981, mm_274, view_974, mm_272, sum_405, sum_406, view_971, mm_270, view_968, mm_268, sum_399, sum_400, view_965, mm_266, view_958, mm_264, sum_393, sum_394, view_955, mm_262, view_952, mm_260, sum_387, sum_388, view_949, mm_258, view_942, mm_256, sum_381, sum_382, view_939, mm_254, view_936, mm_252, sum_375, sum_376, view_933, mm_250, view_926, mm_248, sum_369, sum_370, view_923, mm_246, view_920, mm_244, sum_363, sum_364, view_917, mm_242, view_910, mm_240, sum_357, sum_358, view_907, mm_238, view_904, mm_236, sum_351, sum_352, view_901, mm_234, view_894, mm_232, sum_345, sum_346, view_891, mm_230, view_888, mm_228, sum_339, sum_340, view_885, mm_226, view_878, mm_224, sum_333, sum_334, view_875, mm_222, view_872, mm_220, sum_327, sum_328, view_869, mm_218, view_862, mm_216, sum_321, sum_322, view_859, mm_214, view_856, mm_212, sum_315, sum_316, view_853, mm_210, view_846, mm_208, sum_309, sum_310, view_843, mm_206, view_840, mm_204, sum_303, sum_304, view_837, mm_202, view_830, mm_200, sum_297, sum_298, view_827, mm_198, view_824, mm_196, sum_291, sum_292, view_821, mm_194, view_814, mm_192, sum_285, sum_286, view_811, mm_190, view_808, mm_188, sum_279, sum_280, view_805, mm_186, view_798, mm_184, sum_273, sum_274, view_795, mm_182, view_792, mm_180, sum_267, sum_268, view_789, mm_178, view_782, mm_176, sum_261, sum_262, view_779, mm_174, view_776, mm_172, sum_255, sum_256, view_773, mm_170, view_766, mm_168, sum_249, sum_250, view_763, mm_166, view_760, mm_164, sum_243, sum_244, view_757, mm_162, view_750, mm_160, sum_237, sum_238, view_747, mm_158, view_744, mm_156, sum_231, sum_232, view_741, mm_154, view_734, mm_152, sum_225, sum_226, view_731, mm_150, view_728, mm_148, sum_219, sum_220, view_725, mm_146, view_718, mm_144, sum_213, sum_214, view_715, mm_142, view_712, mm_140, sum_207, sum_208, view_709, mm_138, view_702, mm_136, sum_201, sum_202, view_699, mm_134, view_696, mm_132, sum_195, sum_196, view_693, mm_130, view_686, mm_128, sum_189, sum_190, view_683, mm_126, view_680, mm_124, sum_183, sum_184, view_677, mm_122, view_670, mm_120, sum_177, sum_178, view_667, mm_118, view_664, mm_116, sum_171, sum_172, view_661, mm_114, view_654, mm_112, sum_165, sum_166, view_651, mm_110, view_648, mm_108, sum_159, sum_160, view_645, mm_106, view_638, mm_104, sum_153, sum_154, view_635, mm_102, view_632, mm_100, sum_147, sum_148, view_629, mm_98, view_622, mm_96, sum_141, sum_142, view_619, mm_94, view_616, mm_92, sum_135, sum_136, view_613, mm_90, view_606, mm_88, sum_129, sum_130, view_603, mm_86, view_600, mm_84, sum_123, sum_124, view_597, mm_82, view_590, mm_80, sum_117, sum_118, view_587, mm_78, view_584, mm_76, sum_111, sum_112, view_581, mm_74, view_574, mm_72, sum_105, sum_106, view_571, mm_70, view_568, mm_68, sum_99, sum_100, view_565, mm_66, view_558, mm_64, sum_93, sum_94, view_555, mm_62, view_552, mm_60, sum_87, sum_88, view_549, mm_58, view_542, mm_56, sum_81, sum_82, view_539, mm_54, view_536, mm_52, sum_75, sum_76, view_533, mm_50, view_526, mm_48, sum_69, sum_70, view_523, mm_46, view_520, mm_44, sum_63, sum_64, view_517, mm_42, view_510, mm_40, sum_57, sum_58, view_507, mm_38, view_504, mm_36, sum_51, sum_52, view_501, mm_34, view_494, mm_32, sum_45, sum_46, view_491, mm_30, view_488, mm_28, sum_39, sum_40, view_485, mm_26, view_478, mm_24, sum_33, sum_34, view_475, mm_22, view_472, mm_20, sum_27, sum_28, view_469, mm_18, view_462, mm_16, sum_21, sum_22, view_459, mm_14, view_456, mm_12, sum_15, sum_16, view_453, mm_10, view_446, mm_8, sum_9, sum_10, view_443, mm_6, view_440, mm_4, sum_3, sum_4)

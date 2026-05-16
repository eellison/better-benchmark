"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g77
Pattern hash: df0821f1f304
Shape hash: 9d40ded4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_10: "f32[]", getitem_11: "f32[]", getitem_12: "f32[]", getitem_13: "f32[]", getitem_14: "f32[]", getitem_15: "f32[]", getitem_16: "f32[]", getitem_17: "f32[]", getitem_18: "f32[]", getitem_19: "f32[]", getitem_20: "f32[]", getitem_21: "f32[]", getitem_22: "f32[]", getitem_23: "f32[]", getitem_24: "f32[]", getitem_25: "f32[]", getitem_26: "f32[]", getitem_27: "f32[]", getitem_28: "f32[]", getitem_29: "f32[]", getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]", getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]", getitem_130: "f32[]", getitem_131: "f32[]", getitem_132: "f32[]", getitem_133: "f32[]", getitem_134: "f32[]", getitem_135: "f32[]", getitem_136: "f32[]", getitem_137: "f32[]", getitem_138: "f32[]", getitem_139: "f32[]", getitem_140: "f32[]", getitem_141: "f32[]", getitem_142: "f32[]", getitem_143: "f32[]", getitem_144: "f32[]", getitem_145: "f32[]", getitem_146: "f32[]", getitem_147: "f32[]", getitem_148: "f32[]", getitem_149: "f32[]", getitem_150: "f32[]", getitem_151: "f32[]", getitem_152: "f32[]", getitem_153: "f32[]", getitem_154: "f32[]", getitem_155: "f32[]", getitem_156: "f32[]", getitem_157: "f32[]", getitem_158: "f32[]", getitem_159: "f32[]", getitem_160: "f32[]", getitem_161: "f32[]", getitem_162: "f32[]", getitem_163: "f32[]", getitem_164: "f32[]", getitem_165: "f32[]", getitem_166: "f32[]", getitem_167: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167]);  getitem = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = None
        getitem_168: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_169: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_170: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_171: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_172: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_173: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_174: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_175: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_176: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_177: "f32[]" = _foreach_pow_scalar_and_tensor[9]
        getitem_178: "f32[]" = _foreach_pow_scalar_and_tensor[10]
        getitem_179: "f32[]" = _foreach_pow_scalar_and_tensor[11]
        getitem_180: "f32[]" = _foreach_pow_scalar_and_tensor[12]
        getitem_181: "f32[]" = _foreach_pow_scalar_and_tensor[13]
        getitem_182: "f32[]" = _foreach_pow_scalar_and_tensor[14]
        getitem_183: "f32[]" = _foreach_pow_scalar_and_tensor[15]
        getitem_184: "f32[]" = _foreach_pow_scalar_and_tensor[16]
        getitem_185: "f32[]" = _foreach_pow_scalar_and_tensor[17]
        getitem_186: "f32[]" = _foreach_pow_scalar_and_tensor[18]
        getitem_187: "f32[]" = _foreach_pow_scalar_and_tensor[19]
        getitem_188: "f32[]" = _foreach_pow_scalar_and_tensor[20]
        getitem_189: "f32[]" = _foreach_pow_scalar_and_tensor[21]
        getitem_190: "f32[]" = _foreach_pow_scalar_and_tensor[22]
        getitem_191: "f32[]" = _foreach_pow_scalar_and_tensor[23]
        getitem_192: "f32[]" = _foreach_pow_scalar_and_tensor[24]
        getitem_193: "f32[]" = _foreach_pow_scalar_and_tensor[25]
        getitem_194: "f32[]" = _foreach_pow_scalar_and_tensor[26]
        getitem_195: "f32[]" = _foreach_pow_scalar_and_tensor[27]
        getitem_196: "f32[]" = _foreach_pow_scalar_and_tensor[28]
        getitem_197: "f32[]" = _foreach_pow_scalar_and_tensor[29]
        getitem_198: "f32[]" = _foreach_pow_scalar_and_tensor[30]
        getitem_199: "f32[]" = _foreach_pow_scalar_and_tensor[31]
        getitem_200: "f32[]" = _foreach_pow_scalar_and_tensor[32]
        getitem_201: "f32[]" = _foreach_pow_scalar_and_tensor[33]
        getitem_202: "f32[]" = _foreach_pow_scalar_and_tensor[34]
        getitem_203: "f32[]" = _foreach_pow_scalar_and_tensor[35]
        getitem_204: "f32[]" = _foreach_pow_scalar_and_tensor[36]
        getitem_205: "f32[]" = _foreach_pow_scalar_and_tensor[37]
        getitem_206: "f32[]" = _foreach_pow_scalar_and_tensor[38]
        getitem_207: "f32[]" = _foreach_pow_scalar_and_tensor[39]
        getitem_208: "f32[]" = _foreach_pow_scalar_and_tensor[40]
        getitem_209: "f32[]" = _foreach_pow_scalar_and_tensor[41]
        getitem_210: "f32[]" = _foreach_pow_scalar_and_tensor[42]
        getitem_211: "f32[]" = _foreach_pow_scalar_and_tensor[43]
        getitem_212: "f32[]" = _foreach_pow_scalar_and_tensor[44]
        getitem_213: "f32[]" = _foreach_pow_scalar_and_tensor[45]
        getitem_214: "f32[]" = _foreach_pow_scalar_and_tensor[46]
        getitem_215: "f32[]" = _foreach_pow_scalar_and_tensor[47]
        getitem_216: "f32[]" = _foreach_pow_scalar_and_tensor[48]
        getitem_217: "f32[]" = _foreach_pow_scalar_and_tensor[49]
        getitem_218: "f32[]" = _foreach_pow_scalar_and_tensor[50]
        getitem_219: "f32[]" = _foreach_pow_scalar_and_tensor[51]
        getitem_220: "f32[]" = _foreach_pow_scalar_and_tensor[52]
        getitem_221: "f32[]" = _foreach_pow_scalar_and_tensor[53]
        getitem_222: "f32[]" = _foreach_pow_scalar_and_tensor[54]
        getitem_223: "f32[]" = _foreach_pow_scalar_and_tensor[55]
        getitem_224: "f32[]" = _foreach_pow_scalar_and_tensor[56]
        getitem_225: "f32[]" = _foreach_pow_scalar_and_tensor[57]
        getitem_226: "f32[]" = _foreach_pow_scalar_and_tensor[58]
        getitem_227: "f32[]" = _foreach_pow_scalar_and_tensor[59]
        getitem_228: "f32[]" = _foreach_pow_scalar_and_tensor[60]
        getitem_229: "f32[]" = _foreach_pow_scalar_and_tensor[61]
        getitem_230: "f32[]" = _foreach_pow_scalar_and_tensor[62]
        getitem_231: "f32[]" = _foreach_pow_scalar_and_tensor[63]
        getitem_232: "f32[]" = _foreach_pow_scalar_and_tensor[64]
        getitem_233: "f32[]" = _foreach_pow_scalar_and_tensor[65]
        getitem_234: "f32[]" = _foreach_pow_scalar_and_tensor[66]
        getitem_235: "f32[]" = _foreach_pow_scalar_and_tensor[67]
        getitem_236: "f32[]" = _foreach_pow_scalar_and_tensor[68]
        getitem_237: "f32[]" = _foreach_pow_scalar_and_tensor[69]
        getitem_238: "f32[]" = _foreach_pow_scalar_and_tensor[70]
        getitem_239: "f32[]" = _foreach_pow_scalar_and_tensor[71]
        getitem_240: "f32[]" = _foreach_pow_scalar_and_tensor[72]
        getitem_241: "f32[]" = _foreach_pow_scalar_and_tensor[73]
        getitem_242: "f32[]" = _foreach_pow_scalar_and_tensor[74]
        getitem_243: "f32[]" = _foreach_pow_scalar_and_tensor[75]
        getitem_244: "f32[]" = _foreach_pow_scalar_and_tensor[76]
        getitem_245: "f32[]" = _foreach_pow_scalar_and_tensor[77]
        getitem_246: "f32[]" = _foreach_pow_scalar_and_tensor[78]
        getitem_247: "f32[]" = _foreach_pow_scalar_and_tensor[79]
        getitem_248: "f32[]" = _foreach_pow_scalar_and_tensor[80]
        getitem_249: "f32[]" = _foreach_pow_scalar_and_tensor[81]
        getitem_250: "f32[]" = _foreach_pow_scalar_and_tensor[82]
        getitem_251: "f32[]" = _foreach_pow_scalar_and_tensor[83]
        getitem_252: "f32[]" = _foreach_pow_scalar_and_tensor[84]
        getitem_253: "f32[]" = _foreach_pow_scalar_and_tensor[85]
        getitem_254: "f32[]" = _foreach_pow_scalar_and_tensor[86]
        getitem_255: "f32[]" = _foreach_pow_scalar_and_tensor[87]
        getitem_256: "f32[]" = _foreach_pow_scalar_and_tensor[88]
        getitem_257: "f32[]" = _foreach_pow_scalar_and_tensor[89]
        getitem_258: "f32[]" = _foreach_pow_scalar_and_tensor[90]
        getitem_259: "f32[]" = _foreach_pow_scalar_and_tensor[91]
        getitem_260: "f32[]" = _foreach_pow_scalar_and_tensor[92]
        getitem_261: "f32[]" = _foreach_pow_scalar_and_tensor[93]
        getitem_262: "f32[]" = _foreach_pow_scalar_and_tensor[94]
        getitem_263: "f32[]" = _foreach_pow_scalar_and_tensor[95]
        getitem_264: "f32[]" = _foreach_pow_scalar_and_tensor[96]
        getitem_265: "f32[]" = _foreach_pow_scalar_and_tensor[97]
        getitem_266: "f32[]" = _foreach_pow_scalar_and_tensor[98]
        getitem_267: "f32[]" = _foreach_pow_scalar_and_tensor[99]
        getitem_268: "f32[]" = _foreach_pow_scalar_and_tensor[100]
        getitem_269: "f32[]" = _foreach_pow_scalar_and_tensor[101]
        getitem_270: "f32[]" = _foreach_pow_scalar_and_tensor[102]
        getitem_271: "f32[]" = _foreach_pow_scalar_and_tensor[103]
        getitem_272: "f32[]" = _foreach_pow_scalar_and_tensor[104]
        getitem_273: "f32[]" = _foreach_pow_scalar_and_tensor[105]
        getitem_274: "f32[]" = _foreach_pow_scalar_and_tensor[106]
        getitem_275: "f32[]" = _foreach_pow_scalar_and_tensor[107]
        getitem_276: "f32[]" = _foreach_pow_scalar_and_tensor[108]
        getitem_277: "f32[]" = _foreach_pow_scalar_and_tensor[109]
        getitem_278: "f32[]" = _foreach_pow_scalar_and_tensor[110]
        getitem_279: "f32[]" = _foreach_pow_scalar_and_tensor[111]
        getitem_280: "f32[]" = _foreach_pow_scalar_and_tensor[112]
        getitem_281: "f32[]" = _foreach_pow_scalar_and_tensor[113]
        getitem_282: "f32[]" = _foreach_pow_scalar_and_tensor[114]
        getitem_283: "f32[]" = _foreach_pow_scalar_and_tensor[115]
        getitem_284: "f32[]" = _foreach_pow_scalar_and_tensor[116]
        getitem_285: "f32[]" = _foreach_pow_scalar_and_tensor[117]
        getitem_286: "f32[]" = _foreach_pow_scalar_and_tensor[118]
        getitem_287: "f32[]" = _foreach_pow_scalar_and_tensor[119]
        getitem_288: "f32[]" = _foreach_pow_scalar_and_tensor[120]
        getitem_289: "f32[]" = _foreach_pow_scalar_and_tensor[121]
        getitem_290: "f32[]" = _foreach_pow_scalar_and_tensor[122]
        getitem_291: "f32[]" = _foreach_pow_scalar_and_tensor[123]
        getitem_292: "f32[]" = _foreach_pow_scalar_and_tensor[124]
        getitem_293: "f32[]" = _foreach_pow_scalar_and_tensor[125]
        getitem_294: "f32[]" = _foreach_pow_scalar_and_tensor[126]
        getitem_295: "f32[]" = _foreach_pow_scalar_and_tensor[127]
        getitem_296: "f32[]" = _foreach_pow_scalar_and_tensor[128]
        getitem_297: "f32[]" = _foreach_pow_scalar_and_tensor[129]
        getitem_298: "f32[]" = _foreach_pow_scalar_and_tensor[130]
        getitem_299: "f32[]" = _foreach_pow_scalar_and_tensor[131]
        getitem_300: "f32[]" = _foreach_pow_scalar_and_tensor[132]
        getitem_301: "f32[]" = _foreach_pow_scalar_and_tensor[133]
        getitem_302: "f32[]" = _foreach_pow_scalar_and_tensor[134]
        getitem_303: "f32[]" = _foreach_pow_scalar_and_tensor[135]
        getitem_304: "f32[]" = _foreach_pow_scalar_and_tensor[136]
        getitem_305: "f32[]" = _foreach_pow_scalar_and_tensor[137]
        getitem_306: "f32[]" = _foreach_pow_scalar_and_tensor[138]
        getitem_307: "f32[]" = _foreach_pow_scalar_and_tensor[139]
        getitem_308: "f32[]" = _foreach_pow_scalar_and_tensor[140]
        getitem_309: "f32[]" = _foreach_pow_scalar_and_tensor[141]
        getitem_310: "f32[]" = _foreach_pow_scalar_and_tensor[142]
        getitem_311: "f32[]" = _foreach_pow_scalar_and_tensor[143]
        getitem_312: "f32[]" = _foreach_pow_scalar_and_tensor[144]
        getitem_313: "f32[]" = _foreach_pow_scalar_and_tensor[145]
        getitem_314: "f32[]" = _foreach_pow_scalar_and_tensor[146]
        getitem_315: "f32[]" = _foreach_pow_scalar_and_tensor[147]
        getitem_316: "f32[]" = _foreach_pow_scalar_and_tensor[148]
        getitem_317: "f32[]" = _foreach_pow_scalar_and_tensor[149]
        getitem_318: "f32[]" = _foreach_pow_scalar_and_tensor[150]
        getitem_319: "f32[]" = _foreach_pow_scalar_and_tensor[151]
        getitem_320: "f32[]" = _foreach_pow_scalar_and_tensor[152]
        getitem_321: "f32[]" = _foreach_pow_scalar_and_tensor[153]
        getitem_322: "f32[]" = _foreach_pow_scalar_and_tensor[154]
        getitem_323: "f32[]" = _foreach_pow_scalar_and_tensor[155]
        getitem_324: "f32[]" = _foreach_pow_scalar_and_tensor[156]
        getitem_325: "f32[]" = _foreach_pow_scalar_and_tensor[157]
        getitem_326: "f32[]" = _foreach_pow_scalar_and_tensor[158]
        getitem_327: "f32[]" = _foreach_pow_scalar_and_tensor[159]
        getitem_328: "f32[]" = _foreach_pow_scalar_and_tensor[160]
        getitem_329: "f32[]" = _foreach_pow_scalar_and_tensor[161]
        getitem_330: "f32[]" = _foreach_pow_scalar_and_tensor[162]
        getitem_331: "f32[]" = _foreach_pow_scalar_and_tensor[163]
        getitem_332: "f32[]" = _foreach_pow_scalar_and_tensor[164]
        getitem_333: "f32[]" = _foreach_pow_scalar_and_tensor[165]
        getitem_334: "f32[]" = _foreach_pow_scalar_and_tensor[166]
        getitem_335: "f32[]" = _foreach_pow_scalar_and_tensor[167];  _foreach_pow_scalar_and_tensor = None
        return (getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

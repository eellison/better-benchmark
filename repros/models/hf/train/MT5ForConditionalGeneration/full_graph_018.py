import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "i64[32, 128]", primals_2: "f32[250112, 512]", primals_3: "f32[512]", primals_4: "f32[384, 512]", primals_5: "f32[384, 512]", primals_6: "f32[384, 512]", primals_8: "f32[512, 384]", primals_9: "f32[512]", primals_10: "f32[1024, 512]", primals_11: "f32[1024, 512]", primals_12: "f32[512, 1024]", primals_13: "f32[512]", primals_14: "f32[384, 512]", primals_15: "f32[384, 512]", primals_16: "f32[384, 512]", primals_17: "f32[512, 384]", primals_18: "f32[512]", primals_19: "f32[1024, 512]", primals_20: "f32[1024, 512]", primals_21: "f32[512, 1024]", primals_22: "f32[512]", primals_23: "f32[384, 512]", primals_24: "f32[384, 512]", primals_25: "f32[384, 512]", primals_26: "f32[512, 384]", primals_27: "f32[512]", primals_28: "f32[1024, 512]", primals_29: "f32[1024, 512]", primals_30: "f32[512, 1024]", primals_31: "f32[512]", primals_32: "f32[384, 512]", primals_33: "f32[384, 512]", primals_34: "f32[384, 512]", primals_35: "f32[512, 384]", primals_36: "f32[512]", primals_37: "f32[1024, 512]", primals_38: "f32[1024, 512]", primals_39: "f32[512, 1024]", primals_40: "f32[512]", primals_41: "f32[384, 512]", primals_42: "f32[384, 512]", primals_43: "f32[384, 512]", primals_44: "f32[512, 384]", primals_45: "f32[512]", primals_46: "f32[1024, 512]", primals_47: "f32[1024, 512]", primals_48: "f32[512, 1024]", primals_49: "f32[512]", primals_50: "f32[384, 512]", primals_51: "f32[384, 512]", primals_52: "f32[384, 512]", primals_53: "f32[512, 384]", primals_54: "f32[512]", primals_55: "f32[1024, 512]", primals_56: "f32[1024, 512]", primals_57: "f32[512, 1024]", primals_58: "f32[512]", primals_59: "f32[384, 512]", primals_60: "f32[384, 512]", primals_61: "f32[384, 512]", primals_62: "f32[512, 384]", primals_63: "f32[512]", primals_64: "f32[1024, 512]", primals_65: "f32[1024, 512]", primals_66: "f32[512, 1024]", primals_67: "f32[512]", primals_68: "f32[384, 512]", primals_69: "f32[384, 512]", primals_70: "f32[384, 512]", primals_71: "f32[512, 384]", primals_72: "f32[512]", primals_73: "f32[1024, 512]", primals_74: "f32[1024, 512]", primals_75: "f32[512, 1024]", primals_76: "f32[512]", primals_77: "i64[32, 128]", primals_78: "f32[512]", primals_79: "f32[384, 512]", primals_80: "f32[384, 512]", primals_81: "f32[384, 512]", primals_83: "f32[512, 384]", primals_84: "f32[512]", primals_85: "f32[384, 512]", primals_86: "f32[384, 512]", primals_87: "f32[384, 512]", primals_88: "f32[512, 384]", primals_89: "f32[512]", primals_90: "f32[1024, 512]", primals_91: "f32[1024, 512]", primals_92: "f32[512, 1024]", primals_93: "f32[512]", primals_94: "f32[384, 512]", primals_95: "f32[384, 512]", primals_96: "f32[384, 512]", primals_97: "f32[512, 384]", primals_98: "f32[512]", primals_99: "f32[384, 512]", primals_100: "f32[384, 512]", primals_101: "f32[384, 512]", primals_102: "f32[512, 384]", primals_103: "f32[512]", primals_104: "f32[1024, 512]", primals_105: "f32[1024, 512]", primals_106: "f32[512, 1024]", primals_107: "f32[512]", primals_108: "f32[384, 512]", primals_109: "f32[384, 512]", primals_110: "f32[384, 512]", primals_111: "f32[512, 384]", primals_112: "f32[512]", primals_113: "f32[384, 512]", primals_114: "f32[384, 512]", primals_115: "f32[384, 512]", primals_116: "f32[512, 384]", primals_117: "f32[512]", primals_118: "f32[1024, 512]", primals_119: "f32[1024, 512]", primals_120: "f32[512, 1024]", primals_121: "f32[512]", primals_122: "f32[384, 512]", primals_123: "f32[384, 512]", primals_124: "f32[384, 512]", primals_125: "f32[512, 384]", primals_126: "f32[512]", primals_127: "f32[384, 512]", primals_128: "f32[384, 512]", primals_129: "f32[384, 512]", primals_130: "f32[512, 384]", primals_131: "f32[512]", primals_132: "f32[1024, 512]", primals_133: "f32[1024, 512]", primals_134: "f32[512, 1024]", primals_135: "f32[512]", primals_136: "f32[384, 512]", primals_137: "f32[384, 512]", primals_138: "f32[384, 512]", primals_139: "f32[512, 384]", primals_140: "f32[512]", primals_141: "f32[384, 512]", primals_142: "f32[384, 512]", primals_143: "f32[384, 512]", primals_144: "f32[512, 384]", primals_145: "f32[512]", primals_146: "f32[1024, 512]", primals_147: "f32[1024, 512]", primals_148: "f32[512, 1024]", primals_149: "f32[512]", primals_150: "f32[384, 512]", primals_151: "f32[384, 512]", primals_152: "f32[384, 512]", primals_153: "f32[512, 384]", primals_154: "f32[512]", primals_155: "f32[384, 512]", primals_156: "f32[384, 512]", primals_157: "f32[384, 512]", primals_158: "f32[512, 384]", primals_159: "f32[512]", primals_160: "f32[1024, 512]", primals_161: "f32[1024, 512]", primals_162: "f32[512, 1024]", primals_163: "f32[512]", primals_164: "f32[384, 512]", primals_165: "f32[384, 512]", primals_166: "f32[384, 512]", primals_167: "f32[512, 384]", primals_168: "f32[512]", primals_169: "f32[384, 512]", primals_170: "f32[384, 512]", primals_171: "f32[384, 512]", primals_172: "f32[512, 384]", primals_173: "f32[512]", primals_174: "f32[1024, 512]", primals_175: "f32[1024, 512]", primals_176: "f32[512, 1024]", primals_177: "f32[512]", primals_178: "f32[384, 512]", primals_179: "f32[384, 512]", primals_180: "f32[384, 512]", primals_181: "f32[512, 384]", primals_182: "f32[512]", primals_183: "f32[384, 512]", primals_184: "f32[384, 512]", primals_185: "f32[384, 512]", primals_186: "f32[512, 384]", primals_187: "f32[512]", primals_188: "f32[1024, 512]", primals_189: "f32[1024, 512]", primals_190: "f32[512, 1024]", primals_191: "f32[512]", embedding: "f32[32, 128, 512]", ge: "b8[1, 1, 128, 1]", gt: "b8[32, 128, 512]", rsqrt: "f32[32, 128, 1]", view_1: "f32[4096, 512]", bmm: "f32[192, 128, 128]", add_6: "i64[128, 128]", embedding_1: "f32[128, 128, 6]", amax: "f32[32, 6, 128, 1]", sum_1: "f32[32, 6, 128, 1]", gt_2: "b8[32, 6, 128, 128]", view_20: "f32[4096, 384]", gt_3: "b8[32, 128, 512]", add_9: "f32[32, 128, 512]", rsqrt_1: "f32[32, 128, 1]", view_22: "f32[4096, 512]", mm_4: "f32[4096, 1024]", mm_5: "f32[4096, 1024]", gt_4: "b8[32, 128, 1024]", view_26: "f32[4096, 1024]", gt_5: "b8[32, 128, 512]", add_13: "f32[32, 128, 512]", rsqrt_2: "f32[32, 128, 1]", view_28: "f32[4096, 512]", div_3: "f32[32, 6, 128, 128]", gt_6: "b8[32, 6, 128, 128]", view_47: "f32[4096, 384]", gt_7: "b8[32, 128, 512]", add_16: "f32[32, 128, 512]", rsqrt_3: "f32[32, 128, 1]", view_49: "f32[4096, 512]", mm_11: "f32[4096, 1024]", mm_12: "f32[4096, 1024]", gt_8: "b8[32, 128, 1024]", view_53: "f32[4096, 1024]", gt_9: "b8[32, 128, 512]", add_20: "f32[32, 128, 512]", rsqrt_4: "f32[32, 128, 1]", view_55: "f32[4096, 512]", div_4: "f32[32, 6, 128, 128]", gt_10: "b8[32, 6, 128, 128]", view_74: "f32[4096, 384]", gt_11: "b8[32, 128, 512]", add_23: "f32[32, 128, 512]", rsqrt_5: "f32[32, 128, 1]", view_76: "f32[4096, 512]", mm_18: "f32[4096, 1024]", mm_19: "f32[4096, 1024]", gt_12: "b8[32, 128, 1024]", view_80: "f32[4096, 1024]", gt_13: "b8[32, 128, 512]", add_27: "f32[32, 128, 512]", rsqrt_6: "f32[32, 128, 1]", view_82: "f32[4096, 512]", div_5: "f32[32, 6, 128, 128]", gt_14: "b8[32, 6, 128, 128]", view_101: "f32[4096, 384]", gt_15: "b8[32, 128, 512]", add_30: "f32[32, 128, 512]", rsqrt_7: "f32[32, 128, 1]", view_103: "f32[4096, 512]", mm_25: "f32[4096, 1024]", mm_26: "f32[4096, 1024]", gt_16: "b8[32, 128, 1024]", view_107: "f32[4096, 1024]", gt_17: "b8[32, 128, 512]", add_34: "f32[32, 128, 512]", rsqrt_8: "f32[32, 128, 1]", view_109: "f32[4096, 512]", div_6: "f32[32, 6, 128, 128]", gt_18: "b8[32, 6, 128, 128]", view_128: "f32[4096, 384]", gt_19: "b8[32, 128, 512]", add_37: "f32[32, 128, 512]", rsqrt_9: "f32[32, 128, 1]", view_130: "f32[4096, 512]", mm_32: "f32[4096, 1024]", mm_33: "f32[4096, 1024]", gt_20: "b8[32, 128, 1024]", view_134: "f32[4096, 1024]", gt_21: "b8[32, 128, 512]", add_41: "f32[32, 128, 512]", rsqrt_10: "f32[32, 128, 1]", view_136: "f32[4096, 512]", div_7: "f32[32, 6, 128, 128]", gt_22: "b8[32, 6, 128, 128]", view_155: "f32[4096, 384]", gt_23: "b8[32, 128, 512]", add_44: "f32[32, 128, 512]", rsqrt_11: "f32[32, 128, 1]", view_157: "f32[4096, 512]", mm_39: "f32[4096, 1024]", mm_40: "f32[4096, 1024]", gt_24: "b8[32, 128, 1024]", view_161: "f32[4096, 1024]", gt_25: "b8[32, 128, 512]", add_48: "f32[32, 128, 512]", rsqrt_12: "f32[32, 128, 1]", view_163: "f32[4096, 512]", div_8: "f32[32, 6, 128, 128]", gt_26: "b8[32, 6, 128, 128]", view_182: "f32[4096, 384]", gt_27: "b8[32, 128, 512]", add_51: "f32[32, 128, 512]", rsqrt_13: "f32[32, 128, 1]", view_184: "f32[4096, 512]", mm_46: "f32[4096, 1024]", mm_47: "f32[4096, 1024]", gt_28: "b8[32, 128, 1024]", view_188: "f32[4096, 1024]", gt_29: "b8[32, 128, 512]", add_55: "f32[32, 128, 512]", rsqrt_14: "f32[32, 128, 1]", view_190: "f32[4096, 512]", div_9: "f32[32, 6, 128, 128]", gt_30: "b8[32, 6, 128, 128]", view_209: "f32[4096, 384]", gt_31: "b8[32, 128, 512]", add_58: "f32[32, 128, 512]", rsqrt_15: "f32[32, 128, 1]", view_211: "f32[4096, 512]", mm_53: "f32[4096, 1024]", mm_54: "f32[4096, 1024]", gt_32: "b8[32, 128, 1024]", view_215: "f32[4096, 1024]", gt_33: "b8[32, 128, 512]", add_62: "f32[32, 128, 512]", rsqrt_16: "f32[32, 128, 1]", gt_34: "b8[32, 128, 512]", mul_143: "f32[32, 128, 512]", gt_35: "b8[32, 128, 512]", rsqrt_17: "f32[32, 128, 1]", view_218: "f32[4096, 512]", add_71: "i64[128, 128]", div_12: "f32[32, 6, 128, 128]", gt_36: "b8[32, 6, 128, 128]", view_237: "f32[4096, 384]", gt_37: "b8[32, 128, 512]", add_74: "f32[32, 128, 512]", rsqrt_18: "f32[32, 128, 1]", view_239: "f32[4096, 512]", div_13: "f32[32, 6, 128, 128]", gt_38: "b8[32, 6, 128, 128]", view_258: "f32[4096, 384]", gt_39: "b8[32, 128, 512]", add_78: "f32[32, 128, 512]", rsqrt_19: "f32[32, 128, 1]", view_260: "f32[4096, 512]", mm_64: "f32[4096, 1024]", mm_65: "f32[4096, 1024]", gt_40: "b8[32, 128, 1024]", view_264: "f32[4096, 1024]", gt_41: "b8[32, 128, 512]", add_82: "f32[32, 128, 512]", rsqrt_20: "f32[32, 128, 1]", view_266: "f32[4096, 512]", div_14: "f32[32, 6, 128, 128]", gt_42: "b8[32, 6, 128, 128]", view_285: "f32[4096, 384]", gt_43: "b8[32, 128, 512]", add_85: "f32[32, 128, 512]", rsqrt_21: "f32[32, 128, 1]", view_287: "f32[4096, 512]", div_15: "f32[32, 6, 128, 128]", gt_44: "b8[32, 6, 128, 128]", view_306: "f32[4096, 384]", gt_45: "b8[32, 128, 512]", add_88: "f32[32, 128, 512]", rsqrt_22: "f32[32, 128, 1]", view_308: "f32[4096, 512]", mm_75: "f32[4096, 1024]", mm_76: "f32[4096, 1024]", gt_46: "b8[32, 128, 1024]", view_312: "f32[4096, 1024]", gt_47: "b8[32, 128, 512]", add_92: "f32[32, 128, 512]", rsqrt_23: "f32[32, 128, 1]", view_314: "f32[4096, 512]", div_16: "f32[32, 6, 128, 128]", gt_48: "b8[32, 6, 128, 128]", view_333: "f32[4096, 384]", gt_49: "b8[32, 128, 512]", add_95: "f32[32, 128, 512]", rsqrt_24: "f32[32, 128, 1]", view_335: "f32[4096, 512]", div_17: "f32[32, 6, 128, 128]", gt_50: "b8[32, 6, 128, 128]", view_354: "f32[4096, 384]", gt_51: "b8[32, 128, 512]", add_98: "f32[32, 128, 512]", rsqrt_25: "f32[32, 128, 1]", view_356: "f32[4096, 512]", mm_86: "f32[4096, 1024]", mm_87: "f32[4096, 1024]", gt_52: "b8[32, 128, 1024]", view_360: "f32[4096, 1024]", gt_53: "b8[32, 128, 512]", add_102: "f32[32, 128, 512]", rsqrt_26: "f32[32, 128, 1]", view_362: "f32[4096, 512]", div_18: "f32[32, 6, 128, 128]", gt_54: "b8[32, 6, 128, 128]", view_381: "f32[4096, 384]", gt_55: "b8[32, 128, 512]", add_105: "f32[32, 128, 512]", rsqrt_27: "f32[32, 128, 1]", view_383: "f32[4096, 512]", div_19: "f32[32, 6, 128, 128]", gt_56: "b8[32, 6, 128, 128]", view_402: "f32[4096, 384]", gt_57: "b8[32, 128, 512]", add_108: "f32[32, 128, 512]", rsqrt_28: "f32[32, 128, 1]", view_404: "f32[4096, 512]", mm_97: "f32[4096, 1024]", mm_98: "f32[4096, 1024]", gt_58: "b8[32, 128, 1024]", view_408: "f32[4096, 1024]", gt_59: "b8[32, 128, 512]", add_112: "f32[32, 128, 512]", rsqrt_29: "f32[32, 128, 1]", view_410: "f32[4096, 512]", div_20: "f32[32, 6, 128, 128]", gt_60: "b8[32, 6, 128, 128]", view_429: "f32[4096, 384]", gt_61: "b8[32, 128, 512]", add_115: "f32[32, 128, 512]", rsqrt_30: "f32[32, 128, 1]", view_431: "f32[4096, 512]", div_21: "f32[32, 6, 128, 128]", gt_62: "b8[32, 6, 128, 128]", view_450: "f32[4096, 384]", gt_63: "b8[32, 128, 512]", add_118: "f32[32, 128, 512]", rsqrt_31: "f32[32, 128, 1]", view_452: "f32[4096, 512]", mm_108: "f32[4096, 1024]", mm_109: "f32[4096, 1024]", gt_64: "b8[32, 128, 1024]", view_456: "f32[4096, 1024]", gt_65: "b8[32, 128, 512]", add_122: "f32[32, 128, 512]", rsqrt_32: "f32[32, 128, 1]", view_458: "f32[4096, 512]", div_22: "f32[32, 6, 128, 128]", gt_66: "b8[32, 6, 128, 128]", view_477: "f32[4096, 384]", gt_67: "b8[32, 128, 512]", add_125: "f32[32, 128, 512]", rsqrt_33: "f32[32, 128, 1]", view_479: "f32[4096, 512]", div_23: "f32[32, 6, 128, 128]", gt_68: "b8[32, 6, 128, 128]", view_498: "f32[4096, 384]", gt_69: "b8[32, 128, 512]", add_128: "f32[32, 128, 512]", rsqrt_34: "f32[32, 128, 1]", view_500: "f32[4096, 512]", mm_119: "f32[4096, 1024]", mm_120: "f32[4096, 1024]", gt_70: "b8[32, 128, 1024]", view_504: "f32[4096, 1024]", gt_71: "b8[32, 128, 512]", add_132: "f32[32, 128, 512]", rsqrt_35: "f32[32, 128, 1]", view_506: "f32[4096, 512]", div_24: "f32[32, 6, 128, 128]", gt_72: "b8[32, 6, 128, 128]", view_525: "f32[4096, 384]", gt_73: "b8[32, 128, 512]", add_135: "f32[32, 128, 512]", rsqrt_36: "f32[32, 128, 1]", view_527: "f32[4096, 512]", div_25: "f32[32, 6, 128, 128]", gt_74: "b8[32, 6, 128, 128]", view_546: "f32[4096, 384]", gt_75: "b8[32, 128, 512]", add_138: "f32[32, 128, 512]", rsqrt_37: "f32[32, 128, 1]", view_548: "f32[4096, 512]", mm_130: "f32[4096, 1024]", mm_131: "f32[4096, 1024]", gt_76: "b8[32, 128, 1024]", view_552: "f32[4096, 1024]", gt_77: "b8[32, 128, 512]", add_142: "f32[32, 128, 512]", rsqrt_38: "f32[32, 128, 1]", view_554: "f32[4096, 512]", div_26: "f32[32, 6, 128, 128]", gt_78: "b8[32, 6, 128, 128]", view_573: "f32[4096, 384]", gt_79: "b8[32, 128, 512]", add_145: "f32[32, 128, 512]", rsqrt_39: "f32[32, 128, 1]", view_575: "f32[4096, 512]", div_27: "f32[32, 6, 128, 128]", gt_80: "b8[32, 6, 128, 128]", view_594: "f32[4096, 384]", gt_81: "b8[32, 128, 512]", add_148: "f32[32, 128, 512]", rsqrt_40: "f32[32, 128, 1]", view_596: "f32[4096, 512]", mm_141: "f32[4096, 1024]", mm_142: "f32[4096, 1024]", gt_82: "b8[32, 128, 1024]", view_600: "f32[4096, 1024]", gt_83: "b8[32, 128, 512]", add_152: "f32[32, 128, 512]", rsqrt_41: "f32[32, 128, 1]", gt_84: "b8[32, 128, 512]", view_602: "f32[4096, 512]", view_603: "f32[32, 128, 250112]", amax_24: "f32[4096, 1]", log_2: "f32[4096, 1]", convert_element_type_5: "f32[]", permute_288: "f32[192, 128, 128]", permute_289: "f32[192, 64, 128]", permute_290: "f32[192, 64, 128]", permute_291: "f32[192, 128, 64]", permute_313: "f32[192, 128, 128]", permute_314: "f32[192, 64, 128]", permute_315: "f32[192, 64, 128]", permute_316: "f32[192, 128, 64]", permute_350: "f32[192, 128, 128]", permute_351: "f32[192, 64, 128]", permute_352: "f32[192, 64, 128]", permute_353: "f32[192, 128, 64]", permute_375: "f32[192, 128, 128]", permute_376: "f32[192, 64, 128]", permute_377: "f32[192, 64, 128]", permute_378: "f32[192, 128, 64]", permute_412: "f32[192, 128, 128]", permute_413: "f32[192, 64, 128]", permute_414: "f32[192, 64, 128]", permute_415: "f32[192, 128, 64]", permute_437: "f32[192, 128, 128]", permute_438: "f32[192, 64, 128]", permute_439: "f32[192, 64, 128]", permute_440: "f32[192, 128, 64]", permute_474: "f32[192, 128, 128]", permute_475: "f32[192, 64, 128]", permute_476: "f32[192, 64, 128]", permute_477: "f32[192, 128, 64]", permute_499: "f32[192, 128, 128]", permute_500: "f32[192, 64, 128]", permute_501: "f32[192, 64, 128]", permute_502: "f32[192, 128, 64]", permute_536: "f32[192, 128, 128]", permute_537: "f32[192, 64, 128]", permute_538: "f32[192, 64, 128]", permute_539: "f32[192, 128, 64]", permute_561: "f32[192, 128, 128]", permute_562: "f32[192, 64, 128]", permute_563: "f32[192, 64, 128]", permute_564: "f32[192, 128, 64]", permute_598: "f32[192, 128, 128]", permute_599: "f32[192, 64, 128]", permute_600: "f32[192, 64, 128]", permute_601: "f32[192, 128, 64]", permute_623: "f32[192, 128, 128]", permute_624: "f32[192, 64, 128]", permute_625: "f32[192, 64, 128]", permute_626: "f32[192, 128, 64]", permute_660: "f32[192, 128, 128]", permute_661: "f32[192, 64, 128]", permute_662: "f32[192, 64, 128]", permute_663: "f32[192, 128, 64]", permute_685: "f32[192, 128, 128]", permute_686: "f32[192, 64, 128]", permute_687: "f32[192, 64, 128]", permute_688: "f32[192, 128, 64]", permute_722: "f32[192, 128, 128]", permute_723: "f32[192, 64, 128]", permute_724: "f32[192, 64, 128]", permute_725: "f32[192, 128, 64]", permute_747: "f32[192, 128, 128]", permute_748: "f32[192, 64, 128]", permute_750: "f32[192, 64, 128]", permute_751: "f32[192, 128, 64]", permute_785: "f32[192, 128, 128]", permute_786: "f32[192, 64, 128]", permute_787: "f32[192, 64, 128]", permute_788: "f32[192, 128, 64]", permute_822: "f32[192, 128, 128]", permute_823: "f32[192, 64, 128]", permute_824: "f32[192, 64, 128]", permute_825: "f32[192, 128, 64]", permute_859: "f32[192, 128, 128]", permute_860: "f32[192, 64, 128]", permute_861: "f32[192, 64, 128]", permute_862: "f32[192, 128, 64]", permute_896: "f32[192, 128, 128]", permute_897: "f32[192, 64, 128]", permute_898: "f32[192, 64, 128]", permute_899: "f32[192, 128, 64]", permute_933: "f32[192, 128, 128]", permute_934: "f32[192, 64, 128]", permute_935: "f32[192, 64, 128]", permute_936: "f32[192, 128, 64]", permute_970: "f32[192, 128, 128]", permute_971: "f32[192, 64, 128]", permute_972: "f32[192, 64, 128]", permute_973: "f32[192, 128, 64]", permute_1007: "f32[192, 128, 128]", permute_1008: "f32[192, 64, 128]", permute_1009: "f32[192, 64, 128]", permute_1010: "f32[192, 128, 64]", permute_1044: "f32[192, 128, 128]", permute_1045: "f32[192, 64, 128]", permute_1047: "f32[192, 64, 128]", permute_1048: "f32[192, 128, 64]", tangents_1: "f32[]", tangents_2: "f32[32, 128, 250112]", tangents_3: "f32[32, 128, 512]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        div_29: "f32[]" = torch.ops.aten.div.Tensor(tangents_1, convert_element_type_5);  tangents_1 = convert_element_type_5 = None
        view_605: "i64[4096]" = torch.ops.aten.reshape.default(primals_77, [-1]);  primals_77 = None
        unsqueeze_19: "i64[4096, 1]" = torch.ops.aten.unsqueeze.default(view_605, 1);  view_605 = None
        ne_3: "b8[4096, 1]" = torch.ops.aten.ne.Scalar(unsqueeze_19, -100)
        full_default_10: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "i64[4096, 1]" = torch.ops.aten.where.self(ne_3, unsqueeze_19, full_default_10);  unsqueeze_19 = full_default_10 = None

        # No stacktrace found for following nodes
        iota_default: "i64[250112]" = torch.ops.prims.iota.default(250112, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        view_default: "i64[1, 250112]" = torch.ops.aten.reshape.default(iota_default, [1, 250112]);  iota_default = None
        expand_default: "i64[4096, 250112]" = torch.ops.aten.expand.default(where_7, [4096, 250112]);  where_7 = None
        eq_tensor: "b8[4096, 250112]" = torch.ops.aten.eq.Tensor(expand_default, view_default);  expand_default = view_default = None
        scalar_tensor_default: "f32[]" = torch.ops.aten.scalar_tensor.default(0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))
        scalar_tensor_default_1: "f32[]" = torch.ops.aten.scalar_tensor.default(-1.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0))

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_self: "f32[4096, 250112]" = torch.ops.aten.where.self(eq_tensor, scalar_tensor_default_1, scalar_tensor_default);  eq_tensor = scalar_tensor_default_1 = scalar_tensor_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1150 in forward, code: loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
        where_8: "f32[4096, 1]" = torch.ops.aten.where.self(ne_3, div_29, full_default);  ne_3 = div_29 = None
        mul_335: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(where_self, where_8);  where_self = where_8 = None
        view_604: "f32[4096, 250112]" = torch.ops.aten.reshape.default(view_603, [-1, 250112]);  view_603 = None
        sub_26: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(view_604, amax_24);  view_604 = amax_24 = None
        sub_27: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(sub_26, log_2);  sub_26 = log_2 = None
        exp_25: "f32[4096, 250112]" = torch.ops.aten.exp.default(sub_27);  sub_27 = None
        sum_28: "f32[4096, 1]" = torch.ops.aten.sum.dim_IntList(mul_335, [1], True)
        mul_336: "f32[4096, 250112]" = torch.ops.aten.mul.Tensor(exp_25, sum_28);  exp_25 = sum_28 = None
        sub_28: "f32[4096, 250112]" = torch.ops.aten.sub.Tensor(mul_335, mul_336);  mul_335 = mul_336 = None
        view_606: "f32[32, 128, 250112]" = torch.ops.aten.reshape.default(sub_28, [32, 128, 250112]);  sub_28 = None
        add_154: "f32[32, 128, 250112]" = torch.ops.aten.add.Tensor(tangents_2, view_606);  tangents_2 = view_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:1143 in forward, code: lm_logits = self.lm_head(sequence_output)
        view_607: "f32[4096, 250112]" = torch.ops.aten.reshape.default(add_154, [4096, 250112]);  add_154 = None
        permute_267: "f32[250112, 4096]" = torch.ops.aten.permute.default(view_607, [1, 0])
        mm_145: "f32[250112, 512]" = torch.ops.aten.mm.default(permute_267, view_602);  permute_267 = view_602 = None
        permute_266: "f32[512, 250112]" = torch.ops.aten.permute.default(primals_2, [1, 0]);  primals_2 = None
        permute_269: "f32[250112, 512]" = torch.ops.aten.permute.default(permute_266, [1, 0]);  permute_266 = None
        mm_146: "f32[4096, 512]" = torch.ops.aten.mm.default(view_607, permute_269);  view_607 = permute_269 = None
        view_608: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_146, [32, 128, 512]);  mm_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:766 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_6: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_84, torch.float32);  gt_84 = None
        mul_337: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_6, 1.1111111111111112);  convert_element_type_6 = None
        mul_338: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_608, mul_337);  view_608 = mul_337 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_339: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_338, primals_191);  primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_331: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_152, rsqrt_41)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_340: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_338, mul_331);  mul_338 = mul_331 = None
        sum_29: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_340, [0, 1], True);  mul_340 = None
        view_609: "f32[512]" = torch.ops.aten.reshape.default(sum_29, [512]);  sum_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_341: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_339, add_152)
        mul_342: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_339, rsqrt_41);  mul_339 = None
        sum_30: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_341, [2], True);  mul_341 = None
        pow_59: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_41, 3);  rsqrt_41 = None
        mul_343: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_30, -0.5);  sum_30 = None
        mul_344: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_343, pow_59);  mul_343 = pow_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_99: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_344, [32, 128, 512]);  mul_344 = None
        div_30: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_99, 512);  expand_99 = None
        pow_60: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_152, 1.0);  add_152 = None
        mul_345: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_60, 2.0);  pow_60 = None
        mul_346: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_30, mul_345);  div_30 = mul_345 = None
        add_155: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_342, mul_346);  mul_342 = mul_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_7: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_83, torch.float32);  gt_83 = None
        mul_347: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_7, 1.1111111111111112);  convert_element_type_7 = None
        mul_348: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_155, mul_347);  mul_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_610: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_348, [4096, 512]);  mul_348 = None
        permute_271: "f32[512, 4096]" = torch.ops.aten.permute.default(view_610, [1, 0])
        mm_147: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_271, view_600);  permute_271 = view_600 = None
        permute_265: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_190, [1, 0]);  primals_190 = None
        permute_273: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        mm_148: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_610, permute_273);  view_610 = permute_273 = None
        view_611: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_148, [32, 128, 1024]);  mm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_8: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_82, torch.float32);  gt_82 = None
        mul_349: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_8, 1.1111111111111112);  convert_element_type_8 = None
        mul_350: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_611, mul_349);  view_611 = mul_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_597: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_141, [32, 128, 1024]);  mm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_322: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_597, 0.5)
        pow_57: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_597, 3.0)
        mul_323: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_57, 0.044715);  pow_57 = None
        add_150: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_597, mul_323);  mul_323 = None
        mul_324: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_150, 0.7978845608028654);  add_150 = None
        tanh_15: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_324);  mul_324 = None
        add_151: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_15, 1.0)
        mul_325: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_322, add_151)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_351: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_350, mul_325);  mul_325 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_599: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_142, [32, 128, 1024]);  mm_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_352: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_350, view_599);  mul_350 = view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_612: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_351, [4096, 1024]);  mul_351 = None
        permute_275: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        mm_149: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_275, view_596);  permute_275 = None
        permute_264: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_189, [1, 0]);  primals_189 = None
        permute_277: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_264, [1, 0]);  permute_264 = None
        mm_150: "f32[4096, 512]" = torch.ops.aten.mm.default(view_612, permute_277);  view_612 = permute_277 = None
        view_613: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_150, [32, 128, 512]);  mm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_353: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_352, mul_322);  mul_322 = None
        mul_354: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_352, add_151);  mul_352 = add_151 = None
        mul_355: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_15, tanh_15);  tanh_15 = None
        sub_29: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_355);  mul_355 = None
        mul_356: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_353, sub_29);  mul_353 = sub_29 = None
        mul_357: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_356, 0.7978845608028654);  mul_356 = None
        mul_358: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_357, 0.044715)
        pow_61: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_597, 2.0);  view_597 = None
        mul_359: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_61, 3.0);  pow_61 = None
        mul_360: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_358, mul_359);  mul_358 = mul_359 = None
        add_156: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_357, mul_360);  mul_357 = mul_360 = None
        mul_361: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_354, 0.5);  mul_354 = None
        add_157: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_156, mul_361);  add_156 = mul_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_614: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_157, [4096, 1024]);  add_157 = None
        permute_279: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_614, [1, 0])
        mm_151: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_279, view_596);  permute_279 = view_596 = None
        permute_263: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_188, [1, 0]);  primals_188 = None
        permute_281: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        mm_152: "f32[4096, 512]" = torch.ops.aten.mm.default(view_614, permute_281);  view_614 = permute_281 = None
        view_615: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_152, [32, 128, 512]);  mm_152 = None
        add_158: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_613, view_615);  view_613 = view_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_362: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_158, primals_187);  primals_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_320: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_148, rsqrt_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_363: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_158, mul_320);  add_158 = mul_320 = None
        sum_31: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_363, [0, 1], True);  mul_363 = None
        view_616: "f32[512]" = torch.ops.aten.reshape.default(sum_31, [512]);  sum_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_364: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_362, add_148)
        mul_365: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_362, rsqrt_40);  mul_362 = None
        sum_32: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_364, [2], True);  mul_364 = None
        add_159: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_155, mul_365);  add_155 = mul_365 = None
        pow_62: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_40, 3);  rsqrt_40 = None
        mul_366: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_32, -0.5);  sum_32 = None
        mul_367: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_366, pow_62);  mul_366 = pow_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_100: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_367, [32, 128, 512]);  mul_367 = None
        div_31: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_100, 512);  expand_100 = None
        pow_63: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_148, 1.0);  add_148 = None
        mul_368: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_63, 2.0);  pow_63 = None
        mul_369: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_31, mul_368);  div_31 = mul_368 = None
        add_160: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_159, mul_369);  add_159 = mul_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_9: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_81, torch.float32);  gt_81 = None
        mul_370: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_9, 1.1111111111111112);  convert_element_type_9 = None
        mul_371: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_160, mul_370);  mul_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_617: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_371, [4096, 512]);  mul_371 = None
        permute_283: "f32[512, 4096]" = torch.ops.aten.permute.default(view_617, [1, 0])
        mm_153: "f32[512, 384]" = torch.ops.aten.mm.default(permute_283, view_594);  permute_283 = view_594 = None
        permute_262: "f32[384, 512]" = torch.ops.aten.permute.default(primals_186, [1, 0]);  primals_186 = None
        permute_285: "f32[512, 384]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        mm_154: "f32[4096, 384]" = torch.ops.aten.mm.default(view_617, permute_285);  view_617 = permute_285 = None
        view_618: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_154, [32, 128, 384]);  mm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_619: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_618, [32, 128, 6, 64]);  view_618 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_287: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_619, [0, 2, 1, 3]);  view_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_100: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_287, memory_format = torch.contiguous_format);  permute_287 = None
        view_620: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_100, [192, 128, 64]);  clone_100 = None
        bmm_48: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_288, view_620);  permute_288 = None
        bmm_49: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_620, permute_289);  view_620 = permute_289 = None
        view_621: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_48, [32, 6, 128, 64]);  bmm_48 = None
        view_622: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_49, [32, 6, 128, 128]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_10: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_80, torch.float32);  gt_80 = None
        mul_372: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_10, 1.1111111111111112);  convert_element_type_10 = None
        mul_373: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_622, mul_372);  view_622 = mul_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_374: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_373, div_27);  mul_373 = None
        sum_33: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_374, [-1], True)
        neg_2: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_27);  div_27 = None
        fma: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_2, sum_33, mul_374);  neg_2 = sum_33 = mul_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_623: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma, [192, 128, 128]);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_50: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_290, view_623);  permute_290 = None
        bmm_51: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_623, permute_291);  view_623 = permute_291 = None
        view_628: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_50, [32, 6, 64, 128]);  bmm_50 = None
        view_629: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_51, [32, 6, 128, 64]);  bmm_51 = None
        permute_292: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_628, [0, 1, 3, 2]);  view_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_293: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_621, [0, 2, 1, 3]);  view_621 = None
        clone_103: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_293, memory_format = torch.contiguous_format);  permute_293 = None
        view_630: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_103, [32, 128, 384]);  clone_103 = None
        view_631: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_630, [4096, 384]);  view_630 = None
        permute_294: "f32[384, 4096]" = torch.ops.aten.permute.default(view_631, [1, 0])
        view_581: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_155: "f32[384, 512]" = torch.ops.aten.mm.default(permute_294, view_581);  permute_294 = view_581 = None
        permute_258: "f32[512, 384]" = torch.ops.aten.permute.default(primals_185, [1, 0]);  primals_185 = None
        permute_296: "f32[384, 512]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        mm_156: "f32[4096, 512]" = torch.ops.aten.mm.default(view_631, permute_296);  view_631 = permute_296 = None
        view_632: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_156, [32, 128, 512]);  mm_156 = None
        add_161: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(tangents_3, view_632);  tangents_3 = view_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_298: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_292, [0, 2, 1, 3]);  permute_292 = None
        view_633: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_298, [32, 128, 384]);  permute_298 = None
        clone_104: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_633, memory_format = torch.contiguous_format);  view_633 = None
        view_634: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_104, [4096, 384]);  clone_104 = None
        permute_299: "f32[384, 4096]" = torch.ops.aten.permute.default(view_634, [1, 0])
        view_578: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_157: "f32[384, 512]" = torch.ops.aten.mm.default(permute_299, view_578);  permute_299 = view_578 = None
        permute_256: "f32[512, 384]" = torch.ops.aten.permute.default(primals_184, [1, 0]);  primals_184 = None
        permute_301: "f32[384, 512]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
        mm_158: "f32[4096, 512]" = torch.ops.aten.mm.default(view_634, permute_301);  view_634 = permute_301 = None
        view_635: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_158, [32, 128, 512]);  mm_158 = None
        add_162: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_161, view_635);  add_161 = view_635 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_303: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None
        clone_105: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None
        view_636: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_105, [32, 128, 384]);  clone_105 = None
        view_637: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_636, [4096, 384]);  view_636 = None
        permute_304: "f32[384, 4096]" = torch.ops.aten.permute.default(view_637, [1, 0])
        mm_159: "f32[384, 512]" = torch.ops.aten.mm.default(permute_304, view_575);  permute_304 = view_575 = None
        permute_254: "f32[512, 384]" = torch.ops.aten.permute.default(primals_183, [1, 0]);  primals_183 = None
        permute_306: "f32[384, 512]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        mm_160: "f32[4096, 512]" = torch.ops.aten.mm.default(view_637, permute_306);  view_637 = permute_306 = None
        view_638: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_160, [32, 128, 512]);  mm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_375: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_638, primals_182);  primals_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_314: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_145, rsqrt_39)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_376: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_638, mul_314);  view_638 = mul_314 = None
        sum_34: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_376, [0, 1], True);  mul_376 = None
        view_639: "f32[512]" = torch.ops.aten.reshape.default(sum_34, [512]);  sum_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_377: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_375, add_145)
        mul_378: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_375, rsqrt_39);  mul_375 = None
        sum_35: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_377, [2], True);  mul_377 = None
        add_163: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_160, mul_378);  add_160 = mul_378 = None
        pow_64: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_39, 3);  rsqrt_39 = None
        mul_379: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_35, -0.5);  sum_35 = None
        mul_380: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_379, pow_64);  mul_379 = pow_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_101: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_380, [32, 128, 512]);  mul_380 = None
        div_32: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_101, 512);  expand_101 = None
        pow_65: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_145, 1.0);  add_145 = None
        mul_381: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_65, 2.0);  pow_65 = None
        mul_382: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_32, mul_381);  div_32 = mul_381 = None
        add_164: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_163, mul_382);  add_163 = mul_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_11: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_79, torch.float32);  gt_79 = None
        mul_383: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_11, 1.1111111111111112);  convert_element_type_11 = None
        mul_384: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_164, mul_383);  mul_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_640: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_384, [4096, 512]);  mul_384 = None
        permute_308: "f32[512, 4096]" = torch.ops.aten.permute.default(view_640, [1, 0])
        mm_161: "f32[512, 384]" = torch.ops.aten.mm.default(permute_308, view_573);  permute_308 = view_573 = None
        permute_253: "f32[384, 512]" = torch.ops.aten.permute.default(primals_181, [1, 0]);  primals_181 = None
        permute_310: "f32[512, 384]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        mm_162: "f32[4096, 384]" = torch.ops.aten.mm.default(view_640, permute_310);  view_640 = permute_310 = None
        view_641: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_162, [32, 128, 384]);  mm_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_642: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_641, [32, 128, 6, 64]);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_312: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_642, [0, 2, 1, 3]);  view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_107: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_312, memory_format = torch.contiguous_format);  permute_312 = None
        view_643: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_107, [192, 128, 64]);  clone_107 = None
        bmm_52: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_313, view_643);  permute_313 = None
        bmm_53: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_643, permute_314);  view_643 = permute_314 = None
        view_644: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_52, [32, 6, 128, 64]);  bmm_52 = None
        view_645: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_53, [32, 6, 128, 128]);  bmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_12: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_78, torch.float32);  gt_78 = None
        mul_385: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_12, 1.1111111111111112);  convert_element_type_12 = None
        mul_386: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_645, mul_385);  view_645 = mul_385 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_387: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_386, div_26);  mul_386 = None
        sum_36: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_387, [-1], True)
        neg_3: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_26);  div_26 = None
        fma_1: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_3, sum_36, mul_387);  neg_3 = sum_36 = mul_387 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_646: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_1, [192, 128, 128]);  fma_1 = None
        view_648: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_646, [32, 6, 128, 128]);  view_646 = None
        view_649: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_648, [192, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_54: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_315, view_649);  permute_315 = None
        bmm_55: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_649, permute_316);  view_649 = permute_316 = None
        view_651: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_54, [32, 6, 64, 128]);  bmm_54 = None
        view_652: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_55, [32, 6, 128, 64]);  bmm_55 = None
        permute_317: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_651, [0, 1, 3, 2]);  view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_318: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_644, [0, 2, 1, 3]);  view_644 = None
        clone_110: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_318, memory_format = torch.contiguous_format);  permute_318 = None
        view_653: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_110, [32, 128, 384]);  clone_110 = None
        view_654: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_653, [4096, 384]);  view_653 = None
        permute_319: "f32[384, 4096]" = torch.ops.aten.permute.default(view_654, [1, 0])
        mm_163: "f32[384, 512]" = torch.ops.aten.mm.default(permute_319, view_554);  permute_319 = None
        permute_249: "f32[512, 384]" = torch.ops.aten.permute.default(primals_180, [1, 0]);  primals_180 = None
        permute_321: "f32[384, 512]" = torch.ops.aten.permute.default(permute_249, [1, 0]);  permute_249 = None
        mm_164: "f32[4096, 512]" = torch.ops.aten.mm.default(view_654, permute_321);  view_654 = permute_321 = None
        view_655: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_164, [32, 128, 512]);  mm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_323: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_317, [0, 2, 1, 3]);  permute_317 = None
        view_656: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_323, [32, 128, 384]);  permute_323 = None
        clone_111: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_656, memory_format = torch.contiguous_format);  view_656 = None
        view_657: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_111, [4096, 384]);  clone_111 = None
        permute_324: "f32[384, 4096]" = torch.ops.aten.permute.default(view_657, [1, 0])
        mm_165: "f32[384, 512]" = torch.ops.aten.mm.default(permute_324, view_554);  permute_324 = None
        permute_247: "f32[512, 384]" = torch.ops.aten.permute.default(primals_179, [1, 0]);  primals_179 = None
        permute_326: "f32[384, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        mm_166: "f32[4096, 512]" = torch.ops.aten.mm.default(view_657, permute_326);  view_657 = permute_326 = None
        view_658: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_166, [32, 128, 512]);  mm_166 = None
        add_165: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_655, view_658);  view_655 = view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_328: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_652, [0, 2, 1, 3]);  view_652 = None
        clone_112: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_328, memory_format = torch.contiguous_format);  permute_328 = None
        view_659: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_112, [32, 128, 384]);  clone_112 = None
        view_660: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_659, [4096, 384]);  view_659 = None
        permute_329: "f32[384, 4096]" = torch.ops.aten.permute.default(view_660, [1, 0])
        mm_167: "f32[384, 512]" = torch.ops.aten.mm.default(permute_329, view_554);  permute_329 = view_554 = None
        permute_245: "f32[512, 384]" = torch.ops.aten.permute.default(primals_178, [1, 0]);  primals_178 = None
        permute_331: "f32[384, 512]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        mm_168: "f32[4096, 512]" = torch.ops.aten.mm.default(view_660, permute_331);  view_660 = permute_331 = None
        view_661: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_168, [32, 128, 512]);  mm_168 = None
        add_166: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_165, view_661);  add_165 = view_661 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_388: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_166, primals_177);  primals_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_308: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_142, rsqrt_38)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_389: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_166, mul_308);  add_166 = mul_308 = None
        sum_37: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_389, [0, 1], True);  mul_389 = None
        view_662: "f32[512]" = torch.ops.aten.reshape.default(sum_37, [512]);  sum_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_390: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_388, add_142)
        mul_391: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_388, rsqrt_38);  mul_388 = None
        sum_38: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_390, [2], True);  mul_390 = None
        add_167: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_164, mul_391);  add_164 = mul_391 = None
        pow_66: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_38, 3);  rsqrt_38 = None
        mul_392: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_38, -0.5);  sum_38 = None
        mul_393: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_392, pow_66);  mul_392 = pow_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_102: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_393, [32, 128, 512]);  mul_393 = None
        div_33: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_102, 512);  expand_102 = None
        pow_67: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_142, 1.0);  add_142 = None
        mul_394: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_67, 2.0);  pow_67 = None
        mul_395: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_33, mul_394);  div_33 = mul_394 = None
        add_168: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_167, mul_395);  add_167 = mul_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_13: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_77, torch.float32);  gt_77 = None
        mul_396: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_13, 1.1111111111111112);  convert_element_type_13 = None
        mul_397: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_168, mul_396);  mul_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_663: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_397, [4096, 512]);  mul_397 = None
        permute_333: "f32[512, 4096]" = torch.ops.aten.permute.default(view_663, [1, 0])
        mm_169: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_333, view_552);  permute_333 = view_552 = None
        permute_244: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_176, [1, 0]);  primals_176 = None
        permute_335: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_244, [1, 0]);  permute_244 = None
        mm_170: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_663, permute_335);  view_663 = permute_335 = None
        view_664: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_170, [32, 128, 1024]);  mm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_14: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_76, torch.float32);  gt_76 = None
        mul_398: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_14, 1.1111111111111112);  convert_element_type_14 = None
        mul_399: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_664, mul_398);  view_664 = mul_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_549: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_130, [32, 128, 1024]);  mm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_299: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_549, 0.5)
        pow_53: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_549, 3.0)
        mul_300: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_53, 0.044715);  pow_53 = None
        add_140: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_549, mul_300);  mul_300 = None
        mul_301: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_140, 0.7978845608028654);  add_140 = None
        tanh_14: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_301);  mul_301 = None
        add_141: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_14, 1.0)
        mul_302: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_299, add_141)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_400: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_399, mul_302);  mul_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_551: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_131, [32, 128, 1024]);  mm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_401: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_399, view_551);  mul_399 = view_551 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_665: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_400, [4096, 1024]);  mul_400 = None
        permute_337: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_665, [1, 0])
        mm_171: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_337, view_548);  permute_337 = None
        permute_243: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_175, [1, 0]);  primals_175 = None
        permute_339: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        mm_172: "f32[4096, 512]" = torch.ops.aten.mm.default(view_665, permute_339);  view_665 = permute_339 = None
        view_666: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_172, [32, 128, 512]);  mm_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_402: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_401, mul_299);  mul_299 = None
        mul_403: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_401, add_141);  mul_401 = add_141 = None
        mul_404: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_14, tanh_14);  tanh_14 = None
        sub_30: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_404);  mul_404 = None
        mul_405: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_402, sub_30);  mul_402 = sub_30 = None
        mul_406: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_405, 0.7978845608028654);  mul_405 = None
        mul_407: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_406, 0.044715)
        pow_68: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_549, 2.0);  view_549 = None
        mul_408: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_68, 3.0);  pow_68 = None
        mul_409: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_407, mul_408);  mul_407 = mul_408 = None
        add_169: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_406, mul_409);  mul_406 = mul_409 = None
        mul_410: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_403, 0.5);  mul_403 = None
        add_170: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_169, mul_410);  add_169 = mul_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_667: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_170, [4096, 1024]);  add_170 = None
        permute_341: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_667, [1, 0])
        mm_173: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_341, view_548);  permute_341 = view_548 = None
        permute_242: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_174, [1, 0]);  primals_174 = None
        permute_343: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        mm_174: "f32[4096, 512]" = torch.ops.aten.mm.default(view_667, permute_343);  view_667 = permute_343 = None
        view_668: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_174, [32, 128, 512]);  mm_174 = None
        add_171: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_666, view_668);  view_666 = view_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_411: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_171, primals_173);  primals_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_297: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_138, rsqrt_37)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_412: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_171, mul_297);  add_171 = mul_297 = None
        sum_39: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_412, [0, 1], True);  mul_412 = None
        view_669: "f32[512]" = torch.ops.aten.reshape.default(sum_39, [512]);  sum_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_413: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_411, add_138)
        mul_414: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_411, rsqrt_37);  mul_411 = None
        sum_40: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_413, [2], True);  mul_413 = None
        add_172: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_168, mul_414);  add_168 = mul_414 = None
        pow_69: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_37, 3);  rsqrt_37 = None
        mul_415: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_40, -0.5);  sum_40 = None
        mul_416: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_415, pow_69);  mul_415 = pow_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_103: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_416, [32, 128, 512]);  mul_416 = None
        div_34: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_103, 512);  expand_103 = None
        pow_70: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_138, 1.0);  add_138 = None
        mul_417: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_70, 2.0);  pow_70 = None
        mul_418: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_34, mul_417);  div_34 = mul_417 = None
        add_173: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_172, mul_418);  add_172 = mul_418 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_15: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_75, torch.float32);  gt_75 = None
        mul_419: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_15, 1.1111111111111112);  convert_element_type_15 = None
        mul_420: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_173, mul_419);  mul_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_670: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_420, [4096, 512]);  mul_420 = None
        permute_345: "f32[512, 4096]" = torch.ops.aten.permute.default(view_670, [1, 0])
        mm_175: "f32[512, 384]" = torch.ops.aten.mm.default(permute_345, view_546);  permute_345 = view_546 = None
        permute_241: "f32[384, 512]" = torch.ops.aten.permute.default(primals_172, [1, 0]);  primals_172 = None
        permute_347: "f32[512, 384]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        mm_176: "f32[4096, 384]" = torch.ops.aten.mm.default(view_670, permute_347);  view_670 = permute_347 = None
        view_671: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_176, [32, 128, 384]);  mm_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_672: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_671, [32, 128, 6, 64]);  view_671 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_349: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_672, [0, 2, 1, 3]);  view_672 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_116: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_349, memory_format = torch.contiguous_format);  permute_349 = None
        view_673: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_116, [192, 128, 64]);  clone_116 = None
        bmm_56: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_350, view_673);  permute_350 = None
        bmm_57: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_673, permute_351);  view_673 = permute_351 = None
        view_674: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_56, [32, 6, 128, 64]);  bmm_56 = None
        view_675: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_57, [32, 6, 128, 128]);  bmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_16: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_74, torch.float32);  gt_74 = None
        mul_421: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_16, 1.1111111111111112);  convert_element_type_16 = None
        mul_422: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_675, mul_421);  view_675 = mul_421 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_423: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_422, div_25);  mul_422 = None
        sum_41: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_423, [-1], True)
        neg_4: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_25);  div_25 = None
        fma_2: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_4, sum_41, mul_423);  neg_4 = sum_41 = mul_423 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_676: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_2, [192, 128, 128]);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_58: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_352, view_676);  permute_352 = None
        bmm_59: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_676, permute_353);  view_676 = permute_353 = None
        view_681: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_58, [32, 6, 64, 128]);  bmm_58 = None
        view_682: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_59, [32, 6, 128, 64]);  bmm_59 = None
        permute_354: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_681, [0, 1, 3, 2]);  view_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_355: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_674, [0, 2, 1, 3]);  view_674 = None
        clone_119: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_355, memory_format = torch.contiguous_format);  permute_355 = None
        view_683: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_119, [32, 128, 384]);  clone_119 = None
        view_684: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_683, [4096, 384]);  view_683 = None
        permute_356: "f32[384, 4096]" = torch.ops.aten.permute.default(view_684, [1, 0])
        view_533: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_177: "f32[384, 512]" = torch.ops.aten.mm.default(permute_356, view_533);  permute_356 = view_533 = None
        permute_237: "f32[512, 384]" = torch.ops.aten.permute.default(primals_171, [1, 0]);  primals_171 = None
        permute_358: "f32[384, 512]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        mm_178: "f32[4096, 512]" = torch.ops.aten.mm.default(view_684, permute_358);  view_684 = permute_358 = None
        view_685: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_178, [32, 128, 512]);  mm_178 = None
        add_174: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_162, view_685);  add_162 = view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_360: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_354, [0, 2, 1, 3]);  permute_354 = None
        view_686: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_360, [32, 128, 384]);  permute_360 = None
        clone_120: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_686, memory_format = torch.contiguous_format);  view_686 = None
        view_687: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_120, [4096, 384]);  clone_120 = None
        permute_361: "f32[384, 4096]" = torch.ops.aten.permute.default(view_687, [1, 0])
        view_530: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_179: "f32[384, 512]" = torch.ops.aten.mm.default(permute_361, view_530);  permute_361 = view_530 = None
        permute_235: "f32[512, 384]" = torch.ops.aten.permute.default(primals_170, [1, 0]);  primals_170 = None
        permute_363: "f32[384, 512]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        mm_180: "f32[4096, 512]" = torch.ops.aten.mm.default(view_687, permute_363);  view_687 = permute_363 = None
        view_688: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_180, [32, 128, 512]);  mm_180 = None
        add_175: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_174, view_688);  add_174 = view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_365: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
        clone_121: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_365, memory_format = torch.contiguous_format);  permute_365 = None
        view_689: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_121, [32, 128, 384]);  clone_121 = None
        view_690: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_689, [4096, 384]);  view_689 = None
        permute_366: "f32[384, 4096]" = torch.ops.aten.permute.default(view_690, [1, 0])
        mm_181: "f32[384, 512]" = torch.ops.aten.mm.default(permute_366, view_527);  permute_366 = view_527 = None
        permute_233: "f32[512, 384]" = torch.ops.aten.permute.default(primals_169, [1, 0]);  primals_169 = None
        permute_368: "f32[384, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        mm_182: "f32[4096, 512]" = torch.ops.aten.mm.default(view_690, permute_368);  view_690 = permute_368 = None
        view_691: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_182, [32, 128, 512]);  mm_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_424: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_691, primals_168);  primals_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_291: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_135, rsqrt_36)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_425: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_691, mul_291);  view_691 = mul_291 = None
        sum_42: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_425, [0, 1], True);  mul_425 = None
        view_692: "f32[512]" = torch.ops.aten.reshape.default(sum_42, [512]);  sum_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_426: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_424, add_135)
        mul_427: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_424, rsqrt_36);  mul_424 = None
        sum_43: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_426, [2], True);  mul_426 = None
        add_176: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_173, mul_427);  add_173 = mul_427 = None
        pow_71: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_36, 3);  rsqrt_36 = None
        mul_428: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_43, -0.5);  sum_43 = None
        mul_429: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_428, pow_71);  mul_428 = pow_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_104: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_429, [32, 128, 512]);  mul_429 = None
        div_35: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_104, 512);  expand_104 = None
        pow_72: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_135, 1.0);  add_135 = None
        mul_430: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_72, 2.0);  pow_72 = None
        mul_431: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_35, mul_430);  div_35 = mul_430 = None
        add_177: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_176, mul_431);  add_176 = mul_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_17: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_73, torch.float32);  gt_73 = None
        mul_432: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_17, 1.1111111111111112);  convert_element_type_17 = None
        mul_433: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_177, mul_432);  mul_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_693: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_433, [4096, 512]);  mul_433 = None
        permute_370: "f32[512, 4096]" = torch.ops.aten.permute.default(view_693, [1, 0])
        mm_183: "f32[512, 384]" = torch.ops.aten.mm.default(permute_370, view_525);  permute_370 = view_525 = None
        permute_232: "f32[384, 512]" = torch.ops.aten.permute.default(primals_167, [1, 0]);  primals_167 = None
        permute_372: "f32[512, 384]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        mm_184: "f32[4096, 384]" = torch.ops.aten.mm.default(view_693, permute_372);  view_693 = permute_372 = None
        view_694: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_184, [32, 128, 384]);  mm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_695: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_694, [32, 128, 6, 64]);  view_694 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_374: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_695, [0, 2, 1, 3]);  view_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_123: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_374, memory_format = torch.contiguous_format);  permute_374 = None
        view_696: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_123, [192, 128, 64]);  clone_123 = None
        bmm_60: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_375, view_696);  permute_375 = None
        bmm_61: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_696, permute_376);  view_696 = permute_376 = None
        view_697: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_60, [32, 6, 128, 64]);  bmm_60 = None
        view_698: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_61, [32, 6, 128, 128]);  bmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_18: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_72, torch.float32);  gt_72 = None
        mul_434: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_18, 1.1111111111111112);  convert_element_type_18 = None
        mul_435: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_698, mul_434);  view_698 = mul_434 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_436: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_435, div_24);  mul_435 = None
        sum_44: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_436, [-1], True)
        neg_5: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_24);  div_24 = None
        fma_3: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_5, sum_44, mul_436);  neg_5 = sum_44 = mul_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_699: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_3, [192, 128, 128]);  fma_3 = None
        view_701: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_699, [32, 6, 128, 128]);  view_699 = None
        view_702: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_701, [192, 128, 128])
        add_178: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_648, view_701);  view_648 = view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_62: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_377, view_702);  permute_377 = None
        bmm_63: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_702, permute_378);  view_702 = permute_378 = None
        view_704: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_62, [32, 6, 64, 128]);  bmm_62 = None
        view_705: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_63, [32, 6, 128, 64]);  bmm_63 = None
        permute_379: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_704, [0, 1, 3, 2]);  view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_380: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None
        clone_126: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_380, memory_format = torch.contiguous_format);  permute_380 = None
        view_706: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_126, [32, 128, 384]);  clone_126 = None
        view_707: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_706, [4096, 384]);  view_706 = None
        permute_381: "f32[384, 4096]" = torch.ops.aten.permute.default(view_707, [1, 0])
        mm_185: "f32[384, 512]" = torch.ops.aten.mm.default(permute_381, view_506);  permute_381 = None
        permute_228: "f32[512, 384]" = torch.ops.aten.permute.default(primals_166, [1, 0]);  primals_166 = None
        permute_383: "f32[384, 512]" = torch.ops.aten.permute.default(permute_228, [1, 0]);  permute_228 = None
        mm_186: "f32[4096, 512]" = torch.ops.aten.mm.default(view_707, permute_383);  view_707 = permute_383 = None
        view_708: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_186, [32, 128, 512]);  mm_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_385: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_379, [0, 2, 1, 3]);  permute_379 = None
        view_709: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_385, [32, 128, 384]);  permute_385 = None
        clone_127: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_709, memory_format = torch.contiguous_format);  view_709 = None
        view_710: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_127, [4096, 384]);  clone_127 = None
        permute_386: "f32[384, 4096]" = torch.ops.aten.permute.default(view_710, [1, 0])
        mm_187: "f32[384, 512]" = torch.ops.aten.mm.default(permute_386, view_506);  permute_386 = None
        permute_226: "f32[512, 384]" = torch.ops.aten.permute.default(primals_165, [1, 0]);  primals_165 = None
        permute_388: "f32[384, 512]" = torch.ops.aten.permute.default(permute_226, [1, 0]);  permute_226 = None
        mm_188: "f32[4096, 512]" = torch.ops.aten.mm.default(view_710, permute_388);  view_710 = permute_388 = None
        view_711: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_188, [32, 128, 512]);  mm_188 = None
        add_179: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_708, view_711);  view_708 = view_711 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_390: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_705, [0, 2, 1, 3]);  view_705 = None
        clone_128: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        view_712: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_128, [32, 128, 384]);  clone_128 = None
        view_713: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_712, [4096, 384]);  view_712 = None
        permute_391: "f32[384, 4096]" = torch.ops.aten.permute.default(view_713, [1, 0])
        mm_189: "f32[384, 512]" = torch.ops.aten.mm.default(permute_391, view_506);  permute_391 = view_506 = None
        permute_224: "f32[512, 384]" = torch.ops.aten.permute.default(primals_164, [1, 0]);  primals_164 = None
        permute_393: "f32[384, 512]" = torch.ops.aten.permute.default(permute_224, [1, 0]);  permute_224 = None
        mm_190: "f32[4096, 512]" = torch.ops.aten.mm.default(view_713, permute_393);  view_713 = permute_393 = None
        view_714: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_190, [32, 128, 512]);  mm_190 = None
        add_180: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_179, view_714);  add_179 = view_714 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_437: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_180, primals_163);  primals_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_285: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_132, rsqrt_35)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_438: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_180, mul_285);  add_180 = mul_285 = None
        sum_45: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_438, [0, 1], True);  mul_438 = None
        view_715: "f32[512]" = torch.ops.aten.reshape.default(sum_45, [512]);  sum_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_439: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_437, add_132)
        mul_440: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_437, rsqrt_35);  mul_437 = None
        sum_46: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_439, [2], True);  mul_439 = None
        add_181: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_177, mul_440);  add_177 = mul_440 = None
        pow_73: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_35, 3);  rsqrt_35 = None
        mul_441: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_46, -0.5);  sum_46 = None
        mul_442: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_441, pow_73);  mul_441 = pow_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_105: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_442, [32, 128, 512]);  mul_442 = None
        div_36: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_105, 512);  expand_105 = None
        pow_74: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_132, 1.0);  add_132 = None
        mul_443: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_74, 2.0);  pow_74 = None
        mul_444: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_36, mul_443);  div_36 = mul_443 = None
        add_182: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_181, mul_444);  add_181 = mul_444 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_19: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_71, torch.float32);  gt_71 = None
        mul_445: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_19, 1.1111111111111112);  convert_element_type_19 = None
        mul_446: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_182, mul_445);  mul_445 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_716: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_446, [4096, 512]);  mul_446 = None
        permute_395: "f32[512, 4096]" = torch.ops.aten.permute.default(view_716, [1, 0])
        mm_191: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_395, view_504);  permute_395 = view_504 = None
        permute_223: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_162, [1, 0]);  primals_162 = None
        permute_397: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        mm_192: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_716, permute_397);  view_716 = permute_397 = None
        view_717: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_192, [32, 128, 1024]);  mm_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_20: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_70, torch.float32);  gt_70 = None
        mul_447: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_20, 1.1111111111111112);  convert_element_type_20 = None
        mul_448: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_717, mul_447);  view_717 = mul_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_501: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_119, [32, 128, 1024]);  mm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_276: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_501, 0.5)
        pow_49: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_501, 3.0)
        mul_277: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_49, 0.044715);  pow_49 = None
        add_130: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_501, mul_277);  mul_277 = None
        mul_278: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_130, 0.7978845608028654);  add_130 = None
        tanh_13: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_278);  mul_278 = None
        add_131: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_13, 1.0)
        mul_279: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_276, add_131)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_449: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_448, mul_279);  mul_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_503: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_120, [32, 128, 1024]);  mm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_450: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_448, view_503);  mul_448 = view_503 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_718: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_449, [4096, 1024]);  mul_449 = None
        permute_399: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_718, [1, 0])
        mm_193: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_399, view_500);  permute_399 = None
        permute_222: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_161, [1, 0]);  primals_161 = None
        permute_401: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        mm_194: "f32[4096, 512]" = torch.ops.aten.mm.default(view_718, permute_401);  view_718 = permute_401 = None
        view_719: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_194, [32, 128, 512]);  mm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_451: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_450, mul_276);  mul_276 = None
        mul_452: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_450, add_131);  mul_450 = add_131 = None
        mul_453: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_13, tanh_13);  tanh_13 = None
        sub_31: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_453);  mul_453 = None
        mul_454: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_451, sub_31);  mul_451 = sub_31 = None
        mul_455: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_454, 0.7978845608028654);  mul_454 = None
        mul_456: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_455, 0.044715)
        pow_75: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_501, 2.0);  view_501 = None
        mul_457: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_75, 3.0);  pow_75 = None
        mul_458: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_456, mul_457);  mul_456 = mul_457 = None
        add_183: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_455, mul_458);  mul_455 = mul_458 = None
        mul_459: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_452, 0.5);  mul_452 = None
        add_184: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_183, mul_459);  add_183 = mul_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_720: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_184, [4096, 1024]);  add_184 = None
        permute_403: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_720, [1, 0])
        mm_195: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_403, view_500);  permute_403 = view_500 = None
        permute_221: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_160, [1, 0]);  primals_160 = None
        permute_405: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        mm_196: "f32[4096, 512]" = torch.ops.aten.mm.default(view_720, permute_405);  view_720 = permute_405 = None
        view_721: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_196, [32, 128, 512]);  mm_196 = None
        add_185: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_719, view_721);  view_719 = view_721 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_460: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_185, primals_159);  primals_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_274: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_128, rsqrt_34)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_461: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_185, mul_274);  add_185 = mul_274 = None
        sum_47: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_461, [0, 1], True);  mul_461 = None
        view_722: "f32[512]" = torch.ops.aten.reshape.default(sum_47, [512]);  sum_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_462: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_460, add_128)
        mul_463: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_460, rsqrt_34);  mul_460 = None
        sum_48: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_462, [2], True);  mul_462 = None
        add_186: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_182, mul_463);  add_182 = mul_463 = None
        pow_76: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_34, 3);  rsqrt_34 = None
        mul_464: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_48, -0.5);  sum_48 = None
        mul_465: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_464, pow_76);  mul_464 = pow_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_106: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_465, [32, 128, 512]);  mul_465 = None
        div_37: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_106, 512);  expand_106 = None
        pow_77: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_128, 1.0);  add_128 = None
        mul_466: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_77, 2.0);  pow_77 = None
        mul_467: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_37, mul_466);  div_37 = mul_466 = None
        add_187: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_186, mul_467);  add_186 = mul_467 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_21: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_69, torch.float32);  gt_69 = None
        mul_468: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_21, 1.1111111111111112);  convert_element_type_21 = None
        mul_469: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_187, mul_468);  mul_468 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_723: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_469, [4096, 512]);  mul_469 = None
        permute_407: "f32[512, 4096]" = torch.ops.aten.permute.default(view_723, [1, 0])
        mm_197: "f32[512, 384]" = torch.ops.aten.mm.default(permute_407, view_498);  permute_407 = view_498 = None
        permute_220: "f32[384, 512]" = torch.ops.aten.permute.default(primals_158, [1, 0]);  primals_158 = None
        permute_409: "f32[512, 384]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        mm_198: "f32[4096, 384]" = torch.ops.aten.mm.default(view_723, permute_409);  view_723 = permute_409 = None
        view_724: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_198, [32, 128, 384]);  mm_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_725: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_724, [32, 128, 6, 64]);  view_724 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_411: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_725, [0, 2, 1, 3]);  view_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_132: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_411, memory_format = torch.contiguous_format);  permute_411 = None
        view_726: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_132, [192, 128, 64]);  clone_132 = None
        bmm_64: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_412, view_726);  permute_412 = None
        bmm_65: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_726, permute_413);  view_726 = permute_413 = None
        view_727: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_64, [32, 6, 128, 64]);  bmm_64 = None
        view_728: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_65, [32, 6, 128, 128]);  bmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_22: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_68, torch.float32);  gt_68 = None
        mul_470: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_22, 1.1111111111111112);  convert_element_type_22 = None
        mul_471: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_728, mul_470);  view_728 = mul_470 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_472: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_471, div_23);  mul_471 = None
        sum_49: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_472, [-1], True)
        neg_6: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_23);  div_23 = None
        fma_4: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_6, sum_49, mul_472);  neg_6 = sum_49 = mul_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_729: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_4, [192, 128, 128]);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_66: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_414, view_729);  permute_414 = None
        bmm_67: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_729, permute_415);  view_729 = permute_415 = None
        view_734: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_66, [32, 6, 64, 128]);  bmm_66 = None
        view_735: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_67, [32, 6, 128, 64]);  bmm_67 = None
        permute_416: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_734, [0, 1, 3, 2]);  view_734 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_417: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_727, [0, 2, 1, 3]);  view_727 = None
        clone_135: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None
        view_736: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_135, [32, 128, 384]);  clone_135 = None
        view_737: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_736, [4096, 384]);  view_736 = None
        permute_418: "f32[384, 4096]" = torch.ops.aten.permute.default(view_737, [1, 0])
        view_485: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_199: "f32[384, 512]" = torch.ops.aten.mm.default(permute_418, view_485);  permute_418 = view_485 = None
        permute_216: "f32[512, 384]" = torch.ops.aten.permute.default(primals_157, [1, 0]);  primals_157 = None
        permute_420: "f32[384, 512]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        mm_200: "f32[4096, 512]" = torch.ops.aten.mm.default(view_737, permute_420);  view_737 = permute_420 = None
        view_738: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_200, [32, 128, 512]);  mm_200 = None
        add_188: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_175, view_738);  add_175 = view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_422: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_416, [0, 2, 1, 3]);  permute_416 = None
        view_739: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_422, [32, 128, 384]);  permute_422 = None
        clone_136: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_739, memory_format = torch.contiguous_format);  view_739 = None
        view_740: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_136, [4096, 384]);  clone_136 = None
        permute_423: "f32[384, 4096]" = torch.ops.aten.permute.default(view_740, [1, 0])
        view_482: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_201: "f32[384, 512]" = torch.ops.aten.mm.default(permute_423, view_482);  permute_423 = view_482 = None
        permute_214: "f32[512, 384]" = torch.ops.aten.permute.default(primals_156, [1, 0]);  primals_156 = None
        permute_425: "f32[384, 512]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        mm_202: "f32[4096, 512]" = torch.ops.aten.mm.default(view_740, permute_425);  view_740 = permute_425 = None
        view_741: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_202, [32, 128, 512]);  mm_202 = None
        add_189: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_188, view_741);  add_188 = view_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_427: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_735, [0, 2, 1, 3]);  view_735 = None
        clone_137: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_427, memory_format = torch.contiguous_format);  permute_427 = None
        view_742: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_137, [32, 128, 384]);  clone_137 = None
        view_743: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_742, [4096, 384]);  view_742 = None
        permute_428: "f32[384, 4096]" = torch.ops.aten.permute.default(view_743, [1, 0])
        mm_203: "f32[384, 512]" = torch.ops.aten.mm.default(permute_428, view_479);  permute_428 = view_479 = None
        permute_212: "f32[512, 384]" = torch.ops.aten.permute.default(primals_155, [1, 0]);  primals_155 = None
        permute_430: "f32[384, 512]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        mm_204: "f32[4096, 512]" = torch.ops.aten.mm.default(view_743, permute_430);  view_743 = permute_430 = None
        view_744: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_204, [32, 128, 512]);  mm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_473: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_744, primals_154);  primals_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_268: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_125, rsqrt_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_474: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_744, mul_268);  view_744 = mul_268 = None
        sum_50: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_474, [0, 1], True);  mul_474 = None
        view_745: "f32[512]" = torch.ops.aten.reshape.default(sum_50, [512]);  sum_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_475: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_473, add_125)
        mul_476: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_473, rsqrt_33);  mul_473 = None
        sum_51: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_475, [2], True);  mul_475 = None
        add_190: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_187, mul_476);  add_187 = mul_476 = None
        pow_78: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_33, 3);  rsqrt_33 = None
        mul_477: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_51, -0.5);  sum_51 = None
        mul_478: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_477, pow_78);  mul_477 = pow_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_107: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_478, [32, 128, 512]);  mul_478 = None
        div_38: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_107, 512);  expand_107 = None
        pow_79: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_125, 1.0);  add_125 = None
        mul_479: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_79, 2.0);  pow_79 = None
        mul_480: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_38, mul_479);  div_38 = mul_479 = None
        add_191: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_190, mul_480);  add_190 = mul_480 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_23: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_67, torch.float32);  gt_67 = None
        mul_481: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_23, 1.1111111111111112);  convert_element_type_23 = None
        mul_482: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_191, mul_481);  mul_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_746: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_482, [4096, 512]);  mul_482 = None
        permute_432: "f32[512, 4096]" = torch.ops.aten.permute.default(view_746, [1, 0])
        mm_205: "f32[512, 384]" = torch.ops.aten.mm.default(permute_432, view_477);  permute_432 = view_477 = None
        permute_211: "f32[384, 512]" = torch.ops.aten.permute.default(primals_153, [1, 0]);  primals_153 = None
        permute_434: "f32[512, 384]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        mm_206: "f32[4096, 384]" = torch.ops.aten.mm.default(view_746, permute_434);  view_746 = permute_434 = None
        view_747: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_206, [32, 128, 384]);  mm_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_748: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_747, [32, 128, 6, 64]);  view_747 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_436: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_748, [0, 2, 1, 3]);  view_748 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_139: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None
        view_749: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_139, [192, 128, 64]);  clone_139 = None
        bmm_68: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_437, view_749);  permute_437 = None
        bmm_69: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_749, permute_438);  view_749 = permute_438 = None
        view_750: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_68, [32, 6, 128, 64]);  bmm_68 = None
        view_751: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_69, [32, 6, 128, 128]);  bmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_24: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_66, torch.float32);  gt_66 = None
        mul_483: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_24, 1.1111111111111112);  convert_element_type_24 = None
        mul_484: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_751, mul_483);  view_751 = mul_483 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_485: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_484, div_22);  mul_484 = None
        sum_52: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_485, [-1], True)
        neg_7: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_22);  div_22 = None
        fma_5: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_7, sum_52, mul_485);  neg_7 = sum_52 = mul_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_752: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_5, [192, 128, 128]);  fma_5 = None
        view_754: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_752, [32, 6, 128, 128]);  view_752 = None
        view_755: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_754, [192, 128, 128])
        add_192: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_178, view_754);  add_178 = view_754 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_70: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_439, view_755);  permute_439 = None
        bmm_71: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_755, permute_440);  view_755 = permute_440 = None
        view_757: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_70, [32, 6, 64, 128]);  bmm_70 = None
        view_758: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_71, [32, 6, 128, 64]);  bmm_71 = None
        permute_441: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_757, [0, 1, 3, 2]);  view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_442: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
        clone_142: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_442, memory_format = torch.contiguous_format);  permute_442 = None
        view_759: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_142, [32, 128, 384]);  clone_142 = None
        view_760: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_759, [4096, 384]);  view_759 = None
        permute_443: "f32[384, 4096]" = torch.ops.aten.permute.default(view_760, [1, 0])
        mm_207: "f32[384, 512]" = torch.ops.aten.mm.default(permute_443, view_458);  permute_443 = None
        permute_207: "f32[512, 384]" = torch.ops.aten.permute.default(primals_152, [1, 0]);  primals_152 = None
        permute_445: "f32[384, 512]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        mm_208: "f32[4096, 512]" = torch.ops.aten.mm.default(view_760, permute_445);  view_760 = permute_445 = None
        view_761: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_208, [32, 128, 512]);  mm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_447: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_441, [0, 2, 1, 3]);  permute_441 = None
        view_762: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_447, [32, 128, 384]);  permute_447 = None
        clone_143: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_762, memory_format = torch.contiguous_format);  view_762 = None
        view_763: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_143, [4096, 384]);  clone_143 = None
        permute_448: "f32[384, 4096]" = torch.ops.aten.permute.default(view_763, [1, 0])
        mm_209: "f32[384, 512]" = torch.ops.aten.mm.default(permute_448, view_458);  permute_448 = None
        permute_205: "f32[512, 384]" = torch.ops.aten.permute.default(primals_151, [1, 0]);  primals_151 = None
        permute_450: "f32[384, 512]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        mm_210: "f32[4096, 512]" = torch.ops.aten.mm.default(view_763, permute_450);  view_763 = permute_450 = None
        view_764: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_210, [32, 128, 512]);  mm_210 = None
        add_193: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_761, view_764);  view_761 = view_764 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_452: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_758, [0, 2, 1, 3]);  view_758 = None
        clone_144: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_452, memory_format = torch.contiguous_format);  permute_452 = None
        view_765: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_144, [32, 128, 384]);  clone_144 = None
        view_766: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_765, [4096, 384]);  view_765 = None
        permute_453: "f32[384, 4096]" = torch.ops.aten.permute.default(view_766, [1, 0])
        mm_211: "f32[384, 512]" = torch.ops.aten.mm.default(permute_453, view_458);  permute_453 = view_458 = None
        permute_203: "f32[512, 384]" = torch.ops.aten.permute.default(primals_150, [1, 0]);  primals_150 = None
        permute_455: "f32[384, 512]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        mm_212: "f32[4096, 512]" = torch.ops.aten.mm.default(view_766, permute_455);  view_766 = permute_455 = None
        view_767: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_212, [32, 128, 512]);  mm_212 = None
        add_194: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_193, view_767);  add_193 = view_767 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_486: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_194, primals_149);  primals_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_262: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_122, rsqrt_32)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_487: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_194, mul_262);  add_194 = mul_262 = None
        sum_53: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_487, [0, 1], True);  mul_487 = None
        view_768: "f32[512]" = torch.ops.aten.reshape.default(sum_53, [512]);  sum_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_488: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_486, add_122)
        mul_489: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_486, rsqrt_32);  mul_486 = None
        sum_54: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_488, [2], True);  mul_488 = None
        add_195: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_191, mul_489);  add_191 = mul_489 = None
        pow_80: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_32, 3);  rsqrt_32 = None
        mul_490: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_54, -0.5);  sum_54 = None
        mul_491: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_490, pow_80);  mul_490 = pow_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_108: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_491, [32, 128, 512]);  mul_491 = None
        div_39: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_108, 512);  expand_108 = None
        pow_81: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_122, 1.0);  add_122 = None
        mul_492: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_81, 2.0);  pow_81 = None
        mul_493: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_39, mul_492);  div_39 = mul_492 = None
        add_196: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_195, mul_493);  add_195 = mul_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_25: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_65, torch.float32);  gt_65 = None
        mul_494: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_25, 1.1111111111111112);  convert_element_type_25 = None
        mul_495: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_196, mul_494);  mul_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_769: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_495, [4096, 512]);  mul_495 = None
        permute_457: "f32[512, 4096]" = torch.ops.aten.permute.default(view_769, [1, 0])
        mm_213: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_457, view_456);  permute_457 = view_456 = None
        permute_202: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_148, [1, 0]);  primals_148 = None
        permute_459: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        mm_214: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_769, permute_459);  view_769 = permute_459 = None
        view_770: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_214, [32, 128, 1024]);  mm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_26: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_64, torch.float32);  gt_64 = None
        mul_496: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_26, 1.1111111111111112);  convert_element_type_26 = None
        mul_497: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_770, mul_496);  view_770 = mul_496 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_453: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_108, [32, 128, 1024]);  mm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_253: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_453, 0.5)
        pow_45: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_453, 3.0)
        mul_254: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_45, 0.044715);  pow_45 = None
        add_120: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_453, mul_254);  mul_254 = None
        mul_255: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_120, 0.7978845608028654);  add_120 = None
        tanh_12: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_255);  mul_255 = None
        add_121: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_12, 1.0)
        mul_256: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_253, add_121)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_498: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_497, mul_256);  mul_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_455: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_109, [32, 128, 1024]);  mm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_499: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_497, view_455);  mul_497 = view_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_771: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_498, [4096, 1024]);  mul_498 = None
        permute_461: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_771, [1, 0])
        mm_215: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_461, view_452);  permute_461 = None
        permute_201: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_147, [1, 0]);  primals_147 = None
        permute_463: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        mm_216: "f32[4096, 512]" = torch.ops.aten.mm.default(view_771, permute_463);  view_771 = permute_463 = None
        view_772: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_216, [32, 128, 512]);  mm_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_500: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_499, mul_253);  mul_253 = None
        mul_501: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_499, add_121);  mul_499 = add_121 = None
        mul_502: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_12, tanh_12);  tanh_12 = None
        sub_32: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_502);  mul_502 = None
        mul_503: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_500, sub_32);  mul_500 = sub_32 = None
        mul_504: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_503, 0.7978845608028654);  mul_503 = None
        mul_505: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_504, 0.044715)
        pow_82: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_453, 2.0);  view_453 = None
        mul_506: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_82, 3.0);  pow_82 = None
        mul_507: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_505, mul_506);  mul_505 = mul_506 = None
        add_197: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_504, mul_507);  mul_504 = mul_507 = None
        mul_508: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_501, 0.5);  mul_501 = None
        add_198: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_197, mul_508);  add_197 = mul_508 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_773: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_198, [4096, 1024]);  add_198 = None
        permute_465: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_773, [1, 0])
        mm_217: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_465, view_452);  permute_465 = view_452 = None
        permute_200: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_146, [1, 0]);  primals_146 = None
        permute_467: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        mm_218: "f32[4096, 512]" = torch.ops.aten.mm.default(view_773, permute_467);  view_773 = permute_467 = None
        view_774: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_218, [32, 128, 512]);  mm_218 = None
        add_199: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_772, view_774);  view_772 = view_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_509: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_199, primals_145);  primals_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_251: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_118, rsqrt_31)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_510: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_199, mul_251);  add_199 = mul_251 = None
        sum_55: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_510, [0, 1], True);  mul_510 = None
        view_775: "f32[512]" = torch.ops.aten.reshape.default(sum_55, [512]);  sum_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_511: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_509, add_118)
        mul_512: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_509, rsqrt_31);  mul_509 = None
        sum_56: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_511, [2], True);  mul_511 = None
        add_200: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_196, mul_512);  add_196 = mul_512 = None
        pow_83: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_31, 3);  rsqrt_31 = None
        mul_513: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_56, -0.5);  sum_56 = None
        mul_514: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_513, pow_83);  mul_513 = pow_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_109: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_514, [32, 128, 512]);  mul_514 = None
        div_40: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_109, 512);  expand_109 = None
        pow_84: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_118, 1.0);  add_118 = None
        mul_515: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_84, 2.0);  pow_84 = None
        mul_516: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_40, mul_515);  div_40 = mul_515 = None
        add_201: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_200, mul_516);  add_200 = mul_516 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_27: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_63, torch.float32);  gt_63 = None
        mul_517: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_27, 1.1111111111111112);  convert_element_type_27 = None
        mul_518: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_201, mul_517);  mul_517 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_776: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_518, [4096, 512]);  mul_518 = None
        permute_469: "f32[512, 4096]" = torch.ops.aten.permute.default(view_776, [1, 0])
        mm_219: "f32[512, 384]" = torch.ops.aten.mm.default(permute_469, view_450);  permute_469 = view_450 = None
        permute_199: "f32[384, 512]" = torch.ops.aten.permute.default(primals_144, [1, 0]);  primals_144 = None
        permute_471: "f32[512, 384]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        mm_220: "f32[4096, 384]" = torch.ops.aten.mm.default(view_776, permute_471);  view_776 = permute_471 = None
        view_777: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_220, [32, 128, 384]);  mm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_778: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_777, [32, 128, 6, 64]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_473: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_148: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_473, memory_format = torch.contiguous_format);  permute_473 = None
        view_779: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_148, [192, 128, 64]);  clone_148 = None
        bmm_72: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_474, view_779);  permute_474 = None
        bmm_73: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_779, permute_475);  view_779 = permute_475 = None
        view_780: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_72, [32, 6, 128, 64]);  bmm_72 = None
        view_781: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_73, [32, 6, 128, 128]);  bmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_28: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_62, torch.float32);  gt_62 = None
        mul_519: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_28, 1.1111111111111112);  convert_element_type_28 = None
        mul_520: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_781, mul_519);  view_781 = mul_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_521: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_520, div_21);  mul_520 = None
        sum_57: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_521, [-1], True)
        neg_8: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_21);  div_21 = None
        fma_6: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_8, sum_57, mul_521);  neg_8 = sum_57 = mul_521 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_782: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_6, [192, 128, 128]);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_74: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_476, view_782);  permute_476 = None
        bmm_75: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_782, permute_477);  view_782 = permute_477 = None
        view_787: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_74, [32, 6, 64, 128]);  bmm_74 = None
        view_788: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_75, [32, 6, 128, 64]);  bmm_75 = None
        permute_478: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_787, [0, 1, 3, 2]);  view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_479: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_780, [0, 2, 1, 3]);  view_780 = None
        clone_151: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_479, memory_format = torch.contiguous_format);  permute_479 = None
        view_789: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_151, [32, 128, 384]);  clone_151 = None
        view_790: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_789, [4096, 384]);  view_789 = None
        permute_480: "f32[384, 4096]" = torch.ops.aten.permute.default(view_790, [1, 0])
        view_437: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_221: "f32[384, 512]" = torch.ops.aten.mm.default(permute_480, view_437);  permute_480 = view_437 = None
        permute_195: "f32[512, 384]" = torch.ops.aten.permute.default(primals_143, [1, 0]);  primals_143 = None
        permute_482: "f32[384, 512]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        mm_222: "f32[4096, 512]" = torch.ops.aten.mm.default(view_790, permute_482);  view_790 = permute_482 = None
        view_791: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_222, [32, 128, 512]);  mm_222 = None
        add_202: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_189, view_791);  add_189 = view_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_484: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_478, [0, 2, 1, 3]);  permute_478 = None
        view_792: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_484, [32, 128, 384]);  permute_484 = None
        clone_152: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_792, memory_format = torch.contiguous_format);  view_792 = None
        view_793: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_152, [4096, 384]);  clone_152 = None
        permute_485: "f32[384, 4096]" = torch.ops.aten.permute.default(view_793, [1, 0])
        view_434: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_223: "f32[384, 512]" = torch.ops.aten.mm.default(permute_485, view_434);  permute_485 = view_434 = None
        permute_193: "f32[512, 384]" = torch.ops.aten.permute.default(primals_142, [1, 0]);  primals_142 = None
        permute_487: "f32[384, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        mm_224: "f32[4096, 512]" = torch.ops.aten.mm.default(view_793, permute_487);  view_793 = permute_487 = None
        view_794: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_224, [32, 128, 512]);  mm_224 = None
        add_203: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_202, view_794);  add_202 = view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_489: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_788, [0, 2, 1, 3]);  view_788 = None
        clone_153: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_489, memory_format = torch.contiguous_format);  permute_489 = None
        view_795: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_153, [32, 128, 384]);  clone_153 = None
        view_796: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_795, [4096, 384]);  view_795 = None
        permute_490: "f32[384, 4096]" = torch.ops.aten.permute.default(view_796, [1, 0])
        mm_225: "f32[384, 512]" = torch.ops.aten.mm.default(permute_490, view_431);  permute_490 = view_431 = None
        permute_191: "f32[512, 384]" = torch.ops.aten.permute.default(primals_141, [1, 0]);  primals_141 = None
        permute_492: "f32[384, 512]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        mm_226: "f32[4096, 512]" = torch.ops.aten.mm.default(view_796, permute_492);  view_796 = permute_492 = None
        view_797: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_226, [32, 128, 512]);  mm_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_522: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_797, primals_140);  primals_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_245: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_115, rsqrt_30)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_523: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_797, mul_245);  view_797 = mul_245 = None
        sum_58: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_523, [0, 1], True);  mul_523 = None
        view_798: "f32[512]" = torch.ops.aten.reshape.default(sum_58, [512]);  sum_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_524: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_522, add_115)
        mul_525: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_522, rsqrt_30);  mul_522 = None
        sum_59: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_524, [2], True);  mul_524 = None
        add_204: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_201, mul_525);  add_201 = mul_525 = None
        pow_85: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_30, 3);  rsqrt_30 = None
        mul_526: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_59, -0.5);  sum_59 = None
        mul_527: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_526, pow_85);  mul_526 = pow_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_110: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_527, [32, 128, 512]);  mul_527 = None
        div_41: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_110, 512);  expand_110 = None
        pow_86: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_115, 1.0);  add_115 = None
        mul_528: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_86, 2.0);  pow_86 = None
        mul_529: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_41, mul_528);  div_41 = mul_528 = None
        add_205: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_204, mul_529);  add_204 = mul_529 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_29: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_61, torch.float32);  gt_61 = None
        mul_530: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_29, 1.1111111111111112);  convert_element_type_29 = None
        mul_531: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_205, mul_530);  mul_530 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_799: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_531, [4096, 512]);  mul_531 = None
        permute_494: "f32[512, 4096]" = torch.ops.aten.permute.default(view_799, [1, 0])
        mm_227: "f32[512, 384]" = torch.ops.aten.mm.default(permute_494, view_429);  permute_494 = view_429 = None
        permute_190: "f32[384, 512]" = torch.ops.aten.permute.default(primals_139, [1, 0]);  primals_139 = None
        permute_496: "f32[512, 384]" = torch.ops.aten.permute.default(permute_190, [1, 0]);  permute_190 = None
        mm_228: "f32[4096, 384]" = torch.ops.aten.mm.default(view_799, permute_496);  view_799 = permute_496 = None
        view_800: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_228, [32, 128, 384]);  mm_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_801: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_800, [32, 128, 6, 64]);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_498: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_801, [0, 2, 1, 3]);  view_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_155: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_498, memory_format = torch.contiguous_format);  permute_498 = None
        view_802: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_155, [192, 128, 64]);  clone_155 = None
        bmm_76: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_499, view_802);  permute_499 = None
        bmm_77: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_802, permute_500);  view_802 = permute_500 = None
        view_803: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_76, [32, 6, 128, 64]);  bmm_76 = None
        view_804: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_77, [32, 6, 128, 128]);  bmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_30: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_60, torch.float32);  gt_60 = None
        mul_532: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_30, 1.1111111111111112);  convert_element_type_30 = None
        mul_533: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_804, mul_532);  view_804 = mul_532 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_534: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_533, div_20);  mul_533 = None
        sum_60: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_534, [-1], True)
        neg_9: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_20);  div_20 = None
        fma_7: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_9, sum_60, mul_534);  neg_9 = sum_60 = mul_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_805: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_7, [192, 128, 128]);  fma_7 = None
        view_807: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_805, [32, 6, 128, 128]);  view_805 = None
        view_808: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_807, [192, 128, 128])
        add_206: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_192, view_807);  add_192 = view_807 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_78: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_501, view_808);  permute_501 = None
        bmm_79: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_808, permute_502);  view_808 = permute_502 = None
        view_810: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_78, [32, 6, 64, 128]);  bmm_78 = None
        view_811: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_79, [32, 6, 128, 64]);  bmm_79 = None
        permute_503: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_810, [0, 1, 3, 2]);  view_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_504: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_803, [0, 2, 1, 3]);  view_803 = None
        clone_158: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_504, memory_format = torch.contiguous_format);  permute_504 = None
        view_812: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_158, [32, 128, 384]);  clone_158 = None
        view_813: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_812, [4096, 384]);  view_812 = None
        permute_505: "f32[384, 4096]" = torch.ops.aten.permute.default(view_813, [1, 0])
        mm_229: "f32[384, 512]" = torch.ops.aten.mm.default(permute_505, view_410);  permute_505 = None
        permute_186: "f32[512, 384]" = torch.ops.aten.permute.default(primals_138, [1, 0]);  primals_138 = None
        permute_507: "f32[384, 512]" = torch.ops.aten.permute.default(permute_186, [1, 0]);  permute_186 = None
        mm_230: "f32[4096, 512]" = torch.ops.aten.mm.default(view_813, permute_507);  view_813 = permute_507 = None
        view_814: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_230, [32, 128, 512]);  mm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_509: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_503, [0, 2, 1, 3]);  permute_503 = None
        view_815: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_509, [32, 128, 384]);  permute_509 = None
        clone_159: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_815, memory_format = torch.contiguous_format);  view_815 = None
        view_816: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_159, [4096, 384]);  clone_159 = None
        permute_510: "f32[384, 4096]" = torch.ops.aten.permute.default(view_816, [1, 0])
        mm_231: "f32[384, 512]" = torch.ops.aten.mm.default(permute_510, view_410);  permute_510 = None
        permute_184: "f32[512, 384]" = torch.ops.aten.permute.default(primals_137, [1, 0]);  primals_137 = None
        permute_512: "f32[384, 512]" = torch.ops.aten.permute.default(permute_184, [1, 0]);  permute_184 = None
        mm_232: "f32[4096, 512]" = torch.ops.aten.mm.default(view_816, permute_512);  view_816 = permute_512 = None
        view_817: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_232, [32, 128, 512]);  mm_232 = None
        add_207: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_814, view_817);  view_814 = view_817 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_514: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None
        clone_160: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_514, memory_format = torch.contiguous_format);  permute_514 = None
        view_818: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_160, [32, 128, 384]);  clone_160 = None
        view_819: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_818, [4096, 384]);  view_818 = None
        permute_515: "f32[384, 4096]" = torch.ops.aten.permute.default(view_819, [1, 0])
        mm_233: "f32[384, 512]" = torch.ops.aten.mm.default(permute_515, view_410);  permute_515 = view_410 = None
        permute_182: "f32[512, 384]" = torch.ops.aten.permute.default(primals_136, [1, 0]);  primals_136 = None
        permute_517: "f32[384, 512]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        mm_234: "f32[4096, 512]" = torch.ops.aten.mm.default(view_819, permute_517);  view_819 = permute_517 = None
        view_820: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_234, [32, 128, 512]);  mm_234 = None
        add_208: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_207, view_820);  add_207 = view_820 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_535: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_208, primals_135);  primals_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_239: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_112, rsqrt_29)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_536: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_208, mul_239);  add_208 = mul_239 = None
        sum_61: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_536, [0, 1], True);  mul_536 = None
        view_821: "f32[512]" = torch.ops.aten.reshape.default(sum_61, [512]);  sum_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_537: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_535, add_112)
        mul_538: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_535, rsqrt_29);  mul_535 = None
        sum_62: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_537, [2], True);  mul_537 = None
        add_209: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_205, mul_538);  add_205 = mul_538 = None
        pow_87: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_29, 3);  rsqrt_29 = None
        mul_539: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_62, -0.5);  sum_62 = None
        mul_540: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_539, pow_87);  mul_539 = pow_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_111: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_540, [32, 128, 512]);  mul_540 = None
        div_42: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_111, 512);  expand_111 = None
        pow_88: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_112, 1.0);  add_112 = None
        mul_541: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_88, 2.0);  pow_88 = None
        mul_542: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_42, mul_541);  div_42 = mul_541 = None
        add_210: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_209, mul_542);  add_209 = mul_542 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_31: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_59, torch.float32);  gt_59 = None
        mul_543: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_31, 1.1111111111111112);  convert_element_type_31 = None
        mul_544: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_210, mul_543);  mul_543 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_822: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_544, [4096, 512]);  mul_544 = None
        permute_519: "f32[512, 4096]" = torch.ops.aten.permute.default(view_822, [1, 0])
        mm_235: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_519, view_408);  permute_519 = view_408 = None
        permute_181: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_134, [1, 0]);  primals_134 = None
        permute_521: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        mm_236: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_822, permute_521);  view_822 = permute_521 = None
        view_823: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_236, [32, 128, 1024]);  mm_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_32: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_58, torch.float32);  gt_58 = None
        mul_545: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_32, 1.1111111111111112);  convert_element_type_32 = None
        mul_546: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_823, mul_545);  view_823 = mul_545 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_405: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_97, [32, 128, 1024]);  mm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_230: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_405, 0.5)
        pow_41: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_405, 3.0)
        mul_231: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_41, 0.044715);  pow_41 = None
        add_110: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_405, mul_231);  mul_231 = None
        mul_232: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_110, 0.7978845608028654);  add_110 = None
        tanh_11: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_232);  mul_232 = None
        add_111: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_11, 1.0)
        mul_233: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_230, add_111)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_547: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_546, mul_233);  mul_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_407: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_98, [32, 128, 1024]);  mm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_548: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_546, view_407);  mul_546 = view_407 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_824: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_547, [4096, 1024]);  mul_547 = None
        permute_523: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_824, [1, 0])
        mm_237: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_523, view_404);  permute_523 = None
        permute_180: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_133, [1, 0]);  primals_133 = None
        permute_525: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        mm_238: "f32[4096, 512]" = torch.ops.aten.mm.default(view_824, permute_525);  view_824 = permute_525 = None
        view_825: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_238, [32, 128, 512]);  mm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_549: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_548, mul_230);  mul_230 = None
        mul_550: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_548, add_111);  mul_548 = add_111 = None
        mul_551: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_11, tanh_11);  tanh_11 = None
        sub_33: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_551);  mul_551 = None
        mul_552: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_549, sub_33);  mul_549 = sub_33 = None
        mul_553: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_552, 0.7978845608028654);  mul_552 = None
        mul_554: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_553, 0.044715)
        pow_89: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_405, 2.0);  view_405 = None
        mul_555: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_89, 3.0);  pow_89 = None
        mul_556: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_554, mul_555);  mul_554 = mul_555 = None
        add_211: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_553, mul_556);  mul_553 = mul_556 = None
        mul_557: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_550, 0.5);  mul_550 = None
        add_212: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_211, mul_557);  add_211 = mul_557 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_826: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_212, [4096, 1024]);  add_212 = None
        permute_527: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_826, [1, 0])
        mm_239: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_527, view_404);  permute_527 = view_404 = None
        permute_179: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_132, [1, 0]);  primals_132 = None
        permute_529: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        mm_240: "f32[4096, 512]" = torch.ops.aten.mm.default(view_826, permute_529);  view_826 = permute_529 = None
        view_827: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_240, [32, 128, 512]);  mm_240 = None
        add_213: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_825, view_827);  view_825 = view_827 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_558: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, primals_131);  primals_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_228: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_108, rsqrt_28)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_559: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_213, mul_228);  add_213 = mul_228 = None
        sum_63: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_559, [0, 1], True);  mul_559 = None
        view_828: "f32[512]" = torch.ops.aten.reshape.default(sum_63, [512]);  sum_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_560: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_558, add_108)
        mul_561: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_558, rsqrt_28);  mul_558 = None
        sum_64: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_560, [2], True);  mul_560 = None
        add_214: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_210, mul_561);  add_210 = mul_561 = None
        pow_90: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_28, 3);  rsqrt_28 = None
        mul_562: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_64, -0.5);  sum_64 = None
        mul_563: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_562, pow_90);  mul_562 = pow_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_112: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_563, [32, 128, 512]);  mul_563 = None
        div_43: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_112, 512);  expand_112 = None
        pow_91: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_108, 1.0);  add_108 = None
        mul_564: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_91, 2.0);  pow_91 = None
        mul_565: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_43, mul_564);  div_43 = mul_564 = None
        add_215: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_214, mul_565);  add_214 = mul_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_33: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_57, torch.float32);  gt_57 = None
        mul_566: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_33, 1.1111111111111112);  convert_element_type_33 = None
        mul_567: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_215, mul_566);  mul_566 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_829: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_567, [4096, 512]);  mul_567 = None
        permute_531: "f32[512, 4096]" = torch.ops.aten.permute.default(view_829, [1, 0])
        mm_241: "f32[512, 384]" = torch.ops.aten.mm.default(permute_531, view_402);  permute_531 = view_402 = None
        permute_178: "f32[384, 512]" = torch.ops.aten.permute.default(primals_130, [1, 0]);  primals_130 = None
        permute_533: "f32[512, 384]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        mm_242: "f32[4096, 384]" = torch.ops.aten.mm.default(view_829, permute_533);  view_829 = permute_533 = None
        view_830: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_242, [32, 128, 384]);  mm_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_831: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_830, [32, 128, 6, 64]);  view_830 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_535: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_831, [0, 2, 1, 3]);  view_831 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_164: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_535, memory_format = torch.contiguous_format);  permute_535 = None
        view_832: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_164, [192, 128, 64]);  clone_164 = None
        bmm_80: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_536, view_832);  permute_536 = None
        bmm_81: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_832, permute_537);  view_832 = permute_537 = None
        view_833: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_80, [32, 6, 128, 64]);  bmm_80 = None
        view_834: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_81, [32, 6, 128, 128]);  bmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_34: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_56, torch.float32);  gt_56 = None
        mul_568: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_34, 1.1111111111111112);  convert_element_type_34 = None
        mul_569: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_834, mul_568);  view_834 = mul_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_570: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_569, div_19);  mul_569 = None
        sum_65: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_570, [-1], True)
        neg_10: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_19);  div_19 = None
        fma_8: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_10, sum_65, mul_570);  neg_10 = sum_65 = mul_570 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_835: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_8, [192, 128, 128]);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_82: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_538, view_835);  permute_538 = None
        bmm_83: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_835, permute_539);  view_835 = permute_539 = None
        view_840: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_82, [32, 6, 64, 128]);  bmm_82 = None
        view_841: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_83, [32, 6, 128, 64]);  bmm_83 = None
        permute_540: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_840, [0, 1, 3, 2]);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_541: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None
        clone_167: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_541, memory_format = torch.contiguous_format);  permute_541 = None
        view_842: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_167, [32, 128, 384]);  clone_167 = None
        view_843: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_842, [4096, 384]);  view_842 = None
        permute_542: "f32[384, 4096]" = torch.ops.aten.permute.default(view_843, [1, 0])
        view_389: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_243: "f32[384, 512]" = torch.ops.aten.mm.default(permute_542, view_389);  permute_542 = view_389 = None
        permute_174: "f32[512, 384]" = torch.ops.aten.permute.default(primals_129, [1, 0]);  primals_129 = None
        permute_544: "f32[384, 512]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        mm_244: "f32[4096, 512]" = torch.ops.aten.mm.default(view_843, permute_544);  view_843 = permute_544 = None
        view_844: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_244, [32, 128, 512]);  mm_244 = None
        add_216: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_203, view_844);  add_203 = view_844 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_546: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_540, [0, 2, 1, 3]);  permute_540 = None
        view_845: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_546, [32, 128, 384]);  permute_546 = None
        clone_168: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_845, memory_format = torch.contiguous_format);  view_845 = None
        view_846: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_168, [4096, 384]);  clone_168 = None
        permute_547: "f32[384, 4096]" = torch.ops.aten.permute.default(view_846, [1, 0])
        view_386: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_245: "f32[384, 512]" = torch.ops.aten.mm.default(permute_547, view_386);  permute_547 = view_386 = None
        permute_172: "f32[512, 384]" = torch.ops.aten.permute.default(primals_128, [1, 0]);  primals_128 = None
        permute_549: "f32[384, 512]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        mm_246: "f32[4096, 512]" = torch.ops.aten.mm.default(view_846, permute_549);  view_846 = permute_549 = None
        view_847: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_246, [32, 128, 512]);  mm_246 = None
        add_217: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_216, view_847);  add_216 = view_847 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_551: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_841, [0, 2, 1, 3]);  view_841 = None
        clone_169: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_551, memory_format = torch.contiguous_format);  permute_551 = None
        view_848: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_169, [32, 128, 384]);  clone_169 = None
        view_849: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_848, [4096, 384]);  view_848 = None
        permute_552: "f32[384, 4096]" = torch.ops.aten.permute.default(view_849, [1, 0])
        mm_247: "f32[384, 512]" = torch.ops.aten.mm.default(permute_552, view_383);  permute_552 = view_383 = None
        permute_170: "f32[512, 384]" = torch.ops.aten.permute.default(primals_127, [1, 0]);  primals_127 = None
        permute_554: "f32[384, 512]" = torch.ops.aten.permute.default(permute_170, [1, 0]);  permute_170 = None
        mm_248: "f32[4096, 512]" = torch.ops.aten.mm.default(view_849, permute_554);  view_849 = permute_554 = None
        view_850: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_248, [32, 128, 512]);  mm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_571: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_850, primals_126);  primals_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_222: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_105, rsqrt_27)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_572: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_850, mul_222);  view_850 = mul_222 = None
        sum_66: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_572, [0, 1], True);  mul_572 = None
        view_851: "f32[512]" = torch.ops.aten.reshape.default(sum_66, [512]);  sum_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_573: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_571, add_105)
        mul_574: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_571, rsqrt_27);  mul_571 = None
        sum_67: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_573, [2], True);  mul_573 = None
        add_218: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_215, mul_574);  add_215 = mul_574 = None
        pow_92: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_27, 3);  rsqrt_27 = None
        mul_575: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_67, -0.5);  sum_67 = None
        mul_576: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_575, pow_92);  mul_575 = pow_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_113: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_576, [32, 128, 512]);  mul_576 = None
        div_44: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_113, 512);  expand_113 = None
        pow_93: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_105, 1.0);  add_105 = None
        mul_577: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_93, 2.0);  pow_93 = None
        mul_578: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_44, mul_577);  div_44 = mul_577 = None
        add_219: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_218, mul_578);  add_218 = mul_578 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_35: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_55, torch.float32);  gt_55 = None
        mul_579: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_35, 1.1111111111111112);  convert_element_type_35 = None
        mul_580: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_219, mul_579);  mul_579 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_852: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_580, [4096, 512]);  mul_580 = None
        permute_556: "f32[512, 4096]" = torch.ops.aten.permute.default(view_852, [1, 0])
        mm_249: "f32[512, 384]" = torch.ops.aten.mm.default(permute_556, view_381);  permute_556 = view_381 = None
        permute_169: "f32[384, 512]" = torch.ops.aten.permute.default(primals_125, [1, 0]);  primals_125 = None
        permute_558: "f32[512, 384]" = torch.ops.aten.permute.default(permute_169, [1, 0]);  permute_169 = None
        mm_250: "f32[4096, 384]" = torch.ops.aten.mm.default(view_852, permute_558);  view_852 = permute_558 = None
        view_853: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_250, [32, 128, 384]);  mm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_854: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_853, [32, 128, 6, 64]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_560: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_854, [0, 2, 1, 3]);  view_854 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_171: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_560, memory_format = torch.contiguous_format);  permute_560 = None
        view_855: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_171, [192, 128, 64]);  clone_171 = None
        bmm_84: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_561, view_855);  permute_561 = None
        bmm_85: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_855, permute_562);  view_855 = permute_562 = None
        view_856: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_84, [32, 6, 128, 64]);  bmm_84 = None
        view_857: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_85, [32, 6, 128, 128]);  bmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_36: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_54, torch.float32);  gt_54 = None
        mul_581: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_36, 1.1111111111111112);  convert_element_type_36 = None
        mul_582: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_857, mul_581);  view_857 = mul_581 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_583: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_582, div_18);  mul_582 = None
        sum_68: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_583, [-1], True)
        neg_11: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_18);  div_18 = None
        fma_9: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_11, sum_68, mul_583);  neg_11 = sum_68 = mul_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_858: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_9, [192, 128, 128]);  fma_9 = None
        view_860: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_858, [32, 6, 128, 128]);  view_858 = None
        view_861: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_860, [192, 128, 128])
        add_220: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_206, view_860);  add_206 = view_860 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_86: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_563, view_861);  permute_563 = None
        bmm_87: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_861, permute_564);  view_861 = permute_564 = None
        view_863: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_86, [32, 6, 64, 128]);  bmm_86 = None
        view_864: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_87, [32, 6, 128, 64]);  bmm_87 = None
        permute_565: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_863, [0, 1, 3, 2]);  view_863 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_566: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None
        clone_174: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_566, memory_format = torch.contiguous_format);  permute_566 = None
        view_865: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_174, [32, 128, 384]);  clone_174 = None
        view_866: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_865, [4096, 384]);  view_865 = None
        permute_567: "f32[384, 4096]" = torch.ops.aten.permute.default(view_866, [1, 0])
        mm_251: "f32[384, 512]" = torch.ops.aten.mm.default(permute_567, view_362);  permute_567 = None
        permute_165: "f32[512, 384]" = torch.ops.aten.permute.default(primals_124, [1, 0]);  primals_124 = None
        permute_569: "f32[384, 512]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        mm_252: "f32[4096, 512]" = torch.ops.aten.mm.default(view_866, permute_569);  view_866 = permute_569 = None
        view_867: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_252, [32, 128, 512]);  mm_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_571: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_565, [0, 2, 1, 3]);  permute_565 = None
        view_868: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_571, [32, 128, 384]);  permute_571 = None
        clone_175: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_868, memory_format = torch.contiguous_format);  view_868 = None
        view_869: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_175, [4096, 384]);  clone_175 = None
        permute_572: "f32[384, 4096]" = torch.ops.aten.permute.default(view_869, [1, 0])
        mm_253: "f32[384, 512]" = torch.ops.aten.mm.default(permute_572, view_362);  permute_572 = None
        permute_163: "f32[512, 384]" = torch.ops.aten.permute.default(primals_123, [1, 0]);  primals_123 = None
        permute_574: "f32[384, 512]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        mm_254: "f32[4096, 512]" = torch.ops.aten.mm.default(view_869, permute_574);  view_869 = permute_574 = None
        view_870: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_254, [32, 128, 512]);  mm_254 = None
        add_221: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_867, view_870);  view_867 = view_870 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_576: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_864, [0, 2, 1, 3]);  view_864 = None
        clone_176: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_576, memory_format = torch.contiguous_format);  permute_576 = None
        view_871: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_176, [32, 128, 384]);  clone_176 = None
        view_872: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_871, [4096, 384]);  view_871 = None
        permute_577: "f32[384, 4096]" = torch.ops.aten.permute.default(view_872, [1, 0])
        mm_255: "f32[384, 512]" = torch.ops.aten.mm.default(permute_577, view_362);  permute_577 = view_362 = None
        permute_161: "f32[512, 384]" = torch.ops.aten.permute.default(primals_122, [1, 0]);  primals_122 = None
        permute_579: "f32[384, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        mm_256: "f32[4096, 512]" = torch.ops.aten.mm.default(view_872, permute_579);  view_872 = permute_579 = None
        view_873: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_256, [32, 128, 512]);  mm_256 = None
        add_222: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_221, view_873);  add_221 = view_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_584: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_222, primals_121);  primals_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_216: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_102, rsqrt_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_585: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_222, mul_216);  add_222 = mul_216 = None
        sum_69: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_585, [0, 1], True);  mul_585 = None
        view_874: "f32[512]" = torch.ops.aten.reshape.default(sum_69, [512]);  sum_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_586: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_584, add_102)
        mul_587: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_584, rsqrt_26);  mul_584 = None
        sum_70: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_586, [2], True);  mul_586 = None
        add_223: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_219, mul_587);  add_219 = mul_587 = None
        pow_94: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_26, 3);  rsqrt_26 = None
        mul_588: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_70, -0.5);  sum_70 = None
        mul_589: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_588, pow_94);  mul_588 = pow_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_114: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_589, [32, 128, 512]);  mul_589 = None
        div_45: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_114, 512);  expand_114 = None
        pow_95: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_102, 1.0);  add_102 = None
        mul_590: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_95, 2.0);  pow_95 = None
        mul_591: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_45, mul_590);  div_45 = mul_590 = None
        add_224: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_223, mul_591);  add_223 = mul_591 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_37: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_53, torch.float32);  gt_53 = None
        mul_592: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_37, 1.1111111111111112);  convert_element_type_37 = None
        mul_593: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_224, mul_592);  mul_592 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_875: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_593, [4096, 512]);  mul_593 = None
        permute_581: "f32[512, 4096]" = torch.ops.aten.permute.default(view_875, [1, 0])
        mm_257: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_581, view_360);  permute_581 = view_360 = None
        permute_160: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_120, [1, 0]);  primals_120 = None
        permute_583: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        mm_258: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_875, permute_583);  view_875 = permute_583 = None
        view_876: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_258, [32, 128, 1024]);  mm_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_38: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_52, torch.float32);  gt_52 = None
        mul_594: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_38, 1.1111111111111112);  convert_element_type_38 = None
        mul_595: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_876, mul_594);  view_876 = mul_594 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_357: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_86, [32, 128, 1024]);  mm_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_207: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_357, 0.5)
        pow_37: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_357, 3.0)
        mul_208: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_37, 0.044715);  pow_37 = None
        add_100: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_357, mul_208);  mul_208 = None
        mul_209: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_100, 0.7978845608028654);  add_100 = None
        tanh_10: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_209);  mul_209 = None
        add_101: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_10, 1.0)
        mul_210: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_207, add_101)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_596: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_595, mul_210);  mul_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_359: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_87, [32, 128, 1024]);  mm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_597: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_595, view_359);  mul_595 = view_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_877: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_596, [4096, 1024]);  mul_596 = None
        permute_585: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_877, [1, 0])
        mm_259: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_585, view_356);  permute_585 = None
        permute_159: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_119, [1, 0]);  primals_119 = None
        permute_587: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        mm_260: "f32[4096, 512]" = torch.ops.aten.mm.default(view_877, permute_587);  view_877 = permute_587 = None
        view_878: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_260, [32, 128, 512]);  mm_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_598: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_597, mul_207);  mul_207 = None
        mul_599: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_597, add_101);  mul_597 = add_101 = None
        mul_600: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_10, tanh_10);  tanh_10 = None
        sub_34: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_600);  mul_600 = None
        mul_601: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_598, sub_34);  mul_598 = sub_34 = None
        mul_602: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_601, 0.7978845608028654);  mul_601 = None
        mul_603: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_602, 0.044715)
        pow_96: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_357, 2.0);  view_357 = None
        mul_604: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_96, 3.0);  pow_96 = None
        mul_605: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_603, mul_604);  mul_603 = mul_604 = None
        add_225: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_602, mul_605);  mul_602 = mul_605 = None
        mul_606: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_599, 0.5);  mul_599 = None
        add_226: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_225, mul_606);  add_225 = mul_606 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_879: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_226, [4096, 1024]);  add_226 = None
        permute_589: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_879, [1, 0])
        mm_261: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_589, view_356);  permute_589 = view_356 = None
        permute_158: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_118, [1, 0]);  primals_118 = None
        permute_591: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        mm_262: "f32[4096, 512]" = torch.ops.aten.mm.default(view_879, permute_591);  view_879 = permute_591 = None
        view_880: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_262, [32, 128, 512]);  mm_262 = None
        add_227: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_878, view_880);  view_878 = view_880 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_607: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_227, primals_117);  primals_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_205: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_98, rsqrt_25)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_608: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_227, mul_205);  add_227 = mul_205 = None
        sum_71: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_608, [0, 1], True);  mul_608 = None
        view_881: "f32[512]" = torch.ops.aten.reshape.default(sum_71, [512]);  sum_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_609: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_607, add_98)
        mul_610: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_607, rsqrt_25);  mul_607 = None
        sum_72: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_609, [2], True);  mul_609 = None
        add_228: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_224, mul_610);  add_224 = mul_610 = None
        pow_97: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_25, 3);  rsqrt_25 = None
        mul_611: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_72, -0.5);  sum_72 = None
        mul_612: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_611, pow_97);  mul_611 = pow_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_115: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_612, [32, 128, 512]);  mul_612 = None
        div_46: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_115, 512);  expand_115 = None
        pow_98: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_98, 1.0);  add_98 = None
        mul_613: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_98, 2.0);  pow_98 = None
        mul_614: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_46, mul_613);  div_46 = mul_613 = None
        add_229: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_228, mul_614);  add_228 = mul_614 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_39: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_51, torch.float32);  gt_51 = None
        mul_615: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_39, 1.1111111111111112);  convert_element_type_39 = None
        mul_616: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_229, mul_615);  mul_615 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_882: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_616, [4096, 512]);  mul_616 = None
        permute_593: "f32[512, 4096]" = torch.ops.aten.permute.default(view_882, [1, 0])
        mm_263: "f32[512, 384]" = torch.ops.aten.mm.default(permute_593, view_354);  permute_593 = view_354 = None
        permute_157: "f32[384, 512]" = torch.ops.aten.permute.default(primals_116, [1, 0]);  primals_116 = None
        permute_595: "f32[512, 384]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        mm_264: "f32[4096, 384]" = torch.ops.aten.mm.default(view_882, permute_595);  view_882 = permute_595 = None
        view_883: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_264, [32, 128, 384]);  mm_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_884: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_883, [32, 128, 6, 64]);  view_883 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_597: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_884, [0, 2, 1, 3]);  view_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_180: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_597, memory_format = torch.contiguous_format);  permute_597 = None
        view_885: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_180, [192, 128, 64]);  clone_180 = None
        bmm_88: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_598, view_885);  permute_598 = None
        bmm_89: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_885, permute_599);  view_885 = permute_599 = None
        view_886: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_88, [32, 6, 128, 64]);  bmm_88 = None
        view_887: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_89, [32, 6, 128, 128]);  bmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_40: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_50, torch.float32);  gt_50 = None
        mul_617: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_40, 1.1111111111111112);  convert_element_type_40 = None
        mul_618: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_887, mul_617);  view_887 = mul_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_619: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_618, div_17);  mul_618 = None
        sum_73: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_619, [-1], True)
        neg_12: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_17);  div_17 = None
        fma_10: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_12, sum_73, mul_619);  neg_12 = sum_73 = mul_619 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_888: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_10, [192, 128, 128]);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_90: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_600, view_888);  permute_600 = None
        bmm_91: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_888, permute_601);  view_888 = permute_601 = None
        view_893: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_90, [32, 6, 64, 128]);  bmm_90 = None
        view_894: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_91, [32, 6, 128, 64]);  bmm_91 = None
        permute_602: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_893, [0, 1, 3, 2]);  view_893 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_603: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None
        clone_183: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_603, memory_format = torch.contiguous_format);  permute_603 = None
        view_895: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_183, [32, 128, 384]);  clone_183 = None
        view_896: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_895, [4096, 384]);  view_895 = None
        permute_604: "f32[384, 4096]" = torch.ops.aten.permute.default(view_896, [1, 0])
        view_341: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_265: "f32[384, 512]" = torch.ops.aten.mm.default(permute_604, view_341);  permute_604 = view_341 = None
        permute_153: "f32[512, 384]" = torch.ops.aten.permute.default(primals_115, [1, 0]);  primals_115 = None
        permute_606: "f32[384, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        mm_266: "f32[4096, 512]" = torch.ops.aten.mm.default(view_896, permute_606);  view_896 = permute_606 = None
        view_897: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_266, [32, 128, 512]);  mm_266 = None
        add_230: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_217, view_897);  add_217 = view_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_608: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_602, [0, 2, 1, 3]);  permute_602 = None
        view_898: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_608, [32, 128, 384]);  permute_608 = None
        clone_184: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_898, memory_format = torch.contiguous_format);  view_898 = None
        view_899: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_184, [4096, 384]);  clone_184 = None
        permute_609: "f32[384, 4096]" = torch.ops.aten.permute.default(view_899, [1, 0])
        view_338: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_267: "f32[384, 512]" = torch.ops.aten.mm.default(permute_609, view_338);  permute_609 = view_338 = None
        permute_151: "f32[512, 384]" = torch.ops.aten.permute.default(primals_114, [1, 0]);  primals_114 = None
        permute_611: "f32[384, 512]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        mm_268: "f32[4096, 512]" = torch.ops.aten.mm.default(view_899, permute_611);  view_899 = permute_611 = None
        view_900: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_268, [32, 128, 512]);  mm_268 = None
        add_231: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_230, view_900);  add_230 = view_900 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_613: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_894, [0, 2, 1, 3]);  view_894 = None
        clone_185: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_613, memory_format = torch.contiguous_format);  permute_613 = None
        view_901: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_185, [32, 128, 384]);  clone_185 = None
        view_902: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_901, [4096, 384]);  view_901 = None
        permute_614: "f32[384, 4096]" = torch.ops.aten.permute.default(view_902, [1, 0])
        mm_269: "f32[384, 512]" = torch.ops.aten.mm.default(permute_614, view_335);  permute_614 = view_335 = None
        permute_149: "f32[512, 384]" = torch.ops.aten.permute.default(primals_113, [1, 0]);  primals_113 = None
        permute_616: "f32[384, 512]" = torch.ops.aten.permute.default(permute_149, [1, 0]);  permute_149 = None
        mm_270: "f32[4096, 512]" = torch.ops.aten.mm.default(view_902, permute_616);  view_902 = permute_616 = None
        view_903: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_270, [32, 128, 512]);  mm_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_620: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_903, primals_112);  primals_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_199: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_95, rsqrt_24)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_621: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_903, mul_199);  view_903 = mul_199 = None
        sum_74: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_621, [0, 1], True);  mul_621 = None
        view_904: "f32[512]" = torch.ops.aten.reshape.default(sum_74, [512]);  sum_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_622: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_620, add_95)
        mul_623: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_620, rsqrt_24);  mul_620 = None
        sum_75: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_622, [2], True);  mul_622 = None
        add_232: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_229, mul_623);  add_229 = mul_623 = None
        pow_99: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_24, 3);  rsqrt_24 = None
        mul_624: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_75, -0.5);  sum_75 = None
        mul_625: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_624, pow_99);  mul_624 = pow_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_116: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_625, [32, 128, 512]);  mul_625 = None
        div_47: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_116, 512);  expand_116 = None
        pow_100: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_95, 1.0);  add_95 = None
        mul_626: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_100, 2.0);  pow_100 = None
        mul_627: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_47, mul_626);  div_47 = mul_626 = None
        add_233: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_232, mul_627);  add_232 = mul_627 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_41: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_49, torch.float32);  gt_49 = None
        mul_628: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_41, 1.1111111111111112);  convert_element_type_41 = None
        mul_629: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_233, mul_628);  mul_628 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_905: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_629, [4096, 512]);  mul_629 = None
        permute_618: "f32[512, 4096]" = torch.ops.aten.permute.default(view_905, [1, 0])
        mm_271: "f32[512, 384]" = torch.ops.aten.mm.default(permute_618, view_333);  permute_618 = view_333 = None
        permute_148: "f32[384, 512]" = torch.ops.aten.permute.default(primals_111, [1, 0]);  primals_111 = None
        permute_620: "f32[512, 384]" = torch.ops.aten.permute.default(permute_148, [1, 0]);  permute_148 = None
        mm_272: "f32[4096, 384]" = torch.ops.aten.mm.default(view_905, permute_620);  view_905 = permute_620 = None
        view_906: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_272, [32, 128, 384]);  mm_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_907: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_906, [32, 128, 6, 64]);  view_906 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_622: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_907, [0, 2, 1, 3]);  view_907 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_187: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_622, memory_format = torch.contiguous_format);  permute_622 = None
        view_908: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_187, [192, 128, 64]);  clone_187 = None
        bmm_92: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_623, view_908);  permute_623 = None
        bmm_93: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_908, permute_624);  view_908 = permute_624 = None
        view_909: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_92, [32, 6, 128, 64]);  bmm_92 = None
        view_910: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_93, [32, 6, 128, 128]);  bmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_42: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_48, torch.float32);  gt_48 = None
        mul_630: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_42, 1.1111111111111112);  convert_element_type_42 = None
        mul_631: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_910, mul_630);  view_910 = mul_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_632: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_631, div_16);  mul_631 = None
        sum_76: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_632, [-1], True)
        neg_13: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_16);  div_16 = None
        fma_11: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_13, sum_76, mul_632);  neg_13 = sum_76 = mul_632 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_911: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_11, [192, 128, 128]);  fma_11 = None
        view_913: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_911, [32, 6, 128, 128]);  view_911 = None
        view_914: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_913, [192, 128, 128])
        add_234: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_220, view_913);  add_220 = view_913 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_94: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_625, view_914);  permute_625 = None
        bmm_95: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_914, permute_626);  view_914 = permute_626 = None
        view_916: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_94, [32, 6, 64, 128]);  bmm_94 = None
        view_917: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_95, [32, 6, 128, 64]);  bmm_95 = None
        permute_627: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_916, [0, 1, 3, 2]);  view_916 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_628: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_909, [0, 2, 1, 3]);  view_909 = None
        clone_190: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_628, memory_format = torch.contiguous_format);  permute_628 = None
        view_918: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_190, [32, 128, 384]);  clone_190 = None
        view_919: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_918, [4096, 384]);  view_918 = None
        permute_629: "f32[384, 4096]" = torch.ops.aten.permute.default(view_919, [1, 0])
        mm_273: "f32[384, 512]" = torch.ops.aten.mm.default(permute_629, view_314);  permute_629 = None
        permute_144: "f32[512, 384]" = torch.ops.aten.permute.default(primals_110, [1, 0]);  primals_110 = None
        permute_631: "f32[384, 512]" = torch.ops.aten.permute.default(permute_144, [1, 0]);  permute_144 = None
        mm_274: "f32[4096, 512]" = torch.ops.aten.mm.default(view_919, permute_631);  view_919 = permute_631 = None
        view_920: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_274, [32, 128, 512]);  mm_274 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_633: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_627, [0, 2, 1, 3]);  permute_627 = None
        view_921: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_633, [32, 128, 384]);  permute_633 = None
        clone_191: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_921, memory_format = torch.contiguous_format);  view_921 = None
        view_922: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_191, [4096, 384]);  clone_191 = None
        permute_634: "f32[384, 4096]" = torch.ops.aten.permute.default(view_922, [1, 0])
        mm_275: "f32[384, 512]" = torch.ops.aten.mm.default(permute_634, view_314);  permute_634 = None
        permute_142: "f32[512, 384]" = torch.ops.aten.permute.default(primals_109, [1, 0]);  primals_109 = None
        permute_636: "f32[384, 512]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        mm_276: "f32[4096, 512]" = torch.ops.aten.mm.default(view_922, permute_636);  view_922 = permute_636 = None
        view_923: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_276, [32, 128, 512]);  mm_276 = None
        add_235: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_920, view_923);  view_920 = view_923 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_638: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_917, [0, 2, 1, 3]);  view_917 = None
        clone_192: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_638, memory_format = torch.contiguous_format);  permute_638 = None
        view_924: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_192, [32, 128, 384]);  clone_192 = None
        view_925: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_924, [4096, 384]);  view_924 = None
        permute_639: "f32[384, 4096]" = torch.ops.aten.permute.default(view_925, [1, 0])
        mm_277: "f32[384, 512]" = torch.ops.aten.mm.default(permute_639, view_314);  permute_639 = view_314 = None
        permute_140: "f32[512, 384]" = torch.ops.aten.permute.default(primals_108, [1, 0]);  primals_108 = None
        permute_641: "f32[384, 512]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        mm_278: "f32[4096, 512]" = torch.ops.aten.mm.default(view_925, permute_641);  view_925 = permute_641 = None
        view_926: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_278, [32, 128, 512]);  mm_278 = None
        add_236: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_235, view_926);  add_235 = view_926 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_633: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_236, primals_107);  primals_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_193: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_92, rsqrt_23)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_634: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_236, mul_193);  add_236 = mul_193 = None
        sum_77: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_634, [0, 1], True);  mul_634 = None
        view_927: "f32[512]" = torch.ops.aten.reshape.default(sum_77, [512]);  sum_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_635: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_633, add_92)
        mul_636: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_633, rsqrt_23);  mul_633 = None
        sum_78: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_635, [2], True);  mul_635 = None
        add_237: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_233, mul_636);  add_233 = mul_636 = None
        pow_101: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_23, 3);  rsqrt_23 = None
        mul_637: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_78, -0.5);  sum_78 = None
        mul_638: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_637, pow_101);  mul_637 = pow_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_117: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_638, [32, 128, 512]);  mul_638 = None
        div_48: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_117, 512);  expand_117 = None
        pow_102: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_92, 1.0);  add_92 = None
        mul_639: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_102, 2.0);  pow_102 = None
        mul_640: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_48, mul_639);  div_48 = mul_639 = None
        add_238: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_237, mul_640);  add_237 = mul_640 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_43: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_47, torch.float32);  gt_47 = None
        mul_641: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_43, 1.1111111111111112);  convert_element_type_43 = None
        mul_642: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_238, mul_641);  mul_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_928: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_642, [4096, 512]);  mul_642 = None
        permute_643: "f32[512, 4096]" = torch.ops.aten.permute.default(view_928, [1, 0])
        mm_279: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_643, view_312);  permute_643 = view_312 = None
        permute_139: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_106, [1, 0]);  primals_106 = None
        permute_645: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        mm_280: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_928, permute_645);  view_928 = permute_645 = None
        view_929: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_280, [32, 128, 1024]);  mm_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_44: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_46, torch.float32);  gt_46 = None
        mul_643: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_44, 1.1111111111111112);  convert_element_type_44 = None
        mul_644: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_929, mul_643);  view_929 = mul_643 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_309: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_75, [32, 128, 1024]);  mm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_184: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_309, 0.5)
        pow_33: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_309, 3.0)
        mul_185: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_33, 0.044715);  pow_33 = None
        add_90: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_309, mul_185);  mul_185 = None
        mul_186: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_90, 0.7978845608028654);  add_90 = None
        tanh_9: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_186);  mul_186 = None
        add_91: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_9, 1.0)
        mul_187: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_184, add_91)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_645: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_644, mul_187);  mul_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_311: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_76, [32, 128, 1024]);  mm_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_646: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_644, view_311);  mul_644 = view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_930: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_645, [4096, 1024]);  mul_645 = None
        permute_647: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_930, [1, 0])
        mm_281: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_647, view_308);  permute_647 = None
        permute_138: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_105, [1, 0]);  primals_105 = None
        permute_649: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
        mm_282: "f32[4096, 512]" = torch.ops.aten.mm.default(view_930, permute_649);  view_930 = permute_649 = None
        view_931: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_282, [32, 128, 512]);  mm_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_647: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_646, mul_184);  mul_184 = None
        mul_648: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_646, add_91);  mul_646 = add_91 = None
        mul_649: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_9, tanh_9);  tanh_9 = None
        sub_35: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_649);  mul_649 = None
        mul_650: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_647, sub_35);  mul_647 = sub_35 = None
        mul_651: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_650, 0.7978845608028654);  mul_650 = None
        mul_652: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_651, 0.044715)
        pow_103: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_309, 2.0);  view_309 = None
        mul_653: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_103, 3.0);  pow_103 = None
        mul_654: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_652, mul_653);  mul_652 = mul_653 = None
        add_239: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_651, mul_654);  mul_651 = mul_654 = None
        mul_655: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_648, 0.5);  mul_648 = None
        add_240: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_239, mul_655);  add_239 = mul_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_932: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_240, [4096, 1024]);  add_240 = None
        permute_651: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_932, [1, 0])
        mm_283: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_651, view_308);  permute_651 = view_308 = None
        permute_137: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_104, [1, 0]);  primals_104 = None
        permute_653: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        mm_284: "f32[4096, 512]" = torch.ops.aten.mm.default(view_932, permute_653);  view_932 = permute_653 = None
        view_933: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_284, [32, 128, 512]);  mm_284 = None
        add_241: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_931, view_933);  view_931 = view_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_656: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, primals_103);  primals_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_182: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_88, rsqrt_22)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_657: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_241, mul_182);  add_241 = mul_182 = None
        sum_79: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_657, [0, 1], True);  mul_657 = None
        view_934: "f32[512]" = torch.ops.aten.reshape.default(sum_79, [512]);  sum_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_658: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_656, add_88)
        mul_659: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_656, rsqrt_22);  mul_656 = None
        sum_80: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_658, [2], True);  mul_658 = None
        add_242: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_238, mul_659);  add_238 = mul_659 = None
        pow_104: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_22, 3);  rsqrt_22 = None
        mul_660: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_80, -0.5);  sum_80 = None
        mul_661: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_660, pow_104);  mul_660 = pow_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_118: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_661, [32, 128, 512]);  mul_661 = None
        div_49: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_118, 512);  expand_118 = None
        pow_105: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_88, 1.0);  add_88 = None
        mul_662: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_105, 2.0);  pow_105 = None
        mul_663: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_49, mul_662);  div_49 = mul_662 = None
        add_243: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_242, mul_663);  add_242 = mul_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_45: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_45, torch.float32);  gt_45 = None
        mul_664: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_45, 1.1111111111111112);  convert_element_type_45 = None
        mul_665: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_243, mul_664);  mul_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_935: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_665, [4096, 512]);  mul_665 = None
        permute_655: "f32[512, 4096]" = torch.ops.aten.permute.default(view_935, [1, 0])
        mm_285: "f32[512, 384]" = torch.ops.aten.mm.default(permute_655, view_306);  permute_655 = view_306 = None
        permute_136: "f32[384, 512]" = torch.ops.aten.permute.default(primals_102, [1, 0]);  primals_102 = None
        permute_657: "f32[512, 384]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        mm_286: "f32[4096, 384]" = torch.ops.aten.mm.default(view_935, permute_657);  view_935 = permute_657 = None
        view_936: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_286, [32, 128, 384]);  mm_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_937: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_936, [32, 128, 6, 64]);  view_936 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_659: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_937, [0, 2, 1, 3]);  view_937 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_196: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_659, memory_format = torch.contiguous_format);  permute_659 = None
        view_938: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_196, [192, 128, 64]);  clone_196 = None
        bmm_96: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_660, view_938);  permute_660 = None
        bmm_97: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_938, permute_661);  view_938 = permute_661 = None
        view_939: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_96, [32, 6, 128, 64]);  bmm_96 = None
        view_940: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_97, [32, 6, 128, 128]);  bmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_46: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_44, torch.float32);  gt_44 = None
        mul_666: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_46, 1.1111111111111112);  convert_element_type_46 = None
        mul_667: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_940, mul_666);  view_940 = mul_666 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_668: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_667, div_15);  mul_667 = None
        sum_81: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_668, [-1], True)
        neg_14: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_15);  div_15 = None
        fma_12: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_14, sum_81, mul_668);  neg_14 = sum_81 = mul_668 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_941: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_12, [192, 128, 128]);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_98: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_662, view_941);  permute_662 = None
        bmm_99: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_941, permute_663);  view_941 = permute_663 = None
        view_946: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_98, [32, 6, 64, 128]);  bmm_98 = None
        view_947: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_99, [32, 6, 128, 64]);  bmm_99 = None
        permute_664: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_946, [0, 1, 3, 2]);  view_946 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_665: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_939, [0, 2, 1, 3]);  view_939 = None
        clone_199: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_665, memory_format = torch.contiguous_format);  permute_665 = None
        view_948: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_199, [32, 128, 384]);  clone_199 = None
        view_949: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_948, [4096, 384]);  view_948 = None
        permute_666: "f32[384, 4096]" = torch.ops.aten.permute.default(view_949, [1, 0])
        view_293: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_287: "f32[384, 512]" = torch.ops.aten.mm.default(permute_666, view_293);  permute_666 = view_293 = None
        permute_132: "f32[512, 384]" = torch.ops.aten.permute.default(primals_101, [1, 0]);  primals_101 = None
        permute_668: "f32[384, 512]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        mm_288: "f32[4096, 512]" = torch.ops.aten.mm.default(view_949, permute_668);  view_949 = permute_668 = None
        view_950: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_288, [32, 128, 512]);  mm_288 = None
        add_244: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_231, view_950);  add_231 = view_950 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_670: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_664, [0, 2, 1, 3]);  permute_664 = None
        view_951: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_670, [32, 128, 384]);  permute_670 = None
        clone_200: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_951, memory_format = torch.contiguous_format);  view_951 = None
        view_952: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_200, [4096, 384]);  clone_200 = None
        permute_671: "f32[384, 4096]" = torch.ops.aten.permute.default(view_952, [1, 0])
        view_290: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_289: "f32[384, 512]" = torch.ops.aten.mm.default(permute_671, view_290);  permute_671 = view_290 = None
        permute_130: "f32[512, 384]" = torch.ops.aten.permute.default(primals_100, [1, 0]);  primals_100 = None
        permute_673: "f32[384, 512]" = torch.ops.aten.permute.default(permute_130, [1, 0]);  permute_130 = None
        mm_290: "f32[4096, 512]" = torch.ops.aten.mm.default(view_952, permute_673);  view_952 = permute_673 = None
        view_953: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_290, [32, 128, 512]);  mm_290 = None
        add_245: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_244, view_953);  add_244 = view_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_675: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_947, [0, 2, 1, 3]);  view_947 = None
        clone_201: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_675, memory_format = torch.contiguous_format);  permute_675 = None
        view_954: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_201, [32, 128, 384]);  clone_201 = None
        view_955: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_954, [4096, 384]);  view_954 = None
        permute_676: "f32[384, 4096]" = torch.ops.aten.permute.default(view_955, [1, 0])
        mm_291: "f32[384, 512]" = torch.ops.aten.mm.default(permute_676, view_287);  permute_676 = view_287 = None
        permute_128: "f32[512, 384]" = torch.ops.aten.permute.default(primals_99, [1, 0]);  primals_99 = None
        permute_678: "f32[384, 512]" = torch.ops.aten.permute.default(permute_128, [1, 0]);  permute_128 = None
        mm_292: "f32[4096, 512]" = torch.ops.aten.mm.default(view_955, permute_678);  view_955 = permute_678 = None
        view_956: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_292, [32, 128, 512]);  mm_292 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_669: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_956, primals_98);  primals_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_176: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, rsqrt_21)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_670: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_956, mul_176);  view_956 = mul_176 = None
        sum_82: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_670, [0, 1], True);  mul_670 = None
        view_957: "f32[512]" = torch.ops.aten.reshape.default(sum_82, [512]);  sum_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_671: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_669, add_85)
        mul_672: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_669, rsqrt_21);  mul_669 = None
        sum_83: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_671, [2], True);  mul_671 = None
        add_246: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_243, mul_672);  add_243 = mul_672 = None
        pow_106: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_21, 3);  rsqrt_21 = None
        mul_673: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_83, -0.5);  sum_83 = None
        mul_674: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_673, pow_106);  mul_673 = pow_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_119: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_674, [32, 128, 512]);  mul_674 = None
        div_50: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_119, 512);  expand_119 = None
        pow_107: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_85, 1.0);  add_85 = None
        mul_675: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_107, 2.0);  pow_107 = None
        mul_676: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_50, mul_675);  div_50 = mul_675 = None
        add_247: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_246, mul_676);  add_246 = mul_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_47: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_43, torch.float32);  gt_43 = None
        mul_677: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_47, 1.1111111111111112);  convert_element_type_47 = None
        mul_678: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_247, mul_677);  mul_677 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_958: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_678, [4096, 512]);  mul_678 = None
        permute_680: "f32[512, 4096]" = torch.ops.aten.permute.default(view_958, [1, 0])
        mm_293: "f32[512, 384]" = torch.ops.aten.mm.default(permute_680, view_285);  permute_680 = view_285 = None
        permute_127: "f32[384, 512]" = torch.ops.aten.permute.default(primals_97, [1, 0]);  primals_97 = None
        permute_682: "f32[512, 384]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        mm_294: "f32[4096, 384]" = torch.ops.aten.mm.default(view_958, permute_682);  view_958 = permute_682 = None
        view_959: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_294, [32, 128, 384]);  mm_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_960: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_959, [32, 128, 6, 64]);  view_959 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_684: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_960, [0, 2, 1, 3]);  view_960 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_203: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_684, memory_format = torch.contiguous_format);  permute_684 = None
        view_961: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_203, [192, 128, 64]);  clone_203 = None
        bmm_100: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_685, view_961);  permute_685 = None
        bmm_101: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_961, permute_686);  view_961 = permute_686 = None
        view_962: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_100, [32, 6, 128, 64]);  bmm_100 = None
        view_963: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_101, [32, 6, 128, 128]);  bmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_48: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_42, torch.float32);  gt_42 = None
        mul_679: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_48, 1.1111111111111112);  convert_element_type_48 = None
        mul_680: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_963, mul_679);  view_963 = mul_679 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_681: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_680, div_14);  mul_680 = None
        sum_84: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_681, [-1], True)
        neg_15: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_14);  div_14 = None
        fma_13: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_15, sum_84, mul_681);  neg_15 = sum_84 = mul_681 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_964: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_13, [192, 128, 128]);  fma_13 = None
        view_966: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_964, [32, 6, 128, 128]);  view_964 = None
        view_967: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_966, [192, 128, 128])
        add_248: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_234, view_966);  add_234 = view_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_102: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_687, view_967);  permute_687 = None
        bmm_103: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_967, permute_688);  view_967 = permute_688 = None
        view_969: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_102, [32, 6, 64, 128]);  bmm_102 = None
        view_970: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_103, [32, 6, 128, 64]);  bmm_103 = None
        permute_689: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_969, [0, 1, 3, 2]);  view_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_690: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_962, [0, 2, 1, 3]);  view_962 = None
        clone_206: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_690, memory_format = torch.contiguous_format);  permute_690 = None
        view_971: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_206, [32, 128, 384]);  clone_206 = None
        view_972: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_971, [4096, 384]);  view_971 = None
        permute_691: "f32[384, 4096]" = torch.ops.aten.permute.default(view_972, [1, 0])
        mm_295: "f32[384, 512]" = torch.ops.aten.mm.default(permute_691, view_266);  permute_691 = None
        permute_123: "f32[512, 384]" = torch.ops.aten.permute.default(primals_96, [1, 0]);  primals_96 = None
        permute_693: "f32[384, 512]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        mm_296: "f32[4096, 512]" = torch.ops.aten.mm.default(view_972, permute_693);  view_972 = permute_693 = None
        view_973: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_296, [32, 128, 512]);  mm_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_695: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_689, [0, 2, 1, 3]);  permute_689 = None
        view_974: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_695, [32, 128, 384]);  permute_695 = None
        clone_207: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_974, memory_format = torch.contiguous_format);  view_974 = None
        view_975: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_207, [4096, 384]);  clone_207 = None
        permute_696: "f32[384, 4096]" = torch.ops.aten.permute.default(view_975, [1, 0])
        mm_297: "f32[384, 512]" = torch.ops.aten.mm.default(permute_696, view_266);  permute_696 = None
        permute_121: "f32[512, 384]" = torch.ops.aten.permute.default(primals_95, [1, 0]);  primals_95 = None
        permute_698: "f32[384, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        mm_298: "f32[4096, 512]" = torch.ops.aten.mm.default(view_975, permute_698);  view_975 = permute_698 = None
        view_976: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_298, [32, 128, 512]);  mm_298 = None
        add_249: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_973, view_976);  view_973 = view_976 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_700: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_970, [0, 2, 1, 3]);  view_970 = None
        clone_208: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_700, memory_format = torch.contiguous_format);  permute_700 = None
        view_977: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_208, [32, 128, 384]);  clone_208 = None
        view_978: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_977, [4096, 384]);  view_977 = None
        permute_701: "f32[384, 4096]" = torch.ops.aten.permute.default(view_978, [1, 0])
        mm_299: "f32[384, 512]" = torch.ops.aten.mm.default(permute_701, view_266);  permute_701 = view_266 = None
        permute_119: "f32[512, 384]" = torch.ops.aten.permute.default(primals_94, [1, 0]);  primals_94 = None
        permute_703: "f32[384, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        mm_300: "f32[4096, 512]" = torch.ops.aten.mm.default(view_978, permute_703);  view_978 = permute_703 = None
        view_979: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_300, [32, 128, 512]);  mm_300 = None
        add_250: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_249, view_979);  add_249 = view_979 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_682: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_250, primals_93);  primals_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_170: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_82, rsqrt_20)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_683: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_250, mul_170);  add_250 = mul_170 = None
        sum_85: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_683, [0, 1], True);  mul_683 = None
        view_980: "f32[512]" = torch.ops.aten.reshape.default(sum_85, [512]);  sum_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_684: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_682, add_82)
        mul_685: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_682, rsqrt_20);  mul_682 = None
        sum_86: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_684, [2], True);  mul_684 = None
        add_251: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_247, mul_685);  add_247 = mul_685 = None
        pow_108: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_20, 3);  rsqrt_20 = None
        mul_686: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_86, -0.5);  sum_86 = None
        mul_687: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_686, pow_108);  mul_686 = pow_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_120: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_687, [32, 128, 512]);  mul_687 = None
        div_51: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_120, 512);  expand_120 = None
        pow_109: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_82, 1.0);  add_82 = None
        mul_688: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_109, 2.0);  pow_109 = None
        mul_689: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_51, mul_688);  div_51 = mul_688 = None
        add_252: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_251, mul_689);  add_251 = mul_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_49: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_41, torch.float32);  gt_41 = None
        mul_690: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_49, 1.1111111111111112);  convert_element_type_49 = None
        mul_691: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_252, mul_690);  mul_690 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_981: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_691, [4096, 512]);  mul_691 = None
        permute_705: "f32[512, 4096]" = torch.ops.aten.permute.default(view_981, [1, 0])
        mm_301: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_705, view_264);  permute_705 = view_264 = None
        permute_118: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_92, [1, 0]);  primals_92 = None
        permute_707: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        mm_302: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_981, permute_707);  view_981 = permute_707 = None
        view_982: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_302, [32, 128, 1024]);  mm_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_50: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_40, torch.float32);  gt_40 = None
        mul_692: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_50, 1.1111111111111112);  convert_element_type_50 = None
        mul_693: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_982, mul_692);  view_982 = mul_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_261: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_64, [32, 128, 1024]);  mm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_161: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_261, 0.5)
        pow_29: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_261, 3.0)
        mul_162: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_29, 0.044715);  pow_29 = None
        add_80: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_261, mul_162);  mul_162 = None
        mul_163: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_80, 0.7978845608028654);  add_80 = None
        tanh_8: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_163);  mul_163 = None
        add_81: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_8, 1.0)
        mul_164: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_161, add_81)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_694: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_693, mul_164);  mul_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_263: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_65, [32, 128, 1024]);  mm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_695: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_693, view_263);  mul_693 = view_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_983: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_694, [4096, 1024]);  mul_694 = None
        permute_709: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_983, [1, 0])
        mm_303: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_709, view_260);  permute_709 = None
        permute_117: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_91, [1, 0]);  primals_91 = None
        permute_711: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        mm_304: "f32[4096, 512]" = torch.ops.aten.mm.default(view_983, permute_711);  view_983 = permute_711 = None
        view_984: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_304, [32, 128, 512]);  mm_304 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_696: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_695, mul_161);  mul_161 = None
        mul_697: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_695, add_81);  mul_695 = add_81 = None
        mul_698: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_8, tanh_8);  tanh_8 = None
        sub_36: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_698);  mul_698 = None
        mul_699: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_696, sub_36);  mul_696 = sub_36 = None
        mul_700: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_699, 0.7978845608028654);  mul_699 = None
        mul_701: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_700, 0.044715)
        pow_110: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_261, 2.0);  view_261 = None
        mul_702: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_110, 3.0);  pow_110 = None
        mul_703: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_701, mul_702);  mul_701 = mul_702 = None
        add_253: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_700, mul_703);  mul_700 = mul_703 = None
        mul_704: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_697, 0.5);  mul_697 = None
        add_254: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_253, mul_704);  add_253 = mul_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_985: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_254, [4096, 1024]);  add_254 = None
        permute_713: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_985, [1, 0])
        mm_305: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_713, view_260);  permute_713 = view_260 = None
        permute_116: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_90, [1, 0]);  primals_90 = None
        permute_715: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        mm_306: "f32[4096, 512]" = torch.ops.aten.mm.default(view_985, permute_715);  view_985 = permute_715 = None
        view_986: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_306, [32, 128, 512]);  mm_306 = None
        add_255: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_984, view_986);  view_984 = view_986 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_705: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_255, primals_89);  primals_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_159: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_78, rsqrt_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_706: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_255, mul_159);  add_255 = mul_159 = None
        sum_87: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_706, [0, 1], True);  mul_706 = None
        view_987: "f32[512]" = torch.ops.aten.reshape.default(sum_87, [512]);  sum_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_707: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_705, add_78)
        mul_708: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_705, rsqrt_19);  mul_705 = None
        sum_88: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_707, [2], True);  mul_707 = None
        add_256: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_252, mul_708);  add_252 = mul_708 = None
        pow_111: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_19, 3);  rsqrt_19 = None
        mul_709: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_88, -0.5);  sum_88 = None
        mul_710: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_709, pow_111);  mul_709 = pow_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_121: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_710, [32, 128, 512]);  mul_710 = None
        div_52: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_121, 512);  expand_121 = None
        pow_112: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_78, 1.0);  add_78 = None
        mul_711: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_112, 2.0);  pow_112 = None
        mul_712: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_52, mul_711);  div_52 = mul_711 = None
        add_257: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_256, mul_712);  add_256 = mul_712 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:400 in forward, code: layer_output = hidden_states + self.dropout(attention_output[0])
        convert_element_type_51: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_39, torch.float32);  gt_39 = None
        mul_713: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_51, 1.1111111111111112);  convert_element_type_51 = None
        mul_714: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_257, mul_713);  mul_713 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_988: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_714, [4096, 512]);  mul_714 = None
        permute_717: "f32[512, 4096]" = torch.ops.aten.permute.default(view_988, [1, 0])
        mm_307: "f32[512, 384]" = torch.ops.aten.mm.default(permute_717, view_258);  permute_717 = view_258 = None
        permute_115: "f32[384, 512]" = torch.ops.aten.permute.default(primals_88, [1, 0]);  primals_88 = None
        permute_719: "f32[512, 384]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        mm_308: "f32[4096, 384]" = torch.ops.aten.mm.default(view_988, permute_719);  view_988 = permute_719 = None
        view_989: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_308, [32, 128, 384]);  mm_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_990: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_989, [32, 128, 6, 64]);  view_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_721: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_990, [0, 2, 1, 3]);  view_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_212: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_721, memory_format = torch.contiguous_format);  permute_721 = None
        view_991: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_212, [192, 128, 64]);  clone_212 = None
        bmm_104: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_722, view_991);  permute_722 = None
        bmm_105: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_991, permute_723);  view_991 = permute_723 = None
        view_992: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_104, [32, 6, 128, 64]);  bmm_104 = None
        view_993: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_105, [32, 6, 128, 128]);  bmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_52: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_38, torch.float32);  gt_38 = None
        mul_715: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_52, 1.1111111111111112);  convert_element_type_52 = None
        mul_716: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_993, mul_715);  view_993 = mul_715 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_717: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_716, div_13);  mul_716 = None
        sum_89: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_717, [-1], True)
        neg_16: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_13);  div_13 = None
        fma_14: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_16, sum_89, mul_717);  neg_16 = sum_89 = mul_717 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_994: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_14, [192, 128, 128]);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_106: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_724, view_994);  permute_724 = None
        bmm_107: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_994, permute_725);  view_994 = permute_725 = None
        view_999: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_106, [32, 6, 64, 128]);  bmm_106 = None
        view_1000: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_107, [32, 6, 128, 64]);  bmm_107 = None
        permute_726: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_999, [0, 1, 3, 2]);  view_999 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_727: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_992, [0, 2, 1, 3]);  view_992 = None
        clone_215: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_727, memory_format = torch.contiguous_format);  permute_727 = None
        view_1001: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_215, [32, 128, 384]);  clone_215 = None
        view_1002: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1001, [4096, 384]);  view_1001 = None
        permute_728: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1002, [1, 0])
        view_245: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512])
        mm_309: "f32[384, 512]" = torch.ops.aten.mm.default(permute_728, view_245);  permute_728 = view_245 = None
        permute_111: "f32[512, 384]" = torch.ops.aten.permute.default(primals_87, [1, 0]);  primals_87 = None
        permute_730: "f32[384, 512]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        mm_310: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1002, permute_730);  view_1002 = permute_730 = None
        view_1003: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_310, [32, 128, 512]);  mm_310 = None
        add_258: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_245, view_1003);  add_245 = view_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_732: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_726, [0, 2, 1, 3]);  permute_726 = None
        view_1004: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_732, [32, 128, 384]);  permute_732 = None
        clone_216: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1004, memory_format = torch.contiguous_format);  view_1004 = None
        view_1005: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_216, [4096, 384]);  clone_216 = None
        permute_733: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1005, [1, 0])
        view_242: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_143, [4096, 512]);  mul_143 = None
        mm_311: "f32[384, 512]" = torch.ops.aten.mm.default(permute_733, view_242);  permute_733 = view_242 = None
        permute_109: "f32[512, 384]" = torch.ops.aten.permute.default(primals_86, [1, 0]);  primals_86 = None
        permute_735: "f32[384, 512]" = torch.ops.aten.permute.default(permute_109, [1, 0]);  permute_109 = None
        mm_312: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1005, permute_735);  view_1005 = permute_735 = None
        view_1006: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_312, [32, 128, 512]);  mm_312 = None
        add_259: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_258, view_1006);  add_258 = view_1006 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_737: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1000, [0, 2, 1, 3]);  view_1000 = None
        clone_217: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_737, memory_format = torch.contiguous_format);  permute_737 = None
        view_1007: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_217, [32, 128, 384]);  clone_217 = None
        view_1008: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1007, [4096, 384]);  view_1007 = None
        permute_738: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1008, [1, 0])
        mm_313: "f32[384, 512]" = torch.ops.aten.mm.default(permute_738, view_239);  permute_738 = view_239 = None
        permute_107: "f32[512, 384]" = torch.ops.aten.permute.default(primals_85, [1, 0]);  primals_85 = None
        permute_740: "f32[384, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        mm_314: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1008, permute_740);  view_1008 = permute_740 = None
        view_1009: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_314, [32, 128, 512]);  mm_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_718: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_1009, primals_84);  primals_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_153: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_74, rsqrt_18)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_719: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(view_1009, mul_153);  view_1009 = mul_153 = None
        sum_90: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_719, [0, 1], True);  mul_719 = None
        view_1010: "f32[512]" = torch.ops.aten.reshape.default(sum_90, [512]);  sum_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_720: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_718, add_74)
        mul_721: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_718, rsqrt_18);  mul_718 = None
        sum_91: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_720, [2], True);  mul_720 = None
        add_260: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_257, mul_721);  add_257 = mul_721 = None
        pow_113: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_18, 3);  rsqrt_18 = None
        mul_722: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_91, -0.5);  sum_91 = None
        mul_723: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_722, pow_113);  mul_722 = pow_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_122: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_723, [32, 128, 512]);  mul_723 = None
        div_53: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_122, 512);  expand_122 = None
        pow_114: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_74, 1.0);  add_74 = None
        mul_724: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_114, 2.0);  pow_114 = None
        mul_725: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_53, mul_724);  div_53 = mul_724 = None
        add_261: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_260, mul_725);  add_260 = mul_725 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_53: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_37, torch.float32);  gt_37 = None
        mul_726: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_53, 1.1111111111111112);  convert_element_type_53 = None
        mul_727: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_261, mul_726);  mul_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1011: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_727, [4096, 512]);  mul_727 = None
        permute_742: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1011, [1, 0])
        mm_315: "f32[512, 384]" = torch.ops.aten.mm.default(permute_742, view_237);  permute_742 = view_237 = None
        permute_106: "f32[384, 512]" = torch.ops.aten.permute.default(primals_83, [1, 0]);  primals_83 = None
        permute_744: "f32[512, 384]" = torch.ops.aten.permute.default(permute_106, [1, 0]);  permute_106 = None
        mm_316: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1011, permute_744);  view_1011 = permute_744 = None
        view_1012: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_316, [32, 128, 384]);  mm_316 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1013: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1012, [32, 128, 6, 64]);  view_1012 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_746: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1013, [0, 2, 1, 3]);  view_1013 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_219: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_746, memory_format = torch.contiguous_format);  permute_746 = None
        view_1014: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_219, [192, 128, 64]);  clone_219 = None
        bmm_108: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_747, view_1014);  permute_747 = None
        bmm_109: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1014, permute_748);  view_1014 = permute_748 = None
        view_1015: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_108, [32, 6, 128, 64]);  bmm_108 = None
        view_1016: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_109, [32, 6, 128, 128]);  bmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_54: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_36, torch.float32);  gt_36 = None
        mul_728: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_54, 1.1111111111111112);  convert_element_type_54 = None
        mul_729: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1016, mul_728);  view_1016 = mul_728 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_730: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_729, div_12);  mul_729 = None
        sum_92: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_730, [-1], True)
        neg_17: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_12);  div_12 = None
        fma_15: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_17, sum_92, mul_730);  neg_17 = sum_92 = mul_730 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1017: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_15, [192, 128, 128]);  fma_15 = None
        view_1019: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1017, [32, 6, 128, 128]);  view_1017 = None
        view_1020: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1019, [192, 128, 128])
        add_262: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_248, view_1019);  add_248 = view_1019 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        sum_93: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_262, [0], True);  add_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_1: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_93, 0);  sum_93 = None
        permute_749: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_1, [1, 2, 0]);  squeeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq: "b8[128, 128]" = torch.ops.aten.eq.Scalar(add_71, -1)
        unsqueeze_20: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq, -1);  eq = None
        where_9: "f32[128, 128, 6]" = torch.ops.aten.where.self(unsqueeze_20, full_default, permute_749);  unsqueeze_20 = permute_749 = None
        clone_222: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_9, memory_format = torch.contiguous_format);  where_9 = None
        full_default_16: "f32[32, 6]" = torch.ops.aten.full.default([32, 6], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put: "f32[32, 6]" = torch.ops.aten.index_put.default(full_default_16, [add_71], clone_222, True);  add_71 = clone_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_110: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_750, view_1020);  permute_750 = None
        bmm_111: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1020, permute_751);  view_1020 = permute_751 = None
        view_1022: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_110, [32, 6, 64, 128]);  bmm_110 = None
        view_1023: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_111, [32, 6, 128, 64]);  bmm_111 = None
        permute_752: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1022, [0, 1, 3, 2]);  view_1022 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_753: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1015, [0, 2, 1, 3]);  view_1015 = None
        clone_223: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_753, memory_format = torch.contiguous_format);  permute_753 = None
        view_1024: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_223, [32, 128, 384]);  clone_223 = None
        view_1025: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1024, [4096, 384]);  view_1024 = None
        permute_754: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1025, [1, 0])
        mm_317: "f32[384, 512]" = torch.ops.aten.mm.default(permute_754, view_218);  permute_754 = None
        permute_101: "f32[512, 384]" = torch.ops.aten.permute.default(primals_81, [1, 0]);  primals_81 = None
        permute_756: "f32[384, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        mm_318: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1025, permute_756);  view_1025 = permute_756 = None
        view_1026: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_318, [32, 128, 512]);  mm_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_758: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_752, [0, 2, 1, 3]);  permute_752 = None
        view_1027: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_758, [32, 128, 384]);  permute_758 = None
        clone_224: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1027, memory_format = torch.contiguous_format);  view_1027 = None
        view_1028: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_224, [4096, 384]);  clone_224 = None
        permute_759: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1028, [1, 0])
        mm_319: "f32[384, 512]" = torch.ops.aten.mm.default(permute_759, view_218);  permute_759 = None
        permute_99: "f32[512, 384]" = torch.ops.aten.permute.default(primals_80, [1, 0]);  primals_80 = None
        permute_761: "f32[384, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        mm_320: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1028, permute_761);  view_1028 = permute_761 = None
        view_1029: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_320, [32, 128, 512]);  mm_320 = None
        add_263: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1026, view_1029);  view_1026 = view_1029 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_763: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1023, [0, 2, 1, 3]);  view_1023 = None
        clone_225: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_763, memory_format = torch.contiguous_format);  permute_763 = None
        view_1030: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_225, [32, 128, 384]);  clone_225 = None
        view_1031: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1030, [4096, 384]);  view_1030 = None
        permute_764: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1031, [1, 0])
        mm_321: "f32[384, 512]" = torch.ops.aten.mm.default(permute_764, view_218);  permute_764 = view_218 = None
        permute_97: "f32[512, 384]" = torch.ops.aten.permute.default(primals_79, [1, 0]);  primals_79 = None
        permute_766: "f32[384, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        mm_322: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1031, permute_766);  view_1031 = permute_766 = None
        view_1032: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_322, [32, 128, 512]);  mm_322 = None
        add_264: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_263, view_1032);  add_263 = view_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_731: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_264, primals_78);  primals_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul_144: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt_35, embedding)
        mul_145: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_144, 1.1111111111111112);  mul_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_146: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_145, rsqrt_17)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_732: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_264, mul_146);  add_264 = mul_146 = None
        sum_94: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_732, [0, 1], True);  mul_732 = None
        view_1033: "f32[512]" = torch.ops.aten.reshape.default(sum_94, [512]);  sum_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_733: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_731, mul_145)
        mul_734: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_731, rsqrt_17);  mul_731 = None
        sum_95: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_733, [2], True);  mul_733 = None
        add_265: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_261, mul_734);  add_261 = mul_734 = None
        pow_115: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_17, 3);  rsqrt_17 = None
        mul_735: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_95, -0.5);  sum_95 = None
        mul_736: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_735, pow_115);  mul_735 = pow_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_123: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_736, [32, 128, 512]);  mul_736 = None
        div_54: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_123, 512);  expand_123 = None
        pow_116: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_145, 1.0);  mul_145 = None
        mul_737: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_116, 2.0);  pow_116 = None
        mul_738: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_54, mul_737);  div_54 = mul_737 = None
        add_266: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_265, mul_738);  add_265 = mul_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_55: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_35, torch.float32);  gt_35 = None
        mul_739: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_55, 1.1111111111111112);  convert_element_type_55 = None
        mul_740: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_266, mul_739);  add_266 = mul_739 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        eq_1: "b8[32, 128]" = torch.ops.aten.eq.Scalar(primals_1, -1)
        unsqueeze_21: "b8[32, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_1, -1);  eq_1 = None
        where_10: "f32[32, 128, 512]" = torch.ops.aten.where.self(unsqueeze_21, full_default, mul_740);  mul_740 = None
        full_default_18: "f32[250112, 512]" = torch.ops.aten.full.default([250112, 512], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_1: "f32[250112, 512]" = torch.ops.aten.index_put.default(full_default_18, [primals_1], where_10, True);  where_10 = None
        add_267: "f32[250112, 512]" = torch.ops.aten.add.Tensor(mm_145, index_put_1);  mm_145 = index_put_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:766 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_56: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_34, torch.float32);  gt_34 = None
        mul_741: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_56, 1.1111111111111112);  convert_element_type_56 = None
        mul_742: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_259, mul_741);  add_259 = mul_741 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_743: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_742, primals_76);  primals_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_140: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_62, rsqrt_16)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_744: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_742, mul_140);  mul_742 = mul_140 = None
        sum_96: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_744, [0, 1], True);  mul_744 = None
        view_1034: "f32[512]" = torch.ops.aten.reshape.default(sum_96, [512]);  sum_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_745: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_743, add_62)
        mul_746: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_743, rsqrt_16);  mul_743 = None
        sum_97: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_745, [2], True);  mul_745 = None
        pow_117: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_16, 3);  rsqrt_16 = None
        mul_747: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_97, -0.5);  sum_97 = None
        mul_748: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_747, pow_117);  mul_747 = pow_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_124: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_748, [32, 128, 512]);  mul_748 = None
        div_55: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_124, 512);  expand_124 = None
        pow_118: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_62, 1.0);  add_62 = None
        mul_749: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_118, 2.0);  pow_118 = None
        mul_750: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_55, mul_749);  div_55 = mul_749 = None
        add_268: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(mul_746, mul_750);  mul_746 = mul_750 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_57: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_33, torch.float32);  gt_33 = None
        mul_751: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_57, 1.1111111111111112);  convert_element_type_57 = None
        mul_752: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_268, mul_751);  mul_751 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1035: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_752, [4096, 512]);  mul_752 = None
        permute_768: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1035, [1, 0])
        mm_323: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_768, view_215);  permute_768 = view_215 = None
        permute_96: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_75, [1, 0]);  primals_75 = None
        permute_770: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        mm_324: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1035, permute_770);  view_1035 = permute_770 = None
        view_1036: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_324, [32, 128, 1024]);  mm_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_58: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_32, torch.float32);  gt_32 = None
        mul_753: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_58, 1.1111111111111112);  convert_element_type_58 = None
        mul_754: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1036, mul_753);  view_1036 = mul_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_212: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_53, [32, 128, 1024]);  mm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_131: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_212, 0.5)
        pow_24: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_212, 3.0)
        mul_132: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_24, 0.044715);  pow_24 = None
        add_60: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_212, mul_132);  mul_132 = None
        mul_133: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_60, 0.7978845608028654);  add_60 = None
        tanh_7: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_133);  mul_133 = None
        add_61: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_7, 1.0)
        mul_134: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_131, add_61)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_755: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_754, mul_134);  mul_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_214: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_54, [32, 128, 1024]);  mm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_756: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_754, view_214);  mul_754 = view_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1037: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_755, [4096, 1024]);  mul_755 = None
        permute_772: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        mm_325: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_772, view_211);  permute_772 = None
        permute_95: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_74, [1, 0]);  primals_74 = None
        permute_774: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        mm_326: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1037, permute_774);  view_1037 = permute_774 = None
        view_1038: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_326, [32, 128, 512]);  mm_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_757: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_756, mul_131);  mul_131 = None
        mul_758: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_756, add_61);  mul_756 = add_61 = None
        mul_759: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_7, tanh_7);  tanh_7 = None
        sub_37: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_759);  mul_759 = None
        mul_760: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_757, sub_37);  mul_757 = sub_37 = None
        mul_761: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_760, 0.7978845608028654);  mul_760 = None
        mul_762: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_761, 0.044715)
        pow_119: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_212, 2.0);  view_212 = None
        mul_763: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_119, 3.0);  pow_119 = None
        mul_764: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_762, mul_763);  mul_762 = mul_763 = None
        add_269: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_761, mul_764);  mul_761 = mul_764 = None
        mul_765: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_758, 0.5);  mul_758 = None
        add_270: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_269, mul_765);  add_269 = mul_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1039: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_270, [4096, 1024]);  add_270 = None
        permute_776: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1039, [1, 0])
        mm_327: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_776, view_211);  permute_776 = view_211 = None
        permute_94: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_73, [1, 0]);  primals_73 = None
        permute_778: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        mm_328: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1039, permute_778);  view_1039 = permute_778 = None
        view_1040: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_328, [32, 128, 512]);  mm_328 = None
        add_271: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1038, view_1040);  view_1038 = view_1040 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_766: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_271, primals_72);  primals_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_129: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_58, rsqrt_15)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_767: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_271, mul_129);  add_271 = mul_129 = None
        sum_98: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_767, [0, 1], True);  mul_767 = None
        view_1041: "f32[512]" = torch.ops.aten.reshape.default(sum_98, [512]);  sum_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_768: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_766, add_58)
        mul_769: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_766, rsqrt_15);  mul_766 = None
        sum_99: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_768, [2], True);  mul_768 = None
        add_272: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_268, mul_769);  add_268 = mul_769 = None
        pow_120: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_15, 3);  rsqrt_15 = None
        mul_770: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_99, -0.5);  sum_99 = None
        mul_771: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_770, pow_120);  mul_770 = pow_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_125: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_771, [32, 128, 512]);  mul_771 = None
        div_56: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_125, 512);  expand_125 = None
        pow_121: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_58, 1.0);  add_58 = None
        mul_772: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_121, 2.0);  pow_121 = None
        mul_773: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_56, mul_772);  div_56 = mul_772 = None
        add_273: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_272, mul_773);  add_272 = mul_773 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_59: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_31, torch.float32);  gt_31 = None
        mul_774: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_59, 1.1111111111111112);  convert_element_type_59 = None
        mul_775: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_273, mul_774);  mul_774 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1042: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_775, [4096, 512]);  mul_775 = None
        permute_780: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1042, [1, 0])
        mm_329: "f32[512, 384]" = torch.ops.aten.mm.default(permute_780, view_209);  permute_780 = view_209 = None
        permute_93: "f32[384, 512]" = torch.ops.aten.permute.default(primals_71, [1, 0]);  primals_71 = None
        permute_782: "f32[512, 384]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        mm_330: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1042, permute_782);  view_1042 = permute_782 = None
        view_1043: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_330, [32, 128, 384]);  mm_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1044: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1043, [32, 128, 6, 64]);  view_1043 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_784: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1044, [0, 2, 1, 3]);  view_1044 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_231: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_784, memory_format = torch.contiguous_format);  permute_784 = None
        view_1045: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_231, [192, 128, 64]);  clone_231 = None
        bmm_112: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_785, view_1045);  permute_785 = None
        bmm_113: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1045, permute_786);  view_1045 = permute_786 = None
        view_1046: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_112, [32, 6, 128, 64]);  bmm_112 = None
        view_1047: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_113, [32, 6, 128, 128]);  bmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_60: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_30, torch.float32);  gt_30 = None
        mul_776: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_60, 1.1111111111111112);  convert_element_type_60 = None
        mul_777: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1047, mul_776);  view_1047 = mul_776 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_778: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_777, div_9);  mul_777 = None
        sum_100: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_778, [-1], True)
        neg_18: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_9);  div_9 = None
        fma_16: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_18, sum_100, mul_778);  neg_18 = sum_100 = mul_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1048: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_16, [192, 128, 128]);  fma_16 = None
        view_1050: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1048, [32, 6, 128, 128]);  view_1048 = None
        view_1051: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1050, [192, 128, 128])

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_114: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_787, view_1051);  permute_787 = None
        bmm_115: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1051, permute_788);  view_1051 = permute_788 = None
        view_1053: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_114, [32, 6, 64, 128]);  bmm_114 = None
        view_1054: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_115, [32, 6, 128, 64]);  bmm_115 = None
        permute_789: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1053, [0, 1, 3, 2]);  view_1053 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_790: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1046, [0, 2, 1, 3]);  view_1046 = None
        clone_234: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_790, memory_format = torch.contiguous_format);  permute_790 = None
        view_1055: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_234, [32, 128, 384]);  clone_234 = None
        view_1056: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1055, [4096, 384]);  view_1055 = None
        permute_791: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1056, [1, 0])
        mm_331: "f32[384, 512]" = torch.ops.aten.mm.default(permute_791, view_190);  permute_791 = None
        permute_89: "f32[512, 384]" = torch.ops.aten.permute.default(primals_70, [1, 0]);  primals_70 = None
        permute_793: "f32[384, 512]" = torch.ops.aten.permute.default(permute_89, [1, 0]);  permute_89 = None
        mm_332: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1056, permute_793);  view_1056 = permute_793 = None
        view_1057: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_332, [32, 128, 512]);  mm_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_795: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_789, [0, 2, 1, 3]);  permute_789 = None
        view_1058: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_795, [32, 128, 384]);  permute_795 = None
        clone_235: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1058, memory_format = torch.contiguous_format);  view_1058 = None
        view_1059: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_235, [4096, 384]);  clone_235 = None
        permute_796: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1059, [1, 0])
        mm_333: "f32[384, 512]" = torch.ops.aten.mm.default(permute_796, view_190);  permute_796 = None
        permute_87: "f32[512, 384]" = torch.ops.aten.permute.default(primals_69, [1, 0]);  primals_69 = None
        permute_798: "f32[384, 512]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        mm_334: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1059, permute_798);  view_1059 = permute_798 = None
        view_1060: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_334, [32, 128, 512]);  mm_334 = None
        add_274: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1057, view_1060);  view_1057 = view_1060 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_800: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1054, [0, 2, 1, 3]);  view_1054 = None
        clone_236: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_800, memory_format = torch.contiguous_format);  permute_800 = None
        view_1061: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_236, [32, 128, 384]);  clone_236 = None
        view_1062: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1061, [4096, 384]);  view_1061 = None
        permute_801: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1062, [1, 0])
        mm_335: "f32[384, 512]" = torch.ops.aten.mm.default(permute_801, view_190);  permute_801 = view_190 = None
        permute_85: "f32[512, 384]" = torch.ops.aten.permute.default(primals_68, [1, 0]);  primals_68 = None
        permute_803: "f32[384, 512]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        mm_336: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1062, permute_803);  view_1062 = permute_803 = None
        view_1063: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_336, [32, 128, 512]);  mm_336 = None
        add_275: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_274, view_1063);  add_274 = view_1063 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_779: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_275, primals_67);  primals_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_123: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_55, rsqrt_14)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_780: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_275, mul_123);  add_275 = mul_123 = None
        sum_101: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_780, [0, 1], True);  mul_780 = None
        view_1064: "f32[512]" = torch.ops.aten.reshape.default(sum_101, [512]);  sum_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_781: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_779, add_55)
        mul_782: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_779, rsqrt_14);  mul_779 = None
        sum_102: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_781, [2], True);  mul_781 = None
        add_276: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_273, mul_782);  add_273 = mul_782 = None
        pow_122: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_14, 3);  rsqrt_14 = None
        mul_783: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_102, -0.5);  sum_102 = None
        mul_784: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_783, pow_122);  mul_783 = pow_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_126: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_784, [32, 128, 512]);  mul_784 = None
        div_57: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_126, 512);  expand_126 = None
        pow_123: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_55, 1.0);  add_55 = None
        mul_785: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_123, 2.0);  pow_123 = None
        mul_786: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_57, mul_785);  div_57 = mul_785 = None
        add_277: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_276, mul_786);  add_276 = mul_786 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_61: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_29, torch.float32);  gt_29 = None
        mul_787: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_61, 1.1111111111111112);  convert_element_type_61 = None
        mul_788: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_277, mul_787);  mul_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1065: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_788, [4096, 512]);  mul_788 = None
        permute_805: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1065, [1, 0])
        mm_337: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_805, view_188);  permute_805 = view_188 = None
        permute_84: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_66, [1, 0]);  primals_66 = None
        permute_807: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_84, [1, 0]);  permute_84 = None
        mm_338: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1065, permute_807);  view_1065 = permute_807 = None
        view_1066: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_338, [32, 128, 1024]);  mm_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_62: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_28, torch.float32);  gt_28 = None
        mul_789: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_62, 1.1111111111111112);  convert_element_type_62 = None
        mul_790: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1066, mul_789);  view_1066 = mul_789 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_185: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_46, [32, 128, 1024]);  mm_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_114: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_185, 0.5)
        pow_21: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_185, 3.0)
        mul_115: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_21, 0.044715);  pow_21 = None
        add_53: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_185, mul_115);  mul_115 = None
        mul_116: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_53, 0.7978845608028654);  add_53 = None
        tanh_6: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_116);  mul_116 = None
        add_54: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_6, 1.0)
        mul_117: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_114, add_54)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_791: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_790, mul_117);  mul_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_187: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_47, [32, 128, 1024]);  mm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_792: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_790, view_187);  mul_790 = view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1067: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_791, [4096, 1024]);  mul_791 = None
        permute_809: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1067, [1, 0])
        mm_339: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_809, view_184);  permute_809 = None
        permute_83: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_65, [1, 0]);  primals_65 = None
        permute_811: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        mm_340: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1067, permute_811);  view_1067 = permute_811 = None
        view_1068: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_340, [32, 128, 512]);  mm_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_793: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_792, mul_114);  mul_114 = None
        mul_794: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_792, add_54);  mul_792 = add_54 = None
        mul_795: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_6, tanh_6);  tanh_6 = None
        sub_38: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_795);  mul_795 = None
        mul_796: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_793, sub_38);  mul_793 = sub_38 = None
        mul_797: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_796, 0.7978845608028654);  mul_796 = None
        mul_798: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_797, 0.044715)
        pow_124: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_185, 2.0);  view_185 = None
        mul_799: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_124, 3.0);  pow_124 = None
        mul_800: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_798, mul_799);  mul_798 = mul_799 = None
        add_278: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_797, mul_800);  mul_797 = mul_800 = None
        mul_801: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_794, 0.5);  mul_794 = None
        add_279: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_278, mul_801);  add_278 = mul_801 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1069: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_279, [4096, 1024]);  add_279 = None
        permute_813: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1069, [1, 0])
        mm_341: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_813, view_184);  permute_813 = view_184 = None
        permute_82: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_64, [1, 0]);  primals_64 = None
        permute_815: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        mm_342: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1069, permute_815);  view_1069 = permute_815 = None
        view_1070: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_342, [32, 128, 512]);  mm_342 = None
        add_280: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1068, view_1070);  view_1068 = view_1070 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_802: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_280, primals_63);  primals_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_112: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_51, rsqrt_13)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_803: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_280, mul_112);  add_280 = mul_112 = None
        sum_103: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_803, [0, 1], True);  mul_803 = None
        view_1071: "f32[512]" = torch.ops.aten.reshape.default(sum_103, [512]);  sum_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_804: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_802, add_51)
        mul_805: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_802, rsqrt_13);  mul_802 = None
        sum_104: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_804, [2], True);  mul_804 = None
        add_281: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_277, mul_805);  add_277 = mul_805 = None
        pow_125: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_13, 3);  rsqrt_13 = None
        mul_806: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_104, -0.5);  sum_104 = None
        mul_807: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_806, pow_125);  mul_806 = pow_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_127: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_807, [32, 128, 512]);  mul_807 = None
        div_58: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_127, 512);  expand_127 = None
        pow_126: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_51, 1.0);  add_51 = None
        mul_808: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_126, 2.0);  pow_126 = None
        mul_809: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_58, mul_808);  div_58 = mul_808 = None
        add_282: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_281, mul_809);  add_281 = mul_809 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_63: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_27, torch.float32);  gt_27 = None
        mul_810: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_63, 1.1111111111111112);  convert_element_type_63 = None
        mul_811: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_282, mul_810);  mul_810 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1072: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_811, [4096, 512]);  mul_811 = None
        permute_817: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1072, [1, 0])
        mm_343: "f32[512, 384]" = torch.ops.aten.mm.default(permute_817, view_182);  permute_817 = view_182 = None
        permute_81: "f32[384, 512]" = torch.ops.aten.permute.default(primals_62, [1, 0]);  primals_62 = None
        permute_819: "f32[512, 384]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        mm_344: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1072, permute_819);  view_1072 = permute_819 = None
        view_1073: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_344, [32, 128, 384]);  mm_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1074: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1073, [32, 128, 6, 64]);  view_1073 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_821: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1074, [0, 2, 1, 3]);  view_1074 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_240: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_821, memory_format = torch.contiguous_format);  permute_821 = None
        view_1075: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_240, [192, 128, 64]);  clone_240 = None
        bmm_116: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_822, view_1075);  permute_822 = None
        bmm_117: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1075, permute_823);  view_1075 = permute_823 = None
        view_1076: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_116, [32, 6, 128, 64]);  bmm_116 = None
        view_1077: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_117, [32, 6, 128, 128]);  bmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_64: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_26, torch.float32);  gt_26 = None
        mul_812: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_64, 1.1111111111111112);  convert_element_type_64 = None
        mul_813: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1077, mul_812);  view_1077 = mul_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_814: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_813, div_8);  mul_813 = None
        sum_105: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_814, [-1], True)
        neg_19: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_8);  div_8 = None
        fma_17: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_19, sum_105, mul_814);  neg_19 = sum_105 = mul_814 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1078: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_17, [192, 128, 128]);  fma_17 = None
        view_1080: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1078, [32, 6, 128, 128]);  view_1078 = None
        view_1081: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1080, [192, 128, 128])
        add_283: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_1050, view_1080);  view_1050 = view_1080 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_118: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_824, view_1081);  permute_824 = None
        bmm_119: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1081, permute_825);  view_1081 = permute_825 = None
        view_1083: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_118, [32, 6, 64, 128]);  bmm_118 = None
        view_1084: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_119, [32, 6, 128, 64]);  bmm_119 = None
        permute_826: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1083, [0, 1, 3, 2]);  view_1083 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_827: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1076, [0, 2, 1, 3]);  view_1076 = None
        clone_243: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_827, memory_format = torch.contiguous_format);  permute_827 = None
        view_1085: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_243, [32, 128, 384]);  clone_243 = None
        view_1086: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1085, [4096, 384]);  view_1085 = None
        permute_828: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1086, [1, 0])
        mm_345: "f32[384, 512]" = torch.ops.aten.mm.default(permute_828, view_163);  permute_828 = None
        permute_77: "f32[512, 384]" = torch.ops.aten.permute.default(primals_61, [1, 0]);  primals_61 = None
        permute_830: "f32[384, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        mm_346: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1086, permute_830);  view_1086 = permute_830 = None
        view_1087: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_346, [32, 128, 512]);  mm_346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_832: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_826, [0, 2, 1, 3]);  permute_826 = None
        view_1088: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_832, [32, 128, 384]);  permute_832 = None
        clone_244: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1088, memory_format = torch.contiguous_format);  view_1088 = None
        view_1089: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_244, [4096, 384]);  clone_244 = None
        permute_833: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1089, [1, 0])
        mm_347: "f32[384, 512]" = torch.ops.aten.mm.default(permute_833, view_163);  permute_833 = None
        permute_75: "f32[512, 384]" = torch.ops.aten.permute.default(primals_60, [1, 0]);  primals_60 = None
        permute_835: "f32[384, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        mm_348: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1089, permute_835);  view_1089 = permute_835 = None
        view_1090: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_348, [32, 128, 512]);  mm_348 = None
        add_284: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1087, view_1090);  view_1087 = view_1090 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_837: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1084, [0, 2, 1, 3]);  view_1084 = None
        clone_245: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_837, memory_format = torch.contiguous_format);  permute_837 = None
        view_1091: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_245, [32, 128, 384]);  clone_245 = None
        view_1092: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1091, [4096, 384]);  view_1091 = None
        permute_838: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1092, [1, 0])
        mm_349: "f32[384, 512]" = torch.ops.aten.mm.default(permute_838, view_163);  permute_838 = view_163 = None
        permute_73: "f32[512, 384]" = torch.ops.aten.permute.default(primals_59, [1, 0]);  primals_59 = None
        permute_840: "f32[384, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        mm_350: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1092, permute_840);  view_1092 = permute_840 = None
        view_1093: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_350, [32, 128, 512]);  mm_350 = None
        add_285: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_284, view_1093);  add_284 = view_1093 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_815: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_285, primals_58);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_106: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_48, rsqrt_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_816: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_285, mul_106);  add_285 = mul_106 = None
        sum_106: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_816, [0, 1], True);  mul_816 = None
        view_1094: "f32[512]" = torch.ops.aten.reshape.default(sum_106, [512]);  sum_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_817: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_815, add_48)
        mul_818: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_815, rsqrt_12);  mul_815 = None
        sum_107: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_817, [2], True);  mul_817 = None
        add_286: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_282, mul_818);  add_282 = mul_818 = None
        pow_127: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_12, 3);  rsqrt_12 = None
        mul_819: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_107, -0.5);  sum_107 = None
        mul_820: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_819, pow_127);  mul_819 = pow_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_128: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_820, [32, 128, 512]);  mul_820 = None
        div_59: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_128, 512);  expand_128 = None
        pow_128: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_48, 1.0);  add_48 = None
        mul_821: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_128, 2.0);  pow_128 = None
        mul_822: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_59, mul_821);  div_59 = mul_821 = None
        add_287: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_286, mul_822);  add_286 = mul_822 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_65: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_25, torch.float32);  gt_25 = None
        mul_823: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_65, 1.1111111111111112);  convert_element_type_65 = None
        mul_824: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_287, mul_823);  mul_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1095: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_824, [4096, 512]);  mul_824 = None
        permute_842: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1095, [1, 0])
        mm_351: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_842, view_161);  permute_842 = view_161 = None
        permute_72: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_57, [1, 0]);  primals_57 = None
        permute_844: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        mm_352: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1095, permute_844);  view_1095 = permute_844 = None
        view_1096: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_352, [32, 128, 1024]);  mm_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_66: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_24, torch.float32);  gt_24 = None
        mul_825: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_66, 1.1111111111111112);  convert_element_type_66 = None
        mul_826: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1096, mul_825);  view_1096 = mul_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_158: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_39, [32, 128, 1024]);  mm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_97: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_158, 0.5)
        pow_18: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_158, 3.0)
        mul_98: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_18, 0.044715);  pow_18 = None
        add_46: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_158, mul_98);  mul_98 = None
        mul_99: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_46, 0.7978845608028654);  add_46 = None
        tanh_5: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_99);  mul_99 = None
        add_47: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_5, 1.0)
        mul_100: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_97, add_47)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_827: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_826, mul_100);  mul_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_160: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_40, [32, 128, 1024]);  mm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_828: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_826, view_160);  mul_826 = view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1097: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_827, [4096, 1024]);  mul_827 = None
        permute_846: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1097, [1, 0])
        mm_353: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_846, view_157);  permute_846 = None
        permute_71: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_56, [1, 0]);  primals_56 = None
        permute_848: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        mm_354: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1097, permute_848);  view_1097 = permute_848 = None
        view_1098: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_354, [32, 128, 512]);  mm_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_829: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_828, mul_97);  mul_97 = None
        mul_830: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_828, add_47);  mul_828 = add_47 = None
        mul_831: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_5, tanh_5);  tanh_5 = None
        sub_39: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_831);  mul_831 = None
        mul_832: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_829, sub_39);  mul_829 = sub_39 = None
        mul_833: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_832, 0.7978845608028654);  mul_832 = None
        mul_834: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_833, 0.044715)
        pow_129: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_158, 2.0);  view_158 = None
        mul_835: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_129, 3.0);  pow_129 = None
        mul_836: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_834, mul_835);  mul_834 = mul_835 = None
        add_288: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_833, mul_836);  mul_833 = mul_836 = None
        mul_837: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_830, 0.5);  mul_830 = None
        add_289: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_288, mul_837);  add_288 = mul_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1099: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_289, [4096, 1024]);  add_289 = None
        permute_850: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1099, [1, 0])
        mm_355: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_850, view_157);  permute_850 = view_157 = None
        permute_70: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_55, [1, 0]);  primals_55 = None
        permute_852: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_70, [1, 0]);  permute_70 = None
        mm_356: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1099, permute_852);  view_1099 = permute_852 = None
        view_1100: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_356, [32, 128, 512]);  mm_356 = None
        add_290: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1098, view_1100);  view_1098 = view_1100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_838: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_290, primals_54);  primals_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_95: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_44, rsqrt_11)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_839: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_290, mul_95);  add_290 = mul_95 = None
        sum_108: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_839, [0, 1], True);  mul_839 = None
        view_1101: "f32[512]" = torch.ops.aten.reshape.default(sum_108, [512]);  sum_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_840: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_838, add_44)
        mul_841: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_838, rsqrt_11);  mul_838 = None
        sum_109: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_840, [2], True);  mul_840 = None
        add_291: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_287, mul_841);  add_287 = mul_841 = None
        pow_130: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_11, 3);  rsqrt_11 = None
        mul_842: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_109, -0.5);  sum_109 = None
        mul_843: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_842, pow_130);  mul_842 = pow_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_129: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_843, [32, 128, 512]);  mul_843 = None
        div_60: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_129, 512);  expand_129 = None
        pow_131: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_44, 1.0);  add_44 = None
        mul_844: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_131, 2.0);  pow_131 = None
        mul_845: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_60, mul_844);  div_60 = mul_844 = None
        add_292: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_291, mul_845);  add_291 = mul_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_67: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_23, torch.float32);  gt_23 = None
        mul_846: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_67, 1.1111111111111112);  convert_element_type_67 = None
        mul_847: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_292, mul_846);  mul_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1102: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_847, [4096, 512]);  mul_847 = None
        permute_854: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1102, [1, 0])
        mm_357: "f32[512, 384]" = torch.ops.aten.mm.default(permute_854, view_155);  permute_854 = view_155 = None
        permute_69: "f32[384, 512]" = torch.ops.aten.permute.default(primals_53, [1, 0]);  primals_53 = None
        permute_856: "f32[512, 384]" = torch.ops.aten.permute.default(permute_69, [1, 0]);  permute_69 = None
        mm_358: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1102, permute_856);  view_1102 = permute_856 = None
        view_1103: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_358, [32, 128, 384]);  mm_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1104: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1103, [32, 128, 6, 64]);  view_1103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_858: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1104, [0, 2, 1, 3]);  view_1104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_249: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_858, memory_format = torch.contiguous_format);  permute_858 = None
        view_1105: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_249, [192, 128, 64]);  clone_249 = None
        bmm_120: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_859, view_1105);  permute_859 = None
        bmm_121: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1105, permute_860);  view_1105 = permute_860 = None
        view_1106: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_120, [32, 6, 128, 64]);  bmm_120 = None
        view_1107: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_121, [32, 6, 128, 128]);  bmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_68: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_22, torch.float32);  gt_22 = None
        mul_848: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_68, 1.1111111111111112);  convert_element_type_68 = None
        mul_849: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1107, mul_848);  view_1107 = mul_848 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_850: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_849, div_7);  mul_849 = None
        sum_110: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_850, [-1], True)
        neg_20: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_7);  div_7 = None
        fma_18: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_20, sum_110, mul_850);  neg_20 = sum_110 = mul_850 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1108: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_18, [192, 128, 128]);  fma_18 = None
        view_1110: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1108, [32, 6, 128, 128]);  view_1108 = None
        view_1111: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1110, [192, 128, 128])
        add_293: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_283, view_1110);  add_283 = view_1110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_122: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_861, view_1111);  permute_861 = None
        bmm_123: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1111, permute_862);  view_1111 = permute_862 = None
        view_1113: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_122, [32, 6, 64, 128]);  bmm_122 = None
        view_1114: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_123, [32, 6, 128, 64]);  bmm_123 = None
        permute_863: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1113, [0, 1, 3, 2]);  view_1113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_864: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1106, [0, 2, 1, 3]);  view_1106 = None
        clone_252: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_864, memory_format = torch.contiguous_format);  permute_864 = None
        view_1115: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_252, [32, 128, 384]);  clone_252 = None
        view_1116: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1115, [4096, 384]);  view_1115 = None
        permute_865: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1116, [1, 0])
        mm_359: "f32[384, 512]" = torch.ops.aten.mm.default(permute_865, view_136);  permute_865 = None
        permute_65: "f32[512, 384]" = torch.ops.aten.permute.default(primals_52, [1, 0]);  primals_52 = None
        permute_867: "f32[384, 512]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        mm_360: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1116, permute_867);  view_1116 = permute_867 = None
        view_1117: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_360, [32, 128, 512]);  mm_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_869: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_863, [0, 2, 1, 3]);  permute_863 = None
        view_1118: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_869, [32, 128, 384]);  permute_869 = None
        clone_253: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1118, memory_format = torch.contiguous_format);  view_1118 = None
        view_1119: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_253, [4096, 384]);  clone_253 = None
        permute_870: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1119, [1, 0])
        mm_361: "f32[384, 512]" = torch.ops.aten.mm.default(permute_870, view_136);  permute_870 = None
        permute_63: "f32[512, 384]" = torch.ops.aten.permute.default(primals_51, [1, 0]);  primals_51 = None
        permute_872: "f32[384, 512]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        mm_362: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1119, permute_872);  view_1119 = permute_872 = None
        view_1120: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_362, [32, 128, 512]);  mm_362 = None
        add_294: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1117, view_1120);  view_1117 = view_1120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_874: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1114, [0, 2, 1, 3]);  view_1114 = None
        clone_254: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_874, memory_format = torch.contiguous_format);  permute_874 = None
        view_1121: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_254, [32, 128, 384]);  clone_254 = None
        view_1122: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1121, [4096, 384]);  view_1121 = None
        permute_875: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1122, [1, 0])
        mm_363: "f32[384, 512]" = torch.ops.aten.mm.default(permute_875, view_136);  permute_875 = view_136 = None
        permute_61: "f32[512, 384]" = torch.ops.aten.permute.default(primals_50, [1, 0]);  primals_50 = None
        permute_877: "f32[384, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        mm_364: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1122, permute_877);  view_1122 = permute_877 = None
        view_1123: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_364, [32, 128, 512]);  mm_364 = None
        add_295: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_294, view_1123);  add_294 = view_1123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_851: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_295, primals_49);  primals_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_89: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_41, rsqrt_10)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_852: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_295, mul_89);  add_295 = mul_89 = None
        sum_111: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_852, [0, 1], True);  mul_852 = None
        view_1124: "f32[512]" = torch.ops.aten.reshape.default(sum_111, [512]);  sum_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_853: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_851, add_41)
        mul_854: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_851, rsqrt_10);  mul_851 = None
        sum_112: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_853, [2], True);  mul_853 = None
        add_296: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_292, mul_854);  add_292 = mul_854 = None
        pow_132: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_10, 3);  rsqrt_10 = None
        mul_855: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_112, -0.5);  sum_112 = None
        mul_856: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_855, pow_132);  mul_855 = pow_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_130: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_856, [32, 128, 512]);  mul_856 = None
        div_61: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_130, 512);  expand_130 = None
        pow_133: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_41, 1.0);  add_41 = None
        mul_857: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_133, 2.0);  pow_133 = None
        mul_858: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_61, mul_857);  div_61 = mul_857 = None
        add_297: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_296, mul_858);  add_296 = mul_858 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_69: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_21, torch.float32);  gt_21 = None
        mul_859: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_69, 1.1111111111111112);  convert_element_type_69 = None
        mul_860: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_297, mul_859);  mul_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1125: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_860, [4096, 512]);  mul_860 = None
        permute_879: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1125, [1, 0])
        mm_365: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_879, view_134);  permute_879 = view_134 = None
        permute_60: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_48, [1, 0]);  primals_48 = None
        permute_881: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        mm_366: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1125, permute_881);  view_1125 = permute_881 = None
        view_1126: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_366, [32, 128, 1024]);  mm_366 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_70: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_20, torch.float32);  gt_20 = None
        mul_861: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_70, 1.1111111111111112);  convert_element_type_70 = None
        mul_862: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1126, mul_861);  view_1126 = mul_861 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_131: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_32, [32, 128, 1024]);  mm_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_80: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_131, 0.5)
        pow_15: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 3.0)
        mul_81: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_15, 0.044715);  pow_15 = None
        add_39: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_131, mul_81);  mul_81 = None
        mul_82: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_39, 0.7978845608028654);  add_39 = None
        tanh_4: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_82);  mul_82 = None
        add_40: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_4, 1.0)
        mul_83: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_80, add_40)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_863: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_862, mul_83);  mul_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_133: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_33, [32, 128, 1024]);  mm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_864: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_862, view_133);  mul_862 = view_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1127: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_863, [4096, 1024]);  mul_863 = None
        permute_883: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1127, [1, 0])
        mm_367: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_883, view_130);  permute_883 = None
        permute_59: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_47, [1, 0]);  primals_47 = None
        permute_885: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        mm_368: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1127, permute_885);  view_1127 = permute_885 = None
        view_1128: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_368, [32, 128, 512]);  mm_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_865: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_864, mul_80);  mul_80 = None
        mul_866: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_864, add_40);  mul_864 = add_40 = None
        mul_867: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_4, tanh_4);  tanh_4 = None
        sub_40: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_867);  mul_867 = None
        mul_868: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_865, sub_40);  mul_865 = sub_40 = None
        mul_869: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_868, 0.7978845608028654);  mul_868 = None
        mul_870: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_869, 0.044715)
        pow_134: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_131, 2.0);  view_131 = None
        mul_871: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_134, 3.0);  pow_134 = None
        mul_872: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_870, mul_871);  mul_870 = mul_871 = None
        add_298: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_869, mul_872);  mul_869 = mul_872 = None
        mul_873: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_866, 0.5);  mul_866 = None
        add_299: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_298, mul_873);  add_298 = mul_873 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1129: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_299, [4096, 1024]);  add_299 = None
        permute_887: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        mm_369: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_887, view_130);  permute_887 = view_130 = None
        permute_58: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_46, [1, 0]);  primals_46 = None
        permute_889: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        mm_370: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1129, permute_889);  view_1129 = permute_889 = None
        view_1130: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_370, [32, 128, 512]);  mm_370 = None
        add_300: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1128, view_1130);  view_1128 = view_1130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_874: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_300, primals_45);  primals_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_78: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_37, rsqrt_9)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_875: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_300, mul_78);  add_300 = mul_78 = None
        sum_113: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_875, [0, 1], True);  mul_875 = None
        view_1131: "f32[512]" = torch.ops.aten.reshape.default(sum_113, [512]);  sum_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_876: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_874, add_37)
        mul_877: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_874, rsqrt_9);  mul_874 = None
        sum_114: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_876, [2], True);  mul_876 = None
        add_301: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_297, mul_877);  add_297 = mul_877 = None
        pow_135: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_9, 3);  rsqrt_9 = None
        mul_878: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_114, -0.5);  sum_114 = None
        mul_879: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_878, pow_135);  mul_878 = pow_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_131: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_879, [32, 128, 512]);  mul_879 = None
        div_62: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_131, 512);  expand_131 = None
        pow_136: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_37, 1.0);  add_37 = None
        mul_880: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_136, 2.0);  pow_136 = None
        mul_881: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_62, mul_880);  div_62 = mul_880 = None
        add_302: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_301, mul_881);  add_301 = mul_881 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_71: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_19, torch.float32);  gt_19 = None
        mul_882: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_71, 1.1111111111111112);  convert_element_type_71 = None
        mul_883: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_302, mul_882);  mul_882 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1132: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_883, [4096, 512]);  mul_883 = None
        permute_891: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        mm_371: "f32[512, 384]" = torch.ops.aten.mm.default(permute_891, view_128);  permute_891 = view_128 = None
        permute_57: "f32[384, 512]" = torch.ops.aten.permute.default(primals_44, [1, 0]);  primals_44 = None
        permute_893: "f32[512, 384]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        mm_372: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1132, permute_893);  view_1132 = permute_893 = None
        view_1133: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_372, [32, 128, 384]);  mm_372 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1134: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1133, [32, 128, 6, 64]);  view_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_895: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1134, [0, 2, 1, 3]);  view_1134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_258: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_895, memory_format = torch.contiguous_format);  permute_895 = None
        view_1135: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_258, [192, 128, 64]);  clone_258 = None
        bmm_124: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_896, view_1135);  permute_896 = None
        bmm_125: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1135, permute_897);  view_1135 = permute_897 = None
        view_1136: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_124, [32, 6, 128, 64]);  bmm_124 = None
        view_1137: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_125, [32, 6, 128, 128]);  bmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_72: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_18, torch.float32);  gt_18 = None
        mul_884: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_72, 1.1111111111111112);  convert_element_type_72 = None
        mul_885: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1137, mul_884);  view_1137 = mul_884 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_886: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_885, div_6);  mul_885 = None
        sum_115: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_886, [-1], True)
        neg_21: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_6);  div_6 = None
        fma_19: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_21, sum_115, mul_886);  neg_21 = sum_115 = mul_886 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1138: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_19, [192, 128, 128]);  fma_19 = None
        view_1140: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1138, [32, 6, 128, 128]);  view_1138 = None
        view_1141: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1140, [192, 128, 128])
        add_303: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_293, view_1140);  add_293 = view_1140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_126: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_898, view_1141);  permute_898 = None
        bmm_127: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1141, permute_899);  view_1141 = permute_899 = None
        view_1143: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_126, [32, 6, 64, 128]);  bmm_126 = None
        view_1144: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_127, [32, 6, 128, 64]);  bmm_127 = None
        permute_900: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1143, [0, 1, 3, 2]);  view_1143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_901: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1136, [0, 2, 1, 3]);  view_1136 = None
        clone_261: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_901, memory_format = torch.contiguous_format);  permute_901 = None
        view_1145: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_261, [32, 128, 384]);  clone_261 = None
        view_1146: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1145, [4096, 384]);  view_1145 = None
        permute_902: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1146, [1, 0])
        mm_373: "f32[384, 512]" = torch.ops.aten.mm.default(permute_902, view_109);  permute_902 = None
        permute_53: "f32[512, 384]" = torch.ops.aten.permute.default(primals_43, [1, 0]);  primals_43 = None
        permute_904: "f32[384, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        mm_374: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1146, permute_904);  view_1146 = permute_904 = None
        view_1147: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_374, [32, 128, 512]);  mm_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_906: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_900, [0, 2, 1, 3]);  permute_900 = None
        view_1148: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_906, [32, 128, 384]);  permute_906 = None
        clone_262: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1148, memory_format = torch.contiguous_format);  view_1148 = None
        view_1149: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_262, [4096, 384]);  clone_262 = None
        permute_907: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1149, [1, 0])
        mm_375: "f32[384, 512]" = torch.ops.aten.mm.default(permute_907, view_109);  permute_907 = None
        permute_51: "f32[512, 384]" = torch.ops.aten.permute.default(primals_42, [1, 0]);  primals_42 = None
        permute_909: "f32[384, 512]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        mm_376: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1149, permute_909);  view_1149 = permute_909 = None
        view_1150: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_376, [32, 128, 512]);  mm_376 = None
        add_304: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1147, view_1150);  view_1147 = view_1150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_911: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1144, [0, 2, 1, 3]);  view_1144 = None
        clone_263: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_911, memory_format = torch.contiguous_format);  permute_911 = None
        view_1151: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_263, [32, 128, 384]);  clone_263 = None
        view_1152: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1151, [4096, 384]);  view_1151 = None
        permute_912: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1152, [1, 0])
        mm_377: "f32[384, 512]" = torch.ops.aten.mm.default(permute_912, view_109);  permute_912 = view_109 = None
        permute_49: "f32[512, 384]" = torch.ops.aten.permute.default(primals_41, [1, 0]);  primals_41 = None
        permute_914: "f32[384, 512]" = torch.ops.aten.permute.default(permute_49, [1, 0]);  permute_49 = None
        mm_378: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1152, permute_914);  view_1152 = permute_914 = None
        view_1153: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_378, [32, 128, 512]);  mm_378 = None
        add_305: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_304, view_1153);  add_304 = view_1153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_887: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_305, primals_40);  primals_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_72: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_34, rsqrt_8)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_888: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_305, mul_72);  add_305 = mul_72 = None
        sum_116: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_888, [0, 1], True);  mul_888 = None
        view_1154: "f32[512]" = torch.ops.aten.reshape.default(sum_116, [512]);  sum_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_889: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_887, add_34)
        mul_890: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_887, rsqrt_8);  mul_887 = None
        sum_117: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_889, [2], True);  mul_889 = None
        add_306: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_302, mul_890);  add_302 = mul_890 = None
        pow_137: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_8, 3);  rsqrt_8 = None
        mul_891: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_117, -0.5);  sum_117 = None
        mul_892: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_891, pow_137);  mul_891 = pow_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_132: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_892, [32, 128, 512]);  mul_892 = None
        div_63: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_132, 512);  expand_132 = None
        pow_138: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_34, 1.0);  add_34 = None
        mul_893: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_138, 2.0);  pow_138 = None
        mul_894: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_63, mul_893);  div_63 = mul_893 = None
        add_307: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_306, mul_894);  add_306 = mul_894 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_73: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_17, torch.float32);  gt_17 = None
        mul_895: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_73, 1.1111111111111112);  convert_element_type_73 = None
        mul_896: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_307, mul_895);  mul_895 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1155: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_896, [4096, 512]);  mul_896 = None
        permute_916: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1155, [1, 0])
        mm_379: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_916, view_107);  permute_916 = view_107 = None
        permute_48: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_39, [1, 0]);  primals_39 = None
        permute_918: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_48, [1, 0]);  permute_48 = None
        mm_380: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1155, permute_918);  view_1155 = permute_918 = None
        view_1156: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_380, [32, 128, 1024]);  mm_380 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_74: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_16, torch.float32);  gt_16 = None
        mul_897: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_74, 1.1111111111111112);  convert_element_type_74 = None
        mul_898: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1156, mul_897);  view_1156 = mul_897 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_104: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_25, [32, 128, 1024]);  mm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_63: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_104, 0.5)
        pow_12: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_104, 3.0)
        mul_64: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_12, 0.044715);  pow_12 = None
        add_32: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_104, mul_64);  mul_64 = None
        mul_65: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_32, 0.7978845608028654);  add_32 = None
        tanh_3: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_65);  mul_65 = None
        add_33: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_3, 1.0)
        mul_66: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_63, add_33)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_899: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_898, mul_66);  mul_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_106: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_26, [32, 128, 1024]);  mm_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_900: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_898, view_106);  mul_898 = view_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1157: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_899, [4096, 1024]);  mul_899 = None
        permute_920: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1157, [1, 0])
        mm_381: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_920, view_103);  permute_920 = None
        permute_47: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_38, [1, 0]);  primals_38 = None
        permute_922: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        mm_382: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1157, permute_922);  view_1157 = permute_922 = None
        view_1158: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_382, [32, 128, 512]);  mm_382 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_901: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_900, mul_63);  mul_63 = None
        mul_902: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_900, add_33);  mul_900 = add_33 = None
        mul_903: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_3, tanh_3);  tanh_3 = None
        sub_41: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_903);  mul_903 = None
        mul_904: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_901, sub_41);  mul_901 = sub_41 = None
        mul_905: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_904, 0.7978845608028654);  mul_904 = None
        mul_906: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_905, 0.044715)
        pow_139: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_104, 2.0);  view_104 = None
        mul_907: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_139, 3.0);  pow_139 = None
        mul_908: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_906, mul_907);  mul_906 = mul_907 = None
        add_308: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_905, mul_908);  mul_905 = mul_908 = None
        mul_909: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_902, 0.5);  mul_902 = None
        add_309: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_308, mul_909);  add_308 = mul_909 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1159: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_309, [4096, 1024]);  add_309 = None
        permute_924: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1159, [1, 0])
        mm_383: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_924, view_103);  permute_924 = view_103 = None
        permute_46: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_37, [1, 0]);  primals_37 = None
        permute_926: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_46, [1, 0]);  permute_46 = None
        mm_384: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1159, permute_926);  view_1159 = permute_926 = None
        view_1160: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_384, [32, 128, 512]);  mm_384 = None
        add_310: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1158, view_1160);  view_1158 = view_1160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_910: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_310, primals_36);  primals_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_61: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_30, rsqrt_7)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_911: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_310, mul_61);  add_310 = mul_61 = None
        sum_118: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_911, [0, 1], True);  mul_911 = None
        view_1161: "f32[512]" = torch.ops.aten.reshape.default(sum_118, [512]);  sum_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_912: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_910, add_30)
        mul_913: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_910, rsqrt_7);  mul_910 = None
        sum_119: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_912, [2], True);  mul_912 = None
        add_311: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_307, mul_913);  add_307 = mul_913 = None
        pow_140: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_7, 3);  rsqrt_7 = None
        mul_914: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_119, -0.5);  sum_119 = None
        mul_915: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_914, pow_140);  mul_914 = pow_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_133: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_915, [32, 128, 512]);  mul_915 = None
        div_64: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_133, 512);  expand_133 = None
        pow_141: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_30, 1.0);  add_30 = None
        mul_916: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_141, 2.0);  pow_141 = None
        mul_917: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_64, mul_916);  div_64 = mul_916 = None
        add_312: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_311, mul_917);  add_311 = mul_917 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_75: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_15, torch.float32);  gt_15 = None
        mul_918: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_75, 1.1111111111111112);  convert_element_type_75 = None
        mul_919: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_312, mul_918);  mul_918 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1162: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_919, [4096, 512]);  mul_919 = None
        permute_928: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1162, [1, 0])
        mm_385: "f32[512, 384]" = torch.ops.aten.mm.default(permute_928, view_101);  permute_928 = view_101 = None
        permute_45: "f32[384, 512]" = torch.ops.aten.permute.default(primals_35, [1, 0]);  primals_35 = None
        permute_930: "f32[512, 384]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        mm_386: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1162, permute_930);  view_1162 = permute_930 = None
        view_1163: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_386, [32, 128, 384]);  mm_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1164: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1163, [32, 128, 6, 64]);  view_1163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_932: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1164, [0, 2, 1, 3]);  view_1164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_267: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_932, memory_format = torch.contiguous_format);  permute_932 = None
        view_1165: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_267, [192, 128, 64]);  clone_267 = None
        bmm_128: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_933, view_1165);  permute_933 = None
        bmm_129: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1165, permute_934);  view_1165 = permute_934 = None
        view_1166: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_128, [32, 6, 128, 64]);  bmm_128 = None
        view_1167: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_129, [32, 6, 128, 128]);  bmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_76: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_14, torch.float32);  gt_14 = None
        mul_920: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_76, 1.1111111111111112);  convert_element_type_76 = None
        mul_921: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1167, mul_920);  view_1167 = mul_920 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_922: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_921, div_5);  mul_921 = None
        sum_120: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_922, [-1], True)
        neg_22: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_5);  div_5 = None
        fma_20: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_22, sum_120, mul_922);  neg_22 = sum_120 = mul_922 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1168: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_20, [192, 128, 128]);  fma_20 = None
        view_1170: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1168, [32, 6, 128, 128]);  view_1168 = None
        view_1171: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1170, [192, 128, 128])
        add_313: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_303, view_1170);  add_303 = view_1170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_130: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_935, view_1171);  permute_935 = None
        bmm_131: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1171, permute_936);  view_1171 = permute_936 = None
        view_1173: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_130, [32, 6, 64, 128]);  bmm_130 = None
        view_1174: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_131, [32, 6, 128, 64]);  bmm_131 = None
        permute_937: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1173, [0, 1, 3, 2]);  view_1173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_938: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1166, [0, 2, 1, 3]);  view_1166 = None
        clone_270: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_938, memory_format = torch.contiguous_format);  permute_938 = None
        view_1175: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_270, [32, 128, 384]);  clone_270 = None
        view_1176: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1175, [4096, 384]);  view_1175 = None
        permute_939: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1176, [1, 0])
        mm_387: "f32[384, 512]" = torch.ops.aten.mm.default(permute_939, view_82);  permute_939 = None
        permute_41: "f32[512, 384]" = torch.ops.aten.permute.default(primals_34, [1, 0]);  primals_34 = None
        permute_941: "f32[384, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        mm_388: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1176, permute_941);  view_1176 = permute_941 = None
        view_1177: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_388, [32, 128, 512]);  mm_388 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_943: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_937, [0, 2, 1, 3]);  permute_937 = None
        view_1178: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_943, [32, 128, 384]);  permute_943 = None
        clone_271: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1178, memory_format = torch.contiguous_format);  view_1178 = None
        view_1179: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_271, [4096, 384]);  clone_271 = None
        permute_944: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1179, [1, 0])
        mm_389: "f32[384, 512]" = torch.ops.aten.mm.default(permute_944, view_82);  permute_944 = None
        permute_39: "f32[512, 384]" = torch.ops.aten.permute.default(primals_33, [1, 0]);  primals_33 = None
        permute_946: "f32[384, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        mm_390: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1179, permute_946);  view_1179 = permute_946 = None
        view_1180: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_390, [32, 128, 512]);  mm_390 = None
        add_314: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1177, view_1180);  view_1177 = view_1180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_948: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1174, [0, 2, 1, 3]);  view_1174 = None
        clone_272: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_948, memory_format = torch.contiguous_format);  permute_948 = None
        view_1181: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_272, [32, 128, 384]);  clone_272 = None
        view_1182: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1181, [4096, 384]);  view_1181 = None
        permute_949: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        mm_391: "f32[384, 512]" = torch.ops.aten.mm.default(permute_949, view_82);  permute_949 = view_82 = None
        permute_37: "f32[512, 384]" = torch.ops.aten.permute.default(primals_32, [1, 0]);  primals_32 = None
        permute_951: "f32[384, 512]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        mm_392: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1182, permute_951);  view_1182 = permute_951 = None
        view_1183: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_392, [32, 128, 512]);  mm_392 = None
        add_315: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_314, view_1183);  add_314 = view_1183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_923: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_315, primals_31);  primals_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_55: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_27, rsqrt_6)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_924: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_315, mul_55);  add_315 = mul_55 = None
        sum_121: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_924, [0, 1], True);  mul_924 = None
        view_1184: "f32[512]" = torch.ops.aten.reshape.default(sum_121, [512]);  sum_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_925: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_923, add_27)
        mul_926: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_923, rsqrt_6);  mul_923 = None
        sum_122: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_925, [2], True);  mul_925 = None
        add_316: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_312, mul_926);  add_312 = mul_926 = None
        pow_142: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_6, 3);  rsqrt_6 = None
        mul_927: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_122, -0.5);  sum_122 = None
        mul_928: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_927, pow_142);  mul_927 = pow_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_134: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_928, [32, 128, 512]);  mul_928 = None
        div_65: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_134, 512);  expand_134 = None
        pow_143: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_27, 1.0);  add_27 = None
        mul_929: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_143, 2.0);  pow_143 = None
        mul_930: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_65, mul_929);  div_65 = mul_929 = None
        add_317: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_316, mul_930);  add_316 = mul_930 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_77: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_13, torch.float32);  gt_13 = None
        mul_931: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_77, 1.1111111111111112);  convert_element_type_77 = None
        mul_932: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_317, mul_931);  mul_931 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1185: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_932, [4096, 512]);  mul_932 = None
        permute_953: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1185, [1, 0])
        mm_393: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_953, view_80);  permute_953 = view_80 = None
        permute_36: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_30, [1, 0]);  primals_30 = None
        permute_955: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        mm_394: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1185, permute_955);  view_1185 = permute_955 = None
        view_1186: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_394, [32, 128, 1024]);  mm_394 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_78: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_12, torch.float32);  gt_12 = None
        mul_933: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_78, 1.1111111111111112);  convert_element_type_78 = None
        mul_934: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1186, mul_933);  view_1186 = mul_933 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_77: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_18, [32, 128, 1024]);  mm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_46: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_77, 0.5)
        pow_9: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_77, 3.0)
        mul_47: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_9, 0.044715);  pow_9 = None
        add_25: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_77, mul_47);  mul_47 = None
        mul_48: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_25, 0.7978845608028654);  add_25 = None
        tanh_2: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_48);  mul_48 = None
        add_26: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_2, 1.0)
        mul_49: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_46, add_26)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_935: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_934, mul_49);  mul_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_79: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_19, [32, 128, 1024]);  mm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_936: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_934, view_79);  mul_934 = view_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1187: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_935, [4096, 1024]);  mul_935 = None
        permute_957: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1187, [1, 0])
        mm_395: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_957, view_76);  permute_957 = None
        permute_35: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_29, [1, 0]);  primals_29 = None
        permute_959: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        mm_396: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1187, permute_959);  view_1187 = permute_959 = None
        view_1188: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_396, [32, 128, 512]);  mm_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_937: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_936, mul_46);  mul_46 = None
        mul_938: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_936, add_26);  mul_936 = add_26 = None
        mul_939: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_2, tanh_2);  tanh_2 = None
        sub_42: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_939);  mul_939 = None
        mul_940: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_937, sub_42);  mul_937 = sub_42 = None
        mul_941: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_940, 0.7978845608028654);  mul_940 = None
        mul_942: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_941, 0.044715)
        pow_144: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_77, 2.0);  view_77 = None
        mul_943: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_144, 3.0);  pow_144 = None
        mul_944: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_942, mul_943);  mul_942 = mul_943 = None
        add_318: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_941, mul_944);  mul_941 = mul_944 = None
        mul_945: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_938, 0.5);  mul_938 = None
        add_319: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_318, mul_945);  add_318 = mul_945 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1189: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_319, [4096, 1024]);  add_319 = None
        permute_961: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1189, [1, 0])
        mm_397: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_961, view_76);  permute_961 = view_76 = None
        permute_34: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_28, [1, 0]);  primals_28 = None
        permute_963: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        mm_398: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1189, permute_963);  view_1189 = permute_963 = None
        view_1190: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_398, [32, 128, 512]);  mm_398 = None
        add_320: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1188, view_1190);  view_1188 = view_1190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_946: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_320, primals_27);  primals_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_44: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_23, rsqrt_5)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_947: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_320, mul_44);  add_320 = mul_44 = None
        sum_123: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_947, [0, 1], True);  mul_947 = None
        view_1191: "f32[512]" = torch.ops.aten.reshape.default(sum_123, [512]);  sum_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_948: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_946, add_23)
        mul_949: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_946, rsqrt_5);  mul_946 = None
        sum_124: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_948, [2], True);  mul_948 = None
        add_321: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_317, mul_949);  add_317 = mul_949 = None
        pow_145: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_5, 3);  rsqrt_5 = None
        mul_950: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_124, -0.5);  sum_124 = None
        mul_951: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_950, pow_145);  mul_950 = pow_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_135: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_951, [32, 128, 512]);  mul_951 = None
        div_66: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_135, 512);  expand_135 = None
        pow_146: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_23, 1.0);  add_23 = None
        mul_952: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_146, 2.0);  pow_146 = None
        mul_953: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_66, mul_952);  div_66 = mul_952 = None
        add_322: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_321, mul_953);  add_321 = mul_953 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_79: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_11, torch.float32);  gt_11 = None
        mul_954: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_79, 1.1111111111111112);  convert_element_type_79 = None
        mul_955: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_322, mul_954);  mul_954 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1192: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_955, [4096, 512]);  mul_955 = None
        permute_965: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1192, [1, 0])
        mm_399: "f32[512, 384]" = torch.ops.aten.mm.default(permute_965, view_74);  permute_965 = view_74 = None
        permute_33: "f32[384, 512]" = torch.ops.aten.permute.default(primals_26, [1, 0]);  primals_26 = None
        permute_967: "f32[512, 384]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        mm_400: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1192, permute_967);  view_1192 = permute_967 = None
        view_1193: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_400, [32, 128, 384]);  mm_400 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1194: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1193, [32, 128, 6, 64]);  view_1193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_969: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1194, [0, 2, 1, 3]);  view_1194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_276: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_969, memory_format = torch.contiguous_format);  permute_969 = None
        view_1195: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_276, [192, 128, 64]);  clone_276 = None
        bmm_132: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_970, view_1195);  permute_970 = None
        bmm_133: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1195, permute_971);  view_1195 = permute_971 = None
        view_1196: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_132, [32, 6, 128, 64]);  bmm_132 = None
        view_1197: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_133, [32, 6, 128, 128]);  bmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_80: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_10, torch.float32);  gt_10 = None
        mul_956: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_80, 1.1111111111111112);  convert_element_type_80 = None
        mul_957: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1197, mul_956);  view_1197 = mul_956 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_958: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_957, div_4);  mul_957 = None
        sum_125: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_958, [-1], True)
        neg_23: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_4);  div_4 = None
        fma_21: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_23, sum_125, mul_958);  neg_23 = sum_125 = mul_958 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1198: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_21, [192, 128, 128]);  fma_21 = None
        view_1200: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1198, [32, 6, 128, 128]);  view_1198 = None
        view_1201: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1200, [192, 128, 128])
        add_323: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_313, view_1200);  add_313 = view_1200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_134: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_972, view_1201);  permute_972 = None
        bmm_135: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1201, permute_973);  view_1201 = permute_973 = None
        view_1203: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_134, [32, 6, 64, 128]);  bmm_134 = None
        view_1204: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_135, [32, 6, 128, 64]);  bmm_135 = None
        permute_974: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1203, [0, 1, 3, 2]);  view_1203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_975: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1196, [0, 2, 1, 3]);  view_1196 = None
        clone_279: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_975, memory_format = torch.contiguous_format);  permute_975 = None
        view_1205: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_279, [32, 128, 384]);  clone_279 = None
        view_1206: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1205, [4096, 384]);  view_1205 = None
        permute_976: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1206, [1, 0])
        mm_401: "f32[384, 512]" = torch.ops.aten.mm.default(permute_976, view_55);  permute_976 = None
        permute_29: "f32[512, 384]" = torch.ops.aten.permute.default(primals_25, [1, 0]);  primals_25 = None
        permute_978: "f32[384, 512]" = torch.ops.aten.permute.default(permute_29, [1, 0]);  permute_29 = None
        mm_402: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1206, permute_978);  view_1206 = permute_978 = None
        view_1207: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_402, [32, 128, 512]);  mm_402 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_980: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_974, [0, 2, 1, 3]);  permute_974 = None
        view_1208: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_980, [32, 128, 384]);  permute_980 = None
        clone_280: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1208, memory_format = torch.contiguous_format);  view_1208 = None
        view_1209: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_280, [4096, 384]);  clone_280 = None
        permute_981: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1209, [1, 0])
        mm_403: "f32[384, 512]" = torch.ops.aten.mm.default(permute_981, view_55);  permute_981 = None
        permute_27: "f32[512, 384]" = torch.ops.aten.permute.default(primals_24, [1, 0]);  primals_24 = None
        permute_983: "f32[384, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        mm_404: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1209, permute_983);  view_1209 = permute_983 = None
        view_1210: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_404, [32, 128, 512]);  mm_404 = None
        add_324: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1207, view_1210);  view_1207 = view_1210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_985: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1204, [0, 2, 1, 3]);  view_1204 = None
        clone_281: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_985, memory_format = torch.contiguous_format);  permute_985 = None
        view_1211: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_281, [32, 128, 384]);  clone_281 = None
        view_1212: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1211, [4096, 384]);  view_1211 = None
        permute_986: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1212, [1, 0])
        mm_405: "f32[384, 512]" = torch.ops.aten.mm.default(permute_986, view_55);  permute_986 = view_55 = None
        permute_25: "f32[512, 384]" = torch.ops.aten.permute.default(primals_23, [1, 0]);  primals_23 = None
        permute_988: "f32[384, 512]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        mm_406: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1212, permute_988);  view_1212 = permute_988 = None
        view_1213: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_406, [32, 128, 512]);  mm_406 = None
        add_325: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_324, view_1213);  add_324 = view_1213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_959: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_325, primals_22);  primals_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_38: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_20, rsqrt_4)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_960: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_325, mul_38);  add_325 = mul_38 = None
        sum_126: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_960, [0, 1], True);  mul_960 = None
        view_1214: "f32[512]" = torch.ops.aten.reshape.default(sum_126, [512]);  sum_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_961: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_959, add_20)
        mul_962: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_959, rsqrt_4);  mul_959 = None
        sum_127: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_961, [2], True);  mul_961 = None
        add_326: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_322, mul_962);  add_322 = mul_962 = None
        pow_147: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_4, 3);  rsqrt_4 = None
        mul_963: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_127, -0.5);  sum_127 = None
        mul_964: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_963, pow_147);  mul_963 = pow_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_136: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_964, [32, 128, 512]);  mul_964 = None
        div_67: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_136, 512);  expand_136 = None
        pow_148: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_20, 1.0);  add_20 = None
        mul_965: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_148, 2.0);  pow_148 = None
        mul_966: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_67, mul_965);  div_67 = mul_965 = None
        add_327: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_326, mul_966);  add_326 = mul_966 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_81: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_9, torch.float32);  gt_9 = None
        mul_967: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_81, 1.1111111111111112);  convert_element_type_81 = None
        mul_968: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_327, mul_967);  mul_967 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1215: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_968, [4096, 512]);  mul_968 = None
        permute_990: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1215, [1, 0])
        mm_407: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_990, view_53);  permute_990 = view_53 = None
        permute_24: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_21, [1, 0]);  primals_21 = None
        permute_992: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_24, [1, 0]);  permute_24 = None
        mm_408: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1215, permute_992);  view_1215 = permute_992 = None
        view_1216: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_408, [32, 128, 1024]);  mm_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_82: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_8, torch.float32);  gt_8 = None
        mul_969: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_82, 1.1111111111111112);  convert_element_type_82 = None
        mul_970: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1216, mul_969);  view_1216 = mul_969 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_50: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_11, [32, 128, 1024]);  mm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_29: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_50, 0.5)
        pow_6: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_50, 3.0)
        mul_30: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_6, 0.044715);  pow_6 = None
        add_18: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_50, mul_30);  mul_30 = None
        mul_31: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_18, 0.7978845608028654);  add_18 = None
        tanh_1: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_31);  mul_31 = None
        add_19: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh_1, 1.0)
        mul_32: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_29, add_19)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_971: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_970, mul_32);  mul_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_52: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_12, [32, 128, 1024]);  mm_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_972: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_970, view_52);  mul_970 = view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1217: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_971, [4096, 1024]);  mul_971 = None
        permute_994: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1217, [1, 0])
        mm_409: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_994, view_49);  permute_994 = None
        permute_23: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_20, [1, 0]);  primals_20 = None
        permute_996: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        mm_410: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1217, permute_996);  view_1217 = permute_996 = None
        view_1218: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_410, [32, 128, 512]);  mm_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_973: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_972, mul_29);  mul_29 = None
        mul_974: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_972, add_19);  mul_972 = add_19 = None
        mul_975: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh_1, tanh_1);  tanh_1 = None
        sub_43: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_975);  mul_975 = None
        mul_976: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_973, sub_43);  mul_973 = sub_43 = None
        mul_977: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_976, 0.7978845608028654);  mul_976 = None
        mul_978: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_977, 0.044715)
        pow_149: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_50, 2.0);  view_50 = None
        mul_979: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_149, 3.0);  pow_149 = None
        mul_980: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_978, mul_979);  mul_978 = mul_979 = None
        add_328: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_977, mul_980);  mul_977 = mul_980 = None
        mul_981: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_974, 0.5);  mul_974 = None
        add_329: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_328, mul_981);  add_328 = mul_981 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1219: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_329, [4096, 1024]);  add_329 = None
        permute_998: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1219, [1, 0])
        mm_411: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_998, view_49);  permute_998 = view_49 = None
        permute_22: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_19, [1, 0]);  primals_19 = None
        permute_1000: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        mm_412: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1219, permute_1000);  view_1219 = permute_1000 = None
        view_1220: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_412, [32, 128, 512]);  mm_412 = None
        add_330: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1218, view_1220);  view_1218 = view_1220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_982: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_330, primals_18);  primals_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_27: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_16, rsqrt_3)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_983: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_330, mul_27);  add_330 = mul_27 = None
        sum_128: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_983, [0, 1], True);  mul_983 = None
        view_1221: "f32[512]" = torch.ops.aten.reshape.default(sum_128, [512]);  sum_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_984: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_982, add_16)
        mul_985: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_982, rsqrt_3);  mul_982 = None
        sum_129: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_984, [2], True);  mul_984 = None
        add_331: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_327, mul_985);  add_327 = mul_985 = None
        pow_150: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_3, 3);  rsqrt_3 = None
        mul_986: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_129, -0.5);  sum_129 = None
        mul_987: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_986, pow_150);  mul_986 = pow_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_137: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_987, [32, 128, 512]);  mul_987 = None
        div_68: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_137, 512);  expand_137 = None
        pow_151: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_16, 1.0);  add_16 = None
        mul_988: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_151, 2.0);  pow_151 = None
        mul_989: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_68, mul_988);  div_68 = mul_988 = None
        add_332: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_331, mul_989);  add_331 = mul_989 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_83: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_7, torch.float32);  gt_7 = None
        mul_990: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_83, 1.1111111111111112);  convert_element_type_83 = None
        mul_991: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_332, mul_990);  mul_990 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1222: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_991, [4096, 512]);  mul_991 = None
        permute_1002: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1222, [1, 0])
        mm_413: "f32[512, 384]" = torch.ops.aten.mm.default(permute_1002, view_47);  permute_1002 = view_47 = None
        permute_21: "f32[384, 512]" = torch.ops.aten.permute.default(primals_17, [1, 0]);  primals_17 = None
        permute_1004: "f32[512, 384]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        mm_414: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1222, permute_1004);  view_1222 = permute_1004 = None
        view_1223: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_414, [32, 128, 384]);  mm_414 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1224: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1223, [32, 128, 6, 64]);  view_1223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1006: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1224, [0, 2, 1, 3]);  view_1224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_285: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_1006, memory_format = torch.contiguous_format);  permute_1006 = None
        view_1225: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_285, [192, 128, 64]);  clone_285 = None
        bmm_136: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_1007, view_1225);  permute_1007 = None
        bmm_137: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1225, permute_1008);  view_1225 = permute_1008 = None
        view_1226: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_136, [32, 6, 128, 64]);  bmm_136 = None
        view_1227: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_137, [32, 6, 128, 128]);  bmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_84: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_6, torch.float32);  gt_6 = None
        mul_992: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_84, 1.1111111111111112);  convert_element_type_84 = None
        mul_993: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1227, mul_992);  view_1227 = mul_992 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        mul_994: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_993, div_3);  mul_993 = None
        sum_130: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_994, [-1], True)
        neg_24: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_3);  div_3 = None
        fma_22: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_24, sum_130, mul_994);  neg_24 = sum_130 = mul_994 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1228: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_22, [192, 128, 128]);  fma_22 = None
        view_1230: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1228, [32, 6, 128, 128]);  view_1228 = None
        view_1231: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1230, [192, 128, 128])
        add_333: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_323, view_1230);  add_323 = view_1230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_138: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_1009, view_1231);  permute_1009 = None
        bmm_139: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1231, permute_1010);  view_1231 = permute_1010 = None
        view_1233: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_138, [32, 6, 64, 128]);  bmm_138 = None
        view_1234: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_139, [32, 6, 128, 64]);  bmm_139 = None
        permute_1011: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1233, [0, 1, 3, 2]);  view_1233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1012: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1226, [0, 2, 1, 3]);  view_1226 = None
        clone_288: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1012, memory_format = torch.contiguous_format);  permute_1012 = None
        view_1235: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_288, [32, 128, 384]);  clone_288 = None
        view_1236: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1235, [4096, 384]);  view_1235 = None
        permute_1013: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1236, [1, 0])
        mm_415: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1013, view_28);  permute_1013 = None
        permute_17: "f32[512, 384]" = torch.ops.aten.permute.default(primals_16, [1, 0]);  primals_16 = None
        permute_1015: "f32[384, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        mm_416: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1236, permute_1015);  view_1236 = permute_1015 = None
        view_1237: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_416, [32, 128, 512]);  mm_416 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1017: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_1011, [0, 2, 1, 3]);  permute_1011 = None
        view_1238: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_1017, [32, 128, 384]);  permute_1017 = None
        clone_289: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1238, memory_format = torch.contiguous_format);  view_1238 = None
        view_1239: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_289, [4096, 384]);  clone_289 = None
        permute_1018: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1239, [1, 0])
        mm_417: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1018, view_28);  permute_1018 = None
        permute_15: "f32[512, 384]" = torch.ops.aten.permute.default(primals_15, [1, 0]);  primals_15 = None
        permute_1020: "f32[384, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        mm_418: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1239, permute_1020);  view_1239 = permute_1020 = None
        view_1240: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_418, [32, 128, 512]);  mm_418 = None
        add_334: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1237, view_1240);  view_1237 = view_1240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1022: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1234, [0, 2, 1, 3]);  view_1234 = None
        clone_290: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1022, memory_format = torch.contiguous_format);  permute_1022 = None
        view_1241: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_290, [32, 128, 384]);  clone_290 = None
        view_1242: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1241, [4096, 384]);  view_1241 = None
        permute_1023: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1242, [1, 0])
        mm_419: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1023, view_28);  permute_1023 = view_28 = None
        permute_13: "f32[512, 384]" = torch.ops.aten.permute.default(primals_14, [1, 0]);  primals_14 = None
        permute_1025: "f32[384, 512]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        mm_420: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1242, permute_1025);  view_1242 = permute_1025 = None
        view_1243: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_420, [32, 128, 512]);  mm_420 = None
        add_335: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_334, view_1243);  add_334 = view_1243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_995: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_335, primals_13);  primals_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_21: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_13, rsqrt_2)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_996: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_335, mul_21);  add_335 = mul_21 = None
        sum_131: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_996, [0, 1], True);  mul_996 = None
        view_1244: "f32[512]" = torch.ops.aten.reshape.default(sum_131, [512]);  sum_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_997: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_995, add_13)
        mul_998: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_995, rsqrt_2);  mul_995 = None
        sum_132: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_997, [2], True);  mul_997 = None
        add_336: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_332, mul_998);  add_332 = mul_998 = None
        pow_152: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_2, 3);  rsqrt_2 = None
        mul_999: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_132, -0.5);  sum_132 = None
        mul_1000: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_999, pow_152);  mul_999 = pow_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_138: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_1000, [32, 128, 512]);  mul_1000 = None
        div_69: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_138, 512);  expand_138 = None
        pow_153: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_13, 1.0);  add_13 = None
        mul_1001: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_153, 2.0);  pow_153 = None
        mul_1002: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_69, mul_1001);  div_69 = mul_1001 = None
        add_337: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_336, mul_1002);  add_336 = mul_1002 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:140 in forward, code: hidden_states = hidden_states + self.dropout(forwarded_states)
        convert_element_type_85: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_5, torch.float32);  gt_5 = None
        mul_1003: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_85, 1.1111111111111112);  convert_element_type_85 = None
        mul_1004: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_337, mul_1003);  mul_1003 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:121 in forward, code: hidden_states = self.wo(hidden_states)
        view_1245: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_1004, [4096, 512]);  mul_1004 = None
        permute_1027: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1245, [1, 0])
        mm_421: "f32[512, 1024]" = torch.ops.aten.mm.default(permute_1027, view_26);  permute_1027 = view_26 = None
        permute_12: "f32[1024, 512]" = torch.ops.aten.permute.default(primals_12, [1, 0]);  primals_12 = None
        permute_1029: "f32[512, 1024]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        mm_422: "f32[4096, 1024]" = torch.ops.aten.mm.default(view_1245, permute_1029);  view_1245 = permute_1029 = None
        view_1246: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_422, [32, 128, 1024]);  mm_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:109 in forward, code: hidden_states = self.dropout(hidden_states)
        convert_element_type_86: "f32[32, 128, 1024]" = torch.ops.prims.convert_element_type.default(gt_4, torch.float32);  gt_4 = None
        mul_1005: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(convert_element_type_86, 1.1111111111111112);  convert_element_type_86 = None
        mul_1006: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_1246, mul_1005);  view_1246 = mul_1005 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_23: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_4, [32, 128, 1024]);  mm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_12: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(view_23, 0.5)
        pow_3: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 3.0)
        mul_13: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(pow_3, 0.044715);  pow_3 = None
        add_11: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(view_23, mul_13);  mul_13 = None
        mul_14: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(add_11, 0.7978845608028654);  add_11 = None
        tanh: "f32[32, 128, 1024]" = torch.ops.aten.tanh.default(mul_14);  mul_14 = None
        add_12: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(tanh, 1.0)
        mul_15: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_12, add_12)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_1007: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1006, mul_15);  mul_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_25: "f32[32, 128, 1024]" = torch.ops.aten.reshape.default(mm_5, [32, 128, 1024]);  mm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:108 in forward, code: hidden_states = hidden_gelu * hidden_linear
        mul_1008: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1006, view_25);  mul_1006 = view_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:107 in forward, code: hidden_linear = self.wi_1(hidden_states)
        view_1247: "f32[4096, 1024]" = torch.ops.aten.reshape.default(mul_1007, [4096, 1024]);  mul_1007 = None
        permute_1031: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1247, [1, 0])
        mm_423: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_1031, view_22);  permute_1031 = None
        permute_11: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_11, [1, 0]);  primals_11 = None
        permute_1033: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        mm_424: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1247, permute_1033);  view_1247 = permute_1033 = None
        view_1248: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_424, [32, 128, 512]);  mm_424 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:66 in forward, code: return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))
        mul_1009: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1008, mul_12);  mul_12 = None
        mul_1010: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1008, add_12);  mul_1008 = add_12 = None
        mul_1011: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(tanh, tanh);  tanh = None
        sub_44: "f32[32, 128, 1024]" = torch.ops.aten.sub.Tensor(1, mul_1011);  mul_1011 = None
        mul_1012: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1009, sub_44);  mul_1009 = sub_44 = None
        mul_1013: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1012, 0.7978845608028654);  mul_1012 = None
        mul_1014: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1013, 0.044715)
        pow_154: "f32[32, 128, 1024]" = torch.ops.aten.pow.Tensor_Scalar(view_23, 2.0);  view_23 = None
        mul_1015: "f32[32, 128, 1024]" = torch.ops.aten.mul.Scalar(pow_154, 3.0);  pow_154 = None
        mul_1016: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1014, mul_1015);  mul_1014 = mul_1015 = None
        add_338: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(mul_1013, mul_1016);  mul_1013 = mul_1016 = None
        mul_1017: "f32[32, 128, 1024]" = torch.ops.aten.mul.Tensor(mul_1010, 0.5);  mul_1010 = None
        add_339: "f32[32, 128, 1024]" = torch.ops.aten.add.Tensor(add_338, mul_1017);  add_338 = mul_1017 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:106 in forward, code: hidden_gelu = self.act(self.wi_0(hidden_states))
        view_1249: "f32[4096, 1024]" = torch.ops.aten.reshape.default(add_339, [4096, 1024]);  add_339 = None
        permute_1035: "f32[1024, 4096]" = torch.ops.aten.permute.default(view_1249, [1, 0])
        mm_425: "f32[1024, 512]" = torch.ops.aten.mm.default(permute_1035, view_22);  permute_1035 = view_22 = None
        permute_10: "f32[512, 1024]" = torch.ops.aten.permute.default(primals_10, [1, 0]);  primals_10 = None
        permute_1037: "f32[1024, 512]" = torch.ops.aten.permute.default(permute_10, [1, 0]);  permute_10 = None
        mm_426: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1249, permute_1037);  view_1249 = permute_1037 = None
        view_1250: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_426, [32, 128, 512]);  mm_426 = None
        add_340: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1248, view_1250);  view_1248 = view_1250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_1018: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_340, primals_9);  primals_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_10: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_9, rsqrt_1)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_1019: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_340, mul_10);  add_340 = mul_10 = None
        sum_133: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_1019, [0, 1], True);  mul_1019 = None
        view_1251: "f32[512]" = torch.ops.aten.reshape.default(sum_133, [512]);  sum_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1020: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1018, add_9)
        mul_1021: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1018, rsqrt_1);  mul_1018 = None
        sum_134: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_1020, [2], True);  mul_1020 = None
        add_341: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_337, mul_1021);  add_337 = mul_1021 = None
        pow_155: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt_1, 3);  rsqrt_1 = None
        mul_1022: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_134, -0.5);  sum_134 = None
        mul_1023: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_1022, pow_155);  mul_1022 = pow_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_139: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_1023, [32, 128, 512]);  mul_1023 = None
        div_70: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_139, 512);  expand_139 = None
        pow_156: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(add_9, 1.0);  add_9 = None
        mul_1024: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_156, 2.0);  pow_156 = None
        mul_1025: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_70, mul_1024);  div_70 = mul_1024 = None
        add_342: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_341, mul_1025);  add_341 = mul_1025 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:368 in forward, code: hidden_states = hidden_states + self.dropout(attention_output[0])
        convert_element_type_87: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt_3, torch.float32);  gt_3 = None
        mul_1026: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_87, 1.1111111111111112);  convert_element_type_87 = None
        mul_1027: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_342, mul_1026);  mul_1026 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:330 in forward, code: attn_output = self.o(attn_output)
        view_1252: "f32[4096, 512]" = torch.ops.aten.reshape.default(mul_1027, [4096, 512]);  mul_1027 = None
        permute_1039: "f32[512, 4096]" = torch.ops.aten.permute.default(view_1252, [1, 0])
        mm_427: "f32[512, 384]" = torch.ops.aten.mm.default(permute_1039, view_20);  permute_1039 = view_20 = None
        permute_9: "f32[384, 512]" = torch.ops.aten.permute.default(primals_8, [1, 0]);  primals_8 = None
        permute_1041: "f32[512, 384]" = torch.ops.aten.permute.default(permute_9, [1, 0]);  permute_9 = None
        mm_428: "f32[4096, 384]" = torch.ops.aten.mm.default(view_1252, permute_1041);  view_1252 = permute_1041 = None
        view_1253: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(mm_428, [32, 128, 384]);  mm_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:329 in forward, code: attn_output = attn_output.reshape(*input_shape, -1)
        view_1254: "f32[32, 128, 6, 64]" = torch.ops.aten.reshape.default(view_1253, [32, 128, 6, 64]);  view_1253 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:328 in forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_1043: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1254, [0, 2, 1, 3]);  view_1254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:326 in forward, code: attn_output = torch.matmul(attn_weights, value_states)
        clone_294: "f32[32, 6, 128, 64]" = torch.ops.aten.clone.default(permute_1043, memory_format = torch.contiguous_format);  permute_1043 = None
        view_1255: "f32[192, 128, 64]" = torch.ops.aten.reshape.default(clone_294, [192, 128, 64]);  clone_294 = None
        bmm_140: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(permute_1044, view_1255);  permute_1044 = None
        bmm_141: "f32[192, 128, 128]" = torch.ops.aten.bmm.default(view_1255, permute_1045);  view_1255 = permute_1045 = None
        view_1256: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_140, [32, 6, 128, 64]);  bmm_140 = None
        view_1257: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm_141, [32, 6, 128, 128]);  bmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:324 in forward, code: attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        convert_element_type_88: "f32[32, 6, 128, 128]" = torch.ops.prims.convert_element_type.default(gt_2, torch.float32);  gt_2 = None
        mul_1028: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(convert_element_type_88, 1.1111111111111112);  convert_element_type_88 = None
        mul_1029: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(view_1257, mul_1028);  view_1257 = mul_1028 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[32, 1, 128, 128]" = torch.ops.aten.expand.default(ge, [32, -1, 128, 128]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:612 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        full_default_1: "f32[]" = torch.ops.aten.full.default([], -3.4028234663852886e+38, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[32, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default, full_default_1);  expand = full_default_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        view_12: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(bmm, [32, 6, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        permute_7: "f32[6, 128, 128]" = torch.ops.aten.permute.default(embedding_1, [2, 0, 1]);  embedding_1 = None
        unsqueeze_5: "f32[1, 6, 128, 128]" = torch.ops.aten.unsqueeze.default(permute_7, 0);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        add_7: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(unsqueeze_5, where);  unsqueeze_5 = where = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        add_8: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(view_12, add_7);  view_12 = add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:323 in forward, code: attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        sub_1: "f32[32, 6, 128, 128]" = torch.ops.aten.sub.Tensor(add_8, amax);  add_8 = amax = None
        exp: "f32[32, 6, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        div_2: "f32[32, 6, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        mul_1030: "f32[32, 6, 128, 128]" = torch.ops.aten.mul.Tensor(mul_1029, div_2);  mul_1029 = None
        sum_135: "f32[32, 6, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_1030, [-1], True)
        neg_25: "f32[32, 6, 128, 128]" = torch.ops.aten.neg.default(div_2);  div_2 = None
        fma_23: "f32[32, 6, 128, 128]" = torch.ops.prims.fma.default(neg_25, sum_135, mul_1030);  neg_25 = sum_135 = mul_1030 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:320 in forward, code: scores += position_bias_masked
        view_1258: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(fma_23, [192, 128, 128]);  fma_23 = None
        view_1260: "f32[32, 6, 128, 128]" = torch.ops.aten.reshape.default(view_1258, [32, 6, 128, 128]);  view_1258 = None
        view_1261: "f32[192, 128, 128]" = torch.ops.aten.reshape.default(view_1260, [192, 128, 128])
        add_343: "f32[32, 6, 128, 128]" = torch.ops.aten.add.Tensor(add_333, view_1260);  add_333 = view_1260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:317 in forward, code: position_bias = position_bias + causal_mask
        sum_136: "f32[1, 6, 128, 128]" = torch.ops.aten.sum.dim_IntList(add_343, [0], True);  add_343 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:242 in compute_bias, code: values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        squeeze_2: "f32[6, 128, 128]" = torch.ops.aten.squeeze.dim(sum_136, 0);  sum_136 = None
        permute_1046: "f32[128, 128, 6]" = torch.ops.aten.permute.default(squeeze_2, [1, 2, 0]);  squeeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:241 in compute_bias, code: values = self.relative_attention_bias(relative_position_bucket)  # shape (query_length, key_length, num_heads)
        eq_2: "b8[128, 128]" = torch.ops.aten.eq.Scalar(add_6, -1)
        unsqueeze_22: "b8[128, 128, 1]" = torch.ops.aten.unsqueeze.default(eq_2, -1);  eq_2 = None
        where_11: "f32[128, 128, 6]" = torch.ops.aten.where.self(unsqueeze_22, full_default, permute_1046);  unsqueeze_22 = permute_1046 = None
        clone_297: "f32[128, 128, 6]" = torch.ops.aten.clone.default(where_11, memory_format = torch.contiguous_format);  where_11 = None
        index_put_2: "f32[32, 6]" = torch.ops.aten.index_put.default(full_default_16, [add_6], clone_297, True);  full_default_16 = add_6 = clone_297 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:300 in forward, code: scores = torch.matmul(query_states, key_states.transpose(3, 2))
        bmm_142: "f32[192, 64, 128]" = torch.ops.aten.bmm.default(permute_1047, view_1261);  permute_1047 = None
        bmm_143: "f32[192, 128, 64]" = torch.ops.aten.bmm.default(view_1261, permute_1048);  view_1261 = permute_1048 = None
        view_1263: "f32[32, 6, 64, 128]" = torch.ops.aten.reshape.default(bmm_142, [32, 6, 64, 128]);  bmm_142 = None
        view_1264: "f32[32, 6, 128, 64]" = torch.ops.aten.reshape.default(bmm_143, [32, 6, 128, 64]);  bmm_143 = None
        permute_1049: "f32[32, 6, 128, 64]" = torch.ops.aten.permute.default(view_1263, [0, 1, 3, 2]);  view_1263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:291 in forward, code: value_states = self.v(current_states).view(kv_shape).transpose(1, 2)
        permute_1050: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1256, [0, 2, 1, 3]);  view_1256 = None
        clone_298: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1050, memory_format = torch.contiguous_format);  permute_1050 = None
        view_1265: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_298, [32, 128, 384]);  clone_298 = None
        view_1266: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1265, [4096, 384]);  view_1265 = None
        permute_1051: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1266, [1, 0])
        mm_429: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1051, view_1);  permute_1051 = None
        permute_4: "f32[512, 384]" = torch.ops.aten.permute.default(primals_6, [1, 0]);  primals_6 = None
        permute_1053: "f32[384, 512]" = torch.ops.aten.permute.default(permute_4, [1, 0]);  permute_4 = None
        mm_430: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1266, permute_1053);  view_1266 = permute_1053 = None
        view_1267: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_430, [32, 128, 512]);  mm_430 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:290 in forward, code: key_states = self.k(current_states).view(kv_shape).transpose(1, 2)
        permute_1055: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(permute_1049, [0, 2, 1, 3]);  permute_1049 = None
        view_1268: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(permute_1055, [32, 128, 384]);  permute_1055 = None
        clone_299: "f32[32, 128, 384]" = torch.ops.aten.clone.default(view_1268, memory_format = torch.contiguous_format);  view_1268 = None
        view_1269: "f32[4096, 384]" = torch.ops.aten.reshape.default(clone_299, [4096, 384]);  clone_299 = None
        permute_1056: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1269, [1, 0])
        mm_431: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1056, view_1);  permute_1056 = None
        permute_2: "f32[512, 384]" = torch.ops.aten.permute.default(primals_5, [1, 0]);  primals_5 = None
        permute_1058: "f32[384, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        mm_432: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1269, permute_1058);  view_1269 = permute_1058 = None
        view_1270: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_432, [32, 128, 512]);  mm_432 = None
        add_344: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(view_1267, view_1270);  view_1267 = view_1270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:269 in forward, code: query_states = self.q(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_1060: "f32[32, 128, 6, 64]" = torch.ops.aten.permute.default(view_1264, [0, 2, 1, 3]);  view_1264 = None
        clone_300: "f32[32, 128, 6, 64]" = torch.ops.aten.clone.default(permute_1060, memory_format = torch.contiguous_format);  permute_1060 = None
        view_1271: "f32[32, 128, 384]" = torch.ops.aten.reshape.default(clone_300, [32, 128, 384]);  clone_300 = None
        view_1272: "f32[4096, 384]" = torch.ops.aten.reshape.default(view_1271, [4096, 384]);  view_1271 = None
        permute_1061: "f32[384, 4096]" = torch.ops.aten.permute.default(view_1272, [1, 0])
        mm_433: "f32[384, 512]" = torch.ops.aten.mm.default(permute_1061, view_1);  permute_1061 = view_1 = None
        permute: "f32[512, 384]" = torch.ops.aten.permute.default(primals_4, [1, 0]);  primals_4 = None
        permute_1063: "f32[384, 512]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        mm_434: "f32[4096, 512]" = torch.ops.aten.mm.default(view_1272, permute_1063);  view_1272 = permute_1063 = None
        view_1273: "f32[32, 128, 512]" = torch.ops.aten.reshape.default(mm_434, [32, 128, 512]);  mm_434 = None
        add_345: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_344, view_1273);  add_344 = view_1273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_1031: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_345, primals_3);  primals_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        mul: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(gt, embedding);  embedding = None
        mul_1: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul, 1.1111111111111112);  mul = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_2: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1, rsqrt)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:69 in forward, code: return self.weight * hidden_states
        mul_1032: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_345, mul_2);  add_345 = mul_2 = None
        sum_137: "f32[1, 1, 512]" = torch.ops.aten.sum.dim_IntList(mul_1032, [0, 1], True);  mul_1032 = None
        view_1274: "f32[512]" = torch.ops.aten.reshape.default(sum_137, [512]);  sum_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:63 in forward, code: hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        mul_1033: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1031, mul_1)
        mul_1034: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(mul_1031, rsqrt);  mul_1031 = None
        sum_138: "f32[32, 128, 1]" = torch.ops.aten.sum.dim_IntList(mul_1033, [2], True);  mul_1033 = None
        add_346: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_342, mul_1034);  add_342 = mul_1034 = None
        pow_157: "f32[32, 128, 1]" = torch.ops.aten.pow.Tensor_Scalar(rsqrt, 3);  rsqrt = None
        mul_1035: "f32[32, 128, 1]" = torch.ops.aten.mul.Scalar(sum_138, -0.5);  sum_138 = None
        mul_1036: "f32[32, 128, 1]" = torch.ops.aten.mul.Tensor(mul_1035, pow_157);  mul_1035 = pow_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:62 in forward, code: variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        expand_140: "f32[32, 128, 512]" = torch.ops.aten.expand.default(mul_1036, [32, 128, 512]);  mul_1036 = None
        div_71: "f32[32, 128, 512]" = torch.ops.aten.div.Scalar(expand_140, 512);  expand_140 = None
        pow_158: "f32[32, 128, 512]" = torch.ops.aten.pow.Tensor_Scalar(mul_1, 1.0);  mul_1 = None
        mul_1037: "f32[32, 128, 512]" = torch.ops.aten.mul.Scalar(pow_158, 2.0);  pow_158 = None
        mul_1038: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(div_71, mul_1037);  div_71 = mul_1037 = None
        add_347: "f32[32, 128, 512]" = torch.ops.aten.add.Tensor(add_346, mul_1038);  add_346 = mul_1038 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:732 in forward, code: hidden_states = self.dropout(inputs_embeds)
        convert_element_type_89: "f32[32, 128, 512]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_1039: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(convert_element_type_89, 1.1111111111111112);  convert_element_type_89 = None
        mul_1040: "f32[32, 128, 512]" = torch.ops.aten.mul.Tensor(add_347, mul_1039);  add_347 = mul_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/mt5/modeling_mt5.py:680 in forward, code: inputs_embeds = self.embed_tokens(input_ids)
        where_12: "f32[32, 128, 512]" = torch.ops.aten.where.self(unsqueeze_21, full_default, mul_1040);  unsqueeze_21 = full_default = mul_1040 = None
        index_put_3: "f32[250112, 512]" = torch.ops.aten.index_put.default(full_default_18, [primals_1], where_12, True);  full_default_18 = primals_1 = where_12 = None
        add_348: "f32[250112, 512]" = torch.ops.aten.add.Tensor(add_267, index_put_3);  add_267 = index_put_3 = None
        return (None, add_348, view_1274, mm_433, mm_431, mm_429, index_put_2, mm_427, view_1251, mm_425, mm_423, mm_421, view_1244, mm_419, mm_417, mm_415, mm_413, view_1221, mm_411, mm_409, mm_407, view_1214, mm_405, mm_403, mm_401, mm_399, view_1191, mm_397, mm_395, mm_393, view_1184, mm_391, mm_389, mm_387, mm_385, view_1161, mm_383, mm_381, mm_379, view_1154, mm_377, mm_375, mm_373, mm_371, view_1131, mm_369, mm_367, mm_365, view_1124, mm_363, mm_361, mm_359, mm_357, view_1101, mm_355, mm_353, mm_351, view_1094, mm_349, mm_347, mm_345, mm_343, view_1071, mm_341, mm_339, mm_337, view_1064, mm_335, mm_333, mm_331, mm_329, view_1041, mm_327, mm_325, mm_323, view_1034, None, view_1033, mm_321, mm_319, mm_317, index_put, mm_315, view_1010, mm_313, mm_311, mm_309, mm_307, view_987, mm_305, mm_303, mm_301, view_980, mm_299, mm_297, mm_295, mm_293, view_957, mm_291, mm_289, mm_287, mm_285, view_934, mm_283, mm_281, mm_279, view_927, mm_277, mm_275, mm_273, mm_271, view_904, mm_269, mm_267, mm_265, mm_263, view_881, mm_261, mm_259, mm_257, view_874, mm_255, mm_253, mm_251, mm_249, view_851, mm_247, mm_245, mm_243, mm_241, view_828, mm_239, mm_237, mm_235, view_821, mm_233, mm_231, mm_229, mm_227, view_798, mm_225, mm_223, mm_221, mm_219, view_775, mm_217, mm_215, mm_213, view_768, mm_211, mm_209, mm_207, mm_205, view_745, mm_203, mm_201, mm_199, mm_197, view_722, mm_195, mm_193, mm_191, view_715, mm_189, mm_187, mm_185, mm_183, view_692, mm_181, mm_179, mm_177, mm_175, view_669, mm_173, mm_171, mm_169, view_662, mm_167, mm_165, mm_163, mm_161, view_639, mm_159, mm_157, mm_155, mm_153, view_616, mm_151, mm_149, mm_147, view_609)

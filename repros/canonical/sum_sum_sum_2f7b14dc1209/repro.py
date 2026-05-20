"""
Standalone repro captured via capture_hook.
Label: timm_swin_base_patch4_window7_224_train
Pattern hash: 2f7b14dc1209
Shape hash: 205910e6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([128, 1000], f32), T([128, 7, 7, 1024], f32), T([128, 7, 7, 1024], f32), T([6272, 1024], f32), T([6272, 4096], f32), T([128, 49, 1024], f32), T([128, 49, 1024], f32), T([6272, 1024], f32), T([128, 32, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([6272, 3072], f32), T([128, 7, 7, 1024], f32), T([128, 7, 7, 1024], f32), T([6272, 1024], f32), T([6272, 4096], f32), T([128, 49, 1024], f32), T([128, 49, 1024], f32), T([6272, 1024], f32), T([128, 32, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([6272, 3072], f32), T([128, 7, 7, 1024], f32), T([128, 7, 7, 1024], f32), T([6272, 1024], f32), T([128, 7, 7, 2048], f32), T([128, 7, 7, 2048], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([25088, 2048], f32), T([128, 196, 512], f32), T([128, 196, 512], f32), T([25088, 512], f32), T([512, 16, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([25088, 1536], f32), T([128, 14, 14, 512], f32), T([128, 14, 14, 512], f32), T([25088, 512], f32), T([128, 14, 14, 1024], f32), T([128, 14, 14, 1024], f32), T([100352, 256], f32), T([100352, 1024], f32), T([128, 784, 256], f32), T([128, 784, 256], f32), T([100352, 256], f32), T([2048, 8, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([100352, 768], f32), T([128, 28, 28, 256], f32), T([128, 28, 28, 256], f32), T([100352, 256], f32), T([100352, 1024], f32), T([128, 784, 256], f32), T([128, 784, 256], f32), T([100352, 256], f32), T([2048, 8, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([100352, 768], f32), T([128, 28, 28, 256], f32), T([128, 28, 28, 256], f32), T([100352, 256], f32), T([128, 28, 28, 512], f32), T([128, 28, 28, 512], f32), T([401408, 128], f32), T([401408, 512], f32), T([128, 3136, 128], f32), T([128, 3136, 128], f32), T([401408, 128], f32), T([8192, 4, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([401408, 384], f32), T([128, 56, 56, 128], f32), T([128, 56, 56, 128], f32), T([401408, 128], f32), T([401408, 512], f32), T([128, 3136, 128], f32), T([128, 3136, 128], f32), T([401408, 128], f32), T([8192, 4, 49, 49], f32), T([49, 49], i64, gen=Index(49)), T([401408, 384], f32), T([128, 56, 56, 128], f32), T([128, 56, 56, 128], f32), T([128, 56, 56, 128], f32), T([128, 56, 56, 128], f32), T([128, 128, 56, 56], f32, stride=(401408, 1, 7168, 128)), S([1000]), S([1024]), S([4096]), S([1024]), S([2401, 32]), S([3072]), S([1024]), S([4096]), S([1024]), S([2401, 32]), S([3072]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([512]), S([2048]), S([512]), S([2401, 16]), S([1536]), S([256]), S([1024]), S([256]), S([2401, 8]), S([768]), S([256]), S([1024]), S([256]), S([2401, 8]), S([768]), S([128]), S([512]), S([128]), S([2401, 4]), S([384]), S([128]), S([512]), S([128]), S([2401, 4]), S([384]))"

class Repro(torch.nn.Module):
    def forward(self, tangents_1: "f32[128, 1000]", div_70: "f32[128, 7, 7, 1024]", mul_246: "f32[128, 7, 7, 1024]", view_660: "f32[6272, 1024]", view_663: "f32[6272, 4096]", view_665: "f32[128, 49, 1024]", mul_240: "f32[128, 49, 1024]", view_670: "f32[6272, 1024]", fma: "f32[128, 32, 49, 49]", primals_353: "i64[49, 49]", view_683: "f32[6272, 3072]", view_688: "f32[128, 7, 7, 1024]", mul_236: "f32[128, 7, 7, 1024]", view_690: "f32[6272, 1024]", view_693: "f32[6272, 4096]", view_695: "f32[128, 49, 1024]", mul_230: "f32[128, 49, 1024]", view_700: "f32[6272, 1024]", fma_1: "f32[128, 32, 49, 49]", primals_339: "i64[49, 49]", view_713: "f32[6272, 3072]", view_718: "f32[128, 7, 7, 1024]", mul_226: "f32[128, 7, 7, 1024]", view_719: "f32[6272, 1024]", view_720: "f32[128, 7, 7, 2048]", mul_224: "f32[128, 7, 7, 2048]", view_724: "f32[25088, 512]", view_727: "f32[25088, 2048]", view_729: "f32[128, 196, 512]", mul_218: "f32[128, 196, 512]", view_734: "f32[25088, 512]", fma_2: "f32[512, 16, 49, 49]", primals_322: "i64[49, 49]", view_749: "f32[25088, 1536]", index_71: "f32[128, 14, 14, 512]", mul_214: "f32[128, 14, 14, 512]", view_756: "f32[25088, 512]", view_759: "f32[25088, 2048]", view_761: "f32[128, 196, 512]", mul_208: "f32[128, 196, 512]", view_766: "f32[25088, 512]", fma_3: "f32[512, 16, 49, 49]", primals_307: "i64[49, 49]", view_779: "f32[25088, 1536]", view_784: "f32[128, 14, 14, 512]", mul_204: "f32[128, 14, 14, 512]", view_786: "f32[25088, 512]", view_789: "f32[25088, 2048]", view_791: "f32[128, 196, 512]", mul_198: "f32[128, 196, 512]", view_796: "f32[25088, 512]", fma_4: "f32[512, 16, 49, 49]", primals_293: "i64[49, 49]", view_811: "f32[25088, 1536]", index_75: "f32[128, 14, 14, 512]", mul_194: "f32[128, 14, 14, 512]", view_818: "f32[25088, 512]", view_821: "f32[25088, 2048]", view_823: "f32[128, 196, 512]", mul_188: "f32[128, 196, 512]", view_828: "f32[25088, 512]", fma_5: "f32[512, 16, 49, 49]", primals_278: "i64[49, 49]", view_841: "f32[25088, 1536]", view_846: "f32[128, 14, 14, 512]", mul_184: "f32[128, 14, 14, 512]", view_848: "f32[25088, 512]", view_851: "f32[25088, 2048]", view_853: "f32[128, 196, 512]", mul_178: "f32[128, 196, 512]", view_858: "f32[25088, 512]", fma_6: "f32[512, 16, 49, 49]", primals_264: "i64[49, 49]", view_873: "f32[25088, 1536]", index_79: "f32[128, 14, 14, 512]", mul_174: "f32[128, 14, 14, 512]", view_880: "f32[25088, 512]", view_883: "f32[25088, 2048]", view_885: "f32[128, 196, 512]", mul_168: "f32[128, 196, 512]", view_890: "f32[25088, 512]", fma_7: "f32[512, 16, 49, 49]", primals_249: "i64[49, 49]", view_903: "f32[25088, 1536]", view_908: "f32[128, 14, 14, 512]", mul_164: "f32[128, 14, 14, 512]", view_910: "f32[25088, 512]", view_913: "f32[25088, 2048]", view_915: "f32[128, 196, 512]", mul_158: "f32[128, 196, 512]", view_920: "f32[25088, 512]", fma_8: "f32[512, 16, 49, 49]", primals_235: "i64[49, 49]", view_935: "f32[25088, 1536]", index_83: "f32[128, 14, 14, 512]", mul_154: "f32[128, 14, 14, 512]", view_942: "f32[25088, 512]", view_945: "f32[25088, 2048]", view_947: "f32[128, 196, 512]", mul_148: "f32[128, 196, 512]", view_952: "f32[25088, 512]", fma_9: "f32[512, 16, 49, 49]", primals_220: "i64[49, 49]", view_965: "f32[25088, 1536]", view_970: "f32[128, 14, 14, 512]", mul_144: "f32[128, 14, 14, 512]", view_972: "f32[25088, 512]", view_975: "f32[25088, 2048]", view_977: "f32[128, 196, 512]", mul_138: "f32[128, 196, 512]", view_982: "f32[25088, 512]", fma_10: "f32[512, 16, 49, 49]", primals_206: "i64[49, 49]", view_997: "f32[25088, 1536]", index_87: "f32[128, 14, 14, 512]", mul_134: "f32[128, 14, 14, 512]", view_1004: "f32[25088, 512]", view_1007: "f32[25088, 2048]", view_1009: "f32[128, 196, 512]", mul_128: "f32[128, 196, 512]", view_1014: "f32[25088, 512]", fma_11: "f32[512, 16, 49, 49]", primals_191: "i64[49, 49]", view_1027: "f32[25088, 1536]", view_1032: "f32[128, 14, 14, 512]", mul_124: "f32[128, 14, 14, 512]", view_1034: "f32[25088, 512]", view_1037: "f32[25088, 2048]", view_1039: "f32[128, 196, 512]", mul_118: "f32[128, 196, 512]", view_1044: "f32[25088, 512]", fma_12: "f32[512, 16, 49, 49]", primals_177: "i64[49, 49]", view_1059: "f32[25088, 1536]", index_91: "f32[128, 14, 14, 512]", mul_114: "f32[128, 14, 14, 512]", view_1066: "f32[25088, 512]", view_1069: "f32[25088, 2048]", view_1071: "f32[128, 196, 512]", mul_108: "f32[128, 196, 512]", view_1076: "f32[25088, 512]", fma_13: "f32[512, 16, 49, 49]", primals_162: "i64[49, 49]", view_1089: "f32[25088, 1536]", view_1094: "f32[128, 14, 14, 512]", mul_104: "f32[128, 14, 14, 512]", view_1096: "f32[25088, 512]", view_1099: "f32[25088, 2048]", view_1101: "f32[128, 196, 512]", mul_98: "f32[128, 196, 512]", view_1106: "f32[25088, 512]", fma_14: "f32[512, 16, 49, 49]", primals_148: "i64[49, 49]", view_1121: "f32[25088, 1536]", index_95: "f32[128, 14, 14, 512]", mul_94: "f32[128, 14, 14, 512]", view_1128: "f32[25088, 512]", view_1131: "f32[25088, 2048]", view_1133: "f32[128, 196, 512]", mul_88: "f32[128, 196, 512]", view_1138: "f32[25088, 512]", fma_15: "f32[512, 16, 49, 49]", primals_133: "i64[49, 49]", view_1151: "f32[25088, 1536]", view_1156: "f32[128, 14, 14, 512]", mul_84: "f32[128, 14, 14, 512]", view_1158: "f32[25088, 512]", view_1161: "f32[25088, 2048]", view_1163: "f32[128, 196, 512]", mul_78: "f32[128, 196, 512]", view_1168: "f32[25088, 512]", fma_16: "f32[512, 16, 49, 49]", primals_119: "i64[49, 49]", view_1183: "f32[25088, 1536]", index_99: "f32[128, 14, 14, 512]", mul_74: "f32[128, 14, 14, 512]", view_1190: "f32[25088, 512]", view_1193: "f32[25088, 2048]", view_1195: "f32[128, 196, 512]", mul_68: "f32[128, 196, 512]", view_1200: "f32[25088, 512]", fma_17: "f32[512, 16, 49, 49]", primals_104: "i64[49, 49]", view_1213: "f32[25088, 1536]", view_1218: "f32[128, 14, 14, 512]", mul_64: "f32[128, 14, 14, 512]", view_1220: "f32[25088, 512]", view_1223: "f32[25088, 2048]", view_1225: "f32[128, 196, 512]", mul_58: "f32[128, 196, 512]", view_1230: "f32[25088, 512]", fma_18: "f32[512, 16, 49, 49]", primals_90: "i64[49, 49]", view_1245: "f32[25088, 1536]", index_103: "f32[128, 14, 14, 512]", mul_54: "f32[128, 14, 14, 512]", view_1252: "f32[25088, 512]", view_1255: "f32[25088, 2048]", view_1257: "f32[128, 196, 512]", mul_48: "f32[128, 196, 512]", view_1262: "f32[25088, 512]", fma_19: "f32[512, 16, 49, 49]", primals_75: "i64[49, 49]", view_1275: "f32[25088, 1536]", view_1280: "f32[128, 14, 14, 512]", mul_44: "f32[128, 14, 14, 512]", view_1281: "f32[25088, 512]", view_1282: "f32[128, 14, 14, 1024]", mul_42: "f32[128, 14, 14, 1024]", view_1286: "f32[100352, 256]", view_1289: "f32[100352, 1024]", view_1291: "f32[128, 784, 256]", mul_36: "f32[128, 784, 256]", view_1296: "f32[100352, 256]", fma_20: "f32[2048, 8, 49, 49]", primals_58: "i64[49, 49]", view_1311: "f32[100352, 768]", index_107: "f32[128, 28, 28, 256]", mul_32: "f32[128, 28, 28, 256]", view_1318: "f32[100352, 256]", view_1321: "f32[100352, 1024]", view_1323: "f32[128, 784, 256]", mul_26: "f32[128, 784, 256]", view_1328: "f32[100352, 256]", fma_21: "f32[2048, 8, 49, 49]", primals_43: "i64[49, 49]", view_1341: "f32[100352, 768]", view_1346: "f32[128, 28, 28, 256]", mul_22: "f32[128, 28, 28, 256]", view_1347: "f32[100352, 256]", view_1348: "f32[128, 28, 28, 512]", mul_20: "f32[128, 28, 28, 512]", view_1352: "f32[401408, 128]", view_1355: "f32[401408, 512]", view_1357: "f32[128, 3136, 128]", mul_14: "f32[128, 3136, 128]", view_1362: "f32[401408, 128]", fma_22: "f32[8192, 4, 49, 49]", primals_26: "i64[49, 49]", view_1377: "f32[401408, 384]", index_111: "f32[128, 56, 56, 128]", mul_10: "f32[128, 56, 56, 128]", view_1384: "f32[401408, 128]", view_1387: "f32[401408, 512]", view_1389: "f32[128, 3136, 128]", mul_5: "f32[128, 3136, 128]", view_1394: "f32[401408, 128]", fma_23: "f32[8192, 4, 49, 49]", primals_11: "i64[49, 49]", view_1407: "f32[401408, 384]", view_1412: "f32[128, 56, 56, 128]", mul_2: "f32[128, 56, 56, 128]", add_396: "f32[128, 56, 56, 128]", mul: "f32[128, 56, 56, 128]", permute_891: "f32[128, 128, 56, 56]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/classifier.py:141 in forward, code: x = self.fc(x)
        permute_default: "f32[1000, 128]" = torch.ops.aten.permute.default(tangents_1, [1, 0])
        sum_dim_int_list: "f32[1, 1000]" = torch.ops.aten.sum.dim_IntList(tangents_1, [0], True);  tangents_1 = None
        reshape_default: "f32[1000]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:981 in forward_features, code: x = self.norm(x)
        mul_tensor: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(div_70, mul_246);  mul_246 = None
        sum_dim_int_list_1: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1, 2]);  mul_tensor = None
        sum_dim_int_list_2: "f32[1024]" = torch.ops.aten.sum.dim_IntList(div_70, [0, 1, 2]);  div_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_1: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_660, [1, 0])
        sum_dim_int_list_3: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_660, [0], True);  view_660 = None
        reshape_default_1: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_2: "f32[4096, 6272]" = torch.ops.aten.permute.default(view_663, [1, 0])
        sum_dim_int_list_4: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_663, [0], True);  view_663 = None
        reshape_default_2: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_4, _shape_param_2);  sum_dim_int_list_4 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_1: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_665, mul_240);  mul_240 = None
        sum_dim_int_list_5: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_6: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_665, [0, 1]);  view_665 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_3: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_670, [1, 0])
        sum_dim_int_list_7: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_670, [0], True);  view_670 = None
        reshape_default_3: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_8: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma, [0], True);  fma = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_8, 0);  sum_dim_int_list_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_4: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_dim, [1, 2, 0]);  squeeze_dim = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_4: "f32[2401, 32]" = torch.ops.aten.reshape.default(permute_default_4, _shape_param_4);  permute_default_4 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default: "f32[169, 32]" = torch.ops.aten.full.default([169, 32], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_5: "i64[2401]" = torch.ops.aten.reshape.default(primals_353, [-1]);  primals_353 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default, [reshape_default_5], reshape_default_4, True);  reshape_default_5 = reshape_default_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_5: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_683, [1, 0])
        sum_dim_int_list_9: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_683, [0], True);  view_683 = None
        reshape_default_6: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_9, _shape_param_5);  sum_dim_int_list_9 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_2: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_688, mul_236);  mul_236 = None
        sum_dim_int_list_10: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1, 2]);  mul_tensor_2 = None
        sum_dim_int_list_11: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_688, [0, 1, 2]);  view_688 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_6: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_690, [1, 0])
        sum_dim_int_list_12: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_690, [0], True);  view_690 = None
        reshape_default_7: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_7: "f32[4096, 6272]" = torch.ops.aten.permute.default(view_693, [1, 0])
        sum_dim_int_list_13: "f32[1, 4096]" = torch.ops.aten.sum.dim_IntList(view_693, [0], True);  view_693 = None
        reshape_default_8: "f32[4096]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_3: "f32[128, 49, 1024]" = torch.ops.aten.mul.Tensor(view_695, mul_230);  mul_230 = None
        sum_dim_int_list_14: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_695, [0, 1]);  view_695 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_8: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_700, [1, 0])
        sum_dim_int_list_16: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_700, [0], True);  view_700 = None
        reshape_default_9: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_17: "f32[1, 32, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_1, [0], True);  fma_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_1: "f32[32, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_17, 0);  sum_dim_int_list_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_9: "f32[49, 49, 32]" = torch.ops.aten.permute.default(squeeze_dim_1, [1, 2, 0]);  squeeze_dim_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_10: "f32[2401, 32]" = torch.ops.aten.reshape.default(permute_default_9, _shape_param_9);  permute_default_9 = _shape_param_9 = None
        reshape_default_11: "i64[2401]" = torch.ops.aten.reshape.default(primals_339, [-1]);  primals_339 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_1: "f32[169, 32]" = torch.ops.aten.index_put.default(full_default, [reshape_default_11], reshape_default_10, True);  full_default = reshape_default_11 = reshape_default_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_10: "f32[3072, 6272]" = torch.ops.aten.permute.default(view_713, [1, 0])
        sum_dim_int_list_18: "f32[1, 3072]" = torch.ops.aten.sum.dim_IntList(view_713, [0], True);  view_713 = None
        reshape_default_12: "f32[3072]" = torch.ops.aten.reshape.default(sum_dim_int_list_18, _shape_param_10);  sum_dim_int_list_18 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_4: "f32[128, 7, 7, 1024]" = torch.ops.aten.mul.Tensor(view_718, mul_226);  mul_226 = None
        sum_dim_int_list_19: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1, 2]);  mul_tensor_4 = None
        sum_dim_int_list_20: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_718, [0, 1, 2]);  view_718 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_default_11: "f32[1024, 6272]" = torch.ops.aten.permute.default(view_719, [1, 0]);  view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_tensor_5: "f32[128, 7, 7, 2048]" = torch.ops.aten.mul.Tensor(view_720, mul_224);  mul_224 = None
        sum_dim_int_list_21: "f32[2048]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1, 2]);  mul_tensor_5 = None
        sum_dim_int_list_22: "f32[2048]" = torch.ops.aten.sum.dim_IntList(view_720, [0, 1, 2]);  view_720 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_12: "f32[512, 25088]" = torch.ops.aten.permute.default(view_724, [1, 0])
        sum_dim_int_list_23: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
        reshape_default_13: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_11);  sum_dim_int_list_23 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_13: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_727, [1, 0])
        sum_dim_int_list_24: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_727, [0], True);  view_727 = None
        reshape_default_14: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_24, _shape_param_12);  sum_dim_int_list_24 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_6: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_729, mul_218);  mul_218 = None
        sum_dim_int_list_25: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_26: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_729, [0, 1]);  view_729 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_14: "f32[512, 25088]" = torch.ops.aten.permute.default(view_734, [1, 0])
        sum_dim_int_list_27: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        reshape_default_15: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_13);  sum_dim_int_list_27 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_28: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_2, [0], True);  fma_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_2: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_28, 0);  sum_dim_int_list_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_15: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_2, [1, 2, 0]);  squeeze_dim_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_16: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_15, _shape_param_14);  permute_default_15 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_1: "f32[169, 16]" = torch.ops.aten.full.default([169, 16], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_17: "i64[2401]" = torch.ops.aten.reshape.default(primals_322, [-1]);  primals_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_2: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_17], reshape_default_16, True);  reshape_default_17 = reshape_default_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_16: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_749, [1, 0])
        sum_dim_int_list_29: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_749, [0], True);  view_749 = None
        reshape_default_18: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_29, _shape_param_15);  sum_dim_int_list_29 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_7: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_71, mul_214);  mul_214 = None
        sum_dim_int_list_30: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1, 2]);  mul_tensor_7 = None
        sum_dim_int_list_31: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_71, [0, 1, 2]);  index_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_17: "f32[512, 25088]" = torch.ops.aten.permute.default(view_756, [1, 0])
        sum_dim_int_list_32: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_756, [0], True);  view_756 = None
        reshape_default_19: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_16);  sum_dim_int_list_32 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_18: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_759, [1, 0])
        sum_dim_int_list_33: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_759, [0], True);  view_759 = None
        reshape_default_20: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_17);  sum_dim_int_list_33 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_8: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_761, mul_208);  mul_208 = None
        sum_dim_int_list_34: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_35: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_761, [0, 1]);  view_761 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_19: "f32[512, 25088]" = torch.ops.aten.permute.default(view_766, [1, 0])
        sum_dim_int_list_36: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_766, [0], True);  view_766 = None
        reshape_default_21: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_18);  sum_dim_int_list_36 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_37: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_3, [0], True);  fma_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_3: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_37, 0);  sum_dim_int_list_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_20: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_3, [1, 2, 0]);  squeeze_dim_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_22: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_20, _shape_param_19);  permute_default_20 = _shape_param_19 = None
        reshape_default_23: "i64[2401]" = torch.ops.aten.reshape.default(primals_307, [-1]);  primals_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_3: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_23], reshape_default_22, True);  reshape_default_23 = reshape_default_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_21: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_779, [1, 0])
        sum_dim_int_list_38: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_779, [0], True);  view_779 = None
        reshape_default_24: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_38, _shape_param_20);  sum_dim_int_list_38 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_9: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_784, mul_204);  mul_204 = None
        sum_dim_int_list_39: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1, 2]);  mul_tensor_9 = None
        sum_dim_int_list_40: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_784, [0, 1, 2]);  view_784 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_22: "f32[512, 25088]" = torch.ops.aten.permute.default(view_786, [1, 0])
        sum_dim_int_list_41: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_786, [0], True);  view_786 = None
        reshape_default_25: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_21);  sum_dim_int_list_41 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_23: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_789, [1, 0])
        sum_dim_int_list_42: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_789, [0], True);  view_789 = None
        reshape_default_26: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_22);  sum_dim_int_list_42 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_10: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_791, mul_198);  mul_198 = None
        sum_dim_int_list_43: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_44: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_791, [0, 1]);  view_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_24: "f32[512, 25088]" = torch.ops.aten.permute.default(view_796, [1, 0])
        sum_dim_int_list_45: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        reshape_default_27: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_45, _shape_param_23);  sum_dim_int_list_45 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_46: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_4, [0], True);  fma_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_4: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_46, 0);  sum_dim_int_list_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_25: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_4, [1, 2, 0]);  squeeze_dim_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_28: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_25, _shape_param_24);  permute_default_25 = _shape_param_24 = None
        reshape_default_29: "i64[2401]" = torch.ops.aten.reshape.default(primals_293, [-1]);  primals_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_4: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_29], reshape_default_28, True);  reshape_default_29 = reshape_default_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_26: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_811, [1, 0])
        sum_dim_int_list_47: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_811, [0], True);  view_811 = None
        reshape_default_30: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_25);  sum_dim_int_list_47 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_11: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_75, mul_194);  mul_194 = None
        sum_dim_int_list_48: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1, 2]);  mul_tensor_11 = None
        sum_dim_int_list_49: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_75, [0, 1, 2]);  index_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_27: "f32[512, 25088]" = torch.ops.aten.permute.default(view_818, [1, 0])
        sum_dim_int_list_50: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        reshape_default_31: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_26);  sum_dim_int_list_50 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_28: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_821, [1, 0])
        sum_dim_int_list_51: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        reshape_default_32: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_27);  sum_dim_int_list_51 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_12: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_823, mul_188);  mul_188 = None
        sum_dim_int_list_52: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_53: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_823, [0, 1]);  view_823 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_29: "f32[512, 25088]" = torch.ops.aten.permute.default(view_828, [1, 0])
        sum_dim_int_list_54: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_828, [0], True);  view_828 = None
        reshape_default_33: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_54, _shape_param_28);  sum_dim_int_list_54 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_55: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_5, [0], True);  fma_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_5: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_55, 0);  sum_dim_int_list_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_30: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_5, [1, 2, 0]);  squeeze_dim_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_34: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_30, _shape_param_29);  permute_default_30 = _shape_param_29 = None
        reshape_default_35: "i64[2401]" = torch.ops.aten.reshape.default(primals_278, [-1]);  primals_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_5: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_35], reshape_default_34, True);  reshape_default_35 = reshape_default_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_31: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_841, [1, 0])
        sum_dim_int_list_56: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
        reshape_default_36: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_30);  sum_dim_int_list_56 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_13: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_846, mul_184);  mul_184 = None
        sum_dim_int_list_57: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1, 2]);  mul_tensor_13 = None
        sum_dim_int_list_58: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_846, [0, 1, 2]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_32: "f32[512, 25088]" = torch.ops.aten.permute.default(view_848, [1, 0])
        sum_dim_int_list_59: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_848, [0], True);  view_848 = None
        reshape_default_37: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_59, _shape_param_31);  sum_dim_int_list_59 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_33: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_851, [1, 0])
        sum_dim_int_list_60: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_851, [0], True);  view_851 = None
        reshape_default_38: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_32);  sum_dim_int_list_60 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_14: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_853, mul_178);  mul_178 = None
        sum_dim_int_list_61: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_62: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_853, [0, 1]);  view_853 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_34: "f32[512, 25088]" = torch.ops.aten.permute.default(view_858, [1, 0])
        sum_dim_int_list_63: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_858, [0], True);  view_858 = None
        reshape_default_39: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_33);  sum_dim_int_list_63 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_64: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_6, [0], True);  fma_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_6: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_64, 0);  sum_dim_int_list_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_35: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_6, [1, 2, 0]);  squeeze_dim_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_40: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_35, _shape_param_34);  permute_default_35 = _shape_param_34 = None
        reshape_default_41: "i64[2401]" = torch.ops.aten.reshape.default(primals_264, [-1]);  primals_264 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_6: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_41], reshape_default_40, True);  reshape_default_41 = reshape_default_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_36: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_873, [1, 0])
        sum_dim_int_list_65: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_873, [0], True);  view_873 = None
        reshape_default_42: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_65, _shape_param_35);  sum_dim_int_list_65 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_15: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_79, mul_174);  mul_174 = None
        sum_dim_int_list_66: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1, 2]);  mul_tensor_15 = None
        sum_dim_int_list_67: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_79, [0, 1, 2]);  index_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_37: "f32[512, 25088]" = torch.ops.aten.permute.default(view_880, [1, 0])
        sum_dim_int_list_68: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        reshape_default_43: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_68, _shape_param_36);  sum_dim_int_list_68 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_38: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_883, [1, 0])
        sum_dim_int_list_69: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        reshape_default_44: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_69, _shape_param_37);  sum_dim_int_list_69 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_16: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_885, mul_168);  mul_168 = None
        sum_dim_int_list_70: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_71: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_885, [0, 1]);  view_885 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_39: "f32[512, 25088]" = torch.ops.aten.permute.default(view_890, [1, 0])
        sum_dim_int_list_72: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_890, [0], True);  view_890 = None
        reshape_default_45: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_38);  sum_dim_int_list_72 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_73: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_7, [0], True);  fma_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_7: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_73, 0);  sum_dim_int_list_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_40: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_7, [1, 2, 0]);  squeeze_dim_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_46: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_40, _shape_param_39);  permute_default_40 = _shape_param_39 = None
        reshape_default_47: "i64[2401]" = torch.ops.aten.reshape.default(primals_249, [-1]);  primals_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_7: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_47], reshape_default_46, True);  reshape_default_47 = reshape_default_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_41: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_903, [1, 0])
        sum_dim_int_list_74: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_903, [0], True);  view_903 = None
        reshape_default_48: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_74, _shape_param_40);  sum_dim_int_list_74 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_17: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_908, mul_164);  mul_164 = None
        sum_dim_int_list_75: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1, 2]);  mul_tensor_17 = None
        sum_dim_int_list_76: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_908, [0, 1, 2]);  view_908 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_42: "f32[512, 25088]" = torch.ops.aten.permute.default(view_910, [1, 0])
        sum_dim_int_list_77: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_910, [0], True);  view_910 = None
        reshape_default_49: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_41);  sum_dim_int_list_77 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_43: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_913, [1, 0])
        sum_dim_int_list_78: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_913, [0], True);  view_913 = None
        reshape_default_50: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_78, _shape_param_42);  sum_dim_int_list_78 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_18: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_915, mul_158);  mul_158 = None
        sum_dim_int_list_79: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_80: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_915, [0, 1]);  view_915 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_44: "f32[512, 25088]" = torch.ops.aten.permute.default(view_920, [1, 0])
        sum_dim_int_list_81: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
        reshape_default_51: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_43);  sum_dim_int_list_81 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_82: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_8, [0], True);  fma_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_8: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_82, 0);  sum_dim_int_list_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_45: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_8, [1, 2, 0]);  squeeze_dim_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_52: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_45, _shape_param_44);  permute_default_45 = _shape_param_44 = None
        reshape_default_53: "i64[2401]" = torch.ops.aten.reshape.default(primals_235, [-1]);  primals_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_8: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_53], reshape_default_52, True);  reshape_default_53 = reshape_default_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_46: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_935, [1, 0])
        sum_dim_int_list_83: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_935, [0], True);  view_935 = None
        reshape_default_54: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_45);  sum_dim_int_list_83 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_19: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_83, mul_154);  mul_154 = None
        sum_dim_int_list_84: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1, 2]);  mul_tensor_19 = None
        sum_dim_int_list_85: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_83, [0, 1, 2]);  index_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_47: "f32[512, 25088]" = torch.ops.aten.permute.default(view_942, [1, 0])
        sum_dim_int_list_86: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_942, [0], True);  view_942 = None
        reshape_default_55: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_46);  sum_dim_int_list_86 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_48: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_945, [1, 0])
        sum_dim_int_list_87: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_945, [0], True);  view_945 = None
        reshape_default_56: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_47);  sum_dim_int_list_87 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_20: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_947, mul_148);  mul_148 = None
        sum_dim_int_list_88: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_89: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_947, [0, 1]);  view_947 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_49: "f32[512, 25088]" = torch.ops.aten.permute.default(view_952, [1, 0])
        sum_dim_int_list_90: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_952, [0], True);  view_952 = None
        reshape_default_57: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_48);  sum_dim_int_list_90 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_91: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_9, [0], True);  fma_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_9: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_91, 0);  sum_dim_int_list_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_50: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_9, [1, 2, 0]);  squeeze_dim_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_58: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_50, _shape_param_49);  permute_default_50 = _shape_param_49 = None
        reshape_default_59: "i64[2401]" = torch.ops.aten.reshape.default(primals_220, [-1]);  primals_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_9: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_59], reshape_default_58, True);  reshape_default_59 = reshape_default_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_51: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_965, [1, 0])
        sum_dim_int_list_92: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_965, [0], True);  view_965 = None
        reshape_default_60: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_50);  sum_dim_int_list_92 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_21: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_970, mul_144);  mul_144 = None
        sum_dim_int_list_93: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1, 2]);  mul_tensor_21 = None
        sum_dim_int_list_94: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_970, [0, 1, 2]);  view_970 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_52: "f32[512, 25088]" = torch.ops.aten.permute.default(view_972, [1, 0])
        sum_dim_int_list_95: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_972, [0], True);  view_972 = None
        reshape_default_61: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_95, _shape_param_51);  sum_dim_int_list_95 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_53: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_975, [1, 0])
        sum_dim_int_list_96: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_975, [0], True);  view_975 = None
        reshape_default_62: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_52);  sum_dim_int_list_96 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_22: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_977, mul_138);  mul_138 = None
        sum_dim_int_list_97: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_98: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_977, [0, 1]);  view_977 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_54: "f32[512, 25088]" = torch.ops.aten.permute.default(view_982, [1, 0])
        sum_dim_int_list_99: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_982, [0], True);  view_982 = None
        reshape_default_63: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_99, _shape_param_53);  sum_dim_int_list_99 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_100: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_10, [0], True);  fma_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_10: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_100, 0);  sum_dim_int_list_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_55: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_10, [1, 2, 0]);  squeeze_dim_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_64: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_55, _shape_param_54);  permute_default_55 = _shape_param_54 = None
        reshape_default_65: "i64[2401]" = torch.ops.aten.reshape.default(primals_206, [-1]);  primals_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_10: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_65], reshape_default_64, True);  reshape_default_65 = reshape_default_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_56: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_997, [1, 0])
        sum_dim_int_list_101: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_997, [0], True);  view_997 = None
        reshape_default_66: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_55);  sum_dim_int_list_101 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_23: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_87, mul_134);  mul_134 = None
        sum_dim_int_list_102: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1, 2]);  mul_tensor_23 = None
        sum_dim_int_list_103: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_87, [0, 1, 2]);  index_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_57: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1004, [1, 0])
        sum_dim_int_list_104: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
        reshape_default_67: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_104, _shape_param_56);  sum_dim_int_list_104 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_58: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1007, [1, 0])
        sum_dim_int_list_105: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1007, [0], True);  view_1007 = None
        reshape_default_68: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_105, _shape_param_57);  sum_dim_int_list_105 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_24: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1009, mul_128);  mul_128 = None
        sum_dim_int_list_106: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_107: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1009, [0, 1]);  view_1009 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_59: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        sum_dim_int_list_108: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        reshape_default_69: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_108, _shape_param_58);  sum_dim_int_list_108 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_109: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_11, [0], True);  fma_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_11: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_109, 0);  sum_dim_int_list_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_60: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_11, [1, 2, 0]);  squeeze_dim_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_70: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_60, _shape_param_59);  permute_default_60 = _shape_param_59 = None
        reshape_default_71: "i64[2401]" = torch.ops.aten.reshape.default(primals_191, [-1]);  primals_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_11: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_71], reshape_default_70, True);  reshape_default_71 = reshape_default_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_61: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1027, [1, 0])
        sum_dim_int_list_110: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1027, [0], True);  view_1027 = None
        reshape_default_72: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_60);  sum_dim_int_list_110 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_25: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1032, mul_124);  mul_124 = None
        sum_dim_int_list_111: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1, 2]);  mul_tensor_25 = None
        sum_dim_int_list_112: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1032, [0, 1, 2]);  view_1032 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_62: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1034, [1, 0])
        sum_dim_int_list_113: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1034, [0], True);  view_1034 = None
        reshape_default_73: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_61);  sum_dim_int_list_113 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_63: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        sum_dim_int_list_114: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
        reshape_default_74: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_114, _shape_param_62);  sum_dim_int_list_114 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_26: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1039, mul_118);  mul_118 = None
        sum_dim_int_list_115: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_116: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1039, [0, 1]);  view_1039 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_64: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1044, [1, 0])
        sum_dim_int_list_117: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1044, [0], True);  view_1044 = None
        reshape_default_75: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_63);  sum_dim_int_list_117 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_118: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_12, [0], True);  fma_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_12: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_118, 0);  sum_dim_int_list_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_65: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_12, [1, 2, 0]);  squeeze_dim_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_76: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_65, _shape_param_64);  permute_default_65 = _shape_param_64 = None
        reshape_default_77: "i64[2401]" = torch.ops.aten.reshape.default(primals_177, [-1]);  primals_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_12: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_77], reshape_default_76, True);  reshape_default_77 = reshape_default_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_66: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1059, [1, 0])
        sum_dim_int_list_119: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1059, [0], True);  view_1059 = None
        reshape_default_78: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_119, _shape_param_65);  sum_dim_int_list_119 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_27: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_91, mul_114);  mul_114 = None
        sum_dim_int_list_120: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1, 2]);  mul_tensor_27 = None
        sum_dim_int_list_121: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_91, [0, 1, 2]);  index_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_67: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1066, [1, 0])
        sum_dim_int_list_122: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1066, [0], True);  view_1066 = None
        reshape_default_79: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_66);  sum_dim_int_list_122 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_68: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1069, [1, 0])
        sum_dim_int_list_123: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1069, [0], True);  view_1069 = None
        reshape_default_80: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_67);  sum_dim_int_list_123 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_28: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1071, mul_108);  mul_108 = None
        sum_dim_int_list_124: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_125: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1071, [0, 1]);  view_1071 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_69: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        sum_dim_int_list_126: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        reshape_default_81: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_126, _shape_param_68);  sum_dim_int_list_126 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_127: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_13, [0], True);  fma_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_13: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_127, 0);  sum_dim_int_list_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_70: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_13, [1, 2, 0]);  squeeze_dim_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_82: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_70, _shape_param_69);  permute_default_70 = _shape_param_69 = None
        reshape_default_83: "i64[2401]" = torch.ops.aten.reshape.default(primals_162, [-1]);  primals_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_13: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_83], reshape_default_82, True);  reshape_default_83 = reshape_default_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_71: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1089, [1, 0])
        sum_dim_int_list_128: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1089, [0], True);  view_1089 = None
        reshape_default_84: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_128, _shape_param_70);  sum_dim_int_list_128 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_29: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1094, mul_104);  mul_104 = None
        sum_dim_int_list_129: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1, 2]);  mul_tensor_29 = None
        sum_dim_int_list_130: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1094, [0, 1, 2]);  view_1094 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_72: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1096, [1, 0])
        sum_dim_int_list_131: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1096, [0], True);  view_1096 = None
        reshape_default_85: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_71);  sum_dim_int_list_131 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_73: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1099, [1, 0])
        sum_dim_int_list_132: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1099, [0], True);  view_1099 = None
        reshape_default_86: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_72);  sum_dim_int_list_132 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_30: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1101, mul_98);  mul_98 = None
        sum_dim_int_list_133: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_134: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1101, [0, 1]);  view_1101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_74: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1106, [1, 0])
        sum_dim_int_list_135: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1106, [0], True);  view_1106 = None
        reshape_default_87: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_135, _shape_param_73);  sum_dim_int_list_135 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_136: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_14, [0], True);  fma_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_14: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_136, 0);  sum_dim_int_list_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_75: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_14, [1, 2, 0]);  squeeze_dim_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_88: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_75, _shape_param_74);  permute_default_75 = _shape_param_74 = None
        reshape_default_89: "i64[2401]" = torch.ops.aten.reshape.default(primals_148, [-1]);  primals_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_14: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_89], reshape_default_88, True);  reshape_default_89 = reshape_default_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_76: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1121, [1, 0])
        sum_dim_int_list_137: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
        reshape_default_90: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_137, _shape_param_75);  sum_dim_int_list_137 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_31: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_95, mul_94);  mul_94 = None
        sum_dim_int_list_138: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1, 2]);  mul_tensor_31 = None
        sum_dim_int_list_139: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_95, [0, 1, 2]);  index_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_77: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1128, [1, 0])
        sum_dim_int_list_140: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1128, [0], True);  view_1128 = None
        reshape_default_91: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_140, _shape_param_76);  sum_dim_int_list_140 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_78: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1131, [1, 0])
        sum_dim_int_list_141: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1131, [0], True);  view_1131 = None
        reshape_default_92: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_141, _shape_param_77);  sum_dim_int_list_141 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_32: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1133, mul_88);  mul_88 = None
        sum_dim_int_list_142: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_143: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1133, [0, 1]);  view_1133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_79: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1138, [1, 0])
        sum_dim_int_list_144: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1138, [0], True);  view_1138 = None
        reshape_default_93: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_144, _shape_param_78);  sum_dim_int_list_144 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_145: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_15, [0], True);  fma_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_15: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_145, 0);  sum_dim_int_list_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_80: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_15, [1, 2, 0]);  squeeze_dim_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_94: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_80, _shape_param_79);  permute_default_80 = _shape_param_79 = None
        reshape_default_95: "i64[2401]" = torch.ops.aten.reshape.default(primals_133, [-1]);  primals_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_15: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_95], reshape_default_94, True);  reshape_default_95 = reshape_default_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_81: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1151, [1, 0])
        sum_dim_int_list_146: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1151, [0], True);  view_1151 = None
        reshape_default_96: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_146, _shape_param_80);  sum_dim_int_list_146 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_33: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1156, mul_84);  mul_84 = None
        sum_dim_int_list_147: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1, 2]);  mul_tensor_33 = None
        sum_dim_int_list_148: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1156, [0, 1, 2]);  view_1156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_82: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1158, [1, 0])
        sum_dim_int_list_149: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1158, [0], True);  view_1158 = None
        reshape_default_97: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_149, _shape_param_81);  sum_dim_int_list_149 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_83: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1161, [1, 0])
        sum_dim_int_list_150: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1161, [0], True);  view_1161 = None
        reshape_default_98: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_150, _shape_param_82);  sum_dim_int_list_150 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_34: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1163, mul_78);  mul_78 = None
        sum_dim_int_list_151: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_152: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1163, [0, 1]);  view_1163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_84: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1168, [1, 0])
        sum_dim_int_list_153: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1168, [0], True);  view_1168 = None
        reshape_default_99: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_153, _shape_param_83);  sum_dim_int_list_153 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_154: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_16, [0], True);  fma_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_16: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_154, 0);  sum_dim_int_list_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_85: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_16, [1, 2, 0]);  squeeze_dim_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_100: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_85, _shape_param_84);  permute_default_85 = _shape_param_84 = None
        reshape_default_101: "i64[2401]" = torch.ops.aten.reshape.default(primals_119, [-1]);  primals_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_16: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_101], reshape_default_100, True);  reshape_default_101 = reshape_default_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_86: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1183, [1, 0])
        sum_dim_int_list_155: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1183, [0], True);  view_1183 = None
        reshape_default_102: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_155, _shape_param_85);  sum_dim_int_list_155 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_35: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_99, mul_74);  mul_74 = None
        sum_dim_int_list_156: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1, 2]);  mul_tensor_35 = None
        sum_dim_int_list_157: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_99, [0, 1, 2]);  index_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_87: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1190, [1, 0])
        sum_dim_int_list_158: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1190, [0], True);  view_1190 = None
        reshape_default_103: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_158, _shape_param_86);  sum_dim_int_list_158 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_88: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1193, [1, 0])
        sum_dim_int_list_159: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1193, [0], True);  view_1193 = None
        reshape_default_104: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_159, _shape_param_87);  sum_dim_int_list_159 = _shape_param_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_36: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1195, mul_68);  mul_68 = None
        sum_dim_int_list_160: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_161: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1195, [0, 1]);  view_1195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_89: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1200, [1, 0])
        sum_dim_int_list_162: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True);  view_1200 = None
        reshape_default_105: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_162, _shape_param_88);  sum_dim_int_list_162 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_163: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_17, [0], True);  fma_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_17: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_163, 0);  sum_dim_int_list_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_90: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_17, [1, 2, 0]);  squeeze_dim_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_106: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_90, _shape_param_89);  permute_default_90 = _shape_param_89 = None
        reshape_default_107: "i64[2401]" = torch.ops.aten.reshape.default(primals_104, [-1]);  primals_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_17: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_107], reshape_default_106, True);  reshape_default_107 = reshape_default_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_91: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1213, [1, 0])
        sum_dim_int_list_164: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1213, [0], True);  view_1213 = None
        reshape_default_108: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_164, _shape_param_90);  sum_dim_int_list_164 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_37: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1218, mul_64);  mul_64 = None
        sum_dim_int_list_165: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1, 2]);  mul_tensor_37 = None
        sum_dim_int_list_166: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1218, [0, 1, 2]);  view_1218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_92: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1220, [1, 0])
        sum_dim_int_list_167: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1220, [0], True);  view_1220 = None
        reshape_default_109: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_91);  sum_dim_int_list_167 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_93: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1223, [1, 0])
        sum_dim_int_list_168: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1223, [0], True);  view_1223 = None
        reshape_default_110: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_168, _shape_param_92);  sum_dim_int_list_168 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_38: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1225, mul_58);  mul_58 = None
        sum_dim_int_list_169: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_170: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1225, [0, 1]);  view_1225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_94: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1230, [1, 0])
        sum_dim_int_list_171: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1230, [0], True);  view_1230 = None
        reshape_default_111: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_171, _shape_param_93);  sum_dim_int_list_171 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_172: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_18, [0], True);  fma_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_18: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_172, 0);  sum_dim_int_list_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_95: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_18, [1, 2, 0]);  squeeze_dim_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_112: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_95, _shape_param_94);  permute_default_95 = _shape_param_94 = None
        reshape_default_113: "i64[2401]" = torch.ops.aten.reshape.default(primals_90, [-1]);  primals_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_18: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_113], reshape_default_112, True);  reshape_default_113 = reshape_default_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_96: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1245, [1, 0])
        sum_dim_int_list_173: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1245, [0], True);  view_1245 = None
        reshape_default_114: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_173, _shape_param_95);  sum_dim_int_list_173 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_39: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(index_103, mul_54);  mul_54 = None
        sum_dim_int_list_174: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1, 2]);  mul_tensor_39 = None
        sum_dim_int_list_175: "f32[512]" = torch.ops.aten.sum.dim_IntList(index_103, [0, 1, 2]);  index_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_97: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1252, [1, 0])
        sum_dim_int_list_176: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1252, [0], True);  view_1252 = None
        reshape_default_115: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_176, _shape_param_96);  sum_dim_int_list_176 = _shape_param_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_98: "f32[2048, 25088]" = torch.ops.aten.permute.default(view_1255, [1, 0])
        sum_dim_int_list_177: "f32[1, 2048]" = torch.ops.aten.sum.dim_IntList(view_1255, [0], True);  view_1255 = None
        reshape_default_116: "f32[2048]" = torch.ops.aten.reshape.default(sum_dim_int_list_177, _shape_param_97);  sum_dim_int_list_177 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_40: "f32[128, 196, 512]" = torch.ops.aten.mul.Tensor(view_1257, mul_48);  mul_48 = None
        sum_dim_int_list_178: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_179: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1257, [0, 1]);  view_1257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_99: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1262, [1, 0])
        sum_dim_int_list_180: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1262, [0], True);  view_1262 = None
        reshape_default_117: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_180, _shape_param_98);  sum_dim_int_list_180 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_181: "f32[1, 16, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_19, [0], True);  fma_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_19: "f32[16, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_181, 0);  sum_dim_int_list_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_100: "f32[49, 49, 16]" = torch.ops.aten.permute.default(squeeze_dim_19, [1, 2, 0]);  squeeze_dim_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_118: "f32[2401, 16]" = torch.ops.aten.reshape.default(permute_default_100, _shape_param_99);  permute_default_100 = _shape_param_99 = None
        reshape_default_119: "i64[2401]" = torch.ops.aten.reshape.default(primals_75, [-1]);  primals_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_19: "f32[169, 16]" = torch.ops.aten.index_put.default(full_default_1, [reshape_default_119], reshape_default_118, True);  full_default_1 = reshape_default_119 = reshape_default_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_101: "f32[1536, 25088]" = torch.ops.aten.permute.default(view_1275, [1, 0])
        sum_dim_int_list_182: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1275, [0], True);  view_1275 = None
        reshape_default_120: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_182, _shape_param_100);  sum_dim_int_list_182 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_41: "f32[128, 14, 14, 512]" = torch.ops.aten.mul.Tensor(view_1280, mul_44);  mul_44 = None
        sum_dim_int_list_183: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1, 2]);  mul_tensor_41 = None
        sum_dim_int_list_184: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1280, [0, 1, 2]);  view_1280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_default_102: "f32[512, 25088]" = torch.ops.aten.permute.default(view_1281, [1, 0]);  view_1281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_tensor_42: "f32[128, 14, 14, 1024]" = torch.ops.aten.mul.Tensor(view_1282, mul_42);  mul_42 = None
        sum_dim_int_list_185: "f32[1024]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1, 2]);  mul_tensor_42 = None
        sum_dim_int_list_186: "f32[1024]" = torch.ops.aten.sum.dim_IntList(view_1282, [0, 1, 2]);  view_1282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_103: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1286, [1, 0])
        sum_dim_int_list_187: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1286, [0], True);  view_1286 = None
        reshape_default_121: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_187, _shape_param_101);  sum_dim_int_list_187 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_104: "f32[1024, 100352]" = torch.ops.aten.permute.default(view_1289, [1, 0])
        sum_dim_int_list_188: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1289, [0], True);  view_1289 = None
        reshape_default_122: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_188, _shape_param_102);  sum_dim_int_list_188 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_43: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1291, mul_36);  mul_36 = None
        sum_dim_int_list_189: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_190: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1291, [0, 1]);  view_1291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_105: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1296, [1, 0])
        sum_dim_int_list_191: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1296, [0], True);  view_1296 = None
        reshape_default_123: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_191, _shape_param_103);  sum_dim_int_list_191 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_192: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_20, [0], True);  fma_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_20: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_192, 0);  sum_dim_int_list_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_106: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_dim_20, [1, 2, 0]);  squeeze_dim_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_124: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_default_106, _shape_param_104);  permute_default_106 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_2: "f32[169, 8]" = torch.ops.aten.full.default([169, 8], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_125: "i64[2401]" = torch.ops.aten.reshape.default(primals_58, [-1]);  primals_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_20: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default_2, [reshape_default_125], reshape_default_124, True);  reshape_default_125 = reshape_default_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_107: "f32[768, 100352]" = torch.ops.aten.permute.default(view_1311, [1, 0])
        sum_dim_int_list_193: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1311, [0], True);  view_1311 = None
        reshape_default_126: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_193, _shape_param_105);  sum_dim_int_list_193 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_44: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(index_107, mul_32);  mul_32 = None
        sum_dim_int_list_194: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1, 2]);  mul_tensor_44 = None
        sum_dim_int_list_195: "f32[256]" = torch.ops.aten.sum.dim_IntList(index_107, [0, 1, 2]);  index_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_108: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1318, [1, 0])
        sum_dim_int_list_196: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1318, [0], True);  view_1318 = None
        reshape_default_127: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_196, _shape_param_106);  sum_dim_int_list_196 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_109: "f32[1024, 100352]" = torch.ops.aten.permute.default(view_1321, [1, 0])
        sum_dim_int_list_197: "f32[1, 1024]" = torch.ops.aten.sum.dim_IntList(view_1321, [0], True);  view_1321 = None
        reshape_default_128: "f32[1024]" = torch.ops.aten.reshape.default(sum_dim_int_list_197, _shape_param_107);  sum_dim_int_list_197 = _shape_param_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_45: "f32[128, 784, 256]" = torch.ops.aten.mul.Tensor(view_1323, mul_26);  mul_26 = None
        sum_dim_int_list_198: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_199: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1323, [0, 1]);  view_1323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_110: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1328, [1, 0])
        sum_dim_int_list_200: "f32[1, 256]" = torch.ops.aten.sum.dim_IntList(view_1328, [0], True);  view_1328 = None
        reshape_default_129: "f32[256]" = torch.ops.aten.reshape.default(sum_dim_int_list_200, _shape_param_108);  sum_dim_int_list_200 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_201: "f32[1, 8, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_21, [0], True);  fma_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_21: "f32[8, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_201, 0);  sum_dim_int_list_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_111: "f32[49, 49, 8]" = torch.ops.aten.permute.default(squeeze_dim_21, [1, 2, 0]);  squeeze_dim_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_130: "f32[2401, 8]" = torch.ops.aten.reshape.default(permute_default_111, _shape_param_109);  permute_default_111 = _shape_param_109 = None
        reshape_default_131: "i64[2401]" = torch.ops.aten.reshape.default(primals_43, [-1]);  primals_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_21: "f32[169, 8]" = torch.ops.aten.index_put.default(full_default_2, [reshape_default_131], reshape_default_130, True);  full_default_2 = reshape_default_131 = reshape_default_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_112: "f32[768, 100352]" = torch.ops.aten.permute.default(view_1341, [1, 0])
        sum_dim_int_list_202: "f32[1, 768]" = torch.ops.aten.sum.dim_IntList(view_1341, [0], True);  view_1341 = None
        reshape_default_132: "f32[768]" = torch.ops.aten.reshape.default(sum_dim_int_list_202, _shape_param_110);  sum_dim_int_list_202 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_46: "f32[128, 28, 28, 256]" = torch.ops.aten.mul.Tensor(view_1346, mul_22);  mul_22 = None
        sum_dim_int_list_203: "f32[256]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1, 2]);  mul_tensor_46 = None
        sum_dim_int_list_204: "f32[256]" = torch.ops.aten.sum.dim_IntList(view_1346, [0, 1, 2]);  view_1346 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:541 in forward, code: x = self.reduction(x)
        permute_default_113: "f32[256, 100352]" = torch.ops.aten.permute.default(view_1347, [1, 0]);  view_1347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:540 in forward, code: x = self.norm(x)
        mul_tensor_47: "f32[128, 28, 28, 512]" = torch.ops.aten.mul.Tensor(view_1348, mul_20);  mul_20 = None
        sum_dim_int_list_205: "f32[512]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1, 2]);  mul_tensor_47 = None
        sum_dim_int_list_206: "f32[512]" = torch.ops.aten.sum.dim_IntList(view_1348, [0, 1, 2]);  view_1348 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_114: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1352, [1, 0])
        sum_dim_int_list_207: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1352, [0], True);  view_1352 = None
        reshape_default_133: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_207, _shape_param_111);  sum_dim_int_list_207 = _shape_param_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_115: "f32[512, 401408]" = torch.ops.aten.permute.default(view_1355, [1, 0])
        sum_dim_int_list_208: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1355, [0], True);  view_1355 = None
        reshape_default_134: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_208, _shape_param_112);  sum_dim_int_list_208 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_48: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1357, mul_14);  mul_14 = None
        sum_dim_int_list_209: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 1]);  mul_tensor_48 = None
        sum_dim_int_list_210: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1357, [0, 1]);  view_1357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_116: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1362, [1, 0])
        sum_dim_int_list_211: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1362, [0], True);  view_1362 = None
        reshape_default_135: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_211, _shape_param_113);  sum_dim_int_list_211 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_212: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_22, [0], True);  fma_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_22: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_212, 0);  sum_dim_int_list_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_117: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_dim_22, [1, 2, 0]);  squeeze_dim_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_136: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_default_117, _shape_param_114);  permute_default_117 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        full_default_3: "f32[169, 4]" = torch.ops.aten.full.default([169, 4], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_137: "i64[2401]" = torch.ops.aten.reshape.default(primals_26, [-1]);  primals_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_22: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default_3, [reshape_default_137], reshape_default_136, True);  reshape_default_137 = reshape_default_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_118: "f32[384, 401408]" = torch.ops.aten.permute.default(view_1377, [1, 0])
        sum_dim_int_list_213: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1377, [0], True);  view_1377 = None
        reshape_default_138: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_213, _shape_param_115);  sum_dim_int_list_213 = _shape_param_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_49: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(index_111, mul_10);  mul_10 = None
        sum_dim_int_list_214: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_49, [0, 1, 2]);  mul_tensor_49 = None
        sum_dim_int_list_215: "f32[128]" = torch.ops.aten.sum.dim_IntList(index_111, [0, 1, 2]);  index_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:52 in forward, code: x = self.fc2(x)
        permute_default_119: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1384, [1, 0])
        sum_dim_int_list_216: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1384, [0], True);  view_1384 = None
        reshape_default_139: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_216, _shape_param_116);  sum_dim_int_list_216 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/mlp.py:48 in forward, code: x = self.fc1(x)
        permute_default_120: "f32[512, 401408]" = torch.ops.aten.permute.default(view_1387, [1, 0])
        sum_dim_int_list_217: "f32[1, 512]" = torch.ops.aten.sum.dim_IntList(view_1387, [0], True);  view_1387 = None
        reshape_default_140: "f32[512]" = torch.ops.aten.reshape.default(sum_dim_int_list_217, _shape_param_117);  sum_dim_int_list_217 = _shape_param_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:488 in forward, code: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        mul_tensor_50: "f32[128, 3136, 128]" = torch.ops.aten.mul.Tensor(view_1389, mul_5);  mul_5 = None
        sum_dim_int_list_218: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [0, 1]);  mul_tensor_50 = None
        sum_dim_int_list_219: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1389, [0, 1]);  view_1389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:246 in forward, code: x = self.proj(x)
        permute_default_121: "f32[128, 401408]" = torch.ops.aten.permute.default(view_1394, [1, 0])
        sum_dim_int_list_220: "f32[1, 128]" = torch.ops.aten.sum.dim_IntList(view_1394, [0], True);  view_1394 = None
        reshape_default_141: "f32[128]" = torch.ops.aten.reshape.default(sum_dim_int_list_220, _shape_param_118);  sum_dim_int_list_220 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:236 in forward, code: attn = attn + self._get_rel_pos_bias()
        sum_dim_int_list_221: "f32[1, 4, 49, 49]" = torch.ops.aten.sum.dim_IntList(fma_23, [0], True);  fma_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:206 in _get_rel_pos_bias, code: return relative_position_bias.unsqueeze(0)
        squeeze_dim_23: "f32[4, 49, 49]" = torch.ops.aten.squeeze.dim(sum_dim_int_list_221, 0);  sum_dim_int_list_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:205 in _get_rel_pos_bias, code: relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        permute_default_122: "f32[49, 49, 4]" = torch.ops.aten.permute.default(squeeze_dim_23, [1, 2, 0]);  squeeze_dim_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:204 in _get_rel_pos_bias, code: self.relative_position_index.view(-1)].view(self.window_area, self.window_area, -1)  # Wh*Ww,Wh*Ww,nH
        reshape_default_142: "f32[2401, 4]" = torch.ops.aten.reshape.default(permute_default_122, _shape_param_119);  permute_default_122 = _shape_param_119 = None
        reshape_default_143: "i64[2401]" = torch.ops.aten.reshape.default(primals_11, [-1]);  primals_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:203 in _get_rel_pos_bias, code: relative_position_bias = self.relative_position_bias_table[
        index_put_default_23: "f32[169, 4]" = torch.ops.aten.index_put.default(full_default_3, [reshape_default_143], reshape_default_142, True);  full_default_3 = reshape_default_143 = reshape_default_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:219 in forward, code: qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        permute_default_123: "f32[384, 401408]" = torch.ops.aten.permute.default(view_1407, [1, 0])
        sum_dim_int_list_222: "f32[1, 384]" = torch.ops.aten.sum.dim_IntList(view_1407, [0], True);  view_1407 = None
        reshape_default_144: "f32[384]" = torch.ops.aten.reshape.default(sum_dim_int_list_222, _shape_param_120);  sum_dim_int_list_222 = _shape_param_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/models/swin_transformer.py:486 in forward, code: x = x + self.drop_path1(self._attn(self.norm1(x)))
        mul_tensor_51: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(view_1412, mul_2);  mul_2 = None
        sum_dim_int_list_223: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [0, 1, 2]);  mul_tensor_51 = None
        sum_dim_int_list_224: "f32[128]" = torch.ops.aten.sum.dim_IntList(view_1412, [0, 1, 2]);  view_1412 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:141 in forward, code: x = self.norm(x)
        mul_tensor_52: "f32[128, 56, 56, 128]" = torch.ops.aten.mul.Tensor(add_396, mul);  mul = None
        sum_dim_int_list_225: "f32[128]" = torch.ops.aten.sum.dim_IntList(mul_tensor_52, [0, 1, 2]);  mul_tensor_52 = None
        sum_dim_int_list_226: "f32[128]" = torch.ops.aten.sum.dim_IntList(add_396, [0, 1, 2]);  add_396 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/timm/layers/patch_embed.py:136 in forward, code: x = self.proj(x)
        sum_dim_int_list_227: "f32[128]" = torch.ops.aten.sum.dim_IntList(permute_891, [0, 2, 3]);  permute_891 = None
        return (permute_default, reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default_1, reshape_default_1, permute_default_2, reshape_default_2, sum_dim_int_list_5, sum_dim_int_list_6, permute_default_3, reshape_default_3, index_put_default, permute_default_5, reshape_default_6, sum_dim_int_list_10, sum_dim_int_list_11, permute_default_6, reshape_default_7, permute_default_7, reshape_default_8, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_8, reshape_default_9, index_put_default_1, permute_default_10, reshape_default_12, sum_dim_int_list_19, sum_dim_int_list_20, permute_default_11, sum_dim_int_list_21, sum_dim_int_list_22, permute_default_12, reshape_default_13, permute_default_13, reshape_default_14, sum_dim_int_list_25, sum_dim_int_list_26, permute_default_14, reshape_default_15, index_put_default_2, permute_default_16, reshape_default_18, sum_dim_int_list_30, sum_dim_int_list_31, permute_default_17, reshape_default_19, permute_default_18, reshape_default_20, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_21, index_put_default_3, permute_default_21, reshape_default_24, sum_dim_int_list_39, sum_dim_int_list_40, permute_default_22, reshape_default_25, permute_default_23, reshape_default_26, sum_dim_int_list_43, sum_dim_int_list_44, permute_default_24, reshape_default_27, index_put_default_4, permute_default_26, reshape_default_30, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_31, permute_default_28, reshape_default_32, sum_dim_int_list_52, sum_dim_int_list_53, permute_default_29, reshape_default_33, index_put_default_5, permute_default_31, reshape_default_36, sum_dim_int_list_57, sum_dim_int_list_58, permute_default_32, reshape_default_37, permute_default_33, reshape_default_38, sum_dim_int_list_61, sum_dim_int_list_62, permute_default_34, reshape_default_39, index_put_default_6, permute_default_36, reshape_default_42, sum_dim_int_list_66, sum_dim_int_list_67, permute_default_37, reshape_default_43, permute_default_38, reshape_default_44, sum_dim_int_list_70, sum_dim_int_list_71, permute_default_39, reshape_default_45, index_put_default_7, permute_default_41, reshape_default_48, sum_dim_int_list_75, sum_dim_int_list_76, permute_default_42, reshape_default_49, permute_default_43, reshape_default_50, sum_dim_int_list_79, sum_dim_int_list_80, permute_default_44, reshape_default_51, index_put_default_8, permute_default_46, reshape_default_54, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_47, reshape_default_55, permute_default_48, reshape_default_56, sum_dim_int_list_88, sum_dim_int_list_89, permute_default_49, reshape_default_57, index_put_default_9, permute_default_51, reshape_default_60, sum_dim_int_list_93, sum_dim_int_list_94, permute_default_52, reshape_default_61, permute_default_53, reshape_default_62, sum_dim_int_list_97, sum_dim_int_list_98, permute_default_54, reshape_default_63, index_put_default_10, permute_default_56, reshape_default_66, sum_dim_int_list_102, sum_dim_int_list_103, permute_default_57, reshape_default_67, permute_default_58, reshape_default_68, sum_dim_int_list_106, sum_dim_int_list_107, permute_default_59, reshape_default_69, index_put_default_11, permute_default_61, reshape_default_72, sum_dim_int_list_111, sum_dim_int_list_112, permute_default_62, reshape_default_73, permute_default_63, reshape_default_74, sum_dim_int_list_115, sum_dim_int_list_116, permute_default_64, reshape_default_75, index_put_default_12, permute_default_66, reshape_default_78, sum_dim_int_list_120, sum_dim_int_list_121, permute_default_67, reshape_default_79, permute_default_68, reshape_default_80, sum_dim_int_list_124, sum_dim_int_list_125, permute_default_69, reshape_default_81, index_put_default_13, permute_default_71, reshape_default_84, sum_dim_int_list_129, sum_dim_int_list_130, permute_default_72, reshape_default_85, permute_default_73, reshape_default_86, sum_dim_int_list_133, sum_dim_int_list_134, permute_default_74, reshape_default_87, index_put_default_14, permute_default_76, reshape_default_90, sum_dim_int_list_138, sum_dim_int_list_139, permute_default_77, reshape_default_91, permute_default_78, reshape_default_92, sum_dim_int_list_142, sum_dim_int_list_143, permute_default_79, reshape_default_93, index_put_default_15, permute_default_81, reshape_default_96, sum_dim_int_list_147, sum_dim_int_list_148, permute_default_82, reshape_default_97, permute_default_83, reshape_default_98, sum_dim_int_list_151, sum_dim_int_list_152, permute_default_84, reshape_default_99, index_put_default_16, permute_default_86, reshape_default_102, sum_dim_int_list_156, sum_dim_int_list_157, permute_default_87, reshape_default_103, permute_default_88, reshape_default_104, sum_dim_int_list_160, sum_dim_int_list_161, permute_default_89, reshape_default_105, index_put_default_17, permute_default_91, reshape_default_108, sum_dim_int_list_165, sum_dim_int_list_166, permute_default_92, reshape_default_109, permute_default_93, reshape_default_110, sum_dim_int_list_169, sum_dim_int_list_170, permute_default_94, reshape_default_111, index_put_default_18, permute_default_96, reshape_default_114, sum_dim_int_list_174, sum_dim_int_list_175, permute_default_97, reshape_default_115, permute_default_98, reshape_default_116, sum_dim_int_list_178, sum_dim_int_list_179, permute_default_99, reshape_default_117, index_put_default_19, permute_default_101, reshape_default_120, sum_dim_int_list_183, sum_dim_int_list_184, permute_default_102, sum_dim_int_list_185, sum_dim_int_list_186, permute_default_103, reshape_default_121, permute_default_104, reshape_default_122, sum_dim_int_list_189, sum_dim_int_list_190, permute_default_105, reshape_default_123, index_put_default_20, permute_default_107, reshape_default_126, sum_dim_int_list_194, sum_dim_int_list_195, permute_default_108, reshape_default_127, permute_default_109, reshape_default_128, sum_dim_int_list_198, sum_dim_int_list_199, permute_default_110, reshape_default_129, index_put_default_21, permute_default_112, reshape_default_132, sum_dim_int_list_203, sum_dim_int_list_204, permute_default_113, sum_dim_int_list_205, sum_dim_int_list_206, permute_default_114, reshape_default_133, permute_default_115, reshape_default_134, sum_dim_int_list_209, sum_dim_int_list_210, permute_default_116, reshape_default_135, index_put_default_22, permute_default_118, reshape_default_138, sum_dim_int_list_214, sum_dim_int_list_215, permute_default_119, reshape_default_139, permute_default_120, reshape_default_140, sum_dim_int_list_218, sum_dim_int_list_219, permute_default_121, reshape_default_141, index_put_default_23, permute_default_123, reshape_default_144, sum_dim_int_list_223, sum_dim_int_list_224, sum_dim_int_list_225, sum_dim_int_list_226, sum_dim_int_list_227)


def _default_make_inputs():
    from repro_harness import parse_shapes_config
    return parse_shapes_config(_shapes_config)


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

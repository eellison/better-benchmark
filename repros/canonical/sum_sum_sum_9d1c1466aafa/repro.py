"""
Standalone repro captured via capture_hook.
Label: hf_DebertaV2ForMaskedLM_train
Pattern hash: 9d1c1466aafa
Shape hash: cf39dc8d
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_shapes_config = "(T([4096, 128100], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 6144], f32), T([8, 512, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([4096, 1536], f32), T([8, 512, 1536], b8), T([1536], f32), T([8, 512, 1536], f32), T([1, 512, 1536], f32), T([8, 512, 1], f32), T([8, 512, 1], f32), T([1, 512], i64, gen=Index(512)), T([], f32), T([8, 512], i64, gen=Index(128100)), T([128100, 1536], f32), S([128100]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([1536]), S([1536]), S([1536]), S([6144]), S([1536]), S([1536]), S([8, 512, 1536]), S([1536]), S([8, 512, 1536]), S([1536]), S([8, 512, 1536]))"

class Repro(torch.nn.Module):
    def forward(self, view_535: "f32[4096, 128100]", view_537: "f32[8, 512, 1536]", mul_345: "f32[8, 512, 1536]", view_538: "f32[4096, 1536]", view_540: "f32[8, 512, 1536]", mul_340: "f32[8, 512, 1536]", view_541: "f32[4096, 1536]", view_544: "f32[4096, 6144]", add_179: "f32[8, 512, 1536]", mul_333: "f32[8, 512, 1536]", view_547: "f32[4096, 1536]", view_556: "f32[4096, 1536]", view_561: "f32[4096, 1536]", view_566: "f32[4096, 1536]", add_182: "f32[8, 512, 1536]", mul_326: "f32[8, 512, 1536]", view_569: "f32[4096, 1536]", view_572: "f32[4096, 6144]", add_185: "f32[8, 512, 1536]", mul_319: "f32[8, 512, 1536]", view_575: "f32[4096, 1536]", view_584: "f32[4096, 1536]", view_589: "f32[4096, 1536]", view_594: "f32[4096, 1536]", add_188: "f32[8, 512, 1536]", mul_312: "f32[8, 512, 1536]", view_597: "f32[4096, 1536]", view_600: "f32[4096, 6144]", add_191: "f32[8, 512, 1536]", mul_305: "f32[8, 512, 1536]", view_603: "f32[4096, 1536]", view_612: "f32[4096, 1536]", view_617: "f32[4096, 1536]", view_622: "f32[4096, 1536]", add_194: "f32[8, 512, 1536]", mul_298: "f32[8, 512, 1536]", view_625: "f32[4096, 1536]", view_628: "f32[4096, 6144]", add_197: "f32[8, 512, 1536]", mul_291: "f32[8, 512, 1536]", view_631: "f32[4096, 1536]", view_640: "f32[4096, 1536]", view_645: "f32[4096, 1536]", view_650: "f32[4096, 1536]", add_200: "f32[8, 512, 1536]", mul_284: "f32[8, 512, 1536]", view_653: "f32[4096, 1536]", view_656: "f32[4096, 6144]", add_203: "f32[8, 512, 1536]", mul_277: "f32[8, 512, 1536]", view_659: "f32[4096, 1536]", view_668: "f32[4096, 1536]", view_673: "f32[4096, 1536]", view_678: "f32[4096, 1536]", add_206: "f32[8, 512, 1536]", mul_270: "f32[8, 512, 1536]", view_681: "f32[4096, 1536]", view_684: "f32[4096, 6144]", add_209: "f32[8, 512, 1536]", mul_263: "f32[8, 512, 1536]", view_687: "f32[4096, 1536]", view_696: "f32[4096, 1536]", view_701: "f32[4096, 1536]", view_706: "f32[4096, 1536]", add_212: "f32[8, 512, 1536]", mul_256: "f32[8, 512, 1536]", view_709: "f32[4096, 1536]", view_712: "f32[4096, 6144]", add_215: "f32[8, 512, 1536]", mul_249: "f32[8, 512, 1536]", view_715: "f32[4096, 1536]", view_724: "f32[4096, 1536]", view_729: "f32[4096, 1536]", view_734: "f32[4096, 1536]", add_218: "f32[8, 512, 1536]", mul_242: "f32[8, 512, 1536]", view_737: "f32[4096, 1536]", view_740: "f32[4096, 6144]", add_221: "f32[8, 512, 1536]", mul_235: "f32[8, 512, 1536]", view_743: "f32[4096, 1536]", view_752: "f32[4096, 1536]", view_757: "f32[4096, 1536]", view_762: "f32[4096, 1536]", add_224: "f32[8, 512, 1536]", mul_228: "f32[8, 512, 1536]", view_765: "f32[4096, 1536]", view_768: "f32[4096, 6144]", add_227: "f32[8, 512, 1536]", mul_221: "f32[8, 512, 1536]", view_771: "f32[4096, 1536]", view_780: "f32[4096, 1536]", view_785: "f32[4096, 1536]", view_790: "f32[4096, 1536]", add_230: "f32[8, 512, 1536]", mul_214: "f32[8, 512, 1536]", view_793: "f32[4096, 1536]", view_796: "f32[4096, 6144]", add_233: "f32[8, 512, 1536]", mul_207: "f32[8, 512, 1536]", view_799: "f32[4096, 1536]", view_808: "f32[4096, 1536]", view_813: "f32[4096, 1536]", view_818: "f32[4096, 1536]", add_236: "f32[8, 512, 1536]", mul_200: "f32[8, 512, 1536]", view_821: "f32[4096, 1536]", view_824: "f32[4096, 6144]", add_239: "f32[8, 512, 1536]", mul_193: "f32[8, 512, 1536]", view_827: "f32[4096, 1536]", view_836: "f32[4096, 1536]", view_841: "f32[4096, 1536]", view_846: "f32[4096, 1536]", add_242: "f32[8, 512, 1536]", mul_186: "f32[8, 512, 1536]", view_849: "f32[4096, 1536]", view_852: "f32[4096, 6144]", add_245: "f32[8, 512, 1536]", mul_179: "f32[8, 512, 1536]", view_855: "f32[4096, 1536]", view_864: "f32[4096, 1536]", view_869: "f32[4096, 1536]", view_874: "f32[4096, 1536]", add_248: "f32[8, 512, 1536]", mul_172: "f32[8, 512, 1536]", view_877: "f32[4096, 1536]", view_880: "f32[4096, 6144]", add_251: "f32[8, 512, 1536]", mul_165: "f32[8, 512, 1536]", view_883: "f32[4096, 1536]", view_892: "f32[4096, 1536]", view_897: "f32[4096, 1536]", view_902: "f32[4096, 1536]", add_254: "f32[8, 512, 1536]", mul_158: "f32[8, 512, 1536]", view_905: "f32[4096, 1536]", view_908: "f32[4096, 6144]", add_257: "f32[8, 512, 1536]", mul_151: "f32[8, 512, 1536]", view_911: "f32[4096, 1536]", view_920: "f32[4096, 1536]", view_925: "f32[4096, 1536]", view_930: "f32[4096, 1536]", add_260: "f32[8, 512, 1536]", mul_144: "f32[8, 512, 1536]", view_933: "f32[4096, 1536]", view_936: "f32[4096, 6144]", add_263: "f32[8, 512, 1536]", mul_137: "f32[8, 512, 1536]", view_939: "f32[4096, 1536]", view_948: "f32[4096, 1536]", view_953: "f32[4096, 1536]", view_958: "f32[4096, 1536]", add_266: "f32[8, 512, 1536]", mul_130: "f32[8, 512, 1536]", view_961: "f32[4096, 1536]", view_964: "f32[4096, 6144]", add_269: "f32[8, 512, 1536]", mul_123: "f32[8, 512, 1536]", view_967: "f32[4096, 1536]", view_976: "f32[4096, 1536]", view_981: "f32[4096, 1536]", view_986: "f32[4096, 1536]", add_272: "f32[8, 512, 1536]", mul_116: "f32[8, 512, 1536]", view_989: "f32[4096, 1536]", view_992: "f32[4096, 6144]", add_275: "f32[8, 512, 1536]", mul_109: "f32[8, 512, 1536]", view_995: "f32[4096, 1536]", view_1004: "f32[4096, 1536]", view_1009: "f32[4096, 1536]", view_1014: "f32[4096, 1536]", add_278: "f32[8, 512, 1536]", mul_102: "f32[8, 512, 1536]", view_1017: "f32[4096, 1536]", view_1020: "f32[4096, 6144]", add_281: "f32[8, 512, 1536]", mul_95: "f32[8, 512, 1536]", view_1023: "f32[4096, 1536]", view_1032: "f32[4096, 1536]", view_1037: "f32[4096, 1536]", view_1042: "f32[4096, 1536]", add_284: "f32[8, 512, 1536]", mul_88: "f32[8, 512, 1536]", view_1045: "f32[4096, 1536]", view_1048: "f32[4096, 6144]", add_287: "f32[8, 512, 1536]", mul_81: "f32[8, 512, 1536]", view_1051: "f32[4096, 1536]", view_1060: "f32[4096, 1536]", view_1065: "f32[4096, 1536]", view_1070: "f32[4096, 1536]", add_290: "f32[8, 512, 1536]", mul_74: "f32[8, 512, 1536]", view_1073: "f32[4096, 1536]", view_1076: "f32[4096, 6144]", add_293: "f32[8, 512, 1536]", mul_67: "f32[8, 512, 1536]", view_1079: "f32[4096, 1536]", view_1088: "f32[4096, 1536]", view_1093: "f32[4096, 1536]", view_1098: "f32[4096, 1536]", add_296: "f32[8, 512, 1536]", mul_60: "f32[8, 512, 1536]", view_1101: "f32[4096, 1536]", view_1104: "f32[4096, 6144]", add_299: "f32[8, 512, 1536]", mul_53: "f32[8, 512, 1536]", view_1107: "f32[4096, 1536]", view_1116: "f32[4096, 1536]", view_1121: "f32[4096, 1536]", view_1126: "f32[4096, 1536]", add_302: "f32[8, 512, 1536]", mul_46: "f32[8, 512, 1536]", view_1129: "f32[4096, 1536]", view_1132: "f32[4096, 6144]", add_305: "f32[8, 512, 1536]", mul_39: "f32[8, 512, 1536]", view_1135: "f32[4096, 1536]", view_1144: "f32[4096, 1536]", view_1149: "f32[4096, 1536]", view_1154: "f32[4096, 1536]", add_308: "f32[8, 512, 1536]", mul_32: "f32[8, 512, 1536]", view_1157: "f32[4096, 1536]", view_1160: "f32[4096, 6144]", add_311: "f32[8, 512, 1536]", mul_25: "f32[8, 512, 1536]", view_1163: "f32[4096, 1536]", view_1172: "f32[4096, 1536]", view_1177: "f32[4096, 1536]", view_1182: "f32[4096, 1536]", add_314: "f32[8, 512, 1536]", mul_18: "f32[8, 512, 1536]", view_1185: "f32[4096, 1536]", view_1188: "f32[4096, 6144]", add_317: "f32[8, 512, 1536]", mul_11: "f32[8, 512, 1536]", view_1191: "f32[4096, 1536]", view_1200: "f32[4096, 1536]", mm_286: "f32[4096, 1536]", mul_1028: "f32[8, 512, 1536]", view_1205: "f32[4096, 1536]", mm_288: "f32[4096, 1536]", view_1210: "f32[4096, 1536]", mm_290: "f32[4096, 1536]", gt: "b8[8, 512, 1536]", primals_5: "f32[1536]", embedding: "f32[8, 512, 1536]", embedding_1: "f32[1, 512, 1536]", getitem_1: "f32[8, 512, 1]", rsqrt: "f32[8, 512, 1]", primals_2: "i64[1, 512]", full_default_74: "f32[]", primals_1: "i64[8, 512]", mm_1: "f32[128100, 1536]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:838 in forward, code: hidden_states = self.decoder(hidden_states)
        sum_dim_int_list: "f32[1, 128100]" = torch.ops.aten.sum.dim_IntList(view_535, [0], True);  view_535 = None
        reshape_default: "f32[128100]" = torch.ops.aten.reshape.default(sum_dim_int_list, _shape_param_0);  sum_dim_int_list = _shape_param_0 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:820 in forward, code: hidden_states = self.LayerNorm(hidden_states)
        mul_tensor: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_537, mul_345);  mul_345 = None
        sum_dim_int_list_1: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor, [0, 1]);  mul_tensor = None
        sum_dim_int_list_2: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_537, [0, 1]);  view_537 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:818 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_538, [1, 0])
        sum_dim_int_list_3: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_538, [0], True);  view_538 = None
        reshape_default_1: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_3, _shape_param_1);  sum_dim_int_list_3 = _shape_param_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(view_540, mul_340);  mul_340 = None
        sum_dim_int_list_4: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_1, [0, 1]);  mul_tensor_1 = None
        sum_dim_int_list_5: "f32[1536]" = torch.ops.aten.sum.dim_IntList(view_540, [0, 1]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_1: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_541, [1, 0])
        sum_dim_int_list_6: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_541, [0], True);  view_541 = None
        reshape_default_2: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_6, _shape_param_2);  sum_dim_int_list_6 = _shape_param_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_2: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_544, [1, 0])
        sum_dim_int_list_7: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_544, [0], True);  view_544 = None
        reshape_default_3: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_7, _shape_param_3);  sum_dim_int_list_7 = _shape_param_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_179, mul_333);  mul_333 = None
        sum_dim_int_list_8: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_2, [0, 1]);  mul_tensor_2 = None
        sum_dim_int_list_9: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_179, [0, 1]);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_3: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_547, [1, 0])
        sum_dim_int_list_10: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_547, [0], True);  view_547 = None
        reshape_default_4: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_10, _shape_param_4);  sum_dim_int_list_10 = _shape_param_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_4: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_556, [1, 0])
        sum_dim_int_list_11: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_556, [0], True);  view_556 = None
        reshape_default_5: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_11, _shape_param_5);  sum_dim_int_list_11 = _shape_param_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_5: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_561, [1, 0])
        sum_dim_int_list_12: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_561, [0], True);  view_561 = None
        reshape_default_6: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_12, _shape_param_6);  sum_dim_int_list_12 = _shape_param_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_6: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_566, [1, 0])
        sum_dim_int_list_13: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_566, [0], True);  view_566 = None
        reshape_default_7: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_13, _shape_param_7);  sum_dim_int_list_13 = _shape_param_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_3: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_182, mul_326);  mul_326 = None
        sum_dim_int_list_14: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_3, [0, 1]);  mul_tensor_3 = None
        sum_dim_int_list_15: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_182, [0, 1]);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_7: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_569, [1, 0])
        sum_dim_int_list_16: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_569, [0], True);  view_569 = None
        reshape_default_8: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_16, _shape_param_8);  sum_dim_int_list_16 = _shape_param_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_8: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_572, [1, 0])
        sum_dim_int_list_17: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_572, [0], True);  view_572 = None
        reshape_default_9: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_17, _shape_param_9);  sum_dim_int_list_17 = _shape_param_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_4: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_185, mul_319);  mul_319 = None
        sum_dim_int_list_18: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_4, [0, 1]);  mul_tensor_4 = None
        sum_dim_int_list_19: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_185, [0, 1]);  add_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_9: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_575, [1, 0])
        sum_dim_int_list_20: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_575, [0], True);  view_575 = None
        reshape_default_10: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_20, _shape_param_10);  sum_dim_int_list_20 = _shape_param_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_10: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_584, [1, 0])
        sum_dim_int_list_21: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_584, [0], True);  view_584 = None
        reshape_default_11: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_21, _shape_param_11);  sum_dim_int_list_21 = _shape_param_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_11: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_589, [1, 0])
        sum_dim_int_list_22: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_589, [0], True);  view_589 = None
        reshape_default_12: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_22, _shape_param_12);  sum_dim_int_list_22 = _shape_param_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_12: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_594, [1, 0])
        sum_dim_int_list_23: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_594, [0], True);  view_594 = None
        reshape_default_13: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_23, _shape_param_13);  sum_dim_int_list_23 = _shape_param_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_5: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_188, mul_312);  mul_312 = None
        sum_dim_int_list_24: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_5, [0, 1]);  mul_tensor_5 = None
        sum_dim_int_list_25: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_188, [0, 1]);  add_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_13: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_597, [1, 0])
        sum_dim_int_list_26: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_597, [0], True);  view_597 = None
        reshape_default_14: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_26, _shape_param_14);  sum_dim_int_list_26 = _shape_param_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_14: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_600, [1, 0])
        sum_dim_int_list_27: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_600, [0], True);  view_600 = None
        reshape_default_15: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_27, _shape_param_15);  sum_dim_int_list_27 = _shape_param_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_6: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_191, mul_305);  mul_305 = None
        sum_dim_int_list_28: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_6, [0, 1]);  mul_tensor_6 = None
        sum_dim_int_list_29: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_191, [0, 1]);  add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_15: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_603, [1, 0])
        sum_dim_int_list_30: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_603, [0], True);  view_603 = None
        reshape_default_16: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_30, _shape_param_16);  sum_dim_int_list_30 = _shape_param_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_16: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_612, [1, 0])
        sum_dim_int_list_31: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_612, [0], True);  view_612 = None
        reshape_default_17: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_31, _shape_param_17);  sum_dim_int_list_31 = _shape_param_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_17: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_617, [1, 0])
        sum_dim_int_list_32: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_617, [0], True);  view_617 = None
        reshape_default_18: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_32, _shape_param_18);  sum_dim_int_list_32 = _shape_param_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_18: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_622, [1, 0])
        sum_dim_int_list_33: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_622, [0], True);  view_622 = None
        reshape_default_19: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_33, _shape_param_19);  sum_dim_int_list_33 = _shape_param_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_7: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_194, mul_298);  mul_298 = None
        sum_dim_int_list_34: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_7, [0, 1]);  mul_tensor_7 = None
        sum_dim_int_list_35: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_194, [0, 1]);  add_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_19: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_625, [1, 0])
        sum_dim_int_list_36: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_625, [0], True);  view_625 = None
        reshape_default_20: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_36, _shape_param_20);  sum_dim_int_list_36 = _shape_param_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_20: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_628, [1, 0])
        sum_dim_int_list_37: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_628, [0], True);  view_628 = None
        reshape_default_21: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_37, _shape_param_21);  sum_dim_int_list_37 = _shape_param_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_8: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_197, mul_291);  mul_291 = None
        sum_dim_int_list_38: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_8, [0, 1]);  mul_tensor_8 = None
        sum_dim_int_list_39: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_197, [0, 1]);  add_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_21: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_631, [1, 0])
        sum_dim_int_list_40: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_631, [0], True);  view_631 = None
        reshape_default_22: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_40, _shape_param_22);  sum_dim_int_list_40 = _shape_param_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_22: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_640, [1, 0])
        sum_dim_int_list_41: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_640, [0], True);  view_640 = None
        reshape_default_23: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_41, _shape_param_23);  sum_dim_int_list_41 = _shape_param_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_23: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_645, [1, 0])
        sum_dim_int_list_42: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_645, [0], True);  view_645 = None
        reshape_default_24: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_42, _shape_param_24);  sum_dim_int_list_42 = _shape_param_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_24: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_650, [1, 0])
        sum_dim_int_list_43: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_650, [0], True);  view_650 = None
        reshape_default_25: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_43, _shape_param_25);  sum_dim_int_list_43 = _shape_param_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_9: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_200, mul_284);  mul_284 = None
        sum_dim_int_list_44: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_9, [0, 1]);  mul_tensor_9 = None
        sum_dim_int_list_45: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_200, [0, 1]);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_25: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_653, [1, 0])
        sum_dim_int_list_46: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_653, [0], True);  view_653 = None
        reshape_default_26: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_46, _shape_param_26);  sum_dim_int_list_46 = _shape_param_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_26: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_656, [1, 0])
        sum_dim_int_list_47: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_656, [0], True);  view_656 = None
        reshape_default_27: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_47, _shape_param_27);  sum_dim_int_list_47 = _shape_param_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_10: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_203, mul_277);  mul_277 = None
        sum_dim_int_list_48: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_10, [0, 1]);  mul_tensor_10 = None
        sum_dim_int_list_49: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_203, [0, 1]);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_27: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_659, [1, 0])
        sum_dim_int_list_50: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_659, [0], True);  view_659 = None
        reshape_default_28: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_50, _shape_param_28);  sum_dim_int_list_50 = _shape_param_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_28: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_668, [1, 0])
        sum_dim_int_list_51: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_668, [0], True);  view_668 = None
        reshape_default_29: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_51, _shape_param_29);  sum_dim_int_list_51 = _shape_param_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_29: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_673, [1, 0])
        sum_dim_int_list_52: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_673, [0], True);  view_673 = None
        reshape_default_30: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_52, _shape_param_30);  sum_dim_int_list_52 = _shape_param_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_30: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_678, [1, 0])
        sum_dim_int_list_53: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_678, [0], True);  view_678 = None
        reshape_default_31: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_53, _shape_param_31);  sum_dim_int_list_53 = _shape_param_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_11: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_206, mul_270);  mul_270 = None
        sum_dim_int_list_54: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_11, [0, 1]);  mul_tensor_11 = None
        sum_dim_int_list_55: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_206, [0, 1]);  add_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_31: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_681, [1, 0])
        sum_dim_int_list_56: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_681, [0], True);  view_681 = None
        reshape_default_32: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_56, _shape_param_32);  sum_dim_int_list_56 = _shape_param_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_32: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_684, [1, 0])
        sum_dim_int_list_57: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_684, [0], True);  view_684 = None
        reshape_default_33: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_57, _shape_param_33);  sum_dim_int_list_57 = _shape_param_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_12: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_209, mul_263);  mul_263 = None
        sum_dim_int_list_58: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_12, [0, 1]);  mul_tensor_12 = None
        sum_dim_int_list_59: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_209, [0, 1]);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_33: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_687, [1, 0])
        sum_dim_int_list_60: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_687, [0], True);  view_687 = None
        reshape_default_34: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_60, _shape_param_34);  sum_dim_int_list_60 = _shape_param_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_34: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_696, [1, 0])
        sum_dim_int_list_61: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_696, [0], True);  view_696 = None
        reshape_default_35: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_61, _shape_param_35);  sum_dim_int_list_61 = _shape_param_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_35: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_701, [1, 0])
        sum_dim_int_list_62: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_701, [0], True);  view_701 = None
        reshape_default_36: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_62, _shape_param_36);  sum_dim_int_list_62 = _shape_param_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_36: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_706, [1, 0])
        sum_dim_int_list_63: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_706, [0], True);  view_706 = None
        reshape_default_37: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_63, _shape_param_37);  sum_dim_int_list_63 = _shape_param_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_13: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_212, mul_256);  mul_256 = None
        sum_dim_int_list_64: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_13, [0, 1]);  mul_tensor_13 = None
        sum_dim_int_list_65: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_212, [0, 1]);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_37: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_709, [1, 0])
        sum_dim_int_list_66: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_709, [0], True);  view_709 = None
        reshape_default_38: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_66, _shape_param_38);  sum_dim_int_list_66 = _shape_param_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_38: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_712, [1, 0])
        sum_dim_int_list_67: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_712, [0], True);  view_712 = None
        reshape_default_39: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_67, _shape_param_39);  sum_dim_int_list_67 = _shape_param_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_14: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_215, mul_249);  mul_249 = None
        sum_dim_int_list_68: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_14, [0, 1]);  mul_tensor_14 = None
        sum_dim_int_list_69: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_215, [0, 1]);  add_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_39: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_715, [1, 0])
        sum_dim_int_list_70: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_715, [0], True);  view_715 = None
        reshape_default_40: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_70, _shape_param_40);  sum_dim_int_list_70 = _shape_param_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_40: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_724, [1, 0])
        sum_dim_int_list_71: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_724, [0], True);  view_724 = None
        reshape_default_41: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_71, _shape_param_41);  sum_dim_int_list_71 = _shape_param_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_41: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_729, [1, 0])
        sum_dim_int_list_72: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_729, [0], True);  view_729 = None
        reshape_default_42: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_72, _shape_param_42);  sum_dim_int_list_72 = _shape_param_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_42: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_734, [1, 0])
        sum_dim_int_list_73: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_734, [0], True);  view_734 = None
        reshape_default_43: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_73, _shape_param_43);  sum_dim_int_list_73 = _shape_param_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_15: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_218, mul_242);  mul_242 = None
        sum_dim_int_list_74: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_15, [0, 1]);  mul_tensor_15 = None
        sum_dim_int_list_75: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_218, [0, 1]);  add_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_43: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_737, [1, 0])
        sum_dim_int_list_76: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_737, [0], True);  view_737 = None
        reshape_default_44: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_76, _shape_param_44);  sum_dim_int_list_76 = _shape_param_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_44: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_740, [1, 0])
        sum_dim_int_list_77: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_740, [0], True);  view_740 = None
        reshape_default_45: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_77, _shape_param_45);  sum_dim_int_list_77 = _shape_param_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_16: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_221, mul_235);  mul_235 = None
        sum_dim_int_list_78: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_16, [0, 1]);  mul_tensor_16 = None
        sum_dim_int_list_79: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_221, [0, 1]);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_45: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_743, [1, 0])
        sum_dim_int_list_80: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_743, [0], True);  view_743 = None
        reshape_default_46: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_80, _shape_param_46);  sum_dim_int_list_80 = _shape_param_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_46: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_752, [1, 0])
        sum_dim_int_list_81: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_752, [0], True);  view_752 = None
        reshape_default_47: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_81, _shape_param_47);  sum_dim_int_list_81 = _shape_param_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_47: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_757, [1, 0])
        sum_dim_int_list_82: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_757, [0], True);  view_757 = None
        reshape_default_48: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_82, _shape_param_48);  sum_dim_int_list_82 = _shape_param_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_48: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_762, [1, 0])
        sum_dim_int_list_83: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_762, [0], True);  view_762 = None
        reshape_default_49: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_83, _shape_param_49);  sum_dim_int_list_83 = _shape_param_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_17: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_224, mul_228);  mul_228 = None
        sum_dim_int_list_84: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_17, [0, 1]);  mul_tensor_17 = None
        sum_dim_int_list_85: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_224, [0, 1]);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_49: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_765, [1, 0])
        sum_dim_int_list_86: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_765, [0], True);  view_765 = None
        reshape_default_50: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_86, _shape_param_50);  sum_dim_int_list_86 = _shape_param_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_50: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_768, [1, 0])
        sum_dim_int_list_87: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_768, [0], True);  view_768 = None
        reshape_default_51: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_87, _shape_param_51);  sum_dim_int_list_87 = _shape_param_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_18: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_227, mul_221);  mul_221 = None
        sum_dim_int_list_88: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_18, [0, 1]);  mul_tensor_18 = None
        sum_dim_int_list_89: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_227, [0, 1]);  add_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_51: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_771, [1, 0])
        sum_dim_int_list_90: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_771, [0], True);  view_771 = None
        reshape_default_52: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_90, _shape_param_52);  sum_dim_int_list_90 = _shape_param_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_52: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_780, [1, 0])
        sum_dim_int_list_91: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_780, [0], True);  view_780 = None
        reshape_default_53: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_91, _shape_param_53);  sum_dim_int_list_91 = _shape_param_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_53: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_785, [1, 0])
        sum_dim_int_list_92: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_785, [0], True);  view_785 = None
        reshape_default_54: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_92, _shape_param_54);  sum_dim_int_list_92 = _shape_param_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_54: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_790, [1, 0])
        sum_dim_int_list_93: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_790, [0], True);  view_790 = None
        reshape_default_55: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_93, _shape_param_55);  sum_dim_int_list_93 = _shape_param_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_19: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_230, mul_214);  mul_214 = None
        sum_dim_int_list_94: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_19, [0, 1]);  mul_tensor_19 = None
        sum_dim_int_list_95: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_230, [0, 1]);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_55: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_793, [1, 0])
        sum_dim_int_list_96: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_793, [0], True);  view_793 = None
        reshape_default_56: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_96, _shape_param_56);  sum_dim_int_list_96 = _shape_param_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_56: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_796, [1, 0])
        sum_dim_int_list_97: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_796, [0], True);  view_796 = None
        reshape_default_57: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_97, _shape_param_57);  sum_dim_int_list_97 = _shape_param_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_20: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_233, mul_207);  mul_207 = None
        sum_dim_int_list_98: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_20, [0, 1]);  mul_tensor_20 = None
        sum_dim_int_list_99: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_233, [0, 1]);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_57: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_799, [1, 0])
        sum_dim_int_list_100: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_799, [0], True);  view_799 = None
        reshape_default_58: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_100, _shape_param_58);  sum_dim_int_list_100 = _shape_param_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_58: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_808, [1, 0])
        sum_dim_int_list_101: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_808, [0], True);  view_808 = None
        reshape_default_59: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_101, _shape_param_59);  sum_dim_int_list_101 = _shape_param_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_59: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_813, [1, 0])
        sum_dim_int_list_102: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_813, [0], True);  view_813 = None
        reshape_default_60: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_102, _shape_param_60);  sum_dim_int_list_102 = _shape_param_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_60: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_818, [1, 0])
        sum_dim_int_list_103: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_818, [0], True);  view_818 = None
        reshape_default_61: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_103, _shape_param_61);  sum_dim_int_list_103 = _shape_param_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_21: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_236, mul_200);  mul_200 = None
        sum_dim_int_list_104: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_21, [0, 1]);  mul_tensor_21 = None
        sum_dim_int_list_105: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_236, [0, 1]);  add_236 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_61: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_821, [1, 0])
        sum_dim_int_list_106: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_821, [0], True);  view_821 = None
        reshape_default_62: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_106, _shape_param_62);  sum_dim_int_list_106 = _shape_param_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_62: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_824, [1, 0])
        sum_dim_int_list_107: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_824, [0], True);  view_824 = None
        reshape_default_63: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_107, _shape_param_63);  sum_dim_int_list_107 = _shape_param_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_22: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_239, mul_193);  mul_193 = None
        sum_dim_int_list_108: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_22, [0, 1]);  mul_tensor_22 = None
        sum_dim_int_list_109: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_239, [0, 1]);  add_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_63: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_827, [1, 0])
        sum_dim_int_list_110: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_827, [0], True);  view_827 = None
        reshape_default_64: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_110, _shape_param_64);  sum_dim_int_list_110 = _shape_param_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_64: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_836, [1, 0])
        sum_dim_int_list_111: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_836, [0], True);  view_836 = None
        reshape_default_65: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_111, _shape_param_65);  sum_dim_int_list_111 = _shape_param_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_65: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_841, [1, 0])
        sum_dim_int_list_112: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_841, [0], True);  view_841 = None
        reshape_default_66: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_112, _shape_param_66);  sum_dim_int_list_112 = _shape_param_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_66: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_846, [1, 0])
        sum_dim_int_list_113: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_846, [0], True);  view_846 = None
        reshape_default_67: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_113, _shape_param_67);  sum_dim_int_list_113 = _shape_param_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_23: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_242, mul_186);  mul_186 = None
        sum_dim_int_list_114: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_23, [0, 1]);  mul_tensor_23 = None
        sum_dim_int_list_115: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_242, [0, 1]);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_67: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_849, [1, 0])
        sum_dim_int_list_116: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_849, [0], True);  view_849 = None
        reshape_default_68: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_116, _shape_param_68);  sum_dim_int_list_116 = _shape_param_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_68: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_852, [1, 0])
        sum_dim_int_list_117: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_852, [0], True);  view_852 = None
        reshape_default_69: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_117, _shape_param_69);  sum_dim_int_list_117 = _shape_param_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_24: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_245, mul_179);  mul_179 = None
        sum_dim_int_list_118: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_24, [0, 1]);  mul_tensor_24 = None
        sum_dim_int_list_119: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_245, [0, 1]);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_69: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_855, [1, 0])
        sum_dim_int_list_120: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_855, [0], True);  view_855 = None
        reshape_default_70: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_120, _shape_param_70);  sum_dim_int_list_120 = _shape_param_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_70: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_864, [1, 0])
        sum_dim_int_list_121: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_864, [0], True);  view_864 = None
        reshape_default_71: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_121, _shape_param_71);  sum_dim_int_list_121 = _shape_param_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_71: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_869, [1, 0])
        sum_dim_int_list_122: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_869, [0], True);  view_869 = None
        reshape_default_72: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_122, _shape_param_72);  sum_dim_int_list_122 = _shape_param_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_72: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_874, [1, 0])
        sum_dim_int_list_123: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_874, [0], True);  view_874 = None
        reshape_default_73: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_123, _shape_param_73);  sum_dim_int_list_123 = _shape_param_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_25: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_248, mul_172);  mul_172 = None
        sum_dim_int_list_124: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_25, [0, 1]);  mul_tensor_25 = None
        sum_dim_int_list_125: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_248, [0, 1]);  add_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_73: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_877, [1, 0])
        sum_dim_int_list_126: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_877, [0], True);  view_877 = None
        reshape_default_74: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_126, _shape_param_74);  sum_dim_int_list_126 = _shape_param_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_74: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_880, [1, 0])
        sum_dim_int_list_127: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_880, [0], True);  view_880 = None
        reshape_default_75: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_127, _shape_param_75);  sum_dim_int_list_127 = _shape_param_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_26: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_251, mul_165);  mul_165 = None
        sum_dim_int_list_128: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_26, [0, 1]);  mul_tensor_26 = None
        sum_dim_int_list_129: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_251, [0, 1]);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_75: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_883, [1, 0])
        sum_dim_int_list_130: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_883, [0], True);  view_883 = None
        reshape_default_76: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_130, _shape_param_76);  sum_dim_int_list_130 = _shape_param_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_76: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_892, [1, 0])
        sum_dim_int_list_131: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_892, [0], True);  view_892 = None
        reshape_default_77: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_131, _shape_param_77);  sum_dim_int_list_131 = _shape_param_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_77: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_897, [1, 0])
        sum_dim_int_list_132: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_897, [0], True);  view_897 = None
        reshape_default_78: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_132, _shape_param_78);  sum_dim_int_list_132 = _shape_param_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_78: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_902, [1, 0])
        sum_dim_int_list_133: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_902, [0], True);  view_902 = None
        reshape_default_79: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_133, _shape_param_79);  sum_dim_int_list_133 = _shape_param_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_27: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_254, mul_158);  mul_158 = None
        sum_dim_int_list_134: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_27, [0, 1]);  mul_tensor_27 = None
        sum_dim_int_list_135: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_254, [0, 1]);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_79: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_905, [1, 0])
        sum_dim_int_list_136: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_905, [0], True);  view_905 = None
        reshape_default_80: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_136, _shape_param_80);  sum_dim_int_list_136 = _shape_param_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_80: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_908, [1, 0])
        sum_dim_int_list_137: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_908, [0], True);  view_908 = None
        reshape_default_81: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_137, _shape_param_81);  sum_dim_int_list_137 = _shape_param_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_28: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_257, mul_151);  mul_151 = None
        sum_dim_int_list_138: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_28, [0, 1]);  mul_tensor_28 = None
        sum_dim_int_list_139: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_257, [0, 1]);  add_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_81: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_911, [1, 0])
        sum_dim_int_list_140: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_911, [0], True);  view_911 = None
        reshape_default_82: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_140, _shape_param_82);  sum_dim_int_list_140 = _shape_param_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_82: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_920, [1, 0])
        sum_dim_int_list_141: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_920, [0], True);  view_920 = None
        reshape_default_83: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_141, _shape_param_83);  sum_dim_int_list_141 = _shape_param_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_83: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_925, [1, 0])
        sum_dim_int_list_142: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_925, [0], True);  view_925 = None
        reshape_default_84: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_142, _shape_param_84);  sum_dim_int_list_142 = _shape_param_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_84: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_930, [1, 0])
        sum_dim_int_list_143: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_930, [0], True);  view_930 = None
        reshape_default_85: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_143, _shape_param_85);  sum_dim_int_list_143 = _shape_param_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_29: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_260, mul_144);  mul_144 = None
        sum_dim_int_list_144: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_29, [0, 1]);  mul_tensor_29 = None
        sum_dim_int_list_145: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_260, [0, 1]);  add_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_85: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_933, [1, 0])
        sum_dim_int_list_146: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_933, [0], True);  view_933 = None
        reshape_default_86: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_146, _shape_param_86);  sum_dim_int_list_146 = _shape_param_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_86: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_936, [1, 0])
        sum_dim_int_list_147: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_936, [0], True);  view_936 = None
        reshape_default_87: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_147, _shape_param_87);  sum_dim_int_list_147 = _shape_param_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_30: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_263, mul_137);  mul_137 = None
        sum_dim_int_list_148: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_30, [0, 1]);  mul_tensor_30 = None
        sum_dim_int_list_149: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_263, [0, 1]);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_87: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_939, [1, 0])
        sum_dim_int_list_150: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_939, [0], True);  view_939 = None
        reshape_default_88: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_150, _shape_param_88);  sum_dim_int_list_150 = _shape_param_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_88: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_948, [1, 0])
        sum_dim_int_list_151: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_948, [0], True);  view_948 = None
        reshape_default_89: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_151, _shape_param_89);  sum_dim_int_list_151 = _shape_param_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_89: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_953, [1, 0])
        sum_dim_int_list_152: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_953, [0], True);  view_953 = None
        reshape_default_90: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_152, _shape_param_90);  sum_dim_int_list_152 = _shape_param_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_90: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_958, [1, 0])
        sum_dim_int_list_153: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_958, [0], True);  view_958 = None
        reshape_default_91: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_153, _shape_param_91);  sum_dim_int_list_153 = _shape_param_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_31: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_266, mul_130);  mul_130 = None
        sum_dim_int_list_154: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_31, [0, 1]);  mul_tensor_31 = None
        sum_dim_int_list_155: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_266, [0, 1]);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_91: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_961, [1, 0])
        sum_dim_int_list_156: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_961, [0], True);  view_961 = None
        reshape_default_92: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_156, _shape_param_92);  sum_dim_int_list_156 = _shape_param_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_92: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_964, [1, 0])
        sum_dim_int_list_157: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_964, [0], True);  view_964 = None
        reshape_default_93: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_157, _shape_param_93);  sum_dim_int_list_157 = _shape_param_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_32: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_269, mul_123);  mul_123 = None
        sum_dim_int_list_158: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_32, [0, 1]);  mul_tensor_32 = None
        sum_dim_int_list_159: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_269, [0, 1]);  add_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_93: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_967, [1, 0])
        sum_dim_int_list_160: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_967, [0], True);  view_967 = None
        reshape_default_94: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_160, _shape_param_94);  sum_dim_int_list_160 = _shape_param_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_94: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_976, [1, 0])
        sum_dim_int_list_161: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_976, [0], True);  view_976 = None
        reshape_default_95: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_161, _shape_param_95);  sum_dim_int_list_161 = _shape_param_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_95: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_981, [1, 0])
        sum_dim_int_list_162: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_981, [0], True);  view_981 = None
        reshape_default_96: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_162, _shape_param_96);  sum_dim_int_list_162 = _shape_param_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_96: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_986, [1, 0])
        sum_dim_int_list_163: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_986, [0], True);  view_986 = None
        reshape_default_97: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_163, _shape_param_97);  sum_dim_int_list_163 = _shape_param_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_33: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_272, mul_116);  mul_116 = None
        sum_dim_int_list_164: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_33, [0, 1]);  mul_tensor_33 = None
        sum_dim_int_list_165: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_272, [0, 1]);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_97: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_989, [1, 0])
        sum_dim_int_list_166: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_989, [0], True);  view_989 = None
        reshape_default_98: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_166, _shape_param_98);  sum_dim_int_list_166 = _shape_param_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_98: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_992, [1, 0])
        sum_dim_int_list_167: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_992, [0], True);  view_992 = None
        reshape_default_99: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_167, _shape_param_99);  sum_dim_int_list_167 = _shape_param_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_34: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_275, mul_109);  mul_109 = None
        sum_dim_int_list_168: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_34, [0, 1]);  mul_tensor_34 = None
        sum_dim_int_list_169: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_275, [0, 1]);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_99: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_995, [1, 0])
        sum_dim_int_list_170: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_995, [0], True);  view_995 = None
        reshape_default_100: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_170, _shape_param_100);  sum_dim_int_list_170 = _shape_param_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_100: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1004, [1, 0])
        sum_dim_int_list_171: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1004, [0], True);  view_1004 = None
        reshape_default_101: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_171, _shape_param_101);  sum_dim_int_list_171 = _shape_param_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_101: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1009, [1, 0])
        sum_dim_int_list_172: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1009, [0], True);  view_1009 = None
        reshape_default_102: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_172, _shape_param_102);  sum_dim_int_list_172 = _shape_param_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_102: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1014, [1, 0])
        sum_dim_int_list_173: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1014, [0], True);  view_1014 = None
        reshape_default_103: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_173, _shape_param_103);  sum_dim_int_list_173 = _shape_param_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_35: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_278, mul_102);  mul_102 = None
        sum_dim_int_list_174: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_35, [0, 1]);  mul_tensor_35 = None
        sum_dim_int_list_175: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_278, [0, 1]);  add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_103: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1017, [1, 0])
        sum_dim_int_list_176: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1017, [0], True);  view_1017 = None
        reshape_default_104: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_176, _shape_param_104);  sum_dim_int_list_176 = _shape_param_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_104: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1020, [1, 0])
        sum_dim_int_list_177: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1020, [0], True);  view_1020 = None
        reshape_default_105: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_177, _shape_param_105);  sum_dim_int_list_177 = _shape_param_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_36: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_281, mul_95);  mul_95 = None
        sum_dim_int_list_178: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_36, [0, 1]);  mul_tensor_36 = None
        sum_dim_int_list_179: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_281, [0, 1]);  add_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_105: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1023, [1, 0])
        sum_dim_int_list_180: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1023, [0], True);  view_1023 = None
        reshape_default_106: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_180, _shape_param_106);  sum_dim_int_list_180 = _shape_param_106 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_106: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1032, [1, 0])
        sum_dim_int_list_181: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1032, [0], True);  view_1032 = None
        reshape_default_107: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_181, _shape_param_107);  sum_dim_int_list_181 = _shape_param_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_107: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1037, [1, 0])
        sum_dim_int_list_182: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1037, [0], True);  view_1037 = None
        reshape_default_108: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_182, _shape_param_108);  sum_dim_int_list_182 = _shape_param_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_108: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1042, [1, 0])
        sum_dim_int_list_183: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1042, [0], True);  view_1042 = None
        reshape_default_109: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_183, _shape_param_109);  sum_dim_int_list_183 = _shape_param_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_37: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_284, mul_88);  mul_88 = None
        sum_dim_int_list_184: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_37, [0, 1]);  mul_tensor_37 = None
        sum_dim_int_list_185: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_284, [0, 1]);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_109: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1045, [1, 0])
        sum_dim_int_list_186: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1045, [0], True);  view_1045 = None
        reshape_default_110: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_186, _shape_param_110);  sum_dim_int_list_186 = _shape_param_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_110: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1048, [1, 0])
        sum_dim_int_list_187: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1048, [0], True);  view_1048 = None
        reshape_default_111: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_187, _shape_param_111);  sum_dim_int_list_187 = _shape_param_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_38: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_287, mul_81);  mul_81 = None
        sum_dim_int_list_188: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_38, [0, 1]);  mul_tensor_38 = None
        sum_dim_int_list_189: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_287, [0, 1]);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_111: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1051, [1, 0])
        sum_dim_int_list_190: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1051, [0], True);  view_1051 = None
        reshape_default_112: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_190, _shape_param_112);  sum_dim_int_list_190 = _shape_param_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_112: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1060, [1, 0])
        sum_dim_int_list_191: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1060, [0], True);  view_1060 = None
        reshape_default_113: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_191, _shape_param_113);  sum_dim_int_list_191 = _shape_param_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_113: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1065, [1, 0])
        sum_dim_int_list_192: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1065, [0], True);  view_1065 = None
        reshape_default_114: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_192, _shape_param_114);  sum_dim_int_list_192 = _shape_param_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_114: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1070, [1, 0])
        sum_dim_int_list_193: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1070, [0], True);  view_1070 = None
        reshape_default_115: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_193, _shape_param_115);  sum_dim_int_list_193 = _shape_param_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_39: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_290, mul_74);  mul_74 = None
        sum_dim_int_list_194: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_39, [0, 1]);  mul_tensor_39 = None
        sum_dim_int_list_195: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_290, [0, 1]);  add_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_115: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1073, [1, 0])
        sum_dim_int_list_196: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1073, [0], True);  view_1073 = None
        reshape_default_116: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_196, _shape_param_116);  sum_dim_int_list_196 = _shape_param_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_116: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1076, [1, 0])
        sum_dim_int_list_197: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1076, [0], True);  view_1076 = None
        reshape_default_117: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_197, _shape_param_117);  sum_dim_int_list_197 = _shape_param_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_40: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_293, mul_67);  mul_67 = None
        sum_dim_int_list_198: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_40, [0, 1]);  mul_tensor_40 = None
        sum_dim_int_list_199: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_293, [0, 1]);  add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_117: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1079, [1, 0])
        sum_dim_int_list_200: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1079, [0], True);  view_1079 = None
        reshape_default_118: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_200, _shape_param_118);  sum_dim_int_list_200 = _shape_param_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_118: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1088, [1, 0])
        sum_dim_int_list_201: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1088, [0], True);  view_1088 = None
        reshape_default_119: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_201, _shape_param_119);  sum_dim_int_list_201 = _shape_param_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_119: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1093, [1, 0])
        sum_dim_int_list_202: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1093, [0], True);  view_1093 = None
        reshape_default_120: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_202, _shape_param_120);  sum_dim_int_list_202 = _shape_param_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_120: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1098, [1, 0])
        sum_dim_int_list_203: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1098, [0], True);  view_1098 = None
        reshape_default_121: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_203, _shape_param_121);  sum_dim_int_list_203 = _shape_param_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_41: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_296, mul_60);  mul_60 = None
        sum_dim_int_list_204: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_41, [0, 1]);  mul_tensor_41 = None
        sum_dim_int_list_205: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_296, [0, 1]);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_121: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1101, [1, 0])
        sum_dim_int_list_206: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1101, [0], True);  view_1101 = None
        reshape_default_122: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_206, _shape_param_122);  sum_dim_int_list_206 = _shape_param_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_122: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1104, [1, 0])
        sum_dim_int_list_207: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1104, [0], True);  view_1104 = None
        reshape_default_123: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_207, _shape_param_123);  sum_dim_int_list_207 = _shape_param_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_42: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_299, mul_53);  mul_53 = None
        sum_dim_int_list_208: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_42, [0, 1]);  mul_tensor_42 = None
        sum_dim_int_list_209: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_299, [0, 1]);  add_299 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_123: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1107, [1, 0])
        sum_dim_int_list_210: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1107, [0], True);  view_1107 = None
        reshape_default_124: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_210, _shape_param_124);  sum_dim_int_list_210 = _shape_param_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_124: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1116, [1, 0])
        sum_dim_int_list_211: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1116, [0], True);  view_1116 = None
        reshape_default_125: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_211, _shape_param_125);  sum_dim_int_list_211 = _shape_param_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_125: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1121, [1, 0])
        sum_dim_int_list_212: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1121, [0], True);  view_1121 = None
        reshape_default_126: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_212, _shape_param_126);  sum_dim_int_list_212 = _shape_param_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_126: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1126, [1, 0])
        sum_dim_int_list_213: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1126, [0], True);  view_1126 = None
        reshape_default_127: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_213, _shape_param_127);  sum_dim_int_list_213 = _shape_param_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_43: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_302, mul_46);  mul_46 = None
        sum_dim_int_list_214: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_43, [0, 1]);  mul_tensor_43 = None
        sum_dim_int_list_215: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_302, [0, 1]);  add_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_127: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1129, [1, 0])
        sum_dim_int_list_216: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1129, [0], True);  view_1129 = None
        reshape_default_128: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_216, _shape_param_128);  sum_dim_int_list_216 = _shape_param_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_128: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1132, [1, 0])
        sum_dim_int_list_217: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1132, [0], True);  view_1132 = None
        reshape_default_129: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_217, _shape_param_129);  sum_dim_int_list_217 = _shape_param_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_44: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_305, mul_39);  mul_39 = None
        sum_dim_int_list_218: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_44, [0, 1]);  mul_tensor_44 = None
        sum_dim_int_list_219: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_305, [0, 1]);  add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_129: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1135, [1, 0])
        sum_dim_int_list_220: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1135, [0], True);  view_1135 = None
        reshape_default_130: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_220, _shape_param_130);  sum_dim_int_list_220 = _shape_param_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_130: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1144, [1, 0])
        sum_dim_int_list_221: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1144, [0], True);  view_1144 = None
        reshape_default_131: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_221, _shape_param_131);  sum_dim_int_list_221 = _shape_param_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_131: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1149, [1, 0])
        sum_dim_int_list_222: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1149, [0], True);  view_1149 = None
        reshape_default_132: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_222, _shape_param_132);  sum_dim_int_list_222 = _shape_param_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_132: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1154, [1, 0])
        sum_dim_int_list_223: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1154, [0], True);  view_1154 = None
        reshape_default_133: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_223, _shape_param_133);  sum_dim_int_list_223 = _shape_param_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_45: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_308, mul_32);  mul_32 = None
        sum_dim_int_list_224: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_45, [0, 1]);  mul_tensor_45 = None
        sum_dim_int_list_225: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_308, [0, 1]);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_133: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1157, [1, 0])
        sum_dim_int_list_226: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1157, [0], True);  view_1157 = None
        reshape_default_134: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_226, _shape_param_134);  sum_dim_int_list_226 = _shape_param_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_134: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1160, [1, 0])
        sum_dim_int_list_227: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1160, [0], True);  view_1160 = None
        reshape_default_135: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_227, _shape_param_135);  sum_dim_int_list_227 = _shape_param_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_46: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_311, mul_25);  mul_25 = None
        sum_dim_int_list_228: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_46, [0, 1]);  mul_tensor_46 = None
        sum_dim_int_list_229: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_311, [0, 1]);  add_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_135: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1163, [1, 0])
        sum_dim_int_list_230: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1163, [0], True);  view_1163 = None
        reshape_default_136: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_230, _shape_param_136);  sum_dim_int_list_230 = _shape_param_136 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_136: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1172, [1, 0])
        sum_dim_int_list_231: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1172, [0], True);  view_1172 = None
        reshape_default_137: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_231, _shape_param_137);  sum_dim_int_list_231 = _shape_param_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_137: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1177, [1, 0])
        sum_dim_int_list_232: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1177, [0], True);  view_1177 = None
        reshape_default_138: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_232, _shape_param_138);  sum_dim_int_list_232 = _shape_param_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_138: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1182, [1, 0])
        sum_dim_int_list_233: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1182, [0], True);  view_1182 = None
        reshape_default_139: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_233, _shape_param_139);  sum_dim_int_list_233 = _shape_param_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:411 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_47: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_314, mul_18);  mul_18 = None
        sum_dim_int_list_234: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_47, [0, 1]);  mul_tensor_47 = None
        sum_dim_int_list_235: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_314, [0, 1]);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:409 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_139: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1185, [1, 0])
        sum_dim_int_list_236: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1185, [0], True);  view_1185 = None
        reshape_default_140: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_236, _shape_param_140);  sum_dim_int_list_236 = _shape_param_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:394 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_140: "f32[6144, 4096]" = torch.ops.aten.permute.default(view_1188, [1, 0])
        sum_dim_int_list_237: "f32[1, 6144]" = torch.ops.aten.sum.dim_IntList(view_1188, [0], True);  view_1188 = None
        reshape_default_141: "f32[6144]" = torch.ops.aten.reshape.default(sum_dim_int_list_237, _shape_param_141);  sum_dim_int_list_237 = _shape_param_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:52 in forward, code: hidden_states = self.LayerNorm(hidden_states + input_tensor)
        mul_tensor_48: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_317, mul_11);  mul_11 = None
        sum_dim_int_list_238: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_48, [0, 1]);  mul_tensor_48 = None
        sum_dim_int_list_239: "f32[1536]" = torch.ops.aten.sum.dim_IntList(add_317, [0, 1]);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:50 in forward, code: hidden_states = self.dense(hidden_states)
        permute_default_141: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1191, [1, 0])
        sum_dim_int_list_240: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1191, [0], True);  view_1191 = None
        reshape_default_142: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_240, _shape_param_142);  sum_dim_int_list_240 = _shape_param_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:233 in forward, code: value_layer = self.transpose_for_scores(self.value_proj(hidden_states), self.num_attention_heads)
        permute_default_142: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1200, [1, 0])
        sum_dim_int_list_241: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1200, [0], True);  view_1200 = None
        reshape_default_143: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_241, _shape_param_143);  sum_dim_int_list_241 = _shape_param_143 = None
        reshape_default_144: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_286, _shape_param_144);  mm_286 = _shape_param_144 = None
        add_tensor: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(mul_1028, reshape_default_144);  mul_1028 = reshape_default_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:232 in forward, code: key_layer = self.transpose_for_scores(self.key_proj(hidden_states), self.num_attention_heads)
        permute_default_143: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1205, [1, 0])
        sum_dim_int_list_242: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1205, [0], True);  view_1205 = None
        reshape_default_145: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_242, _shape_param_145);  sum_dim_int_list_242 = _shape_param_145 = None
        reshape_default_146: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_288, _shape_param_146);  mm_288 = _shape_param_146 = None
        add_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_tensor, reshape_default_146);  add_tensor = reshape_default_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:231 in forward, code: query_layer = self.transpose_for_scores(self.query_proj(query_states), self.num_attention_heads)
        permute_default_144: "f32[1536, 4096]" = torch.ops.aten.permute.default(view_1210, [1, 0])
        sum_dim_int_list_243: "f32[1, 1536]" = torch.ops.aten.sum.dim_IntList(view_1210, [0], True);  view_1210 = None
        reshape_default_147: "f32[1536]" = torch.ops.aten.reshape.default(sum_dim_int_list_243, _shape_param_147);  sum_dim_int_list_243 = _shape_param_147 = None
        reshape_default_148: "f32[8, 512, 1536]" = torch.ops.aten.reshape.default(mm_290, _shape_param_148);  mm_290 = _shape_param_148 = None
        add_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(add_tensor_1, reshape_default_148);  add_tensor_1 = reshape_default_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:563 in forward, code: embeddings = self.dropout(embeddings)
        convert_element_type_default: "f32[8, 512, 1536]" = torch.ops.prims.convert_element_type.default(gt, torch.float32);  gt = None
        mul_tensor_49: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(convert_element_type_default, 1.1111111111111112);  convert_element_type_default = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:561 in forward, code: embeddings = embeddings * mask
        mul_tensor_50: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(add_tensor_2, mul_tensor_49);  add_tensor_2 = mul_tensor_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        mul_tensor_51: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_50, primals_5);  primals_5 = None
        mul_tensor_52: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_51, 1536)
        sum_dim_int_list_244: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_51, [2], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        add_tensor_3: "f32[8, 512, 1536]" = torch.ops.aten.add.Tensor(embedding, embedding_1);  embedding = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:552 in forward, code: embeddings = self.LayerNorm(embeddings)
        sub_tensor: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(add_tensor_3, getitem_1);  add_tensor_3 = getitem_1 = None
        mul_tensor_53: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(sub_tensor, rsqrt);  sub_tensor = None
        mul_tensor_54: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_51, mul_tensor_53);  mul_tensor_51 = None
        sum_dim_int_list_245: "f32[8, 512, 1]" = torch.ops.aten.sum.dim_IntList(mul_tensor_54, [2], True);  mul_tensor_54 = None
        mul_tensor_55: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_53, sum_dim_int_list_245);  sum_dim_int_list_245 = None
        sub_tensor_1: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(mul_tensor_52, sum_dim_int_list_244);  mul_tensor_52 = sum_dim_int_list_244 = None
        sub_tensor_2: "f32[8, 512, 1536]" = torch.ops.aten.sub.Tensor(sub_tensor_1, mul_tensor_55);  sub_tensor_1 = mul_tensor_55 = None
        div_tensor: "f32[8, 512, 1]" = torch.ops.aten.div.Tensor(rsqrt, 1536);  rsqrt = None
        mul_tensor_56: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(div_tensor, sub_tensor_2);  div_tensor = sub_tensor_2 = None
        mul_tensor_57: "f32[8, 512, 1536]" = torch.ops.aten.mul.Tensor(mul_tensor_50, mul_tensor_53);  mul_tensor_53 = None
        sum_dim_int_list_246: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_57, [0, 1]);  mul_tensor_57 = None
        sum_dim_int_list_247: "f32[1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_50, [0, 1]);  mul_tensor_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:544 in forward, code: embeddings = embeddings + position_embeddings
        sum_dim_int_list_248: "f32[1, 512, 1536]" = torch.ops.aten.sum.dim_IntList(mul_tensor_56, [0], True)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:538 in forward, code: position_embeddings = self.position_embeddings(position_ids.long())
        eq_scalar: "b8[1, 512]" = torch.ops.aten.eq.Scalar(primals_2, -1)
        unsqueeze_default: "b8[1, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar, -1);  eq_scalar = None
        where_self: "f32[1, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_default, full_default_74, sum_dim_int_list_248);  unsqueeze_default = sum_dim_int_list_248 = None
        full_default: "f32[512, 1536]" = torch.ops.aten.full.default([512, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default: "f32[512, 1536]" = torch.ops.aten.index_put.default(full_default, [primals_2], where_self, True);  full_default = primals_2 = where_self = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/deberta_v2/modeling_deberta_v2.py:535 in forward, code: inputs_embeds = self.word_embeddings(input_ids)
        eq_scalar_1: "b8[8, 512]" = torch.ops.aten.eq.Scalar(primals_1, 0)
        unsqueeze_default_1: "b8[8, 512, 1]" = torch.ops.aten.unsqueeze.default(eq_scalar_1, -1);  eq_scalar_1 = None
        where_self_1: "f32[8, 512, 1536]" = torch.ops.aten.where.self(unsqueeze_default_1, full_default_74, mul_tensor_56);  unsqueeze_default_1 = full_default_74 = mul_tensor_56 = None
        full_default_75: "f32[128100, 1536]" = torch.ops.aten.full.default([128100, 1536], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        index_put_default_1: "f32[128100, 1536]" = torch.ops.aten.index_put.default(full_default_75, [primals_1], where_self_1, True);  full_default_75 = primals_1 = where_self_1 = None
        add_tensor_4: "f32[128100, 1536]" = torch.ops.aten.add.Tensor(mm_1, index_put_default_1);  mm_1 = index_put_default_1 = None
        return (reshape_default, sum_dim_int_list_1, sum_dim_int_list_2, permute_default, reshape_default_1, sum_dim_int_list_4, sum_dim_int_list_5, permute_default_1, reshape_default_2, permute_default_2, reshape_default_3, sum_dim_int_list_8, sum_dim_int_list_9, permute_default_3, reshape_default_4, permute_default_4, reshape_default_5, permute_default_5, reshape_default_6, permute_default_6, reshape_default_7, sum_dim_int_list_14, sum_dim_int_list_15, permute_default_7, reshape_default_8, permute_default_8, reshape_default_9, sum_dim_int_list_18, sum_dim_int_list_19, permute_default_9, reshape_default_10, permute_default_10, reshape_default_11, permute_default_11, reshape_default_12, permute_default_12, reshape_default_13, sum_dim_int_list_24, sum_dim_int_list_25, permute_default_13, reshape_default_14, permute_default_14, reshape_default_15, sum_dim_int_list_28, sum_dim_int_list_29, permute_default_15, reshape_default_16, permute_default_16, reshape_default_17, permute_default_17, reshape_default_18, permute_default_18, reshape_default_19, sum_dim_int_list_34, sum_dim_int_list_35, permute_default_19, reshape_default_20, permute_default_20, reshape_default_21, sum_dim_int_list_38, sum_dim_int_list_39, permute_default_21, reshape_default_22, permute_default_22, reshape_default_23, permute_default_23, reshape_default_24, permute_default_24, reshape_default_25, sum_dim_int_list_44, sum_dim_int_list_45, permute_default_25, reshape_default_26, permute_default_26, reshape_default_27, sum_dim_int_list_48, sum_dim_int_list_49, permute_default_27, reshape_default_28, permute_default_28, reshape_default_29, permute_default_29, reshape_default_30, permute_default_30, reshape_default_31, sum_dim_int_list_54, sum_dim_int_list_55, permute_default_31, reshape_default_32, permute_default_32, reshape_default_33, sum_dim_int_list_58, sum_dim_int_list_59, permute_default_33, reshape_default_34, permute_default_34, reshape_default_35, permute_default_35, reshape_default_36, permute_default_36, reshape_default_37, sum_dim_int_list_64, sum_dim_int_list_65, permute_default_37, reshape_default_38, permute_default_38, reshape_default_39, sum_dim_int_list_68, sum_dim_int_list_69, permute_default_39, reshape_default_40, permute_default_40, reshape_default_41, permute_default_41, reshape_default_42, permute_default_42, reshape_default_43, sum_dim_int_list_74, sum_dim_int_list_75, permute_default_43, reshape_default_44, permute_default_44, reshape_default_45, sum_dim_int_list_78, sum_dim_int_list_79, permute_default_45, reshape_default_46, permute_default_46, reshape_default_47, permute_default_47, reshape_default_48, permute_default_48, reshape_default_49, sum_dim_int_list_84, sum_dim_int_list_85, permute_default_49, reshape_default_50, permute_default_50, reshape_default_51, sum_dim_int_list_88, sum_dim_int_list_89, permute_default_51, reshape_default_52, permute_default_52, reshape_default_53, permute_default_53, reshape_default_54, permute_default_54, reshape_default_55, sum_dim_int_list_94, sum_dim_int_list_95, permute_default_55, reshape_default_56, permute_default_56, reshape_default_57, sum_dim_int_list_98, sum_dim_int_list_99, permute_default_57, reshape_default_58, permute_default_58, reshape_default_59, permute_default_59, reshape_default_60, permute_default_60, reshape_default_61, sum_dim_int_list_104, sum_dim_int_list_105, permute_default_61, reshape_default_62, permute_default_62, reshape_default_63, sum_dim_int_list_108, sum_dim_int_list_109, permute_default_63, reshape_default_64, permute_default_64, reshape_default_65, permute_default_65, reshape_default_66, permute_default_66, reshape_default_67, sum_dim_int_list_114, sum_dim_int_list_115, permute_default_67, reshape_default_68, permute_default_68, reshape_default_69, sum_dim_int_list_118, sum_dim_int_list_119, permute_default_69, reshape_default_70, permute_default_70, reshape_default_71, permute_default_71, reshape_default_72, permute_default_72, reshape_default_73, sum_dim_int_list_124, sum_dim_int_list_125, permute_default_73, reshape_default_74, permute_default_74, reshape_default_75, sum_dim_int_list_128, sum_dim_int_list_129, permute_default_75, reshape_default_76, permute_default_76, reshape_default_77, permute_default_77, reshape_default_78, permute_default_78, reshape_default_79, sum_dim_int_list_134, sum_dim_int_list_135, permute_default_79, reshape_default_80, permute_default_80, reshape_default_81, sum_dim_int_list_138, sum_dim_int_list_139, permute_default_81, reshape_default_82, permute_default_82, reshape_default_83, permute_default_83, reshape_default_84, permute_default_84, reshape_default_85, sum_dim_int_list_144, sum_dim_int_list_145, permute_default_85, reshape_default_86, permute_default_86, reshape_default_87, sum_dim_int_list_148, sum_dim_int_list_149, permute_default_87, reshape_default_88, permute_default_88, reshape_default_89, permute_default_89, reshape_default_90, permute_default_90, reshape_default_91, sum_dim_int_list_154, sum_dim_int_list_155, permute_default_91, reshape_default_92, permute_default_92, reshape_default_93, sum_dim_int_list_158, sum_dim_int_list_159, permute_default_93, reshape_default_94, permute_default_94, reshape_default_95, permute_default_95, reshape_default_96, permute_default_96, reshape_default_97, sum_dim_int_list_164, sum_dim_int_list_165, permute_default_97, reshape_default_98, permute_default_98, reshape_default_99, sum_dim_int_list_168, sum_dim_int_list_169, permute_default_99, reshape_default_100, permute_default_100, reshape_default_101, permute_default_101, reshape_default_102, permute_default_102, reshape_default_103, sum_dim_int_list_174, sum_dim_int_list_175, permute_default_103, reshape_default_104, permute_default_104, reshape_default_105, sum_dim_int_list_178, sum_dim_int_list_179, permute_default_105, reshape_default_106, permute_default_106, reshape_default_107, permute_default_107, reshape_default_108, permute_default_108, reshape_default_109, sum_dim_int_list_184, sum_dim_int_list_185, permute_default_109, reshape_default_110, permute_default_110, reshape_default_111, sum_dim_int_list_188, sum_dim_int_list_189, permute_default_111, reshape_default_112, permute_default_112, reshape_default_113, permute_default_113, reshape_default_114, permute_default_114, reshape_default_115, sum_dim_int_list_194, sum_dim_int_list_195, permute_default_115, reshape_default_116, permute_default_116, reshape_default_117, sum_dim_int_list_198, sum_dim_int_list_199, permute_default_117, reshape_default_118, permute_default_118, reshape_default_119, permute_default_119, reshape_default_120, permute_default_120, reshape_default_121, sum_dim_int_list_204, sum_dim_int_list_205, permute_default_121, reshape_default_122, permute_default_122, reshape_default_123, sum_dim_int_list_208, sum_dim_int_list_209, permute_default_123, reshape_default_124, permute_default_124, reshape_default_125, permute_default_125, reshape_default_126, permute_default_126, reshape_default_127, sum_dim_int_list_214, sum_dim_int_list_215, permute_default_127, reshape_default_128, permute_default_128, reshape_default_129, sum_dim_int_list_218, sum_dim_int_list_219, permute_default_129, reshape_default_130, permute_default_130, reshape_default_131, permute_default_131, reshape_default_132, permute_default_132, reshape_default_133, sum_dim_int_list_224, sum_dim_int_list_225, permute_default_133, reshape_default_134, permute_default_134, reshape_default_135, sum_dim_int_list_228, sum_dim_int_list_229, permute_default_135, reshape_default_136, permute_default_136, reshape_default_137, permute_default_137, reshape_default_138, permute_default_138, reshape_default_139, sum_dim_int_list_234, sum_dim_int_list_235, permute_default_139, reshape_default_140, permute_default_140, reshape_default_141, sum_dim_int_list_238, sum_dim_int_list_239, permute_default_141, reshape_default_142, permute_default_142, reshape_default_143, permute_default_143, reshape_default_145, permute_default_144, reshape_default_147, sum_dim_int_list_246, sum_dim_int_list_247, index_put_default, add_tensor_4)


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

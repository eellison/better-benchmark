"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: 801cd8da00d4
Shape hash: 5d62ee3f
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_prelude import *  # noqa: F401,F403
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_393: "f32[32128, 512]", getitem_394: "f32[512, 512]", getitem_395: "f32[512, 512]", getitem_396: "f32[512, 512]", getitem_397: "f32[512, 512]", getitem_398: "f32[32, 8]", getitem_399: "f32[512]", getitem_400: "f32[2048, 512]", getitem_401: "f32[512, 2048]", getitem_402: "f32[512]", getitem_403: "f32[512, 512]", getitem_404: "f32[512, 512]", getitem_405: "f32[512, 512]", getitem_406: "f32[512, 512]", getitem_407: "f32[512]", getitem_408: "f32[2048, 512]", getitem_409: "f32[512, 2048]", getitem_410: "f32[512]", getitem_411: "f32[512, 512]", getitem_412: "f32[512, 512]", getitem_413: "f32[512, 512]", getitem_414: "f32[512, 512]", getitem_415: "f32[512]", getitem_416: "f32[2048, 512]", getitem_417: "f32[512, 2048]", getitem_418: "f32[512]", getitem_419: "f32[512, 512]", getitem_420: "f32[512, 512]", getitem_421: "f32[512, 512]", getitem_422: "f32[512, 512]", getitem_423: "f32[512]", getitem_424: "f32[2048, 512]", getitem_425: "f32[512, 2048]", getitem_426: "f32[512]", getitem_427: "f32[512, 512]", getitem_428: "f32[512, 512]", getitem_429: "f32[512, 512]", getitem_430: "f32[512, 512]", getitem_431: "f32[512]", getitem_432: "f32[2048, 512]", getitem_433: "f32[512, 2048]", getitem_434: "f32[512]", getitem_435: "f32[512, 512]", getitem_436: "f32[512, 512]", getitem_437: "f32[512, 512]", getitem_438: "f32[512, 512]", getitem_439: "f32[512]", getitem_440: "f32[2048, 512]", getitem_441: "f32[512, 2048]", getitem_442: "f32[512]", getitem_443: "f32[512]", getitem_444: "f32[512, 512]", getitem_445: "f32[512, 512]", getitem_446: "f32[512, 512]", getitem_447: "f32[512, 512]", getitem_448: "f32[32, 8]", getitem_449: "f32[512]", getitem_450: "f32[512, 512]", getitem_451: "f32[512, 512]", getitem_452: "f32[512, 512]", getitem_453: "f32[512, 512]", getitem_454: "f32[512]", getitem_455: "f32[2048, 512]", getitem_456: "f32[512, 2048]", getitem_457: "f32[512]", getitem_458: "f32[512, 512]", getitem_459: "f32[512, 512]", getitem_460: "f32[512, 512]", getitem_461: "f32[512, 512]", getitem_462: "f32[512]", getitem_463: "f32[512, 512]", getitem_464: "f32[512, 512]", getitem_465: "f32[512, 512]", getitem_466: "f32[512, 512]", getitem_467: "f32[512]", getitem_468: "f32[2048, 512]", getitem_469: "f32[512, 2048]", getitem_470: "f32[512]", getitem_471: "f32[512, 512]", getitem_472: "f32[512, 512]", getitem_473: "f32[512, 512]", getitem_474: "f32[512, 512]", getitem_475: "f32[512]", getitem_476: "f32[512, 512]", getitem_477: "f32[512, 512]", getitem_478: "f32[512, 512]", getitem_479: "f32[512, 512]", getitem_480: "f32[512]", getitem_481: "f32[2048, 512]", getitem_482: "f32[512, 2048]", getitem_483: "f32[512]", getitem_484: "f32[512, 512]", getitem_485: "f32[512, 512]", getitem_486: "f32[512, 512]", getitem_487: "f32[512, 512]", getitem_488: "f32[512]", getitem_489: "f32[512, 512]", getitem_490: "f32[512, 512]", getitem_491: "f32[512, 512]", getitem_492: "f32[512, 512]", getitem_493: "f32[512]", getitem_494: "f32[2048, 512]", getitem_495: "f32[512, 2048]", getitem_496: "f32[512]", getitem_497: "f32[512, 512]", getitem_498: "f32[512, 512]", getitem_499: "f32[512, 512]", getitem_500: "f32[512, 512]", getitem_501: "f32[512]", getitem_502: "f32[512, 512]", getitem_503: "f32[512, 512]", getitem_504: "f32[512, 512]", getitem_505: "f32[512, 512]", getitem_506: "f32[512]", getitem_507: "f32[2048, 512]", getitem_508: "f32[512, 2048]", getitem_509: "f32[512]", getitem_510: "f32[512, 512]", getitem_511: "f32[512, 512]", getitem_512: "f32[512, 512]", getitem_513: "f32[512, 512]", getitem_514: "f32[512]", getitem_515: "f32[512, 512]", getitem_516: "f32[512, 512]", getitem_517: "f32[512, 512]", getitem_518: "f32[512, 512]", getitem_519: "f32[512]", getitem_520: "f32[2048, 512]", getitem_521: "f32[512, 2048]", getitem_522: "f32[512]", getitem_523: "f32[512]", arg524_1: "f32[32128, 512]", arg525_1: "f32[512, 512]", arg526_1: "f32[512, 512]", arg527_1: "f32[512, 512]", arg528_1: "f32[512, 512]", arg529_1: "f32[32, 8]", arg530_1: "f32[512]", arg531_1: "f32[2048, 512]", arg532_1: "f32[512, 2048]", arg533_1: "f32[512]", arg534_1: "f32[512, 512]", arg535_1: "f32[512, 512]", arg536_1: "f32[512, 512]", arg537_1: "f32[512, 512]", arg538_1: "f32[512]", arg539_1: "f32[2048, 512]", arg540_1: "f32[512, 2048]", arg541_1: "f32[512]", arg542_1: "f32[512, 512]", arg543_1: "f32[512, 512]", arg544_1: "f32[512, 512]", arg545_1: "f32[512, 512]", arg546_1: "f32[512]", arg547_1: "f32[2048, 512]", arg548_1: "f32[512, 2048]", arg549_1: "f32[512]", arg550_1: "f32[512, 512]", arg551_1: "f32[512, 512]", arg552_1: "f32[512, 512]", arg553_1: "f32[512, 512]", arg554_1: "f32[512]", arg555_1: "f32[2048, 512]", arg556_1: "f32[512, 2048]", arg557_1: "f32[512]", arg558_1: "f32[512, 512]", arg559_1: "f32[512, 512]", arg560_1: "f32[512, 512]", arg561_1: "f32[512, 512]", arg562_1: "f32[512]", arg563_1: "f32[2048, 512]", arg564_1: "f32[512, 2048]", arg565_1: "f32[512]", arg566_1: "f32[512, 512]", arg567_1: "f32[512, 512]", arg568_1: "f32[512, 512]", arg569_1: "f32[512, 512]", arg570_1: "f32[512]", arg571_1: "f32[2048, 512]", arg572_1: "f32[512, 2048]", arg573_1: "f32[512]", arg574_1: "f32[512]", arg575_1: "f32[512, 512]", arg576_1: "f32[512, 512]", arg577_1: "f32[512, 512]", arg578_1: "f32[512, 512]", arg579_1: "f32[32, 8]", arg580_1: "f32[512]", arg581_1: "f32[512, 512]", arg582_1: "f32[512, 512]", arg583_1: "f32[512, 512]", arg584_1: "f32[512, 512]", arg585_1: "f32[512]", arg586_1: "f32[2048, 512]", arg587_1: "f32[512, 2048]", arg588_1: "f32[512]", arg589_1: "f32[512, 512]", arg590_1: "f32[512, 512]", arg591_1: "f32[512, 512]", arg592_1: "f32[512, 512]", arg593_1: "f32[512]", arg594_1: "f32[512, 512]", arg595_1: "f32[512, 512]", arg596_1: "f32[512, 512]", arg597_1: "f32[512, 512]", arg598_1: "f32[512]", arg599_1: "f32[2048, 512]", arg600_1: "f32[512, 2048]", arg601_1: "f32[512]", arg602_1: "f32[512, 512]", arg603_1: "f32[512, 512]", arg604_1: "f32[512, 512]", arg605_1: "f32[512, 512]", arg606_1: "f32[512]", arg607_1: "f32[512, 512]", arg608_1: "f32[512, 512]", arg609_1: "f32[512, 512]", arg610_1: "f32[512, 512]", arg611_1: "f32[512]", arg612_1: "f32[2048, 512]", arg613_1: "f32[512, 2048]", arg614_1: "f32[512]", arg615_1: "f32[512, 512]", arg616_1: "f32[512, 512]", arg617_1: "f32[512, 512]", arg618_1: "f32[512, 512]", arg619_1: "f32[512]", arg620_1: "f32[512, 512]", arg621_1: "f32[512, 512]", arg622_1: "f32[512, 512]", arg623_1: "f32[512, 512]", arg624_1: "f32[512]", arg625_1: "f32[2048, 512]", arg626_1: "f32[512, 2048]", arg627_1: "f32[512]", arg628_1: "f32[512, 512]", arg629_1: "f32[512, 512]", arg630_1: "f32[512, 512]", arg631_1: "f32[512, 512]", arg632_1: "f32[512]", arg633_1: "f32[512, 512]", arg634_1: "f32[512, 512]", arg635_1: "f32[512, 512]", arg636_1: "f32[512, 512]", arg637_1: "f32[512]", arg638_1: "f32[2048, 512]", arg639_1: "f32[512, 2048]", arg640_1: "f32[512]", arg641_1: "f32[512, 512]", arg642_1: "f32[512, 512]", arg643_1: "f32[512, 512]", arg644_1: "f32[512, 512]", arg645_1: "f32[512]", arg646_1: "f32[512, 512]", arg647_1: "f32[512, 512]", arg648_1: "f32[512, 512]", arg649_1: "f32[512, 512]", arg650_1: "f32[512]", arg651_1: "f32[2048, 512]", arg652_1: "f32[512, 2048]", arg653_1: "f32[512]", arg654_1: "f32[512]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_10: "f32[]", getitem_11: "f32[]", getitem_12: "f32[]", getitem_13: "f32[]", getitem_14: "f32[]", getitem_15: "f32[]", getitem_16: "f32[]", getitem_17: "f32[]", getitem_18: "f32[]", getitem_19: "f32[]", getitem_20: "f32[]", getitem_21: "f32[]", getitem_22: "f32[]", getitem_23: "f32[]", getitem_24: "f32[]", getitem_25: "f32[]", getitem_26: "f32[]", getitem_27: "f32[]", getitem_28: "f32[]", getitem_29: "f32[]", getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]", getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]", getitem_130: "f32[]", getitem_1048: "f32[]", getitem_1049: "f32[]", getitem_1050: "f32[]", getitem_1051: "f32[]", getitem_1052: "f32[]", getitem_1053: "f32[]", getitem_1054: "f32[]", getitem_1055: "f32[]", getitem_1056: "f32[]", getitem_1057: "f32[]", getitem_1058: "f32[]", getitem_1059: "f32[]", getitem_1060: "f32[]", getitem_1061: "f32[]", getitem_1062: "f32[]", getitem_1063: "f32[]", getitem_1064: "f32[]", getitem_1065: "f32[]", getitem_1066: "f32[]", getitem_1067: "f32[]", getitem_1068: "f32[]", getitem_1069: "f32[]", getitem_1070: "f32[]", getitem_1071: "f32[]", getitem_1072: "f32[]", getitem_1073: "f32[]", getitem_1074: "f32[]", getitem_1075: "f32[]", getitem_1076: "f32[]", getitem_1077: "f32[]", getitem_1078: "f32[]", getitem_1079: "f32[]", getitem_1080: "f32[]", getitem_1081: "f32[]", getitem_1082: "f32[]", getitem_1083: "f32[]", getitem_1084: "f32[]", getitem_1085: "f32[]", getitem_1086: "f32[]", getitem_1087: "f32[]", getitem_1088: "f32[]", getitem_1089: "f32[]", getitem_1090: "f32[]", getitem_1091: "f32[]", getitem_1092: "f32[]", getitem_1093: "f32[]", getitem_1094: "f32[]", getitem_1095: "f32[]", getitem_1096: "f32[]", getitem_1097: "f32[]", getitem_1098: "f32[]", getitem_1099: "f32[]", getitem_1100: "f32[]", getitem_1101: "f32[]", getitem_1102: "f32[]", getitem_1103: "f32[]", getitem_1104: "f32[]", getitem_1105: "f32[]", getitem_1106: "f32[]", getitem_1107: "f32[]", getitem_1108: "f32[]", getitem_1109: "f32[]", getitem_1110: "f32[]", getitem_1111: "f32[]", getitem_1112: "f32[]", getitem_1113: "f32[]", getitem_1114: "f32[]", getitem_1115: "f32[]", getitem_1116: "f32[]", getitem_1117: "f32[]", getitem_1118: "f32[]", getitem_1119: "f32[]", getitem_1120: "f32[]", getitem_1121: "f32[]", getitem_1122: "f32[]", getitem_1123: "f32[]", getitem_1124: "f32[]", getitem_1125: "f32[]", getitem_1126: "f32[]", getitem_1127: "f32[]", getitem_1128: "f32[]", getitem_1129: "f32[]", getitem_1130: "f32[]", getitem_1131: "f32[]", getitem_1132: "f32[]", getitem_1133: "f32[]", getitem_1134: "f32[]", getitem_1135: "f32[]", getitem_1136: "f32[]", getitem_1137: "f32[]", getitem_1138: "f32[]", getitem_1139: "f32[]", getitem_1140: "f32[]", getitem_1141: "f32[]", getitem_1142: "f32[]", getitem_1143: "f32[]", getitem_1144: "f32[]", getitem_1145: "f32[]", getitem_1146: "f32[]", getitem_1147: "f32[]", getitem_1148: "f32[]", getitem_1149: "f32[]", getitem_1150: "f32[]", getitem_1151: "f32[]", getitem_1152: "f32[]", getitem_1153: "f32[]", getitem_1154: "f32[]", getitem_1155: "f32[]", getitem_1156: "f32[]", getitem_1157: "f32[]", getitem_1158: "f32[]", getitem_1159: "f32[]", getitem_1160: "f32[]", getitem_1161: "f32[]", getitem_1162: "f32[]", getitem_1163: "f32[]", getitem_1164: "f32[]", getitem_1165: "f32[]", getitem_1166: "f32[]", getitem_1167: "f32[]", getitem_1168: "f32[]", getitem_1169: "f32[]", getitem_1170: "f32[]", getitem_1171: "f32[]", getitem_1172: "f32[]", getitem_1173: "f32[]", getitem_1174: "f32[]", getitem_1175: "f32[]", getitem_1176: "f32[]", getitem_1177: "f32[]", getitem_1178: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523], [arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1], [arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1], 0.0010000000000000009);  getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = getitem_409 = getitem_410 = getitem_411 = getitem_412 = getitem_413 = getitem_414 = getitem_415 = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = getitem_433 = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = getitem_439 = getitem_440 = getitem_441 = getitem_442 = getitem_443 = getitem_444 = getitem_445 = getitem_446 = getitem_447 = getitem_448 = getitem_449 = getitem_450 = getitem_451 = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = getitem_457 = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_465 = getitem_466 = getitem_467 = getitem_468 = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = getitem_474 = getitem_475 = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = getitem_481 = getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = getitem_487 = getitem_488 = getitem_489 = getitem_490 = getitem_491 = getitem_492 = getitem_493 = getitem_494 = getitem_495 = getitem_496 = getitem_497 = getitem_498 = getitem_499 = getitem_500 = getitem_501 = getitem_502 = getitem_503 = getitem_504 = getitem_505 = getitem_506 = getitem_507 = getitem_508 = getitem_509 = getitem_510 = getitem_511 = getitem_512 = getitem_513 = getitem_514 = getitem_515 = getitem_516 = getitem_517 = getitem_518 = getitem_519 = getitem_520 = getitem_521 = getitem_522 = getitem_523 = arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = arg540_1 = arg541_1 = arg542_1 = arg543_1 = arg544_1 = arg545_1 = arg546_1 = arg547_1 = arg548_1 = arg549_1 = arg550_1 = arg551_1 = arg552_1 = arg553_1 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = arg558_1 = arg559_1 = arg560_1 = arg561_1 = arg562_1 = arg563_1 = arg564_1 = arg565_1 = arg566_1 = arg567_1 = arg568_1 = arg569_1 = arg570_1 = arg571_1 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = arg576_1 = arg577_1 = arg578_1 = arg579_1 = arg580_1 = arg581_1 = arg582_1 = arg583_1 = arg584_1 = arg585_1 = arg586_1 = arg587_1 = arg588_1 = arg589_1 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = arg594_1 = arg595_1 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = None
        getitem: "f32[32128, 512]" = _foreach_addcmul_scalar[0]
        getitem_524: "f32[512, 512]" = _foreach_addcmul_scalar[1]
        getitem_525: "f32[512, 512]" = _foreach_addcmul_scalar[2]
        getitem_526: "f32[512, 512]" = _foreach_addcmul_scalar[3]
        getitem_527: "f32[512, 512]" = _foreach_addcmul_scalar[4]
        getitem_528: "f32[32, 8]" = _foreach_addcmul_scalar[5]
        getitem_529: "f32[512]" = _foreach_addcmul_scalar[6]
        getitem_530: "f32[2048, 512]" = _foreach_addcmul_scalar[7]
        getitem_531: "f32[512, 2048]" = _foreach_addcmul_scalar[8]
        getitem_532: "f32[512]" = _foreach_addcmul_scalar[9]
        getitem_533: "f32[512, 512]" = _foreach_addcmul_scalar[10]
        getitem_534: "f32[512, 512]" = _foreach_addcmul_scalar[11]
        getitem_535: "f32[512, 512]" = _foreach_addcmul_scalar[12]
        getitem_536: "f32[512, 512]" = _foreach_addcmul_scalar[13]
        getitem_537: "f32[512]" = _foreach_addcmul_scalar[14]
        getitem_538: "f32[2048, 512]" = _foreach_addcmul_scalar[15]
        getitem_539: "f32[512, 2048]" = _foreach_addcmul_scalar[16]
        getitem_540: "f32[512]" = _foreach_addcmul_scalar[17]
        getitem_541: "f32[512, 512]" = _foreach_addcmul_scalar[18]
        getitem_542: "f32[512, 512]" = _foreach_addcmul_scalar[19]
        getitem_543: "f32[512, 512]" = _foreach_addcmul_scalar[20]
        getitem_544: "f32[512, 512]" = _foreach_addcmul_scalar[21]
        getitem_545: "f32[512]" = _foreach_addcmul_scalar[22]
        getitem_546: "f32[2048, 512]" = _foreach_addcmul_scalar[23]
        getitem_547: "f32[512, 2048]" = _foreach_addcmul_scalar[24]
        getitem_548: "f32[512]" = _foreach_addcmul_scalar[25]
        getitem_549: "f32[512, 512]" = _foreach_addcmul_scalar[26]
        getitem_550: "f32[512, 512]" = _foreach_addcmul_scalar[27]
        getitem_551: "f32[512, 512]" = _foreach_addcmul_scalar[28]
        getitem_552: "f32[512, 512]" = _foreach_addcmul_scalar[29]
        getitem_553: "f32[512]" = _foreach_addcmul_scalar[30]
        getitem_554: "f32[2048, 512]" = _foreach_addcmul_scalar[31]
        getitem_555: "f32[512, 2048]" = _foreach_addcmul_scalar[32]
        getitem_556: "f32[512]" = _foreach_addcmul_scalar[33]
        getitem_557: "f32[512, 512]" = _foreach_addcmul_scalar[34]
        getitem_558: "f32[512, 512]" = _foreach_addcmul_scalar[35]
        getitem_559: "f32[512, 512]" = _foreach_addcmul_scalar[36]
        getitem_560: "f32[512, 512]" = _foreach_addcmul_scalar[37]
        getitem_561: "f32[512]" = _foreach_addcmul_scalar[38]
        getitem_562: "f32[2048, 512]" = _foreach_addcmul_scalar[39]
        getitem_563: "f32[512, 2048]" = _foreach_addcmul_scalar[40]
        getitem_564: "f32[512]" = _foreach_addcmul_scalar[41]
        getitem_565: "f32[512, 512]" = _foreach_addcmul_scalar[42]
        getitem_566: "f32[512, 512]" = _foreach_addcmul_scalar[43]
        getitem_567: "f32[512, 512]" = _foreach_addcmul_scalar[44]
        getitem_568: "f32[512, 512]" = _foreach_addcmul_scalar[45]
        getitem_569: "f32[512]" = _foreach_addcmul_scalar[46]
        getitem_570: "f32[2048, 512]" = _foreach_addcmul_scalar[47]
        getitem_571: "f32[512, 2048]" = _foreach_addcmul_scalar[48]
        getitem_572: "f32[512]" = _foreach_addcmul_scalar[49]
        getitem_573: "f32[512]" = _foreach_addcmul_scalar[50]
        getitem_574: "f32[512, 512]" = _foreach_addcmul_scalar[51]
        getitem_575: "f32[512, 512]" = _foreach_addcmul_scalar[52]
        getitem_576: "f32[512, 512]" = _foreach_addcmul_scalar[53]
        getitem_577: "f32[512, 512]" = _foreach_addcmul_scalar[54]
        getitem_578: "f32[32, 8]" = _foreach_addcmul_scalar[55]
        getitem_579: "f32[512]" = _foreach_addcmul_scalar[56]
        getitem_580: "f32[512, 512]" = _foreach_addcmul_scalar[57]
        getitem_581: "f32[512, 512]" = _foreach_addcmul_scalar[58]
        getitem_582: "f32[512, 512]" = _foreach_addcmul_scalar[59]
        getitem_583: "f32[512, 512]" = _foreach_addcmul_scalar[60]
        getitem_584: "f32[512]" = _foreach_addcmul_scalar[61]
        getitem_585: "f32[2048, 512]" = _foreach_addcmul_scalar[62]
        getitem_586: "f32[512, 2048]" = _foreach_addcmul_scalar[63]
        getitem_587: "f32[512]" = _foreach_addcmul_scalar[64]
        getitem_588: "f32[512, 512]" = _foreach_addcmul_scalar[65]
        getitem_589: "f32[512, 512]" = _foreach_addcmul_scalar[66]
        getitem_590: "f32[512, 512]" = _foreach_addcmul_scalar[67]
        getitem_591: "f32[512, 512]" = _foreach_addcmul_scalar[68]
        getitem_592: "f32[512]" = _foreach_addcmul_scalar[69]
        getitem_593: "f32[512, 512]" = _foreach_addcmul_scalar[70]
        getitem_594: "f32[512, 512]" = _foreach_addcmul_scalar[71]
        getitem_595: "f32[512, 512]" = _foreach_addcmul_scalar[72]
        getitem_596: "f32[512, 512]" = _foreach_addcmul_scalar[73]
        getitem_597: "f32[512]" = _foreach_addcmul_scalar[74]
        getitem_598: "f32[2048, 512]" = _foreach_addcmul_scalar[75]
        getitem_599: "f32[512, 2048]" = _foreach_addcmul_scalar[76]
        getitem_600: "f32[512]" = _foreach_addcmul_scalar[77]
        getitem_601: "f32[512, 512]" = _foreach_addcmul_scalar[78]
        getitem_602: "f32[512, 512]" = _foreach_addcmul_scalar[79]
        getitem_603: "f32[512, 512]" = _foreach_addcmul_scalar[80]
        getitem_604: "f32[512, 512]" = _foreach_addcmul_scalar[81]
        getitem_605: "f32[512]" = _foreach_addcmul_scalar[82]
        getitem_606: "f32[512, 512]" = _foreach_addcmul_scalar[83]
        getitem_607: "f32[512, 512]" = _foreach_addcmul_scalar[84]
        getitem_608: "f32[512, 512]" = _foreach_addcmul_scalar[85]
        getitem_609: "f32[512, 512]" = _foreach_addcmul_scalar[86]
        getitem_610: "f32[512]" = _foreach_addcmul_scalar[87]
        getitem_611: "f32[2048, 512]" = _foreach_addcmul_scalar[88]
        getitem_612: "f32[512, 2048]" = _foreach_addcmul_scalar[89]
        getitem_613: "f32[512]" = _foreach_addcmul_scalar[90]
        getitem_614: "f32[512, 512]" = _foreach_addcmul_scalar[91]
        getitem_615: "f32[512, 512]" = _foreach_addcmul_scalar[92]
        getitem_616: "f32[512, 512]" = _foreach_addcmul_scalar[93]
        getitem_617: "f32[512, 512]" = _foreach_addcmul_scalar[94]
        getitem_618: "f32[512]" = _foreach_addcmul_scalar[95]
        getitem_619: "f32[512, 512]" = _foreach_addcmul_scalar[96]
        getitem_620: "f32[512, 512]" = _foreach_addcmul_scalar[97]
        getitem_621: "f32[512, 512]" = _foreach_addcmul_scalar[98]
        getitem_622: "f32[512, 512]" = _foreach_addcmul_scalar[99]
        getitem_623: "f32[512]" = _foreach_addcmul_scalar[100]
        getitem_624: "f32[2048, 512]" = _foreach_addcmul_scalar[101]
        getitem_625: "f32[512, 2048]" = _foreach_addcmul_scalar[102]
        getitem_626: "f32[512]" = _foreach_addcmul_scalar[103]
        getitem_627: "f32[512, 512]" = _foreach_addcmul_scalar[104]
        getitem_628: "f32[512, 512]" = _foreach_addcmul_scalar[105]
        getitem_629: "f32[512, 512]" = _foreach_addcmul_scalar[106]
        getitem_630: "f32[512, 512]" = _foreach_addcmul_scalar[107]
        getitem_631: "f32[512]" = _foreach_addcmul_scalar[108]
        getitem_632: "f32[512, 512]" = _foreach_addcmul_scalar[109]
        getitem_633: "f32[512, 512]" = _foreach_addcmul_scalar[110]
        getitem_634: "f32[512, 512]" = _foreach_addcmul_scalar[111]
        getitem_635: "f32[512, 512]" = _foreach_addcmul_scalar[112]
        getitem_636: "f32[512]" = _foreach_addcmul_scalar[113]
        getitem_637: "f32[2048, 512]" = _foreach_addcmul_scalar[114]
        getitem_638: "f32[512, 2048]" = _foreach_addcmul_scalar[115]
        getitem_639: "f32[512]" = _foreach_addcmul_scalar[116]
        getitem_640: "f32[512, 512]" = _foreach_addcmul_scalar[117]
        getitem_641: "f32[512, 512]" = _foreach_addcmul_scalar[118]
        getitem_642: "f32[512, 512]" = _foreach_addcmul_scalar[119]
        getitem_643: "f32[512, 512]" = _foreach_addcmul_scalar[120]
        getitem_644: "f32[512]" = _foreach_addcmul_scalar[121]
        getitem_645: "f32[512, 512]" = _foreach_addcmul_scalar[122]
        getitem_646: "f32[512, 512]" = _foreach_addcmul_scalar[123]
        getitem_647: "f32[512, 512]" = _foreach_addcmul_scalar[124]
        getitem_648: "f32[512, 512]" = _foreach_addcmul_scalar[125]
        getitem_649: "f32[512]" = _foreach_addcmul_scalar[126]
        getitem_650: "f32[2048, 512]" = _foreach_addcmul_scalar[127]
        getitem_651: "f32[512, 2048]" = _foreach_addcmul_scalar[128]
        getitem_652: "f32[512]" = _foreach_addcmul_scalar[129]
        getitem_653: "f32[512]" = _foreach_addcmul_scalar[130];  _foreach_addcmul_scalar = None
        getitem_654 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_654, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130]);  getitem_654 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = None
        getitem_131: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_132: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_133: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_134: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_135: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_136: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_137: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_138: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_139: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_140: "f32[]" = _foreach_pow_scalar_and_tensor[9]
        getitem_141: "f32[]" = _foreach_pow_scalar_and_tensor[10]
        getitem_142: "f32[]" = _foreach_pow_scalar_and_tensor[11]
        getitem_143: "f32[]" = _foreach_pow_scalar_and_tensor[12]
        getitem_144: "f32[]" = _foreach_pow_scalar_and_tensor[13]
        getitem_145: "f32[]" = _foreach_pow_scalar_and_tensor[14]
        getitem_146: "f32[]" = _foreach_pow_scalar_and_tensor[15]
        getitem_147: "f32[]" = _foreach_pow_scalar_and_tensor[16]
        getitem_148: "f32[]" = _foreach_pow_scalar_and_tensor[17]
        getitem_149: "f32[]" = _foreach_pow_scalar_and_tensor[18]
        getitem_150: "f32[]" = _foreach_pow_scalar_and_tensor[19]
        getitem_151: "f32[]" = _foreach_pow_scalar_and_tensor[20]
        getitem_152: "f32[]" = _foreach_pow_scalar_and_tensor[21]
        getitem_153: "f32[]" = _foreach_pow_scalar_and_tensor[22]
        getitem_154: "f32[]" = _foreach_pow_scalar_and_tensor[23]
        getitem_155: "f32[]" = _foreach_pow_scalar_and_tensor[24]
        getitem_156: "f32[]" = _foreach_pow_scalar_and_tensor[25]
        getitem_157: "f32[]" = _foreach_pow_scalar_and_tensor[26]
        getitem_158: "f32[]" = _foreach_pow_scalar_and_tensor[27]
        getitem_159: "f32[]" = _foreach_pow_scalar_and_tensor[28]
        getitem_160: "f32[]" = _foreach_pow_scalar_and_tensor[29]
        getitem_161: "f32[]" = _foreach_pow_scalar_and_tensor[30]
        getitem_162: "f32[]" = _foreach_pow_scalar_and_tensor[31]
        getitem_163: "f32[]" = _foreach_pow_scalar_and_tensor[32]
        getitem_164: "f32[]" = _foreach_pow_scalar_and_tensor[33]
        getitem_165: "f32[]" = _foreach_pow_scalar_and_tensor[34]
        getitem_166: "f32[]" = _foreach_pow_scalar_and_tensor[35]
        getitem_167: "f32[]" = _foreach_pow_scalar_and_tensor[36]
        getitem_168: "f32[]" = _foreach_pow_scalar_and_tensor[37]
        getitem_169: "f32[]" = _foreach_pow_scalar_and_tensor[38]
        getitem_170: "f32[]" = _foreach_pow_scalar_and_tensor[39]
        getitem_171: "f32[]" = _foreach_pow_scalar_and_tensor[40]
        getitem_172: "f32[]" = _foreach_pow_scalar_and_tensor[41]
        getitem_173: "f32[]" = _foreach_pow_scalar_and_tensor[42]
        getitem_174: "f32[]" = _foreach_pow_scalar_and_tensor[43]
        getitem_175: "f32[]" = _foreach_pow_scalar_and_tensor[44]
        getitem_176: "f32[]" = _foreach_pow_scalar_and_tensor[45]
        getitem_177: "f32[]" = _foreach_pow_scalar_and_tensor[46]
        getitem_178: "f32[]" = _foreach_pow_scalar_and_tensor[47]
        getitem_179: "f32[]" = _foreach_pow_scalar_and_tensor[48]
        getitem_180: "f32[]" = _foreach_pow_scalar_and_tensor[49]
        getitem_181: "f32[]" = _foreach_pow_scalar_and_tensor[50]
        getitem_182: "f32[]" = _foreach_pow_scalar_and_tensor[51]
        getitem_183: "f32[]" = _foreach_pow_scalar_and_tensor[52]
        getitem_184: "f32[]" = _foreach_pow_scalar_and_tensor[53]
        getitem_185: "f32[]" = _foreach_pow_scalar_and_tensor[54]
        getitem_186: "f32[]" = _foreach_pow_scalar_and_tensor[55]
        getitem_187: "f32[]" = _foreach_pow_scalar_and_tensor[56]
        getitem_188: "f32[]" = _foreach_pow_scalar_and_tensor[57]
        getitem_189: "f32[]" = _foreach_pow_scalar_and_tensor[58]
        getitem_190: "f32[]" = _foreach_pow_scalar_and_tensor[59]
        getitem_191: "f32[]" = _foreach_pow_scalar_and_tensor[60]
        getitem_192: "f32[]" = _foreach_pow_scalar_and_tensor[61]
        getitem_193: "f32[]" = _foreach_pow_scalar_and_tensor[62]
        getitem_194: "f32[]" = _foreach_pow_scalar_and_tensor[63]
        getitem_195: "f32[]" = _foreach_pow_scalar_and_tensor[64]
        getitem_196: "f32[]" = _foreach_pow_scalar_and_tensor[65]
        getitem_197: "f32[]" = _foreach_pow_scalar_and_tensor[66]
        getitem_198: "f32[]" = _foreach_pow_scalar_and_tensor[67]
        getitem_199: "f32[]" = _foreach_pow_scalar_and_tensor[68]
        getitem_200: "f32[]" = _foreach_pow_scalar_and_tensor[69]
        getitem_201: "f32[]" = _foreach_pow_scalar_and_tensor[70]
        getitem_202: "f32[]" = _foreach_pow_scalar_and_tensor[71]
        getitem_203: "f32[]" = _foreach_pow_scalar_and_tensor[72]
        getitem_204: "f32[]" = _foreach_pow_scalar_and_tensor[73]
        getitem_205: "f32[]" = _foreach_pow_scalar_and_tensor[74]
        getitem_206: "f32[]" = _foreach_pow_scalar_and_tensor[75]
        getitem_207: "f32[]" = _foreach_pow_scalar_and_tensor[76]
        getitem_208: "f32[]" = _foreach_pow_scalar_and_tensor[77]
        getitem_209: "f32[]" = _foreach_pow_scalar_and_tensor[78]
        getitem_210: "f32[]" = _foreach_pow_scalar_and_tensor[79]
        getitem_211: "f32[]" = _foreach_pow_scalar_and_tensor[80]
        getitem_212: "f32[]" = _foreach_pow_scalar_and_tensor[81]
        getitem_213: "f32[]" = _foreach_pow_scalar_and_tensor[82]
        getitem_214: "f32[]" = _foreach_pow_scalar_and_tensor[83]
        getitem_215: "f32[]" = _foreach_pow_scalar_and_tensor[84]
        getitem_216: "f32[]" = _foreach_pow_scalar_and_tensor[85]
        getitem_217: "f32[]" = _foreach_pow_scalar_and_tensor[86]
        getitem_218: "f32[]" = _foreach_pow_scalar_and_tensor[87]
        getitem_219: "f32[]" = _foreach_pow_scalar_and_tensor[88]
        getitem_220: "f32[]" = _foreach_pow_scalar_and_tensor[89]
        getitem_221: "f32[]" = _foreach_pow_scalar_and_tensor[90]
        getitem_222: "f32[]" = _foreach_pow_scalar_and_tensor[91]
        getitem_223: "f32[]" = _foreach_pow_scalar_and_tensor[92]
        getitem_224: "f32[]" = _foreach_pow_scalar_and_tensor[93]
        getitem_225: "f32[]" = _foreach_pow_scalar_and_tensor[94]
        getitem_226: "f32[]" = _foreach_pow_scalar_and_tensor[95]
        getitem_227: "f32[]" = _foreach_pow_scalar_and_tensor[96]
        getitem_228: "f32[]" = _foreach_pow_scalar_and_tensor[97]
        getitem_229: "f32[]" = _foreach_pow_scalar_and_tensor[98]
        getitem_230: "f32[]" = _foreach_pow_scalar_and_tensor[99]
        getitem_231: "f32[]" = _foreach_pow_scalar_and_tensor[100]
        getitem_232: "f32[]" = _foreach_pow_scalar_and_tensor[101]
        getitem_233: "f32[]" = _foreach_pow_scalar_and_tensor[102]
        getitem_234: "f32[]" = _foreach_pow_scalar_and_tensor[103]
        getitem_235: "f32[]" = _foreach_pow_scalar_and_tensor[104]
        getitem_236: "f32[]" = _foreach_pow_scalar_and_tensor[105]
        getitem_237: "f32[]" = _foreach_pow_scalar_and_tensor[106]
        getitem_238: "f32[]" = _foreach_pow_scalar_and_tensor[107]
        getitem_239: "f32[]" = _foreach_pow_scalar_and_tensor[108]
        getitem_240: "f32[]" = _foreach_pow_scalar_and_tensor[109]
        getitem_241: "f32[]" = _foreach_pow_scalar_and_tensor[110]
        getitem_242: "f32[]" = _foreach_pow_scalar_and_tensor[111]
        getitem_243: "f32[]" = _foreach_pow_scalar_and_tensor[112]
        getitem_244: "f32[]" = _foreach_pow_scalar_and_tensor[113]
        getitem_245: "f32[]" = _foreach_pow_scalar_and_tensor[114]
        getitem_246: "f32[]" = _foreach_pow_scalar_and_tensor[115]
        getitem_247: "f32[]" = _foreach_pow_scalar_and_tensor[116]
        getitem_248: "f32[]" = _foreach_pow_scalar_and_tensor[117]
        getitem_249: "f32[]" = _foreach_pow_scalar_and_tensor[118]
        getitem_250: "f32[]" = _foreach_pow_scalar_and_tensor[119]
        getitem_251: "f32[]" = _foreach_pow_scalar_and_tensor[120]
        getitem_252: "f32[]" = _foreach_pow_scalar_and_tensor[121]
        getitem_253: "f32[]" = _foreach_pow_scalar_and_tensor[122]
        getitem_254: "f32[]" = _foreach_pow_scalar_and_tensor[123]
        getitem_255: "f32[]" = _foreach_pow_scalar_and_tensor[124]
        getitem_256: "f32[]" = _foreach_pow_scalar_and_tensor[125]
        getitem_257: "f32[]" = _foreach_pow_scalar_and_tensor[126]
        getitem_258: "f32[]" = _foreach_pow_scalar_and_tensor[127]
        getitem_259: "f32[]" = _foreach_pow_scalar_and_tensor[128]
        getitem_260: "f32[]" = _foreach_pow_scalar_and_tensor[129]
        getitem_261: "f32[]" = _foreach_pow_scalar_and_tensor[130];  _foreach_pow_scalar_and_tensor = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178]);  getitem_1048 = getitem_1049 = getitem_1050 = getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = getitem_1056 = getitem_1057 = getitem_1058 = getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = getitem_1064 = getitem_1065 = getitem_1066 = getitem_1067 = getitem_1068 = getitem_1069 = getitem_1070 = getitem_1071 = getitem_1072 = getitem_1073 = getitem_1074 = getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = getitem_1080 = getitem_1081 = getitem_1082 = getitem_1083 = getitem_1084 = getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = getitem_1153 = getitem_1154 = getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = None
        getitem_1179: "f32[]" = _foreach_neg_default[0]
        getitem_1180: "f32[]" = _foreach_neg_default[1]
        getitem_1181: "f32[]" = _foreach_neg_default[2]
        getitem_1182: "f32[]" = _foreach_neg_default[3]
        getitem_1183: "f32[]" = _foreach_neg_default[4]
        getitem_1184: "f32[]" = _foreach_neg_default[5]
        getitem_1185: "f32[]" = _foreach_neg_default[6]
        getitem_1186: "f32[]" = _foreach_neg_default[7]
        getitem_1187: "f32[]" = _foreach_neg_default[8]
        getitem_1188: "f32[]" = _foreach_neg_default[9]
        getitem_1189: "f32[]" = _foreach_neg_default[10]
        getitem_1190: "f32[]" = _foreach_neg_default[11]
        getitem_1191: "f32[]" = _foreach_neg_default[12]
        getitem_1192: "f32[]" = _foreach_neg_default[13]
        getitem_1193: "f32[]" = _foreach_neg_default[14]
        getitem_1194: "f32[]" = _foreach_neg_default[15]
        getitem_1195: "f32[]" = _foreach_neg_default[16]
        getitem_1196: "f32[]" = _foreach_neg_default[17]
        getitem_1197: "f32[]" = _foreach_neg_default[18]
        getitem_1198: "f32[]" = _foreach_neg_default[19]
        getitem_1199: "f32[]" = _foreach_neg_default[20]
        getitem_1200: "f32[]" = _foreach_neg_default[21]
        getitem_1201: "f32[]" = _foreach_neg_default[22]
        getitem_1202: "f32[]" = _foreach_neg_default[23]
        getitem_1203: "f32[]" = _foreach_neg_default[24]
        getitem_1204: "f32[]" = _foreach_neg_default[25]
        getitem_1205: "f32[]" = _foreach_neg_default[26]
        getitem_1206: "f32[]" = _foreach_neg_default[27]
        getitem_1207: "f32[]" = _foreach_neg_default[28]
        getitem_1208: "f32[]" = _foreach_neg_default[29]
        getitem_1209: "f32[]" = _foreach_neg_default[30]
        getitem_1210: "f32[]" = _foreach_neg_default[31]
        getitem_1211: "f32[]" = _foreach_neg_default[32]
        getitem_1212: "f32[]" = _foreach_neg_default[33]
        getitem_1213: "f32[]" = _foreach_neg_default[34]
        getitem_1214: "f32[]" = _foreach_neg_default[35]
        getitem_1215: "f32[]" = _foreach_neg_default[36]
        getitem_1216: "f32[]" = _foreach_neg_default[37]
        getitem_1217: "f32[]" = _foreach_neg_default[38]
        getitem_1218: "f32[]" = _foreach_neg_default[39]
        getitem_1219: "f32[]" = _foreach_neg_default[40]
        getitem_1220: "f32[]" = _foreach_neg_default[41]
        getitem_1221: "f32[]" = _foreach_neg_default[42]
        getitem_1222: "f32[]" = _foreach_neg_default[43]
        getitem_1223: "f32[]" = _foreach_neg_default[44]
        getitem_1224: "f32[]" = _foreach_neg_default[45]
        getitem_1225: "f32[]" = _foreach_neg_default[46]
        getitem_1226: "f32[]" = _foreach_neg_default[47]
        getitem_1227: "f32[]" = _foreach_neg_default[48]
        getitem_1228: "f32[]" = _foreach_neg_default[49]
        getitem_1229: "f32[]" = _foreach_neg_default[50]
        getitem_1230: "f32[]" = _foreach_neg_default[51]
        getitem_1231: "f32[]" = _foreach_neg_default[52]
        getitem_1232: "f32[]" = _foreach_neg_default[53]
        getitem_1233: "f32[]" = _foreach_neg_default[54]
        getitem_1234: "f32[]" = _foreach_neg_default[55]
        getitem_1235: "f32[]" = _foreach_neg_default[56]
        getitem_1236: "f32[]" = _foreach_neg_default[57]
        getitem_1237: "f32[]" = _foreach_neg_default[58]
        getitem_1238: "f32[]" = _foreach_neg_default[59]
        getitem_1239: "f32[]" = _foreach_neg_default[60]
        getitem_1240: "f32[]" = _foreach_neg_default[61]
        getitem_1241: "f32[]" = _foreach_neg_default[62]
        getitem_1242: "f32[]" = _foreach_neg_default[63]
        getitem_1243: "f32[]" = _foreach_neg_default[64]
        getitem_1244: "f32[]" = _foreach_neg_default[65]
        getitem_1245: "f32[]" = _foreach_neg_default[66]
        getitem_1246: "f32[]" = _foreach_neg_default[67]
        getitem_1247: "f32[]" = _foreach_neg_default[68]
        getitem_1248: "f32[]" = _foreach_neg_default[69]
        getitem_1249: "f32[]" = _foreach_neg_default[70]
        getitem_1250: "f32[]" = _foreach_neg_default[71]
        getitem_1251: "f32[]" = _foreach_neg_default[72]
        getitem_1252: "f32[]" = _foreach_neg_default[73]
        getitem_1253: "f32[]" = _foreach_neg_default[74]
        getitem_1254: "f32[]" = _foreach_neg_default[75]
        getitem_1255: "f32[]" = _foreach_neg_default[76]
        getitem_1256: "f32[]" = _foreach_neg_default[77]
        getitem_1257: "f32[]" = _foreach_neg_default[78]
        getitem_1258: "f32[]" = _foreach_neg_default[79]
        getitem_1259: "f32[]" = _foreach_neg_default[80]
        getitem_1260: "f32[]" = _foreach_neg_default[81]
        getitem_1261: "f32[]" = _foreach_neg_default[82]
        getitem_1262: "f32[]" = _foreach_neg_default[83]
        getitem_1263: "f32[]" = _foreach_neg_default[84]
        getitem_1264: "f32[]" = _foreach_neg_default[85]
        getitem_1265: "f32[]" = _foreach_neg_default[86]
        getitem_1266: "f32[]" = _foreach_neg_default[87]
        getitem_1267: "f32[]" = _foreach_neg_default[88]
        getitem_1268: "f32[]" = _foreach_neg_default[89]
        getitem_1269: "f32[]" = _foreach_neg_default[90]
        getitem_1270: "f32[]" = _foreach_neg_default[91]
        getitem_1271: "f32[]" = _foreach_neg_default[92]
        getitem_1272: "f32[]" = _foreach_neg_default[93]
        getitem_1273: "f32[]" = _foreach_neg_default[94]
        getitem_1274: "f32[]" = _foreach_neg_default[95]
        getitem_1275: "f32[]" = _foreach_neg_default[96]
        getitem_1276: "f32[]" = _foreach_neg_default[97]
        getitem_1277: "f32[]" = _foreach_neg_default[98]
        getitem_1278: "f32[]" = _foreach_neg_default[99]
        getitem_1279: "f32[]" = _foreach_neg_default[100]
        getitem_1280: "f32[]" = _foreach_neg_default[101]
        getitem_1281: "f32[]" = _foreach_neg_default[102]
        getitem_1282: "f32[]" = _foreach_neg_default[103]
        getitem_1283: "f32[]" = _foreach_neg_default[104]
        getitem_1284: "f32[]" = _foreach_neg_default[105]
        getitem_1285: "f32[]" = _foreach_neg_default[106]
        getitem_1286: "f32[]" = _foreach_neg_default[107]
        getitem_1287: "f32[]" = _foreach_neg_default[108]
        getitem_1288: "f32[]" = _foreach_neg_default[109]
        getitem_1289: "f32[]" = _foreach_neg_default[110]
        getitem_1290: "f32[]" = _foreach_neg_default[111]
        getitem_1291: "f32[]" = _foreach_neg_default[112]
        getitem_1292: "f32[]" = _foreach_neg_default[113]
        getitem_1293: "f32[]" = _foreach_neg_default[114]
        getitem_1294: "f32[]" = _foreach_neg_default[115]
        getitem_1295: "f32[]" = _foreach_neg_default[116]
        getitem_1296: "f32[]" = _foreach_neg_default[117]
        getitem_1297: "f32[]" = _foreach_neg_default[118]
        getitem_1298: "f32[]" = _foreach_neg_default[119]
        getitem_1299: "f32[]" = _foreach_neg_default[120]
        getitem_1300: "f32[]" = _foreach_neg_default[121]
        getitem_1301: "f32[]" = _foreach_neg_default[122]
        getitem_1302: "f32[]" = _foreach_neg_default[123]
        getitem_1303: "f32[]" = _foreach_neg_default[124]
        getitem_1304: "f32[]" = _foreach_neg_default[125]
        getitem_1305: "f32[]" = _foreach_neg_default[126]
        getitem_1306: "f32[]" = _foreach_neg_default[127]
        getitem_1307: "f32[]" = _foreach_neg_default[128]
        getitem_1308: "f32[]" = _foreach_neg_default[129]
        getitem_1309: "f32[]" = _foreach_neg_default[130];  _foreach_neg_default = None
        return (getitem, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309)


def _default_make_inputs():
    return [
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # getitem_654 (unknown shape)
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

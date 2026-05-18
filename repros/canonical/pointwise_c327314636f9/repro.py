"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s2_g53
Pattern hash: c327314636f9
Shape hash: 81307daf
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
    def forward(self, getitem_447: "f32[50257, 768]", getitem_448: "f32[1024, 768]", getitem_449: "f32[768]", getitem_450: "f32[768]", getitem_451: "f32[768, 2304]", getitem_452: "f32[2304]", getitem_453: "f32[768, 768]", getitem_454: "f32[768]", getitem_455: "f32[768]", getitem_456: "f32[768]", getitem_457: "f32[768, 3072]", getitem_458: "f32[3072]", getitem_459: "f32[3072, 768]", getitem_460: "f32[768]", getitem_461: "f32[768]", getitem_462: "f32[768]", getitem_463: "f32[768, 2304]", getitem_464: "f32[2304]", getitem_465: "f32[768, 768]", getitem_466: "f32[768]", getitem_467: "f32[768]", getitem_468: "f32[768]", getitem_469: "f32[768, 3072]", getitem_470: "f32[3072]", getitem_471: "f32[3072, 768]", getitem_472: "f32[768]", getitem_473: "f32[768]", getitem_474: "f32[768]", getitem_475: "f32[768, 2304]", getitem_476: "f32[2304]", getitem_477: "f32[768, 768]", getitem_478: "f32[768]", getitem_479: "f32[768]", getitem_480: "f32[768]", getitem_481: "f32[768, 3072]", getitem_482: "f32[3072]", getitem_483: "f32[3072, 768]", getitem_484: "f32[768]", getitem_485: "f32[768]", getitem_486: "f32[768]", getitem_487: "f32[768, 2304]", getitem_488: "f32[2304]", getitem_489: "f32[768, 768]", getitem_490: "f32[768]", getitem_491: "f32[768]", getitem_492: "f32[768]", getitem_493: "f32[768, 3072]", getitem_494: "f32[3072]", getitem_495: "f32[3072, 768]", getitem_496: "f32[768]", getitem_497: "f32[768]", getitem_498: "f32[768]", getitem_499: "f32[768, 2304]", getitem_500: "f32[2304]", getitem_501: "f32[768, 768]", getitem_502: "f32[768]", getitem_503: "f32[768]", getitem_504: "f32[768]", getitem_505: "f32[768, 3072]", getitem_506: "f32[3072]", getitem_507: "f32[3072, 768]", getitem_508: "f32[768]", getitem_509: "f32[768]", getitem_510: "f32[768]", getitem_511: "f32[768, 2304]", getitem_512: "f32[2304]", getitem_513: "f32[768, 768]", getitem_514: "f32[768]", getitem_515: "f32[768]", getitem_516: "f32[768]", getitem_517: "f32[768, 3072]", getitem_518: "f32[3072]", getitem_519: "f32[3072, 768]", getitem_520: "f32[768]", getitem_521: "f32[768]", getitem_522: "f32[768]", getitem_523: "f32[768, 2304]", getitem_524: "f32[2304]", getitem_525: "f32[768, 768]", getitem_526: "f32[768]", getitem_527: "f32[768]", getitem_528: "f32[768]", getitem_529: "f32[768, 3072]", getitem_530: "f32[3072]", getitem_531: "f32[3072, 768]", getitem_532: "f32[768]", getitem_533: "f32[768]", getitem_534: "f32[768]", getitem_535: "f32[768, 2304]", getitem_536: "f32[2304]", getitem_537: "f32[768, 768]", getitem_538: "f32[768]", getitem_539: "f32[768]", getitem_540: "f32[768]", getitem_541: "f32[768, 3072]", getitem_542: "f32[3072]", getitem_543: "f32[3072, 768]", getitem_544: "f32[768]", getitem_545: "f32[768]", getitem_546: "f32[768]", getitem_547: "f32[768, 2304]", getitem_548: "f32[2304]", getitem_549: "f32[768, 768]", getitem_550: "f32[768]", getitem_551: "f32[768]", getitem_552: "f32[768]", getitem_553: "f32[768, 3072]", getitem_554: "f32[3072]", getitem_555: "f32[3072, 768]", getitem_556: "f32[768]", getitem_557: "f32[768]", getitem_558: "f32[768]", getitem_559: "f32[768, 2304]", getitem_560: "f32[2304]", getitem_561: "f32[768, 768]", getitem_562: "f32[768]", getitem_563: "f32[768]", getitem_564: "f32[768]", getitem_565: "f32[768, 3072]", getitem_566: "f32[3072]", getitem_567: "f32[3072, 768]", getitem_568: "f32[768]", getitem_569: "f32[768]", getitem_570: "f32[768]", getitem_571: "f32[768, 2304]", getitem_572: "f32[2304]", getitem_573: "f32[768, 768]", getitem_574: "f32[768]", getitem_575: "f32[768]", getitem_576: "f32[768]", getitem_577: "f32[768, 3072]", getitem_578: "f32[3072]", getitem_579: "f32[3072, 768]", getitem_580: "f32[768]", getitem_581: "f32[768]", getitem_582: "f32[768]", getitem_583: "f32[768, 2304]", getitem_584: "f32[2304]", getitem_585: "f32[768, 768]", getitem_586: "f32[768]", getitem_587: "f32[768]", getitem_588: "f32[768]", getitem_589: "f32[768, 3072]", getitem_590: "f32[3072]", getitem_591: "f32[3072, 768]", getitem_592: "f32[768]", getitem_593: "f32[768]", getitem_594: "f32[768]", getitem_595: "f32[2, 768]", arg596_1: "f32[50257, 768]", arg597_1: "f32[1024, 768]", arg598_1: "f32[768]", arg599_1: "f32[768]", arg600_1: "f32[768, 2304]", arg601_1: "f32[2304]", arg602_1: "f32[768, 768]", arg603_1: "f32[768]", arg604_1: "f32[768]", arg605_1: "f32[768]", arg606_1: "f32[768, 3072]", arg607_1: "f32[3072]", arg608_1: "f32[3072, 768]", arg609_1: "f32[768]", arg610_1: "f32[768]", arg611_1: "f32[768]", arg612_1: "f32[768, 2304]", arg613_1: "f32[2304]", arg614_1: "f32[768, 768]", arg615_1: "f32[768]", arg616_1: "f32[768]", arg617_1: "f32[768]", arg618_1: "f32[768, 3072]", arg619_1: "f32[3072]", arg620_1: "f32[3072, 768]", arg621_1: "f32[768]", arg622_1: "f32[768]", arg623_1: "f32[768]", arg624_1: "f32[768, 2304]", arg625_1: "f32[2304]", arg626_1: "f32[768, 768]", arg627_1: "f32[768]", arg628_1: "f32[768]", arg629_1: "f32[768]", arg630_1: "f32[768, 3072]", arg631_1: "f32[3072]", arg632_1: "f32[3072, 768]", arg633_1: "f32[768]", arg634_1: "f32[768]", arg635_1: "f32[768]", arg636_1: "f32[768, 2304]", arg637_1: "f32[2304]", arg638_1: "f32[768, 768]", arg639_1: "f32[768]", arg640_1: "f32[768]", arg641_1: "f32[768]", arg642_1: "f32[768, 3072]", arg643_1: "f32[3072]", arg644_1: "f32[3072, 768]", arg645_1: "f32[768]", arg646_1: "f32[768]", arg647_1: "f32[768]", arg648_1: "f32[768, 2304]", arg649_1: "f32[2304]", arg650_1: "f32[768, 768]", arg651_1: "f32[768]", arg652_1: "f32[768]", arg653_1: "f32[768]", arg654_1: "f32[768, 3072]", arg655_1: "f32[3072]", arg656_1: "f32[3072, 768]", arg657_1: "f32[768]", arg658_1: "f32[768]", arg659_1: "f32[768]", arg660_1: "f32[768, 2304]", arg661_1: "f32[2304]", arg662_1: "f32[768, 768]", arg663_1: "f32[768]", arg664_1: "f32[768]", arg665_1: "f32[768]", arg666_1: "f32[768, 3072]", arg667_1: "f32[3072]", arg668_1: "f32[3072, 768]", arg669_1: "f32[768]", arg670_1: "f32[768]", arg671_1: "f32[768]", arg672_1: "f32[768, 2304]", arg673_1: "f32[2304]", arg674_1: "f32[768, 768]", arg675_1: "f32[768]", arg676_1: "f32[768]", arg677_1: "f32[768]", arg678_1: "f32[768, 3072]", arg679_1: "f32[3072]", arg680_1: "f32[3072, 768]", arg681_1: "f32[768]", arg682_1: "f32[768]", arg683_1: "f32[768]", arg684_1: "f32[768, 2304]", arg685_1: "f32[2304]", arg686_1: "f32[768, 768]", arg687_1: "f32[768]", arg688_1: "f32[768]", arg689_1: "f32[768]", arg690_1: "f32[768, 3072]", arg691_1: "f32[3072]", arg692_1: "f32[3072, 768]", arg693_1: "f32[768]", arg694_1: "f32[768]", arg695_1: "f32[768]", arg696_1: "f32[768, 2304]", arg697_1: "f32[2304]", arg698_1: "f32[768, 768]", arg699_1: "f32[768]", arg700_1: "f32[768]", arg701_1: "f32[768]", arg702_1: "f32[768, 3072]", arg703_1: "f32[3072]", arg704_1: "f32[3072, 768]", arg705_1: "f32[768]", arg706_1: "f32[768]", arg707_1: "f32[768]", arg708_1: "f32[768, 2304]", arg709_1: "f32[2304]", arg710_1: "f32[768, 768]", arg711_1: "f32[768]", arg712_1: "f32[768]", arg713_1: "f32[768]", arg714_1: "f32[768, 3072]", arg715_1: "f32[3072]", arg716_1: "f32[3072, 768]", arg717_1: "f32[768]", arg718_1: "f32[768]", arg719_1: "f32[768]", arg720_1: "f32[768, 2304]", arg721_1: "f32[2304]", arg722_1: "f32[768, 768]", arg723_1: "f32[768]", arg724_1: "f32[768]", arg725_1: "f32[768]", arg726_1: "f32[768, 3072]", arg727_1: "f32[3072]", arg728_1: "f32[3072, 768]", arg729_1: "f32[768]", arg730_1: "f32[768]", arg731_1: "f32[768]", arg732_1: "f32[768, 2304]", arg733_1: "f32[2304]", arg734_1: "f32[768, 768]", arg735_1: "f32[768]", arg736_1: "f32[768]", arg737_1: "f32[768]", arg738_1: "f32[768, 3072]", arg739_1: "f32[3072]", arg740_1: "f32[3072, 768]", arg741_1: "f32[768]", arg742_1: "f32[768]", arg743_1: "f32[768]", arg744_1: "f32[2, 768]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_10: "f32[]", getitem_11: "f32[]", getitem_12: "f32[]", getitem_13: "f32[]", getitem_14: "f32[]", getitem_15: "f32[]", getitem_16: "f32[]", getitem_17: "f32[]", getitem_18: "f32[]", getitem_19: "f32[]", getitem_20: "f32[]", getitem_21: "f32[]", getitem_22: "f32[]", getitem_23: "f32[]", getitem_24: "f32[]", getitem_25: "f32[]", getitem_26: "f32[]", getitem_27: "f32[]", getitem_28: "f32[]", getitem_29: "f32[]", getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]", getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]", getitem_130: "f32[]", getitem_131: "f32[]", getitem_132: "f32[]", getitem_133: "f32[]", getitem_134: "f32[]", getitem_135: "f32[]", getitem_136: "f32[]", getitem_137: "f32[]", getitem_138: "f32[]", getitem_139: "f32[]", getitem_140: "f32[]", getitem_141: "f32[]", getitem_142: "f32[]", getitem_143: "f32[]", getitem_144: "f32[]", getitem_145: "f32[]", getitem_146: "f32[]", getitem_147: "f32[]", getitem_148: "f32[]", getitem_1192: "f32[]", getitem_1193: "f32[]", getitem_1194: "f32[]", getitem_1195: "f32[]", getitem_1196: "f32[]", getitem_1197: "f32[]", getitem_1198: "f32[]", getitem_1199: "f32[]", getitem_1200: "f32[]", getitem_1201: "f32[]", getitem_1202: "f32[]", getitem_1203: "f32[]", getitem_1204: "f32[]", getitem_1205: "f32[]", getitem_1206: "f32[]", getitem_1207: "f32[]", getitem_1208: "f32[]", getitem_1209: "f32[]", getitem_1210: "f32[]", getitem_1211: "f32[]", getitem_1212: "f32[]", getitem_1213: "f32[]", getitem_1214: "f32[]", getitem_1215: "f32[]", getitem_1216: "f32[]", getitem_1217: "f32[]", getitem_1218: "f32[]", getitem_1219: "f32[]", getitem_1220: "f32[]", getitem_1221: "f32[]", getitem_1222: "f32[]", getitem_1223: "f32[]", getitem_1224: "f32[]", getitem_1225: "f32[]", getitem_1226: "f32[]", getitem_1227: "f32[]", getitem_1228: "f32[]", getitem_1229: "f32[]", getitem_1230: "f32[]", getitem_1231: "f32[]", getitem_1232: "f32[]", getitem_1233: "f32[]", getitem_1234: "f32[]", getitem_1235: "f32[]", getitem_1236: "f32[]", getitem_1237: "f32[]", getitem_1238: "f32[]", getitem_1239: "f32[]", getitem_1240: "f32[]", getitem_1241: "f32[]", getitem_1242: "f32[]", getitem_1243: "f32[]", getitem_1244: "f32[]", getitem_1245: "f32[]", getitem_1246: "f32[]", getitem_1247: "f32[]", getitem_1248: "f32[]", getitem_1249: "f32[]", getitem_1250: "f32[]", getitem_1251: "f32[]", getitem_1252: "f32[]", getitem_1253: "f32[]", getitem_1254: "f32[]", getitem_1255: "f32[]", getitem_1256: "f32[]", getitem_1257: "f32[]", getitem_1258: "f32[]", getitem_1259: "f32[]", getitem_1260: "f32[]", getitem_1261: "f32[]", getitem_1262: "f32[]", getitem_1263: "f32[]", getitem_1264: "f32[]", getitem_1265: "f32[]", getitem_1266: "f32[]", getitem_1267: "f32[]", getitem_1268: "f32[]", getitem_1269: "f32[]", getitem_1270: "f32[]", getitem_1271: "f32[]", getitem_1272: "f32[]", getitem_1273: "f32[]", getitem_1274: "f32[]", getitem_1275: "f32[]", getitem_1276: "f32[]", getitem_1277: "f32[]", getitem_1278: "f32[]", getitem_1279: "f32[]", getitem_1280: "f32[]", getitem_1281: "f32[]", getitem_1282: "f32[]", getitem_1283: "f32[]", getitem_1284: "f32[]", getitem_1285: "f32[]", getitem_1286: "f32[]", getitem_1287: "f32[]", getitem_1288: "f32[]", getitem_1289: "f32[]", getitem_1290: "f32[]", getitem_1291: "f32[]", getitem_1292: "f32[]", getitem_1293: "f32[]", getitem_1294: "f32[]", getitem_1295: "f32[]", getitem_1296: "f32[]", getitem_1297: "f32[]", getitem_1298: "f32[]", getitem_1299: "f32[]", getitem_1300: "f32[]", getitem_1301: "f32[]", getitem_1302: "f32[]", getitem_1303: "f32[]", getitem_1304: "f32[]", getitem_1305: "f32[]", getitem_1306: "f32[]", getitem_1307: "f32[]", getitem_1308: "f32[]", getitem_1309: "f32[]", getitem_1310: "f32[]", getitem_1311: "f32[]", getitem_1312: "f32[]", getitem_1313: "f32[]", getitem_1314: "f32[]", getitem_1315: "f32[]", getitem_1316: "f32[]", getitem_1317: "f32[]", getitem_1318: "f32[]", getitem_1319: "f32[]", getitem_1320: "f32[]", getitem_1321: "f32[]", getitem_1322: "f32[]", getitem_1323: "f32[]", getitem_1324: "f32[]", getitem_1325: "f32[]", getitem_1326: "f32[]", getitem_1327: "f32[]", getitem_1328: "f32[]", getitem_1329: "f32[]", getitem_1330: "f32[]", getitem_1331: "f32[]", getitem_1332: "f32[]", getitem_1333: "f32[]", getitem_1334: "f32[]", getitem_1335: "f32[]", getitem_1336: "f32[]", getitem_1337: "f32[]", getitem_1338: "f32[]", getitem_1339: "f32[]", getitem_1340: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595], [arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1], [arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1], 0.0010000000000000009);  getitem_447 = getitem_448 = getitem_449 = getitem_450 = getitem_451 = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = getitem_457 = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_465 = getitem_466 = getitem_467 = getitem_468 = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = getitem_474 = getitem_475 = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = getitem_481 = getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = getitem_487 = getitem_488 = getitem_489 = getitem_490 = getitem_491 = getitem_492 = getitem_493 = getitem_494 = getitem_495 = getitem_496 = getitem_497 = getitem_498 = getitem_499 = getitem_500 = getitem_501 = getitem_502 = getitem_503 = getitem_504 = getitem_505 = getitem_506 = getitem_507 = getitem_508 = getitem_509 = getitem_510 = getitem_511 = getitem_512 = getitem_513 = getitem_514 = getitem_515 = getitem_516 = getitem_517 = getitem_518 = getitem_519 = getitem_520 = getitem_521 = getitem_522 = getitem_523 = getitem_524 = getitem_525 = getitem_526 = getitem_527 = getitem_528 = getitem_529 = getitem_530 = getitem_531 = getitem_532 = getitem_533 = getitem_534 = getitem_535 = getitem_536 = getitem_537 = getitem_538 = getitem_539 = getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = getitem_572 = getitem_573 = getitem_574 = getitem_575 = getitem_576 = getitem_577 = getitem_578 = getitem_579 = getitem_580 = getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = arg655_1 = arg656_1 = arg657_1 = arg658_1 = arg659_1 = arg660_1 = arg661_1 = arg662_1 = arg663_1 = arg664_1 = arg665_1 = arg666_1 = arg667_1 = arg668_1 = arg669_1 = arg670_1 = arg671_1 = arg672_1 = arg673_1 = arg674_1 = arg675_1 = arg676_1 = arg677_1 = arg678_1 = arg679_1 = arg680_1 = arg681_1 = arg682_1 = arg683_1 = arg684_1 = arg685_1 = arg686_1 = arg687_1 = arg688_1 = arg689_1 = arg690_1 = arg691_1 = arg692_1 = arg693_1 = arg694_1 = arg695_1 = arg696_1 = arg697_1 = arg698_1 = arg699_1 = arg700_1 = arg701_1 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = arg706_1 = arg707_1 = arg708_1 = arg709_1 = arg710_1 = arg711_1 = arg712_1 = arg713_1 = arg714_1 = arg715_1 = arg716_1 = arg717_1 = arg718_1 = arg719_1 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = arg724_1 = arg725_1 = arg726_1 = arg727_1 = arg728_1 = arg729_1 = arg730_1 = arg731_1 = arg732_1 = arg733_1 = arg734_1 = arg735_1 = arg736_1 = arg737_1 = arg738_1 = arg739_1 = arg740_1 = arg741_1 = arg742_1 = arg743_1 = arg744_1 = None
        getitem: "f32[50257, 768]" = _foreach_addcmul_scalar[0]
        getitem_596: "f32[1024, 768]" = _foreach_addcmul_scalar[1]
        getitem_597: "f32[768]" = _foreach_addcmul_scalar[2]
        getitem_598: "f32[768]" = _foreach_addcmul_scalar[3]
        getitem_599: "f32[768, 2304]" = _foreach_addcmul_scalar[4]
        getitem_600: "f32[2304]" = _foreach_addcmul_scalar[5]
        getitem_601: "f32[768, 768]" = _foreach_addcmul_scalar[6]
        getitem_602: "f32[768]" = _foreach_addcmul_scalar[7]
        getitem_603: "f32[768]" = _foreach_addcmul_scalar[8]
        getitem_604: "f32[768]" = _foreach_addcmul_scalar[9]
        getitem_605: "f32[768, 3072]" = _foreach_addcmul_scalar[10]
        getitem_606: "f32[3072]" = _foreach_addcmul_scalar[11]
        getitem_607: "f32[3072, 768]" = _foreach_addcmul_scalar[12]
        getitem_608: "f32[768]" = _foreach_addcmul_scalar[13]
        getitem_609: "f32[768]" = _foreach_addcmul_scalar[14]
        getitem_610: "f32[768]" = _foreach_addcmul_scalar[15]
        getitem_611: "f32[768, 2304]" = _foreach_addcmul_scalar[16]
        getitem_612: "f32[2304]" = _foreach_addcmul_scalar[17]
        getitem_613: "f32[768, 768]" = _foreach_addcmul_scalar[18]
        getitem_614: "f32[768]" = _foreach_addcmul_scalar[19]
        getitem_615: "f32[768]" = _foreach_addcmul_scalar[20]
        getitem_616: "f32[768]" = _foreach_addcmul_scalar[21]
        getitem_617: "f32[768, 3072]" = _foreach_addcmul_scalar[22]
        getitem_618: "f32[3072]" = _foreach_addcmul_scalar[23]
        getitem_619: "f32[3072, 768]" = _foreach_addcmul_scalar[24]
        getitem_620: "f32[768]" = _foreach_addcmul_scalar[25]
        getitem_621: "f32[768]" = _foreach_addcmul_scalar[26]
        getitem_622: "f32[768]" = _foreach_addcmul_scalar[27]
        getitem_623: "f32[768, 2304]" = _foreach_addcmul_scalar[28]
        getitem_624: "f32[2304]" = _foreach_addcmul_scalar[29]
        getitem_625: "f32[768, 768]" = _foreach_addcmul_scalar[30]
        getitem_626: "f32[768]" = _foreach_addcmul_scalar[31]
        getitem_627: "f32[768]" = _foreach_addcmul_scalar[32]
        getitem_628: "f32[768]" = _foreach_addcmul_scalar[33]
        getitem_629: "f32[768, 3072]" = _foreach_addcmul_scalar[34]
        getitem_630: "f32[3072]" = _foreach_addcmul_scalar[35]
        getitem_631: "f32[3072, 768]" = _foreach_addcmul_scalar[36]
        getitem_632: "f32[768]" = _foreach_addcmul_scalar[37]
        getitem_633: "f32[768]" = _foreach_addcmul_scalar[38]
        getitem_634: "f32[768]" = _foreach_addcmul_scalar[39]
        getitem_635: "f32[768, 2304]" = _foreach_addcmul_scalar[40]
        getitem_636: "f32[2304]" = _foreach_addcmul_scalar[41]
        getitem_637: "f32[768, 768]" = _foreach_addcmul_scalar[42]
        getitem_638: "f32[768]" = _foreach_addcmul_scalar[43]
        getitem_639: "f32[768]" = _foreach_addcmul_scalar[44]
        getitem_640: "f32[768]" = _foreach_addcmul_scalar[45]
        getitem_641: "f32[768, 3072]" = _foreach_addcmul_scalar[46]
        getitem_642: "f32[3072]" = _foreach_addcmul_scalar[47]
        getitem_643: "f32[3072, 768]" = _foreach_addcmul_scalar[48]
        getitem_644: "f32[768]" = _foreach_addcmul_scalar[49]
        getitem_645: "f32[768]" = _foreach_addcmul_scalar[50]
        getitem_646: "f32[768]" = _foreach_addcmul_scalar[51]
        getitem_647: "f32[768, 2304]" = _foreach_addcmul_scalar[52]
        getitem_648: "f32[2304]" = _foreach_addcmul_scalar[53]
        getitem_649: "f32[768, 768]" = _foreach_addcmul_scalar[54]
        getitem_650: "f32[768]" = _foreach_addcmul_scalar[55]
        getitem_651: "f32[768]" = _foreach_addcmul_scalar[56]
        getitem_652: "f32[768]" = _foreach_addcmul_scalar[57]
        getitem_653: "f32[768, 3072]" = _foreach_addcmul_scalar[58]
        getitem_654: "f32[3072]" = _foreach_addcmul_scalar[59]
        getitem_655: "f32[3072, 768]" = _foreach_addcmul_scalar[60]
        getitem_656: "f32[768]" = _foreach_addcmul_scalar[61]
        getitem_657: "f32[768]" = _foreach_addcmul_scalar[62]
        getitem_658: "f32[768]" = _foreach_addcmul_scalar[63]
        getitem_659: "f32[768, 2304]" = _foreach_addcmul_scalar[64]
        getitem_660: "f32[2304]" = _foreach_addcmul_scalar[65]
        getitem_661: "f32[768, 768]" = _foreach_addcmul_scalar[66]
        getitem_662: "f32[768]" = _foreach_addcmul_scalar[67]
        getitem_663: "f32[768]" = _foreach_addcmul_scalar[68]
        getitem_664: "f32[768]" = _foreach_addcmul_scalar[69]
        getitem_665: "f32[768, 3072]" = _foreach_addcmul_scalar[70]
        getitem_666: "f32[3072]" = _foreach_addcmul_scalar[71]
        getitem_667: "f32[3072, 768]" = _foreach_addcmul_scalar[72]
        getitem_668: "f32[768]" = _foreach_addcmul_scalar[73]
        getitem_669: "f32[768]" = _foreach_addcmul_scalar[74]
        getitem_670: "f32[768]" = _foreach_addcmul_scalar[75]
        getitem_671: "f32[768, 2304]" = _foreach_addcmul_scalar[76]
        getitem_672: "f32[2304]" = _foreach_addcmul_scalar[77]
        getitem_673: "f32[768, 768]" = _foreach_addcmul_scalar[78]
        getitem_674: "f32[768]" = _foreach_addcmul_scalar[79]
        getitem_675: "f32[768]" = _foreach_addcmul_scalar[80]
        getitem_676: "f32[768]" = _foreach_addcmul_scalar[81]
        getitem_677: "f32[768, 3072]" = _foreach_addcmul_scalar[82]
        getitem_678: "f32[3072]" = _foreach_addcmul_scalar[83]
        getitem_679: "f32[3072, 768]" = _foreach_addcmul_scalar[84]
        getitem_680: "f32[768]" = _foreach_addcmul_scalar[85]
        getitem_681: "f32[768]" = _foreach_addcmul_scalar[86]
        getitem_682: "f32[768]" = _foreach_addcmul_scalar[87]
        getitem_683: "f32[768, 2304]" = _foreach_addcmul_scalar[88]
        getitem_684: "f32[2304]" = _foreach_addcmul_scalar[89]
        getitem_685: "f32[768, 768]" = _foreach_addcmul_scalar[90]
        getitem_686: "f32[768]" = _foreach_addcmul_scalar[91]
        getitem_687: "f32[768]" = _foreach_addcmul_scalar[92]
        getitem_688: "f32[768]" = _foreach_addcmul_scalar[93]
        getitem_689: "f32[768, 3072]" = _foreach_addcmul_scalar[94]
        getitem_690: "f32[3072]" = _foreach_addcmul_scalar[95]
        getitem_691: "f32[3072, 768]" = _foreach_addcmul_scalar[96]
        getitem_692: "f32[768]" = _foreach_addcmul_scalar[97]
        getitem_693: "f32[768]" = _foreach_addcmul_scalar[98]
        getitem_694: "f32[768]" = _foreach_addcmul_scalar[99]
        getitem_695: "f32[768, 2304]" = _foreach_addcmul_scalar[100]
        getitem_696: "f32[2304]" = _foreach_addcmul_scalar[101]
        getitem_697: "f32[768, 768]" = _foreach_addcmul_scalar[102]
        getitem_698: "f32[768]" = _foreach_addcmul_scalar[103]
        getitem_699: "f32[768]" = _foreach_addcmul_scalar[104]
        getitem_700: "f32[768]" = _foreach_addcmul_scalar[105]
        getitem_701: "f32[768, 3072]" = _foreach_addcmul_scalar[106]
        getitem_702: "f32[3072]" = _foreach_addcmul_scalar[107]
        getitem_703: "f32[3072, 768]" = _foreach_addcmul_scalar[108]
        getitem_704: "f32[768]" = _foreach_addcmul_scalar[109]
        getitem_705: "f32[768]" = _foreach_addcmul_scalar[110]
        getitem_706: "f32[768]" = _foreach_addcmul_scalar[111]
        getitem_707: "f32[768, 2304]" = _foreach_addcmul_scalar[112]
        getitem_708: "f32[2304]" = _foreach_addcmul_scalar[113]
        getitem_709: "f32[768, 768]" = _foreach_addcmul_scalar[114]
        getitem_710: "f32[768]" = _foreach_addcmul_scalar[115]
        getitem_711: "f32[768]" = _foreach_addcmul_scalar[116]
        getitem_712: "f32[768]" = _foreach_addcmul_scalar[117]
        getitem_713: "f32[768, 3072]" = _foreach_addcmul_scalar[118]
        getitem_714: "f32[3072]" = _foreach_addcmul_scalar[119]
        getitem_715: "f32[3072, 768]" = _foreach_addcmul_scalar[120]
        getitem_716: "f32[768]" = _foreach_addcmul_scalar[121]
        getitem_717: "f32[768]" = _foreach_addcmul_scalar[122]
        getitem_718: "f32[768]" = _foreach_addcmul_scalar[123]
        getitem_719: "f32[768, 2304]" = _foreach_addcmul_scalar[124]
        getitem_720: "f32[2304]" = _foreach_addcmul_scalar[125]
        getitem_721: "f32[768, 768]" = _foreach_addcmul_scalar[126]
        getitem_722: "f32[768]" = _foreach_addcmul_scalar[127]
        getitem_723: "f32[768]" = _foreach_addcmul_scalar[128]
        getitem_724: "f32[768]" = _foreach_addcmul_scalar[129]
        getitem_725: "f32[768, 3072]" = _foreach_addcmul_scalar[130]
        getitem_726: "f32[3072]" = _foreach_addcmul_scalar[131]
        getitem_727: "f32[3072, 768]" = _foreach_addcmul_scalar[132]
        getitem_728: "f32[768]" = _foreach_addcmul_scalar[133]
        getitem_729: "f32[768]" = _foreach_addcmul_scalar[134]
        getitem_730: "f32[768]" = _foreach_addcmul_scalar[135]
        getitem_731: "f32[768, 2304]" = _foreach_addcmul_scalar[136]
        getitem_732: "f32[2304]" = _foreach_addcmul_scalar[137]
        getitem_733: "f32[768, 768]" = _foreach_addcmul_scalar[138]
        getitem_734: "f32[768]" = _foreach_addcmul_scalar[139]
        getitem_735: "f32[768]" = _foreach_addcmul_scalar[140]
        getitem_736: "f32[768]" = _foreach_addcmul_scalar[141]
        getitem_737: "f32[768, 3072]" = _foreach_addcmul_scalar[142]
        getitem_738: "f32[3072]" = _foreach_addcmul_scalar[143]
        getitem_739: "f32[3072, 768]" = _foreach_addcmul_scalar[144]
        getitem_740: "f32[768]" = _foreach_addcmul_scalar[145]
        getitem_741: "f32[768]" = _foreach_addcmul_scalar[146]
        getitem_742: "f32[768]" = _foreach_addcmul_scalar[147]
        getitem_743: "f32[2, 768]" = _foreach_addcmul_scalar[148];  _foreach_addcmul_scalar = None
        getitem_744 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_744, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148]);  getitem_744 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = None
        getitem_149: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_150: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_151: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_152: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_153: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_154: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_155: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_156: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_157: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_158: "f32[]" = _foreach_pow_scalar_and_tensor[9]
        getitem_159: "f32[]" = _foreach_pow_scalar_and_tensor[10]
        getitem_160: "f32[]" = _foreach_pow_scalar_and_tensor[11]
        getitem_161: "f32[]" = _foreach_pow_scalar_and_tensor[12]
        getitem_162: "f32[]" = _foreach_pow_scalar_and_tensor[13]
        getitem_163: "f32[]" = _foreach_pow_scalar_and_tensor[14]
        getitem_164: "f32[]" = _foreach_pow_scalar_and_tensor[15]
        getitem_165: "f32[]" = _foreach_pow_scalar_and_tensor[16]
        getitem_166: "f32[]" = _foreach_pow_scalar_and_tensor[17]
        getitem_167: "f32[]" = _foreach_pow_scalar_and_tensor[18]
        getitem_168: "f32[]" = _foreach_pow_scalar_and_tensor[19]
        getitem_169: "f32[]" = _foreach_pow_scalar_and_tensor[20]
        getitem_170: "f32[]" = _foreach_pow_scalar_and_tensor[21]
        getitem_171: "f32[]" = _foreach_pow_scalar_and_tensor[22]
        getitem_172: "f32[]" = _foreach_pow_scalar_and_tensor[23]
        getitem_173: "f32[]" = _foreach_pow_scalar_and_tensor[24]
        getitem_174: "f32[]" = _foreach_pow_scalar_and_tensor[25]
        getitem_175: "f32[]" = _foreach_pow_scalar_and_tensor[26]
        getitem_176: "f32[]" = _foreach_pow_scalar_and_tensor[27]
        getitem_177: "f32[]" = _foreach_pow_scalar_and_tensor[28]
        getitem_178: "f32[]" = _foreach_pow_scalar_and_tensor[29]
        getitem_179: "f32[]" = _foreach_pow_scalar_and_tensor[30]
        getitem_180: "f32[]" = _foreach_pow_scalar_and_tensor[31]
        getitem_181: "f32[]" = _foreach_pow_scalar_and_tensor[32]
        getitem_182: "f32[]" = _foreach_pow_scalar_and_tensor[33]
        getitem_183: "f32[]" = _foreach_pow_scalar_and_tensor[34]
        getitem_184: "f32[]" = _foreach_pow_scalar_and_tensor[35]
        getitem_185: "f32[]" = _foreach_pow_scalar_and_tensor[36]
        getitem_186: "f32[]" = _foreach_pow_scalar_and_tensor[37]
        getitem_187: "f32[]" = _foreach_pow_scalar_and_tensor[38]
        getitem_188: "f32[]" = _foreach_pow_scalar_and_tensor[39]
        getitem_189: "f32[]" = _foreach_pow_scalar_and_tensor[40]
        getitem_190: "f32[]" = _foreach_pow_scalar_and_tensor[41]
        getitem_191: "f32[]" = _foreach_pow_scalar_and_tensor[42]
        getitem_192: "f32[]" = _foreach_pow_scalar_and_tensor[43]
        getitem_193: "f32[]" = _foreach_pow_scalar_and_tensor[44]
        getitem_194: "f32[]" = _foreach_pow_scalar_and_tensor[45]
        getitem_195: "f32[]" = _foreach_pow_scalar_and_tensor[46]
        getitem_196: "f32[]" = _foreach_pow_scalar_and_tensor[47]
        getitem_197: "f32[]" = _foreach_pow_scalar_and_tensor[48]
        getitem_198: "f32[]" = _foreach_pow_scalar_and_tensor[49]
        getitem_199: "f32[]" = _foreach_pow_scalar_and_tensor[50]
        getitem_200: "f32[]" = _foreach_pow_scalar_and_tensor[51]
        getitem_201: "f32[]" = _foreach_pow_scalar_and_tensor[52]
        getitem_202: "f32[]" = _foreach_pow_scalar_and_tensor[53]
        getitem_203: "f32[]" = _foreach_pow_scalar_and_tensor[54]
        getitem_204: "f32[]" = _foreach_pow_scalar_and_tensor[55]
        getitem_205: "f32[]" = _foreach_pow_scalar_and_tensor[56]
        getitem_206: "f32[]" = _foreach_pow_scalar_and_tensor[57]
        getitem_207: "f32[]" = _foreach_pow_scalar_and_tensor[58]
        getitem_208: "f32[]" = _foreach_pow_scalar_and_tensor[59]
        getitem_209: "f32[]" = _foreach_pow_scalar_and_tensor[60]
        getitem_210: "f32[]" = _foreach_pow_scalar_and_tensor[61]
        getitem_211: "f32[]" = _foreach_pow_scalar_and_tensor[62]
        getitem_212: "f32[]" = _foreach_pow_scalar_and_tensor[63]
        getitem_213: "f32[]" = _foreach_pow_scalar_and_tensor[64]
        getitem_214: "f32[]" = _foreach_pow_scalar_and_tensor[65]
        getitem_215: "f32[]" = _foreach_pow_scalar_and_tensor[66]
        getitem_216: "f32[]" = _foreach_pow_scalar_and_tensor[67]
        getitem_217: "f32[]" = _foreach_pow_scalar_and_tensor[68]
        getitem_218: "f32[]" = _foreach_pow_scalar_and_tensor[69]
        getitem_219: "f32[]" = _foreach_pow_scalar_and_tensor[70]
        getitem_220: "f32[]" = _foreach_pow_scalar_and_tensor[71]
        getitem_221: "f32[]" = _foreach_pow_scalar_and_tensor[72]
        getitem_222: "f32[]" = _foreach_pow_scalar_and_tensor[73]
        getitem_223: "f32[]" = _foreach_pow_scalar_and_tensor[74]
        getitem_224: "f32[]" = _foreach_pow_scalar_and_tensor[75]
        getitem_225: "f32[]" = _foreach_pow_scalar_and_tensor[76]
        getitem_226: "f32[]" = _foreach_pow_scalar_and_tensor[77]
        getitem_227: "f32[]" = _foreach_pow_scalar_and_tensor[78]
        getitem_228: "f32[]" = _foreach_pow_scalar_and_tensor[79]
        getitem_229: "f32[]" = _foreach_pow_scalar_and_tensor[80]
        getitem_230: "f32[]" = _foreach_pow_scalar_and_tensor[81]
        getitem_231: "f32[]" = _foreach_pow_scalar_and_tensor[82]
        getitem_232: "f32[]" = _foreach_pow_scalar_and_tensor[83]
        getitem_233: "f32[]" = _foreach_pow_scalar_and_tensor[84]
        getitem_234: "f32[]" = _foreach_pow_scalar_and_tensor[85]
        getitem_235: "f32[]" = _foreach_pow_scalar_and_tensor[86]
        getitem_236: "f32[]" = _foreach_pow_scalar_and_tensor[87]
        getitem_237: "f32[]" = _foreach_pow_scalar_and_tensor[88]
        getitem_238: "f32[]" = _foreach_pow_scalar_and_tensor[89]
        getitem_239: "f32[]" = _foreach_pow_scalar_and_tensor[90]
        getitem_240: "f32[]" = _foreach_pow_scalar_and_tensor[91]
        getitem_241: "f32[]" = _foreach_pow_scalar_and_tensor[92]
        getitem_242: "f32[]" = _foreach_pow_scalar_and_tensor[93]
        getitem_243: "f32[]" = _foreach_pow_scalar_and_tensor[94]
        getitem_244: "f32[]" = _foreach_pow_scalar_and_tensor[95]
        getitem_245: "f32[]" = _foreach_pow_scalar_and_tensor[96]
        getitem_246: "f32[]" = _foreach_pow_scalar_and_tensor[97]
        getitem_247: "f32[]" = _foreach_pow_scalar_and_tensor[98]
        getitem_248: "f32[]" = _foreach_pow_scalar_and_tensor[99]
        getitem_249: "f32[]" = _foreach_pow_scalar_and_tensor[100]
        getitem_250: "f32[]" = _foreach_pow_scalar_and_tensor[101]
        getitem_251: "f32[]" = _foreach_pow_scalar_and_tensor[102]
        getitem_252: "f32[]" = _foreach_pow_scalar_and_tensor[103]
        getitem_253: "f32[]" = _foreach_pow_scalar_and_tensor[104]
        getitem_254: "f32[]" = _foreach_pow_scalar_and_tensor[105]
        getitem_255: "f32[]" = _foreach_pow_scalar_and_tensor[106]
        getitem_256: "f32[]" = _foreach_pow_scalar_and_tensor[107]
        getitem_257: "f32[]" = _foreach_pow_scalar_and_tensor[108]
        getitem_258: "f32[]" = _foreach_pow_scalar_and_tensor[109]
        getitem_259: "f32[]" = _foreach_pow_scalar_and_tensor[110]
        getitem_260: "f32[]" = _foreach_pow_scalar_and_tensor[111]
        getitem_261: "f32[]" = _foreach_pow_scalar_and_tensor[112]
        getitem_262: "f32[]" = _foreach_pow_scalar_and_tensor[113]
        getitem_263: "f32[]" = _foreach_pow_scalar_and_tensor[114]
        getitem_264: "f32[]" = _foreach_pow_scalar_and_tensor[115]
        getitem_265: "f32[]" = _foreach_pow_scalar_and_tensor[116]
        getitem_266: "f32[]" = _foreach_pow_scalar_and_tensor[117]
        getitem_267: "f32[]" = _foreach_pow_scalar_and_tensor[118]
        getitem_268: "f32[]" = _foreach_pow_scalar_and_tensor[119]
        getitem_269: "f32[]" = _foreach_pow_scalar_and_tensor[120]
        getitem_270: "f32[]" = _foreach_pow_scalar_and_tensor[121]
        getitem_271: "f32[]" = _foreach_pow_scalar_and_tensor[122]
        getitem_272: "f32[]" = _foreach_pow_scalar_and_tensor[123]
        getitem_273: "f32[]" = _foreach_pow_scalar_and_tensor[124]
        getitem_274: "f32[]" = _foreach_pow_scalar_and_tensor[125]
        getitem_275: "f32[]" = _foreach_pow_scalar_and_tensor[126]
        getitem_276: "f32[]" = _foreach_pow_scalar_and_tensor[127]
        getitem_277: "f32[]" = _foreach_pow_scalar_and_tensor[128]
        getitem_278: "f32[]" = _foreach_pow_scalar_and_tensor[129]
        getitem_279: "f32[]" = _foreach_pow_scalar_and_tensor[130]
        getitem_280: "f32[]" = _foreach_pow_scalar_and_tensor[131]
        getitem_281: "f32[]" = _foreach_pow_scalar_and_tensor[132]
        getitem_282: "f32[]" = _foreach_pow_scalar_and_tensor[133]
        getitem_283: "f32[]" = _foreach_pow_scalar_and_tensor[134]
        getitem_284: "f32[]" = _foreach_pow_scalar_and_tensor[135]
        getitem_285: "f32[]" = _foreach_pow_scalar_and_tensor[136]
        getitem_286: "f32[]" = _foreach_pow_scalar_and_tensor[137]
        getitem_287: "f32[]" = _foreach_pow_scalar_and_tensor[138]
        getitem_288: "f32[]" = _foreach_pow_scalar_and_tensor[139]
        getitem_289: "f32[]" = _foreach_pow_scalar_and_tensor[140]
        getitem_290: "f32[]" = _foreach_pow_scalar_and_tensor[141]
        getitem_291: "f32[]" = _foreach_pow_scalar_and_tensor[142]
        getitem_292: "f32[]" = _foreach_pow_scalar_and_tensor[143]
        getitem_293: "f32[]" = _foreach_pow_scalar_and_tensor[144]
        getitem_294: "f32[]" = _foreach_pow_scalar_and_tensor[145]
        getitem_295: "f32[]" = _foreach_pow_scalar_and_tensor[146]
        getitem_296: "f32[]" = _foreach_pow_scalar_and_tensor[147]
        getitem_297: "f32[]" = _foreach_pow_scalar_and_tensor[148];  _foreach_pow_scalar_and_tensor = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340]);  getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = getitem_1234 = getitem_1235 = getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = getitem_1242 = getitem_1243 = getitem_1244 = getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = getitem_1250 = getitem_1251 = getitem_1252 = getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = getitem_1258 = getitem_1259 = getitem_1260 = getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = getitem_1269 = getitem_1270 = getitem_1271 = getitem_1272 = getitem_1273 = getitem_1274 = getitem_1275 = getitem_1276 = getitem_1277 = getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = getitem_1285 = getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = getitem_1293 = getitem_1294 = getitem_1295 = getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = getitem_1302 = getitem_1303 = getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = getitem_1310 = getitem_1311 = getitem_1312 = getitem_1313 = getitem_1314 = getitem_1315 = getitem_1316 = getitem_1317 = getitem_1318 = getitem_1319 = getitem_1320 = getitem_1321 = getitem_1322 = getitem_1323 = getitem_1324 = getitem_1325 = getitem_1326 = getitem_1327 = getitem_1328 = getitem_1329 = getitem_1330 = getitem_1331 = getitem_1332 = getitem_1333 = getitem_1334 = getitem_1335 = getitem_1336 = getitem_1337 = getitem_1338 = getitem_1339 = getitem_1340 = None
        getitem_1341: "f32[]" = _foreach_neg_default[0]
        getitem_1342: "f32[]" = _foreach_neg_default[1]
        getitem_1343: "f32[]" = _foreach_neg_default[2]
        getitem_1344: "f32[]" = _foreach_neg_default[3]
        getitem_1345: "f32[]" = _foreach_neg_default[4]
        getitem_1346: "f32[]" = _foreach_neg_default[5]
        getitem_1347: "f32[]" = _foreach_neg_default[6]
        getitem_1348: "f32[]" = _foreach_neg_default[7]
        getitem_1349: "f32[]" = _foreach_neg_default[8]
        getitem_1350: "f32[]" = _foreach_neg_default[9]
        getitem_1351: "f32[]" = _foreach_neg_default[10]
        getitem_1352: "f32[]" = _foreach_neg_default[11]
        getitem_1353: "f32[]" = _foreach_neg_default[12]
        getitem_1354: "f32[]" = _foreach_neg_default[13]
        getitem_1355: "f32[]" = _foreach_neg_default[14]
        getitem_1356: "f32[]" = _foreach_neg_default[15]
        getitem_1357: "f32[]" = _foreach_neg_default[16]
        getitem_1358: "f32[]" = _foreach_neg_default[17]
        getitem_1359: "f32[]" = _foreach_neg_default[18]
        getitem_1360: "f32[]" = _foreach_neg_default[19]
        getitem_1361: "f32[]" = _foreach_neg_default[20]
        getitem_1362: "f32[]" = _foreach_neg_default[21]
        getitem_1363: "f32[]" = _foreach_neg_default[22]
        getitem_1364: "f32[]" = _foreach_neg_default[23]
        getitem_1365: "f32[]" = _foreach_neg_default[24]
        getitem_1366: "f32[]" = _foreach_neg_default[25]
        getitem_1367: "f32[]" = _foreach_neg_default[26]
        getitem_1368: "f32[]" = _foreach_neg_default[27]
        getitem_1369: "f32[]" = _foreach_neg_default[28]
        getitem_1370: "f32[]" = _foreach_neg_default[29]
        getitem_1371: "f32[]" = _foreach_neg_default[30]
        getitem_1372: "f32[]" = _foreach_neg_default[31]
        getitem_1373: "f32[]" = _foreach_neg_default[32]
        getitem_1374: "f32[]" = _foreach_neg_default[33]
        getitem_1375: "f32[]" = _foreach_neg_default[34]
        getitem_1376: "f32[]" = _foreach_neg_default[35]
        getitem_1377: "f32[]" = _foreach_neg_default[36]
        getitem_1378: "f32[]" = _foreach_neg_default[37]
        getitem_1379: "f32[]" = _foreach_neg_default[38]
        getitem_1380: "f32[]" = _foreach_neg_default[39]
        getitem_1381: "f32[]" = _foreach_neg_default[40]
        getitem_1382: "f32[]" = _foreach_neg_default[41]
        getitem_1383: "f32[]" = _foreach_neg_default[42]
        getitem_1384: "f32[]" = _foreach_neg_default[43]
        getitem_1385: "f32[]" = _foreach_neg_default[44]
        getitem_1386: "f32[]" = _foreach_neg_default[45]
        getitem_1387: "f32[]" = _foreach_neg_default[46]
        getitem_1388: "f32[]" = _foreach_neg_default[47]
        getitem_1389: "f32[]" = _foreach_neg_default[48]
        getitem_1390: "f32[]" = _foreach_neg_default[49]
        getitem_1391: "f32[]" = _foreach_neg_default[50]
        getitem_1392: "f32[]" = _foreach_neg_default[51]
        getitem_1393: "f32[]" = _foreach_neg_default[52]
        getitem_1394: "f32[]" = _foreach_neg_default[53]
        getitem_1395: "f32[]" = _foreach_neg_default[54]
        getitem_1396: "f32[]" = _foreach_neg_default[55]
        getitem_1397: "f32[]" = _foreach_neg_default[56]
        getitem_1398: "f32[]" = _foreach_neg_default[57]
        getitem_1399: "f32[]" = _foreach_neg_default[58]
        getitem_1400: "f32[]" = _foreach_neg_default[59]
        getitem_1401: "f32[]" = _foreach_neg_default[60]
        getitem_1402: "f32[]" = _foreach_neg_default[61]
        getitem_1403: "f32[]" = _foreach_neg_default[62]
        getitem_1404: "f32[]" = _foreach_neg_default[63]
        getitem_1405: "f32[]" = _foreach_neg_default[64]
        getitem_1406: "f32[]" = _foreach_neg_default[65]
        getitem_1407: "f32[]" = _foreach_neg_default[66]
        getitem_1408: "f32[]" = _foreach_neg_default[67]
        getitem_1409: "f32[]" = _foreach_neg_default[68]
        getitem_1410: "f32[]" = _foreach_neg_default[69]
        getitem_1411: "f32[]" = _foreach_neg_default[70]
        getitem_1412: "f32[]" = _foreach_neg_default[71]
        getitem_1413: "f32[]" = _foreach_neg_default[72]
        getitem_1414: "f32[]" = _foreach_neg_default[73]
        getitem_1415: "f32[]" = _foreach_neg_default[74]
        getitem_1416: "f32[]" = _foreach_neg_default[75]
        getitem_1417: "f32[]" = _foreach_neg_default[76]
        getitem_1418: "f32[]" = _foreach_neg_default[77]
        getitem_1419: "f32[]" = _foreach_neg_default[78]
        getitem_1420: "f32[]" = _foreach_neg_default[79]
        getitem_1421: "f32[]" = _foreach_neg_default[80]
        getitem_1422: "f32[]" = _foreach_neg_default[81]
        getitem_1423: "f32[]" = _foreach_neg_default[82]
        getitem_1424: "f32[]" = _foreach_neg_default[83]
        getitem_1425: "f32[]" = _foreach_neg_default[84]
        getitem_1426: "f32[]" = _foreach_neg_default[85]
        getitem_1427: "f32[]" = _foreach_neg_default[86]
        getitem_1428: "f32[]" = _foreach_neg_default[87]
        getitem_1429: "f32[]" = _foreach_neg_default[88]
        getitem_1430: "f32[]" = _foreach_neg_default[89]
        getitem_1431: "f32[]" = _foreach_neg_default[90]
        getitem_1432: "f32[]" = _foreach_neg_default[91]
        getitem_1433: "f32[]" = _foreach_neg_default[92]
        getitem_1434: "f32[]" = _foreach_neg_default[93]
        getitem_1435: "f32[]" = _foreach_neg_default[94]
        getitem_1436: "f32[]" = _foreach_neg_default[95]
        getitem_1437: "f32[]" = _foreach_neg_default[96]
        getitem_1438: "f32[]" = _foreach_neg_default[97]
        getitem_1439: "f32[]" = _foreach_neg_default[98]
        getitem_1440: "f32[]" = _foreach_neg_default[99]
        getitem_1441: "f32[]" = _foreach_neg_default[100]
        getitem_1442: "f32[]" = _foreach_neg_default[101]
        getitem_1443: "f32[]" = _foreach_neg_default[102]
        getitem_1444: "f32[]" = _foreach_neg_default[103]
        getitem_1445: "f32[]" = _foreach_neg_default[104]
        getitem_1446: "f32[]" = _foreach_neg_default[105]
        getitem_1447: "f32[]" = _foreach_neg_default[106]
        getitem_1448: "f32[]" = _foreach_neg_default[107]
        getitem_1449: "f32[]" = _foreach_neg_default[108]
        getitem_1450: "f32[]" = _foreach_neg_default[109]
        getitem_1451: "f32[]" = _foreach_neg_default[110]
        getitem_1452: "f32[]" = _foreach_neg_default[111]
        getitem_1453: "f32[]" = _foreach_neg_default[112]
        getitem_1454: "f32[]" = _foreach_neg_default[113]
        getitem_1455: "f32[]" = _foreach_neg_default[114]
        getitem_1456: "f32[]" = _foreach_neg_default[115]
        getitem_1457: "f32[]" = _foreach_neg_default[116]
        getitem_1458: "f32[]" = _foreach_neg_default[117]
        getitem_1459: "f32[]" = _foreach_neg_default[118]
        getitem_1460: "f32[]" = _foreach_neg_default[119]
        getitem_1461: "f32[]" = _foreach_neg_default[120]
        getitem_1462: "f32[]" = _foreach_neg_default[121]
        getitem_1463: "f32[]" = _foreach_neg_default[122]
        getitem_1464: "f32[]" = _foreach_neg_default[123]
        getitem_1465: "f32[]" = _foreach_neg_default[124]
        getitem_1466: "f32[]" = _foreach_neg_default[125]
        getitem_1467: "f32[]" = _foreach_neg_default[126]
        getitem_1468: "f32[]" = _foreach_neg_default[127]
        getitem_1469: "f32[]" = _foreach_neg_default[128]
        getitem_1470: "f32[]" = _foreach_neg_default[129]
        getitem_1471: "f32[]" = _foreach_neg_default[130]
        getitem_1472: "f32[]" = _foreach_neg_default[131]
        getitem_1473: "f32[]" = _foreach_neg_default[132]
        getitem_1474: "f32[]" = _foreach_neg_default[133]
        getitem_1475: "f32[]" = _foreach_neg_default[134]
        getitem_1476: "f32[]" = _foreach_neg_default[135]
        getitem_1477: "f32[]" = _foreach_neg_default[136]
        getitem_1478: "f32[]" = _foreach_neg_default[137]
        getitem_1479: "f32[]" = _foreach_neg_default[138]
        getitem_1480: "f32[]" = _foreach_neg_default[139]
        getitem_1481: "f32[]" = _foreach_neg_default[140]
        getitem_1482: "f32[]" = _foreach_neg_default[141]
        getitem_1483: "f32[]" = _foreach_neg_default[142]
        getitem_1484: "f32[]" = _foreach_neg_default[143]
        getitem_1485: "f32[]" = _foreach_neg_default[144]
        getitem_1486: "f32[]" = _foreach_neg_default[145]
        getitem_1487: "f32[]" = _foreach_neg_default[146]
        getitem_1488: "f32[]" = _foreach_neg_default[147]
        getitem_1489: "f32[]" = _foreach_neg_default[148];  _foreach_neg_default = None
        return (getitem, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489)


def _default_make_inputs():
    return [
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([50257, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 2304], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # getitem_744 (unknown shape)
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g77
Pattern hash: 9d2c61e0cb6c
Shape hash: 687750cd
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
    def forward(self, getitem_504: "f32[64, 3, 3, 3]", getitem_505: "f32[64]", getitem_506: "f32[64]", getitem_507: "f32[64, 3, 1, 1]", getitem_508: "f32[64]", getitem_509: "f32[64]", getitem_510: "f32[96, 64, 3, 3]", getitem_511: "f32[96]", getitem_512: "f32[96]", getitem_513: "f32[96, 64, 1, 1]", getitem_514: "f32[96]", getitem_515: "f32[96]", getitem_516: "f32[96]", getitem_517: "f32[96]", getitem_518: "f32[96, 96, 3, 3]", getitem_519: "f32[96]", getitem_520: "f32[96]", getitem_521: "f32[96, 96, 1, 1]", getitem_522: "f32[96]", getitem_523: "f32[96]", getitem_524: "f32[192, 96, 3, 3]", getitem_525: "f32[192]", getitem_526: "f32[192]", getitem_527: "f32[192, 96, 1, 1]", getitem_528: "f32[192]", getitem_529: "f32[192]", getitem_530: "f32[192]", getitem_531: "f32[192]", getitem_532: "f32[192, 192, 3, 3]", getitem_533: "f32[192]", getitem_534: "f32[192]", getitem_535: "f32[192, 192, 1, 1]", getitem_536: "f32[192]", getitem_537: "f32[192]", getitem_538: "f32[192]", getitem_539: "f32[192]", getitem_540: "f32[192, 192, 3, 3]", getitem_541: "f32[192]", getitem_542: "f32[192]", getitem_543: "f32[192, 192, 1, 1]", getitem_544: "f32[192]", getitem_545: "f32[192]", getitem_546: "f32[192]", getitem_547: "f32[192]", getitem_548: "f32[192, 192, 3, 3]", getitem_549: "f32[192]", getitem_550: "f32[192]", getitem_551: "f32[192, 192, 1, 1]", getitem_552: "f32[192]", getitem_553: "f32[192]", getitem_554: "f32[384, 192, 3, 3]", getitem_555: "f32[384]", getitem_556: "f32[384]", getitem_557: "f32[384, 192, 1, 1]", getitem_558: "f32[384]", getitem_559: "f32[384]", getitem_560: "f32[384]", getitem_561: "f32[384]", getitem_562: "f32[384, 384, 3, 3]", getitem_563: "f32[384]", getitem_564: "f32[384]", getitem_565: "f32[384, 384, 1, 1]", getitem_566: "f32[384]", getitem_567: "f32[384]", getitem_568: "f32[384]", getitem_569: "f32[384]", getitem_570: "f32[384, 384, 3, 3]", getitem_571: "f32[384]", getitem_572: "f32[384]", getitem_573: "f32[384, 384, 1, 1]", getitem_574: "f32[384]", getitem_575: "f32[384]", getitem_576: "f32[384]", getitem_577: "f32[384]", getitem_578: "f32[384, 384, 3, 3]", getitem_579: "f32[384]", getitem_580: "f32[384]", getitem_581: "f32[384, 384, 1, 1]", getitem_582: "f32[384]", getitem_583: "f32[384]", getitem_584: "f32[384]", getitem_585: "f32[384]", getitem_586: "f32[384, 384, 3, 3]", getitem_587: "f32[384]", getitem_588: "f32[384]", getitem_589: "f32[384, 384, 1, 1]", getitem_590: "f32[384]", getitem_591: "f32[384]", getitem_592: "f32[384]", getitem_593: "f32[384]", getitem_594: "f32[384, 384, 3, 3]", getitem_595: "f32[384]", getitem_596: "f32[384]", getitem_597: "f32[384, 384, 1, 1]", getitem_598: "f32[384]", getitem_599: "f32[384]", getitem_600: "f32[384]", getitem_601: "f32[384]", getitem_602: "f32[384, 384, 3, 3]", getitem_603: "f32[384]", getitem_604: "f32[384]", getitem_605: "f32[384, 384, 1, 1]", getitem_606: "f32[384]", getitem_607: "f32[384]", getitem_608: "f32[384]", getitem_609: "f32[384]", getitem_610: "f32[384, 384, 3, 3]", getitem_611: "f32[384]", getitem_612: "f32[384]", getitem_613: "f32[384, 384, 1, 1]", getitem_614: "f32[384]", getitem_615: "f32[384]", getitem_616: "f32[384]", getitem_617: "f32[384]", getitem_618: "f32[384, 384, 3, 3]", getitem_619: "f32[384]", getitem_620: "f32[384]", getitem_621: "f32[384, 384, 1, 1]", getitem_622: "f32[384]", getitem_623: "f32[384]", getitem_624: "f32[384]", getitem_625: "f32[384]", getitem_626: "f32[384, 384, 3, 3]", getitem_627: "f32[384]", getitem_628: "f32[384]", getitem_629: "f32[384, 384, 1, 1]", getitem_630: "f32[384]", getitem_631: "f32[384]", getitem_632: "f32[384]", getitem_633: "f32[384]", getitem_634: "f32[384, 384, 3, 3]", getitem_635: "f32[384]", getitem_636: "f32[384]", getitem_637: "f32[384, 384, 1, 1]", getitem_638: "f32[384]", getitem_639: "f32[384]", getitem_640: "f32[384]", getitem_641: "f32[384]", getitem_642: "f32[384, 384, 3, 3]", getitem_643: "f32[384]", getitem_644: "f32[384]", getitem_645: "f32[384, 384, 1, 1]", getitem_646: "f32[384]", getitem_647: "f32[384]", getitem_648: "f32[384]", getitem_649: "f32[384]", getitem_650: "f32[384, 384, 3, 3]", getitem_651: "f32[384]", getitem_652: "f32[384]", getitem_653: "f32[384, 384, 1, 1]", getitem_654: "f32[384]", getitem_655: "f32[384]", getitem_656: "f32[384]", getitem_657: "f32[384]", getitem_658: "f32[384, 384, 3, 3]", getitem_659: "f32[384]", getitem_660: "f32[384]", getitem_661: "f32[384, 384, 1, 1]", getitem_662: "f32[384]", getitem_663: "f32[384]", getitem_664: "f32[1408, 384, 3, 3]", getitem_665: "f32[1408]", getitem_666: "f32[1408]", getitem_667: "f32[1408, 384, 1, 1]", getitem_668: "f32[1408]", getitem_669: "f32[1408]", getitem_670: "f32[1000, 1408]", getitem_671: "f32[1000]", arg672_1: "f32[64, 3, 3, 3]", arg673_1: "f32[64]", arg674_1: "f32[64]", arg675_1: "f32[64, 3, 1, 1]", arg676_1: "f32[64]", arg677_1: "f32[64]", arg678_1: "f32[96, 64, 3, 3]", arg679_1: "f32[96]", arg680_1: "f32[96]", arg681_1: "f32[96, 64, 1, 1]", arg682_1: "f32[96]", arg683_1: "f32[96]", arg684_1: "f32[96]", arg685_1: "f32[96]", arg686_1: "f32[96, 96, 3, 3]", arg687_1: "f32[96]", arg688_1: "f32[96]", arg689_1: "f32[96, 96, 1, 1]", arg690_1: "f32[96]", arg691_1: "f32[96]", arg692_1: "f32[192, 96, 3, 3]", arg693_1: "f32[192]", arg694_1: "f32[192]", arg695_1: "f32[192, 96, 1, 1]", arg696_1: "f32[192]", arg697_1: "f32[192]", arg698_1: "f32[192]", arg699_1: "f32[192]", arg700_1: "f32[192, 192, 3, 3]", arg701_1: "f32[192]", arg702_1: "f32[192]", arg703_1: "f32[192, 192, 1, 1]", arg704_1: "f32[192]", arg705_1: "f32[192]", arg706_1: "f32[192]", arg707_1: "f32[192]", arg708_1: "f32[192, 192, 3, 3]", arg709_1: "f32[192]", arg710_1: "f32[192]", arg711_1: "f32[192, 192, 1, 1]", arg712_1: "f32[192]", arg713_1: "f32[192]", arg714_1: "f32[192]", arg715_1: "f32[192]", arg716_1: "f32[192, 192, 3, 3]", arg717_1: "f32[192]", arg718_1: "f32[192]", arg719_1: "f32[192, 192, 1, 1]", arg720_1: "f32[192]", arg721_1: "f32[192]", arg722_1: "f32[384, 192, 3, 3]", arg723_1: "f32[384]", arg724_1: "f32[384]", arg725_1: "f32[384, 192, 1, 1]", arg726_1: "f32[384]", arg727_1: "f32[384]", arg728_1: "f32[384]", arg729_1: "f32[384]", arg730_1: "f32[384, 384, 3, 3]", arg731_1: "f32[384]", arg732_1: "f32[384]", arg733_1: "f32[384, 384, 1, 1]", arg734_1: "f32[384]", arg735_1: "f32[384]", arg736_1: "f32[384]", arg737_1: "f32[384]", arg738_1: "f32[384, 384, 3, 3]", arg739_1: "f32[384]", arg740_1: "f32[384]", arg741_1: "f32[384, 384, 1, 1]", arg742_1: "f32[384]", arg743_1: "f32[384]", arg744_1: "f32[384]", arg745_1: "f32[384]", arg746_1: "f32[384, 384, 3, 3]", arg747_1: "f32[384]", arg748_1: "f32[384]", arg749_1: "f32[384, 384, 1, 1]", arg750_1: "f32[384]", arg751_1: "f32[384]", arg752_1: "f32[384]", arg753_1: "f32[384]", arg754_1: "f32[384, 384, 3, 3]", arg755_1: "f32[384]", arg756_1: "f32[384]", arg757_1: "f32[384, 384, 1, 1]", arg758_1: "f32[384]", arg759_1: "f32[384]", arg760_1: "f32[384]", arg761_1: "f32[384]", arg762_1: "f32[384, 384, 3, 3]", arg763_1: "f32[384]", arg764_1: "f32[384]", arg765_1: "f32[384, 384, 1, 1]", arg766_1: "f32[384]", arg767_1: "f32[384]", arg768_1: "f32[384]", arg769_1: "f32[384]", arg770_1: "f32[384, 384, 3, 3]", arg771_1: "f32[384]", arg772_1: "f32[384]", arg773_1: "f32[384, 384, 1, 1]", arg774_1: "f32[384]", arg775_1: "f32[384]", arg776_1: "f32[384]", arg777_1: "f32[384]", arg778_1: "f32[384, 384, 3, 3]", arg779_1: "f32[384]", arg780_1: "f32[384]", arg781_1: "f32[384, 384, 1, 1]", arg782_1: "f32[384]", arg783_1: "f32[384]", arg784_1: "f32[384]", arg785_1: "f32[384]", arg786_1: "f32[384, 384, 3, 3]", arg787_1: "f32[384]", arg788_1: "f32[384]", arg789_1: "f32[384, 384, 1, 1]", arg790_1: "f32[384]", arg791_1: "f32[384]", arg792_1: "f32[384]", arg793_1: "f32[384]", arg794_1: "f32[384, 384, 3, 3]", arg795_1: "f32[384]", arg796_1: "f32[384]", arg797_1: "f32[384, 384, 1, 1]", arg798_1: "f32[384]", arg799_1: "f32[384]", arg800_1: "f32[384]", arg801_1: "f32[384]", arg802_1: "f32[384, 384, 3, 3]", arg803_1: "f32[384]", arg804_1: "f32[384]", arg805_1: "f32[384, 384, 1, 1]", arg806_1: "f32[384]", arg807_1: "f32[384]", arg808_1: "f32[384]", arg809_1: "f32[384]", arg810_1: "f32[384, 384, 3, 3]", arg811_1: "f32[384]", arg812_1: "f32[384]", arg813_1: "f32[384, 384, 1, 1]", arg814_1: "f32[384]", arg815_1: "f32[384]", arg816_1: "f32[384]", arg817_1: "f32[384]", arg818_1: "f32[384, 384, 3, 3]", arg819_1: "f32[384]", arg820_1: "f32[384]", arg821_1: "f32[384, 384, 1, 1]", arg822_1: "f32[384]", arg823_1: "f32[384]", arg824_1: "f32[384]", arg825_1: "f32[384]", arg826_1: "f32[384, 384, 3, 3]", arg827_1: "f32[384]", arg828_1: "f32[384]", arg829_1: "f32[384, 384, 1, 1]", arg830_1: "f32[384]", arg831_1: "f32[384]", arg832_1: "f32[1408, 384, 3, 3]", arg833_1: "f32[1408]", arg834_1: "f32[1408]", arg835_1: "f32[1408, 384, 1, 1]", arg836_1: "f32[1408]", arg837_1: "f32[1408]", arg838_1: "f32[1000, 1408]", arg839_1: "f32[1000]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_10: "f32[]", getitem_11: "f32[]", getitem_12: "f32[]", getitem_13: "f32[]", getitem_14: "f32[]", getitem_15: "f32[]", getitem_16: "f32[]", getitem_17: "f32[]", getitem_18: "f32[]", getitem_19: "f32[]", getitem_20: "f32[]", getitem_21: "f32[]", getitem_22: "f32[]", getitem_23: "f32[]", getitem_24: "f32[]", getitem_25: "f32[]", getitem_26: "f32[]", getitem_27: "f32[]", getitem_28: "f32[]", getitem_29: "f32[]", getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]", getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]", getitem_130: "f32[]", getitem_131: "f32[]", getitem_132: "f32[]", getitem_133: "f32[]", getitem_134: "f32[]", getitem_135: "f32[]", getitem_136: "f32[]", getitem_137: "f32[]", getitem_138: "f32[]", getitem_139: "f32[]", getitem_140: "f32[]", getitem_141: "f32[]", getitem_142: "f32[]", getitem_143: "f32[]", getitem_144: "f32[]", getitem_145: "f32[]", getitem_146: "f32[]", getitem_147: "f32[]", getitem_148: "f32[]", getitem_149: "f32[]", getitem_150: "f32[]", getitem_151: "f32[]", getitem_152: "f32[]", getitem_153: "f32[]", getitem_154: "f32[]", getitem_155: "f32[]", getitem_156: "f32[]", getitem_157: "f32[]", getitem_158: "f32[]", getitem_159: "f32[]", getitem_160: "f32[]", getitem_161: "f32[]", getitem_162: "f32[]", getitem_163: "f32[]", getitem_164: "f32[]", getitem_165: "f32[]", getitem_166: "f32[]", getitem_167: "f32[]", getitem_1344: "f32[]", getitem_1345: "f32[]", getitem_1346: "f32[]", getitem_1347: "f32[]", getitem_1348: "f32[]", getitem_1349: "f32[]", getitem_1350: "f32[]", getitem_1351: "f32[]", getitem_1352: "f32[]", getitem_1353: "f32[]", getitem_1354: "f32[]", getitem_1355: "f32[]", getitem_1356: "f32[]", getitem_1357: "f32[]", getitem_1358: "f32[]", getitem_1359: "f32[]", getitem_1360: "f32[]", getitem_1361: "f32[]", getitem_1362: "f32[]", getitem_1363: "f32[]", getitem_1364: "f32[]", getitem_1365: "f32[]", getitem_1366: "f32[]", getitem_1367: "f32[]", getitem_1368: "f32[]", getitem_1369: "f32[]", getitem_1370: "f32[]", getitem_1371: "f32[]", getitem_1372: "f32[]", getitem_1373: "f32[]", getitem_1374: "f32[]", getitem_1375: "f32[]", getitem_1376: "f32[]", getitem_1377: "f32[]", getitem_1378: "f32[]", getitem_1379: "f32[]", getitem_1380: "f32[]", getitem_1381: "f32[]", getitem_1382: "f32[]", getitem_1383: "f32[]", getitem_1384: "f32[]", getitem_1385: "f32[]", getitem_1386: "f32[]", getitem_1387: "f32[]", getitem_1388: "f32[]", getitem_1389: "f32[]", getitem_1390: "f32[]", getitem_1391: "f32[]", getitem_1392: "f32[]", getitem_1393: "f32[]", getitem_1394: "f32[]", getitem_1395: "f32[]", getitem_1396: "f32[]", getitem_1397: "f32[]", getitem_1398: "f32[]", getitem_1399: "f32[]", getitem_1400: "f32[]", getitem_1401: "f32[]", getitem_1402: "f32[]", getitem_1403: "f32[]", getitem_1404: "f32[]", getitem_1405: "f32[]", getitem_1406: "f32[]", getitem_1407: "f32[]", getitem_1408: "f32[]", getitem_1409: "f32[]", getitem_1410: "f32[]", getitem_1411: "f32[]", getitem_1412: "f32[]", getitem_1413: "f32[]", getitem_1414: "f32[]", getitem_1415: "f32[]", getitem_1416: "f32[]", getitem_1417: "f32[]", getitem_1418: "f32[]", getitem_1419: "f32[]", getitem_1420: "f32[]", getitem_1421: "f32[]", getitem_1422: "f32[]", getitem_1423: "f32[]", getitem_1424: "f32[]", getitem_1425: "f32[]", getitem_1426: "f32[]", getitem_1427: "f32[]", getitem_1428: "f32[]", getitem_1429: "f32[]", getitem_1430: "f32[]", getitem_1431: "f32[]", getitem_1432: "f32[]", getitem_1433: "f32[]", getitem_1434: "f32[]", getitem_1435: "f32[]", getitem_1436: "f32[]", getitem_1437: "f32[]", getitem_1438: "f32[]", getitem_1439: "f32[]", getitem_1440: "f32[]", getitem_1441: "f32[]", getitem_1442: "f32[]", getitem_1443: "f32[]", getitem_1444: "f32[]", getitem_1445: "f32[]", getitem_1446: "f32[]", getitem_1447: "f32[]", getitem_1448: "f32[]", getitem_1449: "f32[]", getitem_1450: "f32[]", getitem_1451: "f32[]", getitem_1452: "f32[]", getitem_1453: "f32[]", getitem_1454: "f32[]", getitem_1455: "f32[]", getitem_1456: "f32[]", getitem_1457: "f32[]", getitem_1458: "f32[]", getitem_1459: "f32[]", getitem_1460: "f32[]", getitem_1461: "f32[]", getitem_1462: "f32[]", getitem_1463: "f32[]", getitem_1464: "f32[]", getitem_1465: "f32[]", getitem_1466: "f32[]", getitem_1467: "f32[]", getitem_1468: "f32[]", getitem_1469: "f32[]", getitem_1470: "f32[]", getitem_1471: "f32[]", getitem_1472: "f32[]", getitem_1473: "f32[]", getitem_1474: "f32[]", getitem_1475: "f32[]", getitem_1476: "f32[]", getitem_1477: "f32[]", getitem_1478: "f32[]", getitem_1479: "f32[]", getitem_1480: "f32[]", getitem_1481: "f32[]", getitem_1482: "f32[]", getitem_1483: "f32[]", getitem_1484: "f32[]", getitem_1485: "f32[]", getitem_1486: "f32[]", getitem_1487: "f32[]", getitem_1488: "f32[]", getitem_1489: "f32[]", getitem_1490: "f32[]", getitem_1491: "f32[]", getitem_1492: "f32[]", getitem_1493: "f32[]", getitem_1494: "f32[]", getitem_1495: "f32[]", getitem_1496: "f32[]", getitem_1497: "f32[]", getitem_1498: "f32[]", getitem_1499: "f32[]", getitem_1500: "f32[]", getitem_1501: "f32[]", getitem_1502: "f32[]", getitem_1503: "f32[]", getitem_1504: "f32[]", getitem_1505: "f32[]", getitem_1506: "f32[]", getitem_1507: "f32[]", getitem_1508: "f32[]", getitem_1509: "f32[]", getitem_1510: "f32[]", getitem_1511: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671], [arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1], [arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1, arg728_1, arg729_1, arg730_1, arg731_1, arg732_1, arg733_1, arg734_1, arg735_1, arg736_1, arg737_1, arg738_1, arg739_1, arg740_1, arg741_1, arg742_1, arg743_1, arg744_1, arg745_1, arg746_1, arg747_1, arg748_1, arg749_1, arg750_1, arg751_1, arg752_1, arg753_1, arg754_1, arg755_1, arg756_1, arg757_1, arg758_1, arg759_1, arg760_1, arg761_1, arg762_1, arg763_1, arg764_1, arg765_1, arg766_1, arg767_1, arg768_1, arg769_1, arg770_1, arg771_1, arg772_1, arg773_1, arg774_1, arg775_1, arg776_1, arg777_1, arg778_1, arg779_1, arg780_1, arg781_1, arg782_1, arg783_1, arg784_1, arg785_1, arg786_1, arg787_1, arg788_1, arg789_1, arg790_1, arg791_1, arg792_1, arg793_1, arg794_1, arg795_1, arg796_1, arg797_1, arg798_1, arg799_1, arg800_1, arg801_1, arg802_1, arg803_1, arg804_1, arg805_1, arg806_1, arg807_1, arg808_1, arg809_1, arg810_1, arg811_1, arg812_1, arg813_1, arg814_1, arg815_1, arg816_1, arg817_1, arg818_1, arg819_1, arg820_1, arg821_1, arg822_1, arg823_1, arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1], 0.0010000000000000009);  getitem_504 = getitem_505 = getitem_506 = getitem_507 = getitem_508 = getitem_509 = getitem_510 = getitem_511 = getitem_512 = getitem_513 = getitem_514 = getitem_515 = getitem_516 = getitem_517 = getitem_518 = getitem_519 = getitem_520 = getitem_521 = getitem_522 = getitem_523 = getitem_524 = getitem_525 = getitem_526 = getitem_527 = getitem_528 = getitem_529 = getitem_530 = getitem_531 = getitem_532 = getitem_533 = getitem_534 = getitem_535 = getitem_536 = getitem_537 = getitem_538 = getitem_539 = getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = getitem_572 = getitem_573 = getitem_574 = getitem_575 = getitem_576 = getitem_577 = getitem_578 = getitem_579 = getitem_580 = getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = getitem_596 = getitem_597 = getitem_598 = getitem_599 = getitem_600 = getitem_601 = getitem_602 = getitem_603 = getitem_604 = getitem_605 = getitem_606 = getitem_607 = getitem_608 = getitem_609 = getitem_610 = getitem_611 = getitem_612 = getitem_613 = getitem_614 = getitem_615 = getitem_616 = getitem_617 = getitem_618 = getitem_619 = getitem_620 = getitem_621 = getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = getitem_646 = getitem_647 = getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = getitem_654 = getitem_655 = getitem_656 = getitem_657 = getitem_658 = getitem_659 = getitem_660 = getitem_661 = getitem_662 = getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = getitem_671 = arg672_1 = arg673_1 = arg674_1 = arg675_1 = arg676_1 = arg677_1 = arg678_1 = arg679_1 = arg680_1 = arg681_1 = arg682_1 = arg683_1 = arg684_1 = arg685_1 = arg686_1 = arg687_1 = arg688_1 = arg689_1 = arg690_1 = arg691_1 = arg692_1 = arg693_1 = arg694_1 = arg695_1 = arg696_1 = arg697_1 = arg698_1 = arg699_1 = arg700_1 = arg701_1 = arg702_1 = arg703_1 = arg704_1 = arg705_1 = arg706_1 = arg707_1 = arg708_1 = arg709_1 = arg710_1 = arg711_1 = arg712_1 = arg713_1 = arg714_1 = arg715_1 = arg716_1 = arg717_1 = arg718_1 = arg719_1 = arg720_1 = arg721_1 = arg722_1 = arg723_1 = arg724_1 = arg725_1 = arg726_1 = arg727_1 = arg728_1 = arg729_1 = arg730_1 = arg731_1 = arg732_1 = arg733_1 = arg734_1 = arg735_1 = arg736_1 = arg737_1 = arg738_1 = arg739_1 = arg740_1 = arg741_1 = arg742_1 = arg743_1 = arg744_1 = arg745_1 = arg746_1 = arg747_1 = arg748_1 = arg749_1 = arg750_1 = arg751_1 = arg752_1 = arg753_1 = arg754_1 = arg755_1 = arg756_1 = arg757_1 = arg758_1 = arg759_1 = arg760_1 = arg761_1 = arg762_1 = arg763_1 = arg764_1 = arg765_1 = arg766_1 = arg767_1 = arg768_1 = arg769_1 = arg770_1 = arg771_1 = arg772_1 = arg773_1 = arg774_1 = arg775_1 = arg776_1 = arg777_1 = arg778_1 = arg779_1 = arg780_1 = arg781_1 = arg782_1 = arg783_1 = arg784_1 = arg785_1 = arg786_1 = arg787_1 = arg788_1 = arg789_1 = arg790_1 = arg791_1 = arg792_1 = arg793_1 = arg794_1 = arg795_1 = arg796_1 = arg797_1 = arg798_1 = arg799_1 = arg800_1 = arg801_1 = arg802_1 = arg803_1 = arg804_1 = arg805_1 = arg806_1 = arg807_1 = arg808_1 = arg809_1 = arg810_1 = arg811_1 = arg812_1 = arg813_1 = arg814_1 = arg815_1 = arg816_1 = arg817_1 = arg818_1 = arg819_1 = arg820_1 = arg821_1 = arg822_1 = arg823_1 = arg824_1 = arg825_1 = arg826_1 = arg827_1 = arg828_1 = arg829_1 = arg830_1 = arg831_1 = arg832_1 = arg833_1 = arg834_1 = arg835_1 = arg836_1 = arg837_1 = arg838_1 = arg839_1 = None
        getitem: "f32[64, 3, 3, 3]" = _foreach_addcmul_scalar[0]
        getitem_672: "f32[64]" = _foreach_addcmul_scalar[1]
        getitem_673: "f32[64]" = _foreach_addcmul_scalar[2]
        getitem_674: "f32[64, 3, 1, 1]" = _foreach_addcmul_scalar[3]
        getitem_675: "f32[64]" = _foreach_addcmul_scalar[4]
        getitem_676: "f32[64]" = _foreach_addcmul_scalar[5]
        getitem_677: "f32[96, 64, 3, 3]" = _foreach_addcmul_scalar[6]
        getitem_678: "f32[96]" = _foreach_addcmul_scalar[7]
        getitem_679: "f32[96]" = _foreach_addcmul_scalar[8]
        getitem_680: "f32[96, 64, 1, 1]" = _foreach_addcmul_scalar[9]
        getitem_681: "f32[96]" = _foreach_addcmul_scalar[10]
        getitem_682: "f32[96]" = _foreach_addcmul_scalar[11]
        getitem_683: "f32[96]" = _foreach_addcmul_scalar[12]
        getitem_684: "f32[96]" = _foreach_addcmul_scalar[13]
        getitem_685: "f32[96, 96, 3, 3]" = _foreach_addcmul_scalar[14]
        getitem_686: "f32[96]" = _foreach_addcmul_scalar[15]
        getitem_687: "f32[96]" = _foreach_addcmul_scalar[16]
        getitem_688: "f32[96, 96, 1, 1]" = _foreach_addcmul_scalar[17]
        getitem_689: "f32[96]" = _foreach_addcmul_scalar[18]
        getitem_690: "f32[96]" = _foreach_addcmul_scalar[19]
        getitem_691: "f32[192, 96, 3, 3]" = _foreach_addcmul_scalar[20]
        getitem_692: "f32[192]" = _foreach_addcmul_scalar[21]
        getitem_693: "f32[192]" = _foreach_addcmul_scalar[22]
        getitem_694: "f32[192, 96, 1, 1]" = _foreach_addcmul_scalar[23]
        getitem_695: "f32[192]" = _foreach_addcmul_scalar[24]
        getitem_696: "f32[192]" = _foreach_addcmul_scalar[25]
        getitem_697: "f32[192]" = _foreach_addcmul_scalar[26]
        getitem_698: "f32[192]" = _foreach_addcmul_scalar[27]
        getitem_699: "f32[192, 192, 3, 3]" = _foreach_addcmul_scalar[28]
        getitem_700: "f32[192]" = _foreach_addcmul_scalar[29]
        getitem_701: "f32[192]" = _foreach_addcmul_scalar[30]
        getitem_702: "f32[192, 192, 1, 1]" = _foreach_addcmul_scalar[31]
        getitem_703: "f32[192]" = _foreach_addcmul_scalar[32]
        getitem_704: "f32[192]" = _foreach_addcmul_scalar[33]
        getitem_705: "f32[192]" = _foreach_addcmul_scalar[34]
        getitem_706: "f32[192]" = _foreach_addcmul_scalar[35]
        getitem_707: "f32[192, 192, 3, 3]" = _foreach_addcmul_scalar[36]
        getitem_708: "f32[192]" = _foreach_addcmul_scalar[37]
        getitem_709: "f32[192]" = _foreach_addcmul_scalar[38]
        getitem_710: "f32[192, 192, 1, 1]" = _foreach_addcmul_scalar[39]
        getitem_711: "f32[192]" = _foreach_addcmul_scalar[40]
        getitem_712: "f32[192]" = _foreach_addcmul_scalar[41]
        getitem_713: "f32[192]" = _foreach_addcmul_scalar[42]
        getitem_714: "f32[192]" = _foreach_addcmul_scalar[43]
        getitem_715: "f32[192, 192, 3, 3]" = _foreach_addcmul_scalar[44]
        getitem_716: "f32[192]" = _foreach_addcmul_scalar[45]
        getitem_717: "f32[192]" = _foreach_addcmul_scalar[46]
        getitem_718: "f32[192, 192, 1, 1]" = _foreach_addcmul_scalar[47]
        getitem_719: "f32[192]" = _foreach_addcmul_scalar[48]
        getitem_720: "f32[192]" = _foreach_addcmul_scalar[49]
        getitem_721: "f32[384, 192, 3, 3]" = _foreach_addcmul_scalar[50]
        getitem_722: "f32[384]" = _foreach_addcmul_scalar[51]
        getitem_723: "f32[384]" = _foreach_addcmul_scalar[52]
        getitem_724: "f32[384, 192, 1, 1]" = _foreach_addcmul_scalar[53]
        getitem_725: "f32[384]" = _foreach_addcmul_scalar[54]
        getitem_726: "f32[384]" = _foreach_addcmul_scalar[55]
        getitem_727: "f32[384]" = _foreach_addcmul_scalar[56]
        getitem_728: "f32[384]" = _foreach_addcmul_scalar[57]
        getitem_729: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[58]
        getitem_730: "f32[384]" = _foreach_addcmul_scalar[59]
        getitem_731: "f32[384]" = _foreach_addcmul_scalar[60]
        getitem_732: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[61]
        getitem_733: "f32[384]" = _foreach_addcmul_scalar[62]
        getitem_734: "f32[384]" = _foreach_addcmul_scalar[63]
        getitem_735: "f32[384]" = _foreach_addcmul_scalar[64]
        getitem_736: "f32[384]" = _foreach_addcmul_scalar[65]
        getitem_737: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[66]
        getitem_738: "f32[384]" = _foreach_addcmul_scalar[67]
        getitem_739: "f32[384]" = _foreach_addcmul_scalar[68]
        getitem_740: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[69]
        getitem_741: "f32[384]" = _foreach_addcmul_scalar[70]
        getitem_742: "f32[384]" = _foreach_addcmul_scalar[71]
        getitem_743: "f32[384]" = _foreach_addcmul_scalar[72]
        getitem_744: "f32[384]" = _foreach_addcmul_scalar[73]
        getitem_745: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[74]
        getitem_746: "f32[384]" = _foreach_addcmul_scalar[75]
        getitem_747: "f32[384]" = _foreach_addcmul_scalar[76]
        getitem_748: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[77]
        getitem_749: "f32[384]" = _foreach_addcmul_scalar[78]
        getitem_750: "f32[384]" = _foreach_addcmul_scalar[79]
        getitem_751: "f32[384]" = _foreach_addcmul_scalar[80]
        getitem_752: "f32[384]" = _foreach_addcmul_scalar[81]
        getitem_753: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[82]
        getitem_754: "f32[384]" = _foreach_addcmul_scalar[83]
        getitem_755: "f32[384]" = _foreach_addcmul_scalar[84]
        getitem_756: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[85]
        getitem_757: "f32[384]" = _foreach_addcmul_scalar[86]
        getitem_758: "f32[384]" = _foreach_addcmul_scalar[87]
        getitem_759: "f32[384]" = _foreach_addcmul_scalar[88]
        getitem_760: "f32[384]" = _foreach_addcmul_scalar[89]
        getitem_761: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[90]
        getitem_762: "f32[384]" = _foreach_addcmul_scalar[91]
        getitem_763: "f32[384]" = _foreach_addcmul_scalar[92]
        getitem_764: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[93]
        getitem_765: "f32[384]" = _foreach_addcmul_scalar[94]
        getitem_766: "f32[384]" = _foreach_addcmul_scalar[95]
        getitem_767: "f32[384]" = _foreach_addcmul_scalar[96]
        getitem_768: "f32[384]" = _foreach_addcmul_scalar[97]
        getitem_769: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[98]
        getitem_770: "f32[384]" = _foreach_addcmul_scalar[99]
        getitem_771: "f32[384]" = _foreach_addcmul_scalar[100]
        getitem_772: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[101]
        getitem_773: "f32[384]" = _foreach_addcmul_scalar[102]
        getitem_774: "f32[384]" = _foreach_addcmul_scalar[103]
        getitem_775: "f32[384]" = _foreach_addcmul_scalar[104]
        getitem_776: "f32[384]" = _foreach_addcmul_scalar[105]
        getitem_777: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[106]
        getitem_778: "f32[384]" = _foreach_addcmul_scalar[107]
        getitem_779: "f32[384]" = _foreach_addcmul_scalar[108]
        getitem_780: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[109]
        getitem_781: "f32[384]" = _foreach_addcmul_scalar[110]
        getitem_782: "f32[384]" = _foreach_addcmul_scalar[111]
        getitem_783: "f32[384]" = _foreach_addcmul_scalar[112]
        getitem_784: "f32[384]" = _foreach_addcmul_scalar[113]
        getitem_785: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[114]
        getitem_786: "f32[384]" = _foreach_addcmul_scalar[115]
        getitem_787: "f32[384]" = _foreach_addcmul_scalar[116]
        getitem_788: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[117]
        getitem_789: "f32[384]" = _foreach_addcmul_scalar[118]
        getitem_790: "f32[384]" = _foreach_addcmul_scalar[119]
        getitem_791: "f32[384]" = _foreach_addcmul_scalar[120]
        getitem_792: "f32[384]" = _foreach_addcmul_scalar[121]
        getitem_793: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[122]
        getitem_794: "f32[384]" = _foreach_addcmul_scalar[123]
        getitem_795: "f32[384]" = _foreach_addcmul_scalar[124]
        getitem_796: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[125]
        getitem_797: "f32[384]" = _foreach_addcmul_scalar[126]
        getitem_798: "f32[384]" = _foreach_addcmul_scalar[127]
        getitem_799: "f32[384]" = _foreach_addcmul_scalar[128]
        getitem_800: "f32[384]" = _foreach_addcmul_scalar[129]
        getitem_801: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[130]
        getitem_802: "f32[384]" = _foreach_addcmul_scalar[131]
        getitem_803: "f32[384]" = _foreach_addcmul_scalar[132]
        getitem_804: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[133]
        getitem_805: "f32[384]" = _foreach_addcmul_scalar[134]
        getitem_806: "f32[384]" = _foreach_addcmul_scalar[135]
        getitem_807: "f32[384]" = _foreach_addcmul_scalar[136]
        getitem_808: "f32[384]" = _foreach_addcmul_scalar[137]
        getitem_809: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[138]
        getitem_810: "f32[384]" = _foreach_addcmul_scalar[139]
        getitem_811: "f32[384]" = _foreach_addcmul_scalar[140]
        getitem_812: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[141]
        getitem_813: "f32[384]" = _foreach_addcmul_scalar[142]
        getitem_814: "f32[384]" = _foreach_addcmul_scalar[143]
        getitem_815: "f32[384]" = _foreach_addcmul_scalar[144]
        getitem_816: "f32[384]" = _foreach_addcmul_scalar[145]
        getitem_817: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[146]
        getitem_818: "f32[384]" = _foreach_addcmul_scalar[147]
        getitem_819: "f32[384]" = _foreach_addcmul_scalar[148]
        getitem_820: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[149]
        getitem_821: "f32[384]" = _foreach_addcmul_scalar[150]
        getitem_822: "f32[384]" = _foreach_addcmul_scalar[151]
        getitem_823: "f32[384]" = _foreach_addcmul_scalar[152]
        getitem_824: "f32[384]" = _foreach_addcmul_scalar[153]
        getitem_825: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[154]
        getitem_826: "f32[384]" = _foreach_addcmul_scalar[155]
        getitem_827: "f32[384]" = _foreach_addcmul_scalar[156]
        getitem_828: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[157]
        getitem_829: "f32[384]" = _foreach_addcmul_scalar[158]
        getitem_830: "f32[384]" = _foreach_addcmul_scalar[159]
        getitem_831: "f32[1408, 384, 3, 3]" = _foreach_addcmul_scalar[160]
        getitem_832: "f32[1408]" = _foreach_addcmul_scalar[161]
        getitem_833: "f32[1408]" = _foreach_addcmul_scalar[162]
        getitem_834: "f32[1408, 384, 1, 1]" = _foreach_addcmul_scalar[163]
        getitem_835: "f32[1408]" = _foreach_addcmul_scalar[164]
        getitem_836: "f32[1408]" = _foreach_addcmul_scalar[165]
        getitem_837: "f32[1000, 1408]" = _foreach_addcmul_scalar[166]
        getitem_838: "f32[1000]" = _foreach_addcmul_scalar[167];  _foreach_addcmul_scalar = None
        getitem_839 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_839, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167]);  getitem_839 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = None
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
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511]);  getitem_1344 = getitem_1345 = getitem_1346 = getitem_1347 = getitem_1348 = getitem_1349 = getitem_1350 = getitem_1351 = getitem_1352 = getitem_1353 = getitem_1354 = getitem_1355 = getitem_1356 = getitem_1357 = getitem_1358 = getitem_1359 = getitem_1360 = getitem_1361 = getitem_1362 = getitem_1363 = getitem_1364 = getitem_1365 = getitem_1366 = getitem_1367 = getitem_1368 = getitem_1369 = getitem_1370 = getitem_1371 = getitem_1372 = getitem_1373 = getitem_1374 = getitem_1375 = getitem_1376 = getitem_1377 = getitem_1378 = getitem_1379 = getitem_1380 = getitem_1381 = getitem_1382 = getitem_1383 = getitem_1384 = getitem_1385 = getitem_1386 = getitem_1387 = getitem_1388 = getitem_1389 = getitem_1390 = getitem_1391 = getitem_1392 = getitem_1393 = getitem_1394 = getitem_1395 = getitem_1396 = getitem_1397 = getitem_1398 = getitem_1399 = getitem_1400 = getitem_1401 = getitem_1402 = getitem_1403 = getitem_1404 = getitem_1405 = getitem_1406 = getitem_1407 = getitem_1408 = getitem_1409 = getitem_1410 = getitem_1411 = getitem_1412 = getitem_1413 = getitem_1414 = getitem_1415 = getitem_1416 = getitem_1417 = getitem_1418 = getitem_1419 = getitem_1420 = getitem_1421 = getitem_1422 = getitem_1423 = getitem_1424 = getitem_1425 = getitem_1426 = getitem_1427 = getitem_1428 = getitem_1429 = getitem_1430 = getitem_1431 = getitem_1432 = getitem_1433 = getitem_1434 = getitem_1435 = getitem_1436 = getitem_1437 = getitem_1438 = getitem_1439 = getitem_1440 = getitem_1441 = getitem_1442 = getitem_1443 = getitem_1444 = getitem_1445 = getitem_1446 = getitem_1447 = getitem_1448 = getitem_1449 = getitem_1450 = getitem_1451 = getitem_1452 = getitem_1453 = getitem_1454 = getitem_1455 = getitem_1456 = getitem_1457 = getitem_1458 = getitem_1459 = getitem_1460 = getitem_1461 = getitem_1462 = getitem_1463 = getitem_1464 = getitem_1465 = getitem_1466 = getitem_1467 = getitem_1468 = getitem_1469 = getitem_1470 = getitem_1471 = getitem_1472 = getitem_1473 = getitem_1474 = getitem_1475 = getitem_1476 = getitem_1477 = getitem_1478 = getitem_1479 = getitem_1480 = getitem_1481 = getitem_1482 = getitem_1483 = getitem_1484 = getitem_1485 = getitem_1486 = getitem_1487 = getitem_1488 = getitem_1489 = getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = None
        getitem_1512: "f32[]" = _foreach_neg_default[0]
        getitem_1513: "f32[]" = _foreach_neg_default[1]
        getitem_1514: "f32[]" = _foreach_neg_default[2]
        getitem_1515: "f32[]" = _foreach_neg_default[3]
        getitem_1516: "f32[]" = _foreach_neg_default[4]
        getitem_1517: "f32[]" = _foreach_neg_default[5]
        getitem_1518: "f32[]" = _foreach_neg_default[6]
        getitem_1519: "f32[]" = _foreach_neg_default[7]
        getitem_1520: "f32[]" = _foreach_neg_default[8]
        getitem_1521: "f32[]" = _foreach_neg_default[9]
        getitem_1522: "f32[]" = _foreach_neg_default[10]
        getitem_1523: "f32[]" = _foreach_neg_default[11]
        getitem_1524: "f32[]" = _foreach_neg_default[12]
        getitem_1525: "f32[]" = _foreach_neg_default[13]
        getitem_1526: "f32[]" = _foreach_neg_default[14]
        getitem_1527: "f32[]" = _foreach_neg_default[15]
        getitem_1528: "f32[]" = _foreach_neg_default[16]
        getitem_1529: "f32[]" = _foreach_neg_default[17]
        getitem_1530: "f32[]" = _foreach_neg_default[18]
        getitem_1531: "f32[]" = _foreach_neg_default[19]
        getitem_1532: "f32[]" = _foreach_neg_default[20]
        getitem_1533: "f32[]" = _foreach_neg_default[21]
        getitem_1534: "f32[]" = _foreach_neg_default[22]
        getitem_1535: "f32[]" = _foreach_neg_default[23]
        getitem_1536: "f32[]" = _foreach_neg_default[24]
        getitem_1537: "f32[]" = _foreach_neg_default[25]
        getitem_1538: "f32[]" = _foreach_neg_default[26]
        getitem_1539: "f32[]" = _foreach_neg_default[27]
        getitem_1540: "f32[]" = _foreach_neg_default[28]
        getitem_1541: "f32[]" = _foreach_neg_default[29]
        getitem_1542: "f32[]" = _foreach_neg_default[30]
        getitem_1543: "f32[]" = _foreach_neg_default[31]
        getitem_1544: "f32[]" = _foreach_neg_default[32]
        getitem_1545: "f32[]" = _foreach_neg_default[33]
        getitem_1546: "f32[]" = _foreach_neg_default[34]
        getitem_1547: "f32[]" = _foreach_neg_default[35]
        getitem_1548: "f32[]" = _foreach_neg_default[36]
        getitem_1549: "f32[]" = _foreach_neg_default[37]
        getitem_1550: "f32[]" = _foreach_neg_default[38]
        getitem_1551: "f32[]" = _foreach_neg_default[39]
        getitem_1552: "f32[]" = _foreach_neg_default[40]
        getitem_1553: "f32[]" = _foreach_neg_default[41]
        getitem_1554: "f32[]" = _foreach_neg_default[42]
        getitem_1555: "f32[]" = _foreach_neg_default[43]
        getitem_1556: "f32[]" = _foreach_neg_default[44]
        getitem_1557: "f32[]" = _foreach_neg_default[45]
        getitem_1558: "f32[]" = _foreach_neg_default[46]
        getitem_1559: "f32[]" = _foreach_neg_default[47]
        getitem_1560: "f32[]" = _foreach_neg_default[48]
        getitem_1561: "f32[]" = _foreach_neg_default[49]
        getitem_1562: "f32[]" = _foreach_neg_default[50]
        getitem_1563: "f32[]" = _foreach_neg_default[51]
        getitem_1564: "f32[]" = _foreach_neg_default[52]
        getitem_1565: "f32[]" = _foreach_neg_default[53]
        getitem_1566: "f32[]" = _foreach_neg_default[54]
        getitem_1567: "f32[]" = _foreach_neg_default[55]
        getitem_1568: "f32[]" = _foreach_neg_default[56]
        getitem_1569: "f32[]" = _foreach_neg_default[57]
        getitem_1570: "f32[]" = _foreach_neg_default[58]
        getitem_1571: "f32[]" = _foreach_neg_default[59]
        getitem_1572: "f32[]" = _foreach_neg_default[60]
        getitem_1573: "f32[]" = _foreach_neg_default[61]
        getitem_1574: "f32[]" = _foreach_neg_default[62]
        getitem_1575: "f32[]" = _foreach_neg_default[63]
        getitem_1576: "f32[]" = _foreach_neg_default[64]
        getitem_1577: "f32[]" = _foreach_neg_default[65]
        getitem_1578: "f32[]" = _foreach_neg_default[66]
        getitem_1579: "f32[]" = _foreach_neg_default[67]
        getitem_1580: "f32[]" = _foreach_neg_default[68]
        getitem_1581: "f32[]" = _foreach_neg_default[69]
        getitem_1582: "f32[]" = _foreach_neg_default[70]
        getitem_1583: "f32[]" = _foreach_neg_default[71]
        getitem_1584: "f32[]" = _foreach_neg_default[72]
        getitem_1585: "f32[]" = _foreach_neg_default[73]
        getitem_1586: "f32[]" = _foreach_neg_default[74]
        getitem_1587: "f32[]" = _foreach_neg_default[75]
        getitem_1588: "f32[]" = _foreach_neg_default[76]
        getitem_1589: "f32[]" = _foreach_neg_default[77]
        getitem_1590: "f32[]" = _foreach_neg_default[78]
        getitem_1591: "f32[]" = _foreach_neg_default[79]
        getitem_1592: "f32[]" = _foreach_neg_default[80]
        getitem_1593: "f32[]" = _foreach_neg_default[81]
        getitem_1594: "f32[]" = _foreach_neg_default[82]
        getitem_1595: "f32[]" = _foreach_neg_default[83]
        getitem_1596: "f32[]" = _foreach_neg_default[84]
        getitem_1597: "f32[]" = _foreach_neg_default[85]
        getitem_1598: "f32[]" = _foreach_neg_default[86]
        getitem_1599: "f32[]" = _foreach_neg_default[87]
        getitem_1600: "f32[]" = _foreach_neg_default[88]
        getitem_1601: "f32[]" = _foreach_neg_default[89]
        getitem_1602: "f32[]" = _foreach_neg_default[90]
        getitem_1603: "f32[]" = _foreach_neg_default[91]
        getitem_1604: "f32[]" = _foreach_neg_default[92]
        getitem_1605: "f32[]" = _foreach_neg_default[93]
        getitem_1606: "f32[]" = _foreach_neg_default[94]
        getitem_1607: "f32[]" = _foreach_neg_default[95]
        getitem_1608: "f32[]" = _foreach_neg_default[96]
        getitem_1609: "f32[]" = _foreach_neg_default[97]
        getitem_1610: "f32[]" = _foreach_neg_default[98]
        getitem_1611: "f32[]" = _foreach_neg_default[99]
        getitem_1612: "f32[]" = _foreach_neg_default[100]
        getitem_1613: "f32[]" = _foreach_neg_default[101]
        getitem_1614: "f32[]" = _foreach_neg_default[102]
        getitem_1615: "f32[]" = _foreach_neg_default[103]
        getitem_1616: "f32[]" = _foreach_neg_default[104]
        getitem_1617: "f32[]" = _foreach_neg_default[105]
        getitem_1618: "f32[]" = _foreach_neg_default[106]
        getitem_1619: "f32[]" = _foreach_neg_default[107]
        getitem_1620: "f32[]" = _foreach_neg_default[108]
        getitem_1621: "f32[]" = _foreach_neg_default[109]
        getitem_1622: "f32[]" = _foreach_neg_default[110]
        getitem_1623: "f32[]" = _foreach_neg_default[111]
        getitem_1624: "f32[]" = _foreach_neg_default[112]
        getitem_1625: "f32[]" = _foreach_neg_default[113]
        getitem_1626: "f32[]" = _foreach_neg_default[114]
        getitem_1627: "f32[]" = _foreach_neg_default[115]
        getitem_1628: "f32[]" = _foreach_neg_default[116]
        getitem_1629: "f32[]" = _foreach_neg_default[117]
        getitem_1630: "f32[]" = _foreach_neg_default[118]
        getitem_1631: "f32[]" = _foreach_neg_default[119]
        getitem_1632: "f32[]" = _foreach_neg_default[120]
        getitem_1633: "f32[]" = _foreach_neg_default[121]
        getitem_1634: "f32[]" = _foreach_neg_default[122]
        getitem_1635: "f32[]" = _foreach_neg_default[123]
        getitem_1636: "f32[]" = _foreach_neg_default[124]
        getitem_1637: "f32[]" = _foreach_neg_default[125]
        getitem_1638: "f32[]" = _foreach_neg_default[126]
        getitem_1639: "f32[]" = _foreach_neg_default[127]
        getitem_1640: "f32[]" = _foreach_neg_default[128]
        getitem_1641: "f32[]" = _foreach_neg_default[129]
        getitem_1642: "f32[]" = _foreach_neg_default[130]
        getitem_1643: "f32[]" = _foreach_neg_default[131]
        getitem_1644: "f32[]" = _foreach_neg_default[132]
        getitem_1645: "f32[]" = _foreach_neg_default[133]
        getitem_1646: "f32[]" = _foreach_neg_default[134]
        getitem_1647: "f32[]" = _foreach_neg_default[135]
        getitem_1648: "f32[]" = _foreach_neg_default[136]
        getitem_1649: "f32[]" = _foreach_neg_default[137]
        getitem_1650: "f32[]" = _foreach_neg_default[138]
        getitem_1651: "f32[]" = _foreach_neg_default[139]
        getitem_1652: "f32[]" = _foreach_neg_default[140]
        getitem_1653: "f32[]" = _foreach_neg_default[141]
        getitem_1654: "f32[]" = _foreach_neg_default[142]
        getitem_1655: "f32[]" = _foreach_neg_default[143]
        getitem_1656: "f32[]" = _foreach_neg_default[144]
        getitem_1657: "f32[]" = _foreach_neg_default[145]
        getitem_1658: "f32[]" = _foreach_neg_default[146]
        getitem_1659: "f32[]" = _foreach_neg_default[147]
        getitem_1660: "f32[]" = _foreach_neg_default[148]
        getitem_1661: "f32[]" = _foreach_neg_default[149]
        getitem_1662: "f32[]" = _foreach_neg_default[150]
        getitem_1663: "f32[]" = _foreach_neg_default[151]
        getitem_1664: "f32[]" = _foreach_neg_default[152]
        getitem_1665: "f32[]" = _foreach_neg_default[153]
        getitem_1666: "f32[]" = _foreach_neg_default[154]
        getitem_1667: "f32[]" = _foreach_neg_default[155]
        getitem_1668: "f32[]" = _foreach_neg_default[156]
        getitem_1669: "f32[]" = _foreach_neg_default[157]
        getitem_1670: "f32[]" = _foreach_neg_default[158]
        getitem_1671: "f32[]" = _foreach_neg_default[159]
        getitem_1672: "f32[]" = _foreach_neg_default[160]
        getitem_1673: "f32[]" = _foreach_neg_default[161]
        getitem_1674: "f32[]" = _foreach_neg_default[162]
        getitem_1675: "f32[]" = _foreach_neg_default[163]
        getitem_1676: "f32[]" = _foreach_neg_default[164]
        getitem_1677: "f32[]" = _foreach_neg_default[165]
        getitem_1678: "f32[]" = _foreach_neg_default[166]
        getitem_1679: "f32[]" = _foreach_neg_default[167];  _foreach_neg_default = None
        return (getitem, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679)


def _default_make_inputs():
    return [
    torch.randn([64, 3, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64, 3, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([64], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 64, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([96], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 96, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 192, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([384], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 3, 3], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408, 384, 1, 1], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 1408], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.tensor(1),  # getitem_839 (unknown shape)
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

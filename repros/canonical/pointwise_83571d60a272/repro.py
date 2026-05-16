"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: 83571d60a272
Shape hash: ee4bf60d
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_655: "f32[]", getitem_656: "f32[]", getitem_657: "f32[]", getitem_658: "f32[]", getitem_659: "f32[]", getitem_660: "f32[]", getitem_661: "f32[]", getitem_662: "f32[]", getitem_663: "f32[]", getitem_664: "f32[]", getitem_665: "f32[]", getitem_666: "f32[]", getitem_667: "f32[]", getitem_668: "f32[]", getitem_669: "f32[]", getitem_670: "f32[]", getitem_671: "f32[]", getitem_672: "f32[]", getitem_673: "f32[]", getitem_674: "f32[]", getitem_675: "f32[]", getitem_676: "f32[]", getitem_677: "f32[]", getitem_678: "f32[]", getitem_679: "f32[]", getitem_680: "f32[]", getitem_681: "f32[]", getitem_682: "f32[]", getitem_683: "f32[]", getitem_684: "f32[]", getitem_685: "f32[]", getitem_686: "f32[]", getitem_687: "f32[]", getitem_688: "f32[]", getitem_689: "f32[]", getitem_690: "f32[]", getitem_691: "f32[]", getitem_692: "f32[]", getitem_693: "f32[]", getitem_694: "f32[]", getitem_695: "f32[]", getitem_696: "f32[]", getitem_697: "f32[]", getitem_698: "f32[]", getitem_699: "f32[]", getitem_700: "f32[]", getitem_701: "f32[]", getitem_702: "f32[]", getitem_703: "f32[]", getitem_704: "f32[]", getitem_705: "f32[]", getitem_706: "f32[]", getitem_707: "f32[]", getitem_708: "f32[]", getitem_709: "f32[]", getitem_710: "f32[]", getitem_711: "f32[]", getitem_712: "f32[]", getitem_713: "f32[]", getitem_714: "f32[]", getitem_715: "f32[]", getitem_716: "f32[]", getitem_717: "f32[]", getitem_718: "f32[]", getitem_719: "f32[]", getitem_720: "f32[]", getitem_721: "f32[]", getitem_722: "f32[]", getitem_723: "f32[]", getitem_724: "f32[]", getitem_725: "f32[]", getitem_726: "f32[]", getitem_727: "f32[]", getitem_728: "f32[]", getitem_729: "f32[]", getitem_730: "f32[]", getitem_731: "f32[]", getitem_732: "f32[]", getitem_733: "f32[]", getitem_734: "f32[]", getitem_735: "f32[]", getitem_736: "f32[]", getitem_737: "f32[]", getitem_738: "f32[]", getitem_739: "f32[]", getitem_740: "f32[]", getitem_741: "f32[]", getitem_742: "f32[]", getitem_743: "f32[]", getitem_744: "f32[]", getitem_745: "f32[]", getitem_746: "f32[]", getitem_747: "f32[]", getitem_748: "f32[]", getitem_749: "f32[]", getitem_750: "f32[]", getitem_751: "f32[]", getitem_752: "f32[]", getitem_753: "f32[]", getitem_754: "f32[]", getitem_755: "f32[]", getitem_756: "f32[]", getitem_757: "f32[]", getitem_758: "f32[]", getitem_759: "f32[]", getitem_760: "f32[]", getitem_761: "f32[]", getitem_762: "f32[]", getitem_763: "f32[]", getitem_764: "f32[]", getitem_765: "f32[]", getitem_766: "f32[]", getitem_767: "f32[]", getitem_768: "f32[]", getitem_769: "f32[]", getitem_770: "f32[]", getitem_771: "f32[]", getitem_772: "f32[]", getitem_773: "f32[]", getitem_774: "f32[]", getitem_775: "f32[]", getitem_776: "f32[]", getitem_777: "f32[]", getitem_778: "f32[]", getitem_779: "f32[]", getitem_780: "f32[]", getitem_781: "f32[]", getitem_782: "f32[]", getitem_783: "f32[]", getitem_784: "f32[]", getitem_785: "f32[]", getitem_1179: "f32[]", getitem_1180: "f32[]", getitem_1181: "f32[]", getitem_1182: "f32[]", getitem_1183: "f32[]", getitem_1184: "f32[]", getitem_1185: "f32[]", getitem_1186: "f32[]", getitem_1187: "f32[]", getitem_1188: "f32[]", getitem_1189: "f32[]", getitem_1190: "f32[]", getitem_1191: "f32[]", getitem_1192: "f32[]", getitem_1193: "f32[]", getitem_1194: "f32[]", getitem_1195: "f32[]", getitem_1196: "f32[]", getitem_1197: "f32[]", getitem_1198: "f32[]", getitem_1199: "f32[]", getitem_1200: "f32[]", getitem_1201: "f32[]", getitem_1202: "f32[]", getitem_1203: "f32[]", getitem_1204: "f32[]", getitem_1205: "f32[]", getitem_1206: "f32[]", getitem_1207: "f32[]", getitem_1208: "f32[]", getitem_1209: "f32[]", getitem_1210: "f32[]", getitem_1211: "f32[]", getitem_1212: "f32[]", getitem_1213: "f32[]", getitem_1214: "f32[]", getitem_1215: "f32[]", getitem_1216: "f32[]", getitem_1217: "f32[]", getitem_1218: "f32[]", getitem_1219: "f32[]", getitem_1220: "f32[]", getitem_1221: "f32[]", getitem_1222: "f32[]", getitem_1223: "f32[]", getitem_1224: "f32[]", getitem_1225: "f32[]", getitem_1226: "f32[]", getitem_1227: "f32[]", getitem_1228: "f32[]", getitem_1229: "f32[]", getitem_1230: "f32[]", getitem_1231: "f32[]", getitem_1232: "f32[]", getitem_1233: "f32[]", getitem_1234: "f32[]", getitem_1235: "f32[]", getitem_1236: "f32[]", getitem_1237: "f32[]", getitem_1238: "f32[]", getitem_1239: "f32[]", getitem_1240: "f32[]", getitem_1241: "f32[]", getitem_1242: "f32[]", getitem_1243: "f32[]", getitem_1244: "f32[]", getitem_1245: "f32[]", getitem_1246: "f32[]", getitem_1247: "f32[]", getitem_1248: "f32[]", getitem_1249: "f32[]", getitem_1250: "f32[]", getitem_1251: "f32[]", getitem_1252: "f32[]", getitem_1253: "f32[]", getitem_1254: "f32[]", getitem_1255: "f32[]", getitem_1256: "f32[]", getitem_1257: "f32[]", getitem_1258: "f32[]", getitem_1259: "f32[]", getitem_1260: "f32[]", getitem_1261: "f32[]", getitem_1262: "f32[]", getitem_1263: "f32[]", getitem_1264: "f32[]", getitem_1265: "f32[]", getitem_1266: "f32[]", getitem_1267: "f32[]", getitem_1268: "f32[]", getitem_1269: "f32[]", getitem_1270: "f32[]", getitem_1271: "f32[]", getitem_1272: "f32[]", getitem_1273: "f32[]", getitem_1274: "f32[]", getitem_1275: "f32[]", getitem_1276: "f32[]", getitem_1277: "f32[]", getitem_1278: "f32[]", getitem_1279: "f32[]", getitem_1280: "f32[]", getitem_1281: "f32[]", getitem_1282: "f32[]", getitem_1283: "f32[]", getitem_1284: "f32[]", getitem_1285: "f32[]", getitem_1286: "f32[]", getitem_1287: "f32[]", getitem_1288: "f32[]", getitem_1289: "f32[]", getitem_1290: "f32[]", getitem_1291: "f32[]", getitem_1292: "f32[]", getitem_1293: "f32[]", getitem_1294: "f32[]", getitem_1295: "f32[]", getitem_1296: "f32[]", getitem_1297: "f32[]", getitem_1298: "f32[]", getitem_1299: "f32[]", getitem_1300: "f32[]", getitem_1301: "f32[]", getitem_1302: "f32[]", getitem_1303: "f32[]", getitem_1304: "f32[]", getitem_1305: "f32[]", getitem_1306: "f32[]", getitem_1307: "f32[]", getitem_1308: "f32[]", getitem_1309: "f32[]", getitem_524: "f32[32128, 512]", getitem_525: "f32[512, 512]", getitem_526: "f32[512, 512]", getitem_527: "f32[512, 512]", getitem_528: "f32[512, 512]", getitem_529: "f32[32, 8]", getitem_530: "f32[512]", getitem_531: "f32[2048, 512]", getitem_532: "f32[512, 2048]", getitem_533: "f32[512]", getitem_534: "f32[512, 512]", getitem_535: "f32[512, 512]", getitem_536: "f32[512, 512]", getitem_537: "f32[512, 512]", getitem_538: "f32[512]", getitem_539: "f32[2048, 512]", getitem_540: "f32[512, 2048]", getitem_541: "f32[512]", getitem_542: "f32[512, 512]", getitem_543: "f32[512, 512]", getitem_544: "f32[512, 512]", getitem_545: "f32[512, 512]", getitem_546: "f32[512]", getitem_547: "f32[2048, 512]", getitem_548: "f32[512, 2048]", getitem_549: "f32[512]", getitem_550: "f32[512, 512]", getitem_551: "f32[512, 512]", getitem_552: "f32[512, 512]", getitem_553: "f32[512, 512]", getitem_554: "f32[512]", getitem_555: "f32[2048, 512]", getitem_556: "f32[512, 2048]", getitem_557: "f32[512]", getitem_558: "f32[512, 512]", getitem_559: "f32[512, 512]", getitem_560: "f32[512, 512]", getitem_561: "f32[512, 512]", getitem_562: "f32[512]", getitem_563: "f32[2048, 512]", getitem_564: "f32[512, 2048]", getitem_565: "f32[512]", getitem_566: "f32[512, 512]", getitem_567: "f32[512, 512]", getitem_568: "f32[512, 512]", getitem_569: "f32[512, 512]", getitem_570: "f32[512]", getitem_571: "f32[2048, 512]", getitem_572: "f32[512, 2048]", getitem_573: "f32[512]", getitem_574: "f32[512]", getitem_575: "f32[512, 512]", getitem_576: "f32[512, 512]", getitem_577: "f32[512, 512]", getitem_578: "f32[512, 512]", getitem_579: "f32[32, 8]", getitem_580: "f32[512]", getitem_581: "f32[512, 512]", getitem_582: "f32[512, 512]", getitem_583: "f32[512, 512]", getitem_584: "f32[512, 512]", getitem_585: "f32[512]", getitem_586: "f32[2048, 512]", getitem_587: "f32[512, 2048]", getitem_588: "f32[512]", getitem_589: "f32[512, 512]", getitem_590: "f32[512, 512]", getitem_591: "f32[512, 512]", getitem_592: "f32[512, 512]", getitem_593: "f32[512]", getitem_594: "f32[512, 512]", getitem_595: "f32[512, 512]", getitem_596: "f32[512, 512]", getitem_597: "f32[512, 512]", getitem_598: "f32[512]", getitem_599: "f32[2048, 512]", getitem_600: "f32[512, 2048]", getitem_601: "f32[512]", getitem_602: "f32[512, 512]", getitem_603: "f32[512, 512]", getitem_604: "f32[512, 512]", getitem_605: "f32[512, 512]", getitem_606: "f32[512]", getitem_607: "f32[512, 512]", getitem_608: "f32[512, 512]", getitem_609: "f32[512, 512]", getitem_610: "f32[512, 512]", getitem_611: "f32[512]", getitem_612: "f32[2048, 512]", getitem_613: "f32[512, 2048]", getitem_614: "f32[512]", getitem_615: "f32[512, 512]", getitem_616: "f32[512, 512]", getitem_617: "f32[512, 512]", getitem_618: "f32[512, 512]", getitem_619: "f32[512]", getitem_620: "f32[512, 512]", getitem_621: "f32[512, 512]", getitem_622: "f32[512, 512]", getitem_623: "f32[512, 512]", getitem_624: "f32[512]", getitem_625: "f32[2048, 512]", getitem_626: "f32[512, 2048]", getitem_627: "f32[512]", getitem_628: "f32[512, 512]", getitem_629: "f32[512, 512]", getitem_630: "f32[512, 512]", getitem_631: "f32[512, 512]", getitem_632: "f32[512]", getitem_633: "f32[512, 512]", getitem_634: "f32[512, 512]", getitem_635: "f32[512, 512]", getitem_636: "f32[512, 512]", getitem_637: "f32[512]", getitem_638: "f32[2048, 512]", getitem_639: "f32[512, 2048]", getitem_640: "f32[512]", getitem_641: "f32[512, 512]", getitem_642: "f32[512, 512]", getitem_643: "f32[512, 512]", getitem_644: "f32[512, 512]", getitem_645: "f32[512]", getitem_646: "f32[512, 512]", getitem_647: "f32[512, 512]", getitem_648: "f32[512, 512]", getitem_649: "f32[512, 512]", getitem_650: "f32[512]", getitem_651: "f32[2048, 512]", getitem_652: "f32[512, 2048]", getitem_653: "f32[512]", getitem_654: "f32[512]"):
        # No stacktrace found for following nodes
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785], 1);  getitem_655 = getitem_656 = getitem_657 = getitem_658 = getitem_659 = getitem_660 = getitem_661 = getitem_662 = getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = getitem_671 = getitem_672 = getitem_673 = getitem_674 = getitem_675 = getitem_676 = getitem_677 = getitem_678 = getitem_679 = getitem_680 = getitem_681 = getitem_682 = getitem_683 = getitem_684 = getitem_685 = getitem_686 = getitem_687 = getitem_688 = getitem_689 = getitem_690 = getitem_691 = getitem_692 = getitem_693 = getitem_694 = getitem_695 = getitem_696 = getitem_697 = getitem_698 = getitem_699 = getitem_700 = getitem_701 = getitem_702 = getitem_703 = getitem_704 = getitem_705 = getitem_706 = getitem_707 = getitem_708 = getitem_709 = getitem_710 = getitem_711 = getitem_712 = getitem_713 = getitem_714 = getitem_715 = getitem_716 = getitem_717 = getitem_718 = getitem_719 = getitem_720 = getitem_721 = getitem_722 = getitem_723 = getitem_724 = getitem_725 = getitem_726 = getitem_727 = getitem_728 = getitem_729 = getitem_730 = getitem_731 = getitem_732 = getitem_733 = getitem_734 = getitem_735 = getitem_736 = getitem_737 = getitem_738 = getitem_739 = getitem_740 = getitem_741 = getitem_742 = getitem_743 = getitem_744 = getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = getitem_752 = getitem_753 = getitem_754 = getitem_755 = getitem_756 = getitem_757 = getitem_758 = getitem_759 = getitem_760 = getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = getitem_768 = getitem_769 = getitem_770 = getitem_771 = getitem_772 = getitem_773 = getitem_774 = getitem_775 = getitem_776 = getitem_777 = getitem_778 = getitem_779 = getitem_780 = getitem_781 = getitem_782 = getitem_783 = getitem_784 = getitem_785 = None
        getitem: "f32[]" = _foreach_sub_scalar[0]
        getitem_786: "f32[]" = _foreach_sub_scalar[1]
        getitem_787: "f32[]" = _foreach_sub_scalar[2]
        getitem_788: "f32[]" = _foreach_sub_scalar[3]
        getitem_789: "f32[]" = _foreach_sub_scalar[4]
        getitem_790: "f32[]" = _foreach_sub_scalar[5]
        getitem_791: "f32[]" = _foreach_sub_scalar[6]
        getitem_792: "f32[]" = _foreach_sub_scalar[7]
        getitem_793: "f32[]" = _foreach_sub_scalar[8]
        getitem_794: "f32[]" = _foreach_sub_scalar[9]
        getitem_795: "f32[]" = _foreach_sub_scalar[10]
        getitem_796: "f32[]" = _foreach_sub_scalar[11]
        getitem_797: "f32[]" = _foreach_sub_scalar[12]
        getitem_798: "f32[]" = _foreach_sub_scalar[13]
        getitem_799: "f32[]" = _foreach_sub_scalar[14]
        getitem_800: "f32[]" = _foreach_sub_scalar[15]
        getitem_801: "f32[]" = _foreach_sub_scalar[16]
        getitem_802: "f32[]" = _foreach_sub_scalar[17]
        getitem_803: "f32[]" = _foreach_sub_scalar[18]
        getitem_804: "f32[]" = _foreach_sub_scalar[19]
        getitem_805: "f32[]" = _foreach_sub_scalar[20]
        getitem_806: "f32[]" = _foreach_sub_scalar[21]
        getitem_807: "f32[]" = _foreach_sub_scalar[22]
        getitem_808: "f32[]" = _foreach_sub_scalar[23]
        getitem_809: "f32[]" = _foreach_sub_scalar[24]
        getitem_810: "f32[]" = _foreach_sub_scalar[25]
        getitem_811: "f32[]" = _foreach_sub_scalar[26]
        getitem_812: "f32[]" = _foreach_sub_scalar[27]
        getitem_813: "f32[]" = _foreach_sub_scalar[28]
        getitem_814: "f32[]" = _foreach_sub_scalar[29]
        getitem_815: "f32[]" = _foreach_sub_scalar[30]
        getitem_816: "f32[]" = _foreach_sub_scalar[31]
        getitem_817: "f32[]" = _foreach_sub_scalar[32]
        getitem_818: "f32[]" = _foreach_sub_scalar[33]
        getitem_819: "f32[]" = _foreach_sub_scalar[34]
        getitem_820: "f32[]" = _foreach_sub_scalar[35]
        getitem_821: "f32[]" = _foreach_sub_scalar[36]
        getitem_822: "f32[]" = _foreach_sub_scalar[37]
        getitem_823: "f32[]" = _foreach_sub_scalar[38]
        getitem_824: "f32[]" = _foreach_sub_scalar[39]
        getitem_825: "f32[]" = _foreach_sub_scalar[40]
        getitem_826: "f32[]" = _foreach_sub_scalar[41]
        getitem_827: "f32[]" = _foreach_sub_scalar[42]
        getitem_828: "f32[]" = _foreach_sub_scalar[43]
        getitem_829: "f32[]" = _foreach_sub_scalar[44]
        getitem_830: "f32[]" = _foreach_sub_scalar[45]
        getitem_831: "f32[]" = _foreach_sub_scalar[46]
        getitem_832: "f32[]" = _foreach_sub_scalar[47]
        getitem_833: "f32[]" = _foreach_sub_scalar[48]
        getitem_834: "f32[]" = _foreach_sub_scalar[49]
        getitem_835: "f32[]" = _foreach_sub_scalar[50]
        getitem_836: "f32[]" = _foreach_sub_scalar[51]
        getitem_837: "f32[]" = _foreach_sub_scalar[52]
        getitem_838: "f32[]" = _foreach_sub_scalar[53]
        getitem_839: "f32[]" = _foreach_sub_scalar[54]
        getitem_840: "f32[]" = _foreach_sub_scalar[55]
        getitem_841: "f32[]" = _foreach_sub_scalar[56]
        getitem_842: "f32[]" = _foreach_sub_scalar[57]
        getitem_843: "f32[]" = _foreach_sub_scalar[58]
        getitem_844: "f32[]" = _foreach_sub_scalar[59]
        getitem_845: "f32[]" = _foreach_sub_scalar[60]
        getitem_846: "f32[]" = _foreach_sub_scalar[61]
        getitem_847: "f32[]" = _foreach_sub_scalar[62]
        getitem_848: "f32[]" = _foreach_sub_scalar[63]
        getitem_849: "f32[]" = _foreach_sub_scalar[64]
        getitem_850: "f32[]" = _foreach_sub_scalar[65]
        getitem_851: "f32[]" = _foreach_sub_scalar[66]
        getitem_852: "f32[]" = _foreach_sub_scalar[67]
        getitem_853: "f32[]" = _foreach_sub_scalar[68]
        getitem_854: "f32[]" = _foreach_sub_scalar[69]
        getitem_855: "f32[]" = _foreach_sub_scalar[70]
        getitem_856: "f32[]" = _foreach_sub_scalar[71]
        getitem_857: "f32[]" = _foreach_sub_scalar[72]
        getitem_858: "f32[]" = _foreach_sub_scalar[73]
        getitem_859: "f32[]" = _foreach_sub_scalar[74]
        getitem_860: "f32[]" = _foreach_sub_scalar[75]
        getitem_861: "f32[]" = _foreach_sub_scalar[76]
        getitem_862: "f32[]" = _foreach_sub_scalar[77]
        getitem_863: "f32[]" = _foreach_sub_scalar[78]
        getitem_864: "f32[]" = _foreach_sub_scalar[79]
        getitem_865: "f32[]" = _foreach_sub_scalar[80]
        getitem_866: "f32[]" = _foreach_sub_scalar[81]
        getitem_867: "f32[]" = _foreach_sub_scalar[82]
        getitem_868: "f32[]" = _foreach_sub_scalar[83]
        getitem_869: "f32[]" = _foreach_sub_scalar[84]
        getitem_870: "f32[]" = _foreach_sub_scalar[85]
        getitem_871: "f32[]" = _foreach_sub_scalar[86]
        getitem_872: "f32[]" = _foreach_sub_scalar[87]
        getitem_873: "f32[]" = _foreach_sub_scalar[88]
        getitem_874: "f32[]" = _foreach_sub_scalar[89]
        getitem_875: "f32[]" = _foreach_sub_scalar[90]
        getitem_876: "f32[]" = _foreach_sub_scalar[91]
        getitem_877: "f32[]" = _foreach_sub_scalar[92]
        getitem_878: "f32[]" = _foreach_sub_scalar[93]
        getitem_879: "f32[]" = _foreach_sub_scalar[94]
        getitem_880: "f32[]" = _foreach_sub_scalar[95]
        getitem_881: "f32[]" = _foreach_sub_scalar[96]
        getitem_882: "f32[]" = _foreach_sub_scalar[97]
        getitem_883: "f32[]" = _foreach_sub_scalar[98]
        getitem_884: "f32[]" = _foreach_sub_scalar[99]
        getitem_885: "f32[]" = _foreach_sub_scalar[100]
        getitem_886: "f32[]" = _foreach_sub_scalar[101]
        getitem_887: "f32[]" = _foreach_sub_scalar[102]
        getitem_888: "f32[]" = _foreach_sub_scalar[103]
        getitem_889: "f32[]" = _foreach_sub_scalar[104]
        getitem_890: "f32[]" = _foreach_sub_scalar[105]
        getitem_891: "f32[]" = _foreach_sub_scalar[106]
        getitem_892: "f32[]" = _foreach_sub_scalar[107]
        getitem_893: "f32[]" = _foreach_sub_scalar[108]
        getitem_894: "f32[]" = _foreach_sub_scalar[109]
        getitem_895: "f32[]" = _foreach_sub_scalar[110]
        getitem_896: "f32[]" = _foreach_sub_scalar[111]
        getitem_897: "f32[]" = _foreach_sub_scalar[112]
        getitem_898: "f32[]" = _foreach_sub_scalar[113]
        getitem_899: "f32[]" = _foreach_sub_scalar[114]
        getitem_900: "f32[]" = _foreach_sub_scalar[115]
        getitem_901: "f32[]" = _foreach_sub_scalar[116]
        getitem_902: "f32[]" = _foreach_sub_scalar[117]
        getitem_903: "f32[]" = _foreach_sub_scalar[118]
        getitem_904: "f32[]" = _foreach_sub_scalar[119]
        getitem_905: "f32[]" = _foreach_sub_scalar[120]
        getitem_906: "f32[]" = _foreach_sub_scalar[121]
        getitem_907: "f32[]" = _foreach_sub_scalar[122]
        getitem_908: "f32[]" = _foreach_sub_scalar[123]
        getitem_909: "f32[]" = _foreach_sub_scalar[124]
        getitem_910: "f32[]" = _foreach_sub_scalar[125]
        getitem_911: "f32[]" = _foreach_sub_scalar[126]
        getitem_912: "f32[]" = _foreach_sub_scalar[127]
        getitem_913: "f32[]" = _foreach_sub_scalar[128]
        getitem_914: "f32[]" = _foreach_sub_scalar[129]
        getitem_915: "f32[]" = _foreach_sub_scalar[130];  _foreach_sub_scalar = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309]);  getitem_1179 = getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = getitem_1187 = getitem_1188 = getitem_1189 = getitem_1190 = getitem_1191 = getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = getitem_1234 = getitem_1235 = getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = getitem_1242 = getitem_1243 = getitem_1244 = getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = getitem_1250 = getitem_1251 = getitem_1252 = getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = getitem_1258 = getitem_1259 = getitem_1260 = getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = getitem_1269 = getitem_1270 = getitem_1271 = getitem_1272 = getitem_1273 = getitem_1274 = getitem_1275 = getitem_1276 = getitem_1277 = getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = getitem_1285 = getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = getitem_1293 = getitem_1294 = getitem_1295 = getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = getitem_1302 = getitem_1303 = getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = None
        getitem_1310: "f32[]" = _foreach_sqrt_default[0]
        getitem_1311: "f32[]" = _foreach_sqrt_default[1]
        getitem_1312: "f32[]" = _foreach_sqrt_default[2]
        getitem_1313: "f32[]" = _foreach_sqrt_default[3]
        getitem_1314: "f32[]" = _foreach_sqrt_default[4]
        getitem_1315: "f32[]" = _foreach_sqrt_default[5]
        getitem_1316: "f32[]" = _foreach_sqrt_default[6]
        getitem_1317: "f32[]" = _foreach_sqrt_default[7]
        getitem_1318: "f32[]" = _foreach_sqrt_default[8]
        getitem_1319: "f32[]" = _foreach_sqrt_default[9]
        getitem_1320: "f32[]" = _foreach_sqrt_default[10]
        getitem_1321: "f32[]" = _foreach_sqrt_default[11]
        getitem_1322: "f32[]" = _foreach_sqrt_default[12]
        getitem_1323: "f32[]" = _foreach_sqrt_default[13]
        getitem_1324: "f32[]" = _foreach_sqrt_default[14]
        getitem_1325: "f32[]" = _foreach_sqrt_default[15]
        getitem_1326: "f32[]" = _foreach_sqrt_default[16]
        getitem_1327: "f32[]" = _foreach_sqrt_default[17]
        getitem_1328: "f32[]" = _foreach_sqrt_default[18]
        getitem_1329: "f32[]" = _foreach_sqrt_default[19]
        getitem_1330: "f32[]" = _foreach_sqrt_default[20]
        getitem_1331: "f32[]" = _foreach_sqrt_default[21]
        getitem_1332: "f32[]" = _foreach_sqrt_default[22]
        getitem_1333: "f32[]" = _foreach_sqrt_default[23]
        getitem_1334: "f32[]" = _foreach_sqrt_default[24]
        getitem_1335: "f32[]" = _foreach_sqrt_default[25]
        getitem_1336: "f32[]" = _foreach_sqrt_default[26]
        getitem_1337: "f32[]" = _foreach_sqrt_default[27]
        getitem_1338: "f32[]" = _foreach_sqrt_default[28]
        getitem_1339: "f32[]" = _foreach_sqrt_default[29]
        getitem_1340: "f32[]" = _foreach_sqrt_default[30]
        getitem_1341: "f32[]" = _foreach_sqrt_default[31]
        getitem_1342: "f32[]" = _foreach_sqrt_default[32]
        getitem_1343: "f32[]" = _foreach_sqrt_default[33]
        getitem_1344: "f32[]" = _foreach_sqrt_default[34]
        getitem_1345: "f32[]" = _foreach_sqrt_default[35]
        getitem_1346: "f32[]" = _foreach_sqrt_default[36]
        getitem_1347: "f32[]" = _foreach_sqrt_default[37]
        getitem_1348: "f32[]" = _foreach_sqrt_default[38]
        getitem_1349: "f32[]" = _foreach_sqrt_default[39]
        getitem_1350: "f32[]" = _foreach_sqrt_default[40]
        getitem_1351: "f32[]" = _foreach_sqrt_default[41]
        getitem_1352: "f32[]" = _foreach_sqrt_default[42]
        getitem_1353: "f32[]" = _foreach_sqrt_default[43]
        getitem_1354: "f32[]" = _foreach_sqrt_default[44]
        getitem_1355: "f32[]" = _foreach_sqrt_default[45]
        getitem_1356: "f32[]" = _foreach_sqrt_default[46]
        getitem_1357: "f32[]" = _foreach_sqrt_default[47]
        getitem_1358: "f32[]" = _foreach_sqrt_default[48]
        getitem_1359: "f32[]" = _foreach_sqrt_default[49]
        getitem_1360: "f32[]" = _foreach_sqrt_default[50]
        getitem_1361: "f32[]" = _foreach_sqrt_default[51]
        getitem_1362: "f32[]" = _foreach_sqrt_default[52]
        getitem_1363: "f32[]" = _foreach_sqrt_default[53]
        getitem_1364: "f32[]" = _foreach_sqrt_default[54]
        getitem_1365: "f32[]" = _foreach_sqrt_default[55]
        getitem_1366: "f32[]" = _foreach_sqrt_default[56]
        getitem_1367: "f32[]" = _foreach_sqrt_default[57]
        getitem_1368: "f32[]" = _foreach_sqrt_default[58]
        getitem_1369: "f32[]" = _foreach_sqrt_default[59]
        getitem_1370: "f32[]" = _foreach_sqrt_default[60]
        getitem_1371: "f32[]" = _foreach_sqrt_default[61]
        getitem_1372: "f32[]" = _foreach_sqrt_default[62]
        getitem_1373: "f32[]" = _foreach_sqrt_default[63]
        getitem_1374: "f32[]" = _foreach_sqrt_default[64]
        getitem_1375: "f32[]" = _foreach_sqrt_default[65]
        getitem_1376: "f32[]" = _foreach_sqrt_default[66]
        getitem_1377: "f32[]" = _foreach_sqrt_default[67]
        getitem_1378: "f32[]" = _foreach_sqrt_default[68]
        getitem_1379: "f32[]" = _foreach_sqrt_default[69]
        getitem_1380: "f32[]" = _foreach_sqrt_default[70]
        getitem_1381: "f32[]" = _foreach_sqrt_default[71]
        getitem_1382: "f32[]" = _foreach_sqrt_default[72]
        getitem_1383: "f32[]" = _foreach_sqrt_default[73]
        getitem_1384: "f32[]" = _foreach_sqrt_default[74]
        getitem_1385: "f32[]" = _foreach_sqrt_default[75]
        getitem_1386: "f32[]" = _foreach_sqrt_default[76]
        getitem_1387: "f32[]" = _foreach_sqrt_default[77]
        getitem_1388: "f32[]" = _foreach_sqrt_default[78]
        getitem_1389: "f32[]" = _foreach_sqrt_default[79]
        getitem_1390: "f32[]" = _foreach_sqrt_default[80]
        getitem_1391: "f32[]" = _foreach_sqrt_default[81]
        getitem_1392: "f32[]" = _foreach_sqrt_default[82]
        getitem_1393: "f32[]" = _foreach_sqrt_default[83]
        getitem_1394: "f32[]" = _foreach_sqrt_default[84]
        getitem_1395: "f32[]" = _foreach_sqrt_default[85]
        getitem_1396: "f32[]" = _foreach_sqrt_default[86]
        getitem_1397: "f32[]" = _foreach_sqrt_default[87]
        getitem_1398: "f32[]" = _foreach_sqrt_default[88]
        getitem_1399: "f32[]" = _foreach_sqrt_default[89]
        getitem_1400: "f32[]" = _foreach_sqrt_default[90]
        getitem_1401: "f32[]" = _foreach_sqrt_default[91]
        getitem_1402: "f32[]" = _foreach_sqrt_default[92]
        getitem_1403: "f32[]" = _foreach_sqrt_default[93]
        getitem_1404: "f32[]" = _foreach_sqrt_default[94]
        getitem_1405: "f32[]" = _foreach_sqrt_default[95]
        getitem_1406: "f32[]" = _foreach_sqrt_default[96]
        getitem_1407: "f32[]" = _foreach_sqrt_default[97]
        getitem_1408: "f32[]" = _foreach_sqrt_default[98]
        getitem_1409: "f32[]" = _foreach_sqrt_default[99]
        getitem_1410: "f32[]" = _foreach_sqrt_default[100]
        getitem_1411: "f32[]" = _foreach_sqrt_default[101]
        getitem_1412: "f32[]" = _foreach_sqrt_default[102]
        getitem_1413: "f32[]" = _foreach_sqrt_default[103]
        getitem_1414: "f32[]" = _foreach_sqrt_default[104]
        getitem_1415: "f32[]" = _foreach_sqrt_default[105]
        getitem_1416: "f32[]" = _foreach_sqrt_default[106]
        getitem_1417: "f32[]" = _foreach_sqrt_default[107]
        getitem_1418: "f32[]" = _foreach_sqrt_default[108]
        getitem_1419: "f32[]" = _foreach_sqrt_default[109]
        getitem_1420: "f32[]" = _foreach_sqrt_default[110]
        getitem_1421: "f32[]" = _foreach_sqrt_default[111]
        getitem_1422: "f32[]" = _foreach_sqrt_default[112]
        getitem_1423: "f32[]" = _foreach_sqrt_default[113]
        getitem_1424: "f32[]" = _foreach_sqrt_default[114]
        getitem_1425: "f32[]" = _foreach_sqrt_default[115]
        getitem_1426: "f32[]" = _foreach_sqrt_default[116]
        getitem_1427: "f32[]" = _foreach_sqrt_default[117]
        getitem_1428: "f32[]" = _foreach_sqrt_default[118]
        getitem_1429: "f32[]" = _foreach_sqrt_default[119]
        getitem_1430: "f32[]" = _foreach_sqrt_default[120]
        getitem_1431: "f32[]" = _foreach_sqrt_default[121]
        getitem_1432: "f32[]" = _foreach_sqrt_default[122]
        getitem_1433: "f32[]" = _foreach_sqrt_default[123]
        getitem_1434: "f32[]" = _foreach_sqrt_default[124]
        getitem_1435: "f32[]" = _foreach_sqrt_default[125]
        getitem_1436: "f32[]" = _foreach_sqrt_default[126]
        getitem_1437: "f32[]" = _foreach_sqrt_default[127]
        getitem_1438: "f32[]" = _foreach_sqrt_default[128]
        getitem_1439: "f32[]" = _foreach_sqrt_default[129]
        getitem_1440: "f32[]" = _foreach_sqrt_default[130];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539, getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654]);  getitem_524 = getitem_525 = getitem_526 = getitem_527 = getitem_528 = getitem_529 = getitem_530 = getitem_531 = getitem_532 = getitem_533 = getitem_534 = getitem_535 = getitem_536 = getitem_537 = getitem_538 = getitem_539 = getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = getitem_572 = getitem_573 = getitem_574 = getitem_575 = getitem_576 = getitem_577 = getitem_578 = getitem_579 = getitem_580 = getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = getitem_596 = getitem_597 = getitem_598 = getitem_599 = getitem_600 = getitem_601 = getitem_602 = getitem_603 = getitem_604 = getitem_605 = getitem_606 = getitem_607 = getitem_608 = getitem_609 = getitem_610 = getitem_611 = getitem_612 = getitem_613 = getitem_614 = getitem_615 = getitem_616 = getitem_617 = getitem_618 = getitem_619 = getitem_620 = getitem_621 = getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = getitem_646 = getitem_647 = getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = getitem_654 = None
        getitem_916: "f32[32128, 512]" = _foreach_sqrt_default_1[0]
        getitem_917: "f32[512, 512]" = _foreach_sqrt_default_1[1]
        getitem_918: "f32[512, 512]" = _foreach_sqrt_default_1[2]
        getitem_919: "f32[512, 512]" = _foreach_sqrt_default_1[3]
        getitem_920: "f32[512, 512]" = _foreach_sqrt_default_1[4]
        getitem_921: "f32[32, 8]" = _foreach_sqrt_default_1[5]
        getitem_922: "f32[512]" = _foreach_sqrt_default_1[6]
        getitem_923: "f32[2048, 512]" = _foreach_sqrt_default_1[7]
        getitem_924: "f32[512, 2048]" = _foreach_sqrt_default_1[8]
        getitem_925: "f32[512]" = _foreach_sqrt_default_1[9]
        getitem_926: "f32[512, 512]" = _foreach_sqrt_default_1[10]
        getitem_927: "f32[512, 512]" = _foreach_sqrt_default_1[11]
        getitem_928: "f32[512, 512]" = _foreach_sqrt_default_1[12]
        getitem_929: "f32[512, 512]" = _foreach_sqrt_default_1[13]
        getitem_930: "f32[512]" = _foreach_sqrt_default_1[14]
        getitem_931: "f32[2048, 512]" = _foreach_sqrt_default_1[15]
        getitem_932: "f32[512, 2048]" = _foreach_sqrt_default_1[16]
        getitem_933: "f32[512]" = _foreach_sqrt_default_1[17]
        getitem_934: "f32[512, 512]" = _foreach_sqrt_default_1[18]
        getitem_935: "f32[512, 512]" = _foreach_sqrt_default_1[19]
        getitem_936: "f32[512, 512]" = _foreach_sqrt_default_1[20]
        getitem_937: "f32[512, 512]" = _foreach_sqrt_default_1[21]
        getitem_938: "f32[512]" = _foreach_sqrt_default_1[22]
        getitem_939: "f32[2048, 512]" = _foreach_sqrt_default_1[23]
        getitem_940: "f32[512, 2048]" = _foreach_sqrt_default_1[24]
        getitem_941: "f32[512]" = _foreach_sqrt_default_1[25]
        getitem_942: "f32[512, 512]" = _foreach_sqrt_default_1[26]
        getitem_943: "f32[512, 512]" = _foreach_sqrt_default_1[27]
        getitem_944: "f32[512, 512]" = _foreach_sqrt_default_1[28]
        getitem_945: "f32[512, 512]" = _foreach_sqrt_default_1[29]
        getitem_946: "f32[512]" = _foreach_sqrt_default_1[30]
        getitem_947: "f32[2048, 512]" = _foreach_sqrt_default_1[31]
        getitem_948: "f32[512, 2048]" = _foreach_sqrt_default_1[32]
        getitem_949: "f32[512]" = _foreach_sqrt_default_1[33]
        getitem_950: "f32[512, 512]" = _foreach_sqrt_default_1[34]
        getitem_951: "f32[512, 512]" = _foreach_sqrt_default_1[35]
        getitem_952: "f32[512, 512]" = _foreach_sqrt_default_1[36]
        getitem_953: "f32[512, 512]" = _foreach_sqrt_default_1[37]
        getitem_954: "f32[512]" = _foreach_sqrt_default_1[38]
        getitem_955: "f32[2048, 512]" = _foreach_sqrt_default_1[39]
        getitem_956: "f32[512, 2048]" = _foreach_sqrt_default_1[40]
        getitem_957: "f32[512]" = _foreach_sqrt_default_1[41]
        getitem_958: "f32[512, 512]" = _foreach_sqrt_default_1[42]
        getitem_959: "f32[512, 512]" = _foreach_sqrt_default_1[43]
        getitem_960: "f32[512, 512]" = _foreach_sqrt_default_1[44]
        getitem_961: "f32[512, 512]" = _foreach_sqrt_default_1[45]
        getitem_962: "f32[512]" = _foreach_sqrt_default_1[46]
        getitem_963: "f32[2048, 512]" = _foreach_sqrt_default_1[47]
        getitem_964: "f32[512, 2048]" = _foreach_sqrt_default_1[48]
        getitem_965: "f32[512]" = _foreach_sqrt_default_1[49]
        getitem_966: "f32[512]" = _foreach_sqrt_default_1[50]
        getitem_967: "f32[512, 512]" = _foreach_sqrt_default_1[51]
        getitem_968: "f32[512, 512]" = _foreach_sqrt_default_1[52]
        getitem_969: "f32[512, 512]" = _foreach_sqrt_default_1[53]
        getitem_970: "f32[512, 512]" = _foreach_sqrt_default_1[54]
        getitem_971: "f32[32, 8]" = _foreach_sqrt_default_1[55]
        getitem_972: "f32[512]" = _foreach_sqrt_default_1[56]
        getitem_973: "f32[512, 512]" = _foreach_sqrt_default_1[57]
        getitem_974: "f32[512, 512]" = _foreach_sqrt_default_1[58]
        getitem_975: "f32[512, 512]" = _foreach_sqrt_default_1[59]
        getitem_976: "f32[512, 512]" = _foreach_sqrt_default_1[60]
        getitem_977: "f32[512]" = _foreach_sqrt_default_1[61]
        getitem_978: "f32[2048, 512]" = _foreach_sqrt_default_1[62]
        getitem_979: "f32[512, 2048]" = _foreach_sqrt_default_1[63]
        getitem_980: "f32[512]" = _foreach_sqrt_default_1[64]
        getitem_981: "f32[512, 512]" = _foreach_sqrt_default_1[65]
        getitem_982: "f32[512, 512]" = _foreach_sqrt_default_1[66]
        getitem_983: "f32[512, 512]" = _foreach_sqrt_default_1[67]
        getitem_984: "f32[512, 512]" = _foreach_sqrt_default_1[68]
        getitem_985: "f32[512]" = _foreach_sqrt_default_1[69]
        getitem_986: "f32[512, 512]" = _foreach_sqrt_default_1[70]
        getitem_987: "f32[512, 512]" = _foreach_sqrt_default_1[71]
        getitem_988: "f32[512, 512]" = _foreach_sqrt_default_1[72]
        getitem_989: "f32[512, 512]" = _foreach_sqrt_default_1[73]
        getitem_990: "f32[512]" = _foreach_sqrt_default_1[74]
        getitem_991: "f32[2048, 512]" = _foreach_sqrt_default_1[75]
        getitem_992: "f32[512, 2048]" = _foreach_sqrt_default_1[76]
        getitem_993: "f32[512]" = _foreach_sqrt_default_1[77]
        getitem_994: "f32[512, 512]" = _foreach_sqrt_default_1[78]
        getitem_995: "f32[512, 512]" = _foreach_sqrt_default_1[79]
        getitem_996: "f32[512, 512]" = _foreach_sqrt_default_1[80]
        getitem_997: "f32[512, 512]" = _foreach_sqrt_default_1[81]
        getitem_998: "f32[512]" = _foreach_sqrt_default_1[82]
        getitem_999: "f32[512, 512]" = _foreach_sqrt_default_1[83]
        getitem_1000: "f32[512, 512]" = _foreach_sqrt_default_1[84]
        getitem_1001: "f32[512, 512]" = _foreach_sqrt_default_1[85]
        getitem_1002: "f32[512, 512]" = _foreach_sqrt_default_1[86]
        getitem_1003: "f32[512]" = _foreach_sqrt_default_1[87]
        getitem_1004: "f32[2048, 512]" = _foreach_sqrt_default_1[88]
        getitem_1005: "f32[512, 2048]" = _foreach_sqrt_default_1[89]
        getitem_1006: "f32[512]" = _foreach_sqrt_default_1[90]
        getitem_1007: "f32[512, 512]" = _foreach_sqrt_default_1[91]
        getitem_1008: "f32[512, 512]" = _foreach_sqrt_default_1[92]
        getitem_1009: "f32[512, 512]" = _foreach_sqrt_default_1[93]
        getitem_1010: "f32[512, 512]" = _foreach_sqrt_default_1[94]
        getitem_1011: "f32[512]" = _foreach_sqrt_default_1[95]
        getitem_1012: "f32[512, 512]" = _foreach_sqrt_default_1[96]
        getitem_1013: "f32[512, 512]" = _foreach_sqrt_default_1[97]
        getitem_1014: "f32[512, 512]" = _foreach_sqrt_default_1[98]
        getitem_1015: "f32[512, 512]" = _foreach_sqrt_default_1[99]
        getitem_1016: "f32[512]" = _foreach_sqrt_default_1[100]
        getitem_1017: "f32[2048, 512]" = _foreach_sqrt_default_1[101]
        getitem_1018: "f32[512, 2048]" = _foreach_sqrt_default_1[102]
        getitem_1019: "f32[512]" = _foreach_sqrt_default_1[103]
        getitem_1020: "f32[512, 512]" = _foreach_sqrt_default_1[104]
        getitem_1021: "f32[512, 512]" = _foreach_sqrt_default_1[105]
        getitem_1022: "f32[512, 512]" = _foreach_sqrt_default_1[106]
        getitem_1023: "f32[512, 512]" = _foreach_sqrt_default_1[107]
        getitem_1024: "f32[512]" = _foreach_sqrt_default_1[108]
        getitem_1025: "f32[512, 512]" = _foreach_sqrt_default_1[109]
        getitem_1026: "f32[512, 512]" = _foreach_sqrt_default_1[110]
        getitem_1027: "f32[512, 512]" = _foreach_sqrt_default_1[111]
        getitem_1028: "f32[512, 512]" = _foreach_sqrt_default_1[112]
        getitem_1029: "f32[512]" = _foreach_sqrt_default_1[113]
        getitem_1030: "f32[2048, 512]" = _foreach_sqrt_default_1[114]
        getitem_1031: "f32[512, 2048]" = _foreach_sqrt_default_1[115]
        getitem_1032: "f32[512]" = _foreach_sqrt_default_1[116]
        getitem_1033: "f32[512, 512]" = _foreach_sqrt_default_1[117]
        getitem_1034: "f32[512, 512]" = _foreach_sqrt_default_1[118]
        getitem_1035: "f32[512, 512]" = _foreach_sqrt_default_1[119]
        getitem_1036: "f32[512, 512]" = _foreach_sqrt_default_1[120]
        getitem_1037: "f32[512]" = _foreach_sqrt_default_1[121]
        getitem_1038: "f32[512, 512]" = _foreach_sqrt_default_1[122]
        getitem_1039: "f32[512, 512]" = _foreach_sqrt_default_1[123]
        getitem_1040: "f32[512, 512]" = _foreach_sqrt_default_1[124]
        getitem_1041: "f32[512, 512]" = _foreach_sqrt_default_1[125]
        getitem_1042: "f32[512]" = _foreach_sqrt_default_1[126]
        getitem_1043: "f32[2048, 512]" = _foreach_sqrt_default_1[127]
        getitem_1044: "f32[512, 2048]" = _foreach_sqrt_default_1[128]
        getitem_1045: "f32[512]" = _foreach_sqrt_default_1[129]
        getitem_1046: "f32[512]" = _foreach_sqrt_default_1[130];  _foreach_sqrt_default_1 = None
        return (getitem, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_916, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

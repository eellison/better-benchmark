"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g77
Pattern hash: 7f5c97a29b76
Shape hash: 86dd8c2e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_775: "f32[]", getitem_776: "f32[]", getitem_777: "f32[]", getitem_778: "f32[]", getitem_779: "f32[]", getitem_780: "f32[]", getitem_781: "f32[]", getitem_782: "f32[]", getitem_783: "f32[]", getitem_784: "f32[]", getitem_785: "f32[]", getitem_786: "f32[]", getitem_787: "f32[]", getitem_788: "f32[]", getitem_789: "f32[]", getitem_790: "f32[]", getitem_791: "f32[]", getitem_792: "f32[]", getitem_793: "f32[]", getitem_794: "f32[]", getitem_795: "f32[]", getitem_796: "f32[]", getitem_797: "f32[]", getitem_798: "f32[]", getitem_799: "f32[]", getitem_800: "f32[]", getitem_801: "f32[]", getitem_802: "f32[]", getitem_803: "f32[]", getitem_804: "f32[]", getitem_805: "f32[]", getitem_806: "f32[]", getitem_807: "f32[]", getitem_808: "f32[]", getitem_809: "f32[]", getitem_810: "f32[]", getitem_811: "f32[]", getitem_812: "f32[]", getitem_813: "f32[]", getitem_814: "f32[]", getitem_815: "f32[]", getitem_816: "f32[]", getitem_817: "f32[]", getitem_818: "f32[]", getitem_819: "f32[]", getitem_820: "f32[]", getitem_821: "f32[]", getitem_822: "f32[]", getitem_823: "f32[]", getitem_824: "f32[]", getitem_825: "f32[]", getitem_826: "f32[]", getitem_827: "f32[]", getitem_828: "f32[]", getitem_829: "f32[]", getitem_830: "f32[]", getitem_831: "f32[]", getitem_832: "f32[]", getitem_833: "f32[]", getitem_834: "f32[]", getitem_835: "f32[]", getitem_836: "f32[]", getitem_837: "f32[]", getitem_838: "f32[]", getitem_839: "f32[]", getitem_840: "f32[]", getitem_841: "f32[]", getitem_842: "f32[]", getitem_843: "f32[]", getitem_844: "f32[]", getitem_845: "f32[]", getitem_846: "f32[]", getitem_847: "f32[]", getitem_848: "f32[]", getitem_849: "f32[]", getitem_850: "f32[]", getitem_851: "f32[]", getitem_852: "f32[]", getitem_853: "f32[]", getitem_854: "f32[]", getitem_855: "f32[]", getitem_856: "f32[]", getitem_857: "f32[]", getitem_858: "f32[]", getitem_859: "f32[]", getitem_860: "f32[]", getitem_861: "f32[]", getitem_862: "f32[]", getitem_863: "f32[]", getitem_864: "f32[]", getitem_865: "f32[]", getitem_866: "f32[]", getitem_867: "f32[]", getitem_868: "f32[]", getitem_869: "f32[]", getitem_870: "f32[]", getitem_871: "f32[]", getitem_872: "f32[]", getitem_873: "f32[]", getitem_874: "f32[]", getitem_875: "f32[]", getitem_876: "f32[]", getitem_877: "f32[]", getitem_878: "f32[]", getitem_879: "f32[]", getitem_880: "f32[]", getitem_881: "f32[]", getitem_882: "f32[]", getitem_883: "f32[]", getitem_884: "f32[]", getitem_885: "f32[]", getitem_886: "f32[]", getitem_887: "f32[]", getitem_888: "f32[]", getitem_889: "f32[]", getitem_890: "f32[]", getitem_891: "f32[]", getitem_892: "f32[]", getitem_893: "f32[]", getitem_894: "f32[]", getitem_895: "f32[]", getitem_896: "f32[]", getitem_897: "f32[]", getitem_898: "f32[]", getitem_899: "f32[]", getitem_900: "f32[]", getitem_901: "f32[]", getitem_902: "f32[]", getitem_903: "f32[]", getitem_904: "f32[]", getitem_905: "f32[]", getitem_906: "f32[]", getitem_907: "f32[]", getitem_908: "f32[]", getitem_909: "f32[]", getitem_910: "f32[]", getitem_911: "f32[]", getitem_912: "f32[]", getitem_913: "f32[]", getitem_914: "f32[]", getitem_915: "f32[]", getitem_916: "f32[]", getitem_917: "f32[]", getitem_918: "f32[]", getitem_919: "f32[]", getitem_920: "f32[]", getitem_921: "f32[]", getitem_922: "f32[]", getitem_923: "f32[]", getitem_924: "f32[]", getitem_925: "f32[]", getitem_926: "f32[]", getitem_927: "f32[]", getitem_928: "f32[]", getitem_929: "f32[]", getitem_1395: "f32[]", getitem_1396: "f32[]", getitem_1397: "f32[]", getitem_1398: "f32[]", getitem_1399: "f32[]", getitem_1400: "f32[]", getitem_1401: "f32[]", getitem_1402: "f32[]", getitem_1403: "f32[]", getitem_1404: "f32[]", getitem_1405: "f32[]", getitem_1406: "f32[]", getitem_1407: "f32[]", getitem_1408: "f32[]", getitem_1409: "f32[]", getitem_1410: "f32[]", getitem_1411: "f32[]", getitem_1412: "f32[]", getitem_1413: "f32[]", getitem_1414: "f32[]", getitem_1415: "f32[]", getitem_1416: "f32[]", getitem_1417: "f32[]", getitem_1418: "f32[]", getitem_1419: "f32[]", getitem_1420: "f32[]", getitem_1421: "f32[]", getitem_1422: "f32[]", getitem_1423: "f32[]", getitem_1424: "f32[]", getitem_1425: "f32[]", getitem_1426: "f32[]", getitem_1427: "f32[]", getitem_1428: "f32[]", getitem_1429: "f32[]", getitem_1430: "f32[]", getitem_1431: "f32[]", getitem_1432: "f32[]", getitem_1433: "f32[]", getitem_1434: "f32[]", getitem_1435: "f32[]", getitem_1436: "f32[]", getitem_1437: "f32[]", getitem_1438: "f32[]", getitem_1439: "f32[]", getitem_1440: "f32[]", getitem_1441: "f32[]", getitem_1442: "f32[]", getitem_1443: "f32[]", getitem_1444: "f32[]", getitem_1445: "f32[]", getitem_1446: "f32[]", getitem_1447: "f32[]", getitem_1448: "f32[]", getitem_1449: "f32[]", getitem_1450: "f32[]", getitem_1451: "f32[]", getitem_1452: "f32[]", getitem_1453: "f32[]", getitem_1454: "f32[]", getitem_1455: "f32[]", getitem_1456: "f32[]", getitem_1457: "f32[]", getitem_1458: "f32[]", getitem_1459: "f32[]", getitem_1460: "f32[]", getitem_1461: "f32[]", getitem_1462: "f32[]", getitem_1463: "f32[]", getitem_1464: "f32[]", getitem_1465: "f32[]", getitem_1466: "f32[]", getitem_1467: "f32[]", getitem_1468: "f32[]", getitem_1469: "f32[]", getitem_1470: "f32[]", getitem_1471: "f32[]", getitem_1472: "f32[]", getitem_1473: "f32[]", getitem_1474: "f32[]", getitem_1475: "f32[]", getitem_1476: "f32[]", getitem_1477: "f32[]", getitem_1478: "f32[]", getitem_1479: "f32[]", getitem_1480: "f32[]", getitem_1481: "f32[]", getitem_1482: "f32[]", getitem_1483: "f32[]", getitem_1484: "f32[]", getitem_1485: "f32[]", getitem_1486: "f32[]", getitem_1487: "f32[]", getitem_1488: "f32[]", getitem_1489: "f32[]", getitem_1490: "f32[]", getitem_1491: "f32[]", getitem_1492: "f32[]", getitem_1493: "f32[]", getitem_1494: "f32[]", getitem_1495: "f32[]", getitem_1496: "f32[]", getitem_1497: "f32[]", getitem_1498: "f32[]", getitem_1499: "f32[]", getitem_1500: "f32[]", getitem_1501: "f32[]", getitem_1502: "f32[]", getitem_1503: "f32[]", getitem_1504: "f32[]", getitem_1505: "f32[]", getitem_1506: "f32[]", getitem_1507: "f32[]", getitem_1508: "f32[]", getitem_1509: "f32[]", getitem_1510: "f32[]", getitem_1511: "f32[]", getitem_1512: "f32[]", getitem_1513: "f32[]", getitem_1514: "f32[]", getitem_1515: "f32[]", getitem_1516: "f32[]", getitem_1517: "f32[]", getitem_1518: "f32[]", getitem_1519: "f32[]", getitem_1520: "f32[]", getitem_1521: "f32[]", getitem_1522: "f32[]", getitem_1523: "f32[]", getitem_1524: "f32[]", getitem_1525: "f32[]", getitem_1526: "f32[]", getitem_1527: "f32[]", getitem_1528: "f32[]", getitem_1529: "f32[]", getitem_1530: "f32[]", getitem_1531: "f32[]", getitem_1532: "f32[]", getitem_1533: "f32[]", getitem_1534: "f32[]", getitem_1535: "f32[]", getitem_1536: "f32[]", getitem_1537: "f32[]", getitem_1538: "f32[]", getitem_1539: "f32[]", getitem_1540: "f32[]", getitem_1541: "f32[]", getitem_1542: "f32[]", getitem_1543: "f32[]", getitem_1544: "f32[]", getitem_1545: "f32[]", getitem_1546: "f32[]", getitem_1547: "f32[]", getitem_1548: "f32[]", getitem_1549: "f32[]", getitem_620: "f32[1, 1, 768]", getitem_621: "f32[1, 198, 768]", getitem_622: "f32[1, 1, 768]", getitem_623: "f32[768, 3, 16, 16]", getitem_624: "f32[768]", getitem_625: "f32[768]", getitem_626: "f32[768]", getitem_627: "f32[2304, 768]", getitem_628: "f32[2304]", getitem_629: "f32[768, 768]", getitem_630: "f32[768]", getitem_631: "f32[768]", getitem_632: "f32[768]", getitem_633: "f32[3072, 768]", getitem_634: "f32[3072]", getitem_635: "f32[768, 3072]", getitem_636: "f32[768]", getitem_637: "f32[768]", getitem_638: "f32[768]", getitem_639: "f32[2304, 768]", getitem_640: "f32[2304]", getitem_641: "f32[768, 768]", getitem_642: "f32[768]", getitem_643: "f32[768]", getitem_644: "f32[768]", getitem_645: "f32[3072, 768]", getitem_646: "f32[3072]", getitem_647: "f32[768, 3072]", getitem_648: "f32[768]", getitem_649: "f32[768]", getitem_650: "f32[768]", getitem_651: "f32[2304, 768]", getitem_652: "f32[2304]", getitem_653: "f32[768, 768]", getitem_654: "f32[768]", getitem_655: "f32[768]", getitem_656: "f32[768]", getitem_657: "f32[3072, 768]", getitem_658: "f32[3072]", getitem_659: "f32[768, 3072]", getitem_660: "f32[768]", getitem_661: "f32[768]", getitem_662: "f32[768]", getitem_663: "f32[2304, 768]", getitem_664: "f32[2304]", getitem_665: "f32[768, 768]", getitem_666: "f32[768]", getitem_667: "f32[768]", getitem_668: "f32[768]", getitem_669: "f32[3072, 768]", getitem_670: "f32[3072]", getitem_671: "f32[768, 3072]", getitem_672: "f32[768]", getitem_673: "f32[768]", getitem_674: "f32[768]", getitem_675: "f32[2304, 768]", getitem_676: "f32[2304]", getitem_677: "f32[768, 768]", getitem_678: "f32[768]", getitem_679: "f32[768]", getitem_680: "f32[768]", getitem_681: "f32[3072, 768]", getitem_682: "f32[3072]", getitem_683: "f32[768, 3072]", getitem_684: "f32[768]", getitem_685: "f32[768]", getitem_686: "f32[768]", getitem_687: "f32[2304, 768]", getitem_688: "f32[2304]", getitem_689: "f32[768, 768]", getitem_690: "f32[768]", getitem_691: "f32[768]", getitem_692: "f32[768]", getitem_693: "f32[3072, 768]", getitem_694: "f32[3072]", getitem_695: "f32[768, 3072]", getitem_696: "f32[768]", getitem_697: "f32[768]", getitem_698: "f32[768]", getitem_699: "f32[2304, 768]", getitem_700: "f32[2304]", getitem_701: "f32[768, 768]", getitem_702: "f32[768]", getitem_703: "f32[768]", getitem_704: "f32[768]", getitem_705: "f32[3072, 768]", getitem_706: "f32[3072]", getitem_707: "f32[768, 3072]", getitem_708: "f32[768]", getitem_709: "f32[768]", getitem_710: "f32[768]", getitem_711: "f32[2304, 768]", getitem_712: "f32[2304]", getitem_713: "f32[768, 768]", getitem_714: "f32[768]", getitem_715: "f32[768]", getitem_716: "f32[768]", getitem_717: "f32[3072, 768]", getitem_718: "f32[3072]", getitem_719: "f32[768, 3072]", getitem_720: "f32[768]", getitem_721: "f32[768]", getitem_722: "f32[768]", getitem_723: "f32[2304, 768]", getitem_724: "f32[2304]", getitem_725: "f32[768, 768]", getitem_726: "f32[768]", getitem_727: "f32[768]", getitem_728: "f32[768]", getitem_729: "f32[3072, 768]", getitem_730: "f32[3072]", getitem_731: "f32[768, 3072]", getitem_732: "f32[768]", getitem_733: "f32[768]", getitem_734: "f32[768]", getitem_735: "f32[2304, 768]", getitem_736: "f32[2304]", getitem_737: "f32[768, 768]", getitem_738: "f32[768]", getitem_739: "f32[768]", getitem_740: "f32[768]", getitem_741: "f32[3072, 768]", getitem_742: "f32[3072]", getitem_743: "f32[768, 3072]", getitem_744: "f32[768]", getitem_745: "f32[768]", getitem_746: "f32[768]", getitem_747: "f32[2304, 768]", getitem_748: "f32[2304]", getitem_749: "f32[768, 768]", getitem_750: "f32[768]", getitem_751: "f32[768]", getitem_752: "f32[768]", getitem_753: "f32[3072, 768]", getitem_754: "f32[3072]", getitem_755: "f32[768, 3072]", getitem_756: "f32[768]", getitem_757: "f32[768]", getitem_758: "f32[768]", getitem_759: "f32[2304, 768]", getitem_760: "f32[2304]", getitem_761: "f32[768, 768]", getitem_762: "f32[768]", getitem_763: "f32[768]", getitem_764: "f32[768]", getitem_765: "f32[3072, 768]", getitem_766: "f32[3072]", getitem_767: "f32[768, 3072]", getitem_768: "f32[768]", getitem_769: "f32[768]", getitem_770: "f32[768]", getitem_771: "f32[1000, 768]", getitem_772: "f32[1000]", getitem_773: "f32[1000, 768]", getitem_774: "f32[1000]"):
        # No stacktrace found for following nodes
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929], 1);  getitem_775 = getitem_776 = getitem_777 = getitem_778 = getitem_779 = getitem_780 = getitem_781 = getitem_782 = getitem_783 = getitem_784 = getitem_785 = getitem_786 = getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = getitem_794 = getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = getitem_802 = getitem_803 = getitem_804 = getitem_805 = getitem_806 = getitem_807 = getitem_808 = getitem_809 = getitem_810 = getitem_811 = getitem_812 = getitem_813 = getitem_814 = getitem_815 = getitem_816 = getitem_817 = getitem_818 = getitem_819 = getitem_820 = getitem_821 = getitem_822 = getitem_823 = getitem_824 = getitem_825 = getitem_826 = getitem_827 = getitem_828 = getitem_829 = getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = getitem_835 = getitem_836 = getitem_837 = getitem_838 = getitem_839 = getitem_840 = getitem_841 = getitem_842 = getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = getitem_849 = getitem_850 = getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = getitem_857 = getitem_858 = getitem_859 = getitem_860 = getitem_861 = getitem_862 = getitem_863 = getitem_864 = getitem_865 = getitem_866 = getitem_867 = getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = getitem_873 = getitem_874 = getitem_875 = getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = getitem_900 = getitem_901 = getitem_902 = getitem_903 = getitem_904 = getitem_905 = getitem_906 = getitem_907 = getitem_908 = getitem_909 = getitem_910 = getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = getitem_917 = getitem_918 = getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = getitem_925 = getitem_926 = getitem_927 = getitem_928 = getitem_929 = None
        getitem: "f32[]" = _foreach_sub_scalar[0]
        getitem_930: "f32[]" = _foreach_sub_scalar[1]
        getitem_931: "f32[]" = _foreach_sub_scalar[2]
        getitem_932: "f32[]" = _foreach_sub_scalar[3]
        getitem_933: "f32[]" = _foreach_sub_scalar[4]
        getitem_934: "f32[]" = _foreach_sub_scalar[5]
        getitem_935: "f32[]" = _foreach_sub_scalar[6]
        getitem_936: "f32[]" = _foreach_sub_scalar[7]
        getitem_937: "f32[]" = _foreach_sub_scalar[8]
        getitem_938: "f32[]" = _foreach_sub_scalar[9]
        getitem_939: "f32[]" = _foreach_sub_scalar[10]
        getitem_940: "f32[]" = _foreach_sub_scalar[11]
        getitem_941: "f32[]" = _foreach_sub_scalar[12]
        getitem_942: "f32[]" = _foreach_sub_scalar[13]
        getitem_943: "f32[]" = _foreach_sub_scalar[14]
        getitem_944: "f32[]" = _foreach_sub_scalar[15]
        getitem_945: "f32[]" = _foreach_sub_scalar[16]
        getitem_946: "f32[]" = _foreach_sub_scalar[17]
        getitem_947: "f32[]" = _foreach_sub_scalar[18]
        getitem_948: "f32[]" = _foreach_sub_scalar[19]
        getitem_949: "f32[]" = _foreach_sub_scalar[20]
        getitem_950: "f32[]" = _foreach_sub_scalar[21]
        getitem_951: "f32[]" = _foreach_sub_scalar[22]
        getitem_952: "f32[]" = _foreach_sub_scalar[23]
        getitem_953: "f32[]" = _foreach_sub_scalar[24]
        getitem_954: "f32[]" = _foreach_sub_scalar[25]
        getitem_955: "f32[]" = _foreach_sub_scalar[26]
        getitem_956: "f32[]" = _foreach_sub_scalar[27]
        getitem_957: "f32[]" = _foreach_sub_scalar[28]
        getitem_958: "f32[]" = _foreach_sub_scalar[29]
        getitem_959: "f32[]" = _foreach_sub_scalar[30]
        getitem_960: "f32[]" = _foreach_sub_scalar[31]
        getitem_961: "f32[]" = _foreach_sub_scalar[32]
        getitem_962: "f32[]" = _foreach_sub_scalar[33]
        getitem_963: "f32[]" = _foreach_sub_scalar[34]
        getitem_964: "f32[]" = _foreach_sub_scalar[35]
        getitem_965: "f32[]" = _foreach_sub_scalar[36]
        getitem_966: "f32[]" = _foreach_sub_scalar[37]
        getitem_967: "f32[]" = _foreach_sub_scalar[38]
        getitem_968: "f32[]" = _foreach_sub_scalar[39]
        getitem_969: "f32[]" = _foreach_sub_scalar[40]
        getitem_970: "f32[]" = _foreach_sub_scalar[41]
        getitem_971: "f32[]" = _foreach_sub_scalar[42]
        getitem_972: "f32[]" = _foreach_sub_scalar[43]
        getitem_973: "f32[]" = _foreach_sub_scalar[44]
        getitem_974: "f32[]" = _foreach_sub_scalar[45]
        getitem_975: "f32[]" = _foreach_sub_scalar[46]
        getitem_976: "f32[]" = _foreach_sub_scalar[47]
        getitem_977: "f32[]" = _foreach_sub_scalar[48]
        getitem_978: "f32[]" = _foreach_sub_scalar[49]
        getitem_979: "f32[]" = _foreach_sub_scalar[50]
        getitem_980: "f32[]" = _foreach_sub_scalar[51]
        getitem_981: "f32[]" = _foreach_sub_scalar[52]
        getitem_982: "f32[]" = _foreach_sub_scalar[53]
        getitem_983: "f32[]" = _foreach_sub_scalar[54]
        getitem_984: "f32[]" = _foreach_sub_scalar[55]
        getitem_985: "f32[]" = _foreach_sub_scalar[56]
        getitem_986: "f32[]" = _foreach_sub_scalar[57]
        getitem_987: "f32[]" = _foreach_sub_scalar[58]
        getitem_988: "f32[]" = _foreach_sub_scalar[59]
        getitem_989: "f32[]" = _foreach_sub_scalar[60]
        getitem_990: "f32[]" = _foreach_sub_scalar[61]
        getitem_991: "f32[]" = _foreach_sub_scalar[62]
        getitem_992: "f32[]" = _foreach_sub_scalar[63]
        getitem_993: "f32[]" = _foreach_sub_scalar[64]
        getitem_994: "f32[]" = _foreach_sub_scalar[65]
        getitem_995: "f32[]" = _foreach_sub_scalar[66]
        getitem_996: "f32[]" = _foreach_sub_scalar[67]
        getitem_997: "f32[]" = _foreach_sub_scalar[68]
        getitem_998: "f32[]" = _foreach_sub_scalar[69]
        getitem_999: "f32[]" = _foreach_sub_scalar[70]
        getitem_1000: "f32[]" = _foreach_sub_scalar[71]
        getitem_1001: "f32[]" = _foreach_sub_scalar[72]
        getitem_1002: "f32[]" = _foreach_sub_scalar[73]
        getitem_1003: "f32[]" = _foreach_sub_scalar[74]
        getitem_1004: "f32[]" = _foreach_sub_scalar[75]
        getitem_1005: "f32[]" = _foreach_sub_scalar[76]
        getitem_1006: "f32[]" = _foreach_sub_scalar[77]
        getitem_1007: "f32[]" = _foreach_sub_scalar[78]
        getitem_1008: "f32[]" = _foreach_sub_scalar[79]
        getitem_1009: "f32[]" = _foreach_sub_scalar[80]
        getitem_1010: "f32[]" = _foreach_sub_scalar[81]
        getitem_1011: "f32[]" = _foreach_sub_scalar[82]
        getitem_1012: "f32[]" = _foreach_sub_scalar[83]
        getitem_1013: "f32[]" = _foreach_sub_scalar[84]
        getitem_1014: "f32[]" = _foreach_sub_scalar[85]
        getitem_1015: "f32[]" = _foreach_sub_scalar[86]
        getitem_1016: "f32[]" = _foreach_sub_scalar[87]
        getitem_1017: "f32[]" = _foreach_sub_scalar[88]
        getitem_1018: "f32[]" = _foreach_sub_scalar[89]
        getitem_1019: "f32[]" = _foreach_sub_scalar[90]
        getitem_1020: "f32[]" = _foreach_sub_scalar[91]
        getitem_1021: "f32[]" = _foreach_sub_scalar[92]
        getitem_1022: "f32[]" = _foreach_sub_scalar[93]
        getitem_1023: "f32[]" = _foreach_sub_scalar[94]
        getitem_1024: "f32[]" = _foreach_sub_scalar[95]
        getitem_1025: "f32[]" = _foreach_sub_scalar[96]
        getitem_1026: "f32[]" = _foreach_sub_scalar[97]
        getitem_1027: "f32[]" = _foreach_sub_scalar[98]
        getitem_1028: "f32[]" = _foreach_sub_scalar[99]
        getitem_1029: "f32[]" = _foreach_sub_scalar[100]
        getitem_1030: "f32[]" = _foreach_sub_scalar[101]
        getitem_1031: "f32[]" = _foreach_sub_scalar[102]
        getitem_1032: "f32[]" = _foreach_sub_scalar[103]
        getitem_1033: "f32[]" = _foreach_sub_scalar[104]
        getitem_1034: "f32[]" = _foreach_sub_scalar[105]
        getitem_1035: "f32[]" = _foreach_sub_scalar[106]
        getitem_1036: "f32[]" = _foreach_sub_scalar[107]
        getitem_1037: "f32[]" = _foreach_sub_scalar[108]
        getitem_1038: "f32[]" = _foreach_sub_scalar[109]
        getitem_1039: "f32[]" = _foreach_sub_scalar[110]
        getitem_1040: "f32[]" = _foreach_sub_scalar[111]
        getitem_1041: "f32[]" = _foreach_sub_scalar[112]
        getitem_1042: "f32[]" = _foreach_sub_scalar[113]
        getitem_1043: "f32[]" = _foreach_sub_scalar[114]
        getitem_1044: "f32[]" = _foreach_sub_scalar[115]
        getitem_1045: "f32[]" = _foreach_sub_scalar[116]
        getitem_1046: "f32[]" = _foreach_sub_scalar[117]
        getitem_1047: "f32[]" = _foreach_sub_scalar[118]
        getitem_1048: "f32[]" = _foreach_sub_scalar[119]
        getitem_1049: "f32[]" = _foreach_sub_scalar[120]
        getitem_1050: "f32[]" = _foreach_sub_scalar[121]
        getitem_1051: "f32[]" = _foreach_sub_scalar[122]
        getitem_1052: "f32[]" = _foreach_sub_scalar[123]
        getitem_1053: "f32[]" = _foreach_sub_scalar[124]
        getitem_1054: "f32[]" = _foreach_sub_scalar[125]
        getitem_1055: "f32[]" = _foreach_sub_scalar[126]
        getitem_1056: "f32[]" = _foreach_sub_scalar[127]
        getitem_1057: "f32[]" = _foreach_sub_scalar[128]
        getitem_1058: "f32[]" = _foreach_sub_scalar[129]
        getitem_1059: "f32[]" = _foreach_sub_scalar[130]
        getitem_1060: "f32[]" = _foreach_sub_scalar[131]
        getitem_1061: "f32[]" = _foreach_sub_scalar[132]
        getitem_1062: "f32[]" = _foreach_sub_scalar[133]
        getitem_1063: "f32[]" = _foreach_sub_scalar[134]
        getitem_1064: "f32[]" = _foreach_sub_scalar[135]
        getitem_1065: "f32[]" = _foreach_sub_scalar[136]
        getitem_1066: "f32[]" = _foreach_sub_scalar[137]
        getitem_1067: "f32[]" = _foreach_sub_scalar[138]
        getitem_1068: "f32[]" = _foreach_sub_scalar[139]
        getitem_1069: "f32[]" = _foreach_sub_scalar[140]
        getitem_1070: "f32[]" = _foreach_sub_scalar[141]
        getitem_1071: "f32[]" = _foreach_sub_scalar[142]
        getitem_1072: "f32[]" = _foreach_sub_scalar[143]
        getitem_1073: "f32[]" = _foreach_sub_scalar[144]
        getitem_1074: "f32[]" = _foreach_sub_scalar[145]
        getitem_1075: "f32[]" = _foreach_sub_scalar[146]
        getitem_1076: "f32[]" = _foreach_sub_scalar[147]
        getitem_1077: "f32[]" = _foreach_sub_scalar[148]
        getitem_1078: "f32[]" = _foreach_sub_scalar[149]
        getitem_1079: "f32[]" = _foreach_sub_scalar[150]
        getitem_1080: "f32[]" = _foreach_sub_scalar[151]
        getitem_1081: "f32[]" = _foreach_sub_scalar[152]
        getitem_1082: "f32[]" = _foreach_sub_scalar[153]
        getitem_1083: "f32[]" = _foreach_sub_scalar[154];  _foreach_sub_scalar = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549]);  getitem_1395 = getitem_1396 = getitem_1397 = getitem_1398 = getitem_1399 = getitem_1400 = getitem_1401 = getitem_1402 = getitem_1403 = getitem_1404 = getitem_1405 = getitem_1406 = getitem_1407 = getitem_1408 = getitem_1409 = getitem_1410 = getitem_1411 = getitem_1412 = getitem_1413 = getitem_1414 = getitem_1415 = getitem_1416 = getitem_1417 = getitem_1418 = getitem_1419 = getitem_1420 = getitem_1421 = getitem_1422 = getitem_1423 = getitem_1424 = getitem_1425 = getitem_1426 = getitem_1427 = getitem_1428 = getitem_1429 = getitem_1430 = getitem_1431 = getitem_1432 = getitem_1433 = getitem_1434 = getitem_1435 = getitem_1436 = getitem_1437 = getitem_1438 = getitem_1439 = getitem_1440 = getitem_1441 = getitem_1442 = getitem_1443 = getitem_1444 = getitem_1445 = getitem_1446 = getitem_1447 = getitem_1448 = getitem_1449 = getitem_1450 = getitem_1451 = getitem_1452 = getitem_1453 = getitem_1454 = getitem_1455 = getitem_1456 = getitem_1457 = getitem_1458 = getitem_1459 = getitem_1460 = getitem_1461 = getitem_1462 = getitem_1463 = getitem_1464 = getitem_1465 = getitem_1466 = getitem_1467 = getitem_1468 = getitem_1469 = getitem_1470 = getitem_1471 = getitem_1472 = getitem_1473 = getitem_1474 = getitem_1475 = getitem_1476 = getitem_1477 = getitem_1478 = getitem_1479 = getitem_1480 = getitem_1481 = getitem_1482 = getitem_1483 = getitem_1484 = getitem_1485 = getitem_1486 = getitem_1487 = getitem_1488 = getitem_1489 = getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = getitem_1512 = getitem_1513 = getitem_1514 = getitem_1515 = getitem_1516 = getitem_1517 = getitem_1518 = getitem_1519 = getitem_1520 = getitem_1521 = getitem_1522 = getitem_1523 = getitem_1524 = getitem_1525 = getitem_1526 = getitem_1527 = getitem_1528 = getitem_1529 = getitem_1530 = getitem_1531 = getitem_1532 = getitem_1533 = getitem_1534 = getitem_1535 = getitem_1536 = getitem_1537 = getitem_1538 = getitem_1539 = getitem_1540 = getitem_1541 = getitem_1542 = getitem_1543 = getitem_1544 = getitem_1545 = getitem_1546 = getitem_1547 = getitem_1548 = getitem_1549 = None
        getitem_1550: "f32[]" = _foreach_sqrt_default[0]
        getitem_1551: "f32[]" = _foreach_sqrt_default[1]
        getitem_1552: "f32[]" = _foreach_sqrt_default[2]
        getitem_1553: "f32[]" = _foreach_sqrt_default[3]
        getitem_1554: "f32[]" = _foreach_sqrt_default[4]
        getitem_1555: "f32[]" = _foreach_sqrt_default[5]
        getitem_1556: "f32[]" = _foreach_sqrt_default[6]
        getitem_1557: "f32[]" = _foreach_sqrt_default[7]
        getitem_1558: "f32[]" = _foreach_sqrt_default[8]
        getitem_1559: "f32[]" = _foreach_sqrt_default[9]
        getitem_1560: "f32[]" = _foreach_sqrt_default[10]
        getitem_1561: "f32[]" = _foreach_sqrt_default[11]
        getitem_1562: "f32[]" = _foreach_sqrt_default[12]
        getitem_1563: "f32[]" = _foreach_sqrt_default[13]
        getitem_1564: "f32[]" = _foreach_sqrt_default[14]
        getitem_1565: "f32[]" = _foreach_sqrt_default[15]
        getitem_1566: "f32[]" = _foreach_sqrt_default[16]
        getitem_1567: "f32[]" = _foreach_sqrt_default[17]
        getitem_1568: "f32[]" = _foreach_sqrt_default[18]
        getitem_1569: "f32[]" = _foreach_sqrt_default[19]
        getitem_1570: "f32[]" = _foreach_sqrt_default[20]
        getitem_1571: "f32[]" = _foreach_sqrt_default[21]
        getitem_1572: "f32[]" = _foreach_sqrt_default[22]
        getitem_1573: "f32[]" = _foreach_sqrt_default[23]
        getitem_1574: "f32[]" = _foreach_sqrt_default[24]
        getitem_1575: "f32[]" = _foreach_sqrt_default[25]
        getitem_1576: "f32[]" = _foreach_sqrt_default[26]
        getitem_1577: "f32[]" = _foreach_sqrt_default[27]
        getitem_1578: "f32[]" = _foreach_sqrt_default[28]
        getitem_1579: "f32[]" = _foreach_sqrt_default[29]
        getitem_1580: "f32[]" = _foreach_sqrt_default[30]
        getitem_1581: "f32[]" = _foreach_sqrt_default[31]
        getitem_1582: "f32[]" = _foreach_sqrt_default[32]
        getitem_1583: "f32[]" = _foreach_sqrt_default[33]
        getitem_1584: "f32[]" = _foreach_sqrt_default[34]
        getitem_1585: "f32[]" = _foreach_sqrt_default[35]
        getitem_1586: "f32[]" = _foreach_sqrt_default[36]
        getitem_1587: "f32[]" = _foreach_sqrt_default[37]
        getitem_1588: "f32[]" = _foreach_sqrt_default[38]
        getitem_1589: "f32[]" = _foreach_sqrt_default[39]
        getitem_1590: "f32[]" = _foreach_sqrt_default[40]
        getitem_1591: "f32[]" = _foreach_sqrt_default[41]
        getitem_1592: "f32[]" = _foreach_sqrt_default[42]
        getitem_1593: "f32[]" = _foreach_sqrt_default[43]
        getitem_1594: "f32[]" = _foreach_sqrt_default[44]
        getitem_1595: "f32[]" = _foreach_sqrt_default[45]
        getitem_1596: "f32[]" = _foreach_sqrt_default[46]
        getitem_1597: "f32[]" = _foreach_sqrt_default[47]
        getitem_1598: "f32[]" = _foreach_sqrt_default[48]
        getitem_1599: "f32[]" = _foreach_sqrt_default[49]
        getitem_1600: "f32[]" = _foreach_sqrt_default[50]
        getitem_1601: "f32[]" = _foreach_sqrt_default[51]
        getitem_1602: "f32[]" = _foreach_sqrt_default[52]
        getitem_1603: "f32[]" = _foreach_sqrt_default[53]
        getitem_1604: "f32[]" = _foreach_sqrt_default[54]
        getitem_1605: "f32[]" = _foreach_sqrt_default[55]
        getitem_1606: "f32[]" = _foreach_sqrt_default[56]
        getitem_1607: "f32[]" = _foreach_sqrt_default[57]
        getitem_1608: "f32[]" = _foreach_sqrt_default[58]
        getitem_1609: "f32[]" = _foreach_sqrt_default[59]
        getitem_1610: "f32[]" = _foreach_sqrt_default[60]
        getitem_1611: "f32[]" = _foreach_sqrt_default[61]
        getitem_1612: "f32[]" = _foreach_sqrt_default[62]
        getitem_1613: "f32[]" = _foreach_sqrt_default[63]
        getitem_1614: "f32[]" = _foreach_sqrt_default[64]
        getitem_1615: "f32[]" = _foreach_sqrt_default[65]
        getitem_1616: "f32[]" = _foreach_sqrt_default[66]
        getitem_1617: "f32[]" = _foreach_sqrt_default[67]
        getitem_1618: "f32[]" = _foreach_sqrt_default[68]
        getitem_1619: "f32[]" = _foreach_sqrt_default[69]
        getitem_1620: "f32[]" = _foreach_sqrt_default[70]
        getitem_1621: "f32[]" = _foreach_sqrt_default[71]
        getitem_1622: "f32[]" = _foreach_sqrt_default[72]
        getitem_1623: "f32[]" = _foreach_sqrt_default[73]
        getitem_1624: "f32[]" = _foreach_sqrt_default[74]
        getitem_1625: "f32[]" = _foreach_sqrt_default[75]
        getitem_1626: "f32[]" = _foreach_sqrt_default[76]
        getitem_1627: "f32[]" = _foreach_sqrt_default[77]
        getitem_1628: "f32[]" = _foreach_sqrt_default[78]
        getitem_1629: "f32[]" = _foreach_sqrt_default[79]
        getitem_1630: "f32[]" = _foreach_sqrt_default[80]
        getitem_1631: "f32[]" = _foreach_sqrt_default[81]
        getitem_1632: "f32[]" = _foreach_sqrt_default[82]
        getitem_1633: "f32[]" = _foreach_sqrt_default[83]
        getitem_1634: "f32[]" = _foreach_sqrt_default[84]
        getitem_1635: "f32[]" = _foreach_sqrt_default[85]
        getitem_1636: "f32[]" = _foreach_sqrt_default[86]
        getitem_1637: "f32[]" = _foreach_sqrt_default[87]
        getitem_1638: "f32[]" = _foreach_sqrt_default[88]
        getitem_1639: "f32[]" = _foreach_sqrt_default[89]
        getitem_1640: "f32[]" = _foreach_sqrt_default[90]
        getitem_1641: "f32[]" = _foreach_sqrt_default[91]
        getitem_1642: "f32[]" = _foreach_sqrt_default[92]
        getitem_1643: "f32[]" = _foreach_sqrt_default[93]
        getitem_1644: "f32[]" = _foreach_sqrt_default[94]
        getitem_1645: "f32[]" = _foreach_sqrt_default[95]
        getitem_1646: "f32[]" = _foreach_sqrt_default[96]
        getitem_1647: "f32[]" = _foreach_sqrt_default[97]
        getitem_1648: "f32[]" = _foreach_sqrt_default[98]
        getitem_1649: "f32[]" = _foreach_sqrt_default[99]
        getitem_1650: "f32[]" = _foreach_sqrt_default[100]
        getitem_1651: "f32[]" = _foreach_sqrt_default[101]
        getitem_1652: "f32[]" = _foreach_sqrt_default[102]
        getitem_1653: "f32[]" = _foreach_sqrt_default[103]
        getitem_1654: "f32[]" = _foreach_sqrt_default[104]
        getitem_1655: "f32[]" = _foreach_sqrt_default[105]
        getitem_1656: "f32[]" = _foreach_sqrt_default[106]
        getitem_1657: "f32[]" = _foreach_sqrt_default[107]
        getitem_1658: "f32[]" = _foreach_sqrt_default[108]
        getitem_1659: "f32[]" = _foreach_sqrt_default[109]
        getitem_1660: "f32[]" = _foreach_sqrt_default[110]
        getitem_1661: "f32[]" = _foreach_sqrt_default[111]
        getitem_1662: "f32[]" = _foreach_sqrt_default[112]
        getitem_1663: "f32[]" = _foreach_sqrt_default[113]
        getitem_1664: "f32[]" = _foreach_sqrt_default[114]
        getitem_1665: "f32[]" = _foreach_sqrt_default[115]
        getitem_1666: "f32[]" = _foreach_sqrt_default[116]
        getitem_1667: "f32[]" = _foreach_sqrt_default[117]
        getitem_1668: "f32[]" = _foreach_sqrt_default[118]
        getitem_1669: "f32[]" = _foreach_sqrt_default[119]
        getitem_1670: "f32[]" = _foreach_sqrt_default[120]
        getitem_1671: "f32[]" = _foreach_sqrt_default[121]
        getitem_1672: "f32[]" = _foreach_sqrt_default[122]
        getitem_1673: "f32[]" = _foreach_sqrt_default[123]
        getitem_1674: "f32[]" = _foreach_sqrt_default[124]
        getitem_1675: "f32[]" = _foreach_sqrt_default[125]
        getitem_1676: "f32[]" = _foreach_sqrt_default[126]
        getitem_1677: "f32[]" = _foreach_sqrt_default[127]
        getitem_1678: "f32[]" = _foreach_sqrt_default[128]
        getitem_1679: "f32[]" = _foreach_sqrt_default[129]
        getitem_1680: "f32[]" = _foreach_sqrt_default[130]
        getitem_1681: "f32[]" = _foreach_sqrt_default[131]
        getitem_1682: "f32[]" = _foreach_sqrt_default[132]
        getitem_1683: "f32[]" = _foreach_sqrt_default[133]
        getitem_1684: "f32[]" = _foreach_sqrt_default[134]
        getitem_1685: "f32[]" = _foreach_sqrt_default[135]
        getitem_1686: "f32[]" = _foreach_sqrt_default[136]
        getitem_1687: "f32[]" = _foreach_sqrt_default[137]
        getitem_1688: "f32[]" = _foreach_sqrt_default[138]
        getitem_1689: "f32[]" = _foreach_sqrt_default[139]
        getitem_1690: "f32[]" = _foreach_sqrt_default[140]
        getitem_1691: "f32[]" = _foreach_sqrt_default[141]
        getitem_1692: "f32[]" = _foreach_sqrt_default[142]
        getitem_1693: "f32[]" = _foreach_sqrt_default[143]
        getitem_1694: "f32[]" = _foreach_sqrt_default[144]
        getitem_1695: "f32[]" = _foreach_sqrt_default[145]
        getitem_1696: "f32[]" = _foreach_sqrt_default[146]
        getitem_1697: "f32[]" = _foreach_sqrt_default[147]
        getitem_1698: "f32[]" = _foreach_sqrt_default[148]
        getitem_1699: "f32[]" = _foreach_sqrt_default[149]
        getitem_1700: "f32[]" = _foreach_sqrt_default[150]
        getitem_1701: "f32[]" = _foreach_sqrt_default[151]
        getitem_1702: "f32[]" = _foreach_sqrt_default[152]
        getitem_1703: "f32[]" = _foreach_sqrt_default[153]
        getitem_1704: "f32[]" = _foreach_sqrt_default[154];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774]);  getitem_620 = getitem_621 = getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = getitem_646 = getitem_647 = getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = getitem_654 = getitem_655 = getitem_656 = getitem_657 = getitem_658 = getitem_659 = getitem_660 = getitem_661 = getitem_662 = getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = getitem_671 = getitem_672 = getitem_673 = getitem_674 = getitem_675 = getitem_676 = getitem_677 = getitem_678 = getitem_679 = getitem_680 = getitem_681 = getitem_682 = getitem_683 = getitem_684 = getitem_685 = getitem_686 = getitem_687 = getitem_688 = getitem_689 = getitem_690 = getitem_691 = getitem_692 = getitem_693 = getitem_694 = getitem_695 = getitem_696 = getitem_697 = getitem_698 = getitem_699 = getitem_700 = getitem_701 = getitem_702 = getitem_703 = getitem_704 = getitem_705 = getitem_706 = getitem_707 = getitem_708 = getitem_709 = getitem_710 = getitem_711 = getitem_712 = getitem_713 = getitem_714 = getitem_715 = getitem_716 = getitem_717 = getitem_718 = getitem_719 = getitem_720 = getitem_721 = getitem_722 = getitem_723 = getitem_724 = getitem_725 = getitem_726 = getitem_727 = getitem_728 = getitem_729 = getitem_730 = getitem_731 = getitem_732 = getitem_733 = getitem_734 = getitem_735 = getitem_736 = getitem_737 = getitem_738 = getitem_739 = getitem_740 = getitem_741 = getitem_742 = getitem_743 = getitem_744 = getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = getitem_752 = getitem_753 = getitem_754 = getitem_755 = getitem_756 = getitem_757 = getitem_758 = getitem_759 = getitem_760 = getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = getitem_768 = getitem_769 = getitem_770 = getitem_771 = getitem_772 = getitem_773 = getitem_774 = None
        getitem_1084: "f32[1, 1, 768]" = _foreach_sqrt_default_1[0]
        getitem_1085: "f32[1, 198, 768]" = _foreach_sqrt_default_1[1]
        getitem_1086: "f32[1, 1, 768]" = _foreach_sqrt_default_1[2]
        getitem_1087: "f32[768, 3, 16, 16]" = _foreach_sqrt_default_1[3]
        getitem_1088: "f32[768]" = _foreach_sqrt_default_1[4]
        getitem_1089: "f32[768]" = _foreach_sqrt_default_1[5]
        getitem_1090: "f32[768]" = _foreach_sqrt_default_1[6]
        getitem_1091: "f32[2304, 768]" = _foreach_sqrt_default_1[7]
        getitem_1092: "f32[2304]" = _foreach_sqrt_default_1[8]
        getitem_1093: "f32[768, 768]" = _foreach_sqrt_default_1[9]
        getitem_1094: "f32[768]" = _foreach_sqrt_default_1[10]
        getitem_1095: "f32[768]" = _foreach_sqrt_default_1[11]
        getitem_1096: "f32[768]" = _foreach_sqrt_default_1[12]
        getitem_1097: "f32[3072, 768]" = _foreach_sqrt_default_1[13]
        getitem_1098: "f32[3072]" = _foreach_sqrt_default_1[14]
        getitem_1099: "f32[768, 3072]" = _foreach_sqrt_default_1[15]
        getitem_1100: "f32[768]" = _foreach_sqrt_default_1[16]
        getitem_1101: "f32[768]" = _foreach_sqrt_default_1[17]
        getitem_1102: "f32[768]" = _foreach_sqrt_default_1[18]
        getitem_1103: "f32[2304, 768]" = _foreach_sqrt_default_1[19]
        getitem_1104: "f32[2304]" = _foreach_sqrt_default_1[20]
        getitem_1105: "f32[768, 768]" = _foreach_sqrt_default_1[21]
        getitem_1106: "f32[768]" = _foreach_sqrt_default_1[22]
        getitem_1107: "f32[768]" = _foreach_sqrt_default_1[23]
        getitem_1108: "f32[768]" = _foreach_sqrt_default_1[24]
        getitem_1109: "f32[3072, 768]" = _foreach_sqrt_default_1[25]
        getitem_1110: "f32[3072]" = _foreach_sqrt_default_1[26]
        getitem_1111: "f32[768, 3072]" = _foreach_sqrt_default_1[27]
        getitem_1112: "f32[768]" = _foreach_sqrt_default_1[28]
        getitem_1113: "f32[768]" = _foreach_sqrt_default_1[29]
        getitem_1114: "f32[768]" = _foreach_sqrt_default_1[30]
        getitem_1115: "f32[2304, 768]" = _foreach_sqrt_default_1[31]
        getitem_1116: "f32[2304]" = _foreach_sqrt_default_1[32]
        getitem_1117: "f32[768, 768]" = _foreach_sqrt_default_1[33]
        getitem_1118: "f32[768]" = _foreach_sqrt_default_1[34]
        getitem_1119: "f32[768]" = _foreach_sqrt_default_1[35]
        getitem_1120: "f32[768]" = _foreach_sqrt_default_1[36]
        getitem_1121: "f32[3072, 768]" = _foreach_sqrt_default_1[37]
        getitem_1122: "f32[3072]" = _foreach_sqrt_default_1[38]
        getitem_1123: "f32[768, 3072]" = _foreach_sqrt_default_1[39]
        getitem_1124: "f32[768]" = _foreach_sqrt_default_1[40]
        getitem_1125: "f32[768]" = _foreach_sqrt_default_1[41]
        getitem_1126: "f32[768]" = _foreach_sqrt_default_1[42]
        getitem_1127: "f32[2304, 768]" = _foreach_sqrt_default_1[43]
        getitem_1128: "f32[2304]" = _foreach_sqrt_default_1[44]
        getitem_1129: "f32[768, 768]" = _foreach_sqrt_default_1[45]
        getitem_1130: "f32[768]" = _foreach_sqrt_default_1[46]
        getitem_1131: "f32[768]" = _foreach_sqrt_default_1[47]
        getitem_1132: "f32[768]" = _foreach_sqrt_default_1[48]
        getitem_1133: "f32[3072, 768]" = _foreach_sqrt_default_1[49]
        getitem_1134: "f32[3072]" = _foreach_sqrt_default_1[50]
        getitem_1135: "f32[768, 3072]" = _foreach_sqrt_default_1[51]
        getitem_1136: "f32[768]" = _foreach_sqrt_default_1[52]
        getitem_1137: "f32[768]" = _foreach_sqrt_default_1[53]
        getitem_1138: "f32[768]" = _foreach_sqrt_default_1[54]
        getitem_1139: "f32[2304, 768]" = _foreach_sqrt_default_1[55]
        getitem_1140: "f32[2304]" = _foreach_sqrt_default_1[56]
        getitem_1141: "f32[768, 768]" = _foreach_sqrt_default_1[57]
        getitem_1142: "f32[768]" = _foreach_sqrt_default_1[58]
        getitem_1143: "f32[768]" = _foreach_sqrt_default_1[59]
        getitem_1144: "f32[768]" = _foreach_sqrt_default_1[60]
        getitem_1145: "f32[3072, 768]" = _foreach_sqrt_default_1[61]
        getitem_1146: "f32[3072]" = _foreach_sqrt_default_1[62]
        getitem_1147: "f32[768, 3072]" = _foreach_sqrt_default_1[63]
        getitem_1148: "f32[768]" = _foreach_sqrt_default_1[64]
        getitem_1149: "f32[768]" = _foreach_sqrt_default_1[65]
        getitem_1150: "f32[768]" = _foreach_sqrt_default_1[66]
        getitem_1151: "f32[2304, 768]" = _foreach_sqrt_default_1[67]
        getitem_1152: "f32[2304]" = _foreach_sqrt_default_1[68]
        getitem_1153: "f32[768, 768]" = _foreach_sqrt_default_1[69]
        getitem_1154: "f32[768]" = _foreach_sqrt_default_1[70]
        getitem_1155: "f32[768]" = _foreach_sqrt_default_1[71]
        getitem_1156: "f32[768]" = _foreach_sqrt_default_1[72]
        getitem_1157: "f32[3072, 768]" = _foreach_sqrt_default_1[73]
        getitem_1158: "f32[3072]" = _foreach_sqrt_default_1[74]
        getitem_1159: "f32[768, 3072]" = _foreach_sqrt_default_1[75]
        getitem_1160: "f32[768]" = _foreach_sqrt_default_1[76]
        getitem_1161: "f32[768]" = _foreach_sqrt_default_1[77]
        getitem_1162: "f32[768]" = _foreach_sqrt_default_1[78]
        getitem_1163: "f32[2304, 768]" = _foreach_sqrt_default_1[79]
        getitem_1164: "f32[2304]" = _foreach_sqrt_default_1[80]
        getitem_1165: "f32[768, 768]" = _foreach_sqrt_default_1[81]
        getitem_1166: "f32[768]" = _foreach_sqrt_default_1[82]
        getitem_1167: "f32[768]" = _foreach_sqrt_default_1[83]
        getitem_1168: "f32[768]" = _foreach_sqrt_default_1[84]
        getitem_1169: "f32[3072, 768]" = _foreach_sqrt_default_1[85]
        getitem_1170: "f32[3072]" = _foreach_sqrt_default_1[86]
        getitem_1171: "f32[768, 3072]" = _foreach_sqrt_default_1[87]
        getitem_1172: "f32[768]" = _foreach_sqrt_default_1[88]
        getitem_1173: "f32[768]" = _foreach_sqrt_default_1[89]
        getitem_1174: "f32[768]" = _foreach_sqrt_default_1[90]
        getitem_1175: "f32[2304, 768]" = _foreach_sqrt_default_1[91]
        getitem_1176: "f32[2304]" = _foreach_sqrt_default_1[92]
        getitem_1177: "f32[768, 768]" = _foreach_sqrt_default_1[93]
        getitem_1178: "f32[768]" = _foreach_sqrt_default_1[94]
        getitem_1179: "f32[768]" = _foreach_sqrt_default_1[95]
        getitem_1180: "f32[768]" = _foreach_sqrt_default_1[96]
        getitem_1181: "f32[3072, 768]" = _foreach_sqrt_default_1[97]
        getitem_1182: "f32[3072]" = _foreach_sqrt_default_1[98]
        getitem_1183: "f32[768, 3072]" = _foreach_sqrt_default_1[99]
        getitem_1184: "f32[768]" = _foreach_sqrt_default_1[100]
        getitem_1185: "f32[768]" = _foreach_sqrt_default_1[101]
        getitem_1186: "f32[768]" = _foreach_sqrt_default_1[102]
        getitem_1187: "f32[2304, 768]" = _foreach_sqrt_default_1[103]
        getitem_1188: "f32[2304]" = _foreach_sqrt_default_1[104]
        getitem_1189: "f32[768, 768]" = _foreach_sqrt_default_1[105]
        getitem_1190: "f32[768]" = _foreach_sqrt_default_1[106]
        getitem_1191: "f32[768]" = _foreach_sqrt_default_1[107]
        getitem_1192: "f32[768]" = _foreach_sqrt_default_1[108]
        getitem_1193: "f32[3072, 768]" = _foreach_sqrt_default_1[109]
        getitem_1194: "f32[3072]" = _foreach_sqrt_default_1[110]
        getitem_1195: "f32[768, 3072]" = _foreach_sqrt_default_1[111]
        getitem_1196: "f32[768]" = _foreach_sqrt_default_1[112]
        getitem_1197: "f32[768]" = _foreach_sqrt_default_1[113]
        getitem_1198: "f32[768]" = _foreach_sqrt_default_1[114]
        getitem_1199: "f32[2304, 768]" = _foreach_sqrt_default_1[115]
        getitem_1200: "f32[2304]" = _foreach_sqrt_default_1[116]
        getitem_1201: "f32[768, 768]" = _foreach_sqrt_default_1[117]
        getitem_1202: "f32[768]" = _foreach_sqrt_default_1[118]
        getitem_1203: "f32[768]" = _foreach_sqrt_default_1[119]
        getitem_1204: "f32[768]" = _foreach_sqrt_default_1[120]
        getitem_1205: "f32[3072, 768]" = _foreach_sqrt_default_1[121]
        getitem_1206: "f32[3072]" = _foreach_sqrt_default_1[122]
        getitem_1207: "f32[768, 3072]" = _foreach_sqrt_default_1[123]
        getitem_1208: "f32[768]" = _foreach_sqrt_default_1[124]
        getitem_1209: "f32[768]" = _foreach_sqrt_default_1[125]
        getitem_1210: "f32[768]" = _foreach_sqrt_default_1[126]
        getitem_1211: "f32[2304, 768]" = _foreach_sqrt_default_1[127]
        getitem_1212: "f32[2304]" = _foreach_sqrt_default_1[128]
        getitem_1213: "f32[768, 768]" = _foreach_sqrt_default_1[129]
        getitem_1214: "f32[768]" = _foreach_sqrt_default_1[130]
        getitem_1215: "f32[768]" = _foreach_sqrt_default_1[131]
        getitem_1216: "f32[768]" = _foreach_sqrt_default_1[132]
        getitem_1217: "f32[3072, 768]" = _foreach_sqrt_default_1[133]
        getitem_1218: "f32[3072]" = _foreach_sqrt_default_1[134]
        getitem_1219: "f32[768, 3072]" = _foreach_sqrt_default_1[135]
        getitem_1220: "f32[768]" = _foreach_sqrt_default_1[136]
        getitem_1221: "f32[768]" = _foreach_sqrt_default_1[137]
        getitem_1222: "f32[768]" = _foreach_sqrt_default_1[138]
        getitem_1223: "f32[2304, 768]" = _foreach_sqrt_default_1[139]
        getitem_1224: "f32[2304]" = _foreach_sqrt_default_1[140]
        getitem_1225: "f32[768, 768]" = _foreach_sqrt_default_1[141]
        getitem_1226: "f32[768]" = _foreach_sqrt_default_1[142]
        getitem_1227: "f32[768]" = _foreach_sqrt_default_1[143]
        getitem_1228: "f32[768]" = _foreach_sqrt_default_1[144]
        getitem_1229: "f32[3072, 768]" = _foreach_sqrt_default_1[145]
        getitem_1230: "f32[3072]" = _foreach_sqrt_default_1[146]
        getitem_1231: "f32[768, 3072]" = _foreach_sqrt_default_1[147]
        getitem_1232: "f32[768]" = _foreach_sqrt_default_1[148]
        getitem_1233: "f32[768]" = _foreach_sqrt_default_1[149]
        getitem_1234: "f32[768]" = _foreach_sqrt_default_1[150]
        getitem_1235: "f32[1000, 768]" = _foreach_sqrt_default_1[151]
        getitem_1236: "f32[1000]" = _foreach_sqrt_default_1[152]
        getitem_1237: "f32[1000, 768]" = _foreach_sqrt_default_1[153]
        getitem_1238: "f32[1000]" = _foreach_sqrt_default_1[154];  _foreach_sqrt_default_1 = None
        return (getitem, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047, getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 198, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

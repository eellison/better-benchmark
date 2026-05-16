"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: eb843608b24f
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
    def forward(self, getitem_917: "f32[]", getitem_918: "f32[]", getitem_919: "f32[]", getitem_920: "f32[]", getitem_921: "f32[]", getitem_922: "f32[]", getitem_923: "f32[]", getitem_924: "f32[]", getitem_925: "f32[]", getitem_926: "f32[]", getitem_927: "f32[]", getitem_928: "f32[]", getitem_929: "f32[]", getitem_930: "f32[]", getitem_931: "f32[]", getitem_932: "f32[]", getitem_933: "f32[]", getitem_934: "f32[]", getitem_935: "f32[]", getitem_936: "f32[]", getitem_937: "f32[]", getitem_938: "f32[]", getitem_939: "f32[]", getitem_940: "f32[]", getitem_941: "f32[]", getitem_942: "f32[]", getitem_943: "f32[]", getitem_944: "f32[]", getitem_945: "f32[]", getitem_946: "f32[]", getitem_947: "f32[]", getitem_948: "f32[]", getitem_949: "f32[]", getitem_950: "f32[]", getitem_951: "f32[]", getitem_952: "f32[]", getitem_953: "f32[]", getitem_954: "f32[]", getitem_955: "f32[]", getitem_956: "f32[]", getitem_957: "f32[]", getitem_958: "f32[]", getitem_959: "f32[]", getitem_960: "f32[]", getitem_961: "f32[]", getitem_962: "f32[]", getitem_963: "f32[]", getitem_964: "f32[]", getitem_965: "f32[]", getitem_966: "f32[]", getitem_967: "f32[]", getitem_968: "f32[]", getitem_969: "f32[]", getitem_970: "f32[]", getitem_971: "f32[]", getitem_972: "f32[]", getitem_973: "f32[]", getitem_974: "f32[]", getitem_975: "f32[]", getitem_976: "f32[]", getitem_977: "f32[]", getitem_978: "f32[]", getitem_979: "f32[]", getitem_980: "f32[]", getitem_981: "f32[]", getitem_982: "f32[]", getitem_983: "f32[]", getitem_984: "f32[]", getitem_985: "f32[]", getitem_986: "f32[]", getitem_987: "f32[]", getitem_988: "f32[]", getitem_989: "f32[]", getitem_990: "f32[]", getitem_991: "f32[]", getitem_992: "f32[]", getitem_993: "f32[]", getitem_994: "f32[]", getitem_995: "f32[]", getitem_996: "f32[]", getitem_997: "f32[]", getitem_998: "f32[]", getitem_999: "f32[]", getitem_1000: "f32[]", getitem_1001: "f32[]", getitem_1002: "f32[]", getitem_1003: "f32[]", getitem_1004: "f32[]", getitem_1005: "f32[]", getitem_1006: "f32[]", getitem_1007: "f32[]", getitem_1008: "f32[]", getitem_1009: "f32[]", getitem_1010: "f32[]", getitem_1011: "f32[]", getitem_1012: "f32[]", getitem_1013: "f32[]", getitem_1014: "f32[]", getitem_1015: "f32[]", getitem_1016: "f32[]", getitem_1017: "f32[]", getitem_1018: "f32[]", getitem_1019: "f32[]", getitem_1020: "f32[]", getitem_1021: "f32[]", getitem_1022: "f32[]", getitem_1023: "f32[]", getitem_1024: "f32[]", getitem_1025: "f32[]", getitem_1026: "f32[]", getitem_1027: "f32[]", getitem_1028: "f32[]", getitem_1029: "f32[]", getitem_1030: "f32[]", getitem_1031: "f32[]", getitem_1032: "f32[]", getitem_1033: "f32[]", getitem_1034: "f32[]", getitem_1035: "f32[]", getitem_1036: "f32[]", getitem_1037: "f32[]", getitem_1038: "f32[]", getitem_1039: "f32[]", getitem_1040: "f32[]", getitem_1041: "f32[]", getitem_1042: "f32[]", getitem_1043: "f32[]", getitem_1044: "f32[]", getitem_1045: "f32[]", getitem_1046: "f32[]", getitem_1047: "f32[]", getitem_1703: "f32[32128, 512]", getitem_1704: "f32[512, 512]", getitem_1705: "f32[512, 512]", getitem_1706: "f32[512, 512]", getitem_1707: "f32[512, 512]", getitem_1708: "f32[32, 8]", getitem_1709: "f32[512]", getitem_1710: "f32[2048, 512]", getitem_1711: "f32[512, 2048]", getitem_1712: "f32[512]", getitem_1713: "f32[512, 512]", getitem_1714: "f32[512, 512]", getitem_1715: "f32[512, 512]", getitem_1716: "f32[512, 512]", getitem_1717: "f32[512]", getitem_1718: "f32[2048, 512]", getitem_1719: "f32[512, 2048]", getitem_1720: "f32[512]", getitem_1721: "f32[512, 512]", getitem_1722: "f32[512, 512]", getitem_1723: "f32[512, 512]", getitem_1724: "f32[512, 512]", getitem_1725: "f32[512]", getitem_1726: "f32[2048, 512]", getitem_1727: "f32[512, 2048]", getitem_1728: "f32[512]", getitem_1729: "f32[512, 512]", getitem_1730: "f32[512, 512]", getitem_1731: "f32[512, 512]", getitem_1732: "f32[512, 512]", getitem_1733: "f32[512]", getitem_1734: "f32[2048, 512]", getitem_1735: "f32[512, 2048]", getitem_1736: "f32[512]", getitem_1737: "f32[512, 512]", getitem_1738: "f32[512, 512]", getitem_1739: "f32[512, 512]", getitem_1740: "f32[512, 512]", getitem_1741: "f32[512]", getitem_1742: "f32[2048, 512]", getitem_1743: "f32[512, 2048]", getitem_1744: "f32[512]", getitem_1745: "f32[512, 512]", getitem_1746: "f32[512, 512]", getitem_1747: "f32[512, 512]", getitem_1748: "f32[512, 512]", getitem_1749: "f32[512]", getitem_1750: "f32[2048, 512]", getitem_1751: "f32[512, 2048]", getitem_1752: "f32[512]", getitem_1753: "f32[512]", getitem_1754: "f32[512, 512]", getitem_1755: "f32[512, 512]", getitem_1756: "f32[512, 512]", getitem_1757: "f32[512, 512]", getitem_1758: "f32[32, 8]", getitem_1759: "f32[512]", getitem_1760: "f32[512, 512]", getitem_1761: "f32[512, 512]", getitem_1762: "f32[512, 512]", getitem_1763: "f32[512, 512]", getitem_1764: "f32[512]", getitem_1765: "f32[2048, 512]", getitem_1766: "f32[512, 2048]", getitem_1767: "f32[512]", getitem_1768: "f32[512, 512]", getitem_1769: "f32[512, 512]", getitem_1770: "f32[512, 512]", getitem_1771: "f32[512, 512]", getitem_1772: "f32[512]", getitem_1773: "f32[512, 512]", getitem_1774: "f32[512, 512]", getitem_1775: "f32[512, 512]", getitem_1776: "f32[512, 512]", getitem_1777: "f32[512]", getitem_1778: "f32[2048, 512]", getitem_1779: "f32[512, 2048]", getitem_1780: "f32[512]", getitem_1781: "f32[512, 512]", getitem_1782: "f32[512, 512]", getitem_1783: "f32[512, 512]", getitem_1784: "f32[512, 512]", getitem_1785: "f32[512]", getitem_1786: "f32[512, 512]", getitem_1787: "f32[512, 512]", getitem_1788: "f32[512, 512]", getitem_1789: "f32[512, 512]", getitem_1790: "f32[512]", getitem_1791: "f32[2048, 512]", getitem_1792: "f32[512, 2048]", getitem_1793: "f32[512]", getitem_1794: "f32[512, 512]", getitem_1795: "f32[512, 512]", getitem_1796: "f32[512, 512]", getitem_1797: "f32[512, 512]", getitem_1798: "f32[512]", getitem_1799: "f32[512, 512]", getitem_1800: "f32[512, 512]", getitem_1801: "f32[512, 512]", getitem_1802: "f32[512, 512]", getitem_1803: "f32[512]", getitem_1804: "f32[2048, 512]", getitem_1805: "f32[512, 2048]", getitem_1806: "f32[512]", getitem_1807: "f32[512, 512]", getitem_1808: "f32[512, 512]", getitem_1809: "f32[512, 512]", getitem_1810: "f32[512, 512]", getitem_1811: "f32[512]", getitem_1812: "f32[512, 512]", getitem_1813: "f32[512, 512]", getitem_1814: "f32[512, 512]", getitem_1815: "f32[512, 512]", getitem_1816: "f32[512]", getitem_1817: "f32[2048, 512]", getitem_1818: "f32[512, 2048]", getitem_1819: "f32[512]", getitem_1820: "f32[512, 512]", getitem_1821: "f32[512, 512]", getitem_1822: "f32[512, 512]", getitem_1823: "f32[512, 512]", getitem_1824: "f32[512]", getitem_1825: "f32[512, 512]", getitem_1826: "f32[512, 512]", getitem_1827: "f32[512, 512]", getitem_1828: "f32[512, 512]", getitem_1829: "f32[512]", getitem_1830: "f32[2048, 512]", getitem_1831: "f32[512, 2048]", getitem_1832: "f32[512]", getitem_1833: "f32[512]", getitem_1572: "f32[]", getitem_1573: "f32[]", getitem_1574: "f32[]", getitem_1575: "f32[]", getitem_1576: "f32[]", getitem_1577: "f32[]", getitem_1578: "f32[]", getitem_1579: "f32[]", getitem_1580: "f32[]", getitem_1581: "f32[]", getitem_1582: "f32[]", getitem_1583: "f32[]", getitem_1584: "f32[]", getitem_1585: "f32[]", getitem_1586: "f32[]", getitem_1587: "f32[]", getitem_1588: "f32[]", getitem_1589: "f32[]", getitem_1590: "f32[]", getitem_1591: "f32[]", getitem_1592: "f32[]", getitem_1593: "f32[]", getitem_1594: "f32[]", getitem_1595: "f32[]", getitem_1596: "f32[]", getitem_1597: "f32[]", getitem_1598: "f32[]", getitem_1599: "f32[]", getitem_1600: "f32[]", getitem_1601: "f32[]", getitem_1602: "f32[]", getitem_1603: "f32[]", getitem_1604: "f32[]", getitem_1605: "f32[]", getitem_1606: "f32[]", getitem_1607: "f32[]", getitem_1608: "f32[]", getitem_1609: "f32[]", getitem_1610: "f32[]", getitem_1611: "f32[]", getitem_1612: "f32[]", getitem_1613: "f32[]", getitem_1614: "f32[]", getitem_1615: "f32[]", getitem_1616: "f32[]", getitem_1617: "f32[]", getitem_1618: "f32[]", getitem_1619: "f32[]", getitem_1620: "f32[]", getitem_1621: "f32[]", getitem_1622: "f32[]", getitem_1623: "f32[]", getitem_1624: "f32[]", getitem_1625: "f32[]", getitem_1626: "f32[]", getitem_1627: "f32[]", getitem_1628: "f32[]", getitem_1629: "f32[]", getitem_1630: "f32[]", getitem_1631: "f32[]", getitem_1632: "f32[]", getitem_1633: "f32[]", getitem_1634: "f32[]", getitem_1635: "f32[]", getitem_1636: "f32[]", getitem_1637: "f32[]", getitem_1638: "f32[]", getitem_1639: "f32[]", getitem_1640: "f32[]", getitem_1641: "f32[]", getitem_1642: "f32[]", getitem_1643: "f32[]", getitem_1644: "f32[]", getitem_1645: "f32[]", getitem_1646: "f32[]", getitem_1647: "f32[]", getitem_1648: "f32[]", getitem_1649: "f32[]", getitem_1650: "f32[]", getitem_1651: "f32[]", getitem_1652: "f32[]", getitem_1653: "f32[]", getitem_1654: "f32[]", getitem_1655: "f32[]", getitem_1656: "f32[]", getitem_1657: "f32[]", getitem_1658: "f32[]", getitem_1659: "f32[]", getitem_1660: "f32[]", getitem_1661: "f32[]", getitem_1662: "f32[]", getitem_1663: "f32[]", getitem_1664: "f32[]", getitem_1665: "f32[]", getitem_1666: "f32[]", getitem_1667: "f32[]", getitem_1668: "f32[]", getitem_1669: "f32[]", getitem_1670: "f32[]", getitem_1671: "f32[]", getitem_1672: "f32[]", getitem_1673: "f32[]", getitem_1674: "f32[]", getitem_1675: "f32[]", getitem_1676: "f32[]", getitem_1677: "f32[]", getitem_1678: "f32[]", getitem_1679: "f32[]", getitem_1680: "f32[]", getitem_1681: "f32[]", getitem_1682: "f32[]", getitem_1683: "f32[]", getitem_1684: "f32[]", getitem_1685: "f32[]", getitem_1686: "f32[]", getitem_1687: "f32[]", getitem_1688: "f32[]", getitem_1689: "f32[]", getitem_1690: "f32[]", getitem_1691: "f32[]", getitem_1692: "f32[]", getitem_1693: "f32[]", getitem_1694: "f32[]", getitem_1695: "f32[]", getitem_1696: "f32[]", getitem_1697: "f32[]", getitem_1698: "f32[]", getitem_1699: "f32[]", getitem_1700: "f32[]", getitem_1701: "f32[]", getitem_1702: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047], 0.01);  getitem_917 = getitem_918 = getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = getitem_925 = getitem_926 = getitem_927 = getitem_928 = getitem_929 = getitem_930 = getitem_931 = getitem_932 = getitem_933 = getitem_934 = getitem_935 = getitem_936 = getitem_937 = getitem_938 = getitem_939 = getitem_940 = getitem_941 = getitem_942 = getitem_943 = getitem_944 = getitem_945 = getitem_946 = getitem_947 = getitem_948 = getitem_949 = getitem_950 = getitem_951 = getitem_952 = getitem_953 = getitem_954 = getitem_955 = getitem_956 = getitem_957 = getitem_958 = getitem_959 = getitem_960 = getitem_961 = getitem_962 = getitem_963 = getitem_964 = getitem_965 = getitem_966 = getitem_967 = getitem_968 = getitem_969 = getitem_970 = getitem_971 = getitem_972 = getitem_973 = getitem_974 = getitem_975 = getitem_976 = getitem_977 = getitem_978 = getitem_979 = getitem_980 = getitem_981 = getitem_982 = getitem_983 = getitem_984 = getitem_985 = getitem_986 = getitem_987 = getitem_988 = getitem_989 = getitem_990 = getitem_991 = getitem_992 = getitem_993 = getitem_994 = getitem_995 = getitem_996 = getitem_997 = getitem_998 = getitem_999 = getitem_1000 = getitem_1001 = getitem_1002 = getitem_1003 = getitem_1004 = getitem_1005 = getitem_1006 = getitem_1007 = getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = getitem_1013 = getitem_1014 = getitem_1015 = getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = getitem_1023 = getitem_1024 = getitem_1025 = getitem_1026 = getitem_1027 = getitem_1028 = getitem_1029 = getitem_1030 = getitem_1031 = getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = None
        getitem: "f32[]" = _foreach_div_scalar[0]
        getitem_1048: "f32[]" = _foreach_div_scalar[1]
        getitem_1049: "f32[]" = _foreach_div_scalar[2]
        getitem_1050: "f32[]" = _foreach_div_scalar[3]
        getitem_1051: "f32[]" = _foreach_div_scalar[4]
        getitem_1052: "f32[]" = _foreach_div_scalar[5]
        getitem_1053: "f32[]" = _foreach_div_scalar[6]
        getitem_1054: "f32[]" = _foreach_div_scalar[7]
        getitem_1055: "f32[]" = _foreach_div_scalar[8]
        getitem_1056: "f32[]" = _foreach_div_scalar[9]
        getitem_1057: "f32[]" = _foreach_div_scalar[10]
        getitem_1058: "f32[]" = _foreach_div_scalar[11]
        getitem_1059: "f32[]" = _foreach_div_scalar[12]
        getitem_1060: "f32[]" = _foreach_div_scalar[13]
        getitem_1061: "f32[]" = _foreach_div_scalar[14]
        getitem_1062: "f32[]" = _foreach_div_scalar[15]
        getitem_1063: "f32[]" = _foreach_div_scalar[16]
        getitem_1064: "f32[]" = _foreach_div_scalar[17]
        getitem_1065: "f32[]" = _foreach_div_scalar[18]
        getitem_1066: "f32[]" = _foreach_div_scalar[19]
        getitem_1067: "f32[]" = _foreach_div_scalar[20]
        getitem_1068: "f32[]" = _foreach_div_scalar[21]
        getitem_1069: "f32[]" = _foreach_div_scalar[22]
        getitem_1070: "f32[]" = _foreach_div_scalar[23]
        getitem_1071: "f32[]" = _foreach_div_scalar[24]
        getitem_1072: "f32[]" = _foreach_div_scalar[25]
        getitem_1073: "f32[]" = _foreach_div_scalar[26]
        getitem_1074: "f32[]" = _foreach_div_scalar[27]
        getitem_1075: "f32[]" = _foreach_div_scalar[28]
        getitem_1076: "f32[]" = _foreach_div_scalar[29]
        getitem_1077: "f32[]" = _foreach_div_scalar[30]
        getitem_1078: "f32[]" = _foreach_div_scalar[31]
        getitem_1079: "f32[]" = _foreach_div_scalar[32]
        getitem_1080: "f32[]" = _foreach_div_scalar[33]
        getitem_1081: "f32[]" = _foreach_div_scalar[34]
        getitem_1082: "f32[]" = _foreach_div_scalar[35]
        getitem_1083: "f32[]" = _foreach_div_scalar[36]
        getitem_1084: "f32[]" = _foreach_div_scalar[37]
        getitem_1085: "f32[]" = _foreach_div_scalar[38]
        getitem_1086: "f32[]" = _foreach_div_scalar[39]
        getitem_1087: "f32[]" = _foreach_div_scalar[40]
        getitem_1088: "f32[]" = _foreach_div_scalar[41]
        getitem_1089: "f32[]" = _foreach_div_scalar[42]
        getitem_1090: "f32[]" = _foreach_div_scalar[43]
        getitem_1091: "f32[]" = _foreach_div_scalar[44]
        getitem_1092: "f32[]" = _foreach_div_scalar[45]
        getitem_1093: "f32[]" = _foreach_div_scalar[46]
        getitem_1094: "f32[]" = _foreach_div_scalar[47]
        getitem_1095: "f32[]" = _foreach_div_scalar[48]
        getitem_1096: "f32[]" = _foreach_div_scalar[49]
        getitem_1097: "f32[]" = _foreach_div_scalar[50]
        getitem_1098: "f32[]" = _foreach_div_scalar[51]
        getitem_1099: "f32[]" = _foreach_div_scalar[52]
        getitem_1100: "f32[]" = _foreach_div_scalar[53]
        getitem_1101: "f32[]" = _foreach_div_scalar[54]
        getitem_1102: "f32[]" = _foreach_div_scalar[55]
        getitem_1103: "f32[]" = _foreach_div_scalar[56]
        getitem_1104: "f32[]" = _foreach_div_scalar[57]
        getitem_1105: "f32[]" = _foreach_div_scalar[58]
        getitem_1106: "f32[]" = _foreach_div_scalar[59]
        getitem_1107: "f32[]" = _foreach_div_scalar[60]
        getitem_1108: "f32[]" = _foreach_div_scalar[61]
        getitem_1109: "f32[]" = _foreach_div_scalar[62]
        getitem_1110: "f32[]" = _foreach_div_scalar[63]
        getitem_1111: "f32[]" = _foreach_div_scalar[64]
        getitem_1112: "f32[]" = _foreach_div_scalar[65]
        getitem_1113: "f32[]" = _foreach_div_scalar[66]
        getitem_1114: "f32[]" = _foreach_div_scalar[67]
        getitem_1115: "f32[]" = _foreach_div_scalar[68]
        getitem_1116: "f32[]" = _foreach_div_scalar[69]
        getitem_1117: "f32[]" = _foreach_div_scalar[70]
        getitem_1118: "f32[]" = _foreach_div_scalar[71]
        getitem_1119: "f32[]" = _foreach_div_scalar[72]
        getitem_1120: "f32[]" = _foreach_div_scalar[73]
        getitem_1121: "f32[]" = _foreach_div_scalar[74]
        getitem_1122: "f32[]" = _foreach_div_scalar[75]
        getitem_1123: "f32[]" = _foreach_div_scalar[76]
        getitem_1124: "f32[]" = _foreach_div_scalar[77]
        getitem_1125: "f32[]" = _foreach_div_scalar[78]
        getitem_1126: "f32[]" = _foreach_div_scalar[79]
        getitem_1127: "f32[]" = _foreach_div_scalar[80]
        getitem_1128: "f32[]" = _foreach_div_scalar[81]
        getitem_1129: "f32[]" = _foreach_div_scalar[82]
        getitem_1130: "f32[]" = _foreach_div_scalar[83]
        getitem_1131: "f32[]" = _foreach_div_scalar[84]
        getitem_1132: "f32[]" = _foreach_div_scalar[85]
        getitem_1133: "f32[]" = _foreach_div_scalar[86]
        getitem_1134: "f32[]" = _foreach_div_scalar[87]
        getitem_1135: "f32[]" = _foreach_div_scalar[88]
        getitem_1136: "f32[]" = _foreach_div_scalar[89]
        getitem_1137: "f32[]" = _foreach_div_scalar[90]
        getitem_1138: "f32[]" = _foreach_div_scalar[91]
        getitem_1139: "f32[]" = _foreach_div_scalar[92]
        getitem_1140: "f32[]" = _foreach_div_scalar[93]
        getitem_1141: "f32[]" = _foreach_div_scalar[94]
        getitem_1142: "f32[]" = _foreach_div_scalar[95]
        getitem_1143: "f32[]" = _foreach_div_scalar[96]
        getitem_1144: "f32[]" = _foreach_div_scalar[97]
        getitem_1145: "f32[]" = _foreach_div_scalar[98]
        getitem_1146: "f32[]" = _foreach_div_scalar[99]
        getitem_1147: "f32[]" = _foreach_div_scalar[100]
        getitem_1148: "f32[]" = _foreach_div_scalar[101]
        getitem_1149: "f32[]" = _foreach_div_scalar[102]
        getitem_1150: "f32[]" = _foreach_div_scalar[103]
        getitem_1151: "f32[]" = _foreach_div_scalar[104]
        getitem_1152: "f32[]" = _foreach_div_scalar[105]
        getitem_1153: "f32[]" = _foreach_div_scalar[106]
        getitem_1154: "f32[]" = _foreach_div_scalar[107]
        getitem_1155: "f32[]" = _foreach_div_scalar[108]
        getitem_1156: "f32[]" = _foreach_div_scalar[109]
        getitem_1157: "f32[]" = _foreach_div_scalar[110]
        getitem_1158: "f32[]" = _foreach_div_scalar[111]
        getitem_1159: "f32[]" = _foreach_div_scalar[112]
        getitem_1160: "f32[]" = _foreach_div_scalar[113]
        getitem_1161: "f32[]" = _foreach_div_scalar[114]
        getitem_1162: "f32[]" = _foreach_div_scalar[115]
        getitem_1163: "f32[]" = _foreach_div_scalar[116]
        getitem_1164: "f32[]" = _foreach_div_scalar[117]
        getitem_1165: "f32[]" = _foreach_div_scalar[118]
        getitem_1166: "f32[]" = _foreach_div_scalar[119]
        getitem_1167: "f32[]" = _foreach_div_scalar[120]
        getitem_1168: "f32[]" = _foreach_div_scalar[121]
        getitem_1169: "f32[]" = _foreach_div_scalar[122]
        getitem_1170: "f32[]" = _foreach_div_scalar[123]
        getitem_1171: "f32[]" = _foreach_div_scalar[124]
        getitem_1172: "f32[]" = _foreach_div_scalar[125]
        getitem_1173: "f32[]" = _foreach_div_scalar[126]
        getitem_1174: "f32[]" = _foreach_div_scalar[127]
        getitem_1175: "f32[]" = _foreach_div_scalar[128]
        getitem_1176: "f32[]" = _foreach_div_scalar[129]
        getitem_1177: "f32[]" = _foreach_div_scalar[130];  _foreach_div_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833], [getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702]);  getitem_1703 = getitem_1704 = getitem_1705 = getitem_1706 = getitem_1707 = getitem_1708 = getitem_1709 = getitem_1710 = getitem_1711 = getitem_1712 = getitem_1713 = getitem_1714 = getitem_1715 = getitem_1716 = getitem_1717 = getitem_1718 = getitem_1719 = getitem_1720 = getitem_1721 = getitem_1722 = getitem_1723 = getitem_1724 = getitem_1725 = getitem_1726 = getitem_1727 = getitem_1728 = getitem_1729 = getitem_1730 = getitem_1731 = getitem_1732 = getitem_1733 = getitem_1734 = getitem_1735 = getitem_1736 = getitem_1737 = getitem_1738 = getitem_1739 = getitem_1740 = getitem_1741 = getitem_1742 = getitem_1743 = getitem_1744 = getitem_1745 = getitem_1746 = getitem_1747 = getitem_1748 = getitem_1749 = getitem_1750 = getitem_1751 = getitem_1752 = getitem_1753 = getitem_1754 = getitem_1755 = getitem_1756 = getitem_1757 = getitem_1758 = getitem_1759 = getitem_1760 = getitem_1761 = getitem_1762 = getitem_1763 = getitem_1764 = getitem_1765 = getitem_1766 = getitem_1767 = getitem_1768 = getitem_1769 = getitem_1770 = getitem_1771 = getitem_1772 = getitem_1773 = getitem_1774 = getitem_1775 = getitem_1776 = getitem_1777 = getitem_1778 = getitem_1779 = getitem_1780 = getitem_1781 = getitem_1782 = getitem_1783 = getitem_1784 = getitem_1785 = getitem_1786 = getitem_1787 = getitem_1788 = getitem_1789 = getitem_1790 = getitem_1791 = getitem_1792 = getitem_1793 = getitem_1794 = getitem_1795 = getitem_1796 = getitem_1797 = getitem_1798 = getitem_1799 = getitem_1800 = getitem_1801 = getitem_1802 = getitem_1803 = getitem_1804 = getitem_1805 = getitem_1806 = getitem_1807 = getitem_1808 = getitem_1809 = getitem_1810 = getitem_1811 = getitem_1812 = getitem_1813 = getitem_1814 = getitem_1815 = getitem_1816 = getitem_1817 = getitem_1818 = getitem_1819 = getitem_1820 = getitem_1821 = getitem_1822 = getitem_1823 = getitem_1824 = getitem_1825 = getitem_1826 = getitem_1827 = getitem_1828 = getitem_1829 = getitem_1830 = getitem_1831 = getitem_1832 = getitem_1833 = getitem_1572 = getitem_1573 = getitem_1574 = getitem_1575 = getitem_1576 = getitem_1577 = getitem_1578 = getitem_1579 = getitem_1580 = getitem_1581 = getitem_1582 = getitem_1583 = getitem_1584 = getitem_1585 = getitem_1586 = getitem_1587 = getitem_1588 = getitem_1589 = getitem_1590 = getitem_1591 = getitem_1592 = getitem_1593 = getitem_1594 = getitem_1595 = getitem_1596 = getitem_1597 = getitem_1598 = getitem_1599 = getitem_1600 = getitem_1601 = getitem_1602 = getitem_1603 = getitem_1604 = getitem_1605 = getitem_1606 = getitem_1607 = getitem_1608 = getitem_1609 = getitem_1610 = getitem_1611 = getitem_1612 = getitem_1613 = getitem_1614 = getitem_1615 = getitem_1616 = getitem_1617 = getitem_1618 = getitem_1619 = getitem_1620 = getitem_1621 = getitem_1622 = getitem_1623 = getitem_1624 = getitem_1625 = getitem_1626 = getitem_1627 = getitem_1628 = getitem_1629 = getitem_1630 = getitem_1631 = getitem_1632 = getitem_1633 = getitem_1634 = getitem_1635 = getitem_1636 = getitem_1637 = getitem_1638 = getitem_1639 = getitem_1640 = getitem_1641 = getitem_1642 = getitem_1643 = getitem_1644 = getitem_1645 = getitem_1646 = getitem_1647 = getitem_1648 = getitem_1649 = getitem_1650 = getitem_1651 = getitem_1652 = getitem_1653 = getitem_1654 = getitem_1655 = getitem_1656 = getitem_1657 = getitem_1658 = getitem_1659 = getitem_1660 = getitem_1661 = getitem_1662 = getitem_1663 = getitem_1664 = getitem_1665 = getitem_1666 = getitem_1667 = getitem_1668 = getitem_1669 = getitem_1670 = getitem_1671 = getitem_1672 = getitem_1673 = getitem_1674 = getitem_1675 = getitem_1676 = getitem_1677 = getitem_1678 = getitem_1679 = getitem_1680 = getitem_1681 = getitem_1682 = getitem_1683 = getitem_1684 = getitem_1685 = getitem_1686 = getitem_1687 = getitem_1688 = getitem_1689 = getitem_1690 = getitem_1691 = getitem_1692 = getitem_1693 = getitem_1694 = getitem_1695 = getitem_1696 = getitem_1697 = getitem_1698 = getitem_1699 = getitem_1700 = getitem_1701 = getitem_1702 = None
        getitem_1834: "f32[32128, 512]" = _foreach_div_list[0]
        getitem_1835: "f32[512, 512]" = _foreach_div_list[1]
        getitem_1836: "f32[512, 512]" = _foreach_div_list[2]
        getitem_1837: "f32[512, 512]" = _foreach_div_list[3]
        getitem_1838: "f32[512, 512]" = _foreach_div_list[4]
        getitem_1839: "f32[32, 8]" = _foreach_div_list[5]
        getitem_1840: "f32[512]" = _foreach_div_list[6]
        getitem_1841: "f32[2048, 512]" = _foreach_div_list[7]
        getitem_1842: "f32[512, 2048]" = _foreach_div_list[8]
        getitem_1843: "f32[512]" = _foreach_div_list[9]
        getitem_1844: "f32[512, 512]" = _foreach_div_list[10]
        getitem_1845: "f32[512, 512]" = _foreach_div_list[11]
        getitem_1846: "f32[512, 512]" = _foreach_div_list[12]
        getitem_1847: "f32[512, 512]" = _foreach_div_list[13]
        getitem_1848: "f32[512]" = _foreach_div_list[14]
        getitem_1849: "f32[2048, 512]" = _foreach_div_list[15]
        getitem_1850: "f32[512, 2048]" = _foreach_div_list[16]
        getitem_1851: "f32[512]" = _foreach_div_list[17]
        getitem_1852: "f32[512, 512]" = _foreach_div_list[18]
        getitem_1853: "f32[512, 512]" = _foreach_div_list[19]
        getitem_1854: "f32[512, 512]" = _foreach_div_list[20]
        getitem_1855: "f32[512, 512]" = _foreach_div_list[21]
        getitem_1856: "f32[512]" = _foreach_div_list[22]
        getitem_1857: "f32[2048, 512]" = _foreach_div_list[23]
        getitem_1858: "f32[512, 2048]" = _foreach_div_list[24]
        getitem_1859: "f32[512]" = _foreach_div_list[25]
        getitem_1860: "f32[512, 512]" = _foreach_div_list[26]
        getitem_1861: "f32[512, 512]" = _foreach_div_list[27]
        getitem_1862: "f32[512, 512]" = _foreach_div_list[28]
        getitem_1863: "f32[512, 512]" = _foreach_div_list[29]
        getitem_1864: "f32[512]" = _foreach_div_list[30]
        getitem_1865: "f32[2048, 512]" = _foreach_div_list[31]
        getitem_1866: "f32[512, 2048]" = _foreach_div_list[32]
        getitem_1867: "f32[512]" = _foreach_div_list[33]
        getitem_1868: "f32[512, 512]" = _foreach_div_list[34]
        getitem_1869: "f32[512, 512]" = _foreach_div_list[35]
        getitem_1870: "f32[512, 512]" = _foreach_div_list[36]
        getitem_1871: "f32[512, 512]" = _foreach_div_list[37]
        getitem_1872: "f32[512]" = _foreach_div_list[38]
        getitem_1873: "f32[2048, 512]" = _foreach_div_list[39]
        getitem_1874: "f32[512, 2048]" = _foreach_div_list[40]
        getitem_1875: "f32[512]" = _foreach_div_list[41]
        getitem_1876: "f32[512, 512]" = _foreach_div_list[42]
        getitem_1877: "f32[512, 512]" = _foreach_div_list[43]
        getitem_1878: "f32[512, 512]" = _foreach_div_list[44]
        getitem_1879: "f32[512, 512]" = _foreach_div_list[45]
        getitem_1880: "f32[512]" = _foreach_div_list[46]
        getitem_1881: "f32[2048, 512]" = _foreach_div_list[47]
        getitem_1882: "f32[512, 2048]" = _foreach_div_list[48]
        getitem_1883: "f32[512]" = _foreach_div_list[49]
        getitem_1884: "f32[512]" = _foreach_div_list[50]
        getitem_1885: "f32[512, 512]" = _foreach_div_list[51]
        getitem_1886: "f32[512, 512]" = _foreach_div_list[52]
        getitem_1887: "f32[512, 512]" = _foreach_div_list[53]
        getitem_1888: "f32[512, 512]" = _foreach_div_list[54]
        getitem_1889: "f32[32, 8]" = _foreach_div_list[55]
        getitem_1890: "f32[512]" = _foreach_div_list[56]
        getitem_1891: "f32[512, 512]" = _foreach_div_list[57]
        getitem_1892: "f32[512, 512]" = _foreach_div_list[58]
        getitem_1893: "f32[512, 512]" = _foreach_div_list[59]
        getitem_1894: "f32[512, 512]" = _foreach_div_list[60]
        getitem_1895: "f32[512]" = _foreach_div_list[61]
        getitem_1896: "f32[2048, 512]" = _foreach_div_list[62]
        getitem_1897: "f32[512, 2048]" = _foreach_div_list[63]
        getitem_1898: "f32[512]" = _foreach_div_list[64]
        getitem_1899: "f32[512, 512]" = _foreach_div_list[65]
        getitem_1900: "f32[512, 512]" = _foreach_div_list[66]
        getitem_1901: "f32[512, 512]" = _foreach_div_list[67]
        getitem_1902: "f32[512, 512]" = _foreach_div_list[68]
        getitem_1903: "f32[512]" = _foreach_div_list[69]
        getitem_1904: "f32[512, 512]" = _foreach_div_list[70]
        getitem_1905: "f32[512, 512]" = _foreach_div_list[71]
        getitem_1906: "f32[512, 512]" = _foreach_div_list[72]
        getitem_1907: "f32[512, 512]" = _foreach_div_list[73]
        getitem_1908: "f32[512]" = _foreach_div_list[74]
        getitem_1909: "f32[2048, 512]" = _foreach_div_list[75]
        getitem_1910: "f32[512, 2048]" = _foreach_div_list[76]
        getitem_1911: "f32[512]" = _foreach_div_list[77]
        getitem_1912: "f32[512, 512]" = _foreach_div_list[78]
        getitem_1913: "f32[512, 512]" = _foreach_div_list[79]
        getitem_1914: "f32[512, 512]" = _foreach_div_list[80]
        getitem_1915: "f32[512, 512]" = _foreach_div_list[81]
        getitem_1916: "f32[512]" = _foreach_div_list[82]
        getitem_1917: "f32[512, 512]" = _foreach_div_list[83]
        getitem_1918: "f32[512, 512]" = _foreach_div_list[84]
        getitem_1919: "f32[512, 512]" = _foreach_div_list[85]
        getitem_1920: "f32[512, 512]" = _foreach_div_list[86]
        getitem_1921: "f32[512]" = _foreach_div_list[87]
        getitem_1922: "f32[2048, 512]" = _foreach_div_list[88]
        getitem_1923: "f32[512, 2048]" = _foreach_div_list[89]
        getitem_1924: "f32[512]" = _foreach_div_list[90]
        getitem_1925: "f32[512, 512]" = _foreach_div_list[91]
        getitem_1926: "f32[512, 512]" = _foreach_div_list[92]
        getitem_1927: "f32[512, 512]" = _foreach_div_list[93]
        getitem_1928: "f32[512, 512]" = _foreach_div_list[94]
        getitem_1929: "f32[512]" = _foreach_div_list[95]
        getitem_1930: "f32[512, 512]" = _foreach_div_list[96]
        getitem_1931: "f32[512, 512]" = _foreach_div_list[97]
        getitem_1932: "f32[512, 512]" = _foreach_div_list[98]
        getitem_1933: "f32[512, 512]" = _foreach_div_list[99]
        getitem_1934: "f32[512]" = _foreach_div_list[100]
        getitem_1935: "f32[2048, 512]" = _foreach_div_list[101]
        getitem_1936: "f32[512, 2048]" = _foreach_div_list[102]
        getitem_1937: "f32[512]" = _foreach_div_list[103]
        getitem_1938: "f32[512, 512]" = _foreach_div_list[104]
        getitem_1939: "f32[512, 512]" = _foreach_div_list[105]
        getitem_1940: "f32[512, 512]" = _foreach_div_list[106]
        getitem_1941: "f32[512, 512]" = _foreach_div_list[107]
        getitem_1942: "f32[512]" = _foreach_div_list[108]
        getitem_1943: "f32[512, 512]" = _foreach_div_list[109]
        getitem_1944: "f32[512, 512]" = _foreach_div_list[110]
        getitem_1945: "f32[512, 512]" = _foreach_div_list[111]
        getitem_1946: "f32[512, 512]" = _foreach_div_list[112]
        getitem_1947: "f32[512]" = _foreach_div_list[113]
        getitem_1948: "f32[2048, 512]" = _foreach_div_list[114]
        getitem_1949: "f32[512, 2048]" = _foreach_div_list[115]
        getitem_1950: "f32[512]" = _foreach_div_list[116]
        getitem_1951: "f32[512, 512]" = _foreach_div_list[117]
        getitem_1952: "f32[512, 512]" = _foreach_div_list[118]
        getitem_1953: "f32[512, 512]" = _foreach_div_list[119]
        getitem_1954: "f32[512, 512]" = _foreach_div_list[120]
        getitem_1955: "f32[512]" = _foreach_div_list[121]
        getitem_1956: "f32[512, 512]" = _foreach_div_list[122]
        getitem_1957: "f32[512, 512]" = _foreach_div_list[123]
        getitem_1958: "f32[512, 512]" = _foreach_div_list[124]
        getitem_1959: "f32[512, 512]" = _foreach_div_list[125]
        getitem_1960: "f32[512]" = _foreach_div_list[126]
        getitem_1961: "f32[2048, 512]" = _foreach_div_list[127]
        getitem_1962: "f32[512, 2048]" = _foreach_div_list[128]
        getitem_1963: "f32[512]" = _foreach_div_list[129]
        getitem_1964: "f32[512]" = _foreach_div_list[130];  _foreach_div_list = None
        return (getitem, getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

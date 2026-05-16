"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: eb1d58ae0860
Shape hash: 36649dda
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, getitem_1030: "f32[]", getitem_1031: "f32[]", getitem_1032: "f32[]", getitem_1033: "f32[]", getitem_1034: "f32[]", getitem_1035: "f32[]", getitem_1036: "f32[]", getitem_1037: "f32[]", getitem_1038: "f32[]", getitem_1039: "f32[]", getitem_1040: "f32[]", getitem_1041: "f32[]", getitem_1042: "f32[]", getitem_1043: "f32[]", getitem_1044: "f32[]", getitem_1045: "f32[]", getitem_1046: "f32[]", getitem_1047: "f32[]", getitem_1048: "f32[]", getitem_1049: "f32[]", getitem_1050: "f32[]", getitem_1051: "f32[]", getitem_1052: "f32[]", getitem_1053: "f32[]", getitem_1054: "f32[]", getitem_1055: "f32[]", getitem_1056: "f32[]", getitem_1057: "f32[]", getitem_1058: "f32[]", getitem_1059: "f32[]", getitem_1060: "f32[]", getitem_1061: "f32[]", getitem_1062: "f32[]", getitem_1063: "f32[]", getitem_1064: "f32[]", getitem_1065: "f32[]", getitem_1066: "f32[]", getitem_1067: "f32[]", getitem_1068: "f32[]", getitem_1069: "f32[]", getitem_1070: "f32[]", getitem_1071: "f32[]", getitem_1072: "f32[]", getitem_1073: "f32[]", getitem_1074: "f32[]", getitem_1075: "f32[]", getitem_1076: "f32[]", getitem_1077: "f32[]", getitem_1078: "f32[]", getitem_1079: "f32[]", getitem_1080: "f32[]", getitem_1081: "f32[]", getitem_1082: "f32[]", getitem_1083: "f32[]", getitem_1084: "f32[]", getitem_1085: "f32[]", getitem_1086: "f32[]", getitem_1087: "f32[]", getitem_1088: "f32[]", getitem_1089: "f32[]", getitem_1090: "f32[]", getitem_1091: "f32[]", getitem_1092: "f32[]", getitem_1093: "f32[]", getitem_1094: "f32[]", getitem_1095: "f32[]", getitem_1096: "f32[]", getitem_1097: "f32[]", getitem_1098: "f32[]", getitem_1099: "f32[]", getitem_1100: "f32[]", getitem_1101: "f32[]", getitem_1102: "f32[]", getitem_1103: "f32[]", getitem_1104: "f32[]", getitem_1105: "f32[]", getitem_1106: "f32[]", getitem_1107: "f32[]", getitem_1108: "f32[]", getitem_1109: "f32[]", getitem_1110: "f32[]", getitem_1111: "f32[]", getitem_1112: "f32[]", getitem_1113: "f32[]", getitem_1114: "f32[]", getitem_1115: "f32[]", getitem_1116: "f32[]", getitem_1117: "f32[]", getitem_1118: "f32[]", getitem_1119: "f32[]", getitem_1120: "f32[]", getitem_1121: "f32[]", getitem_1122: "f32[]", getitem_1123: "f32[]", getitem_1124: "f32[]", getitem_1125: "f32[]", getitem_1126: "f32[]", getitem_1127: "f32[]", getitem_1128: "f32[]", getitem_1129: "f32[]", getitem_1130: "f32[]", getitem_1131: "f32[]", getitem_1132: "f32[]", getitem_1133: "f32[]", getitem_1134: "f32[]", getitem_1135: "f32[]", getitem_1136: "f32[]", getitem_1137: "f32[]", getitem_1138: "f32[]", getitem_1139: "f32[]", getitem_1140: "f32[]", getitem_1141: "f32[]", getitem_1142: "f32[]", getitem_1143: "f32[]", getitem_1144: "f32[]", getitem_1145: "f32[]", getitem_1146: "f32[]", getitem_1147: "f32[]", getitem_1148: "f32[]", getitem_1149: "f32[]", getitem_1150: "f32[]", getitem_1151: "f32[]", getitem_1152: "f32[]", getitem_1153: "f32[]", getitem_1154: "f32[]", getitem_1155: "f32[]", getitem_1156: "f32[]", getitem_1157: "f32[]", getitem_1158: "f32[]", getitem_1159: "f32[]", getitem_1160: "f32[]", getitem_1161: "f32[]", getitem_1162: "f32[]", getitem_1163: "f32[]", getitem_1164: "f32[]", getitem_1165: "f32[]", getitem_1166: "f32[]", getitem_1167: "f32[]", getitem_1168: "f32[]", getitem_1169: "f32[]", getitem_1170: "f32[]", getitem_1171: "f32[]", getitem_1172: "f32[]", getitem_1173: "f32[]", getitem_1174: "f32[]", getitem_1175: "f32[]", getitem_1176: "f32[]", getitem_1177: "f32[]", getitem_1178: "f32[]", getitem_1179: "f32[]", getitem_1180: "f32[]", getitem_1181: "f32[]", getitem_1182: "f32[]", getitem_1183: "f32[]", getitem_1184: "f32[]", getitem_1185: "f32[]", getitem_1186: "f32[]", getitem_1187: "f32[]", getitem_1188: "f32[]", getitem_1189: "f32[]", getitem_1190: "f32[]", getitem_1191: "f32[]", getitem_1192: "f32[]", getitem_1193: "f32[]", getitem_1194: "f32[]", getitem_1195: "f32[]", getitem_1196: "f32[]", getitem_1197: "f32[]", getitem_1198: "f32[]", getitem_1199: "f32[]", getitem_1200: "f32[]", getitem_1201: "f32[]", getitem_1202: "f32[]", getitem_1203: "f32[]", getitem_1204: "f32[]", getitem_1205: "f32[]", getitem_1206: "f32[]", getitem_1207: "f32[]", getitem_1208: "f32[]", getitem_1209: "f32[]", getitem_1210: "f32[]", getitem_1211: "f32[]", getitem_1212: "f32[]", getitem_1213: "f32[]", getitem_1214: "f32[]", getitem_1215: "f32[]", getitem_1216: "f32[]", getitem_1217: "f32[]", getitem_1218: "f32[]", getitem_1219: "f32[]", getitem_1220: "f32[]", getitem_1221: "f32[]", getitem_1222: "f32[]", getitem_1223: "f32[]", getitem_1224: "f32[]", getitem_1225: "f32[]", getitem_1226: "f32[]", getitem_1227: "f32[]", getitem_1228: "f32[]", getitem_1229: "f32[]", getitem_1230: "f32[]", getitem_1231: "f32[]", getitem_1232: "f32[]", getitem_1233: "f32[]", getitem_1234: "f32[]", getitem_1235: "f32[]", getitem_1854: "f32[]", getitem_1855: "f32[]", getitem_1856: "f32[]", getitem_1857: "f32[]", getitem_1858: "f32[]", getitem_1859: "f32[]", getitem_1860: "f32[]", getitem_1861: "f32[]", getitem_1862: "f32[]", getitem_1863: "f32[]", getitem_1864: "f32[]", getitem_1865: "f32[]", getitem_1866: "f32[]", getitem_1867: "f32[]", getitem_1868: "f32[]", getitem_1869: "f32[]", getitem_1870: "f32[]", getitem_1871: "f32[]", getitem_1872: "f32[]", getitem_1873: "f32[]", getitem_1874: "f32[]", getitem_1875: "f32[]", getitem_1876: "f32[]", getitem_1877: "f32[]", getitem_1878: "f32[]", getitem_1879: "f32[]", getitem_1880: "f32[]", getitem_1881: "f32[]", getitem_1882: "f32[]", getitem_1883: "f32[]", getitem_1884: "f32[]", getitem_1885: "f32[]", getitem_1886: "f32[]", getitem_1887: "f32[]", getitem_1888: "f32[]", getitem_1889: "f32[]", getitem_1890: "f32[]", getitem_1891: "f32[]", getitem_1892: "f32[]", getitem_1893: "f32[]", getitem_1894: "f32[]", getitem_1895: "f32[]", getitem_1896: "f32[]", getitem_1897: "f32[]", getitem_1898: "f32[]", getitem_1899: "f32[]", getitem_1900: "f32[]", getitem_1901: "f32[]", getitem_1902: "f32[]", getitem_1903: "f32[]", getitem_1904: "f32[]", getitem_1905: "f32[]", getitem_1906: "f32[]", getitem_1907: "f32[]", getitem_1908: "f32[]", getitem_1909: "f32[]", getitem_1910: "f32[]", getitem_1911: "f32[]", getitem_1912: "f32[]", getitem_1913: "f32[]", getitem_1914: "f32[]", getitem_1915: "f32[]", getitem_1916: "f32[]", getitem_1917: "f32[]", getitem_1918: "f32[]", getitem_1919: "f32[]", getitem_1920: "f32[]", getitem_1921: "f32[]", getitem_1922: "f32[]", getitem_1923: "f32[]", getitem_1924: "f32[]", getitem_1925: "f32[]", getitem_1926: "f32[]", getitem_1927: "f32[]", getitem_1928: "f32[]", getitem_1929: "f32[]", getitem_1930: "f32[]", getitem_1931: "f32[]", getitem_1932: "f32[]", getitem_1933: "f32[]", getitem_1934: "f32[]", getitem_1935: "f32[]", getitem_1936: "f32[]", getitem_1937: "f32[]", getitem_1938: "f32[]", getitem_1939: "f32[]", getitem_1940: "f32[]", getitem_1941: "f32[]", getitem_1942: "f32[]", getitem_1943: "f32[]", getitem_1944: "f32[]", getitem_1945: "f32[]", getitem_1946: "f32[]", getitem_1947: "f32[]", getitem_1948: "f32[]", getitem_1949: "f32[]", getitem_1950: "f32[]", getitem_1951: "f32[]", getitem_1952: "f32[]", getitem_1953: "f32[]", getitem_1954: "f32[]", getitem_1955: "f32[]", getitem_1956: "f32[]", getitem_1957: "f32[]", getitem_1958: "f32[]", getitem_1959: "f32[]", getitem_1960: "f32[]", getitem_1961: "f32[]", getitem_1962: "f32[]", getitem_1963: "f32[]", getitem_1964: "f32[]", getitem_1965: "f32[]", getitem_1966: "f32[]", getitem_1967: "f32[]", getitem_1968: "f32[]", getitem_1969: "f32[]", getitem_1970: "f32[]", getitem_1971: "f32[]", getitem_1972: "f32[]", getitem_1973: "f32[]", getitem_1974: "f32[]", getitem_1975: "f32[]", getitem_1976: "f32[]", getitem_1977: "f32[]", getitem_1978: "f32[]", getitem_1979: "f32[]", getitem_1980: "f32[]", getitem_1981: "f32[]", getitem_1982: "f32[]", getitem_1983: "f32[]", getitem_1984: "f32[]", getitem_1985: "f32[]", getitem_1986: "f32[]", getitem_1987: "f32[]", getitem_1988: "f32[]", getitem_1989: "f32[]", getitem_1990: "f32[]", getitem_1991: "f32[]", getitem_1992: "f32[]", getitem_1993: "f32[]", getitem_1994: "f32[]", getitem_1995: "f32[]", getitem_1996: "f32[]", getitem_1997: "f32[]", getitem_1998: "f32[]", getitem_1999: "f32[]", getitem_2000: "f32[]", getitem_2001: "f32[]", getitem_2002: "f32[]", getitem_2003: "f32[]", getitem_2004: "f32[]", getitem_2005: "f32[]", getitem_2006: "f32[]", getitem_2007: "f32[]", getitem_2008: "f32[]", getitem_2009: "f32[]", getitem_2010: "f32[]", getitem_2011: "f32[]", getitem_2012: "f32[]", getitem_2013: "f32[]", getitem_2014: "f32[]", getitem_2015: "f32[]", getitem_2016: "f32[]", getitem_2017: "f32[]", getitem_2018: "f32[]", getitem_2019: "f32[]", getitem_2020: "f32[]", getitem_2021: "f32[]", getitem_2022: "f32[]", getitem_2023: "f32[]", getitem_2024: "f32[]", getitem_2025: "f32[]", getitem_2026: "f32[]", getitem_2027: "f32[]", getitem_2028: "f32[]", getitem_2029: "f32[]", getitem_2030: "f32[]", getitem_2031: "f32[]", getitem_2032: "f32[]", getitem_2033: "f32[]", getitem_2034: "f32[]", getitem_2035: "f32[]", getitem_2036: "f32[]", getitem_2037: "f32[]", getitem_2038: "f32[]", getitem_2039: "f32[]", getitem_2040: "f32[]", getitem_2041: "f32[]", getitem_2042: "f32[]", getitem_2043: "f32[]", getitem_2044: "f32[]", getitem_2045: "f32[]", getitem_2046: "f32[]", getitem_2047: "f32[]", getitem_2048: "f32[]", getitem_2049: "f32[]", getitem_2050: "f32[]", getitem_2051: "f32[]", getitem_2052: "f32[]", getitem_2053: "f32[]", getitem_2054: "f32[]", getitem_2055: "f32[]", getitem_2056: "f32[]", getitem_2057: "f32[]", getitem_2058: "f32[]", getitem_2059: "f32[]", getitem_824: "f32[30522, 768]", getitem_825: "f32[512, 768]", getitem_826: "f32[1024, 768]", getitem_827: "f32[1024, 768]", getitem_828: "f32[1024, 768]", getitem_829: "f32[1024, 768]", getitem_830: "f32[2, 768]", getitem_831: "f32[768]", getitem_832: "f32[768]", getitem_833: "f32[768, 768]", getitem_834: "f32[768]", getitem_835: "f32[768, 768]", getitem_836: "f32[768]", getitem_837: "f32[768, 768]", getitem_838: "f32[768]", getitem_839: "f32[768, 768]", getitem_840: "f32[768]", getitem_841: "f32[768]", getitem_842: "f32[768]", getitem_843: "f32[3072, 768]", getitem_844: "f32[3072]", getitem_845: "f32[768, 3072]", getitem_846: "f32[768]", getitem_847: "f32[768]", getitem_848: "f32[768]", getitem_849: "f32[768, 768]", getitem_850: "f32[768]", getitem_851: "f32[768, 768]", getitem_852: "f32[768]", getitem_853: "f32[768, 768]", getitem_854: "f32[768]", getitem_855: "f32[768, 768]", getitem_856: "f32[768]", getitem_857: "f32[768]", getitem_858: "f32[768]", getitem_859: "f32[3072, 768]", getitem_860: "f32[3072]", getitem_861: "f32[768, 3072]", getitem_862: "f32[768]", getitem_863: "f32[768]", getitem_864: "f32[768]", getitem_865: "f32[768, 768]", getitem_866: "f32[768]", getitem_867: "f32[768, 768]", getitem_868: "f32[768]", getitem_869: "f32[768, 768]", getitem_870: "f32[768]", getitem_871: "f32[768, 768]", getitem_872: "f32[768]", getitem_873: "f32[768]", getitem_874: "f32[768]", getitem_875: "f32[3072, 768]", getitem_876: "f32[3072]", getitem_877: "f32[768, 3072]", getitem_878: "f32[768]", getitem_879: "f32[768]", getitem_880: "f32[768]", getitem_881: "f32[768, 768]", getitem_882: "f32[768]", getitem_883: "f32[768, 768]", getitem_884: "f32[768]", getitem_885: "f32[768, 768]", getitem_886: "f32[768]", getitem_887: "f32[768, 768]", getitem_888: "f32[768]", getitem_889: "f32[768]", getitem_890: "f32[768]", getitem_891: "f32[3072, 768]", getitem_892: "f32[3072]", getitem_893: "f32[768, 3072]", getitem_894: "f32[768]", getitem_895: "f32[768]", getitem_896: "f32[768]", getitem_897: "f32[768, 768]", getitem_898: "f32[768]", getitem_899: "f32[768, 768]", getitem_900: "f32[768]", getitem_901: "f32[768, 768]", getitem_902: "f32[768]", getitem_903: "f32[768, 768]", getitem_904: "f32[768]", getitem_905: "f32[768]", getitem_906: "f32[768]", getitem_907: "f32[3072, 768]", getitem_908: "f32[3072]", getitem_909: "f32[768, 3072]", getitem_910: "f32[768]", getitem_911: "f32[768]", getitem_912: "f32[768]", getitem_913: "f32[768, 768]", getitem_914: "f32[768]", getitem_915: "f32[768, 768]", getitem_916: "f32[768]", getitem_917: "f32[768, 768]", getitem_918: "f32[768]", getitem_919: "f32[768, 768]", getitem_920: "f32[768]", getitem_921: "f32[768]", getitem_922: "f32[768]", getitem_923: "f32[3072, 768]", getitem_924: "f32[3072]", getitem_925: "f32[768, 3072]", getitem_926: "f32[768]", getitem_927: "f32[768]", getitem_928: "f32[768]", getitem_929: "f32[768, 768]", getitem_930: "f32[768]", getitem_931: "f32[768, 768]", getitem_932: "f32[768]", getitem_933: "f32[768, 768]", getitem_934: "f32[768]", getitem_935: "f32[768, 768]", getitem_936: "f32[768]", getitem_937: "f32[768]", getitem_938: "f32[768]", getitem_939: "f32[3072, 768]", getitem_940: "f32[3072]", getitem_941: "f32[768, 3072]", getitem_942: "f32[768]", getitem_943: "f32[768]", getitem_944: "f32[768]", getitem_945: "f32[768, 768]", getitem_946: "f32[768]", getitem_947: "f32[768, 768]", getitem_948: "f32[768]", getitem_949: "f32[768, 768]", getitem_950: "f32[768]", getitem_951: "f32[768, 768]", getitem_952: "f32[768]", getitem_953: "f32[768]", getitem_954: "f32[768]", getitem_955: "f32[3072, 768]", getitem_956: "f32[3072]", getitem_957: "f32[768, 3072]", getitem_958: "f32[768]", getitem_959: "f32[768]", getitem_960: "f32[768]", getitem_961: "f32[768, 768]", getitem_962: "f32[768]", getitem_963: "f32[768, 768]", getitem_964: "f32[768]", getitem_965: "f32[768, 768]", getitem_966: "f32[768]", getitem_967: "f32[768, 768]", getitem_968: "f32[768]", getitem_969: "f32[768]", getitem_970: "f32[768]", getitem_971: "f32[3072, 768]", getitem_972: "f32[3072]", getitem_973: "f32[768, 3072]", getitem_974: "f32[768]", getitem_975: "f32[768]", getitem_976: "f32[768]", getitem_977: "f32[768, 768]", getitem_978: "f32[768]", getitem_979: "f32[768, 768]", getitem_980: "f32[768]", getitem_981: "f32[768, 768]", getitem_982: "f32[768]", getitem_983: "f32[768, 768]", getitem_984: "f32[768]", getitem_985: "f32[768]", getitem_986: "f32[768]", getitem_987: "f32[3072, 768]", getitem_988: "f32[3072]", getitem_989: "f32[768, 3072]", getitem_990: "f32[768]", getitem_991: "f32[768]", getitem_992: "f32[768]", getitem_993: "f32[768, 768]", getitem_994: "f32[768]", getitem_995: "f32[768, 768]", getitem_996: "f32[768]", getitem_997: "f32[768, 768]", getitem_998: "f32[768]", getitem_999: "f32[768, 768]", getitem_1000: "f32[768]", getitem_1001: "f32[768]", getitem_1002: "f32[768]", getitem_1003: "f32[3072, 768]", getitem_1004: "f32[3072]", getitem_1005: "f32[768, 3072]", getitem_1006: "f32[768]", getitem_1007: "f32[768]", getitem_1008: "f32[768]", getitem_1009: "f32[768, 768]", getitem_1010: "f32[768]", getitem_1011: "f32[768, 768]", getitem_1012: "f32[768]", getitem_1013: "f32[768, 768]", getitem_1014: "f32[768]", getitem_1015: "f32[768, 768]", getitem_1016: "f32[768]", getitem_1017: "f32[768]", getitem_1018: "f32[768]", getitem_1019: "f32[3072, 768]", getitem_1020: "f32[3072]", getitem_1021: "f32[768, 3072]", getitem_1022: "f32[768]", getitem_1023: "f32[768]", getitem_1024: "f32[768]", getitem_1025: "f32[30522]", getitem_1026: "f32[768, 768]", getitem_1027: "f32[768]", getitem_1028: "f32[768]", getitem_1029: "f32[768]"):
        # No stacktrace found for following nodes
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047, getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235], 1);  getitem_1030 = getitem_1031 = getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = getitem_1048 = getitem_1049 = getitem_1050 = getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = getitem_1056 = getitem_1057 = getitem_1058 = getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = getitem_1064 = getitem_1065 = getitem_1066 = getitem_1067 = getitem_1068 = getitem_1069 = getitem_1070 = getitem_1071 = getitem_1072 = getitem_1073 = getitem_1074 = getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = getitem_1080 = getitem_1081 = getitem_1082 = getitem_1083 = getitem_1084 = getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = getitem_1153 = getitem_1154 = getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = getitem_1179 = getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = getitem_1187 = getitem_1188 = getitem_1189 = getitem_1190 = getitem_1191 = getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = getitem_1234 = getitem_1235 = None
        getitem: "f32[]" = _foreach_sub_scalar[0]
        getitem_1236: "f32[]" = _foreach_sub_scalar[1]
        getitem_1237: "f32[]" = _foreach_sub_scalar[2]
        getitem_1238: "f32[]" = _foreach_sub_scalar[3]
        getitem_1239: "f32[]" = _foreach_sub_scalar[4]
        getitem_1240: "f32[]" = _foreach_sub_scalar[5]
        getitem_1241: "f32[]" = _foreach_sub_scalar[6]
        getitem_1242: "f32[]" = _foreach_sub_scalar[7]
        getitem_1243: "f32[]" = _foreach_sub_scalar[8]
        getitem_1244: "f32[]" = _foreach_sub_scalar[9]
        getitem_1245: "f32[]" = _foreach_sub_scalar[10]
        getitem_1246: "f32[]" = _foreach_sub_scalar[11]
        getitem_1247: "f32[]" = _foreach_sub_scalar[12]
        getitem_1248: "f32[]" = _foreach_sub_scalar[13]
        getitem_1249: "f32[]" = _foreach_sub_scalar[14]
        getitem_1250: "f32[]" = _foreach_sub_scalar[15]
        getitem_1251: "f32[]" = _foreach_sub_scalar[16]
        getitem_1252: "f32[]" = _foreach_sub_scalar[17]
        getitem_1253: "f32[]" = _foreach_sub_scalar[18]
        getitem_1254: "f32[]" = _foreach_sub_scalar[19]
        getitem_1255: "f32[]" = _foreach_sub_scalar[20]
        getitem_1256: "f32[]" = _foreach_sub_scalar[21]
        getitem_1257: "f32[]" = _foreach_sub_scalar[22]
        getitem_1258: "f32[]" = _foreach_sub_scalar[23]
        getitem_1259: "f32[]" = _foreach_sub_scalar[24]
        getitem_1260: "f32[]" = _foreach_sub_scalar[25]
        getitem_1261: "f32[]" = _foreach_sub_scalar[26]
        getitem_1262: "f32[]" = _foreach_sub_scalar[27]
        getitem_1263: "f32[]" = _foreach_sub_scalar[28]
        getitem_1264: "f32[]" = _foreach_sub_scalar[29]
        getitem_1265: "f32[]" = _foreach_sub_scalar[30]
        getitem_1266: "f32[]" = _foreach_sub_scalar[31]
        getitem_1267: "f32[]" = _foreach_sub_scalar[32]
        getitem_1268: "f32[]" = _foreach_sub_scalar[33]
        getitem_1269: "f32[]" = _foreach_sub_scalar[34]
        getitem_1270: "f32[]" = _foreach_sub_scalar[35]
        getitem_1271: "f32[]" = _foreach_sub_scalar[36]
        getitem_1272: "f32[]" = _foreach_sub_scalar[37]
        getitem_1273: "f32[]" = _foreach_sub_scalar[38]
        getitem_1274: "f32[]" = _foreach_sub_scalar[39]
        getitem_1275: "f32[]" = _foreach_sub_scalar[40]
        getitem_1276: "f32[]" = _foreach_sub_scalar[41]
        getitem_1277: "f32[]" = _foreach_sub_scalar[42]
        getitem_1278: "f32[]" = _foreach_sub_scalar[43]
        getitem_1279: "f32[]" = _foreach_sub_scalar[44]
        getitem_1280: "f32[]" = _foreach_sub_scalar[45]
        getitem_1281: "f32[]" = _foreach_sub_scalar[46]
        getitem_1282: "f32[]" = _foreach_sub_scalar[47]
        getitem_1283: "f32[]" = _foreach_sub_scalar[48]
        getitem_1284: "f32[]" = _foreach_sub_scalar[49]
        getitem_1285: "f32[]" = _foreach_sub_scalar[50]
        getitem_1286: "f32[]" = _foreach_sub_scalar[51]
        getitem_1287: "f32[]" = _foreach_sub_scalar[52]
        getitem_1288: "f32[]" = _foreach_sub_scalar[53]
        getitem_1289: "f32[]" = _foreach_sub_scalar[54]
        getitem_1290: "f32[]" = _foreach_sub_scalar[55]
        getitem_1291: "f32[]" = _foreach_sub_scalar[56]
        getitem_1292: "f32[]" = _foreach_sub_scalar[57]
        getitem_1293: "f32[]" = _foreach_sub_scalar[58]
        getitem_1294: "f32[]" = _foreach_sub_scalar[59]
        getitem_1295: "f32[]" = _foreach_sub_scalar[60]
        getitem_1296: "f32[]" = _foreach_sub_scalar[61]
        getitem_1297: "f32[]" = _foreach_sub_scalar[62]
        getitem_1298: "f32[]" = _foreach_sub_scalar[63]
        getitem_1299: "f32[]" = _foreach_sub_scalar[64]
        getitem_1300: "f32[]" = _foreach_sub_scalar[65]
        getitem_1301: "f32[]" = _foreach_sub_scalar[66]
        getitem_1302: "f32[]" = _foreach_sub_scalar[67]
        getitem_1303: "f32[]" = _foreach_sub_scalar[68]
        getitem_1304: "f32[]" = _foreach_sub_scalar[69]
        getitem_1305: "f32[]" = _foreach_sub_scalar[70]
        getitem_1306: "f32[]" = _foreach_sub_scalar[71]
        getitem_1307: "f32[]" = _foreach_sub_scalar[72]
        getitem_1308: "f32[]" = _foreach_sub_scalar[73]
        getitem_1309: "f32[]" = _foreach_sub_scalar[74]
        getitem_1310: "f32[]" = _foreach_sub_scalar[75]
        getitem_1311: "f32[]" = _foreach_sub_scalar[76]
        getitem_1312: "f32[]" = _foreach_sub_scalar[77]
        getitem_1313: "f32[]" = _foreach_sub_scalar[78]
        getitem_1314: "f32[]" = _foreach_sub_scalar[79]
        getitem_1315: "f32[]" = _foreach_sub_scalar[80]
        getitem_1316: "f32[]" = _foreach_sub_scalar[81]
        getitem_1317: "f32[]" = _foreach_sub_scalar[82]
        getitem_1318: "f32[]" = _foreach_sub_scalar[83]
        getitem_1319: "f32[]" = _foreach_sub_scalar[84]
        getitem_1320: "f32[]" = _foreach_sub_scalar[85]
        getitem_1321: "f32[]" = _foreach_sub_scalar[86]
        getitem_1322: "f32[]" = _foreach_sub_scalar[87]
        getitem_1323: "f32[]" = _foreach_sub_scalar[88]
        getitem_1324: "f32[]" = _foreach_sub_scalar[89]
        getitem_1325: "f32[]" = _foreach_sub_scalar[90]
        getitem_1326: "f32[]" = _foreach_sub_scalar[91]
        getitem_1327: "f32[]" = _foreach_sub_scalar[92]
        getitem_1328: "f32[]" = _foreach_sub_scalar[93]
        getitem_1329: "f32[]" = _foreach_sub_scalar[94]
        getitem_1330: "f32[]" = _foreach_sub_scalar[95]
        getitem_1331: "f32[]" = _foreach_sub_scalar[96]
        getitem_1332: "f32[]" = _foreach_sub_scalar[97]
        getitem_1333: "f32[]" = _foreach_sub_scalar[98]
        getitem_1334: "f32[]" = _foreach_sub_scalar[99]
        getitem_1335: "f32[]" = _foreach_sub_scalar[100]
        getitem_1336: "f32[]" = _foreach_sub_scalar[101]
        getitem_1337: "f32[]" = _foreach_sub_scalar[102]
        getitem_1338: "f32[]" = _foreach_sub_scalar[103]
        getitem_1339: "f32[]" = _foreach_sub_scalar[104]
        getitem_1340: "f32[]" = _foreach_sub_scalar[105]
        getitem_1341: "f32[]" = _foreach_sub_scalar[106]
        getitem_1342: "f32[]" = _foreach_sub_scalar[107]
        getitem_1343: "f32[]" = _foreach_sub_scalar[108]
        getitem_1344: "f32[]" = _foreach_sub_scalar[109]
        getitem_1345: "f32[]" = _foreach_sub_scalar[110]
        getitem_1346: "f32[]" = _foreach_sub_scalar[111]
        getitem_1347: "f32[]" = _foreach_sub_scalar[112]
        getitem_1348: "f32[]" = _foreach_sub_scalar[113]
        getitem_1349: "f32[]" = _foreach_sub_scalar[114]
        getitem_1350: "f32[]" = _foreach_sub_scalar[115]
        getitem_1351: "f32[]" = _foreach_sub_scalar[116]
        getitem_1352: "f32[]" = _foreach_sub_scalar[117]
        getitem_1353: "f32[]" = _foreach_sub_scalar[118]
        getitem_1354: "f32[]" = _foreach_sub_scalar[119]
        getitem_1355: "f32[]" = _foreach_sub_scalar[120]
        getitem_1356: "f32[]" = _foreach_sub_scalar[121]
        getitem_1357: "f32[]" = _foreach_sub_scalar[122]
        getitem_1358: "f32[]" = _foreach_sub_scalar[123]
        getitem_1359: "f32[]" = _foreach_sub_scalar[124]
        getitem_1360: "f32[]" = _foreach_sub_scalar[125]
        getitem_1361: "f32[]" = _foreach_sub_scalar[126]
        getitem_1362: "f32[]" = _foreach_sub_scalar[127]
        getitem_1363: "f32[]" = _foreach_sub_scalar[128]
        getitem_1364: "f32[]" = _foreach_sub_scalar[129]
        getitem_1365: "f32[]" = _foreach_sub_scalar[130]
        getitem_1366: "f32[]" = _foreach_sub_scalar[131]
        getitem_1367: "f32[]" = _foreach_sub_scalar[132]
        getitem_1368: "f32[]" = _foreach_sub_scalar[133]
        getitem_1369: "f32[]" = _foreach_sub_scalar[134]
        getitem_1370: "f32[]" = _foreach_sub_scalar[135]
        getitem_1371: "f32[]" = _foreach_sub_scalar[136]
        getitem_1372: "f32[]" = _foreach_sub_scalar[137]
        getitem_1373: "f32[]" = _foreach_sub_scalar[138]
        getitem_1374: "f32[]" = _foreach_sub_scalar[139]
        getitem_1375: "f32[]" = _foreach_sub_scalar[140]
        getitem_1376: "f32[]" = _foreach_sub_scalar[141]
        getitem_1377: "f32[]" = _foreach_sub_scalar[142]
        getitem_1378: "f32[]" = _foreach_sub_scalar[143]
        getitem_1379: "f32[]" = _foreach_sub_scalar[144]
        getitem_1380: "f32[]" = _foreach_sub_scalar[145]
        getitem_1381: "f32[]" = _foreach_sub_scalar[146]
        getitem_1382: "f32[]" = _foreach_sub_scalar[147]
        getitem_1383: "f32[]" = _foreach_sub_scalar[148]
        getitem_1384: "f32[]" = _foreach_sub_scalar[149]
        getitem_1385: "f32[]" = _foreach_sub_scalar[150]
        getitem_1386: "f32[]" = _foreach_sub_scalar[151]
        getitem_1387: "f32[]" = _foreach_sub_scalar[152]
        getitem_1388: "f32[]" = _foreach_sub_scalar[153]
        getitem_1389: "f32[]" = _foreach_sub_scalar[154]
        getitem_1390: "f32[]" = _foreach_sub_scalar[155]
        getitem_1391: "f32[]" = _foreach_sub_scalar[156]
        getitem_1392: "f32[]" = _foreach_sub_scalar[157]
        getitem_1393: "f32[]" = _foreach_sub_scalar[158]
        getitem_1394: "f32[]" = _foreach_sub_scalar[159]
        getitem_1395: "f32[]" = _foreach_sub_scalar[160]
        getitem_1396: "f32[]" = _foreach_sub_scalar[161]
        getitem_1397: "f32[]" = _foreach_sub_scalar[162]
        getitem_1398: "f32[]" = _foreach_sub_scalar[163]
        getitem_1399: "f32[]" = _foreach_sub_scalar[164]
        getitem_1400: "f32[]" = _foreach_sub_scalar[165]
        getitem_1401: "f32[]" = _foreach_sub_scalar[166]
        getitem_1402: "f32[]" = _foreach_sub_scalar[167]
        getitem_1403: "f32[]" = _foreach_sub_scalar[168]
        getitem_1404: "f32[]" = _foreach_sub_scalar[169]
        getitem_1405: "f32[]" = _foreach_sub_scalar[170]
        getitem_1406: "f32[]" = _foreach_sub_scalar[171]
        getitem_1407: "f32[]" = _foreach_sub_scalar[172]
        getitem_1408: "f32[]" = _foreach_sub_scalar[173]
        getitem_1409: "f32[]" = _foreach_sub_scalar[174]
        getitem_1410: "f32[]" = _foreach_sub_scalar[175]
        getitem_1411: "f32[]" = _foreach_sub_scalar[176]
        getitem_1412: "f32[]" = _foreach_sub_scalar[177]
        getitem_1413: "f32[]" = _foreach_sub_scalar[178]
        getitem_1414: "f32[]" = _foreach_sub_scalar[179]
        getitem_1415: "f32[]" = _foreach_sub_scalar[180]
        getitem_1416: "f32[]" = _foreach_sub_scalar[181]
        getitem_1417: "f32[]" = _foreach_sub_scalar[182]
        getitem_1418: "f32[]" = _foreach_sub_scalar[183]
        getitem_1419: "f32[]" = _foreach_sub_scalar[184]
        getitem_1420: "f32[]" = _foreach_sub_scalar[185]
        getitem_1421: "f32[]" = _foreach_sub_scalar[186]
        getitem_1422: "f32[]" = _foreach_sub_scalar[187]
        getitem_1423: "f32[]" = _foreach_sub_scalar[188]
        getitem_1424: "f32[]" = _foreach_sub_scalar[189]
        getitem_1425: "f32[]" = _foreach_sub_scalar[190]
        getitem_1426: "f32[]" = _foreach_sub_scalar[191]
        getitem_1427: "f32[]" = _foreach_sub_scalar[192]
        getitem_1428: "f32[]" = _foreach_sub_scalar[193]
        getitem_1429: "f32[]" = _foreach_sub_scalar[194]
        getitem_1430: "f32[]" = _foreach_sub_scalar[195]
        getitem_1431: "f32[]" = _foreach_sub_scalar[196]
        getitem_1432: "f32[]" = _foreach_sub_scalar[197]
        getitem_1433: "f32[]" = _foreach_sub_scalar[198]
        getitem_1434: "f32[]" = _foreach_sub_scalar[199]
        getitem_1435: "f32[]" = _foreach_sub_scalar[200]
        getitem_1436: "f32[]" = _foreach_sub_scalar[201]
        getitem_1437: "f32[]" = _foreach_sub_scalar[202]
        getitem_1438: "f32[]" = _foreach_sub_scalar[203]
        getitem_1439: "f32[]" = _foreach_sub_scalar[204]
        getitem_1440: "f32[]" = _foreach_sub_scalar[205];  _foreach_sub_scalar = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059]);  getitem_1854 = getitem_1855 = getitem_1856 = getitem_1857 = getitem_1858 = getitem_1859 = getitem_1860 = getitem_1861 = getitem_1862 = getitem_1863 = getitem_1864 = getitem_1865 = getitem_1866 = getitem_1867 = getitem_1868 = getitem_1869 = getitem_1870 = getitem_1871 = getitem_1872 = getitem_1873 = getitem_1874 = getitem_1875 = getitem_1876 = getitem_1877 = getitem_1878 = getitem_1879 = getitem_1880 = getitem_1881 = getitem_1882 = getitem_1883 = getitem_1884 = getitem_1885 = getitem_1886 = getitem_1887 = getitem_1888 = getitem_1889 = getitem_1890 = getitem_1891 = getitem_1892 = getitem_1893 = getitem_1894 = getitem_1895 = getitem_1896 = getitem_1897 = getitem_1898 = getitem_1899 = getitem_1900 = getitem_1901 = getitem_1902 = getitem_1903 = getitem_1904 = getitem_1905 = getitem_1906 = getitem_1907 = getitem_1908 = getitem_1909 = getitem_1910 = getitem_1911 = getitem_1912 = getitem_1913 = getitem_1914 = getitem_1915 = getitem_1916 = getitem_1917 = getitem_1918 = getitem_1919 = getitem_1920 = getitem_1921 = getitem_1922 = getitem_1923 = getitem_1924 = getitem_1925 = getitem_1926 = getitem_1927 = getitem_1928 = getitem_1929 = getitem_1930 = getitem_1931 = getitem_1932 = getitem_1933 = getitem_1934 = getitem_1935 = getitem_1936 = getitem_1937 = getitem_1938 = getitem_1939 = getitem_1940 = getitem_1941 = getitem_1942 = getitem_1943 = getitem_1944 = getitem_1945 = getitem_1946 = getitem_1947 = getitem_1948 = getitem_1949 = getitem_1950 = getitem_1951 = getitem_1952 = getitem_1953 = getitem_1954 = getitem_1955 = getitem_1956 = getitem_1957 = getitem_1958 = getitem_1959 = getitem_1960 = getitem_1961 = getitem_1962 = getitem_1963 = getitem_1964 = getitem_1965 = getitem_1966 = getitem_1967 = getitem_1968 = getitem_1969 = getitem_1970 = getitem_1971 = getitem_1972 = getitem_1973 = getitem_1974 = getitem_1975 = getitem_1976 = getitem_1977 = getitem_1978 = getitem_1979 = getitem_1980 = getitem_1981 = getitem_1982 = getitem_1983 = getitem_1984 = getitem_1985 = getitem_1986 = getitem_1987 = getitem_1988 = getitem_1989 = getitem_1990 = getitem_1991 = getitem_1992 = getitem_1993 = getitem_1994 = getitem_1995 = getitem_1996 = getitem_1997 = getitem_1998 = getitem_1999 = getitem_2000 = getitem_2001 = getitem_2002 = getitem_2003 = getitem_2004 = getitem_2005 = getitem_2006 = getitem_2007 = getitem_2008 = getitem_2009 = getitem_2010 = getitem_2011 = getitem_2012 = getitem_2013 = getitem_2014 = getitem_2015 = getitem_2016 = getitem_2017 = getitem_2018 = getitem_2019 = getitem_2020 = getitem_2021 = getitem_2022 = getitem_2023 = getitem_2024 = getitem_2025 = getitem_2026 = getitem_2027 = getitem_2028 = getitem_2029 = getitem_2030 = getitem_2031 = getitem_2032 = getitem_2033 = getitem_2034 = getitem_2035 = getitem_2036 = getitem_2037 = getitem_2038 = getitem_2039 = getitem_2040 = getitem_2041 = getitem_2042 = getitem_2043 = getitem_2044 = getitem_2045 = getitem_2046 = getitem_2047 = getitem_2048 = getitem_2049 = getitem_2050 = getitem_2051 = getitem_2052 = getitem_2053 = getitem_2054 = getitem_2055 = getitem_2056 = getitem_2057 = getitem_2058 = getitem_2059 = None
        getitem_2060: "f32[]" = _foreach_sqrt_default[0]
        getitem_2061: "f32[]" = _foreach_sqrt_default[1]
        getitem_2062: "f32[]" = _foreach_sqrt_default[2]
        getitem_2063: "f32[]" = _foreach_sqrt_default[3]
        getitem_2064: "f32[]" = _foreach_sqrt_default[4]
        getitem_2065: "f32[]" = _foreach_sqrt_default[5]
        getitem_2066: "f32[]" = _foreach_sqrt_default[6]
        getitem_2067: "f32[]" = _foreach_sqrt_default[7]
        getitem_2068: "f32[]" = _foreach_sqrt_default[8]
        getitem_2069: "f32[]" = _foreach_sqrt_default[9]
        getitem_2070: "f32[]" = _foreach_sqrt_default[10]
        getitem_2071: "f32[]" = _foreach_sqrt_default[11]
        getitem_2072: "f32[]" = _foreach_sqrt_default[12]
        getitem_2073: "f32[]" = _foreach_sqrt_default[13]
        getitem_2074: "f32[]" = _foreach_sqrt_default[14]
        getitem_2075: "f32[]" = _foreach_sqrt_default[15]
        getitem_2076: "f32[]" = _foreach_sqrt_default[16]
        getitem_2077: "f32[]" = _foreach_sqrt_default[17]
        getitem_2078: "f32[]" = _foreach_sqrt_default[18]
        getitem_2079: "f32[]" = _foreach_sqrt_default[19]
        getitem_2080: "f32[]" = _foreach_sqrt_default[20]
        getitem_2081: "f32[]" = _foreach_sqrt_default[21]
        getitem_2082: "f32[]" = _foreach_sqrt_default[22]
        getitem_2083: "f32[]" = _foreach_sqrt_default[23]
        getitem_2084: "f32[]" = _foreach_sqrt_default[24]
        getitem_2085: "f32[]" = _foreach_sqrt_default[25]
        getitem_2086: "f32[]" = _foreach_sqrt_default[26]
        getitem_2087: "f32[]" = _foreach_sqrt_default[27]
        getitem_2088: "f32[]" = _foreach_sqrt_default[28]
        getitem_2089: "f32[]" = _foreach_sqrt_default[29]
        getitem_2090: "f32[]" = _foreach_sqrt_default[30]
        getitem_2091: "f32[]" = _foreach_sqrt_default[31]
        getitem_2092: "f32[]" = _foreach_sqrt_default[32]
        getitem_2093: "f32[]" = _foreach_sqrt_default[33]
        getitem_2094: "f32[]" = _foreach_sqrt_default[34]
        getitem_2095: "f32[]" = _foreach_sqrt_default[35]
        getitem_2096: "f32[]" = _foreach_sqrt_default[36]
        getitem_2097: "f32[]" = _foreach_sqrt_default[37]
        getitem_2098: "f32[]" = _foreach_sqrt_default[38]
        getitem_2099: "f32[]" = _foreach_sqrt_default[39]
        getitem_2100: "f32[]" = _foreach_sqrt_default[40]
        getitem_2101: "f32[]" = _foreach_sqrt_default[41]
        getitem_2102: "f32[]" = _foreach_sqrt_default[42]
        getitem_2103: "f32[]" = _foreach_sqrt_default[43]
        getitem_2104: "f32[]" = _foreach_sqrt_default[44]
        getitem_2105: "f32[]" = _foreach_sqrt_default[45]
        getitem_2106: "f32[]" = _foreach_sqrt_default[46]
        getitem_2107: "f32[]" = _foreach_sqrt_default[47]
        getitem_2108: "f32[]" = _foreach_sqrt_default[48]
        getitem_2109: "f32[]" = _foreach_sqrt_default[49]
        getitem_2110: "f32[]" = _foreach_sqrt_default[50]
        getitem_2111: "f32[]" = _foreach_sqrt_default[51]
        getitem_2112: "f32[]" = _foreach_sqrt_default[52]
        getitem_2113: "f32[]" = _foreach_sqrt_default[53]
        getitem_2114: "f32[]" = _foreach_sqrt_default[54]
        getitem_2115: "f32[]" = _foreach_sqrt_default[55]
        getitem_2116: "f32[]" = _foreach_sqrt_default[56]
        getitem_2117: "f32[]" = _foreach_sqrt_default[57]
        getitem_2118: "f32[]" = _foreach_sqrt_default[58]
        getitem_2119: "f32[]" = _foreach_sqrt_default[59]
        getitem_2120: "f32[]" = _foreach_sqrt_default[60]
        getitem_2121: "f32[]" = _foreach_sqrt_default[61]
        getitem_2122: "f32[]" = _foreach_sqrt_default[62]
        getitem_2123: "f32[]" = _foreach_sqrt_default[63]
        getitem_2124: "f32[]" = _foreach_sqrt_default[64]
        getitem_2125: "f32[]" = _foreach_sqrt_default[65]
        getitem_2126: "f32[]" = _foreach_sqrt_default[66]
        getitem_2127: "f32[]" = _foreach_sqrt_default[67]
        getitem_2128: "f32[]" = _foreach_sqrt_default[68]
        getitem_2129: "f32[]" = _foreach_sqrt_default[69]
        getitem_2130: "f32[]" = _foreach_sqrt_default[70]
        getitem_2131: "f32[]" = _foreach_sqrt_default[71]
        getitem_2132: "f32[]" = _foreach_sqrt_default[72]
        getitem_2133: "f32[]" = _foreach_sqrt_default[73]
        getitem_2134: "f32[]" = _foreach_sqrt_default[74]
        getitem_2135: "f32[]" = _foreach_sqrt_default[75]
        getitem_2136: "f32[]" = _foreach_sqrt_default[76]
        getitem_2137: "f32[]" = _foreach_sqrt_default[77]
        getitem_2138: "f32[]" = _foreach_sqrt_default[78]
        getitem_2139: "f32[]" = _foreach_sqrt_default[79]
        getitem_2140: "f32[]" = _foreach_sqrt_default[80]
        getitem_2141: "f32[]" = _foreach_sqrt_default[81]
        getitem_2142: "f32[]" = _foreach_sqrt_default[82]
        getitem_2143: "f32[]" = _foreach_sqrt_default[83]
        getitem_2144: "f32[]" = _foreach_sqrt_default[84]
        getitem_2145: "f32[]" = _foreach_sqrt_default[85]
        getitem_2146: "f32[]" = _foreach_sqrt_default[86]
        getitem_2147: "f32[]" = _foreach_sqrt_default[87]
        getitem_2148: "f32[]" = _foreach_sqrt_default[88]
        getitem_2149: "f32[]" = _foreach_sqrt_default[89]
        getitem_2150: "f32[]" = _foreach_sqrt_default[90]
        getitem_2151: "f32[]" = _foreach_sqrt_default[91]
        getitem_2152: "f32[]" = _foreach_sqrt_default[92]
        getitem_2153: "f32[]" = _foreach_sqrt_default[93]
        getitem_2154: "f32[]" = _foreach_sqrt_default[94]
        getitem_2155: "f32[]" = _foreach_sqrt_default[95]
        getitem_2156: "f32[]" = _foreach_sqrt_default[96]
        getitem_2157: "f32[]" = _foreach_sqrt_default[97]
        getitem_2158: "f32[]" = _foreach_sqrt_default[98]
        getitem_2159: "f32[]" = _foreach_sqrt_default[99]
        getitem_2160: "f32[]" = _foreach_sqrt_default[100]
        getitem_2161: "f32[]" = _foreach_sqrt_default[101]
        getitem_2162: "f32[]" = _foreach_sqrt_default[102]
        getitem_2163: "f32[]" = _foreach_sqrt_default[103]
        getitem_2164: "f32[]" = _foreach_sqrt_default[104]
        getitem_2165: "f32[]" = _foreach_sqrt_default[105]
        getitem_2166: "f32[]" = _foreach_sqrt_default[106]
        getitem_2167: "f32[]" = _foreach_sqrt_default[107]
        getitem_2168: "f32[]" = _foreach_sqrt_default[108]
        getitem_2169: "f32[]" = _foreach_sqrt_default[109]
        getitem_2170: "f32[]" = _foreach_sqrt_default[110]
        getitem_2171: "f32[]" = _foreach_sqrt_default[111]
        getitem_2172: "f32[]" = _foreach_sqrt_default[112]
        getitem_2173: "f32[]" = _foreach_sqrt_default[113]
        getitem_2174: "f32[]" = _foreach_sqrt_default[114]
        getitem_2175: "f32[]" = _foreach_sqrt_default[115]
        getitem_2176: "f32[]" = _foreach_sqrt_default[116]
        getitem_2177: "f32[]" = _foreach_sqrt_default[117]
        getitem_2178: "f32[]" = _foreach_sqrt_default[118]
        getitem_2179: "f32[]" = _foreach_sqrt_default[119]
        getitem_2180: "f32[]" = _foreach_sqrt_default[120]
        getitem_2181: "f32[]" = _foreach_sqrt_default[121]
        getitem_2182: "f32[]" = _foreach_sqrt_default[122]
        getitem_2183: "f32[]" = _foreach_sqrt_default[123]
        getitem_2184: "f32[]" = _foreach_sqrt_default[124]
        getitem_2185: "f32[]" = _foreach_sqrt_default[125]
        getitem_2186: "f32[]" = _foreach_sqrt_default[126]
        getitem_2187: "f32[]" = _foreach_sqrt_default[127]
        getitem_2188: "f32[]" = _foreach_sqrt_default[128]
        getitem_2189: "f32[]" = _foreach_sqrt_default[129]
        getitem_2190: "f32[]" = _foreach_sqrt_default[130]
        getitem_2191: "f32[]" = _foreach_sqrt_default[131]
        getitem_2192: "f32[]" = _foreach_sqrt_default[132]
        getitem_2193: "f32[]" = _foreach_sqrt_default[133]
        getitem_2194: "f32[]" = _foreach_sqrt_default[134]
        getitem_2195: "f32[]" = _foreach_sqrt_default[135]
        getitem_2196: "f32[]" = _foreach_sqrt_default[136]
        getitem_2197: "f32[]" = _foreach_sqrt_default[137]
        getitem_2198: "f32[]" = _foreach_sqrt_default[138]
        getitem_2199: "f32[]" = _foreach_sqrt_default[139]
        getitem_2200: "f32[]" = _foreach_sqrt_default[140]
        getitem_2201: "f32[]" = _foreach_sqrt_default[141]
        getitem_2202: "f32[]" = _foreach_sqrt_default[142]
        getitem_2203: "f32[]" = _foreach_sqrt_default[143]
        getitem_2204: "f32[]" = _foreach_sqrt_default[144]
        getitem_2205: "f32[]" = _foreach_sqrt_default[145]
        getitem_2206: "f32[]" = _foreach_sqrt_default[146]
        getitem_2207: "f32[]" = _foreach_sqrt_default[147]
        getitem_2208: "f32[]" = _foreach_sqrt_default[148]
        getitem_2209: "f32[]" = _foreach_sqrt_default[149]
        getitem_2210: "f32[]" = _foreach_sqrt_default[150]
        getitem_2211: "f32[]" = _foreach_sqrt_default[151]
        getitem_2212: "f32[]" = _foreach_sqrt_default[152]
        getitem_2213: "f32[]" = _foreach_sqrt_default[153]
        getitem_2214: "f32[]" = _foreach_sqrt_default[154]
        getitem_2215: "f32[]" = _foreach_sqrt_default[155]
        getitem_2216: "f32[]" = _foreach_sqrt_default[156]
        getitem_2217: "f32[]" = _foreach_sqrt_default[157]
        getitem_2218: "f32[]" = _foreach_sqrt_default[158]
        getitem_2219: "f32[]" = _foreach_sqrt_default[159]
        getitem_2220: "f32[]" = _foreach_sqrt_default[160]
        getitem_2221: "f32[]" = _foreach_sqrt_default[161]
        getitem_2222: "f32[]" = _foreach_sqrt_default[162]
        getitem_2223: "f32[]" = _foreach_sqrt_default[163]
        getitem_2224: "f32[]" = _foreach_sqrt_default[164]
        getitem_2225: "f32[]" = _foreach_sqrt_default[165]
        getitem_2226: "f32[]" = _foreach_sqrt_default[166]
        getitem_2227: "f32[]" = _foreach_sqrt_default[167]
        getitem_2228: "f32[]" = _foreach_sqrt_default[168]
        getitem_2229: "f32[]" = _foreach_sqrt_default[169]
        getitem_2230: "f32[]" = _foreach_sqrt_default[170]
        getitem_2231: "f32[]" = _foreach_sqrt_default[171]
        getitem_2232: "f32[]" = _foreach_sqrt_default[172]
        getitem_2233: "f32[]" = _foreach_sqrt_default[173]
        getitem_2234: "f32[]" = _foreach_sqrt_default[174]
        getitem_2235: "f32[]" = _foreach_sqrt_default[175]
        getitem_2236: "f32[]" = _foreach_sqrt_default[176]
        getitem_2237: "f32[]" = _foreach_sqrt_default[177]
        getitem_2238: "f32[]" = _foreach_sqrt_default[178]
        getitem_2239: "f32[]" = _foreach_sqrt_default[179]
        getitem_2240: "f32[]" = _foreach_sqrt_default[180]
        getitem_2241: "f32[]" = _foreach_sqrt_default[181]
        getitem_2242: "f32[]" = _foreach_sqrt_default[182]
        getitem_2243: "f32[]" = _foreach_sqrt_default[183]
        getitem_2244: "f32[]" = _foreach_sqrt_default[184]
        getitem_2245: "f32[]" = _foreach_sqrt_default[185]
        getitem_2246: "f32[]" = _foreach_sqrt_default[186]
        getitem_2247: "f32[]" = _foreach_sqrt_default[187]
        getitem_2248: "f32[]" = _foreach_sqrt_default[188]
        getitem_2249: "f32[]" = _foreach_sqrt_default[189]
        getitem_2250: "f32[]" = _foreach_sqrt_default[190]
        getitem_2251: "f32[]" = _foreach_sqrt_default[191]
        getitem_2252: "f32[]" = _foreach_sqrt_default[192]
        getitem_2253: "f32[]" = _foreach_sqrt_default[193]
        getitem_2254: "f32[]" = _foreach_sqrt_default[194]
        getitem_2255: "f32[]" = _foreach_sqrt_default[195]
        getitem_2256: "f32[]" = _foreach_sqrt_default[196]
        getitem_2257: "f32[]" = _foreach_sqrt_default[197]
        getitem_2258: "f32[]" = _foreach_sqrt_default[198]
        getitem_2259: "f32[]" = _foreach_sqrt_default[199]
        getitem_2260: "f32[]" = _foreach_sqrt_default[200]
        getitem_2261: "f32[]" = _foreach_sqrt_default[201]
        getitem_2262: "f32[]" = _foreach_sqrt_default[202]
        getitem_2263: "f32[]" = _foreach_sqrt_default[203]
        getitem_2264: "f32[]" = _foreach_sqrt_default[204]
        getitem_2265: "f32[]" = _foreach_sqrt_default[205];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029]);  getitem_824 = getitem_825 = getitem_826 = getitem_827 = getitem_828 = getitem_829 = getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = getitem_835 = getitem_836 = getitem_837 = getitem_838 = getitem_839 = getitem_840 = getitem_841 = getitem_842 = getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = getitem_849 = getitem_850 = getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = getitem_857 = getitem_858 = getitem_859 = getitem_860 = getitem_861 = getitem_862 = getitem_863 = getitem_864 = getitem_865 = getitem_866 = getitem_867 = getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = getitem_873 = getitem_874 = getitem_875 = getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = getitem_900 = getitem_901 = getitem_902 = getitem_903 = getitem_904 = getitem_905 = getitem_906 = getitem_907 = getitem_908 = getitem_909 = getitem_910 = getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = getitem_917 = getitem_918 = getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = getitem_925 = getitem_926 = getitem_927 = getitem_928 = getitem_929 = getitem_930 = getitem_931 = getitem_932 = getitem_933 = getitem_934 = getitem_935 = getitem_936 = getitem_937 = getitem_938 = getitem_939 = getitem_940 = getitem_941 = getitem_942 = getitem_943 = getitem_944 = getitem_945 = getitem_946 = getitem_947 = getitem_948 = getitem_949 = getitem_950 = getitem_951 = getitem_952 = getitem_953 = getitem_954 = getitem_955 = getitem_956 = getitem_957 = getitem_958 = getitem_959 = getitem_960 = getitem_961 = getitem_962 = getitem_963 = getitem_964 = getitem_965 = getitem_966 = getitem_967 = getitem_968 = getitem_969 = getitem_970 = getitem_971 = getitem_972 = getitem_973 = getitem_974 = getitem_975 = getitem_976 = getitem_977 = getitem_978 = getitem_979 = getitem_980 = getitem_981 = getitem_982 = getitem_983 = getitem_984 = getitem_985 = getitem_986 = getitem_987 = getitem_988 = getitem_989 = getitem_990 = getitem_991 = getitem_992 = getitem_993 = getitem_994 = getitem_995 = getitem_996 = getitem_997 = getitem_998 = getitem_999 = getitem_1000 = getitem_1001 = getitem_1002 = getitem_1003 = getitem_1004 = getitem_1005 = getitem_1006 = getitem_1007 = getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = getitem_1013 = getitem_1014 = getitem_1015 = getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = getitem_1023 = getitem_1024 = getitem_1025 = getitem_1026 = getitem_1027 = getitem_1028 = getitem_1029 = None
        getitem_1441: "f32[30522, 768]" = _foreach_sqrt_default_1[0]
        getitem_1442: "f32[512, 768]" = _foreach_sqrt_default_1[1]
        getitem_1443: "f32[1024, 768]" = _foreach_sqrt_default_1[2]
        getitem_1444: "f32[1024, 768]" = _foreach_sqrt_default_1[3]
        getitem_1445: "f32[1024, 768]" = _foreach_sqrt_default_1[4]
        getitem_1446: "f32[1024, 768]" = _foreach_sqrt_default_1[5]
        getitem_1447: "f32[2, 768]" = _foreach_sqrt_default_1[6]
        getitem_1448: "f32[768]" = _foreach_sqrt_default_1[7]
        getitem_1449: "f32[768]" = _foreach_sqrt_default_1[8]
        getitem_1450: "f32[768, 768]" = _foreach_sqrt_default_1[9]
        getitem_1451: "f32[768]" = _foreach_sqrt_default_1[10]
        getitem_1452: "f32[768, 768]" = _foreach_sqrt_default_1[11]
        getitem_1453: "f32[768]" = _foreach_sqrt_default_1[12]
        getitem_1454: "f32[768, 768]" = _foreach_sqrt_default_1[13]
        getitem_1455: "f32[768]" = _foreach_sqrt_default_1[14]
        getitem_1456: "f32[768, 768]" = _foreach_sqrt_default_1[15]
        getitem_1457: "f32[768]" = _foreach_sqrt_default_1[16]
        getitem_1458: "f32[768]" = _foreach_sqrt_default_1[17]
        getitem_1459: "f32[768]" = _foreach_sqrt_default_1[18]
        getitem_1460: "f32[3072, 768]" = _foreach_sqrt_default_1[19]
        getitem_1461: "f32[3072]" = _foreach_sqrt_default_1[20]
        getitem_1462: "f32[768, 3072]" = _foreach_sqrt_default_1[21]
        getitem_1463: "f32[768]" = _foreach_sqrt_default_1[22]
        getitem_1464: "f32[768]" = _foreach_sqrt_default_1[23]
        getitem_1465: "f32[768]" = _foreach_sqrt_default_1[24]
        getitem_1466: "f32[768, 768]" = _foreach_sqrt_default_1[25]
        getitem_1467: "f32[768]" = _foreach_sqrt_default_1[26]
        getitem_1468: "f32[768, 768]" = _foreach_sqrt_default_1[27]
        getitem_1469: "f32[768]" = _foreach_sqrt_default_1[28]
        getitem_1470: "f32[768, 768]" = _foreach_sqrt_default_1[29]
        getitem_1471: "f32[768]" = _foreach_sqrt_default_1[30]
        getitem_1472: "f32[768, 768]" = _foreach_sqrt_default_1[31]
        getitem_1473: "f32[768]" = _foreach_sqrt_default_1[32]
        getitem_1474: "f32[768]" = _foreach_sqrt_default_1[33]
        getitem_1475: "f32[768]" = _foreach_sqrt_default_1[34]
        getitem_1476: "f32[3072, 768]" = _foreach_sqrt_default_1[35]
        getitem_1477: "f32[3072]" = _foreach_sqrt_default_1[36]
        getitem_1478: "f32[768, 3072]" = _foreach_sqrt_default_1[37]
        getitem_1479: "f32[768]" = _foreach_sqrt_default_1[38]
        getitem_1480: "f32[768]" = _foreach_sqrt_default_1[39]
        getitem_1481: "f32[768]" = _foreach_sqrt_default_1[40]
        getitem_1482: "f32[768, 768]" = _foreach_sqrt_default_1[41]
        getitem_1483: "f32[768]" = _foreach_sqrt_default_1[42]
        getitem_1484: "f32[768, 768]" = _foreach_sqrt_default_1[43]
        getitem_1485: "f32[768]" = _foreach_sqrt_default_1[44]
        getitem_1486: "f32[768, 768]" = _foreach_sqrt_default_1[45]
        getitem_1487: "f32[768]" = _foreach_sqrt_default_1[46]
        getitem_1488: "f32[768, 768]" = _foreach_sqrt_default_1[47]
        getitem_1489: "f32[768]" = _foreach_sqrt_default_1[48]
        getitem_1490: "f32[768]" = _foreach_sqrt_default_1[49]
        getitem_1491: "f32[768]" = _foreach_sqrt_default_1[50]
        getitem_1492: "f32[3072, 768]" = _foreach_sqrt_default_1[51]
        getitem_1493: "f32[3072]" = _foreach_sqrt_default_1[52]
        getitem_1494: "f32[768, 3072]" = _foreach_sqrt_default_1[53]
        getitem_1495: "f32[768]" = _foreach_sqrt_default_1[54]
        getitem_1496: "f32[768]" = _foreach_sqrt_default_1[55]
        getitem_1497: "f32[768]" = _foreach_sqrt_default_1[56]
        getitem_1498: "f32[768, 768]" = _foreach_sqrt_default_1[57]
        getitem_1499: "f32[768]" = _foreach_sqrt_default_1[58]
        getitem_1500: "f32[768, 768]" = _foreach_sqrt_default_1[59]
        getitem_1501: "f32[768]" = _foreach_sqrt_default_1[60]
        getitem_1502: "f32[768, 768]" = _foreach_sqrt_default_1[61]
        getitem_1503: "f32[768]" = _foreach_sqrt_default_1[62]
        getitem_1504: "f32[768, 768]" = _foreach_sqrt_default_1[63]
        getitem_1505: "f32[768]" = _foreach_sqrt_default_1[64]
        getitem_1506: "f32[768]" = _foreach_sqrt_default_1[65]
        getitem_1507: "f32[768]" = _foreach_sqrt_default_1[66]
        getitem_1508: "f32[3072, 768]" = _foreach_sqrt_default_1[67]
        getitem_1509: "f32[3072]" = _foreach_sqrt_default_1[68]
        getitem_1510: "f32[768, 3072]" = _foreach_sqrt_default_1[69]
        getitem_1511: "f32[768]" = _foreach_sqrt_default_1[70]
        getitem_1512: "f32[768]" = _foreach_sqrt_default_1[71]
        getitem_1513: "f32[768]" = _foreach_sqrt_default_1[72]
        getitem_1514: "f32[768, 768]" = _foreach_sqrt_default_1[73]
        getitem_1515: "f32[768]" = _foreach_sqrt_default_1[74]
        getitem_1516: "f32[768, 768]" = _foreach_sqrt_default_1[75]
        getitem_1517: "f32[768]" = _foreach_sqrt_default_1[76]
        getitem_1518: "f32[768, 768]" = _foreach_sqrt_default_1[77]
        getitem_1519: "f32[768]" = _foreach_sqrt_default_1[78]
        getitem_1520: "f32[768, 768]" = _foreach_sqrt_default_1[79]
        getitem_1521: "f32[768]" = _foreach_sqrt_default_1[80]
        getitem_1522: "f32[768]" = _foreach_sqrt_default_1[81]
        getitem_1523: "f32[768]" = _foreach_sqrt_default_1[82]
        getitem_1524: "f32[3072, 768]" = _foreach_sqrt_default_1[83]
        getitem_1525: "f32[3072]" = _foreach_sqrt_default_1[84]
        getitem_1526: "f32[768, 3072]" = _foreach_sqrt_default_1[85]
        getitem_1527: "f32[768]" = _foreach_sqrt_default_1[86]
        getitem_1528: "f32[768]" = _foreach_sqrt_default_1[87]
        getitem_1529: "f32[768]" = _foreach_sqrt_default_1[88]
        getitem_1530: "f32[768, 768]" = _foreach_sqrt_default_1[89]
        getitem_1531: "f32[768]" = _foreach_sqrt_default_1[90]
        getitem_1532: "f32[768, 768]" = _foreach_sqrt_default_1[91]
        getitem_1533: "f32[768]" = _foreach_sqrt_default_1[92]
        getitem_1534: "f32[768, 768]" = _foreach_sqrt_default_1[93]
        getitem_1535: "f32[768]" = _foreach_sqrt_default_1[94]
        getitem_1536: "f32[768, 768]" = _foreach_sqrt_default_1[95]
        getitem_1537: "f32[768]" = _foreach_sqrt_default_1[96]
        getitem_1538: "f32[768]" = _foreach_sqrt_default_1[97]
        getitem_1539: "f32[768]" = _foreach_sqrt_default_1[98]
        getitem_1540: "f32[3072, 768]" = _foreach_sqrt_default_1[99]
        getitem_1541: "f32[3072]" = _foreach_sqrt_default_1[100]
        getitem_1542: "f32[768, 3072]" = _foreach_sqrt_default_1[101]
        getitem_1543: "f32[768]" = _foreach_sqrt_default_1[102]
        getitem_1544: "f32[768]" = _foreach_sqrt_default_1[103]
        getitem_1545: "f32[768]" = _foreach_sqrt_default_1[104]
        getitem_1546: "f32[768, 768]" = _foreach_sqrt_default_1[105]
        getitem_1547: "f32[768]" = _foreach_sqrt_default_1[106]
        getitem_1548: "f32[768, 768]" = _foreach_sqrt_default_1[107]
        getitem_1549: "f32[768]" = _foreach_sqrt_default_1[108]
        getitem_1550: "f32[768, 768]" = _foreach_sqrt_default_1[109]
        getitem_1551: "f32[768]" = _foreach_sqrt_default_1[110]
        getitem_1552: "f32[768, 768]" = _foreach_sqrt_default_1[111]
        getitem_1553: "f32[768]" = _foreach_sqrt_default_1[112]
        getitem_1554: "f32[768]" = _foreach_sqrt_default_1[113]
        getitem_1555: "f32[768]" = _foreach_sqrt_default_1[114]
        getitem_1556: "f32[3072, 768]" = _foreach_sqrt_default_1[115]
        getitem_1557: "f32[3072]" = _foreach_sqrt_default_1[116]
        getitem_1558: "f32[768, 3072]" = _foreach_sqrt_default_1[117]
        getitem_1559: "f32[768]" = _foreach_sqrt_default_1[118]
        getitem_1560: "f32[768]" = _foreach_sqrt_default_1[119]
        getitem_1561: "f32[768]" = _foreach_sqrt_default_1[120]
        getitem_1562: "f32[768, 768]" = _foreach_sqrt_default_1[121]
        getitem_1563: "f32[768]" = _foreach_sqrt_default_1[122]
        getitem_1564: "f32[768, 768]" = _foreach_sqrt_default_1[123]
        getitem_1565: "f32[768]" = _foreach_sqrt_default_1[124]
        getitem_1566: "f32[768, 768]" = _foreach_sqrt_default_1[125]
        getitem_1567: "f32[768]" = _foreach_sqrt_default_1[126]
        getitem_1568: "f32[768, 768]" = _foreach_sqrt_default_1[127]
        getitem_1569: "f32[768]" = _foreach_sqrt_default_1[128]
        getitem_1570: "f32[768]" = _foreach_sqrt_default_1[129]
        getitem_1571: "f32[768]" = _foreach_sqrt_default_1[130]
        getitem_1572: "f32[3072, 768]" = _foreach_sqrt_default_1[131]
        getitem_1573: "f32[3072]" = _foreach_sqrt_default_1[132]
        getitem_1574: "f32[768, 3072]" = _foreach_sqrt_default_1[133]
        getitem_1575: "f32[768]" = _foreach_sqrt_default_1[134]
        getitem_1576: "f32[768]" = _foreach_sqrt_default_1[135]
        getitem_1577: "f32[768]" = _foreach_sqrt_default_1[136]
        getitem_1578: "f32[768, 768]" = _foreach_sqrt_default_1[137]
        getitem_1579: "f32[768]" = _foreach_sqrt_default_1[138]
        getitem_1580: "f32[768, 768]" = _foreach_sqrt_default_1[139]
        getitem_1581: "f32[768]" = _foreach_sqrt_default_1[140]
        getitem_1582: "f32[768, 768]" = _foreach_sqrt_default_1[141]
        getitem_1583: "f32[768]" = _foreach_sqrt_default_1[142]
        getitem_1584: "f32[768, 768]" = _foreach_sqrt_default_1[143]
        getitem_1585: "f32[768]" = _foreach_sqrt_default_1[144]
        getitem_1586: "f32[768]" = _foreach_sqrt_default_1[145]
        getitem_1587: "f32[768]" = _foreach_sqrt_default_1[146]
        getitem_1588: "f32[3072, 768]" = _foreach_sqrt_default_1[147]
        getitem_1589: "f32[3072]" = _foreach_sqrt_default_1[148]
        getitem_1590: "f32[768, 3072]" = _foreach_sqrt_default_1[149]
        getitem_1591: "f32[768]" = _foreach_sqrt_default_1[150]
        getitem_1592: "f32[768]" = _foreach_sqrt_default_1[151]
        getitem_1593: "f32[768]" = _foreach_sqrt_default_1[152]
        getitem_1594: "f32[768, 768]" = _foreach_sqrt_default_1[153]
        getitem_1595: "f32[768]" = _foreach_sqrt_default_1[154]
        getitem_1596: "f32[768, 768]" = _foreach_sqrt_default_1[155]
        getitem_1597: "f32[768]" = _foreach_sqrt_default_1[156]
        getitem_1598: "f32[768, 768]" = _foreach_sqrt_default_1[157]
        getitem_1599: "f32[768]" = _foreach_sqrt_default_1[158]
        getitem_1600: "f32[768, 768]" = _foreach_sqrt_default_1[159]
        getitem_1601: "f32[768]" = _foreach_sqrt_default_1[160]
        getitem_1602: "f32[768]" = _foreach_sqrt_default_1[161]
        getitem_1603: "f32[768]" = _foreach_sqrt_default_1[162]
        getitem_1604: "f32[3072, 768]" = _foreach_sqrt_default_1[163]
        getitem_1605: "f32[3072]" = _foreach_sqrt_default_1[164]
        getitem_1606: "f32[768, 3072]" = _foreach_sqrt_default_1[165]
        getitem_1607: "f32[768]" = _foreach_sqrt_default_1[166]
        getitem_1608: "f32[768]" = _foreach_sqrt_default_1[167]
        getitem_1609: "f32[768]" = _foreach_sqrt_default_1[168]
        getitem_1610: "f32[768, 768]" = _foreach_sqrt_default_1[169]
        getitem_1611: "f32[768]" = _foreach_sqrt_default_1[170]
        getitem_1612: "f32[768, 768]" = _foreach_sqrt_default_1[171]
        getitem_1613: "f32[768]" = _foreach_sqrt_default_1[172]
        getitem_1614: "f32[768, 768]" = _foreach_sqrt_default_1[173]
        getitem_1615: "f32[768]" = _foreach_sqrt_default_1[174]
        getitem_1616: "f32[768, 768]" = _foreach_sqrt_default_1[175]
        getitem_1617: "f32[768]" = _foreach_sqrt_default_1[176]
        getitem_1618: "f32[768]" = _foreach_sqrt_default_1[177]
        getitem_1619: "f32[768]" = _foreach_sqrt_default_1[178]
        getitem_1620: "f32[3072, 768]" = _foreach_sqrt_default_1[179]
        getitem_1621: "f32[3072]" = _foreach_sqrt_default_1[180]
        getitem_1622: "f32[768, 3072]" = _foreach_sqrt_default_1[181]
        getitem_1623: "f32[768]" = _foreach_sqrt_default_1[182]
        getitem_1624: "f32[768]" = _foreach_sqrt_default_1[183]
        getitem_1625: "f32[768]" = _foreach_sqrt_default_1[184]
        getitem_1626: "f32[768, 768]" = _foreach_sqrt_default_1[185]
        getitem_1627: "f32[768]" = _foreach_sqrt_default_1[186]
        getitem_1628: "f32[768, 768]" = _foreach_sqrt_default_1[187]
        getitem_1629: "f32[768]" = _foreach_sqrt_default_1[188]
        getitem_1630: "f32[768, 768]" = _foreach_sqrt_default_1[189]
        getitem_1631: "f32[768]" = _foreach_sqrt_default_1[190]
        getitem_1632: "f32[768, 768]" = _foreach_sqrt_default_1[191]
        getitem_1633: "f32[768]" = _foreach_sqrt_default_1[192]
        getitem_1634: "f32[768]" = _foreach_sqrt_default_1[193]
        getitem_1635: "f32[768]" = _foreach_sqrt_default_1[194]
        getitem_1636: "f32[3072, 768]" = _foreach_sqrt_default_1[195]
        getitem_1637: "f32[3072]" = _foreach_sqrt_default_1[196]
        getitem_1638: "f32[768, 3072]" = _foreach_sqrt_default_1[197]
        getitem_1639: "f32[768]" = _foreach_sqrt_default_1[198]
        getitem_1640: "f32[768]" = _foreach_sqrt_default_1[199]
        getitem_1641: "f32[768]" = _foreach_sqrt_default_1[200]
        getitem_1642: "f32[30522]" = _foreach_sqrt_default_1[201]
        getitem_1643: "f32[768, 768]" = _foreach_sqrt_default_1[202]
        getitem_1644: "f32[768]" = _foreach_sqrt_default_1[203]
        getitem_1645: "f32[768]" = _foreach_sqrt_default_1[204]
        getitem_1646: "f32[768]" = _foreach_sqrt_default_1[205];  _foreach_sqrt_default_1 = None
        return (getitem, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403, getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511, getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619, getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([30522, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([1024, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([30522], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

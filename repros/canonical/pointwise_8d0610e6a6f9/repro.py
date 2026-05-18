"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g77
Pattern hash: 8d0610e6a6f9
Shape hash: 86dd8c2e
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
    def forward(self, getitem_1085: "f32[]", getitem_1086: "f32[]", getitem_1087: "f32[]", getitem_1088: "f32[]", getitem_1089: "f32[]", getitem_1090: "f32[]", getitem_1091: "f32[]", getitem_1092: "f32[]", getitem_1093: "f32[]", getitem_1094: "f32[]", getitem_1095: "f32[]", getitem_1096: "f32[]", getitem_1097: "f32[]", getitem_1098: "f32[]", getitem_1099: "f32[]", getitem_1100: "f32[]", getitem_1101: "f32[]", getitem_1102: "f32[]", getitem_1103: "f32[]", getitem_1104: "f32[]", getitem_1105: "f32[]", getitem_1106: "f32[]", getitem_1107: "f32[]", getitem_1108: "f32[]", getitem_1109: "f32[]", getitem_1110: "f32[]", getitem_1111: "f32[]", getitem_1112: "f32[]", getitem_1113: "f32[]", getitem_1114: "f32[]", getitem_1115: "f32[]", getitem_1116: "f32[]", getitem_1117: "f32[]", getitem_1118: "f32[]", getitem_1119: "f32[]", getitem_1120: "f32[]", getitem_1121: "f32[]", getitem_1122: "f32[]", getitem_1123: "f32[]", getitem_1124: "f32[]", getitem_1125: "f32[]", getitem_1126: "f32[]", getitem_1127: "f32[]", getitem_1128: "f32[]", getitem_1129: "f32[]", getitem_1130: "f32[]", getitem_1131: "f32[]", getitem_1132: "f32[]", getitem_1133: "f32[]", getitem_1134: "f32[]", getitem_1135: "f32[]", getitem_1136: "f32[]", getitem_1137: "f32[]", getitem_1138: "f32[]", getitem_1139: "f32[]", getitem_1140: "f32[]", getitem_1141: "f32[]", getitem_1142: "f32[]", getitem_1143: "f32[]", getitem_1144: "f32[]", getitem_1145: "f32[]", getitem_1146: "f32[]", getitem_1147: "f32[]", getitem_1148: "f32[]", getitem_1149: "f32[]", getitem_1150: "f32[]", getitem_1151: "f32[]", getitem_1152: "f32[]", getitem_1153: "f32[]", getitem_1154: "f32[]", getitem_1155: "f32[]", getitem_1156: "f32[]", getitem_1157: "f32[]", getitem_1158: "f32[]", getitem_1159: "f32[]", getitem_1160: "f32[]", getitem_1161: "f32[]", getitem_1162: "f32[]", getitem_1163: "f32[]", getitem_1164: "f32[]", getitem_1165: "f32[]", getitem_1166: "f32[]", getitem_1167: "f32[]", getitem_1168: "f32[]", getitem_1169: "f32[]", getitem_1170: "f32[]", getitem_1171: "f32[]", getitem_1172: "f32[]", getitem_1173: "f32[]", getitem_1174: "f32[]", getitem_1175: "f32[]", getitem_1176: "f32[]", getitem_1177: "f32[]", getitem_1178: "f32[]", getitem_1179: "f32[]", getitem_1180: "f32[]", getitem_1181: "f32[]", getitem_1182: "f32[]", getitem_1183: "f32[]", getitem_1184: "f32[]", getitem_1185: "f32[]", getitem_1186: "f32[]", getitem_1187: "f32[]", getitem_1188: "f32[]", getitem_1189: "f32[]", getitem_1190: "f32[]", getitem_1191: "f32[]", getitem_1192: "f32[]", getitem_1193: "f32[]", getitem_1194: "f32[]", getitem_1195: "f32[]", getitem_1196: "f32[]", getitem_1197: "f32[]", getitem_1198: "f32[]", getitem_1199: "f32[]", getitem_1200: "f32[]", getitem_1201: "f32[]", getitem_1202: "f32[]", getitem_1203: "f32[]", getitem_1204: "f32[]", getitem_1205: "f32[]", getitem_1206: "f32[]", getitem_1207: "f32[]", getitem_1208: "f32[]", getitem_1209: "f32[]", getitem_1210: "f32[]", getitem_1211: "f32[]", getitem_1212: "f32[]", getitem_1213: "f32[]", getitem_1214: "f32[]", getitem_1215: "f32[]", getitem_1216: "f32[]", getitem_1217: "f32[]", getitem_1218: "f32[]", getitem_1219: "f32[]", getitem_1220: "f32[]", getitem_1221: "f32[]", getitem_1222: "f32[]", getitem_1223: "f32[]", getitem_1224: "f32[]", getitem_1225: "f32[]", getitem_1226: "f32[]", getitem_1227: "f32[]", getitem_1228: "f32[]", getitem_1229: "f32[]", getitem_1230: "f32[]", getitem_1231: "f32[]", getitem_1232: "f32[]", getitem_1233: "f32[]", getitem_1234: "f32[]", getitem_1235: "f32[]", getitem_1236: "f32[]", getitem_1237: "f32[]", getitem_1238: "f32[]", getitem_1239: "f32[]", getitem_2015: "f32[1, 1, 768]", getitem_2016: "f32[1, 198, 768]", getitem_2017: "f32[1, 1, 768]", getitem_2018: "f32[768, 3, 16, 16]", getitem_2019: "f32[768]", getitem_2020: "f32[768]", getitem_2021: "f32[768]", getitem_2022: "f32[2304, 768]", getitem_2023: "f32[2304]", getitem_2024: "f32[768, 768]", getitem_2025: "f32[768]", getitem_2026: "f32[768]", getitem_2027: "f32[768]", getitem_2028: "f32[3072, 768]", getitem_2029: "f32[3072]", getitem_2030: "f32[768, 3072]", getitem_2031: "f32[768]", getitem_2032: "f32[768]", getitem_2033: "f32[768]", getitem_2034: "f32[2304, 768]", getitem_2035: "f32[2304]", getitem_2036: "f32[768, 768]", getitem_2037: "f32[768]", getitem_2038: "f32[768]", getitem_2039: "f32[768]", getitem_2040: "f32[3072, 768]", getitem_2041: "f32[3072]", getitem_2042: "f32[768, 3072]", getitem_2043: "f32[768]", getitem_2044: "f32[768]", getitem_2045: "f32[768]", getitem_2046: "f32[2304, 768]", getitem_2047: "f32[2304]", getitem_2048: "f32[768, 768]", getitem_2049: "f32[768]", getitem_2050: "f32[768]", getitem_2051: "f32[768]", getitem_2052: "f32[3072, 768]", getitem_2053: "f32[3072]", getitem_2054: "f32[768, 3072]", getitem_2055: "f32[768]", getitem_2056: "f32[768]", getitem_2057: "f32[768]", getitem_2058: "f32[2304, 768]", getitem_2059: "f32[2304]", getitem_2060: "f32[768, 768]", getitem_2061: "f32[768]", getitem_2062: "f32[768]", getitem_2063: "f32[768]", getitem_2064: "f32[3072, 768]", getitem_2065: "f32[3072]", getitem_2066: "f32[768, 3072]", getitem_2067: "f32[768]", getitem_2068: "f32[768]", getitem_2069: "f32[768]", getitem_2070: "f32[2304, 768]", getitem_2071: "f32[2304]", getitem_2072: "f32[768, 768]", getitem_2073: "f32[768]", getitem_2074: "f32[768]", getitem_2075: "f32[768]", getitem_2076: "f32[3072, 768]", getitem_2077: "f32[3072]", getitem_2078: "f32[768, 3072]", getitem_2079: "f32[768]", getitem_2080: "f32[768]", getitem_2081: "f32[768]", getitem_2082: "f32[2304, 768]", getitem_2083: "f32[2304]", getitem_2084: "f32[768, 768]", getitem_2085: "f32[768]", getitem_2086: "f32[768]", getitem_2087: "f32[768]", getitem_2088: "f32[3072, 768]", getitem_2089: "f32[3072]", getitem_2090: "f32[768, 3072]", getitem_2091: "f32[768]", getitem_2092: "f32[768]", getitem_2093: "f32[768]", getitem_2094: "f32[2304, 768]", getitem_2095: "f32[2304]", getitem_2096: "f32[768, 768]", getitem_2097: "f32[768]", getitem_2098: "f32[768]", getitem_2099: "f32[768]", getitem_2100: "f32[3072, 768]", getitem_2101: "f32[3072]", getitem_2102: "f32[768, 3072]", getitem_2103: "f32[768]", getitem_2104: "f32[768]", getitem_2105: "f32[768]", getitem_2106: "f32[2304, 768]", getitem_2107: "f32[2304]", getitem_2108: "f32[768, 768]", getitem_2109: "f32[768]", getitem_2110: "f32[768]", getitem_2111: "f32[768]", getitem_2112: "f32[3072, 768]", getitem_2113: "f32[3072]", getitem_2114: "f32[768, 3072]", getitem_2115: "f32[768]", getitem_2116: "f32[768]", getitem_2117: "f32[768]", getitem_2118: "f32[2304, 768]", getitem_2119: "f32[2304]", getitem_2120: "f32[768, 768]", getitem_2121: "f32[768]", getitem_2122: "f32[768]", getitem_2123: "f32[768]", getitem_2124: "f32[3072, 768]", getitem_2125: "f32[3072]", getitem_2126: "f32[768, 3072]", getitem_2127: "f32[768]", getitem_2128: "f32[768]", getitem_2129: "f32[768]", getitem_2130: "f32[2304, 768]", getitem_2131: "f32[2304]", getitem_2132: "f32[768, 768]", getitem_2133: "f32[768]", getitem_2134: "f32[768]", getitem_2135: "f32[768]", getitem_2136: "f32[3072, 768]", getitem_2137: "f32[3072]", getitem_2138: "f32[768, 3072]", getitem_2139: "f32[768]", getitem_2140: "f32[768]", getitem_2141: "f32[768]", getitem_2142: "f32[2304, 768]", getitem_2143: "f32[2304]", getitem_2144: "f32[768, 768]", getitem_2145: "f32[768]", getitem_2146: "f32[768]", getitem_2147: "f32[768]", getitem_2148: "f32[3072, 768]", getitem_2149: "f32[3072]", getitem_2150: "f32[768, 3072]", getitem_2151: "f32[768]", getitem_2152: "f32[768]", getitem_2153: "f32[768]", getitem_2154: "f32[2304, 768]", getitem_2155: "f32[2304]", getitem_2156: "f32[768, 768]", getitem_2157: "f32[768]", getitem_2158: "f32[768]", getitem_2159: "f32[768]", getitem_2160: "f32[3072, 768]", getitem_2161: "f32[3072]", getitem_2162: "f32[768, 3072]", getitem_2163: "f32[768]", getitem_2164: "f32[768]", getitem_2165: "f32[768]", getitem_2166: "f32[1000, 768]", getitem_2167: "f32[1000]", getitem_2168: "f32[1000, 768]", getitem_2169: "f32[1000]", getitem_1860: "f32[]", getitem_1861: "f32[]", getitem_1862: "f32[]", getitem_1863: "f32[]", getitem_1864: "f32[]", getitem_1865: "f32[]", getitem_1866: "f32[]", getitem_1867: "f32[]", getitem_1868: "f32[]", getitem_1869: "f32[]", getitem_1870: "f32[]", getitem_1871: "f32[]", getitem_1872: "f32[]", getitem_1873: "f32[]", getitem_1874: "f32[]", getitem_1875: "f32[]", getitem_1876: "f32[]", getitem_1877: "f32[]", getitem_1878: "f32[]", getitem_1879: "f32[]", getitem_1880: "f32[]", getitem_1881: "f32[]", getitem_1882: "f32[]", getitem_1883: "f32[]", getitem_1884: "f32[]", getitem_1885: "f32[]", getitem_1886: "f32[]", getitem_1887: "f32[]", getitem_1888: "f32[]", getitem_1889: "f32[]", getitem_1890: "f32[]", getitem_1891: "f32[]", getitem_1892: "f32[]", getitem_1893: "f32[]", getitem_1894: "f32[]", getitem_1895: "f32[]", getitem_1896: "f32[]", getitem_1897: "f32[]", getitem_1898: "f32[]", getitem_1899: "f32[]", getitem_1900: "f32[]", getitem_1901: "f32[]", getitem_1902: "f32[]", getitem_1903: "f32[]", getitem_1904: "f32[]", getitem_1905: "f32[]", getitem_1906: "f32[]", getitem_1907: "f32[]", getitem_1908: "f32[]", getitem_1909: "f32[]", getitem_1910: "f32[]", getitem_1911: "f32[]", getitem_1912: "f32[]", getitem_1913: "f32[]", getitem_1914: "f32[]", getitem_1915: "f32[]", getitem_1916: "f32[]", getitem_1917: "f32[]", getitem_1918: "f32[]", getitem_1919: "f32[]", getitem_1920: "f32[]", getitem_1921: "f32[]", getitem_1922: "f32[]", getitem_1923: "f32[]", getitem_1924: "f32[]", getitem_1925: "f32[]", getitem_1926: "f32[]", getitem_1927: "f32[]", getitem_1928: "f32[]", getitem_1929: "f32[]", getitem_1930: "f32[]", getitem_1931: "f32[]", getitem_1932: "f32[]", getitem_1933: "f32[]", getitem_1934: "f32[]", getitem_1935: "f32[]", getitem_1936: "f32[]", getitem_1937: "f32[]", getitem_1938: "f32[]", getitem_1939: "f32[]", getitem_1940: "f32[]", getitem_1941: "f32[]", getitem_1942: "f32[]", getitem_1943: "f32[]", getitem_1944: "f32[]", getitem_1945: "f32[]", getitem_1946: "f32[]", getitem_1947: "f32[]", getitem_1948: "f32[]", getitem_1949: "f32[]", getitem_1950: "f32[]", getitem_1951: "f32[]", getitem_1952: "f32[]", getitem_1953: "f32[]", getitem_1954: "f32[]", getitem_1955: "f32[]", getitem_1956: "f32[]", getitem_1957: "f32[]", getitem_1958: "f32[]", getitem_1959: "f32[]", getitem_1960: "f32[]", getitem_1961: "f32[]", getitem_1962: "f32[]", getitem_1963: "f32[]", getitem_1964: "f32[]", getitem_1965: "f32[]", getitem_1966: "f32[]", getitem_1967: "f32[]", getitem_1968: "f32[]", getitem_1969: "f32[]", getitem_1970: "f32[]", getitem_1971: "f32[]", getitem_1972: "f32[]", getitem_1973: "f32[]", getitem_1974: "f32[]", getitem_1975: "f32[]", getitem_1976: "f32[]", getitem_1977: "f32[]", getitem_1978: "f32[]", getitem_1979: "f32[]", getitem_1980: "f32[]", getitem_1981: "f32[]", getitem_1982: "f32[]", getitem_1983: "f32[]", getitem_1984: "f32[]", getitem_1985: "f32[]", getitem_1986: "f32[]", getitem_1987: "f32[]", getitem_1988: "f32[]", getitem_1989: "f32[]", getitem_1990: "f32[]", getitem_1991: "f32[]", getitem_1992: "f32[]", getitem_1993: "f32[]", getitem_1994: "f32[]", getitem_1995: "f32[]", getitem_1996: "f32[]", getitem_1997: "f32[]", getitem_1998: "f32[]", getitem_1999: "f32[]", getitem_2000: "f32[]", getitem_2001: "f32[]", getitem_2002: "f32[]", getitem_2003: "f32[]", getitem_2004: "f32[]", getitem_2005: "f32[]", getitem_2006: "f32[]", getitem_2007: "f32[]", getitem_2008: "f32[]", getitem_2009: "f32[]", getitem_2010: "f32[]", getitem_2011: "f32[]", getitem_2012: "f32[]", getitem_2013: "f32[]", getitem_2014: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239], 0.01);  getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = getitem_1153 = getitem_1154 = getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = getitem_1179 = getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = getitem_1187 = getitem_1188 = getitem_1189 = getitem_1190 = getitem_1191 = getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = getitem_1234 = getitem_1235 = getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = None
        getitem: "f32[]" = _foreach_div_scalar[0]
        getitem_1240: "f32[]" = _foreach_div_scalar[1]
        getitem_1241: "f32[]" = _foreach_div_scalar[2]
        getitem_1242: "f32[]" = _foreach_div_scalar[3]
        getitem_1243: "f32[]" = _foreach_div_scalar[4]
        getitem_1244: "f32[]" = _foreach_div_scalar[5]
        getitem_1245: "f32[]" = _foreach_div_scalar[6]
        getitem_1246: "f32[]" = _foreach_div_scalar[7]
        getitem_1247: "f32[]" = _foreach_div_scalar[8]
        getitem_1248: "f32[]" = _foreach_div_scalar[9]
        getitem_1249: "f32[]" = _foreach_div_scalar[10]
        getitem_1250: "f32[]" = _foreach_div_scalar[11]
        getitem_1251: "f32[]" = _foreach_div_scalar[12]
        getitem_1252: "f32[]" = _foreach_div_scalar[13]
        getitem_1253: "f32[]" = _foreach_div_scalar[14]
        getitem_1254: "f32[]" = _foreach_div_scalar[15]
        getitem_1255: "f32[]" = _foreach_div_scalar[16]
        getitem_1256: "f32[]" = _foreach_div_scalar[17]
        getitem_1257: "f32[]" = _foreach_div_scalar[18]
        getitem_1258: "f32[]" = _foreach_div_scalar[19]
        getitem_1259: "f32[]" = _foreach_div_scalar[20]
        getitem_1260: "f32[]" = _foreach_div_scalar[21]
        getitem_1261: "f32[]" = _foreach_div_scalar[22]
        getitem_1262: "f32[]" = _foreach_div_scalar[23]
        getitem_1263: "f32[]" = _foreach_div_scalar[24]
        getitem_1264: "f32[]" = _foreach_div_scalar[25]
        getitem_1265: "f32[]" = _foreach_div_scalar[26]
        getitem_1266: "f32[]" = _foreach_div_scalar[27]
        getitem_1267: "f32[]" = _foreach_div_scalar[28]
        getitem_1268: "f32[]" = _foreach_div_scalar[29]
        getitem_1269: "f32[]" = _foreach_div_scalar[30]
        getitem_1270: "f32[]" = _foreach_div_scalar[31]
        getitem_1271: "f32[]" = _foreach_div_scalar[32]
        getitem_1272: "f32[]" = _foreach_div_scalar[33]
        getitem_1273: "f32[]" = _foreach_div_scalar[34]
        getitem_1274: "f32[]" = _foreach_div_scalar[35]
        getitem_1275: "f32[]" = _foreach_div_scalar[36]
        getitem_1276: "f32[]" = _foreach_div_scalar[37]
        getitem_1277: "f32[]" = _foreach_div_scalar[38]
        getitem_1278: "f32[]" = _foreach_div_scalar[39]
        getitem_1279: "f32[]" = _foreach_div_scalar[40]
        getitem_1280: "f32[]" = _foreach_div_scalar[41]
        getitem_1281: "f32[]" = _foreach_div_scalar[42]
        getitem_1282: "f32[]" = _foreach_div_scalar[43]
        getitem_1283: "f32[]" = _foreach_div_scalar[44]
        getitem_1284: "f32[]" = _foreach_div_scalar[45]
        getitem_1285: "f32[]" = _foreach_div_scalar[46]
        getitem_1286: "f32[]" = _foreach_div_scalar[47]
        getitem_1287: "f32[]" = _foreach_div_scalar[48]
        getitem_1288: "f32[]" = _foreach_div_scalar[49]
        getitem_1289: "f32[]" = _foreach_div_scalar[50]
        getitem_1290: "f32[]" = _foreach_div_scalar[51]
        getitem_1291: "f32[]" = _foreach_div_scalar[52]
        getitem_1292: "f32[]" = _foreach_div_scalar[53]
        getitem_1293: "f32[]" = _foreach_div_scalar[54]
        getitem_1294: "f32[]" = _foreach_div_scalar[55]
        getitem_1295: "f32[]" = _foreach_div_scalar[56]
        getitem_1296: "f32[]" = _foreach_div_scalar[57]
        getitem_1297: "f32[]" = _foreach_div_scalar[58]
        getitem_1298: "f32[]" = _foreach_div_scalar[59]
        getitem_1299: "f32[]" = _foreach_div_scalar[60]
        getitem_1300: "f32[]" = _foreach_div_scalar[61]
        getitem_1301: "f32[]" = _foreach_div_scalar[62]
        getitem_1302: "f32[]" = _foreach_div_scalar[63]
        getitem_1303: "f32[]" = _foreach_div_scalar[64]
        getitem_1304: "f32[]" = _foreach_div_scalar[65]
        getitem_1305: "f32[]" = _foreach_div_scalar[66]
        getitem_1306: "f32[]" = _foreach_div_scalar[67]
        getitem_1307: "f32[]" = _foreach_div_scalar[68]
        getitem_1308: "f32[]" = _foreach_div_scalar[69]
        getitem_1309: "f32[]" = _foreach_div_scalar[70]
        getitem_1310: "f32[]" = _foreach_div_scalar[71]
        getitem_1311: "f32[]" = _foreach_div_scalar[72]
        getitem_1312: "f32[]" = _foreach_div_scalar[73]
        getitem_1313: "f32[]" = _foreach_div_scalar[74]
        getitem_1314: "f32[]" = _foreach_div_scalar[75]
        getitem_1315: "f32[]" = _foreach_div_scalar[76]
        getitem_1316: "f32[]" = _foreach_div_scalar[77]
        getitem_1317: "f32[]" = _foreach_div_scalar[78]
        getitem_1318: "f32[]" = _foreach_div_scalar[79]
        getitem_1319: "f32[]" = _foreach_div_scalar[80]
        getitem_1320: "f32[]" = _foreach_div_scalar[81]
        getitem_1321: "f32[]" = _foreach_div_scalar[82]
        getitem_1322: "f32[]" = _foreach_div_scalar[83]
        getitem_1323: "f32[]" = _foreach_div_scalar[84]
        getitem_1324: "f32[]" = _foreach_div_scalar[85]
        getitem_1325: "f32[]" = _foreach_div_scalar[86]
        getitem_1326: "f32[]" = _foreach_div_scalar[87]
        getitem_1327: "f32[]" = _foreach_div_scalar[88]
        getitem_1328: "f32[]" = _foreach_div_scalar[89]
        getitem_1329: "f32[]" = _foreach_div_scalar[90]
        getitem_1330: "f32[]" = _foreach_div_scalar[91]
        getitem_1331: "f32[]" = _foreach_div_scalar[92]
        getitem_1332: "f32[]" = _foreach_div_scalar[93]
        getitem_1333: "f32[]" = _foreach_div_scalar[94]
        getitem_1334: "f32[]" = _foreach_div_scalar[95]
        getitem_1335: "f32[]" = _foreach_div_scalar[96]
        getitem_1336: "f32[]" = _foreach_div_scalar[97]
        getitem_1337: "f32[]" = _foreach_div_scalar[98]
        getitem_1338: "f32[]" = _foreach_div_scalar[99]
        getitem_1339: "f32[]" = _foreach_div_scalar[100]
        getitem_1340: "f32[]" = _foreach_div_scalar[101]
        getitem_1341: "f32[]" = _foreach_div_scalar[102]
        getitem_1342: "f32[]" = _foreach_div_scalar[103]
        getitem_1343: "f32[]" = _foreach_div_scalar[104]
        getitem_1344: "f32[]" = _foreach_div_scalar[105]
        getitem_1345: "f32[]" = _foreach_div_scalar[106]
        getitem_1346: "f32[]" = _foreach_div_scalar[107]
        getitem_1347: "f32[]" = _foreach_div_scalar[108]
        getitem_1348: "f32[]" = _foreach_div_scalar[109]
        getitem_1349: "f32[]" = _foreach_div_scalar[110]
        getitem_1350: "f32[]" = _foreach_div_scalar[111]
        getitem_1351: "f32[]" = _foreach_div_scalar[112]
        getitem_1352: "f32[]" = _foreach_div_scalar[113]
        getitem_1353: "f32[]" = _foreach_div_scalar[114]
        getitem_1354: "f32[]" = _foreach_div_scalar[115]
        getitem_1355: "f32[]" = _foreach_div_scalar[116]
        getitem_1356: "f32[]" = _foreach_div_scalar[117]
        getitem_1357: "f32[]" = _foreach_div_scalar[118]
        getitem_1358: "f32[]" = _foreach_div_scalar[119]
        getitem_1359: "f32[]" = _foreach_div_scalar[120]
        getitem_1360: "f32[]" = _foreach_div_scalar[121]
        getitem_1361: "f32[]" = _foreach_div_scalar[122]
        getitem_1362: "f32[]" = _foreach_div_scalar[123]
        getitem_1363: "f32[]" = _foreach_div_scalar[124]
        getitem_1364: "f32[]" = _foreach_div_scalar[125]
        getitem_1365: "f32[]" = _foreach_div_scalar[126]
        getitem_1366: "f32[]" = _foreach_div_scalar[127]
        getitem_1367: "f32[]" = _foreach_div_scalar[128]
        getitem_1368: "f32[]" = _foreach_div_scalar[129]
        getitem_1369: "f32[]" = _foreach_div_scalar[130]
        getitem_1370: "f32[]" = _foreach_div_scalar[131]
        getitem_1371: "f32[]" = _foreach_div_scalar[132]
        getitem_1372: "f32[]" = _foreach_div_scalar[133]
        getitem_1373: "f32[]" = _foreach_div_scalar[134]
        getitem_1374: "f32[]" = _foreach_div_scalar[135]
        getitem_1375: "f32[]" = _foreach_div_scalar[136]
        getitem_1376: "f32[]" = _foreach_div_scalar[137]
        getitem_1377: "f32[]" = _foreach_div_scalar[138]
        getitem_1378: "f32[]" = _foreach_div_scalar[139]
        getitem_1379: "f32[]" = _foreach_div_scalar[140]
        getitem_1380: "f32[]" = _foreach_div_scalar[141]
        getitem_1381: "f32[]" = _foreach_div_scalar[142]
        getitem_1382: "f32[]" = _foreach_div_scalar[143]
        getitem_1383: "f32[]" = _foreach_div_scalar[144]
        getitem_1384: "f32[]" = _foreach_div_scalar[145]
        getitem_1385: "f32[]" = _foreach_div_scalar[146]
        getitem_1386: "f32[]" = _foreach_div_scalar[147]
        getitem_1387: "f32[]" = _foreach_div_scalar[148]
        getitem_1388: "f32[]" = _foreach_div_scalar[149]
        getitem_1389: "f32[]" = _foreach_div_scalar[150]
        getitem_1390: "f32[]" = _foreach_div_scalar[151]
        getitem_1391: "f32[]" = _foreach_div_scalar[152]
        getitem_1392: "f32[]" = _foreach_div_scalar[153]
        getitem_1393: "f32[]" = _foreach_div_scalar[154];  _foreach_div_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169], [getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014]);  getitem_2015 = getitem_2016 = getitem_2017 = getitem_2018 = getitem_2019 = getitem_2020 = getitem_2021 = getitem_2022 = getitem_2023 = getitem_2024 = getitem_2025 = getitem_2026 = getitem_2027 = getitem_2028 = getitem_2029 = getitem_2030 = getitem_2031 = getitem_2032 = getitem_2033 = getitem_2034 = getitem_2035 = getitem_2036 = getitem_2037 = getitem_2038 = getitem_2039 = getitem_2040 = getitem_2041 = getitem_2042 = getitem_2043 = getitem_2044 = getitem_2045 = getitem_2046 = getitem_2047 = getitem_2048 = getitem_2049 = getitem_2050 = getitem_2051 = getitem_2052 = getitem_2053 = getitem_2054 = getitem_2055 = getitem_2056 = getitem_2057 = getitem_2058 = getitem_2059 = getitem_2060 = getitem_2061 = getitem_2062 = getitem_2063 = getitem_2064 = getitem_2065 = getitem_2066 = getitem_2067 = getitem_2068 = getitem_2069 = getitem_2070 = getitem_2071 = getitem_2072 = getitem_2073 = getitem_2074 = getitem_2075 = getitem_2076 = getitem_2077 = getitem_2078 = getitem_2079 = getitem_2080 = getitem_2081 = getitem_2082 = getitem_2083 = getitem_2084 = getitem_2085 = getitem_2086 = getitem_2087 = getitem_2088 = getitem_2089 = getitem_2090 = getitem_2091 = getitem_2092 = getitem_2093 = getitem_2094 = getitem_2095 = getitem_2096 = getitem_2097 = getitem_2098 = getitem_2099 = getitem_2100 = getitem_2101 = getitem_2102 = getitem_2103 = getitem_2104 = getitem_2105 = getitem_2106 = getitem_2107 = getitem_2108 = getitem_2109 = getitem_2110 = getitem_2111 = getitem_2112 = getitem_2113 = getitem_2114 = getitem_2115 = getitem_2116 = getitem_2117 = getitem_2118 = getitem_2119 = getitem_2120 = getitem_2121 = getitem_2122 = getitem_2123 = getitem_2124 = getitem_2125 = getitem_2126 = getitem_2127 = getitem_2128 = getitem_2129 = getitem_2130 = getitem_2131 = getitem_2132 = getitem_2133 = getitem_2134 = getitem_2135 = getitem_2136 = getitem_2137 = getitem_2138 = getitem_2139 = getitem_2140 = getitem_2141 = getitem_2142 = getitem_2143 = getitem_2144 = getitem_2145 = getitem_2146 = getitem_2147 = getitem_2148 = getitem_2149 = getitem_2150 = getitem_2151 = getitem_2152 = getitem_2153 = getitem_2154 = getitem_2155 = getitem_2156 = getitem_2157 = getitem_2158 = getitem_2159 = getitem_2160 = getitem_2161 = getitem_2162 = getitem_2163 = getitem_2164 = getitem_2165 = getitem_2166 = getitem_2167 = getitem_2168 = getitem_2169 = getitem_1860 = getitem_1861 = getitem_1862 = getitem_1863 = getitem_1864 = getitem_1865 = getitem_1866 = getitem_1867 = getitem_1868 = getitem_1869 = getitem_1870 = getitem_1871 = getitem_1872 = getitem_1873 = getitem_1874 = getitem_1875 = getitem_1876 = getitem_1877 = getitem_1878 = getitem_1879 = getitem_1880 = getitem_1881 = getitem_1882 = getitem_1883 = getitem_1884 = getitem_1885 = getitem_1886 = getitem_1887 = getitem_1888 = getitem_1889 = getitem_1890 = getitem_1891 = getitem_1892 = getitem_1893 = getitem_1894 = getitem_1895 = getitem_1896 = getitem_1897 = getitem_1898 = getitem_1899 = getitem_1900 = getitem_1901 = getitem_1902 = getitem_1903 = getitem_1904 = getitem_1905 = getitem_1906 = getitem_1907 = getitem_1908 = getitem_1909 = getitem_1910 = getitem_1911 = getitem_1912 = getitem_1913 = getitem_1914 = getitem_1915 = getitem_1916 = getitem_1917 = getitem_1918 = getitem_1919 = getitem_1920 = getitem_1921 = getitem_1922 = getitem_1923 = getitem_1924 = getitem_1925 = getitem_1926 = getitem_1927 = getitem_1928 = getitem_1929 = getitem_1930 = getitem_1931 = getitem_1932 = getitem_1933 = getitem_1934 = getitem_1935 = getitem_1936 = getitem_1937 = getitem_1938 = getitem_1939 = getitem_1940 = getitem_1941 = getitem_1942 = getitem_1943 = getitem_1944 = getitem_1945 = getitem_1946 = getitem_1947 = getitem_1948 = getitem_1949 = getitem_1950 = getitem_1951 = getitem_1952 = getitem_1953 = getitem_1954 = getitem_1955 = getitem_1956 = getitem_1957 = getitem_1958 = getitem_1959 = getitem_1960 = getitem_1961 = getitem_1962 = getitem_1963 = getitem_1964 = getitem_1965 = getitem_1966 = getitem_1967 = getitem_1968 = getitem_1969 = getitem_1970 = getitem_1971 = getitem_1972 = getitem_1973 = getitem_1974 = getitem_1975 = getitem_1976 = getitem_1977 = getitem_1978 = getitem_1979 = getitem_1980 = getitem_1981 = getitem_1982 = getitem_1983 = getitem_1984 = getitem_1985 = getitem_1986 = getitem_1987 = getitem_1988 = getitem_1989 = getitem_1990 = getitem_1991 = getitem_1992 = getitem_1993 = getitem_1994 = getitem_1995 = getitem_1996 = getitem_1997 = getitem_1998 = getitem_1999 = getitem_2000 = getitem_2001 = getitem_2002 = getitem_2003 = getitem_2004 = getitem_2005 = getitem_2006 = getitem_2007 = getitem_2008 = getitem_2009 = getitem_2010 = getitem_2011 = getitem_2012 = getitem_2013 = getitem_2014 = None
        getitem_2170: "f32[1, 1, 768]" = _foreach_div_list[0]
        getitem_2171: "f32[1, 198, 768]" = _foreach_div_list[1]
        getitem_2172: "f32[1, 1, 768]" = _foreach_div_list[2]
        getitem_2173: "f32[768, 3, 16, 16]" = _foreach_div_list[3]
        getitem_2174: "f32[768]" = _foreach_div_list[4]
        getitem_2175: "f32[768]" = _foreach_div_list[5]
        getitem_2176: "f32[768]" = _foreach_div_list[6]
        getitem_2177: "f32[2304, 768]" = _foreach_div_list[7]
        getitem_2178: "f32[2304]" = _foreach_div_list[8]
        getitem_2179: "f32[768, 768]" = _foreach_div_list[9]
        getitem_2180: "f32[768]" = _foreach_div_list[10]
        getitem_2181: "f32[768]" = _foreach_div_list[11]
        getitem_2182: "f32[768]" = _foreach_div_list[12]
        getitem_2183: "f32[3072, 768]" = _foreach_div_list[13]
        getitem_2184: "f32[3072]" = _foreach_div_list[14]
        getitem_2185: "f32[768, 3072]" = _foreach_div_list[15]
        getitem_2186: "f32[768]" = _foreach_div_list[16]
        getitem_2187: "f32[768]" = _foreach_div_list[17]
        getitem_2188: "f32[768]" = _foreach_div_list[18]
        getitem_2189: "f32[2304, 768]" = _foreach_div_list[19]
        getitem_2190: "f32[2304]" = _foreach_div_list[20]
        getitem_2191: "f32[768, 768]" = _foreach_div_list[21]
        getitem_2192: "f32[768]" = _foreach_div_list[22]
        getitem_2193: "f32[768]" = _foreach_div_list[23]
        getitem_2194: "f32[768]" = _foreach_div_list[24]
        getitem_2195: "f32[3072, 768]" = _foreach_div_list[25]
        getitem_2196: "f32[3072]" = _foreach_div_list[26]
        getitem_2197: "f32[768, 3072]" = _foreach_div_list[27]
        getitem_2198: "f32[768]" = _foreach_div_list[28]
        getitem_2199: "f32[768]" = _foreach_div_list[29]
        getitem_2200: "f32[768]" = _foreach_div_list[30]
        getitem_2201: "f32[2304, 768]" = _foreach_div_list[31]
        getitem_2202: "f32[2304]" = _foreach_div_list[32]
        getitem_2203: "f32[768, 768]" = _foreach_div_list[33]
        getitem_2204: "f32[768]" = _foreach_div_list[34]
        getitem_2205: "f32[768]" = _foreach_div_list[35]
        getitem_2206: "f32[768]" = _foreach_div_list[36]
        getitem_2207: "f32[3072, 768]" = _foreach_div_list[37]
        getitem_2208: "f32[3072]" = _foreach_div_list[38]
        getitem_2209: "f32[768, 3072]" = _foreach_div_list[39]
        getitem_2210: "f32[768]" = _foreach_div_list[40]
        getitem_2211: "f32[768]" = _foreach_div_list[41]
        getitem_2212: "f32[768]" = _foreach_div_list[42]
        getitem_2213: "f32[2304, 768]" = _foreach_div_list[43]
        getitem_2214: "f32[2304]" = _foreach_div_list[44]
        getitem_2215: "f32[768, 768]" = _foreach_div_list[45]
        getitem_2216: "f32[768]" = _foreach_div_list[46]
        getitem_2217: "f32[768]" = _foreach_div_list[47]
        getitem_2218: "f32[768]" = _foreach_div_list[48]
        getitem_2219: "f32[3072, 768]" = _foreach_div_list[49]
        getitem_2220: "f32[3072]" = _foreach_div_list[50]
        getitem_2221: "f32[768, 3072]" = _foreach_div_list[51]
        getitem_2222: "f32[768]" = _foreach_div_list[52]
        getitem_2223: "f32[768]" = _foreach_div_list[53]
        getitem_2224: "f32[768]" = _foreach_div_list[54]
        getitem_2225: "f32[2304, 768]" = _foreach_div_list[55]
        getitem_2226: "f32[2304]" = _foreach_div_list[56]
        getitem_2227: "f32[768, 768]" = _foreach_div_list[57]
        getitem_2228: "f32[768]" = _foreach_div_list[58]
        getitem_2229: "f32[768]" = _foreach_div_list[59]
        getitem_2230: "f32[768]" = _foreach_div_list[60]
        getitem_2231: "f32[3072, 768]" = _foreach_div_list[61]
        getitem_2232: "f32[3072]" = _foreach_div_list[62]
        getitem_2233: "f32[768, 3072]" = _foreach_div_list[63]
        getitem_2234: "f32[768]" = _foreach_div_list[64]
        getitem_2235: "f32[768]" = _foreach_div_list[65]
        getitem_2236: "f32[768]" = _foreach_div_list[66]
        getitem_2237: "f32[2304, 768]" = _foreach_div_list[67]
        getitem_2238: "f32[2304]" = _foreach_div_list[68]
        getitem_2239: "f32[768, 768]" = _foreach_div_list[69]
        getitem_2240: "f32[768]" = _foreach_div_list[70]
        getitem_2241: "f32[768]" = _foreach_div_list[71]
        getitem_2242: "f32[768]" = _foreach_div_list[72]
        getitem_2243: "f32[3072, 768]" = _foreach_div_list[73]
        getitem_2244: "f32[3072]" = _foreach_div_list[74]
        getitem_2245: "f32[768, 3072]" = _foreach_div_list[75]
        getitem_2246: "f32[768]" = _foreach_div_list[76]
        getitem_2247: "f32[768]" = _foreach_div_list[77]
        getitem_2248: "f32[768]" = _foreach_div_list[78]
        getitem_2249: "f32[2304, 768]" = _foreach_div_list[79]
        getitem_2250: "f32[2304]" = _foreach_div_list[80]
        getitem_2251: "f32[768, 768]" = _foreach_div_list[81]
        getitem_2252: "f32[768]" = _foreach_div_list[82]
        getitem_2253: "f32[768]" = _foreach_div_list[83]
        getitem_2254: "f32[768]" = _foreach_div_list[84]
        getitem_2255: "f32[3072, 768]" = _foreach_div_list[85]
        getitem_2256: "f32[3072]" = _foreach_div_list[86]
        getitem_2257: "f32[768, 3072]" = _foreach_div_list[87]
        getitem_2258: "f32[768]" = _foreach_div_list[88]
        getitem_2259: "f32[768]" = _foreach_div_list[89]
        getitem_2260: "f32[768]" = _foreach_div_list[90]
        getitem_2261: "f32[2304, 768]" = _foreach_div_list[91]
        getitem_2262: "f32[2304]" = _foreach_div_list[92]
        getitem_2263: "f32[768, 768]" = _foreach_div_list[93]
        getitem_2264: "f32[768]" = _foreach_div_list[94]
        getitem_2265: "f32[768]" = _foreach_div_list[95]
        getitem_2266: "f32[768]" = _foreach_div_list[96]
        getitem_2267: "f32[3072, 768]" = _foreach_div_list[97]
        getitem_2268: "f32[3072]" = _foreach_div_list[98]
        getitem_2269: "f32[768, 3072]" = _foreach_div_list[99]
        getitem_2270: "f32[768]" = _foreach_div_list[100]
        getitem_2271: "f32[768]" = _foreach_div_list[101]
        getitem_2272: "f32[768]" = _foreach_div_list[102]
        getitem_2273: "f32[2304, 768]" = _foreach_div_list[103]
        getitem_2274: "f32[2304]" = _foreach_div_list[104]
        getitem_2275: "f32[768, 768]" = _foreach_div_list[105]
        getitem_2276: "f32[768]" = _foreach_div_list[106]
        getitem_2277: "f32[768]" = _foreach_div_list[107]
        getitem_2278: "f32[768]" = _foreach_div_list[108]
        getitem_2279: "f32[3072, 768]" = _foreach_div_list[109]
        getitem_2280: "f32[3072]" = _foreach_div_list[110]
        getitem_2281: "f32[768, 3072]" = _foreach_div_list[111]
        getitem_2282: "f32[768]" = _foreach_div_list[112]
        getitem_2283: "f32[768]" = _foreach_div_list[113]
        getitem_2284: "f32[768]" = _foreach_div_list[114]
        getitem_2285: "f32[2304, 768]" = _foreach_div_list[115]
        getitem_2286: "f32[2304]" = _foreach_div_list[116]
        getitem_2287: "f32[768, 768]" = _foreach_div_list[117]
        getitem_2288: "f32[768]" = _foreach_div_list[118]
        getitem_2289: "f32[768]" = _foreach_div_list[119]
        getitem_2290: "f32[768]" = _foreach_div_list[120]
        getitem_2291: "f32[3072, 768]" = _foreach_div_list[121]
        getitem_2292: "f32[3072]" = _foreach_div_list[122]
        getitem_2293: "f32[768, 3072]" = _foreach_div_list[123]
        getitem_2294: "f32[768]" = _foreach_div_list[124]
        getitem_2295: "f32[768]" = _foreach_div_list[125]
        getitem_2296: "f32[768]" = _foreach_div_list[126]
        getitem_2297: "f32[2304, 768]" = _foreach_div_list[127]
        getitem_2298: "f32[2304]" = _foreach_div_list[128]
        getitem_2299: "f32[768, 768]" = _foreach_div_list[129]
        getitem_2300: "f32[768]" = _foreach_div_list[130]
        getitem_2301: "f32[768]" = _foreach_div_list[131]
        getitem_2302: "f32[768]" = _foreach_div_list[132]
        getitem_2303: "f32[3072, 768]" = _foreach_div_list[133]
        getitem_2304: "f32[3072]" = _foreach_div_list[134]
        getitem_2305: "f32[768, 3072]" = _foreach_div_list[135]
        getitem_2306: "f32[768]" = _foreach_div_list[136]
        getitem_2307: "f32[768]" = _foreach_div_list[137]
        getitem_2308: "f32[768]" = _foreach_div_list[138]
        getitem_2309: "f32[2304, 768]" = _foreach_div_list[139]
        getitem_2310: "f32[2304]" = _foreach_div_list[140]
        getitem_2311: "f32[768, 768]" = _foreach_div_list[141]
        getitem_2312: "f32[768]" = _foreach_div_list[142]
        getitem_2313: "f32[768]" = _foreach_div_list[143]
        getitem_2314: "f32[768]" = _foreach_div_list[144]
        getitem_2315: "f32[3072, 768]" = _foreach_div_list[145]
        getitem_2316: "f32[3072]" = _foreach_div_list[146]
        getitem_2317: "f32[768, 3072]" = _foreach_div_list[147]
        getitem_2318: "f32[768]" = _foreach_div_list[148]
        getitem_2319: "f32[768]" = _foreach_div_list[149]
        getitem_2320: "f32[768]" = _foreach_div_list[150]
        getitem_2321: "f32[1000, 768]" = _foreach_div_list[151]
        getitem_2322: "f32[1000]" = _foreach_div_list[152]
        getitem_2323: "f32[1000, 768]" = _foreach_div_list[153]
        getitem_2324: "f32[1000]" = _foreach_div_list[154];  _foreach_div_list = None
        return (getitem, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

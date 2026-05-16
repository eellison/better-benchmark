"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: 03175854b672
Shape hash: 2fb8692e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg305_1: "f32[768]", arg303_1: "f32[50, 768]", arg905_1: "f32[768, 512]", arg906_1: "f32[768, 3, 32, 32]", arg907_1: "f32[768]", arg908_1: "f32[768]", arg909_1: "f32[2304, 768]", arg910_1: "f32[2304]", arg911_1: "f32[768, 768]", arg912_1: "f32[768]", arg913_1: "f32[3072, 768]", arg914_1: "f32[3072]", arg915_1: "f32[768, 3072]", arg916_1: "f32[768]", arg917_1: "f32[768]", arg918_1: "f32[768]", arg919_1: "f32[768]", arg920_1: "f32[768]", arg921_1: "f32[2304, 768]", arg922_1: "f32[2304]", arg923_1: "f32[768, 768]", arg924_1: "f32[768]", arg925_1: "f32[3072, 768]", arg926_1: "f32[3072]", arg927_1: "f32[768, 3072]", arg928_1: "f32[768]", arg929_1: "f32[768]", arg930_1: "f32[768]", arg931_1: "f32[768]", arg932_1: "f32[768]", arg933_1: "f32[2304, 768]", arg934_1: "f32[2304]", arg935_1: "f32[768, 768]", arg936_1: "f32[768]", arg937_1: "f32[3072, 768]", arg938_1: "f32[3072]", arg939_1: "f32[768, 3072]", arg940_1: "f32[768]", arg941_1: "f32[768]", arg942_1: "f32[768]", arg943_1: "f32[768]", arg944_1: "f32[768]", arg945_1: "f32[2304, 768]", arg946_1: "f32[2304]", arg947_1: "f32[768, 768]", arg948_1: "f32[768]", arg949_1: "f32[3072, 768]", arg950_1: "f32[3072]", arg951_1: "f32[768, 3072]", arg952_1: "f32[768]", arg953_1: "f32[768]", arg954_1: "f32[768]", arg955_1: "f32[768]", arg956_1: "f32[768]", arg957_1: "f32[2304, 768]", arg958_1: "f32[2304]", arg959_1: "f32[768, 768]", arg960_1: "f32[768]", arg961_1: "f32[3072, 768]", arg962_1: "f32[3072]", arg963_1: "f32[768, 3072]", arg964_1: "f32[768]", arg965_1: "f32[768]", arg966_1: "f32[768]", arg967_1: "f32[768]", arg968_1: "f32[768]", arg969_1: "f32[2304, 768]", arg970_1: "f32[2304]", arg971_1: "f32[768, 768]", arg972_1: "f32[768]", arg973_1: "f32[3072, 768]", arg974_1: "f32[3072]", arg975_1: "f32[768, 3072]", arg976_1: "f32[768]", arg977_1: "f32[768]", arg978_1: "f32[768]", arg979_1: "f32[768]", arg980_1: "f32[768]", arg981_1: "f32[2304, 768]", arg982_1: "f32[2304]", arg983_1: "f32[768, 768]", arg984_1: "f32[768]", arg985_1: "f32[3072, 768]", arg986_1: "f32[3072]", arg987_1: "f32[768, 3072]", arg988_1: "f32[768]", arg989_1: "f32[768]", arg990_1: "f32[768]", arg991_1: "f32[768]", arg992_1: "f32[768]", arg993_1: "f32[2304, 768]", arg994_1: "f32[2304]", arg995_1: "f32[768, 768]", arg996_1: "f32[768]", arg997_1: "f32[3072, 768]", arg998_1: "f32[3072]", arg999_1: "f32[768, 3072]", arg1000_1: "f32[768]", arg1001_1: "f32[768]", arg1002_1: "f32[768]", arg1003_1: "f32[768]", arg1004_1: "f32[768]", arg1005_1: "f32[2304, 768]", arg1006_1: "f32[2304]", arg1007_1: "f32[768, 768]", arg1008_1: "f32[768]", arg1009_1: "f32[3072, 768]", arg1010_1: "f32[3072]", arg1011_1: "f32[768, 3072]", arg1012_1: "f32[768]", arg1013_1: "f32[768]", arg1014_1: "f32[768]", arg1015_1: "f32[768]", arg1016_1: "f32[768]", arg1017_1: "f32[2304, 768]", arg1018_1: "f32[2304]", arg1019_1: "f32[768, 768]", arg1020_1: "f32[768]", arg1021_1: "f32[3072, 768]", arg1022_1: "f32[3072]", arg1023_1: "f32[768, 3072]", arg1024_1: "f32[768]", arg1025_1: "f32[768]", arg1026_1: "f32[768]", arg1027_1: "f32[768]", arg1028_1: "f32[768]", arg1029_1: "f32[2304, 768]", arg1030_1: "f32[2304]", arg1031_1: "f32[768, 768]", arg1032_1: "f32[768]", arg1033_1: "f32[3072, 768]", arg1034_1: "f32[3072]", arg1035_1: "f32[768, 3072]", arg1036_1: "f32[768]", arg1037_1: "f32[768]", arg1038_1: "f32[768]", arg1039_1: "f32[768]", arg1040_1: "f32[768]", arg1041_1: "f32[2304, 768]", arg1042_1: "f32[2304]", arg1043_1: "f32[768, 768]", arg1044_1: "f32[768]", arg1045_1: "f32[3072, 768]", arg1046_1: "f32[3072]", arg1047_1: "f32[768, 3072]", arg1048_1: "f32[768]", arg1049_1: "f32[768]", arg1050_1: "f32[768]", arg1051_1: "f32[768]", arg1052_1: "f32[768]", arg1053_1: "f32[768]", arg1054_1: "f32[768]", arg1055_1: "f32[77, 512]", arg1056_1: "f32[49408, 512]", arg1057_1: "f32[1536, 512]", arg1058_1: "f32[1536]", arg1059_1: "f32[512, 512]", arg1060_1: "f32[512]", arg1061_1: "f32[2048, 512]", arg1062_1: "f32[2048]", arg1063_1: "f32[512, 2048]", arg1064_1: "f32[512]", arg1065_1: "f32[512]", arg1066_1: "f32[512]", arg1067_1: "f32[512]", arg1068_1: "f32[512]", arg1069_1: "f32[1536, 512]", arg1070_1: "f32[1536]", arg1071_1: "f32[512, 512]", arg1072_1: "f32[512]", arg1073_1: "f32[2048, 512]", arg1074_1: "f32[2048]", arg1075_1: "f32[512, 2048]", arg1076_1: "f32[512]", arg1077_1: "f32[512]", arg1078_1: "f32[512]", arg1079_1: "f32[512]", arg1080_1: "f32[512]", arg1081_1: "f32[1536, 512]", arg1082_1: "f32[1536]", arg1083_1: "f32[512, 512]", arg1084_1: "f32[512]", arg1085_1: "f32[2048, 512]", arg1086_1: "f32[2048]", arg1087_1: "f32[512, 2048]", arg1088_1: "f32[512]", arg1089_1: "f32[512]", arg1090_1: "f32[512]", arg1091_1: "f32[512]", arg1092_1: "f32[512]", arg1093_1: "f32[1536, 512]", arg1094_1: "f32[1536]", arg1095_1: "f32[512, 512]", arg1096_1: "f32[512]", arg1097_1: "f32[2048, 512]", arg1098_1: "f32[2048]", arg1099_1: "f32[512, 2048]", arg1100_1: "f32[512]", arg1101_1: "f32[512]", arg1102_1: "f32[512]", arg1103_1: "f32[512]", arg1104_1: "f32[512]", arg1105_1: "f32[1536, 512]", arg1106_1: "f32[1536]", arg1107_1: "f32[512, 512]", arg1108_1: "f32[512]", arg1109_1: "f32[2048, 512]", arg1110_1: "f32[2048]", arg1111_1: "f32[512, 2048]", arg1112_1: "f32[512]", arg1113_1: "f32[512]", arg1114_1: "f32[512]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", arg1117_1: "f32[1536, 512]", arg1118_1: "f32[1536]", arg1119_1: "f32[512, 512]", arg1120_1: "f32[512]", arg1121_1: "f32[2048, 512]", arg1122_1: "f32[2048]", arg1123_1: "f32[512, 2048]", arg1124_1: "f32[512]", arg1125_1: "f32[512]", arg1126_1: "f32[512]", arg1127_1: "f32[512]", arg1128_1: "f32[512]", arg1129_1: "f32[1536, 512]", arg1130_1: "f32[1536]", arg1131_1: "f32[512, 512]", arg1132_1: "f32[512]", arg1133_1: "f32[2048, 512]", arg1134_1: "f32[2048]", arg1135_1: "f32[512, 2048]", arg1136_1: "f32[512]", arg1137_1: "f32[512]", arg1138_1: "f32[512]", arg1139_1: "f32[512]", arg1140_1: "f32[512]", arg1141_1: "f32[1536, 512]", arg1142_1: "f32[1536]", arg1143_1: "f32[512, 512]", arg1144_1: "f32[512]", arg1145_1: "f32[2048, 512]", arg1146_1: "f32[2048]", arg1147_1: "f32[512, 2048]", arg1148_1: "f32[512]", arg1149_1: "f32[512]", arg1150_1: "f32[512]", arg1151_1: "f32[512]", arg1152_1: "f32[512]", arg1153_1: "f32[1536, 512]", arg1154_1: "f32[1536]", arg1155_1: "f32[512, 512]", arg1156_1: "f32[512]", arg1157_1: "f32[2048, 512]", arg1158_1: "f32[2048]", arg1159_1: "f32[512, 2048]", arg1160_1: "f32[512]", arg1161_1: "f32[512]", arg1162_1: "f32[512]", arg1163_1: "f32[512]", arg1164_1: "f32[512]", arg1165_1: "f32[1536, 512]", arg1166_1: "f32[1536]", arg1167_1: "f32[512, 512]", arg1168_1: "f32[512]", arg1169_1: "f32[2048, 512]", arg1170_1: "f32[2048]", arg1171_1: "f32[512, 2048]", arg1172_1: "f32[512]", arg1173_1: "f32[512]", arg1174_1: "f32[512]", arg1175_1: "f32[512]", arg1176_1: "f32[512]", arg1177_1: "f32[1536, 512]", arg1178_1: "f32[1536]", arg1179_1: "f32[512, 512]", arg1180_1: "f32[512]", arg1181_1: "f32[2048, 512]", arg1182_1: "f32[2048]", arg1183_1: "f32[512, 2048]", arg1184_1: "f32[512]", arg1185_1: "f32[512]", arg1186_1: "f32[512]", arg1187_1: "f32[512]", arg1188_1: "f32[512]", arg1189_1: "f32[1536, 512]", arg1190_1: "f32[1536]", arg1191_1: "f32[512, 512]", arg1192_1: "f32[512]", arg1193_1: "f32[2048, 512]", arg1194_1: "f32[2048]", arg1195_1: "f32[512, 2048]", arg1196_1: "f32[512]", arg1197_1: "f32[512]", arg1198_1: "f32[512]", arg1199_1: "f32[512]", arg1200_1: "f32[512]", arg1201_1: "f32[512]", arg1202_1: "f32[512]", arg1203_1: "f32[512, 512]", getitem_1806: "f32[]", getitem_1807: "f32[]", getitem_1808: "f32[]", getitem_1809: "f32[]", getitem_1810: "f32[]", getitem_1811: "f32[]", getitem_1812: "f32[]", getitem_1813: "f32[]", getitem_1814: "f32[]", getitem_1815: "f32[]", getitem_1816: "f32[]", getitem_1817: "f32[]", getitem_1818: "f32[]", getitem_1819: "f32[]", getitem_1820: "f32[]", getitem_1821: "f32[]", getitem_1822: "f32[]", getitem_1823: "f32[]", getitem_1824: "f32[]", getitem_1825: "f32[]", getitem_1826: "f32[]", getitem_1827: "f32[]", getitem_1828: "f32[]", getitem_1829: "f32[]", getitem_1830: "f32[]", getitem_1831: "f32[]", getitem_1832: "f32[]", getitem_1833: "f32[]", getitem_1834: "f32[]", getitem_1835: "f32[]", getitem_1836: "f32[]", getitem_1837: "f32[]", getitem_1838: "f32[]", getitem_1839: "f32[]", getitem_1840: "f32[]", getitem_1841: "f32[]", getitem_1842: "f32[]", getitem_1843: "f32[]", getitem_1844: "f32[]", getitem_1845: "f32[]", getitem_1846: "f32[]", getitem_1847: "f32[]", getitem_1848: "f32[]", getitem_1849: "f32[]", getitem_1850: "f32[]", getitem_1851: "f32[]", getitem_1852: "f32[]", getitem_1853: "f32[]", getitem_1854: "f32[]", getitem_1855: "f32[]", getitem_1856: "f32[]", getitem_1857: "f32[]", getitem_1858: "f32[]", getitem_1859: "f32[]", getitem_1860: "f32[]", getitem_1861: "f32[]", getitem_1862: "f32[]", getitem_1863: "f32[]", getitem_1864: "f32[]", getitem_1865: "f32[]", getitem_1866: "f32[]", getitem_1867: "f32[]", getitem_1868: "f32[]", getitem_1869: "f32[]", getitem_1870: "f32[]", getitem_1871: "f32[]", getitem_1872: "f32[]", getitem_1873: "f32[]", getitem_1874: "f32[]", getitem_1875: "f32[]", getitem_1876: "f32[]", getitem_1877: "f32[]", getitem_1878: "f32[]", getitem_1879: "f32[]", getitem_1880: "f32[]", getitem_1881: "f32[]", getitem_1882: "f32[]", getitem_1883: "f32[]", getitem_1884: "f32[]", getitem_1885: "f32[]", getitem_1886: "f32[]", getitem_1887: "f32[]", getitem_1888: "f32[]", getitem_1889: "f32[]", getitem_1890: "f32[]", getitem_1891: "f32[]", getitem_1892: "f32[]", getitem_1893: "f32[]", getitem_1894: "f32[]", getitem_1895: "f32[]", getitem_1896: "f32[]", getitem_1897: "f32[]", getitem_1898: "f32[]", getitem_1899: "f32[]", getitem_1900: "f32[]", getitem_1901: "f32[]", getitem_1902: "f32[]", getitem_1903: "f32[]", getitem_1904: "f32[]", getitem_1905: "f32[]", getitem_1906: "f32[]", getitem_1907: "f32[]", getitem_1908: "f32[]", getitem_1909: "f32[]", getitem_1910: "f32[]", getitem_1911: "f32[]", getitem_1912: "f32[]", getitem_1913: "f32[]", getitem_1914: "f32[]", getitem_1915: "f32[]", getitem_1916: "f32[]", getitem_1917: "f32[]", getitem_1918: "f32[]", getitem_1919: "f32[]", getitem_1920: "f32[]", getitem_1921: "f32[]", getitem_1922: "f32[]", getitem_1923: "f32[]", getitem_1924: "f32[]", getitem_1925: "f32[]", getitem_1926: "f32[]", getitem_1927: "f32[]", getitem_1928: "f32[]", getitem_1929: "f32[]", getitem_1930: "f32[]", getitem_1931: "f32[]", getitem_1932: "f32[]", getitem_1933: "f32[]", getitem_1934: "f32[]", getitem_1935: "f32[]", getitem_1936: "f32[]", getitem_1937: "f32[]", getitem_1938: "f32[]", getitem_1939: "f32[]", getitem_1940: "f32[]", getitem_1941: "f32[]", getitem_1942: "f32[]", getitem_1943: "f32[]", getitem_1944: "f32[]", getitem_1945: "f32[]", getitem_1946: "f32[]", getitem_1947: "f32[]", getitem_1948: "f32[]", getitem_1949: "f32[]", getitem_1950: "f32[]", getitem_1951: "f32[]", getitem_1952: "f32[]", getitem_1953: "f32[]", getitem_1954: "f32[]", getitem_1955: "f32[]", getitem_1956: "f32[]", getitem_1957: "f32[]", getitem_1958: "f32[]", getitem_1959: "f32[]", getitem_1960: "f32[]", getitem_1961: "f32[]", getitem_1962: "f32[]", getitem_1963: "f32[]", getitem_1964: "f32[]", getitem_1965: "f32[]", getitem_1966: "f32[]", getitem_1967: "f32[]", getitem_1968: "f32[]", getitem_1969: "f32[]", getitem_1970: "f32[]", getitem_1971: "f32[]", getitem_1972: "f32[]", getitem_1973: "f32[]", getitem_1974: "f32[]", getitem_1975: "f32[]", getitem_1976: "f32[]", getitem_1977: "f32[]", getitem_1978: "f32[]", getitem_1979: "f32[]", getitem_1980: "f32[]", getitem_1981: "f32[]", getitem_1982: "f32[]", getitem_1983: "f32[]", getitem_1984: "f32[]", getitem_1985: "f32[]", getitem_1986: "f32[]", getitem_1987: "f32[]", getitem_1988: "f32[]", getitem_1989: "f32[]", getitem_1990: "f32[]", getitem_1991: "f32[]", getitem_1992: "f32[]", getitem_1993: "f32[]", getitem_1994: "f32[]", getitem_1995: "f32[]", getitem_1996: "f32[]", getitem_1997: "f32[]", getitem_1998: "f32[]", getitem_1999: "f32[]", getitem_2000: "f32[]", getitem_2001: "f32[]", getitem_2002: "f32[]", getitem_2003: "f32[]", getitem_2004: "f32[]", getitem_2005: "f32[]", getitem_2006: "f32[]", getitem_2007: "f32[]", getitem_2008: "f32[]", getitem_2009: "f32[]", getitem_2010: "f32[]", getitem_2011: "f32[]", getitem_2012: "f32[]", getitem_2013: "f32[]", getitem_2014: "f32[]", getitem_2015: "f32[]", getitem_2016: "f32[]", getitem_2017: "f32[]", getitem_2018: "f32[]", getitem_2019: "f32[]", getitem_2020: "f32[]", getitem_2021: "f32[]", getitem_2022: "f32[]", getitem_2023: "f32[]", getitem_2024: "f32[]", getitem_2025: "f32[]", getitem_2026: "f32[]", getitem_2027: "f32[]", getitem_2028: "f32[]", getitem_2029: "f32[]", getitem_2030: "f32[]", getitem_2031: "f32[]", getitem_2032: "f32[]", getitem_2033: "f32[]", getitem_2034: "f32[]", getitem_2035: "f32[]", getitem_2036: "f32[]", getitem_2037: "f32[]", getitem_2038: "f32[]", getitem_2039: "f32[]", getitem_2040: "f32[]", getitem_2041: "f32[]", getitem_2042: "f32[]", getitem_2043: "f32[]", getitem_2044: "f32[]", getitem_2045: "f32[]", getitem_2046: "f32[]", getitem_2047: "f32[]", getitem_2048: "f32[]", getitem_2049: "f32[]", getitem_2050: "f32[]", getitem_2051: "f32[]", getitem_2052: "f32[]", getitem_2053: "f32[]", getitem_2054: "f32[]", getitem_2055: "f32[]", getitem_2056: "f32[]", getitem_2057: "f32[]", getitem_2058: "f32[]", getitem_2059: "f32[]", getitem_2060: "f32[]", getitem_2061: "f32[]", getitem_2062: "f32[]", getitem_2063: "f32[]", getitem_2064: "f32[]", getitem_2065: "f32[]", getitem_2066: "f32[]", getitem_2067: "f32[]", getitem_2068: "f32[]", getitem_2069: "f32[]", getitem_2070: "f32[]", getitem_2071: "f32[]", getitem_2072: "f32[]", getitem_2073: "f32[]", getitem_2074: "f32[]", getitem_2075: "f32[]", getitem_2076: "f32[]", getitem_2077: "f32[]", getitem_2078: "f32[]", getitem_2079: "f32[]", getitem_2080: "f32[]", getitem_2081: "f32[]", getitem_2082: "f32[]", getitem_2083: "f32[]", getitem_2084: "f32[]", getitem_2085: "f32[]", getitem_2086: "f32[]", getitem_2087: "f32[]", getitem_2088: "f32[]", getitem_2089: "f32[]", getitem_2090: "f32[]", getitem_2091: "f32[]", getitem_2092: "f32[]", getitem_2093: "f32[]", getitem_2094: "f32[]", getitem_2095: "f32[]", getitem_2096: "f32[]", getitem_2097: "f32[]", getitem_2098: "f32[]", getitem_2099: "f32[]", getitem_2100: "f32[]", getitem_2101: "f32[]", getitem_2102: "f32[]", getitem_2103: "f32[]", getitem_2104: "f32[]", getitem_2105: "f32[]", getitem_2106: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg305_1, arg303_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1, arg1030_1, arg1031_1, arg1032_1, arg1033_1, arg1034_1, arg1035_1, arg1036_1, arg1037_1, arg1038_1, arg1039_1, arg1040_1, arg1041_1, arg1042_1, arg1043_1, arg1044_1, arg1045_1, arg1046_1, arg1047_1, arg1048_1, arg1049_1, arg1050_1, arg1051_1, arg1052_1, arg1053_1, arg1054_1, arg1055_1, arg1056_1, arg1057_1, arg1058_1, arg1059_1, arg1060_1, arg1061_1, arg1062_1, arg1063_1, arg1064_1, arg1065_1, arg1066_1, arg1067_1, arg1068_1, arg1069_1, arg1070_1, arg1071_1, arg1072_1, arg1073_1, arg1074_1, arg1075_1, arg1076_1, arg1077_1, arg1078_1, arg1079_1, arg1080_1, arg1081_1, arg1082_1, arg1083_1, arg1084_1, arg1085_1, arg1086_1, arg1087_1, arg1088_1, arg1089_1, arg1090_1, arg1091_1, arg1092_1, arg1093_1, arg1094_1, arg1095_1, arg1096_1, arg1097_1, arg1098_1, arg1099_1, arg1100_1, arg1101_1, arg1102_1, arg1103_1, arg1104_1, arg1105_1, arg1106_1, arg1107_1, arg1108_1, arg1109_1, arg1110_1, arg1111_1, arg1112_1, arg1113_1, arg1114_1, arg1115_1, arg1116_1, arg1117_1, arg1118_1, arg1119_1, arg1120_1, arg1121_1, arg1122_1, arg1123_1, arg1124_1, arg1125_1, arg1126_1, arg1127_1, arg1128_1, arg1129_1, arg1130_1, arg1131_1, arg1132_1, arg1133_1, arg1134_1, arg1135_1, arg1136_1, arg1137_1, arg1138_1, arg1139_1, arg1140_1, arg1141_1, arg1142_1, arg1143_1, arg1144_1, arg1145_1, arg1146_1, arg1147_1, arg1148_1, arg1149_1, arg1150_1, arg1151_1, arg1152_1, arg1153_1, arg1154_1, arg1155_1, arg1156_1, arg1157_1, arg1158_1, arg1159_1, arg1160_1, arg1161_1, arg1162_1, arg1163_1, arg1164_1, arg1165_1, arg1166_1, arg1167_1, arg1168_1, arg1169_1, arg1170_1, arg1171_1, arg1172_1, arg1173_1, arg1174_1, arg1175_1, arg1176_1, arg1177_1, arg1178_1, arg1179_1, arg1180_1, arg1181_1, arg1182_1, arg1183_1, arg1184_1, arg1185_1, arg1186_1, arg1187_1, arg1188_1, arg1189_1, arg1190_1, arg1191_1, arg1192_1, arg1193_1, arg1194_1, arg1195_1, arg1196_1, arg1197_1, arg1198_1, arg1199_1, arg1200_1, arg1201_1, arg1202_1, arg1203_1], 0.999);  arg305_1 = arg303_1 = arg905_1 = arg906_1 = arg907_1 = arg908_1 = arg909_1 = arg910_1 = arg911_1 = arg912_1 = arg913_1 = arg914_1 = arg915_1 = arg916_1 = arg917_1 = arg918_1 = arg919_1 = arg920_1 = arg921_1 = arg922_1 = arg923_1 = arg924_1 = arg925_1 = arg926_1 = arg927_1 = arg928_1 = arg929_1 = arg930_1 = arg931_1 = arg932_1 = arg933_1 = arg934_1 = arg935_1 = arg936_1 = arg937_1 = arg938_1 = arg939_1 = arg940_1 = arg941_1 = arg942_1 = arg943_1 = arg944_1 = arg945_1 = arg946_1 = arg947_1 = arg948_1 = arg949_1 = arg950_1 = arg951_1 = arg952_1 = arg953_1 = arg954_1 = arg955_1 = arg956_1 = arg957_1 = arg958_1 = arg959_1 = arg960_1 = arg961_1 = arg962_1 = arg963_1 = arg964_1 = arg965_1 = arg966_1 = arg967_1 = arg968_1 = arg969_1 = arg970_1 = arg971_1 = arg972_1 = arg973_1 = arg974_1 = arg975_1 = arg976_1 = arg977_1 = arg978_1 = arg979_1 = arg980_1 = arg981_1 = arg982_1 = arg983_1 = arg984_1 = arg985_1 = arg986_1 = arg987_1 = arg988_1 = arg989_1 = arg990_1 = arg991_1 = arg992_1 = arg993_1 = arg994_1 = arg995_1 = arg996_1 = arg997_1 = arg998_1 = arg999_1 = arg1000_1 = arg1001_1 = arg1002_1 = arg1003_1 = arg1004_1 = arg1005_1 = arg1006_1 = arg1007_1 = arg1008_1 = arg1009_1 = arg1010_1 = arg1011_1 = arg1012_1 = arg1013_1 = arg1014_1 = arg1015_1 = arg1016_1 = arg1017_1 = arg1018_1 = arg1019_1 = arg1020_1 = arg1021_1 = arg1022_1 = arg1023_1 = arg1024_1 = arg1025_1 = arg1026_1 = arg1027_1 = arg1028_1 = arg1029_1 = arg1030_1 = arg1031_1 = arg1032_1 = arg1033_1 = arg1034_1 = arg1035_1 = arg1036_1 = arg1037_1 = arg1038_1 = arg1039_1 = arg1040_1 = arg1041_1 = arg1042_1 = arg1043_1 = arg1044_1 = arg1045_1 = arg1046_1 = arg1047_1 = arg1048_1 = arg1049_1 = arg1050_1 = arg1051_1 = arg1052_1 = arg1053_1 = arg1054_1 = arg1055_1 = arg1056_1 = arg1057_1 = arg1058_1 = arg1059_1 = arg1060_1 = arg1061_1 = arg1062_1 = arg1063_1 = arg1064_1 = arg1065_1 = arg1066_1 = arg1067_1 = arg1068_1 = arg1069_1 = arg1070_1 = arg1071_1 = arg1072_1 = arg1073_1 = arg1074_1 = arg1075_1 = arg1076_1 = arg1077_1 = arg1078_1 = arg1079_1 = arg1080_1 = arg1081_1 = arg1082_1 = arg1083_1 = arg1084_1 = arg1085_1 = arg1086_1 = arg1087_1 = arg1088_1 = arg1089_1 = arg1090_1 = arg1091_1 = arg1092_1 = arg1093_1 = arg1094_1 = arg1095_1 = arg1096_1 = arg1097_1 = arg1098_1 = arg1099_1 = arg1100_1 = arg1101_1 = arg1102_1 = arg1103_1 = arg1104_1 = arg1105_1 = arg1106_1 = arg1107_1 = arg1108_1 = arg1109_1 = arg1110_1 = arg1111_1 = arg1112_1 = arg1113_1 = arg1114_1 = arg1115_1 = arg1116_1 = arg1117_1 = arg1118_1 = arg1119_1 = arg1120_1 = arg1121_1 = arg1122_1 = arg1123_1 = arg1124_1 = arg1125_1 = arg1126_1 = arg1127_1 = arg1128_1 = arg1129_1 = arg1130_1 = arg1131_1 = arg1132_1 = arg1133_1 = arg1134_1 = arg1135_1 = arg1136_1 = arg1137_1 = arg1138_1 = arg1139_1 = arg1140_1 = arg1141_1 = arg1142_1 = arg1143_1 = arg1144_1 = arg1145_1 = arg1146_1 = arg1147_1 = arg1148_1 = arg1149_1 = arg1150_1 = arg1151_1 = arg1152_1 = arg1153_1 = arg1154_1 = arg1155_1 = arg1156_1 = arg1157_1 = arg1158_1 = arg1159_1 = arg1160_1 = arg1161_1 = arg1162_1 = arg1163_1 = arg1164_1 = arg1165_1 = arg1166_1 = arg1167_1 = arg1168_1 = arg1169_1 = arg1170_1 = arg1171_1 = arg1172_1 = arg1173_1 = arg1174_1 = arg1175_1 = arg1176_1 = arg1177_1 = arg1178_1 = arg1179_1 = arg1180_1 = arg1181_1 = arg1182_1 = arg1183_1 = arg1184_1 = arg1185_1 = arg1186_1 = arg1187_1 = arg1188_1 = arg1189_1 = arg1190_1 = arg1191_1 = arg1192_1 = arg1193_1 = arg1194_1 = arg1195_1 = arg1196_1 = arg1197_1 = arg1198_1 = arg1199_1 = arg1200_1 = arg1201_1 = arg1202_1 = arg1203_1 = None
        getitem: "f32[768]" = _foreach_mul_scalar[0]
        getitem_1: "f32[50, 768]" = _foreach_mul_scalar[1]
        getitem_2: "f32[768, 512]" = _foreach_mul_scalar[2]
        getitem_3: "f32[768, 3, 32, 32]" = _foreach_mul_scalar[3]
        getitem_4: "f32[768]" = _foreach_mul_scalar[4]
        getitem_5: "f32[768]" = _foreach_mul_scalar[5]
        getitem_6: "f32[2304, 768]" = _foreach_mul_scalar[6]
        getitem_7: "f32[2304]" = _foreach_mul_scalar[7]
        getitem_8: "f32[768, 768]" = _foreach_mul_scalar[8]
        getitem_9: "f32[768]" = _foreach_mul_scalar[9]
        getitem_10: "f32[3072, 768]" = _foreach_mul_scalar[10]
        getitem_11: "f32[3072]" = _foreach_mul_scalar[11]
        getitem_12: "f32[768, 3072]" = _foreach_mul_scalar[12]
        getitem_13: "f32[768]" = _foreach_mul_scalar[13]
        getitem_14: "f32[768]" = _foreach_mul_scalar[14]
        getitem_15: "f32[768]" = _foreach_mul_scalar[15]
        getitem_16: "f32[768]" = _foreach_mul_scalar[16]
        getitem_17: "f32[768]" = _foreach_mul_scalar[17]
        getitem_18: "f32[2304, 768]" = _foreach_mul_scalar[18]
        getitem_19: "f32[2304]" = _foreach_mul_scalar[19]
        getitem_20: "f32[768, 768]" = _foreach_mul_scalar[20]
        getitem_21: "f32[768]" = _foreach_mul_scalar[21]
        getitem_22: "f32[3072, 768]" = _foreach_mul_scalar[22]
        getitem_23: "f32[3072]" = _foreach_mul_scalar[23]
        getitem_24: "f32[768, 3072]" = _foreach_mul_scalar[24]
        getitem_25: "f32[768]" = _foreach_mul_scalar[25]
        getitem_26: "f32[768]" = _foreach_mul_scalar[26]
        getitem_27: "f32[768]" = _foreach_mul_scalar[27]
        getitem_28: "f32[768]" = _foreach_mul_scalar[28]
        getitem_29: "f32[768]" = _foreach_mul_scalar[29]
        getitem_30: "f32[2304, 768]" = _foreach_mul_scalar[30]
        getitem_31: "f32[2304]" = _foreach_mul_scalar[31]
        getitem_32: "f32[768, 768]" = _foreach_mul_scalar[32]
        getitem_33: "f32[768]" = _foreach_mul_scalar[33]
        getitem_34: "f32[3072, 768]" = _foreach_mul_scalar[34]
        getitem_35: "f32[3072]" = _foreach_mul_scalar[35]
        getitem_36: "f32[768, 3072]" = _foreach_mul_scalar[36]
        getitem_37: "f32[768]" = _foreach_mul_scalar[37]
        getitem_38: "f32[768]" = _foreach_mul_scalar[38]
        getitem_39: "f32[768]" = _foreach_mul_scalar[39]
        getitem_40: "f32[768]" = _foreach_mul_scalar[40]
        getitem_41: "f32[768]" = _foreach_mul_scalar[41]
        getitem_42: "f32[2304, 768]" = _foreach_mul_scalar[42]
        getitem_43: "f32[2304]" = _foreach_mul_scalar[43]
        getitem_44: "f32[768, 768]" = _foreach_mul_scalar[44]
        getitem_45: "f32[768]" = _foreach_mul_scalar[45]
        getitem_46: "f32[3072, 768]" = _foreach_mul_scalar[46]
        getitem_47: "f32[3072]" = _foreach_mul_scalar[47]
        getitem_48: "f32[768, 3072]" = _foreach_mul_scalar[48]
        getitem_49: "f32[768]" = _foreach_mul_scalar[49]
        getitem_50: "f32[768]" = _foreach_mul_scalar[50]
        getitem_51: "f32[768]" = _foreach_mul_scalar[51]
        getitem_52: "f32[768]" = _foreach_mul_scalar[52]
        getitem_53: "f32[768]" = _foreach_mul_scalar[53]
        getitem_54: "f32[2304, 768]" = _foreach_mul_scalar[54]
        getitem_55: "f32[2304]" = _foreach_mul_scalar[55]
        getitem_56: "f32[768, 768]" = _foreach_mul_scalar[56]
        getitem_57: "f32[768]" = _foreach_mul_scalar[57]
        getitem_58: "f32[3072, 768]" = _foreach_mul_scalar[58]
        getitem_59: "f32[3072]" = _foreach_mul_scalar[59]
        getitem_60: "f32[768, 3072]" = _foreach_mul_scalar[60]
        getitem_61: "f32[768]" = _foreach_mul_scalar[61]
        getitem_62: "f32[768]" = _foreach_mul_scalar[62]
        getitem_63: "f32[768]" = _foreach_mul_scalar[63]
        getitem_64: "f32[768]" = _foreach_mul_scalar[64]
        getitem_65: "f32[768]" = _foreach_mul_scalar[65]
        getitem_66: "f32[2304, 768]" = _foreach_mul_scalar[66]
        getitem_67: "f32[2304]" = _foreach_mul_scalar[67]
        getitem_68: "f32[768, 768]" = _foreach_mul_scalar[68]
        getitem_69: "f32[768]" = _foreach_mul_scalar[69]
        getitem_70: "f32[3072, 768]" = _foreach_mul_scalar[70]
        getitem_71: "f32[3072]" = _foreach_mul_scalar[71]
        getitem_72: "f32[768, 3072]" = _foreach_mul_scalar[72]
        getitem_73: "f32[768]" = _foreach_mul_scalar[73]
        getitem_74: "f32[768]" = _foreach_mul_scalar[74]
        getitem_75: "f32[768]" = _foreach_mul_scalar[75]
        getitem_76: "f32[768]" = _foreach_mul_scalar[76]
        getitem_77: "f32[768]" = _foreach_mul_scalar[77]
        getitem_78: "f32[2304, 768]" = _foreach_mul_scalar[78]
        getitem_79: "f32[2304]" = _foreach_mul_scalar[79]
        getitem_80: "f32[768, 768]" = _foreach_mul_scalar[80]
        getitem_81: "f32[768]" = _foreach_mul_scalar[81]
        getitem_82: "f32[3072, 768]" = _foreach_mul_scalar[82]
        getitem_83: "f32[3072]" = _foreach_mul_scalar[83]
        getitem_84: "f32[768, 3072]" = _foreach_mul_scalar[84]
        getitem_85: "f32[768]" = _foreach_mul_scalar[85]
        getitem_86: "f32[768]" = _foreach_mul_scalar[86]
        getitem_87: "f32[768]" = _foreach_mul_scalar[87]
        getitem_88: "f32[768]" = _foreach_mul_scalar[88]
        getitem_89: "f32[768]" = _foreach_mul_scalar[89]
        getitem_90: "f32[2304, 768]" = _foreach_mul_scalar[90]
        getitem_91: "f32[2304]" = _foreach_mul_scalar[91]
        getitem_92: "f32[768, 768]" = _foreach_mul_scalar[92]
        getitem_93: "f32[768]" = _foreach_mul_scalar[93]
        getitem_94: "f32[3072, 768]" = _foreach_mul_scalar[94]
        getitem_95: "f32[3072]" = _foreach_mul_scalar[95]
        getitem_96: "f32[768, 3072]" = _foreach_mul_scalar[96]
        getitem_97: "f32[768]" = _foreach_mul_scalar[97]
        getitem_98: "f32[768]" = _foreach_mul_scalar[98]
        getitem_99: "f32[768]" = _foreach_mul_scalar[99]
        getitem_100: "f32[768]" = _foreach_mul_scalar[100]
        getitem_101: "f32[768]" = _foreach_mul_scalar[101]
        getitem_102: "f32[2304, 768]" = _foreach_mul_scalar[102]
        getitem_103: "f32[2304]" = _foreach_mul_scalar[103]
        getitem_104: "f32[768, 768]" = _foreach_mul_scalar[104]
        getitem_105: "f32[768]" = _foreach_mul_scalar[105]
        getitem_106: "f32[3072, 768]" = _foreach_mul_scalar[106]
        getitem_107: "f32[3072]" = _foreach_mul_scalar[107]
        getitem_108: "f32[768, 3072]" = _foreach_mul_scalar[108]
        getitem_109: "f32[768]" = _foreach_mul_scalar[109]
        getitem_110: "f32[768]" = _foreach_mul_scalar[110]
        getitem_111: "f32[768]" = _foreach_mul_scalar[111]
        getitem_112: "f32[768]" = _foreach_mul_scalar[112]
        getitem_113: "f32[768]" = _foreach_mul_scalar[113]
        getitem_114: "f32[2304, 768]" = _foreach_mul_scalar[114]
        getitem_115: "f32[2304]" = _foreach_mul_scalar[115]
        getitem_116: "f32[768, 768]" = _foreach_mul_scalar[116]
        getitem_117: "f32[768]" = _foreach_mul_scalar[117]
        getitem_118: "f32[3072, 768]" = _foreach_mul_scalar[118]
        getitem_119: "f32[3072]" = _foreach_mul_scalar[119]
        getitem_120: "f32[768, 3072]" = _foreach_mul_scalar[120]
        getitem_121: "f32[768]" = _foreach_mul_scalar[121]
        getitem_122: "f32[768]" = _foreach_mul_scalar[122]
        getitem_123: "f32[768]" = _foreach_mul_scalar[123]
        getitem_124: "f32[768]" = _foreach_mul_scalar[124]
        getitem_125: "f32[768]" = _foreach_mul_scalar[125]
        getitem_126: "f32[2304, 768]" = _foreach_mul_scalar[126]
        getitem_127: "f32[2304]" = _foreach_mul_scalar[127]
        getitem_128: "f32[768, 768]" = _foreach_mul_scalar[128]
        getitem_129: "f32[768]" = _foreach_mul_scalar[129]
        getitem_130: "f32[3072, 768]" = _foreach_mul_scalar[130]
        getitem_131: "f32[3072]" = _foreach_mul_scalar[131]
        getitem_132: "f32[768, 3072]" = _foreach_mul_scalar[132]
        getitem_133: "f32[768]" = _foreach_mul_scalar[133]
        getitem_134: "f32[768]" = _foreach_mul_scalar[134]
        getitem_135: "f32[768]" = _foreach_mul_scalar[135]
        getitem_136: "f32[768]" = _foreach_mul_scalar[136]
        getitem_137: "f32[768]" = _foreach_mul_scalar[137]
        getitem_138: "f32[2304, 768]" = _foreach_mul_scalar[138]
        getitem_139: "f32[2304]" = _foreach_mul_scalar[139]
        getitem_140: "f32[768, 768]" = _foreach_mul_scalar[140]
        getitem_141: "f32[768]" = _foreach_mul_scalar[141]
        getitem_142: "f32[3072, 768]" = _foreach_mul_scalar[142]
        getitem_143: "f32[3072]" = _foreach_mul_scalar[143]
        getitem_144: "f32[768, 3072]" = _foreach_mul_scalar[144]
        getitem_145: "f32[768]" = _foreach_mul_scalar[145]
        getitem_146: "f32[768]" = _foreach_mul_scalar[146]
        getitem_147: "f32[768]" = _foreach_mul_scalar[147]
        getitem_148: "f32[768]" = _foreach_mul_scalar[148]
        getitem_149: "f32[768]" = _foreach_mul_scalar[149]
        getitem_150: "f32[768]" = _foreach_mul_scalar[150]
        getitem_151: "f32[768]" = _foreach_mul_scalar[151]
        getitem_152: "f32[77, 512]" = _foreach_mul_scalar[152]
        getitem_153: "f32[49408, 512]" = _foreach_mul_scalar[153]
        getitem_154: "f32[1536, 512]" = _foreach_mul_scalar[154]
        getitem_155: "f32[1536]" = _foreach_mul_scalar[155]
        getitem_156: "f32[512, 512]" = _foreach_mul_scalar[156]
        getitem_157: "f32[512]" = _foreach_mul_scalar[157]
        getitem_158: "f32[2048, 512]" = _foreach_mul_scalar[158]
        getitem_159: "f32[2048]" = _foreach_mul_scalar[159]
        getitem_160: "f32[512, 2048]" = _foreach_mul_scalar[160]
        getitem_161: "f32[512]" = _foreach_mul_scalar[161]
        getitem_162: "f32[512]" = _foreach_mul_scalar[162]
        getitem_163: "f32[512]" = _foreach_mul_scalar[163]
        getitem_164: "f32[512]" = _foreach_mul_scalar[164]
        getitem_165: "f32[512]" = _foreach_mul_scalar[165]
        getitem_166: "f32[1536, 512]" = _foreach_mul_scalar[166]
        getitem_167: "f32[1536]" = _foreach_mul_scalar[167]
        getitem_168: "f32[512, 512]" = _foreach_mul_scalar[168]
        getitem_169: "f32[512]" = _foreach_mul_scalar[169]
        getitem_170: "f32[2048, 512]" = _foreach_mul_scalar[170]
        getitem_171: "f32[2048]" = _foreach_mul_scalar[171]
        getitem_172: "f32[512, 2048]" = _foreach_mul_scalar[172]
        getitem_173: "f32[512]" = _foreach_mul_scalar[173]
        getitem_174: "f32[512]" = _foreach_mul_scalar[174]
        getitem_175: "f32[512]" = _foreach_mul_scalar[175]
        getitem_176: "f32[512]" = _foreach_mul_scalar[176]
        getitem_177: "f32[512]" = _foreach_mul_scalar[177]
        getitem_178: "f32[1536, 512]" = _foreach_mul_scalar[178]
        getitem_179: "f32[1536]" = _foreach_mul_scalar[179]
        getitem_180: "f32[512, 512]" = _foreach_mul_scalar[180]
        getitem_181: "f32[512]" = _foreach_mul_scalar[181]
        getitem_182: "f32[2048, 512]" = _foreach_mul_scalar[182]
        getitem_183: "f32[2048]" = _foreach_mul_scalar[183]
        getitem_184: "f32[512, 2048]" = _foreach_mul_scalar[184]
        getitem_185: "f32[512]" = _foreach_mul_scalar[185]
        getitem_186: "f32[512]" = _foreach_mul_scalar[186]
        getitem_187: "f32[512]" = _foreach_mul_scalar[187]
        getitem_188: "f32[512]" = _foreach_mul_scalar[188]
        getitem_189: "f32[512]" = _foreach_mul_scalar[189]
        getitem_190: "f32[1536, 512]" = _foreach_mul_scalar[190]
        getitem_191: "f32[1536]" = _foreach_mul_scalar[191]
        getitem_192: "f32[512, 512]" = _foreach_mul_scalar[192]
        getitem_193: "f32[512]" = _foreach_mul_scalar[193]
        getitem_194: "f32[2048, 512]" = _foreach_mul_scalar[194]
        getitem_195: "f32[2048]" = _foreach_mul_scalar[195]
        getitem_196: "f32[512, 2048]" = _foreach_mul_scalar[196]
        getitem_197: "f32[512]" = _foreach_mul_scalar[197]
        getitem_198: "f32[512]" = _foreach_mul_scalar[198]
        getitem_199: "f32[512]" = _foreach_mul_scalar[199]
        getitem_200: "f32[512]" = _foreach_mul_scalar[200]
        getitem_201: "f32[512]" = _foreach_mul_scalar[201]
        getitem_202: "f32[1536, 512]" = _foreach_mul_scalar[202]
        getitem_203: "f32[1536]" = _foreach_mul_scalar[203]
        getitem_204: "f32[512, 512]" = _foreach_mul_scalar[204]
        getitem_205: "f32[512]" = _foreach_mul_scalar[205]
        getitem_206: "f32[2048, 512]" = _foreach_mul_scalar[206]
        getitem_207: "f32[2048]" = _foreach_mul_scalar[207]
        getitem_208: "f32[512, 2048]" = _foreach_mul_scalar[208]
        getitem_209: "f32[512]" = _foreach_mul_scalar[209]
        getitem_210: "f32[512]" = _foreach_mul_scalar[210]
        getitem_211: "f32[512]" = _foreach_mul_scalar[211]
        getitem_212: "f32[512]" = _foreach_mul_scalar[212]
        getitem_213: "f32[512]" = _foreach_mul_scalar[213]
        getitem_214: "f32[1536, 512]" = _foreach_mul_scalar[214]
        getitem_215: "f32[1536]" = _foreach_mul_scalar[215]
        getitem_216: "f32[512, 512]" = _foreach_mul_scalar[216]
        getitem_217: "f32[512]" = _foreach_mul_scalar[217]
        getitem_218: "f32[2048, 512]" = _foreach_mul_scalar[218]
        getitem_219: "f32[2048]" = _foreach_mul_scalar[219]
        getitem_220: "f32[512, 2048]" = _foreach_mul_scalar[220]
        getitem_221: "f32[512]" = _foreach_mul_scalar[221]
        getitem_222: "f32[512]" = _foreach_mul_scalar[222]
        getitem_223: "f32[512]" = _foreach_mul_scalar[223]
        getitem_224: "f32[512]" = _foreach_mul_scalar[224]
        getitem_225: "f32[512]" = _foreach_mul_scalar[225]
        getitem_226: "f32[1536, 512]" = _foreach_mul_scalar[226]
        getitem_227: "f32[1536]" = _foreach_mul_scalar[227]
        getitem_228: "f32[512, 512]" = _foreach_mul_scalar[228]
        getitem_229: "f32[512]" = _foreach_mul_scalar[229]
        getitem_230: "f32[2048, 512]" = _foreach_mul_scalar[230]
        getitem_231: "f32[2048]" = _foreach_mul_scalar[231]
        getitem_232: "f32[512, 2048]" = _foreach_mul_scalar[232]
        getitem_233: "f32[512]" = _foreach_mul_scalar[233]
        getitem_234: "f32[512]" = _foreach_mul_scalar[234]
        getitem_235: "f32[512]" = _foreach_mul_scalar[235]
        getitem_236: "f32[512]" = _foreach_mul_scalar[236]
        getitem_237: "f32[512]" = _foreach_mul_scalar[237]
        getitem_238: "f32[1536, 512]" = _foreach_mul_scalar[238]
        getitem_239: "f32[1536]" = _foreach_mul_scalar[239]
        getitem_240: "f32[512, 512]" = _foreach_mul_scalar[240]
        getitem_241: "f32[512]" = _foreach_mul_scalar[241]
        getitem_242: "f32[2048, 512]" = _foreach_mul_scalar[242]
        getitem_243: "f32[2048]" = _foreach_mul_scalar[243]
        getitem_244: "f32[512, 2048]" = _foreach_mul_scalar[244]
        getitem_245: "f32[512]" = _foreach_mul_scalar[245]
        getitem_246: "f32[512]" = _foreach_mul_scalar[246]
        getitem_247: "f32[512]" = _foreach_mul_scalar[247]
        getitem_248: "f32[512]" = _foreach_mul_scalar[248]
        getitem_249: "f32[512]" = _foreach_mul_scalar[249]
        getitem_250: "f32[1536, 512]" = _foreach_mul_scalar[250]
        getitem_251: "f32[1536]" = _foreach_mul_scalar[251]
        getitem_252: "f32[512, 512]" = _foreach_mul_scalar[252]
        getitem_253: "f32[512]" = _foreach_mul_scalar[253]
        getitem_254: "f32[2048, 512]" = _foreach_mul_scalar[254]
        getitem_255: "f32[2048]" = _foreach_mul_scalar[255]
        getitem_256: "f32[512, 2048]" = _foreach_mul_scalar[256]
        getitem_257: "f32[512]" = _foreach_mul_scalar[257]
        getitem_258: "f32[512]" = _foreach_mul_scalar[258]
        getitem_259: "f32[512]" = _foreach_mul_scalar[259]
        getitem_260: "f32[512]" = _foreach_mul_scalar[260]
        getitem_261: "f32[512]" = _foreach_mul_scalar[261]
        getitem_262: "f32[1536, 512]" = _foreach_mul_scalar[262]
        getitem_263: "f32[1536]" = _foreach_mul_scalar[263]
        getitem_264: "f32[512, 512]" = _foreach_mul_scalar[264]
        getitem_265: "f32[512]" = _foreach_mul_scalar[265]
        getitem_266: "f32[2048, 512]" = _foreach_mul_scalar[266]
        getitem_267: "f32[2048]" = _foreach_mul_scalar[267]
        getitem_268: "f32[512, 2048]" = _foreach_mul_scalar[268]
        getitem_269: "f32[512]" = _foreach_mul_scalar[269]
        getitem_270: "f32[512]" = _foreach_mul_scalar[270]
        getitem_271: "f32[512]" = _foreach_mul_scalar[271]
        getitem_272: "f32[512]" = _foreach_mul_scalar[272]
        getitem_273: "f32[512]" = _foreach_mul_scalar[273]
        getitem_274: "f32[1536, 512]" = _foreach_mul_scalar[274]
        getitem_275: "f32[1536]" = _foreach_mul_scalar[275]
        getitem_276: "f32[512, 512]" = _foreach_mul_scalar[276]
        getitem_277: "f32[512]" = _foreach_mul_scalar[277]
        getitem_278: "f32[2048, 512]" = _foreach_mul_scalar[278]
        getitem_279: "f32[2048]" = _foreach_mul_scalar[279]
        getitem_280: "f32[512, 2048]" = _foreach_mul_scalar[280]
        getitem_281: "f32[512]" = _foreach_mul_scalar[281]
        getitem_282: "f32[512]" = _foreach_mul_scalar[282]
        getitem_283: "f32[512]" = _foreach_mul_scalar[283]
        getitem_284: "f32[512]" = _foreach_mul_scalar[284]
        getitem_285: "f32[512]" = _foreach_mul_scalar[285]
        getitem_286: "f32[1536, 512]" = _foreach_mul_scalar[286]
        getitem_287: "f32[1536]" = _foreach_mul_scalar[287]
        getitem_288: "f32[512, 512]" = _foreach_mul_scalar[288]
        getitem_289: "f32[512]" = _foreach_mul_scalar[289]
        getitem_290: "f32[2048, 512]" = _foreach_mul_scalar[290]
        getitem_291: "f32[2048]" = _foreach_mul_scalar[291]
        getitem_292: "f32[512, 2048]" = _foreach_mul_scalar[292]
        getitem_293: "f32[512]" = _foreach_mul_scalar[293]
        getitem_294: "f32[512]" = _foreach_mul_scalar[294]
        getitem_295: "f32[512]" = _foreach_mul_scalar[295]
        getitem_296: "f32[512]" = _foreach_mul_scalar[296]
        getitem_297: "f32[512]" = _foreach_mul_scalar[297]
        getitem_298: "f32[512]" = _foreach_mul_scalar[298]
        getitem_299: "f32[512]" = _foreach_mul_scalar[299]
        getitem_300: "f32[512, 512]" = _foreach_mul_scalar[300];  _foreach_mul_scalar = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106], 1);  getitem_1806 = getitem_1807 = getitem_1808 = getitem_1809 = getitem_1810 = getitem_1811 = getitem_1812 = getitem_1813 = getitem_1814 = getitem_1815 = getitem_1816 = getitem_1817 = getitem_1818 = getitem_1819 = getitem_1820 = getitem_1821 = getitem_1822 = getitem_1823 = getitem_1824 = getitem_1825 = getitem_1826 = getitem_1827 = getitem_1828 = getitem_1829 = getitem_1830 = getitem_1831 = getitem_1832 = getitem_1833 = getitem_1834 = getitem_1835 = getitem_1836 = getitem_1837 = getitem_1838 = getitem_1839 = getitem_1840 = getitem_1841 = getitem_1842 = getitem_1843 = getitem_1844 = getitem_1845 = getitem_1846 = getitem_1847 = getitem_1848 = getitem_1849 = getitem_1850 = getitem_1851 = getitem_1852 = getitem_1853 = getitem_1854 = getitem_1855 = getitem_1856 = getitem_1857 = getitem_1858 = getitem_1859 = getitem_1860 = getitem_1861 = getitem_1862 = getitem_1863 = getitem_1864 = getitem_1865 = getitem_1866 = getitem_1867 = getitem_1868 = getitem_1869 = getitem_1870 = getitem_1871 = getitem_1872 = getitem_1873 = getitem_1874 = getitem_1875 = getitem_1876 = getitem_1877 = getitem_1878 = getitem_1879 = getitem_1880 = getitem_1881 = getitem_1882 = getitem_1883 = getitem_1884 = getitem_1885 = getitem_1886 = getitem_1887 = getitem_1888 = getitem_1889 = getitem_1890 = getitem_1891 = getitem_1892 = getitem_1893 = getitem_1894 = getitem_1895 = getitem_1896 = getitem_1897 = getitem_1898 = getitem_1899 = getitem_1900 = getitem_1901 = getitem_1902 = getitem_1903 = getitem_1904 = getitem_1905 = getitem_1906 = getitem_1907 = getitem_1908 = getitem_1909 = getitem_1910 = getitem_1911 = getitem_1912 = getitem_1913 = getitem_1914 = getitem_1915 = getitem_1916 = getitem_1917 = getitem_1918 = getitem_1919 = getitem_1920 = getitem_1921 = getitem_1922 = getitem_1923 = getitem_1924 = getitem_1925 = getitem_1926 = getitem_1927 = getitem_1928 = getitem_1929 = getitem_1930 = getitem_1931 = getitem_1932 = getitem_1933 = getitem_1934 = getitem_1935 = getitem_1936 = getitem_1937 = getitem_1938 = getitem_1939 = getitem_1940 = getitem_1941 = getitem_1942 = getitem_1943 = getitem_1944 = getitem_1945 = getitem_1946 = getitem_1947 = getitem_1948 = getitem_1949 = getitem_1950 = getitem_1951 = getitem_1952 = getitem_1953 = getitem_1954 = getitem_1955 = getitem_1956 = getitem_1957 = getitem_1958 = getitem_1959 = getitem_1960 = getitem_1961 = getitem_1962 = getitem_1963 = getitem_1964 = getitem_1965 = getitem_1966 = getitem_1967 = getitem_1968 = getitem_1969 = getitem_1970 = getitem_1971 = getitem_1972 = getitem_1973 = getitem_1974 = getitem_1975 = getitem_1976 = getitem_1977 = getitem_1978 = getitem_1979 = getitem_1980 = getitem_1981 = getitem_1982 = getitem_1983 = getitem_1984 = getitem_1985 = getitem_1986 = getitem_1987 = getitem_1988 = getitem_1989 = getitem_1990 = getitem_1991 = getitem_1992 = getitem_1993 = getitem_1994 = getitem_1995 = getitem_1996 = getitem_1997 = getitem_1998 = getitem_1999 = getitem_2000 = getitem_2001 = getitem_2002 = getitem_2003 = getitem_2004 = getitem_2005 = getitem_2006 = getitem_2007 = getitem_2008 = getitem_2009 = getitem_2010 = getitem_2011 = getitem_2012 = getitem_2013 = getitem_2014 = getitem_2015 = getitem_2016 = getitem_2017 = getitem_2018 = getitem_2019 = getitem_2020 = getitem_2021 = getitem_2022 = getitem_2023 = getitem_2024 = getitem_2025 = getitem_2026 = getitem_2027 = getitem_2028 = getitem_2029 = getitem_2030 = getitem_2031 = getitem_2032 = getitem_2033 = getitem_2034 = getitem_2035 = getitem_2036 = getitem_2037 = getitem_2038 = getitem_2039 = getitem_2040 = getitem_2041 = getitem_2042 = getitem_2043 = getitem_2044 = getitem_2045 = getitem_2046 = getitem_2047 = getitem_2048 = getitem_2049 = getitem_2050 = getitem_2051 = getitem_2052 = getitem_2053 = getitem_2054 = getitem_2055 = getitem_2056 = getitem_2057 = getitem_2058 = getitem_2059 = getitem_2060 = getitem_2061 = getitem_2062 = getitem_2063 = getitem_2064 = getitem_2065 = getitem_2066 = getitem_2067 = getitem_2068 = getitem_2069 = getitem_2070 = getitem_2071 = getitem_2072 = getitem_2073 = getitem_2074 = getitem_2075 = getitem_2076 = getitem_2077 = getitem_2078 = getitem_2079 = getitem_2080 = getitem_2081 = getitem_2082 = getitem_2083 = getitem_2084 = getitem_2085 = getitem_2086 = getitem_2087 = getitem_2088 = getitem_2089 = getitem_2090 = getitem_2091 = getitem_2092 = getitem_2093 = getitem_2094 = getitem_2095 = getitem_2096 = getitem_2097 = getitem_2098 = getitem_2099 = getitem_2100 = getitem_2101 = getitem_2102 = getitem_2103 = getitem_2104 = getitem_2105 = getitem_2106 = None
        getitem_2107: "f32[]" = _foreach_sub_scalar[0]
        getitem_2108: "f32[]" = _foreach_sub_scalar[1]
        getitem_2109: "f32[]" = _foreach_sub_scalar[2]
        getitem_2110: "f32[]" = _foreach_sub_scalar[3]
        getitem_2111: "f32[]" = _foreach_sub_scalar[4]
        getitem_2112: "f32[]" = _foreach_sub_scalar[5]
        getitem_2113: "f32[]" = _foreach_sub_scalar[6]
        getitem_2114: "f32[]" = _foreach_sub_scalar[7]
        getitem_2115: "f32[]" = _foreach_sub_scalar[8]
        getitem_2116: "f32[]" = _foreach_sub_scalar[9]
        getitem_2117: "f32[]" = _foreach_sub_scalar[10]
        getitem_2118: "f32[]" = _foreach_sub_scalar[11]
        getitem_2119: "f32[]" = _foreach_sub_scalar[12]
        getitem_2120: "f32[]" = _foreach_sub_scalar[13]
        getitem_2121: "f32[]" = _foreach_sub_scalar[14]
        getitem_2122: "f32[]" = _foreach_sub_scalar[15]
        getitem_2123: "f32[]" = _foreach_sub_scalar[16]
        getitem_2124: "f32[]" = _foreach_sub_scalar[17]
        getitem_2125: "f32[]" = _foreach_sub_scalar[18]
        getitem_2126: "f32[]" = _foreach_sub_scalar[19]
        getitem_2127: "f32[]" = _foreach_sub_scalar[20]
        getitem_2128: "f32[]" = _foreach_sub_scalar[21]
        getitem_2129: "f32[]" = _foreach_sub_scalar[22]
        getitem_2130: "f32[]" = _foreach_sub_scalar[23]
        getitem_2131: "f32[]" = _foreach_sub_scalar[24]
        getitem_2132: "f32[]" = _foreach_sub_scalar[25]
        getitem_2133: "f32[]" = _foreach_sub_scalar[26]
        getitem_2134: "f32[]" = _foreach_sub_scalar[27]
        getitem_2135: "f32[]" = _foreach_sub_scalar[28]
        getitem_2136: "f32[]" = _foreach_sub_scalar[29]
        getitem_2137: "f32[]" = _foreach_sub_scalar[30]
        getitem_2138: "f32[]" = _foreach_sub_scalar[31]
        getitem_2139: "f32[]" = _foreach_sub_scalar[32]
        getitem_2140: "f32[]" = _foreach_sub_scalar[33]
        getitem_2141: "f32[]" = _foreach_sub_scalar[34]
        getitem_2142: "f32[]" = _foreach_sub_scalar[35]
        getitem_2143: "f32[]" = _foreach_sub_scalar[36]
        getitem_2144: "f32[]" = _foreach_sub_scalar[37]
        getitem_2145: "f32[]" = _foreach_sub_scalar[38]
        getitem_2146: "f32[]" = _foreach_sub_scalar[39]
        getitem_2147: "f32[]" = _foreach_sub_scalar[40]
        getitem_2148: "f32[]" = _foreach_sub_scalar[41]
        getitem_2149: "f32[]" = _foreach_sub_scalar[42]
        getitem_2150: "f32[]" = _foreach_sub_scalar[43]
        getitem_2151: "f32[]" = _foreach_sub_scalar[44]
        getitem_2152: "f32[]" = _foreach_sub_scalar[45]
        getitem_2153: "f32[]" = _foreach_sub_scalar[46]
        getitem_2154: "f32[]" = _foreach_sub_scalar[47]
        getitem_2155: "f32[]" = _foreach_sub_scalar[48]
        getitem_2156: "f32[]" = _foreach_sub_scalar[49]
        getitem_2157: "f32[]" = _foreach_sub_scalar[50]
        getitem_2158: "f32[]" = _foreach_sub_scalar[51]
        getitem_2159: "f32[]" = _foreach_sub_scalar[52]
        getitem_2160: "f32[]" = _foreach_sub_scalar[53]
        getitem_2161: "f32[]" = _foreach_sub_scalar[54]
        getitem_2162: "f32[]" = _foreach_sub_scalar[55]
        getitem_2163: "f32[]" = _foreach_sub_scalar[56]
        getitem_2164: "f32[]" = _foreach_sub_scalar[57]
        getitem_2165: "f32[]" = _foreach_sub_scalar[58]
        getitem_2166: "f32[]" = _foreach_sub_scalar[59]
        getitem_2167: "f32[]" = _foreach_sub_scalar[60]
        getitem_2168: "f32[]" = _foreach_sub_scalar[61]
        getitem_2169: "f32[]" = _foreach_sub_scalar[62]
        getitem_2170: "f32[]" = _foreach_sub_scalar[63]
        getitem_2171: "f32[]" = _foreach_sub_scalar[64]
        getitem_2172: "f32[]" = _foreach_sub_scalar[65]
        getitem_2173: "f32[]" = _foreach_sub_scalar[66]
        getitem_2174: "f32[]" = _foreach_sub_scalar[67]
        getitem_2175: "f32[]" = _foreach_sub_scalar[68]
        getitem_2176: "f32[]" = _foreach_sub_scalar[69]
        getitem_2177: "f32[]" = _foreach_sub_scalar[70]
        getitem_2178: "f32[]" = _foreach_sub_scalar[71]
        getitem_2179: "f32[]" = _foreach_sub_scalar[72]
        getitem_2180: "f32[]" = _foreach_sub_scalar[73]
        getitem_2181: "f32[]" = _foreach_sub_scalar[74]
        getitem_2182: "f32[]" = _foreach_sub_scalar[75]
        getitem_2183: "f32[]" = _foreach_sub_scalar[76]
        getitem_2184: "f32[]" = _foreach_sub_scalar[77]
        getitem_2185: "f32[]" = _foreach_sub_scalar[78]
        getitem_2186: "f32[]" = _foreach_sub_scalar[79]
        getitem_2187: "f32[]" = _foreach_sub_scalar[80]
        getitem_2188: "f32[]" = _foreach_sub_scalar[81]
        getitem_2189: "f32[]" = _foreach_sub_scalar[82]
        getitem_2190: "f32[]" = _foreach_sub_scalar[83]
        getitem_2191: "f32[]" = _foreach_sub_scalar[84]
        getitem_2192: "f32[]" = _foreach_sub_scalar[85]
        getitem_2193: "f32[]" = _foreach_sub_scalar[86]
        getitem_2194: "f32[]" = _foreach_sub_scalar[87]
        getitem_2195: "f32[]" = _foreach_sub_scalar[88]
        getitem_2196: "f32[]" = _foreach_sub_scalar[89]
        getitem_2197: "f32[]" = _foreach_sub_scalar[90]
        getitem_2198: "f32[]" = _foreach_sub_scalar[91]
        getitem_2199: "f32[]" = _foreach_sub_scalar[92]
        getitem_2200: "f32[]" = _foreach_sub_scalar[93]
        getitem_2201: "f32[]" = _foreach_sub_scalar[94]
        getitem_2202: "f32[]" = _foreach_sub_scalar[95]
        getitem_2203: "f32[]" = _foreach_sub_scalar[96]
        getitem_2204: "f32[]" = _foreach_sub_scalar[97]
        getitem_2205: "f32[]" = _foreach_sub_scalar[98]
        getitem_2206: "f32[]" = _foreach_sub_scalar[99]
        getitem_2207: "f32[]" = _foreach_sub_scalar[100]
        getitem_2208: "f32[]" = _foreach_sub_scalar[101]
        getitem_2209: "f32[]" = _foreach_sub_scalar[102]
        getitem_2210: "f32[]" = _foreach_sub_scalar[103]
        getitem_2211: "f32[]" = _foreach_sub_scalar[104]
        getitem_2212: "f32[]" = _foreach_sub_scalar[105]
        getitem_2213: "f32[]" = _foreach_sub_scalar[106]
        getitem_2214: "f32[]" = _foreach_sub_scalar[107]
        getitem_2215: "f32[]" = _foreach_sub_scalar[108]
        getitem_2216: "f32[]" = _foreach_sub_scalar[109]
        getitem_2217: "f32[]" = _foreach_sub_scalar[110]
        getitem_2218: "f32[]" = _foreach_sub_scalar[111]
        getitem_2219: "f32[]" = _foreach_sub_scalar[112]
        getitem_2220: "f32[]" = _foreach_sub_scalar[113]
        getitem_2221: "f32[]" = _foreach_sub_scalar[114]
        getitem_2222: "f32[]" = _foreach_sub_scalar[115]
        getitem_2223: "f32[]" = _foreach_sub_scalar[116]
        getitem_2224: "f32[]" = _foreach_sub_scalar[117]
        getitem_2225: "f32[]" = _foreach_sub_scalar[118]
        getitem_2226: "f32[]" = _foreach_sub_scalar[119]
        getitem_2227: "f32[]" = _foreach_sub_scalar[120]
        getitem_2228: "f32[]" = _foreach_sub_scalar[121]
        getitem_2229: "f32[]" = _foreach_sub_scalar[122]
        getitem_2230: "f32[]" = _foreach_sub_scalar[123]
        getitem_2231: "f32[]" = _foreach_sub_scalar[124]
        getitem_2232: "f32[]" = _foreach_sub_scalar[125]
        getitem_2233: "f32[]" = _foreach_sub_scalar[126]
        getitem_2234: "f32[]" = _foreach_sub_scalar[127]
        getitem_2235: "f32[]" = _foreach_sub_scalar[128]
        getitem_2236: "f32[]" = _foreach_sub_scalar[129]
        getitem_2237: "f32[]" = _foreach_sub_scalar[130]
        getitem_2238: "f32[]" = _foreach_sub_scalar[131]
        getitem_2239: "f32[]" = _foreach_sub_scalar[132]
        getitem_2240: "f32[]" = _foreach_sub_scalar[133]
        getitem_2241: "f32[]" = _foreach_sub_scalar[134]
        getitem_2242: "f32[]" = _foreach_sub_scalar[135]
        getitem_2243: "f32[]" = _foreach_sub_scalar[136]
        getitem_2244: "f32[]" = _foreach_sub_scalar[137]
        getitem_2245: "f32[]" = _foreach_sub_scalar[138]
        getitem_2246: "f32[]" = _foreach_sub_scalar[139]
        getitem_2247: "f32[]" = _foreach_sub_scalar[140]
        getitem_2248: "f32[]" = _foreach_sub_scalar[141]
        getitem_2249: "f32[]" = _foreach_sub_scalar[142]
        getitem_2250: "f32[]" = _foreach_sub_scalar[143]
        getitem_2251: "f32[]" = _foreach_sub_scalar[144]
        getitem_2252: "f32[]" = _foreach_sub_scalar[145]
        getitem_2253: "f32[]" = _foreach_sub_scalar[146]
        getitem_2254: "f32[]" = _foreach_sub_scalar[147]
        getitem_2255: "f32[]" = _foreach_sub_scalar[148]
        getitem_2256: "f32[]" = _foreach_sub_scalar[149]
        getitem_2257: "f32[]" = _foreach_sub_scalar[150]
        getitem_2258: "f32[]" = _foreach_sub_scalar[151]
        getitem_2259: "f32[]" = _foreach_sub_scalar[152]
        getitem_2260: "f32[]" = _foreach_sub_scalar[153]
        getitem_2261: "f32[]" = _foreach_sub_scalar[154]
        getitem_2262: "f32[]" = _foreach_sub_scalar[155]
        getitem_2263: "f32[]" = _foreach_sub_scalar[156]
        getitem_2264: "f32[]" = _foreach_sub_scalar[157]
        getitem_2265: "f32[]" = _foreach_sub_scalar[158]
        getitem_2266: "f32[]" = _foreach_sub_scalar[159]
        getitem_2267: "f32[]" = _foreach_sub_scalar[160]
        getitem_2268: "f32[]" = _foreach_sub_scalar[161]
        getitem_2269: "f32[]" = _foreach_sub_scalar[162]
        getitem_2270: "f32[]" = _foreach_sub_scalar[163]
        getitem_2271: "f32[]" = _foreach_sub_scalar[164]
        getitem_2272: "f32[]" = _foreach_sub_scalar[165]
        getitem_2273: "f32[]" = _foreach_sub_scalar[166]
        getitem_2274: "f32[]" = _foreach_sub_scalar[167]
        getitem_2275: "f32[]" = _foreach_sub_scalar[168]
        getitem_2276: "f32[]" = _foreach_sub_scalar[169]
        getitem_2277: "f32[]" = _foreach_sub_scalar[170]
        getitem_2278: "f32[]" = _foreach_sub_scalar[171]
        getitem_2279: "f32[]" = _foreach_sub_scalar[172]
        getitem_2280: "f32[]" = _foreach_sub_scalar[173]
        getitem_2281: "f32[]" = _foreach_sub_scalar[174]
        getitem_2282: "f32[]" = _foreach_sub_scalar[175]
        getitem_2283: "f32[]" = _foreach_sub_scalar[176]
        getitem_2284: "f32[]" = _foreach_sub_scalar[177]
        getitem_2285: "f32[]" = _foreach_sub_scalar[178]
        getitem_2286: "f32[]" = _foreach_sub_scalar[179]
        getitem_2287: "f32[]" = _foreach_sub_scalar[180]
        getitem_2288: "f32[]" = _foreach_sub_scalar[181]
        getitem_2289: "f32[]" = _foreach_sub_scalar[182]
        getitem_2290: "f32[]" = _foreach_sub_scalar[183]
        getitem_2291: "f32[]" = _foreach_sub_scalar[184]
        getitem_2292: "f32[]" = _foreach_sub_scalar[185]
        getitem_2293: "f32[]" = _foreach_sub_scalar[186]
        getitem_2294: "f32[]" = _foreach_sub_scalar[187]
        getitem_2295: "f32[]" = _foreach_sub_scalar[188]
        getitem_2296: "f32[]" = _foreach_sub_scalar[189]
        getitem_2297: "f32[]" = _foreach_sub_scalar[190]
        getitem_2298: "f32[]" = _foreach_sub_scalar[191]
        getitem_2299: "f32[]" = _foreach_sub_scalar[192]
        getitem_2300: "f32[]" = _foreach_sub_scalar[193]
        getitem_2301: "f32[]" = _foreach_sub_scalar[194]
        getitem_2302: "f32[]" = _foreach_sub_scalar[195]
        getitem_2303: "f32[]" = _foreach_sub_scalar[196]
        getitem_2304: "f32[]" = _foreach_sub_scalar[197]
        getitem_2305: "f32[]" = _foreach_sub_scalar[198]
        getitem_2306: "f32[]" = _foreach_sub_scalar[199]
        getitem_2307: "f32[]" = _foreach_sub_scalar[200]
        getitem_2308: "f32[]" = _foreach_sub_scalar[201]
        getitem_2309: "f32[]" = _foreach_sub_scalar[202]
        getitem_2310: "f32[]" = _foreach_sub_scalar[203]
        getitem_2311: "f32[]" = _foreach_sub_scalar[204]
        getitem_2312: "f32[]" = _foreach_sub_scalar[205]
        getitem_2313: "f32[]" = _foreach_sub_scalar[206]
        getitem_2314: "f32[]" = _foreach_sub_scalar[207]
        getitem_2315: "f32[]" = _foreach_sub_scalar[208]
        getitem_2316: "f32[]" = _foreach_sub_scalar[209]
        getitem_2317: "f32[]" = _foreach_sub_scalar[210]
        getitem_2318: "f32[]" = _foreach_sub_scalar[211]
        getitem_2319: "f32[]" = _foreach_sub_scalar[212]
        getitem_2320: "f32[]" = _foreach_sub_scalar[213]
        getitem_2321: "f32[]" = _foreach_sub_scalar[214]
        getitem_2322: "f32[]" = _foreach_sub_scalar[215]
        getitem_2323: "f32[]" = _foreach_sub_scalar[216]
        getitem_2324: "f32[]" = _foreach_sub_scalar[217]
        getitem_2325: "f32[]" = _foreach_sub_scalar[218]
        getitem_2326: "f32[]" = _foreach_sub_scalar[219]
        getitem_2327: "f32[]" = _foreach_sub_scalar[220]
        getitem_2328: "f32[]" = _foreach_sub_scalar[221]
        getitem_2329: "f32[]" = _foreach_sub_scalar[222]
        getitem_2330: "f32[]" = _foreach_sub_scalar[223]
        getitem_2331: "f32[]" = _foreach_sub_scalar[224]
        getitem_2332: "f32[]" = _foreach_sub_scalar[225]
        getitem_2333: "f32[]" = _foreach_sub_scalar[226]
        getitem_2334: "f32[]" = _foreach_sub_scalar[227]
        getitem_2335: "f32[]" = _foreach_sub_scalar[228]
        getitem_2336: "f32[]" = _foreach_sub_scalar[229]
        getitem_2337: "f32[]" = _foreach_sub_scalar[230]
        getitem_2338: "f32[]" = _foreach_sub_scalar[231]
        getitem_2339: "f32[]" = _foreach_sub_scalar[232]
        getitem_2340: "f32[]" = _foreach_sub_scalar[233]
        getitem_2341: "f32[]" = _foreach_sub_scalar[234]
        getitem_2342: "f32[]" = _foreach_sub_scalar[235]
        getitem_2343: "f32[]" = _foreach_sub_scalar[236]
        getitem_2344: "f32[]" = _foreach_sub_scalar[237]
        getitem_2345: "f32[]" = _foreach_sub_scalar[238]
        getitem_2346: "f32[]" = _foreach_sub_scalar[239]
        getitem_2347: "f32[]" = _foreach_sub_scalar[240]
        getitem_2348: "f32[]" = _foreach_sub_scalar[241]
        getitem_2349: "f32[]" = _foreach_sub_scalar[242]
        getitem_2350: "f32[]" = _foreach_sub_scalar[243]
        getitem_2351: "f32[]" = _foreach_sub_scalar[244]
        getitem_2352: "f32[]" = _foreach_sub_scalar[245]
        getitem_2353: "f32[]" = _foreach_sub_scalar[246]
        getitem_2354: "f32[]" = _foreach_sub_scalar[247]
        getitem_2355: "f32[]" = _foreach_sub_scalar[248]
        getitem_2356: "f32[]" = _foreach_sub_scalar[249]
        getitem_2357: "f32[]" = _foreach_sub_scalar[250]
        getitem_2358: "f32[]" = _foreach_sub_scalar[251]
        getitem_2359: "f32[]" = _foreach_sub_scalar[252]
        getitem_2360: "f32[]" = _foreach_sub_scalar[253]
        getitem_2361: "f32[]" = _foreach_sub_scalar[254]
        getitem_2362: "f32[]" = _foreach_sub_scalar[255]
        getitem_2363: "f32[]" = _foreach_sub_scalar[256]
        getitem_2364: "f32[]" = _foreach_sub_scalar[257]
        getitem_2365: "f32[]" = _foreach_sub_scalar[258]
        getitem_2366: "f32[]" = _foreach_sub_scalar[259]
        getitem_2367: "f32[]" = _foreach_sub_scalar[260]
        getitem_2368: "f32[]" = _foreach_sub_scalar[261]
        getitem_2369: "f32[]" = _foreach_sub_scalar[262]
        getitem_2370: "f32[]" = _foreach_sub_scalar[263]
        getitem_2371: "f32[]" = _foreach_sub_scalar[264]
        getitem_2372: "f32[]" = _foreach_sub_scalar[265]
        getitem_2373: "f32[]" = _foreach_sub_scalar[266]
        getitem_2374: "f32[]" = _foreach_sub_scalar[267]
        getitem_2375: "f32[]" = _foreach_sub_scalar[268]
        getitem_2376: "f32[]" = _foreach_sub_scalar[269]
        getitem_2377: "f32[]" = _foreach_sub_scalar[270]
        getitem_2378: "f32[]" = _foreach_sub_scalar[271]
        getitem_2379: "f32[]" = _foreach_sub_scalar[272]
        getitem_2380: "f32[]" = _foreach_sub_scalar[273]
        getitem_2381: "f32[]" = _foreach_sub_scalar[274]
        getitem_2382: "f32[]" = _foreach_sub_scalar[275]
        getitem_2383: "f32[]" = _foreach_sub_scalar[276]
        getitem_2384: "f32[]" = _foreach_sub_scalar[277]
        getitem_2385: "f32[]" = _foreach_sub_scalar[278]
        getitem_2386: "f32[]" = _foreach_sub_scalar[279]
        getitem_2387: "f32[]" = _foreach_sub_scalar[280]
        getitem_2388: "f32[]" = _foreach_sub_scalar[281]
        getitem_2389: "f32[]" = _foreach_sub_scalar[282]
        getitem_2390: "f32[]" = _foreach_sub_scalar[283]
        getitem_2391: "f32[]" = _foreach_sub_scalar[284]
        getitem_2392: "f32[]" = _foreach_sub_scalar[285]
        getitem_2393: "f32[]" = _foreach_sub_scalar[286]
        getitem_2394: "f32[]" = _foreach_sub_scalar[287]
        getitem_2395: "f32[]" = _foreach_sub_scalar[288]
        getitem_2396: "f32[]" = _foreach_sub_scalar[289]
        getitem_2397: "f32[]" = _foreach_sub_scalar[290]
        getitem_2398: "f32[]" = _foreach_sub_scalar[291]
        getitem_2399: "f32[]" = _foreach_sub_scalar[292]
        getitem_2400: "f32[]" = _foreach_sub_scalar[293]
        getitem_2401: "f32[]" = _foreach_sub_scalar[294]
        getitem_2402: "f32[]" = _foreach_sub_scalar[295]
        getitem_2403: "f32[]" = _foreach_sub_scalar[296]
        getitem_2404: "f32[]" = _foreach_sub_scalar[297]
        getitem_2405: "f32[]" = _foreach_sub_scalar[298]
        getitem_2406: "f32[]" = _foreach_sub_scalar[299]
        getitem_2407: "f32[]" = _foreach_sub_scalar[300];  _foreach_sub_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406, getitem_2407)


def _default_make_inputs():
    return [
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([50, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768, 512], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3, 32, 32], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([2304, 768], dtype=torch.float32, device='cuda'),
    torch.randn([2304], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([77, 512], dtype=torch.float32, device='cuda'),
    torch.randn([49408, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([1536, 512], dtype=torch.float32, device='cuda'),
    torch.randn([1536], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([2048], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

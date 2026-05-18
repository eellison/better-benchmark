"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: abbd4bc08e6c
Shape hash: 869e899c
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
    def forward(self, arg824_1: "f32[30522, 768]", arg825_1: "f32[512, 768]", arg826_1: "f32[1024, 768]", arg827_1: "f32[1024, 768]", arg828_1: "f32[1024, 768]", arg829_1: "f32[1024, 768]", arg830_1: "f32[2, 768]", arg831_1: "f32[768]", arg832_1: "f32[768]", arg833_1: "f32[768, 768]", arg834_1: "f32[768]", arg835_1: "f32[768, 768]", arg836_1: "f32[768]", arg837_1: "f32[768, 768]", arg838_1: "f32[768]", arg839_1: "f32[768, 768]", arg840_1: "f32[768]", arg841_1: "f32[768]", arg842_1: "f32[768]", arg843_1: "f32[3072, 768]", arg844_1: "f32[3072]", arg845_1: "f32[768, 3072]", arg846_1: "f32[768]", arg847_1: "f32[768]", arg848_1: "f32[768]", arg849_1: "f32[768, 768]", arg850_1: "f32[768]", arg851_1: "f32[768, 768]", arg852_1: "f32[768]", arg853_1: "f32[768, 768]", arg854_1: "f32[768]", arg855_1: "f32[768, 768]", arg856_1: "f32[768]", arg857_1: "f32[768]", arg858_1: "f32[768]", arg859_1: "f32[3072, 768]", arg860_1: "f32[3072]", arg861_1: "f32[768, 3072]", arg862_1: "f32[768]", arg863_1: "f32[768]", arg864_1: "f32[768]", arg865_1: "f32[768, 768]", arg866_1: "f32[768]", arg867_1: "f32[768, 768]", arg868_1: "f32[768]", arg869_1: "f32[768, 768]", arg870_1: "f32[768]", arg871_1: "f32[768, 768]", arg872_1: "f32[768]", arg873_1: "f32[768]", arg874_1: "f32[768]", arg875_1: "f32[3072, 768]", arg876_1: "f32[3072]", arg877_1: "f32[768, 3072]", arg878_1: "f32[768]", arg879_1: "f32[768]", arg880_1: "f32[768]", arg881_1: "f32[768, 768]", arg882_1: "f32[768]", arg883_1: "f32[768, 768]", arg884_1: "f32[768]", arg885_1: "f32[768, 768]", arg886_1: "f32[768]", arg887_1: "f32[768, 768]", arg888_1: "f32[768]", arg889_1: "f32[768]", arg890_1: "f32[768]", arg891_1: "f32[3072, 768]", arg892_1: "f32[3072]", arg893_1: "f32[768, 3072]", arg894_1: "f32[768]", arg895_1: "f32[768]", arg896_1: "f32[768]", arg897_1: "f32[768, 768]", arg898_1: "f32[768]", arg899_1: "f32[768, 768]", arg900_1: "f32[768]", arg901_1: "f32[768, 768]", arg902_1: "f32[768]", arg903_1: "f32[768, 768]", arg904_1: "f32[768]", arg905_1: "f32[768]", arg906_1: "f32[768]", arg907_1: "f32[3072, 768]", arg908_1: "f32[3072]", arg909_1: "f32[768, 3072]", arg910_1: "f32[768]", arg911_1: "f32[768]", arg912_1: "f32[768]", arg913_1: "f32[768, 768]", arg914_1: "f32[768]", arg915_1: "f32[768, 768]", arg916_1: "f32[768]", arg917_1: "f32[768, 768]", arg918_1: "f32[768]", arg919_1: "f32[768, 768]", arg920_1: "f32[768]", arg921_1: "f32[768]", arg922_1: "f32[768]", arg923_1: "f32[3072, 768]", arg924_1: "f32[3072]", arg925_1: "f32[768, 3072]", arg926_1: "f32[768]", arg927_1: "f32[768]", arg928_1: "f32[768]", arg929_1: "f32[768, 768]", arg930_1: "f32[768]", arg931_1: "f32[768, 768]", arg932_1: "f32[768]", arg933_1: "f32[768, 768]", arg934_1: "f32[768]", arg935_1: "f32[768, 768]", arg936_1: "f32[768]", arg937_1: "f32[768]", arg938_1: "f32[768]", arg939_1: "f32[3072, 768]", arg940_1: "f32[3072]", arg941_1: "f32[768, 3072]", arg942_1: "f32[768]", arg943_1: "f32[768]", arg944_1: "f32[768]", arg945_1: "f32[768, 768]", arg946_1: "f32[768]", arg947_1: "f32[768, 768]", arg948_1: "f32[768]", arg949_1: "f32[768, 768]", arg950_1: "f32[768]", arg951_1: "f32[768, 768]", arg952_1: "f32[768]", arg953_1: "f32[768]", arg954_1: "f32[768]", arg955_1: "f32[3072, 768]", arg956_1: "f32[3072]", arg957_1: "f32[768, 3072]", arg958_1: "f32[768]", arg959_1: "f32[768]", arg960_1: "f32[768]", arg961_1: "f32[768, 768]", arg962_1: "f32[768]", arg963_1: "f32[768, 768]", arg964_1: "f32[768]", arg965_1: "f32[768, 768]", arg966_1: "f32[768]", arg967_1: "f32[768, 768]", arg968_1: "f32[768]", arg969_1: "f32[768]", arg970_1: "f32[768]", arg971_1: "f32[3072, 768]", arg972_1: "f32[3072]", arg973_1: "f32[768, 3072]", arg974_1: "f32[768]", arg975_1: "f32[768]", arg976_1: "f32[768]", arg977_1: "f32[768, 768]", arg978_1: "f32[768]", arg979_1: "f32[768, 768]", arg980_1: "f32[768]", arg981_1: "f32[768, 768]", arg982_1: "f32[768]", arg983_1: "f32[768, 768]", arg984_1: "f32[768]", arg985_1: "f32[768]", arg986_1: "f32[768]", arg987_1: "f32[3072, 768]", arg988_1: "f32[3072]", arg989_1: "f32[768, 3072]", arg990_1: "f32[768]", arg991_1: "f32[768]", arg992_1: "f32[768]", arg993_1: "f32[768, 768]", arg994_1: "f32[768]", arg995_1: "f32[768, 768]", arg996_1: "f32[768]", arg997_1: "f32[768, 768]", arg998_1: "f32[768]", arg999_1: "f32[768, 768]", arg1000_1: "f32[768]", arg1001_1: "f32[768]", arg1002_1: "f32[768]", arg1003_1: "f32[3072, 768]", arg1004_1: "f32[3072]", arg1005_1: "f32[768, 3072]", arg1006_1: "f32[768]", arg1007_1: "f32[768]", arg1008_1: "f32[768]", arg1009_1: "f32[768, 768]", arg1010_1: "f32[768]", arg1011_1: "f32[768, 768]", arg1012_1: "f32[768]", arg1013_1: "f32[768, 768]", arg1014_1: "f32[768]", arg1015_1: "f32[768, 768]", arg1016_1: "f32[768]", arg1017_1: "f32[768]", arg1018_1: "f32[768]", arg1019_1: "f32[3072, 768]", arg1020_1: "f32[3072]", arg1021_1: "f32[768, 3072]", arg1022_1: "f32[768]", arg1023_1: "f32[768]", arg1024_1: "f32[768]", arg1025_1: "f32[30522]", arg1026_1: "f32[768, 768]", arg1027_1: "f32[768]", arg1028_1: "f32[768]", arg1029_1: "f32[768]", arg209_1: "f32[30522, 768]", arg207_1: "f32[512, 768]", arg416_1: "f32[1024, 768]", arg417_1: "f32[1024, 768]", arg418_1: "f32[1024, 768]", arg419_1: "f32[1024, 768]", arg420_1: "f32[2, 768]", arg421_1: "f32[768]", arg422_1: "f32[768]", arg423_1: "f32[768, 768]", arg424_1: "f32[768]", arg425_1: "f32[768, 768]", arg426_1: "f32[768]", arg427_1: "f32[768, 768]", arg428_1: "f32[768]", arg429_1: "f32[768, 768]", arg430_1: "f32[768]", arg431_1: "f32[768]", arg432_1: "f32[768]", arg433_1: "f32[3072, 768]", arg434_1: "f32[3072]", arg435_1: "f32[768, 3072]", arg436_1: "f32[768]", arg437_1: "f32[768]", arg438_1: "f32[768]", arg439_1: "f32[768, 768]", arg440_1: "f32[768]", arg441_1: "f32[768, 768]", arg442_1: "f32[768]", arg443_1: "f32[768, 768]", arg444_1: "f32[768]", arg445_1: "f32[768, 768]", arg446_1: "f32[768]", arg447_1: "f32[768]", arg448_1: "f32[768]", arg449_1: "f32[3072, 768]", arg450_1: "f32[3072]", arg451_1: "f32[768, 3072]", arg452_1: "f32[768]", arg453_1: "f32[768]", arg454_1: "f32[768]", arg455_1: "f32[768, 768]", arg456_1: "f32[768]", arg457_1: "f32[768, 768]", arg458_1: "f32[768]", arg459_1: "f32[768, 768]", arg460_1: "f32[768]", arg461_1: "f32[768, 768]", arg462_1: "f32[768]", arg463_1: "f32[768]", arg464_1: "f32[768]", arg465_1: "f32[3072, 768]", arg466_1: "f32[3072]", arg467_1: "f32[768, 3072]", arg468_1: "f32[768]", arg469_1: "f32[768]", arg470_1: "f32[768]", arg471_1: "f32[768, 768]", arg472_1: "f32[768]", arg473_1: "f32[768, 768]", arg474_1: "f32[768]", arg475_1: "f32[768, 768]", arg476_1: "f32[768]", arg477_1: "f32[768, 768]", arg478_1: "f32[768]", arg479_1: "f32[768]", arg480_1: "f32[768]", arg481_1: "f32[3072, 768]", arg482_1: "f32[3072]", arg483_1: "f32[768, 3072]", arg484_1: "f32[768]", arg485_1: "f32[768]", arg486_1: "f32[768]", arg487_1: "f32[768, 768]", arg488_1: "f32[768]", arg489_1: "f32[768, 768]", arg490_1: "f32[768]", arg491_1: "f32[768, 768]", arg492_1: "f32[768]", arg493_1: "f32[768, 768]", arg494_1: "f32[768]", arg495_1: "f32[768]", arg496_1: "f32[768]", arg497_1: "f32[3072, 768]", arg498_1: "f32[3072]", arg499_1: "f32[768, 3072]", arg500_1: "f32[768]", arg501_1: "f32[768]", arg502_1: "f32[768]", arg503_1: "f32[768, 768]", arg504_1: "f32[768]", arg505_1: "f32[768, 768]", arg506_1: "f32[768]", arg507_1: "f32[768, 768]", arg508_1: "f32[768]", arg509_1: "f32[768, 768]", arg510_1: "f32[768]", arg511_1: "f32[768]", arg512_1: "f32[768]", arg513_1: "f32[3072, 768]", arg514_1: "f32[3072]", arg515_1: "f32[768, 3072]", arg516_1: "f32[768]", arg517_1: "f32[768]", arg518_1: "f32[768]", arg519_1: "f32[768, 768]", arg520_1: "f32[768]", arg521_1: "f32[768, 768]", arg522_1: "f32[768]", arg523_1: "f32[768, 768]", arg524_1: "f32[768]", arg525_1: "f32[768, 768]", arg526_1: "f32[768]", arg527_1: "f32[768]", arg528_1: "f32[768]", arg529_1: "f32[3072, 768]", arg530_1: "f32[3072]", arg531_1: "f32[768, 3072]", arg532_1: "f32[768]", arg533_1: "f32[768]", arg534_1: "f32[768]", arg535_1: "f32[768, 768]", arg536_1: "f32[768]", arg537_1: "f32[768, 768]", arg538_1: "f32[768]", arg539_1: "f32[768, 768]", arg540_1: "f32[768]", arg541_1: "f32[768, 768]", arg542_1: "f32[768]", arg543_1: "f32[768]", arg544_1: "f32[768]", arg545_1: "f32[3072, 768]", arg546_1: "f32[3072]", arg547_1: "f32[768, 3072]", arg548_1: "f32[768]", arg549_1: "f32[768]", arg550_1: "f32[768]", arg551_1: "f32[768, 768]", arg552_1: "f32[768]", arg553_1: "f32[768, 768]", arg554_1: "f32[768]", arg555_1: "f32[768, 768]", arg556_1: "f32[768]", arg557_1: "f32[768, 768]", arg558_1: "f32[768]", arg559_1: "f32[768]", arg560_1: "f32[768]", arg561_1: "f32[3072, 768]", arg562_1: "f32[3072]", arg563_1: "f32[768, 3072]", arg564_1: "f32[768]", arg565_1: "f32[768]", arg566_1: "f32[768]", arg567_1: "f32[768, 768]", arg568_1: "f32[768]", arg569_1: "f32[768, 768]", arg570_1: "f32[768]", arg571_1: "f32[768, 768]", arg572_1: "f32[768]", arg573_1: "f32[768, 768]", arg574_1: "f32[768]", arg575_1: "f32[768]", arg576_1: "f32[768]", arg577_1: "f32[3072, 768]", arg578_1: "f32[3072]", arg579_1: "f32[768, 3072]", arg580_1: "f32[768]", arg581_1: "f32[768]", arg582_1: "f32[768]", arg583_1: "f32[768, 768]", arg584_1: "f32[768]", arg585_1: "f32[768, 768]", arg586_1: "f32[768]", arg587_1: "f32[768, 768]", arg588_1: "f32[768]", arg589_1: "f32[768, 768]", arg590_1: "f32[768]", arg591_1: "f32[768]", arg592_1: "f32[768]", arg593_1: "f32[3072, 768]", arg594_1: "f32[3072]", arg595_1: "f32[768, 3072]", arg596_1: "f32[768]", arg597_1: "f32[768]", arg598_1: "f32[768]", arg599_1: "f32[768, 768]", arg600_1: "f32[768]", arg601_1: "f32[768, 768]", arg602_1: "f32[768]", arg603_1: "f32[768, 768]", arg604_1: "f32[768]", arg605_1: "f32[768, 768]", arg606_1: "f32[768]", arg607_1: "f32[768]", arg608_1: "f32[768]", arg609_1: "f32[3072, 768]", arg610_1: "f32[3072]", arg611_1: "f32[768, 3072]", arg612_1: "f32[768]", arg613_1: "f32[768]", arg614_1: "f32[768]", arg615_1: "f32[30522]", arg616_1: "f32[768, 768]", arg617_1: "f32[768]", arg618_1: "f32[768]", arg619_1: "f32[768]", getitem_2060: "f32[]", getitem_2061: "f32[]", getitem_2062: "f32[]", getitem_2063: "f32[]", getitem_2064: "f32[]", getitem_2065: "f32[]", getitem_2066: "f32[]", getitem_2067: "f32[]", getitem_2068: "f32[]", getitem_2069: "f32[]", getitem_2070: "f32[]", getitem_2071: "f32[]", getitem_2072: "f32[]", getitem_2073: "f32[]", getitem_2074: "f32[]", getitem_2075: "f32[]", getitem_2076: "f32[]", getitem_2077: "f32[]", getitem_2078: "f32[]", getitem_2079: "f32[]", getitem_2080: "f32[]", getitem_2081: "f32[]", getitem_2082: "f32[]", getitem_2083: "f32[]", getitem_2084: "f32[]", getitem_2085: "f32[]", getitem_2086: "f32[]", getitem_2087: "f32[]", getitem_2088: "f32[]", getitem_2089: "f32[]", getitem_2090: "f32[]", getitem_2091: "f32[]", getitem_2092: "f32[]", getitem_2093: "f32[]", getitem_2094: "f32[]", getitem_2095: "f32[]", getitem_2096: "f32[]", getitem_2097: "f32[]", getitem_2098: "f32[]", getitem_2099: "f32[]", getitem_2100: "f32[]", getitem_2101: "f32[]", getitem_2102: "f32[]", getitem_2103: "f32[]", getitem_2104: "f32[]", getitem_2105: "f32[]", getitem_2106: "f32[]", getitem_2107: "f32[]", getitem_2108: "f32[]", getitem_2109: "f32[]", getitem_2110: "f32[]", getitem_2111: "f32[]", getitem_2112: "f32[]", getitem_2113: "f32[]", getitem_2114: "f32[]", getitem_2115: "f32[]", getitem_2116: "f32[]", getitem_2117: "f32[]", getitem_2118: "f32[]", getitem_2119: "f32[]", getitem_2120: "f32[]", getitem_2121: "f32[]", getitem_2122: "f32[]", getitem_2123: "f32[]", getitem_2124: "f32[]", getitem_2125: "f32[]", getitem_2126: "f32[]", getitem_2127: "f32[]", getitem_2128: "f32[]", getitem_2129: "f32[]", getitem_2130: "f32[]", getitem_2131: "f32[]", getitem_2132: "f32[]", getitem_2133: "f32[]", getitem_2134: "f32[]", getitem_2135: "f32[]", getitem_2136: "f32[]", getitem_2137: "f32[]", getitem_2138: "f32[]", getitem_2139: "f32[]", getitem_2140: "f32[]", getitem_2141: "f32[]", getitem_2142: "f32[]", getitem_2143: "f32[]", getitem_2144: "f32[]", getitem_2145: "f32[]", getitem_2146: "f32[]", getitem_2147: "f32[]", getitem_2148: "f32[]", getitem_2149: "f32[]", getitem_2150: "f32[]", getitem_2151: "f32[]", getitem_2152: "f32[]", getitem_2153: "f32[]", getitem_2154: "f32[]", getitem_2155: "f32[]", getitem_2156: "f32[]", getitem_2157: "f32[]", getitem_2158: "f32[]", getitem_2159: "f32[]", getitem_2160: "f32[]", getitem_2161: "f32[]", getitem_2162: "f32[]", getitem_2163: "f32[]", getitem_2164: "f32[]", getitem_2165: "f32[]", getitem_2166: "f32[]", getitem_2167: "f32[]", getitem_2168: "f32[]", getitem_2169: "f32[]", getitem_2170: "f32[]", getitem_2171: "f32[]", getitem_2172: "f32[]", getitem_2173: "f32[]", getitem_2174: "f32[]", getitem_2175: "f32[]", getitem_2176: "f32[]", getitem_2177: "f32[]", getitem_2178: "f32[]", getitem_2179: "f32[]", getitem_2180: "f32[]", getitem_2181: "f32[]", getitem_2182: "f32[]", getitem_2183: "f32[]", getitem_2184: "f32[]", getitem_2185: "f32[]", getitem_2186: "f32[]", getitem_2187: "f32[]", getitem_2188: "f32[]", getitem_2189: "f32[]", getitem_2190: "f32[]", getitem_2191: "f32[]", getitem_2192: "f32[]", getitem_2193: "f32[]", getitem_2194: "f32[]", getitem_2195: "f32[]", getitem_2196: "f32[]", getitem_2197: "f32[]", getitem_2198: "f32[]", getitem_2199: "f32[]", getitem_2200: "f32[]", getitem_2201: "f32[]", getitem_2202: "f32[]", getitem_2203: "f32[]", getitem_2204: "f32[]", getitem_2205: "f32[]", getitem_2206: "f32[]", getitem_2207: "f32[]", getitem_2208: "f32[]", getitem_2209: "f32[]", getitem_2210: "f32[]", getitem_2211: "f32[]", getitem_2212: "f32[]", getitem_2213: "f32[]", getitem_2214: "f32[]", getitem_2215: "f32[]", getitem_2216: "f32[]", getitem_2217: "f32[]", getitem_2218: "f32[]", getitem_2219: "f32[]", getitem_2220: "f32[]", getitem_2221: "f32[]", getitem_2222: "f32[]", getitem_2223: "f32[]", getitem_2224: "f32[]", getitem_2225: "f32[]", getitem_2226: "f32[]", getitem_2227: "f32[]", getitem_2228: "f32[]", getitem_2229: "f32[]", getitem_2230: "f32[]", getitem_2231: "f32[]", getitem_2232: "f32[]", getitem_2233: "f32[]", getitem_2234: "f32[]", getitem_2235: "f32[]", getitem_2236: "f32[]", getitem_2237: "f32[]", getitem_2238: "f32[]", getitem_2239: "f32[]", getitem_2240: "f32[]", getitem_2241: "f32[]", getitem_2242: "f32[]", getitem_2243: "f32[]", getitem_2244: "f32[]", getitem_2245: "f32[]", getitem_2246: "f32[]", getitem_2247: "f32[]", getitem_2248: "f32[]", getitem_2249: "f32[]", getitem_2250: "f32[]", getitem_2251: "f32[]", getitem_2252: "f32[]", getitem_2253: "f32[]", getitem_2254: "f32[]", getitem_2255: "f32[]", getitem_2256: "f32[]", getitem_2257: "f32[]", getitem_2258: "f32[]", getitem_2259: "f32[]", getitem_2260: "f32[]", getitem_2261: "f32[]", getitem_2262: "f32[]", getitem_2263: "f32[]", getitem_2264: "f32[]", getitem_2265: "f32[]", getitem_2884: "f32[30522, 768]", getitem_2885: "f32[512, 768]", getitem_2886: "f32[1024, 768]", getitem_2887: "f32[1024, 768]", getitem_2888: "f32[1024, 768]", getitem_2889: "f32[1024, 768]", getitem_2890: "f32[2, 768]", getitem_2891: "f32[768]", getitem_2892: "f32[768]", getitem_2893: "f32[768, 768]", getitem_2894: "f32[768]", getitem_2895: "f32[768, 768]", getitem_2896: "f32[768]", getitem_2897: "f32[768, 768]", getitem_2898: "f32[768]", getitem_2899: "f32[768, 768]", getitem_2900: "f32[768]", getitem_2901: "f32[768]", getitem_2902: "f32[768]", getitem_2903: "f32[3072, 768]", getitem_2904: "f32[3072]", getitem_2905: "f32[768, 3072]", getitem_2906: "f32[768]", getitem_2907: "f32[768]", getitem_2908: "f32[768]", getitem_2909: "f32[768, 768]", getitem_2910: "f32[768]", getitem_2911: "f32[768, 768]", getitem_2912: "f32[768]", getitem_2913: "f32[768, 768]", getitem_2914: "f32[768]", getitem_2915: "f32[768, 768]", getitem_2916: "f32[768]", getitem_2917: "f32[768]", getitem_2918: "f32[768]", getitem_2919: "f32[3072, 768]", getitem_2920: "f32[3072]", getitem_2921: "f32[768, 3072]", getitem_2922: "f32[768]", getitem_2923: "f32[768]", getitem_2924: "f32[768]", getitem_2925: "f32[768, 768]", getitem_2926: "f32[768]", getitem_2927: "f32[768, 768]", getitem_2928: "f32[768]", getitem_2929: "f32[768, 768]", getitem_2930: "f32[768]", getitem_2931: "f32[768, 768]", getitem_2932: "f32[768]", getitem_2933: "f32[768]", getitem_2934: "f32[768]", getitem_2935: "f32[3072, 768]", getitem_2936: "f32[3072]", getitem_2937: "f32[768, 3072]", getitem_2938: "f32[768]", getitem_2939: "f32[768]", getitem_2940: "f32[768]", getitem_2941: "f32[768, 768]", getitem_2942: "f32[768]", getitem_2943: "f32[768, 768]", getitem_2944: "f32[768]", getitem_2945: "f32[768, 768]", getitem_2946: "f32[768]", getitem_2947: "f32[768, 768]", getitem_2948: "f32[768]", getitem_2949: "f32[768]", getitem_2950: "f32[768]", getitem_2951: "f32[3072, 768]", getitem_2952: "f32[3072]", getitem_2953: "f32[768, 3072]", getitem_2954: "f32[768]", getitem_2955: "f32[768]", getitem_2956: "f32[768]", getitem_2957: "f32[768, 768]", getitem_2958: "f32[768]", getitem_2959: "f32[768, 768]", getitem_2960: "f32[768]", getitem_2961: "f32[768, 768]", getitem_2962: "f32[768]", getitem_2963: "f32[768, 768]", getitem_2964: "f32[768]", getitem_2965: "f32[768]", getitem_2966: "f32[768]", getitem_2967: "f32[3072, 768]", getitem_2968: "f32[3072]", getitem_2969: "f32[768, 3072]", getitem_2970: "f32[768]", getitem_2971: "f32[768]", getitem_2972: "f32[768]", getitem_2973: "f32[768, 768]", getitem_2974: "f32[768]", getitem_2975: "f32[768, 768]", getitem_2976: "f32[768]", getitem_2977: "f32[768, 768]", getitem_2978: "f32[768]", getitem_2979: "f32[768, 768]", getitem_2980: "f32[768]", getitem_2981: "f32[768]", getitem_2982: "f32[768]", getitem_2983: "f32[3072, 768]", getitem_2984: "f32[3072]", getitem_2985: "f32[768, 3072]", getitem_2986: "f32[768]", getitem_2987: "f32[768]", getitem_2988: "f32[768]", getitem_2989: "f32[768, 768]", getitem_2990: "f32[768]", getitem_2991: "f32[768, 768]", getitem_2992: "f32[768]", getitem_2993: "f32[768, 768]", getitem_2994: "f32[768]", getitem_2995: "f32[768, 768]", getitem_2996: "f32[768]", getitem_2997: "f32[768]", getitem_2998: "f32[768]", getitem_2999: "f32[3072, 768]", getitem_3000: "f32[3072]", getitem_3001: "f32[768, 3072]", getitem_3002: "f32[768]", getitem_3003: "f32[768]", getitem_3004: "f32[768]", getitem_3005: "f32[768, 768]", getitem_3006: "f32[768]", getitem_3007: "f32[768, 768]", getitem_3008: "f32[768]", getitem_3009: "f32[768, 768]", getitem_3010: "f32[768]", getitem_3011: "f32[768, 768]", getitem_3012: "f32[768]", getitem_3013: "f32[768]", getitem_3014: "f32[768]", getitem_3015: "f32[3072, 768]", getitem_3016: "f32[3072]", getitem_3017: "f32[768, 3072]", getitem_3018: "f32[768]", getitem_3019: "f32[768]", getitem_3020: "f32[768]", getitem_3021: "f32[768, 768]", getitem_3022: "f32[768]", getitem_3023: "f32[768, 768]", getitem_3024: "f32[768]", getitem_3025: "f32[768, 768]", getitem_3026: "f32[768]", getitem_3027: "f32[768, 768]", getitem_3028: "f32[768]", getitem_3029: "f32[768]", getitem_3030: "f32[768]", getitem_3031: "f32[3072, 768]", getitem_3032: "f32[3072]", getitem_3033: "f32[768, 3072]", getitem_3034: "f32[768]", getitem_3035: "f32[768]", getitem_3036: "f32[768]", getitem_3037: "f32[768, 768]", getitem_3038: "f32[768]", getitem_3039: "f32[768, 768]", getitem_3040: "f32[768]", getitem_3041: "f32[768, 768]", getitem_3042: "f32[768]", getitem_3043: "f32[768, 768]", getitem_3044: "f32[768]", getitem_3045: "f32[768]", getitem_3046: "f32[768]", getitem_3047: "f32[3072, 768]", getitem_3048: "f32[3072]", getitem_3049: "f32[768, 3072]", getitem_3050: "f32[768]", getitem_3051: "f32[768]", getitem_3052: "f32[768]", getitem_3053: "f32[768, 768]", getitem_3054: "f32[768]", getitem_3055: "f32[768, 768]", getitem_3056: "f32[768]", getitem_3057: "f32[768, 768]", getitem_3058: "f32[768]", getitem_3059: "f32[768, 768]", getitem_3060: "f32[768]", getitem_3061: "f32[768]", getitem_3062: "f32[768]", getitem_3063: "f32[3072, 768]", getitem_3064: "f32[3072]", getitem_3065: "f32[768, 3072]", getitem_3066: "f32[768]", getitem_3067: "f32[768]", getitem_3068: "f32[768]", getitem_3069: "f32[768, 768]", getitem_3070: "f32[768]", getitem_3071: "f32[768, 768]", getitem_3072: "f32[768]", getitem_3073: "f32[768, 768]", getitem_3074: "f32[768]", getitem_3075: "f32[768, 768]", getitem_3076: "f32[768]", getitem_3077: "f32[768]", getitem_3078: "f32[768]", getitem_3079: "f32[3072, 768]", getitem_3080: "f32[3072]", getitem_3081: "f32[768, 3072]", getitem_3082: "f32[768]", getitem_3083: "f32[768]", getitem_3084: "f32[768]", getitem_3085: "f32[30522]", getitem_3086: "f32[768, 768]", getitem_3087: "f32[768]", getitem_3088: "f32[768]", getitem_3089: "f32[768]"):
        # No stacktrace found for following nodes
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1], [arg209_1, arg207_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1]);  arg824_1 = arg825_1 = arg826_1 = arg827_1 = arg828_1 = arg829_1 = arg830_1 = arg831_1 = arg832_1 = arg833_1 = arg834_1 = arg835_1 = arg836_1 = arg837_1 = arg838_1 = arg839_1 = arg840_1 = arg841_1 = arg842_1 = arg843_1 = arg844_1 = arg845_1 = arg846_1 = arg847_1 = arg848_1 = arg849_1 = arg850_1 = arg851_1 = arg852_1 = arg853_1 = arg854_1 = arg855_1 = arg856_1 = arg857_1 = arg858_1 = arg859_1 = arg860_1 = arg861_1 = arg862_1 = arg863_1 = arg864_1 = arg865_1 = arg866_1 = arg867_1 = arg868_1 = arg869_1 = arg870_1 = arg871_1 = arg872_1 = arg873_1 = arg874_1 = arg875_1 = arg876_1 = arg877_1 = arg878_1 = arg879_1 = arg880_1 = arg881_1 = arg882_1 = arg883_1 = arg884_1 = arg885_1 = arg886_1 = arg887_1 = arg888_1 = arg889_1 = arg890_1 = arg891_1 = arg892_1 = arg893_1 = arg894_1 = arg895_1 = arg896_1 = arg897_1 = arg898_1 = arg899_1 = arg900_1 = arg901_1 = arg902_1 = arg903_1 = arg904_1 = arg905_1 = arg906_1 = arg907_1 = arg908_1 = arg909_1 = arg910_1 = arg911_1 = arg912_1 = arg913_1 = arg914_1 = arg915_1 = arg916_1 = arg917_1 = arg918_1 = arg919_1 = arg920_1 = arg921_1 = arg922_1 = arg923_1 = arg924_1 = arg925_1 = arg926_1 = arg927_1 = arg928_1 = arg929_1 = arg930_1 = arg931_1 = arg932_1 = arg933_1 = arg934_1 = arg935_1 = arg936_1 = arg937_1 = arg938_1 = arg939_1 = arg940_1 = arg941_1 = arg942_1 = arg943_1 = arg944_1 = arg945_1 = arg946_1 = arg947_1 = arg948_1 = arg949_1 = arg950_1 = arg951_1 = arg952_1 = arg953_1 = arg954_1 = arg955_1 = arg956_1 = arg957_1 = arg958_1 = arg959_1 = arg960_1 = arg961_1 = arg962_1 = arg963_1 = arg964_1 = arg965_1 = arg966_1 = arg967_1 = arg968_1 = arg969_1 = arg970_1 = arg971_1 = arg972_1 = arg973_1 = arg974_1 = arg975_1 = arg976_1 = arg977_1 = arg978_1 = arg979_1 = arg980_1 = arg981_1 = arg982_1 = arg983_1 = arg984_1 = arg985_1 = arg986_1 = arg987_1 = arg988_1 = arg989_1 = arg990_1 = arg991_1 = arg992_1 = arg993_1 = arg994_1 = arg995_1 = arg996_1 = arg997_1 = arg998_1 = arg999_1 = arg1000_1 = arg1001_1 = arg1002_1 = arg1003_1 = arg1004_1 = arg1005_1 = arg1006_1 = arg1007_1 = arg1008_1 = arg1009_1 = arg1010_1 = arg1011_1 = arg1012_1 = arg1013_1 = arg1014_1 = arg1015_1 = arg1016_1 = arg1017_1 = arg1018_1 = arg1019_1 = arg1020_1 = arg1021_1 = arg1022_1 = arg1023_1 = arg1024_1 = arg1025_1 = arg1026_1 = arg1027_1 = arg1028_1 = arg1029_1 = arg209_1 = arg207_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = arg506_1 = arg507_1 = arg508_1 = arg509_1 = arg510_1 = arg511_1 = arg512_1 = arg513_1 = arg514_1 = arg515_1 = arg516_1 = arg517_1 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = arg522_1 = arg523_1 = arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = arg540_1 = arg541_1 = arg542_1 = arg543_1 = arg544_1 = arg545_1 = arg546_1 = arg547_1 = arg548_1 = arg549_1 = arg550_1 = arg551_1 = arg552_1 = arg553_1 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = arg558_1 = arg559_1 = arg560_1 = arg561_1 = arg562_1 = arg563_1 = arg564_1 = arg565_1 = arg566_1 = arg567_1 = arg568_1 = arg569_1 = arg570_1 = arg571_1 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = arg576_1 = arg577_1 = arg578_1 = arg579_1 = arg580_1 = arg581_1 = arg582_1 = arg583_1 = arg584_1 = arg585_1 = arg586_1 = arg587_1 = arg588_1 = arg589_1 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = arg594_1 = arg595_1 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = None
        getitem: "f32[30522, 768]" = _foreach_sub_list[0]
        getitem_1: "f32[512, 768]" = _foreach_sub_list[1]
        getitem_2: "f32[1024, 768]" = _foreach_sub_list[2]
        getitem_3: "f32[1024, 768]" = _foreach_sub_list[3]
        getitem_4: "f32[1024, 768]" = _foreach_sub_list[4]
        getitem_5: "f32[1024, 768]" = _foreach_sub_list[5]
        getitem_6: "f32[2, 768]" = _foreach_sub_list[6]
        getitem_7: "f32[768]" = _foreach_sub_list[7]
        getitem_8: "f32[768]" = _foreach_sub_list[8]
        getitem_9: "f32[768, 768]" = _foreach_sub_list[9]
        getitem_10: "f32[768]" = _foreach_sub_list[10]
        getitem_11: "f32[768, 768]" = _foreach_sub_list[11]
        getitem_12: "f32[768]" = _foreach_sub_list[12]
        getitem_13: "f32[768, 768]" = _foreach_sub_list[13]
        getitem_14: "f32[768]" = _foreach_sub_list[14]
        getitem_15: "f32[768, 768]" = _foreach_sub_list[15]
        getitem_16: "f32[768]" = _foreach_sub_list[16]
        getitem_17: "f32[768]" = _foreach_sub_list[17]
        getitem_18: "f32[768]" = _foreach_sub_list[18]
        getitem_19: "f32[3072, 768]" = _foreach_sub_list[19]
        getitem_20: "f32[3072]" = _foreach_sub_list[20]
        getitem_21: "f32[768, 3072]" = _foreach_sub_list[21]
        getitem_22: "f32[768]" = _foreach_sub_list[22]
        getitem_23: "f32[768]" = _foreach_sub_list[23]
        getitem_24: "f32[768]" = _foreach_sub_list[24]
        getitem_25: "f32[768, 768]" = _foreach_sub_list[25]
        getitem_26: "f32[768]" = _foreach_sub_list[26]
        getitem_27: "f32[768, 768]" = _foreach_sub_list[27]
        getitem_28: "f32[768]" = _foreach_sub_list[28]
        getitem_29: "f32[768, 768]" = _foreach_sub_list[29]
        getitem_30: "f32[768]" = _foreach_sub_list[30]
        getitem_31: "f32[768, 768]" = _foreach_sub_list[31]
        getitem_32: "f32[768]" = _foreach_sub_list[32]
        getitem_33: "f32[768]" = _foreach_sub_list[33]
        getitem_34: "f32[768]" = _foreach_sub_list[34]
        getitem_35: "f32[3072, 768]" = _foreach_sub_list[35]
        getitem_36: "f32[3072]" = _foreach_sub_list[36]
        getitem_37: "f32[768, 3072]" = _foreach_sub_list[37]
        getitem_38: "f32[768]" = _foreach_sub_list[38]
        getitem_39: "f32[768]" = _foreach_sub_list[39]
        getitem_40: "f32[768]" = _foreach_sub_list[40]
        getitem_41: "f32[768, 768]" = _foreach_sub_list[41]
        getitem_42: "f32[768]" = _foreach_sub_list[42]
        getitem_43: "f32[768, 768]" = _foreach_sub_list[43]
        getitem_44: "f32[768]" = _foreach_sub_list[44]
        getitem_45: "f32[768, 768]" = _foreach_sub_list[45]
        getitem_46: "f32[768]" = _foreach_sub_list[46]
        getitem_47: "f32[768, 768]" = _foreach_sub_list[47]
        getitem_48: "f32[768]" = _foreach_sub_list[48]
        getitem_49: "f32[768]" = _foreach_sub_list[49]
        getitem_50: "f32[768]" = _foreach_sub_list[50]
        getitem_51: "f32[3072, 768]" = _foreach_sub_list[51]
        getitem_52: "f32[3072]" = _foreach_sub_list[52]
        getitem_53: "f32[768, 3072]" = _foreach_sub_list[53]
        getitem_54: "f32[768]" = _foreach_sub_list[54]
        getitem_55: "f32[768]" = _foreach_sub_list[55]
        getitem_56: "f32[768]" = _foreach_sub_list[56]
        getitem_57: "f32[768, 768]" = _foreach_sub_list[57]
        getitem_58: "f32[768]" = _foreach_sub_list[58]
        getitem_59: "f32[768, 768]" = _foreach_sub_list[59]
        getitem_60: "f32[768]" = _foreach_sub_list[60]
        getitem_61: "f32[768, 768]" = _foreach_sub_list[61]
        getitem_62: "f32[768]" = _foreach_sub_list[62]
        getitem_63: "f32[768, 768]" = _foreach_sub_list[63]
        getitem_64: "f32[768]" = _foreach_sub_list[64]
        getitem_65: "f32[768]" = _foreach_sub_list[65]
        getitem_66: "f32[768]" = _foreach_sub_list[66]
        getitem_67: "f32[3072, 768]" = _foreach_sub_list[67]
        getitem_68: "f32[3072]" = _foreach_sub_list[68]
        getitem_69: "f32[768, 3072]" = _foreach_sub_list[69]
        getitem_70: "f32[768]" = _foreach_sub_list[70]
        getitem_71: "f32[768]" = _foreach_sub_list[71]
        getitem_72: "f32[768]" = _foreach_sub_list[72]
        getitem_73: "f32[768, 768]" = _foreach_sub_list[73]
        getitem_74: "f32[768]" = _foreach_sub_list[74]
        getitem_75: "f32[768, 768]" = _foreach_sub_list[75]
        getitem_76: "f32[768]" = _foreach_sub_list[76]
        getitem_77: "f32[768, 768]" = _foreach_sub_list[77]
        getitem_78: "f32[768]" = _foreach_sub_list[78]
        getitem_79: "f32[768, 768]" = _foreach_sub_list[79]
        getitem_80: "f32[768]" = _foreach_sub_list[80]
        getitem_81: "f32[768]" = _foreach_sub_list[81]
        getitem_82: "f32[768]" = _foreach_sub_list[82]
        getitem_83: "f32[3072, 768]" = _foreach_sub_list[83]
        getitem_84: "f32[3072]" = _foreach_sub_list[84]
        getitem_85: "f32[768, 3072]" = _foreach_sub_list[85]
        getitem_86: "f32[768]" = _foreach_sub_list[86]
        getitem_87: "f32[768]" = _foreach_sub_list[87]
        getitem_88: "f32[768]" = _foreach_sub_list[88]
        getitem_89: "f32[768, 768]" = _foreach_sub_list[89]
        getitem_90: "f32[768]" = _foreach_sub_list[90]
        getitem_91: "f32[768, 768]" = _foreach_sub_list[91]
        getitem_92: "f32[768]" = _foreach_sub_list[92]
        getitem_93: "f32[768, 768]" = _foreach_sub_list[93]
        getitem_94: "f32[768]" = _foreach_sub_list[94]
        getitem_95: "f32[768, 768]" = _foreach_sub_list[95]
        getitem_96: "f32[768]" = _foreach_sub_list[96]
        getitem_97: "f32[768]" = _foreach_sub_list[97]
        getitem_98: "f32[768]" = _foreach_sub_list[98]
        getitem_99: "f32[3072, 768]" = _foreach_sub_list[99]
        getitem_100: "f32[3072]" = _foreach_sub_list[100]
        getitem_101: "f32[768, 3072]" = _foreach_sub_list[101]
        getitem_102: "f32[768]" = _foreach_sub_list[102]
        getitem_103: "f32[768]" = _foreach_sub_list[103]
        getitem_104: "f32[768]" = _foreach_sub_list[104]
        getitem_105: "f32[768, 768]" = _foreach_sub_list[105]
        getitem_106: "f32[768]" = _foreach_sub_list[106]
        getitem_107: "f32[768, 768]" = _foreach_sub_list[107]
        getitem_108: "f32[768]" = _foreach_sub_list[108]
        getitem_109: "f32[768, 768]" = _foreach_sub_list[109]
        getitem_110: "f32[768]" = _foreach_sub_list[110]
        getitem_111: "f32[768, 768]" = _foreach_sub_list[111]
        getitem_112: "f32[768]" = _foreach_sub_list[112]
        getitem_113: "f32[768]" = _foreach_sub_list[113]
        getitem_114: "f32[768]" = _foreach_sub_list[114]
        getitem_115: "f32[3072, 768]" = _foreach_sub_list[115]
        getitem_116: "f32[3072]" = _foreach_sub_list[116]
        getitem_117: "f32[768, 3072]" = _foreach_sub_list[117]
        getitem_118: "f32[768]" = _foreach_sub_list[118]
        getitem_119: "f32[768]" = _foreach_sub_list[119]
        getitem_120: "f32[768]" = _foreach_sub_list[120]
        getitem_121: "f32[768, 768]" = _foreach_sub_list[121]
        getitem_122: "f32[768]" = _foreach_sub_list[122]
        getitem_123: "f32[768, 768]" = _foreach_sub_list[123]
        getitem_124: "f32[768]" = _foreach_sub_list[124]
        getitem_125: "f32[768, 768]" = _foreach_sub_list[125]
        getitem_126: "f32[768]" = _foreach_sub_list[126]
        getitem_127: "f32[768, 768]" = _foreach_sub_list[127]
        getitem_128: "f32[768]" = _foreach_sub_list[128]
        getitem_129: "f32[768]" = _foreach_sub_list[129]
        getitem_130: "f32[768]" = _foreach_sub_list[130]
        getitem_131: "f32[3072, 768]" = _foreach_sub_list[131]
        getitem_132: "f32[3072]" = _foreach_sub_list[132]
        getitem_133: "f32[768, 3072]" = _foreach_sub_list[133]
        getitem_134: "f32[768]" = _foreach_sub_list[134]
        getitem_135: "f32[768]" = _foreach_sub_list[135]
        getitem_136: "f32[768]" = _foreach_sub_list[136]
        getitem_137: "f32[768, 768]" = _foreach_sub_list[137]
        getitem_138: "f32[768]" = _foreach_sub_list[138]
        getitem_139: "f32[768, 768]" = _foreach_sub_list[139]
        getitem_140: "f32[768]" = _foreach_sub_list[140]
        getitem_141: "f32[768, 768]" = _foreach_sub_list[141]
        getitem_142: "f32[768]" = _foreach_sub_list[142]
        getitem_143: "f32[768, 768]" = _foreach_sub_list[143]
        getitem_144: "f32[768]" = _foreach_sub_list[144]
        getitem_145: "f32[768]" = _foreach_sub_list[145]
        getitem_146: "f32[768]" = _foreach_sub_list[146]
        getitem_147: "f32[3072, 768]" = _foreach_sub_list[147]
        getitem_148: "f32[3072]" = _foreach_sub_list[148]
        getitem_149: "f32[768, 3072]" = _foreach_sub_list[149]
        getitem_150: "f32[768]" = _foreach_sub_list[150]
        getitem_151: "f32[768]" = _foreach_sub_list[151]
        getitem_152: "f32[768]" = _foreach_sub_list[152]
        getitem_153: "f32[768, 768]" = _foreach_sub_list[153]
        getitem_154: "f32[768]" = _foreach_sub_list[154]
        getitem_155: "f32[768, 768]" = _foreach_sub_list[155]
        getitem_156: "f32[768]" = _foreach_sub_list[156]
        getitem_157: "f32[768, 768]" = _foreach_sub_list[157]
        getitem_158: "f32[768]" = _foreach_sub_list[158]
        getitem_159: "f32[768, 768]" = _foreach_sub_list[159]
        getitem_160: "f32[768]" = _foreach_sub_list[160]
        getitem_161: "f32[768]" = _foreach_sub_list[161]
        getitem_162: "f32[768]" = _foreach_sub_list[162]
        getitem_163: "f32[3072, 768]" = _foreach_sub_list[163]
        getitem_164: "f32[3072]" = _foreach_sub_list[164]
        getitem_165: "f32[768, 3072]" = _foreach_sub_list[165]
        getitem_166: "f32[768]" = _foreach_sub_list[166]
        getitem_167: "f32[768]" = _foreach_sub_list[167]
        getitem_168: "f32[768]" = _foreach_sub_list[168]
        getitem_169: "f32[768, 768]" = _foreach_sub_list[169]
        getitem_170: "f32[768]" = _foreach_sub_list[170]
        getitem_171: "f32[768, 768]" = _foreach_sub_list[171]
        getitem_172: "f32[768]" = _foreach_sub_list[172]
        getitem_173: "f32[768, 768]" = _foreach_sub_list[173]
        getitem_174: "f32[768]" = _foreach_sub_list[174]
        getitem_175: "f32[768, 768]" = _foreach_sub_list[175]
        getitem_176: "f32[768]" = _foreach_sub_list[176]
        getitem_177: "f32[768]" = _foreach_sub_list[177]
        getitem_178: "f32[768]" = _foreach_sub_list[178]
        getitem_179: "f32[3072, 768]" = _foreach_sub_list[179]
        getitem_180: "f32[3072]" = _foreach_sub_list[180]
        getitem_181: "f32[768, 3072]" = _foreach_sub_list[181]
        getitem_182: "f32[768]" = _foreach_sub_list[182]
        getitem_183: "f32[768]" = _foreach_sub_list[183]
        getitem_184: "f32[768]" = _foreach_sub_list[184]
        getitem_185: "f32[768, 768]" = _foreach_sub_list[185]
        getitem_186: "f32[768]" = _foreach_sub_list[186]
        getitem_187: "f32[768, 768]" = _foreach_sub_list[187]
        getitem_188: "f32[768]" = _foreach_sub_list[188]
        getitem_189: "f32[768, 768]" = _foreach_sub_list[189]
        getitem_190: "f32[768]" = _foreach_sub_list[190]
        getitem_191: "f32[768, 768]" = _foreach_sub_list[191]
        getitem_192: "f32[768]" = _foreach_sub_list[192]
        getitem_193: "f32[768]" = _foreach_sub_list[193]
        getitem_194: "f32[768]" = _foreach_sub_list[194]
        getitem_195: "f32[3072, 768]" = _foreach_sub_list[195]
        getitem_196: "f32[3072]" = _foreach_sub_list[196]
        getitem_197: "f32[768, 3072]" = _foreach_sub_list[197]
        getitem_198: "f32[768]" = _foreach_sub_list[198]
        getitem_199: "f32[768]" = _foreach_sub_list[199]
        getitem_200: "f32[768]" = _foreach_sub_list[200]
        getitem_201: "f32[30522]" = _foreach_sub_list[201]
        getitem_202: "f32[768, 768]" = _foreach_sub_list[202]
        getitem_203: "f32[768]" = _foreach_sub_list[203]
        getitem_204: "f32[768]" = _foreach_sub_list[204]
        getitem_205: "f32[768]" = _foreach_sub_list[205];  _foreach_sub_list = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183, getitem_2184, getitem_2185, getitem_2186, getitem_2187, getitem_2188, getitem_2189, getitem_2190, getitem_2191, getitem_2192, getitem_2193, getitem_2194, getitem_2195, getitem_2196, getitem_2197, getitem_2198, getitem_2199, getitem_2200, getitem_2201, getitem_2202, getitem_2203, getitem_2204, getitem_2205, getitem_2206, getitem_2207, getitem_2208, getitem_2209, getitem_2210, getitem_2211, getitem_2212, getitem_2213, getitem_2214, getitem_2215, getitem_2216, getitem_2217, getitem_2218, getitem_2219, getitem_2220, getitem_2221, getitem_2222, getitem_2223, getitem_2224, getitem_2225, getitem_2226, getitem_2227, getitem_2228, getitem_2229, getitem_2230, getitem_2231, getitem_2232, getitem_2233, getitem_2234, getitem_2235, getitem_2236, getitem_2237, getitem_2238, getitem_2239, getitem_2240, getitem_2241, getitem_2242, getitem_2243, getitem_2244, getitem_2245, getitem_2246, getitem_2247, getitem_2248, getitem_2249, getitem_2250, getitem_2251, getitem_2252, getitem_2253, getitem_2254, getitem_2255, getitem_2256, getitem_2257, getitem_2258, getitem_2259, getitem_2260, getitem_2261, getitem_2262, getitem_2263, getitem_2264, getitem_2265]);  getitem_2060 = getitem_2061 = getitem_2062 = getitem_2063 = getitem_2064 = getitem_2065 = getitem_2066 = getitem_2067 = getitem_2068 = getitem_2069 = getitem_2070 = getitem_2071 = getitem_2072 = getitem_2073 = getitem_2074 = getitem_2075 = getitem_2076 = getitem_2077 = getitem_2078 = getitem_2079 = getitem_2080 = getitem_2081 = getitem_2082 = getitem_2083 = getitem_2084 = getitem_2085 = getitem_2086 = getitem_2087 = getitem_2088 = getitem_2089 = getitem_2090 = getitem_2091 = getitem_2092 = getitem_2093 = getitem_2094 = getitem_2095 = getitem_2096 = getitem_2097 = getitem_2098 = getitem_2099 = getitem_2100 = getitem_2101 = getitem_2102 = getitem_2103 = getitem_2104 = getitem_2105 = getitem_2106 = getitem_2107 = getitem_2108 = getitem_2109 = getitem_2110 = getitem_2111 = getitem_2112 = getitem_2113 = getitem_2114 = getitem_2115 = getitem_2116 = getitem_2117 = getitem_2118 = getitem_2119 = getitem_2120 = getitem_2121 = getitem_2122 = getitem_2123 = getitem_2124 = getitem_2125 = getitem_2126 = getitem_2127 = getitem_2128 = getitem_2129 = getitem_2130 = getitem_2131 = getitem_2132 = getitem_2133 = getitem_2134 = getitem_2135 = getitem_2136 = getitem_2137 = getitem_2138 = getitem_2139 = getitem_2140 = getitem_2141 = getitem_2142 = getitem_2143 = getitem_2144 = getitem_2145 = getitem_2146 = getitem_2147 = getitem_2148 = getitem_2149 = getitem_2150 = getitem_2151 = getitem_2152 = getitem_2153 = getitem_2154 = getitem_2155 = getitem_2156 = getitem_2157 = getitem_2158 = getitem_2159 = getitem_2160 = getitem_2161 = getitem_2162 = getitem_2163 = getitem_2164 = getitem_2165 = getitem_2166 = getitem_2167 = getitem_2168 = getitem_2169 = getitem_2170 = getitem_2171 = getitem_2172 = getitem_2173 = getitem_2174 = getitem_2175 = getitem_2176 = getitem_2177 = getitem_2178 = getitem_2179 = getitem_2180 = getitem_2181 = getitem_2182 = getitem_2183 = getitem_2184 = getitem_2185 = getitem_2186 = getitem_2187 = getitem_2188 = getitem_2189 = getitem_2190 = getitem_2191 = getitem_2192 = getitem_2193 = getitem_2194 = getitem_2195 = getitem_2196 = getitem_2197 = getitem_2198 = getitem_2199 = getitem_2200 = getitem_2201 = getitem_2202 = getitem_2203 = getitem_2204 = getitem_2205 = getitem_2206 = getitem_2207 = getitem_2208 = getitem_2209 = getitem_2210 = getitem_2211 = getitem_2212 = getitem_2213 = getitem_2214 = getitem_2215 = getitem_2216 = getitem_2217 = getitem_2218 = getitem_2219 = getitem_2220 = getitem_2221 = getitem_2222 = getitem_2223 = getitem_2224 = getitem_2225 = getitem_2226 = getitem_2227 = getitem_2228 = getitem_2229 = getitem_2230 = getitem_2231 = getitem_2232 = getitem_2233 = getitem_2234 = getitem_2235 = getitem_2236 = getitem_2237 = getitem_2238 = getitem_2239 = getitem_2240 = getitem_2241 = getitem_2242 = getitem_2243 = getitem_2244 = getitem_2245 = getitem_2246 = getitem_2247 = getitem_2248 = getitem_2249 = getitem_2250 = getitem_2251 = getitem_2252 = getitem_2253 = getitem_2254 = getitem_2255 = getitem_2256 = getitem_2257 = getitem_2258 = getitem_2259 = getitem_2260 = getitem_2261 = getitem_2262 = getitem_2263 = getitem_2264 = getitem_2265 = None
        getitem_2266: "f32[]" = _foreach_reciprocal_default[0]
        getitem_2267: "f32[]" = _foreach_reciprocal_default[1]
        getitem_2268: "f32[]" = _foreach_reciprocal_default[2]
        getitem_2269: "f32[]" = _foreach_reciprocal_default[3]
        getitem_2270: "f32[]" = _foreach_reciprocal_default[4]
        getitem_2271: "f32[]" = _foreach_reciprocal_default[5]
        getitem_2272: "f32[]" = _foreach_reciprocal_default[6]
        getitem_2273: "f32[]" = _foreach_reciprocal_default[7]
        getitem_2274: "f32[]" = _foreach_reciprocal_default[8]
        getitem_2275: "f32[]" = _foreach_reciprocal_default[9]
        getitem_2276: "f32[]" = _foreach_reciprocal_default[10]
        getitem_2277: "f32[]" = _foreach_reciprocal_default[11]
        getitem_2278: "f32[]" = _foreach_reciprocal_default[12]
        getitem_2279: "f32[]" = _foreach_reciprocal_default[13]
        getitem_2280: "f32[]" = _foreach_reciprocal_default[14]
        getitem_2281: "f32[]" = _foreach_reciprocal_default[15]
        getitem_2282: "f32[]" = _foreach_reciprocal_default[16]
        getitem_2283: "f32[]" = _foreach_reciprocal_default[17]
        getitem_2284: "f32[]" = _foreach_reciprocal_default[18]
        getitem_2285: "f32[]" = _foreach_reciprocal_default[19]
        getitem_2286: "f32[]" = _foreach_reciprocal_default[20]
        getitem_2287: "f32[]" = _foreach_reciprocal_default[21]
        getitem_2288: "f32[]" = _foreach_reciprocal_default[22]
        getitem_2289: "f32[]" = _foreach_reciprocal_default[23]
        getitem_2290: "f32[]" = _foreach_reciprocal_default[24]
        getitem_2291: "f32[]" = _foreach_reciprocal_default[25]
        getitem_2292: "f32[]" = _foreach_reciprocal_default[26]
        getitem_2293: "f32[]" = _foreach_reciprocal_default[27]
        getitem_2294: "f32[]" = _foreach_reciprocal_default[28]
        getitem_2295: "f32[]" = _foreach_reciprocal_default[29]
        getitem_2296: "f32[]" = _foreach_reciprocal_default[30]
        getitem_2297: "f32[]" = _foreach_reciprocal_default[31]
        getitem_2298: "f32[]" = _foreach_reciprocal_default[32]
        getitem_2299: "f32[]" = _foreach_reciprocal_default[33]
        getitem_2300: "f32[]" = _foreach_reciprocal_default[34]
        getitem_2301: "f32[]" = _foreach_reciprocal_default[35]
        getitem_2302: "f32[]" = _foreach_reciprocal_default[36]
        getitem_2303: "f32[]" = _foreach_reciprocal_default[37]
        getitem_2304: "f32[]" = _foreach_reciprocal_default[38]
        getitem_2305: "f32[]" = _foreach_reciprocal_default[39]
        getitem_2306: "f32[]" = _foreach_reciprocal_default[40]
        getitem_2307: "f32[]" = _foreach_reciprocal_default[41]
        getitem_2308: "f32[]" = _foreach_reciprocal_default[42]
        getitem_2309: "f32[]" = _foreach_reciprocal_default[43]
        getitem_2310: "f32[]" = _foreach_reciprocal_default[44]
        getitem_2311: "f32[]" = _foreach_reciprocal_default[45]
        getitem_2312: "f32[]" = _foreach_reciprocal_default[46]
        getitem_2313: "f32[]" = _foreach_reciprocal_default[47]
        getitem_2314: "f32[]" = _foreach_reciprocal_default[48]
        getitem_2315: "f32[]" = _foreach_reciprocal_default[49]
        getitem_2316: "f32[]" = _foreach_reciprocal_default[50]
        getitem_2317: "f32[]" = _foreach_reciprocal_default[51]
        getitem_2318: "f32[]" = _foreach_reciprocal_default[52]
        getitem_2319: "f32[]" = _foreach_reciprocal_default[53]
        getitem_2320: "f32[]" = _foreach_reciprocal_default[54]
        getitem_2321: "f32[]" = _foreach_reciprocal_default[55]
        getitem_2322: "f32[]" = _foreach_reciprocal_default[56]
        getitem_2323: "f32[]" = _foreach_reciprocal_default[57]
        getitem_2324: "f32[]" = _foreach_reciprocal_default[58]
        getitem_2325: "f32[]" = _foreach_reciprocal_default[59]
        getitem_2326: "f32[]" = _foreach_reciprocal_default[60]
        getitem_2327: "f32[]" = _foreach_reciprocal_default[61]
        getitem_2328: "f32[]" = _foreach_reciprocal_default[62]
        getitem_2329: "f32[]" = _foreach_reciprocal_default[63]
        getitem_2330: "f32[]" = _foreach_reciprocal_default[64]
        getitem_2331: "f32[]" = _foreach_reciprocal_default[65]
        getitem_2332: "f32[]" = _foreach_reciprocal_default[66]
        getitem_2333: "f32[]" = _foreach_reciprocal_default[67]
        getitem_2334: "f32[]" = _foreach_reciprocal_default[68]
        getitem_2335: "f32[]" = _foreach_reciprocal_default[69]
        getitem_2336: "f32[]" = _foreach_reciprocal_default[70]
        getitem_2337: "f32[]" = _foreach_reciprocal_default[71]
        getitem_2338: "f32[]" = _foreach_reciprocal_default[72]
        getitem_2339: "f32[]" = _foreach_reciprocal_default[73]
        getitem_2340: "f32[]" = _foreach_reciprocal_default[74]
        getitem_2341: "f32[]" = _foreach_reciprocal_default[75]
        getitem_2342: "f32[]" = _foreach_reciprocal_default[76]
        getitem_2343: "f32[]" = _foreach_reciprocal_default[77]
        getitem_2344: "f32[]" = _foreach_reciprocal_default[78]
        getitem_2345: "f32[]" = _foreach_reciprocal_default[79]
        getitem_2346: "f32[]" = _foreach_reciprocal_default[80]
        getitem_2347: "f32[]" = _foreach_reciprocal_default[81]
        getitem_2348: "f32[]" = _foreach_reciprocal_default[82]
        getitem_2349: "f32[]" = _foreach_reciprocal_default[83]
        getitem_2350: "f32[]" = _foreach_reciprocal_default[84]
        getitem_2351: "f32[]" = _foreach_reciprocal_default[85]
        getitem_2352: "f32[]" = _foreach_reciprocal_default[86]
        getitem_2353: "f32[]" = _foreach_reciprocal_default[87]
        getitem_2354: "f32[]" = _foreach_reciprocal_default[88]
        getitem_2355: "f32[]" = _foreach_reciprocal_default[89]
        getitem_2356: "f32[]" = _foreach_reciprocal_default[90]
        getitem_2357: "f32[]" = _foreach_reciprocal_default[91]
        getitem_2358: "f32[]" = _foreach_reciprocal_default[92]
        getitem_2359: "f32[]" = _foreach_reciprocal_default[93]
        getitem_2360: "f32[]" = _foreach_reciprocal_default[94]
        getitem_2361: "f32[]" = _foreach_reciprocal_default[95]
        getitem_2362: "f32[]" = _foreach_reciprocal_default[96]
        getitem_2363: "f32[]" = _foreach_reciprocal_default[97]
        getitem_2364: "f32[]" = _foreach_reciprocal_default[98]
        getitem_2365: "f32[]" = _foreach_reciprocal_default[99]
        getitem_2366: "f32[]" = _foreach_reciprocal_default[100]
        getitem_2367: "f32[]" = _foreach_reciprocal_default[101]
        getitem_2368: "f32[]" = _foreach_reciprocal_default[102]
        getitem_2369: "f32[]" = _foreach_reciprocal_default[103]
        getitem_2370: "f32[]" = _foreach_reciprocal_default[104]
        getitem_2371: "f32[]" = _foreach_reciprocal_default[105]
        getitem_2372: "f32[]" = _foreach_reciprocal_default[106]
        getitem_2373: "f32[]" = _foreach_reciprocal_default[107]
        getitem_2374: "f32[]" = _foreach_reciprocal_default[108]
        getitem_2375: "f32[]" = _foreach_reciprocal_default[109]
        getitem_2376: "f32[]" = _foreach_reciprocal_default[110]
        getitem_2377: "f32[]" = _foreach_reciprocal_default[111]
        getitem_2378: "f32[]" = _foreach_reciprocal_default[112]
        getitem_2379: "f32[]" = _foreach_reciprocal_default[113]
        getitem_2380: "f32[]" = _foreach_reciprocal_default[114]
        getitem_2381: "f32[]" = _foreach_reciprocal_default[115]
        getitem_2382: "f32[]" = _foreach_reciprocal_default[116]
        getitem_2383: "f32[]" = _foreach_reciprocal_default[117]
        getitem_2384: "f32[]" = _foreach_reciprocal_default[118]
        getitem_2385: "f32[]" = _foreach_reciprocal_default[119]
        getitem_2386: "f32[]" = _foreach_reciprocal_default[120]
        getitem_2387: "f32[]" = _foreach_reciprocal_default[121]
        getitem_2388: "f32[]" = _foreach_reciprocal_default[122]
        getitem_2389: "f32[]" = _foreach_reciprocal_default[123]
        getitem_2390: "f32[]" = _foreach_reciprocal_default[124]
        getitem_2391: "f32[]" = _foreach_reciprocal_default[125]
        getitem_2392: "f32[]" = _foreach_reciprocal_default[126]
        getitem_2393: "f32[]" = _foreach_reciprocal_default[127]
        getitem_2394: "f32[]" = _foreach_reciprocal_default[128]
        getitem_2395: "f32[]" = _foreach_reciprocal_default[129]
        getitem_2396: "f32[]" = _foreach_reciprocal_default[130]
        getitem_2397: "f32[]" = _foreach_reciprocal_default[131]
        getitem_2398: "f32[]" = _foreach_reciprocal_default[132]
        getitem_2399: "f32[]" = _foreach_reciprocal_default[133]
        getitem_2400: "f32[]" = _foreach_reciprocal_default[134]
        getitem_2401: "f32[]" = _foreach_reciprocal_default[135]
        getitem_2402: "f32[]" = _foreach_reciprocal_default[136]
        getitem_2403: "f32[]" = _foreach_reciprocal_default[137]
        getitem_2404: "f32[]" = _foreach_reciprocal_default[138]
        getitem_2405: "f32[]" = _foreach_reciprocal_default[139]
        getitem_2406: "f32[]" = _foreach_reciprocal_default[140]
        getitem_2407: "f32[]" = _foreach_reciprocal_default[141]
        getitem_2408: "f32[]" = _foreach_reciprocal_default[142]
        getitem_2409: "f32[]" = _foreach_reciprocal_default[143]
        getitem_2410: "f32[]" = _foreach_reciprocal_default[144]
        getitem_2411: "f32[]" = _foreach_reciprocal_default[145]
        getitem_2412: "f32[]" = _foreach_reciprocal_default[146]
        getitem_2413: "f32[]" = _foreach_reciprocal_default[147]
        getitem_2414: "f32[]" = _foreach_reciprocal_default[148]
        getitem_2415: "f32[]" = _foreach_reciprocal_default[149]
        getitem_2416: "f32[]" = _foreach_reciprocal_default[150]
        getitem_2417: "f32[]" = _foreach_reciprocal_default[151]
        getitem_2418: "f32[]" = _foreach_reciprocal_default[152]
        getitem_2419: "f32[]" = _foreach_reciprocal_default[153]
        getitem_2420: "f32[]" = _foreach_reciprocal_default[154]
        getitem_2421: "f32[]" = _foreach_reciprocal_default[155]
        getitem_2422: "f32[]" = _foreach_reciprocal_default[156]
        getitem_2423: "f32[]" = _foreach_reciprocal_default[157]
        getitem_2424: "f32[]" = _foreach_reciprocal_default[158]
        getitem_2425: "f32[]" = _foreach_reciprocal_default[159]
        getitem_2426: "f32[]" = _foreach_reciprocal_default[160]
        getitem_2427: "f32[]" = _foreach_reciprocal_default[161]
        getitem_2428: "f32[]" = _foreach_reciprocal_default[162]
        getitem_2429: "f32[]" = _foreach_reciprocal_default[163]
        getitem_2430: "f32[]" = _foreach_reciprocal_default[164]
        getitem_2431: "f32[]" = _foreach_reciprocal_default[165]
        getitem_2432: "f32[]" = _foreach_reciprocal_default[166]
        getitem_2433: "f32[]" = _foreach_reciprocal_default[167]
        getitem_2434: "f32[]" = _foreach_reciprocal_default[168]
        getitem_2435: "f32[]" = _foreach_reciprocal_default[169]
        getitem_2436: "f32[]" = _foreach_reciprocal_default[170]
        getitem_2437: "f32[]" = _foreach_reciprocal_default[171]
        getitem_2438: "f32[]" = _foreach_reciprocal_default[172]
        getitem_2439: "f32[]" = _foreach_reciprocal_default[173]
        getitem_2440: "f32[]" = _foreach_reciprocal_default[174]
        getitem_2441: "f32[]" = _foreach_reciprocal_default[175]
        getitem_2442: "f32[]" = _foreach_reciprocal_default[176]
        getitem_2443: "f32[]" = _foreach_reciprocal_default[177]
        getitem_2444: "f32[]" = _foreach_reciprocal_default[178]
        getitem_2445: "f32[]" = _foreach_reciprocal_default[179]
        getitem_2446: "f32[]" = _foreach_reciprocal_default[180]
        getitem_2447: "f32[]" = _foreach_reciprocal_default[181]
        getitem_2448: "f32[]" = _foreach_reciprocal_default[182]
        getitem_2449: "f32[]" = _foreach_reciprocal_default[183]
        getitem_2450: "f32[]" = _foreach_reciprocal_default[184]
        getitem_2451: "f32[]" = _foreach_reciprocal_default[185]
        getitem_2452: "f32[]" = _foreach_reciprocal_default[186]
        getitem_2453: "f32[]" = _foreach_reciprocal_default[187]
        getitem_2454: "f32[]" = _foreach_reciprocal_default[188]
        getitem_2455: "f32[]" = _foreach_reciprocal_default[189]
        getitem_2456: "f32[]" = _foreach_reciprocal_default[190]
        getitem_2457: "f32[]" = _foreach_reciprocal_default[191]
        getitem_2458: "f32[]" = _foreach_reciprocal_default[192]
        getitem_2459: "f32[]" = _foreach_reciprocal_default[193]
        getitem_2460: "f32[]" = _foreach_reciprocal_default[194]
        getitem_2461: "f32[]" = _foreach_reciprocal_default[195]
        getitem_2462: "f32[]" = _foreach_reciprocal_default[196]
        getitem_2463: "f32[]" = _foreach_reciprocal_default[197]
        getitem_2464: "f32[]" = _foreach_reciprocal_default[198]
        getitem_2465: "f32[]" = _foreach_reciprocal_default[199]
        getitem_2466: "f32[]" = _foreach_reciprocal_default[200]
        getitem_2467: "f32[]" = _foreach_reciprocal_default[201]
        getitem_2468: "f32[]" = _foreach_reciprocal_default[202]
        getitem_2469: "f32[]" = _foreach_reciprocal_default[203]
        getitem_2470: "f32[]" = _foreach_reciprocal_default[204]
        getitem_2471: "f32[]" = _foreach_reciprocal_default[205];  _foreach_reciprocal_default = None
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([getitem_2884, getitem_2885, getitem_2886, getitem_2887, getitem_2888, getitem_2889, getitem_2890, getitem_2891, getitem_2892, getitem_2893, getitem_2894, getitem_2895, getitem_2896, getitem_2897, getitem_2898, getitem_2899, getitem_2900, getitem_2901, getitem_2902, getitem_2903, getitem_2904, getitem_2905, getitem_2906, getitem_2907, getitem_2908, getitem_2909, getitem_2910, getitem_2911, getitem_2912, getitem_2913, getitem_2914, getitem_2915, getitem_2916, getitem_2917, getitem_2918, getitem_2919, getitem_2920, getitem_2921, getitem_2922, getitem_2923, getitem_2924, getitem_2925, getitem_2926, getitem_2927, getitem_2928, getitem_2929, getitem_2930, getitem_2931, getitem_2932, getitem_2933, getitem_2934, getitem_2935, getitem_2936, getitem_2937, getitem_2938, getitem_2939, getitem_2940, getitem_2941, getitem_2942, getitem_2943, getitem_2944, getitem_2945, getitem_2946, getitem_2947, getitem_2948, getitem_2949, getitem_2950, getitem_2951, getitem_2952, getitem_2953, getitem_2954, getitem_2955, getitem_2956, getitem_2957, getitem_2958, getitem_2959, getitem_2960, getitem_2961, getitem_2962, getitem_2963, getitem_2964, getitem_2965, getitem_2966, getitem_2967, getitem_2968, getitem_2969, getitem_2970, getitem_2971, getitem_2972, getitem_2973, getitem_2974, getitem_2975, getitem_2976, getitem_2977, getitem_2978, getitem_2979, getitem_2980, getitem_2981, getitem_2982, getitem_2983, getitem_2984, getitem_2985, getitem_2986, getitem_2987, getitem_2988, getitem_2989, getitem_2990, getitem_2991, getitem_2992, getitem_2993, getitem_2994, getitem_2995, getitem_2996, getitem_2997, getitem_2998, getitem_2999, getitem_3000, getitem_3001, getitem_3002, getitem_3003, getitem_3004, getitem_3005, getitem_3006, getitem_3007, getitem_3008, getitem_3009, getitem_3010, getitem_3011, getitem_3012, getitem_3013, getitem_3014, getitem_3015, getitem_3016, getitem_3017, getitem_3018, getitem_3019, getitem_3020, getitem_3021, getitem_3022, getitem_3023, getitem_3024, getitem_3025, getitem_3026, getitem_3027, getitem_3028, getitem_3029, getitem_3030, getitem_3031, getitem_3032, getitem_3033, getitem_3034, getitem_3035, getitem_3036, getitem_3037, getitem_3038, getitem_3039, getitem_3040, getitem_3041, getitem_3042, getitem_3043, getitem_3044, getitem_3045, getitem_3046, getitem_3047, getitem_3048, getitem_3049, getitem_3050, getitem_3051, getitem_3052, getitem_3053, getitem_3054, getitem_3055, getitem_3056, getitem_3057, getitem_3058, getitem_3059, getitem_3060, getitem_3061, getitem_3062, getitem_3063, getitem_3064, getitem_3065, getitem_3066, getitem_3067, getitem_3068, getitem_3069, getitem_3070, getitem_3071, getitem_3072, getitem_3073, getitem_3074, getitem_3075, getitem_3076, getitem_3077, getitem_3078, getitem_3079, getitem_3080, getitem_3081, getitem_3082, getitem_3083, getitem_3084, getitem_3085, getitem_3086, getitem_3087, getitem_3088, getitem_3089], 1e-08);  getitem_2884 = getitem_2885 = getitem_2886 = getitem_2887 = getitem_2888 = getitem_2889 = getitem_2890 = getitem_2891 = getitem_2892 = getitem_2893 = getitem_2894 = getitem_2895 = getitem_2896 = getitem_2897 = getitem_2898 = getitem_2899 = getitem_2900 = getitem_2901 = getitem_2902 = getitem_2903 = getitem_2904 = getitem_2905 = getitem_2906 = getitem_2907 = getitem_2908 = getitem_2909 = getitem_2910 = getitem_2911 = getitem_2912 = getitem_2913 = getitem_2914 = getitem_2915 = getitem_2916 = getitem_2917 = getitem_2918 = getitem_2919 = getitem_2920 = getitem_2921 = getitem_2922 = getitem_2923 = getitem_2924 = getitem_2925 = getitem_2926 = getitem_2927 = getitem_2928 = getitem_2929 = getitem_2930 = getitem_2931 = getitem_2932 = getitem_2933 = getitem_2934 = getitem_2935 = getitem_2936 = getitem_2937 = getitem_2938 = getitem_2939 = getitem_2940 = getitem_2941 = getitem_2942 = getitem_2943 = getitem_2944 = getitem_2945 = getitem_2946 = getitem_2947 = getitem_2948 = getitem_2949 = getitem_2950 = getitem_2951 = getitem_2952 = getitem_2953 = getitem_2954 = getitem_2955 = getitem_2956 = getitem_2957 = getitem_2958 = getitem_2959 = getitem_2960 = getitem_2961 = getitem_2962 = getitem_2963 = getitem_2964 = getitem_2965 = getitem_2966 = getitem_2967 = getitem_2968 = getitem_2969 = getitem_2970 = getitem_2971 = getitem_2972 = getitem_2973 = getitem_2974 = getitem_2975 = getitem_2976 = getitem_2977 = getitem_2978 = getitem_2979 = getitem_2980 = getitem_2981 = getitem_2982 = getitem_2983 = getitem_2984 = getitem_2985 = getitem_2986 = getitem_2987 = getitem_2988 = getitem_2989 = getitem_2990 = getitem_2991 = getitem_2992 = getitem_2993 = getitem_2994 = getitem_2995 = getitem_2996 = getitem_2997 = getitem_2998 = getitem_2999 = getitem_3000 = getitem_3001 = getitem_3002 = getitem_3003 = getitem_3004 = getitem_3005 = getitem_3006 = getitem_3007 = getitem_3008 = getitem_3009 = getitem_3010 = getitem_3011 = getitem_3012 = getitem_3013 = getitem_3014 = getitem_3015 = getitem_3016 = getitem_3017 = getitem_3018 = getitem_3019 = getitem_3020 = getitem_3021 = getitem_3022 = getitem_3023 = getitem_3024 = getitem_3025 = getitem_3026 = getitem_3027 = getitem_3028 = getitem_3029 = getitem_3030 = getitem_3031 = getitem_3032 = getitem_3033 = getitem_3034 = getitem_3035 = getitem_3036 = getitem_3037 = getitem_3038 = getitem_3039 = getitem_3040 = getitem_3041 = getitem_3042 = getitem_3043 = getitem_3044 = getitem_3045 = getitem_3046 = getitem_3047 = getitem_3048 = getitem_3049 = getitem_3050 = getitem_3051 = getitem_3052 = getitem_3053 = getitem_3054 = getitem_3055 = getitem_3056 = getitem_3057 = getitem_3058 = getitem_3059 = getitem_3060 = getitem_3061 = getitem_3062 = getitem_3063 = getitem_3064 = getitem_3065 = getitem_3066 = getitem_3067 = getitem_3068 = getitem_3069 = getitem_3070 = getitem_3071 = getitem_3072 = getitem_3073 = getitem_3074 = getitem_3075 = getitem_3076 = getitem_3077 = getitem_3078 = getitem_3079 = getitem_3080 = getitem_3081 = getitem_3082 = getitem_3083 = getitem_3084 = getitem_3085 = getitem_3086 = getitem_3087 = getitem_3088 = getitem_3089 = None
        getitem_3090: "f32[30522, 768]" = _foreach_add_scalar[0]
        getitem_3091: "f32[512, 768]" = _foreach_add_scalar[1]
        getitem_3092: "f32[1024, 768]" = _foreach_add_scalar[2]
        getitem_3093: "f32[1024, 768]" = _foreach_add_scalar[3]
        getitem_3094: "f32[1024, 768]" = _foreach_add_scalar[4]
        getitem_3095: "f32[1024, 768]" = _foreach_add_scalar[5]
        getitem_3096: "f32[2, 768]" = _foreach_add_scalar[6]
        getitem_3097: "f32[768]" = _foreach_add_scalar[7]
        getitem_3098: "f32[768]" = _foreach_add_scalar[8]
        getitem_3099: "f32[768, 768]" = _foreach_add_scalar[9]
        getitem_3100: "f32[768]" = _foreach_add_scalar[10]
        getitem_3101: "f32[768, 768]" = _foreach_add_scalar[11]
        getitem_3102: "f32[768]" = _foreach_add_scalar[12]
        getitem_3103: "f32[768, 768]" = _foreach_add_scalar[13]
        getitem_3104: "f32[768]" = _foreach_add_scalar[14]
        getitem_3105: "f32[768, 768]" = _foreach_add_scalar[15]
        getitem_3106: "f32[768]" = _foreach_add_scalar[16]
        getitem_3107: "f32[768]" = _foreach_add_scalar[17]
        getitem_3108: "f32[768]" = _foreach_add_scalar[18]
        getitem_3109: "f32[3072, 768]" = _foreach_add_scalar[19]
        getitem_3110: "f32[3072]" = _foreach_add_scalar[20]
        getitem_3111: "f32[768, 3072]" = _foreach_add_scalar[21]
        getitem_3112: "f32[768]" = _foreach_add_scalar[22]
        getitem_3113: "f32[768]" = _foreach_add_scalar[23]
        getitem_3114: "f32[768]" = _foreach_add_scalar[24]
        getitem_3115: "f32[768, 768]" = _foreach_add_scalar[25]
        getitem_3116: "f32[768]" = _foreach_add_scalar[26]
        getitem_3117: "f32[768, 768]" = _foreach_add_scalar[27]
        getitem_3118: "f32[768]" = _foreach_add_scalar[28]
        getitem_3119: "f32[768, 768]" = _foreach_add_scalar[29]
        getitem_3120: "f32[768]" = _foreach_add_scalar[30]
        getitem_3121: "f32[768, 768]" = _foreach_add_scalar[31]
        getitem_3122: "f32[768]" = _foreach_add_scalar[32]
        getitem_3123: "f32[768]" = _foreach_add_scalar[33]
        getitem_3124: "f32[768]" = _foreach_add_scalar[34]
        getitem_3125: "f32[3072, 768]" = _foreach_add_scalar[35]
        getitem_3126: "f32[3072]" = _foreach_add_scalar[36]
        getitem_3127: "f32[768, 3072]" = _foreach_add_scalar[37]
        getitem_3128: "f32[768]" = _foreach_add_scalar[38]
        getitem_3129: "f32[768]" = _foreach_add_scalar[39]
        getitem_3130: "f32[768]" = _foreach_add_scalar[40]
        getitem_3131: "f32[768, 768]" = _foreach_add_scalar[41]
        getitem_3132: "f32[768]" = _foreach_add_scalar[42]
        getitem_3133: "f32[768, 768]" = _foreach_add_scalar[43]
        getitem_3134: "f32[768]" = _foreach_add_scalar[44]
        getitem_3135: "f32[768, 768]" = _foreach_add_scalar[45]
        getitem_3136: "f32[768]" = _foreach_add_scalar[46]
        getitem_3137: "f32[768, 768]" = _foreach_add_scalar[47]
        getitem_3138: "f32[768]" = _foreach_add_scalar[48]
        getitem_3139: "f32[768]" = _foreach_add_scalar[49]
        getitem_3140: "f32[768]" = _foreach_add_scalar[50]
        getitem_3141: "f32[3072, 768]" = _foreach_add_scalar[51]
        getitem_3142: "f32[3072]" = _foreach_add_scalar[52]
        getitem_3143: "f32[768, 3072]" = _foreach_add_scalar[53]
        getitem_3144: "f32[768]" = _foreach_add_scalar[54]
        getitem_3145: "f32[768]" = _foreach_add_scalar[55]
        getitem_3146: "f32[768]" = _foreach_add_scalar[56]
        getitem_3147: "f32[768, 768]" = _foreach_add_scalar[57]
        getitem_3148: "f32[768]" = _foreach_add_scalar[58]
        getitem_3149: "f32[768, 768]" = _foreach_add_scalar[59]
        getitem_3150: "f32[768]" = _foreach_add_scalar[60]
        getitem_3151: "f32[768, 768]" = _foreach_add_scalar[61]
        getitem_3152: "f32[768]" = _foreach_add_scalar[62]
        getitem_3153: "f32[768, 768]" = _foreach_add_scalar[63]
        getitem_3154: "f32[768]" = _foreach_add_scalar[64]
        getitem_3155: "f32[768]" = _foreach_add_scalar[65]
        getitem_3156: "f32[768]" = _foreach_add_scalar[66]
        getitem_3157: "f32[3072, 768]" = _foreach_add_scalar[67]
        getitem_3158: "f32[3072]" = _foreach_add_scalar[68]
        getitem_3159: "f32[768, 3072]" = _foreach_add_scalar[69]
        getitem_3160: "f32[768]" = _foreach_add_scalar[70]
        getitem_3161: "f32[768]" = _foreach_add_scalar[71]
        getitem_3162: "f32[768]" = _foreach_add_scalar[72]
        getitem_3163: "f32[768, 768]" = _foreach_add_scalar[73]
        getitem_3164: "f32[768]" = _foreach_add_scalar[74]
        getitem_3165: "f32[768, 768]" = _foreach_add_scalar[75]
        getitem_3166: "f32[768]" = _foreach_add_scalar[76]
        getitem_3167: "f32[768, 768]" = _foreach_add_scalar[77]
        getitem_3168: "f32[768]" = _foreach_add_scalar[78]
        getitem_3169: "f32[768, 768]" = _foreach_add_scalar[79]
        getitem_3170: "f32[768]" = _foreach_add_scalar[80]
        getitem_3171: "f32[768]" = _foreach_add_scalar[81]
        getitem_3172: "f32[768]" = _foreach_add_scalar[82]
        getitem_3173: "f32[3072, 768]" = _foreach_add_scalar[83]
        getitem_3174: "f32[3072]" = _foreach_add_scalar[84]
        getitem_3175: "f32[768, 3072]" = _foreach_add_scalar[85]
        getitem_3176: "f32[768]" = _foreach_add_scalar[86]
        getitem_3177: "f32[768]" = _foreach_add_scalar[87]
        getitem_3178: "f32[768]" = _foreach_add_scalar[88]
        getitem_3179: "f32[768, 768]" = _foreach_add_scalar[89]
        getitem_3180: "f32[768]" = _foreach_add_scalar[90]
        getitem_3181: "f32[768, 768]" = _foreach_add_scalar[91]
        getitem_3182: "f32[768]" = _foreach_add_scalar[92]
        getitem_3183: "f32[768, 768]" = _foreach_add_scalar[93]
        getitem_3184: "f32[768]" = _foreach_add_scalar[94]
        getitem_3185: "f32[768, 768]" = _foreach_add_scalar[95]
        getitem_3186: "f32[768]" = _foreach_add_scalar[96]
        getitem_3187: "f32[768]" = _foreach_add_scalar[97]
        getitem_3188: "f32[768]" = _foreach_add_scalar[98]
        getitem_3189: "f32[3072, 768]" = _foreach_add_scalar[99]
        getitem_3190: "f32[3072]" = _foreach_add_scalar[100]
        getitem_3191: "f32[768, 3072]" = _foreach_add_scalar[101]
        getitem_3192: "f32[768]" = _foreach_add_scalar[102]
        getitem_3193: "f32[768]" = _foreach_add_scalar[103]
        getitem_3194: "f32[768]" = _foreach_add_scalar[104]
        getitem_3195: "f32[768, 768]" = _foreach_add_scalar[105]
        getitem_3196: "f32[768]" = _foreach_add_scalar[106]
        getitem_3197: "f32[768, 768]" = _foreach_add_scalar[107]
        getitem_3198: "f32[768]" = _foreach_add_scalar[108]
        getitem_3199: "f32[768, 768]" = _foreach_add_scalar[109]
        getitem_3200: "f32[768]" = _foreach_add_scalar[110]
        getitem_3201: "f32[768, 768]" = _foreach_add_scalar[111]
        getitem_3202: "f32[768]" = _foreach_add_scalar[112]
        getitem_3203: "f32[768]" = _foreach_add_scalar[113]
        getitem_3204: "f32[768]" = _foreach_add_scalar[114]
        getitem_3205: "f32[3072, 768]" = _foreach_add_scalar[115]
        getitem_3206: "f32[3072]" = _foreach_add_scalar[116]
        getitem_3207: "f32[768, 3072]" = _foreach_add_scalar[117]
        getitem_3208: "f32[768]" = _foreach_add_scalar[118]
        getitem_3209: "f32[768]" = _foreach_add_scalar[119]
        getitem_3210: "f32[768]" = _foreach_add_scalar[120]
        getitem_3211: "f32[768, 768]" = _foreach_add_scalar[121]
        getitem_3212: "f32[768]" = _foreach_add_scalar[122]
        getitem_3213: "f32[768, 768]" = _foreach_add_scalar[123]
        getitem_3214: "f32[768]" = _foreach_add_scalar[124]
        getitem_3215: "f32[768, 768]" = _foreach_add_scalar[125]
        getitem_3216: "f32[768]" = _foreach_add_scalar[126]
        getitem_3217: "f32[768, 768]" = _foreach_add_scalar[127]
        getitem_3218: "f32[768]" = _foreach_add_scalar[128]
        getitem_3219: "f32[768]" = _foreach_add_scalar[129]
        getitem_3220: "f32[768]" = _foreach_add_scalar[130]
        getitem_3221: "f32[3072, 768]" = _foreach_add_scalar[131]
        getitem_3222: "f32[3072]" = _foreach_add_scalar[132]
        getitem_3223: "f32[768, 3072]" = _foreach_add_scalar[133]
        getitem_3224: "f32[768]" = _foreach_add_scalar[134]
        getitem_3225: "f32[768]" = _foreach_add_scalar[135]
        getitem_3226: "f32[768]" = _foreach_add_scalar[136]
        getitem_3227: "f32[768, 768]" = _foreach_add_scalar[137]
        getitem_3228: "f32[768]" = _foreach_add_scalar[138]
        getitem_3229: "f32[768, 768]" = _foreach_add_scalar[139]
        getitem_3230: "f32[768]" = _foreach_add_scalar[140]
        getitem_3231: "f32[768, 768]" = _foreach_add_scalar[141]
        getitem_3232: "f32[768]" = _foreach_add_scalar[142]
        getitem_3233: "f32[768, 768]" = _foreach_add_scalar[143]
        getitem_3234: "f32[768]" = _foreach_add_scalar[144]
        getitem_3235: "f32[768]" = _foreach_add_scalar[145]
        getitem_3236: "f32[768]" = _foreach_add_scalar[146]
        getitem_3237: "f32[3072, 768]" = _foreach_add_scalar[147]
        getitem_3238: "f32[3072]" = _foreach_add_scalar[148]
        getitem_3239: "f32[768, 3072]" = _foreach_add_scalar[149]
        getitem_3240: "f32[768]" = _foreach_add_scalar[150]
        getitem_3241: "f32[768]" = _foreach_add_scalar[151]
        getitem_3242: "f32[768]" = _foreach_add_scalar[152]
        getitem_3243: "f32[768, 768]" = _foreach_add_scalar[153]
        getitem_3244: "f32[768]" = _foreach_add_scalar[154]
        getitem_3245: "f32[768, 768]" = _foreach_add_scalar[155]
        getitem_3246: "f32[768]" = _foreach_add_scalar[156]
        getitem_3247: "f32[768, 768]" = _foreach_add_scalar[157]
        getitem_3248: "f32[768]" = _foreach_add_scalar[158]
        getitem_3249: "f32[768, 768]" = _foreach_add_scalar[159]
        getitem_3250: "f32[768]" = _foreach_add_scalar[160]
        getitem_3251: "f32[768]" = _foreach_add_scalar[161]
        getitem_3252: "f32[768]" = _foreach_add_scalar[162]
        getitem_3253: "f32[3072, 768]" = _foreach_add_scalar[163]
        getitem_3254: "f32[3072]" = _foreach_add_scalar[164]
        getitem_3255: "f32[768, 3072]" = _foreach_add_scalar[165]
        getitem_3256: "f32[768]" = _foreach_add_scalar[166]
        getitem_3257: "f32[768]" = _foreach_add_scalar[167]
        getitem_3258: "f32[768]" = _foreach_add_scalar[168]
        getitem_3259: "f32[768, 768]" = _foreach_add_scalar[169]
        getitem_3260: "f32[768]" = _foreach_add_scalar[170]
        getitem_3261: "f32[768, 768]" = _foreach_add_scalar[171]
        getitem_3262: "f32[768]" = _foreach_add_scalar[172]
        getitem_3263: "f32[768, 768]" = _foreach_add_scalar[173]
        getitem_3264: "f32[768]" = _foreach_add_scalar[174]
        getitem_3265: "f32[768, 768]" = _foreach_add_scalar[175]
        getitem_3266: "f32[768]" = _foreach_add_scalar[176]
        getitem_3267: "f32[768]" = _foreach_add_scalar[177]
        getitem_3268: "f32[768]" = _foreach_add_scalar[178]
        getitem_3269: "f32[3072, 768]" = _foreach_add_scalar[179]
        getitem_3270: "f32[3072]" = _foreach_add_scalar[180]
        getitem_3271: "f32[768, 3072]" = _foreach_add_scalar[181]
        getitem_3272: "f32[768]" = _foreach_add_scalar[182]
        getitem_3273: "f32[768]" = _foreach_add_scalar[183]
        getitem_3274: "f32[768]" = _foreach_add_scalar[184]
        getitem_3275: "f32[768, 768]" = _foreach_add_scalar[185]
        getitem_3276: "f32[768]" = _foreach_add_scalar[186]
        getitem_3277: "f32[768, 768]" = _foreach_add_scalar[187]
        getitem_3278: "f32[768]" = _foreach_add_scalar[188]
        getitem_3279: "f32[768, 768]" = _foreach_add_scalar[189]
        getitem_3280: "f32[768]" = _foreach_add_scalar[190]
        getitem_3281: "f32[768, 768]" = _foreach_add_scalar[191]
        getitem_3282: "f32[768]" = _foreach_add_scalar[192]
        getitem_3283: "f32[768]" = _foreach_add_scalar[193]
        getitem_3284: "f32[768]" = _foreach_add_scalar[194]
        getitem_3285: "f32[3072, 768]" = _foreach_add_scalar[195]
        getitem_3286: "f32[3072]" = _foreach_add_scalar[196]
        getitem_3287: "f32[768, 3072]" = _foreach_add_scalar[197]
        getitem_3288: "f32[768]" = _foreach_add_scalar[198]
        getitem_3289: "f32[768]" = _foreach_add_scalar[199]
        getitem_3290: "f32[768]" = _foreach_add_scalar[200]
        getitem_3291: "f32[30522]" = _foreach_add_scalar[201]
        getitem_3292: "f32[768, 768]" = _foreach_add_scalar[202]
        getitem_3293: "f32[768]" = _foreach_add_scalar[203]
        getitem_3294: "f32[768]" = _foreach_add_scalar[204]
        getitem_3295: "f32[768]" = _foreach_add_scalar[205];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_2266, getitem_2267, getitem_2268, getitem_2269, getitem_2270, getitem_2271, getitem_2272, getitem_2273, getitem_2274, getitem_2275, getitem_2276, getitem_2277, getitem_2278, getitem_2279, getitem_2280, getitem_2281, getitem_2282, getitem_2283, getitem_2284, getitem_2285, getitem_2286, getitem_2287, getitem_2288, getitem_2289, getitem_2290, getitem_2291, getitem_2292, getitem_2293, getitem_2294, getitem_2295, getitem_2296, getitem_2297, getitem_2298, getitem_2299, getitem_2300, getitem_2301, getitem_2302, getitem_2303, getitem_2304, getitem_2305, getitem_2306, getitem_2307, getitem_2308, getitem_2309, getitem_2310, getitem_2311, getitem_2312, getitem_2313, getitem_2314, getitem_2315, getitem_2316, getitem_2317, getitem_2318, getitem_2319, getitem_2320, getitem_2321, getitem_2322, getitem_2323, getitem_2324, getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406, getitem_2407, getitem_2408, getitem_2409, getitem_2410, getitem_2411, getitem_2412, getitem_2413, getitem_2414, getitem_2415, getitem_2416, getitem_2417, getitem_2418, getitem_2419, getitem_2420, getitem_2421, getitem_2422, getitem_2423, getitem_2424, getitem_2425, getitem_2426, getitem_2427, getitem_2428, getitem_2429, getitem_2430, getitem_2431, getitem_2432, getitem_2433, getitem_2434, getitem_2435, getitem_2436, getitem_2437, getitem_2438, getitem_2439, getitem_2440, getitem_2441, getitem_2442, getitem_2443, getitem_2444, getitem_2445, getitem_2446, getitem_2447, getitem_2448, getitem_2449, getitem_2450, getitem_2451, getitem_2452, getitem_2453, getitem_2454, getitem_2455, getitem_2456, getitem_2457, getitem_2458, getitem_2459, getitem_2460, getitem_2461, getitem_2462, getitem_2463, getitem_2464, getitem_2465, getitem_2466, getitem_2467, getitem_2468, getitem_2469, getitem_2470, getitem_2471, getitem_3090, getitem_3091, getitem_3092, getitem_3093, getitem_3094, getitem_3095, getitem_3096, getitem_3097, getitem_3098, getitem_3099, getitem_3100, getitem_3101, getitem_3102, getitem_3103, getitem_3104, getitem_3105, getitem_3106, getitem_3107, getitem_3108, getitem_3109, getitem_3110, getitem_3111, getitem_3112, getitem_3113, getitem_3114, getitem_3115, getitem_3116, getitem_3117, getitem_3118, getitem_3119, getitem_3120, getitem_3121, getitem_3122, getitem_3123, getitem_3124, getitem_3125, getitem_3126, getitem_3127, getitem_3128, getitem_3129, getitem_3130, getitem_3131, getitem_3132, getitem_3133, getitem_3134, getitem_3135, getitem_3136, getitem_3137, getitem_3138, getitem_3139, getitem_3140, getitem_3141, getitem_3142, getitem_3143, getitem_3144, getitem_3145, getitem_3146, getitem_3147, getitem_3148, getitem_3149, getitem_3150, getitem_3151, getitem_3152, getitem_3153, getitem_3154, getitem_3155, getitem_3156, getitem_3157, getitem_3158, getitem_3159, getitem_3160, getitem_3161, getitem_3162, getitem_3163, getitem_3164, getitem_3165, getitem_3166, getitem_3167, getitem_3168, getitem_3169, getitem_3170, getitem_3171, getitem_3172, getitem_3173, getitem_3174, getitem_3175, getitem_3176, getitem_3177, getitem_3178, getitem_3179, getitem_3180, getitem_3181, getitem_3182, getitem_3183, getitem_3184, getitem_3185, getitem_3186, getitem_3187, getitem_3188, getitem_3189, getitem_3190, getitem_3191, getitem_3192, getitem_3193, getitem_3194, getitem_3195, getitem_3196, getitem_3197, getitem_3198, getitem_3199, getitem_3200, getitem_3201, getitem_3202, getitem_3203, getitem_3204, getitem_3205, getitem_3206, getitem_3207, getitem_3208, getitem_3209, getitem_3210, getitem_3211, getitem_3212, getitem_3213, getitem_3214, getitem_3215, getitem_3216, getitem_3217, getitem_3218, getitem_3219, getitem_3220, getitem_3221, getitem_3222, getitem_3223, getitem_3224, getitem_3225, getitem_3226, getitem_3227, getitem_3228, getitem_3229, getitem_3230, getitem_3231, getitem_3232, getitem_3233, getitem_3234, getitem_3235, getitem_3236, getitem_3237, getitem_3238, getitem_3239, getitem_3240, getitem_3241, getitem_3242, getitem_3243, getitem_3244, getitem_3245, getitem_3246, getitem_3247, getitem_3248, getitem_3249, getitem_3250, getitem_3251, getitem_3252, getitem_3253, getitem_3254, getitem_3255, getitem_3256, getitem_3257, getitem_3258, getitem_3259, getitem_3260, getitem_3261, getitem_3262, getitem_3263, getitem_3264, getitem_3265, getitem_3266, getitem_3267, getitem_3268, getitem_3269, getitem_3270, getitem_3271, getitem_3272, getitem_3273, getitem_3274, getitem_3275, getitem_3276, getitem_3277, getitem_3278, getitem_3279, getitem_3280, getitem_3281, getitem_3282, getitem_3283, getitem_3284, getitem_3285, getitem_3286, getitem_3287, getitem_3288, getitem_3289, getitem_3290, getitem_3291, getitem_3292, getitem_3293, getitem_3294, getitem_3295)


def _default_make_inputs():
    return [
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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

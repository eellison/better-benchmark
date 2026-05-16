"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g77
Pattern hash: 8a74d6667e85
Shape hash: 0f19318e
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg158_1: "f32[1, 1, 768]", arg156_1: "f32[1, 198, 768]", arg314_1: "f32[1, 1, 768]", arg315_1: "f32[768, 3, 16, 16]", arg316_1: "f32[768]", arg317_1: "f32[768]", arg318_1: "f32[768]", arg319_1: "f32[2304, 768]", arg320_1: "f32[2304]", arg321_1: "f32[768, 768]", arg322_1: "f32[768]", arg323_1: "f32[768]", arg324_1: "f32[768]", arg325_1: "f32[3072, 768]", arg326_1: "f32[3072]", arg327_1: "f32[768, 3072]", arg328_1: "f32[768]", arg329_1: "f32[768]", arg330_1: "f32[768]", arg331_1: "f32[2304, 768]", arg332_1: "f32[2304]", arg333_1: "f32[768, 768]", arg334_1: "f32[768]", arg335_1: "f32[768]", arg336_1: "f32[768]", arg337_1: "f32[3072, 768]", arg338_1: "f32[3072]", arg339_1: "f32[768, 3072]", arg340_1: "f32[768]", arg341_1: "f32[768]", arg342_1: "f32[768]", arg343_1: "f32[2304, 768]", arg344_1: "f32[2304]", arg345_1: "f32[768, 768]", arg346_1: "f32[768]", arg347_1: "f32[768]", arg348_1: "f32[768]", arg349_1: "f32[3072, 768]", arg350_1: "f32[3072]", arg351_1: "f32[768, 3072]", arg352_1: "f32[768]", arg353_1: "f32[768]", arg354_1: "f32[768]", arg355_1: "f32[2304, 768]", arg356_1: "f32[2304]", arg357_1: "f32[768, 768]", arg358_1: "f32[768]", arg359_1: "f32[768]", arg360_1: "f32[768]", arg361_1: "f32[3072, 768]", arg362_1: "f32[3072]", arg363_1: "f32[768, 3072]", arg364_1: "f32[768]", arg365_1: "f32[768]", arg366_1: "f32[768]", arg367_1: "f32[2304, 768]", arg368_1: "f32[2304]", arg369_1: "f32[768, 768]", arg370_1: "f32[768]", arg371_1: "f32[768]", arg372_1: "f32[768]", arg373_1: "f32[3072, 768]", arg374_1: "f32[3072]", arg375_1: "f32[768, 3072]", arg376_1: "f32[768]", arg377_1: "f32[768]", arg378_1: "f32[768]", arg379_1: "f32[2304, 768]", arg380_1: "f32[2304]", arg381_1: "f32[768, 768]", arg382_1: "f32[768]", arg383_1: "f32[768]", arg384_1: "f32[768]", arg385_1: "f32[3072, 768]", arg386_1: "f32[3072]", arg387_1: "f32[768, 3072]", arg388_1: "f32[768]", arg389_1: "f32[768]", arg390_1: "f32[768]", arg391_1: "f32[2304, 768]", arg392_1: "f32[2304]", arg393_1: "f32[768, 768]", arg394_1: "f32[768]", arg395_1: "f32[768]", arg396_1: "f32[768]", arg397_1: "f32[3072, 768]", arg398_1: "f32[3072]", arg399_1: "f32[768, 3072]", arg400_1: "f32[768]", arg401_1: "f32[768]", arg402_1: "f32[768]", arg403_1: "f32[2304, 768]", arg404_1: "f32[2304]", arg405_1: "f32[768, 768]", arg406_1: "f32[768]", arg407_1: "f32[768]", arg408_1: "f32[768]", arg409_1: "f32[3072, 768]", arg410_1: "f32[3072]", arg411_1: "f32[768, 3072]", arg412_1: "f32[768]", arg413_1: "f32[768]", arg414_1: "f32[768]", arg415_1: "f32[2304, 768]", arg416_1: "f32[2304]", arg417_1: "f32[768, 768]", arg418_1: "f32[768]", arg419_1: "f32[768]", arg420_1: "f32[768]", arg421_1: "f32[3072, 768]", arg422_1: "f32[3072]", arg423_1: "f32[768, 3072]", arg424_1: "f32[768]", arg425_1: "f32[768]", arg426_1: "f32[768]", arg427_1: "f32[2304, 768]", arg428_1: "f32[2304]", arg429_1: "f32[768, 768]", arg430_1: "f32[768]", arg431_1: "f32[768]", arg432_1: "f32[768]", arg433_1: "f32[3072, 768]", arg434_1: "f32[3072]", arg435_1: "f32[768, 3072]", arg436_1: "f32[768]", arg437_1: "f32[768]", arg438_1: "f32[768]", arg439_1: "f32[2304, 768]", arg440_1: "f32[2304]", arg441_1: "f32[768, 768]", arg442_1: "f32[768]", arg443_1: "f32[768]", arg444_1: "f32[768]", arg445_1: "f32[3072, 768]", arg446_1: "f32[3072]", arg447_1: "f32[768, 3072]", arg448_1: "f32[768]", arg449_1: "f32[768]", arg450_1: "f32[768]", arg451_1: "f32[2304, 768]", arg452_1: "f32[2304]", arg453_1: "f32[768, 768]", arg454_1: "f32[768]", arg455_1: "f32[768]", arg456_1: "f32[768]", arg457_1: "f32[3072, 768]", arg458_1: "f32[3072]", arg459_1: "f32[768, 3072]", arg460_1: "f32[768]", arg461_1: "f32[768]", arg462_1: "f32[768]", arg463_1: "f32[1000, 768]", arg464_1: "f32[1000]", arg465_1: "f32[1000, 768]", arg466_1: "f32[1000]", getitem_155: "f32[1, 1, 768]", getitem_156: "f32[1, 198, 768]", getitem_157: "f32[1, 1, 768]", getitem_158: "f32[768, 3, 16, 16]", getitem_159: "f32[768]", getitem_160: "f32[768]", getitem_161: "f32[768]", getitem_162: "f32[2304, 768]", getitem_163: "f32[2304]", getitem_164: "f32[768, 768]", getitem_165: "f32[768]", getitem_166: "f32[768]", getitem_167: "f32[768]", getitem_168: "f32[3072, 768]", getitem_169: "f32[3072]", getitem_170: "f32[768, 3072]", getitem_171: "f32[768]", getitem_172: "f32[768]", getitem_173: "f32[768]", getitem_174: "f32[2304, 768]", getitem_175: "f32[2304]", getitem_176: "f32[768, 768]", getitem_177: "f32[768]", getitem_178: "f32[768]", getitem_179: "f32[768]", getitem_180: "f32[3072, 768]", getitem_181: "f32[3072]", getitem_182: "f32[768, 3072]", getitem_183: "f32[768]", getitem_184: "f32[768]", getitem_185: "f32[768]", getitem_186: "f32[2304, 768]", getitem_187: "f32[2304]", getitem_188: "f32[768, 768]", getitem_189: "f32[768]", getitem_190: "f32[768]", getitem_191: "f32[768]", getitem_192: "f32[3072, 768]", getitem_193: "f32[3072]", getitem_194: "f32[768, 3072]", getitem_195: "f32[768]", getitem_196: "f32[768]", getitem_197: "f32[768]", getitem_198: "f32[2304, 768]", getitem_199: "f32[2304]", getitem_200: "f32[768, 768]", getitem_201: "f32[768]", getitem_202: "f32[768]", getitem_203: "f32[768]", getitem_204: "f32[3072, 768]", getitem_205: "f32[3072]", getitem_206: "f32[768, 3072]", getitem_207: "f32[768]", getitem_208: "f32[768]", getitem_209: "f32[768]", getitem_210: "f32[2304, 768]", getitem_211: "f32[2304]", getitem_212: "f32[768, 768]", getitem_213: "f32[768]", getitem_214: "f32[768]", getitem_215: "f32[768]", getitem_216: "f32[3072, 768]", getitem_217: "f32[3072]", getitem_218: "f32[768, 3072]", getitem_219: "f32[768]", getitem_220: "f32[768]", getitem_221: "f32[768]", getitem_222: "f32[2304, 768]", getitem_223: "f32[2304]", getitem_224: "f32[768, 768]", getitem_225: "f32[768]", getitem_226: "f32[768]", getitem_227: "f32[768]", getitem_228: "f32[3072, 768]", getitem_229: "f32[3072]", getitem_230: "f32[768, 3072]", getitem_231: "f32[768]", getitem_232: "f32[768]", getitem_233: "f32[768]", getitem_234: "f32[2304, 768]", getitem_235: "f32[2304]", getitem_236: "f32[768, 768]", getitem_237: "f32[768]", getitem_238: "f32[768]", getitem_239: "f32[768]", getitem_240: "f32[3072, 768]", getitem_241: "f32[3072]", getitem_242: "f32[768, 3072]", getitem_243: "f32[768]", getitem_244: "f32[768]", getitem_245: "f32[768]", getitem_246: "f32[2304, 768]", getitem_247: "f32[2304]", getitem_248: "f32[768, 768]", getitem_249: "f32[768]", getitem_250: "f32[768]", getitem_251: "f32[768]", getitem_252: "f32[3072, 768]", getitem_253: "f32[3072]", getitem_254: "f32[768, 3072]", getitem_255: "f32[768]", getitem_256: "f32[768]", getitem_257: "f32[768]", getitem_258: "f32[2304, 768]", getitem_259: "f32[2304]", getitem_260: "f32[768, 768]", getitem_261: "f32[768]", getitem_262: "f32[768]", getitem_263: "f32[768]", getitem_264: "f32[3072, 768]", getitem_265: "f32[3072]", getitem_266: "f32[768, 3072]", getitem_267: "f32[768]", getitem_268: "f32[768]", getitem_269: "f32[768]", getitem_270: "f32[2304, 768]", getitem_271: "f32[2304]", getitem_272: "f32[768, 768]", getitem_273: "f32[768]", getitem_274: "f32[768]", getitem_275: "f32[768]", getitem_276: "f32[3072, 768]", getitem_277: "f32[3072]", getitem_278: "f32[768, 3072]", getitem_279: "f32[768]", getitem_280: "f32[768]", getitem_281: "f32[768]", getitem_282: "f32[2304, 768]", getitem_283: "f32[2304]", getitem_284: "f32[768, 768]", getitem_285: "f32[768]", getitem_286: "f32[768]", getitem_287: "f32[768]", getitem_288: "f32[3072, 768]", getitem_289: "f32[3072]", getitem_290: "f32[768, 3072]", getitem_291: "f32[768]", getitem_292: "f32[768]", getitem_293: "f32[768]", getitem_294: "f32[2304, 768]", getitem_295: "f32[2304]", getitem_296: "f32[768, 768]", getitem_297: "f32[768]", getitem_298: "f32[768]", getitem_299: "f32[768]", getitem_300: "f32[3072, 768]", getitem_301: "f32[3072]", getitem_302: "f32[768, 3072]", getitem_303: "f32[768]", getitem_304: "f32[768]", getitem_305: "f32[768]", getitem_306: "f32[1000, 768]", getitem_307: "f32[1000]", getitem_308: "f32[1000, 768]", getitem_309: "f32[1000]", getitem_2325: "f32[1, 1, 768]", getitem_2326: "f32[1, 198, 768]", getitem_2327: "f32[1, 1, 768]", getitem_2328: "f32[768, 3, 16, 16]", getitem_2329: "f32[768]", getitem_2330: "f32[768]", getitem_2331: "f32[768]", getitem_2332: "f32[2304, 768]", getitem_2333: "f32[2304]", getitem_2334: "f32[768, 768]", getitem_2335: "f32[768]", getitem_2336: "f32[768]", getitem_2337: "f32[768]", getitem_2338: "f32[3072, 768]", getitem_2339: "f32[3072]", getitem_2340: "f32[768, 3072]", getitem_2341: "f32[768]", getitem_2342: "f32[768]", getitem_2343: "f32[768]", getitem_2344: "f32[2304, 768]", getitem_2345: "f32[2304]", getitem_2346: "f32[768, 768]", getitem_2347: "f32[768]", getitem_2348: "f32[768]", getitem_2349: "f32[768]", getitem_2350: "f32[3072, 768]", getitem_2351: "f32[3072]", getitem_2352: "f32[768, 3072]", getitem_2353: "f32[768]", getitem_2354: "f32[768]", getitem_2355: "f32[768]", getitem_2356: "f32[2304, 768]", getitem_2357: "f32[2304]", getitem_2358: "f32[768, 768]", getitem_2359: "f32[768]", getitem_2360: "f32[768]", getitem_2361: "f32[768]", getitem_2362: "f32[3072, 768]", getitem_2363: "f32[3072]", getitem_2364: "f32[768, 3072]", getitem_2365: "f32[768]", getitem_2366: "f32[768]", getitem_2367: "f32[768]", getitem_2368: "f32[2304, 768]", getitem_2369: "f32[2304]", getitem_2370: "f32[768, 768]", getitem_2371: "f32[768]", getitem_2372: "f32[768]", getitem_2373: "f32[768]", getitem_2374: "f32[3072, 768]", getitem_2375: "f32[3072]", getitem_2376: "f32[768, 3072]", getitem_2377: "f32[768]", getitem_2378: "f32[768]", getitem_2379: "f32[768]", getitem_2380: "f32[2304, 768]", getitem_2381: "f32[2304]", getitem_2382: "f32[768, 768]", getitem_2383: "f32[768]", getitem_2384: "f32[768]", getitem_2385: "f32[768]", getitem_2386: "f32[3072, 768]", getitem_2387: "f32[3072]", getitem_2388: "f32[768, 3072]", getitem_2389: "f32[768]", getitem_2390: "f32[768]", getitem_2391: "f32[768]", getitem_2392: "f32[2304, 768]", getitem_2393: "f32[2304]", getitem_2394: "f32[768, 768]", getitem_2395: "f32[768]", getitem_2396: "f32[768]", getitem_2397: "f32[768]", getitem_2398: "f32[3072, 768]", getitem_2399: "f32[3072]", getitem_2400: "f32[768, 3072]", getitem_2401: "f32[768]", getitem_2402: "f32[768]", getitem_2403: "f32[768]", getitem_2404: "f32[2304, 768]", getitem_2405: "f32[2304]", getitem_2406: "f32[768, 768]", getitem_2407: "f32[768]", getitem_2408: "f32[768]", getitem_2409: "f32[768]", getitem_2410: "f32[3072, 768]", getitem_2411: "f32[3072]", getitem_2412: "f32[768, 3072]", getitem_2413: "f32[768]", getitem_2414: "f32[768]", getitem_2415: "f32[768]", getitem_2416: "f32[2304, 768]", getitem_2417: "f32[2304]", getitem_2418: "f32[768, 768]", getitem_2419: "f32[768]", getitem_2420: "f32[768]", getitem_2421: "f32[768]", getitem_2422: "f32[3072, 768]", getitem_2423: "f32[3072]", getitem_2424: "f32[768, 3072]", getitem_2425: "f32[768]", getitem_2426: "f32[768]", getitem_2427: "f32[768]", getitem_2428: "f32[2304, 768]", getitem_2429: "f32[2304]", getitem_2430: "f32[768, 768]", getitem_2431: "f32[768]", getitem_2432: "f32[768]", getitem_2433: "f32[768]", getitem_2434: "f32[3072, 768]", getitem_2435: "f32[3072]", getitem_2436: "f32[768, 3072]", getitem_2437: "f32[768]", getitem_2438: "f32[768]", getitem_2439: "f32[768]", getitem_2440: "f32[2304, 768]", getitem_2441: "f32[2304]", getitem_2442: "f32[768, 768]", getitem_2443: "f32[768]", getitem_2444: "f32[768]", getitem_2445: "f32[768]", getitem_2446: "f32[3072, 768]", getitem_2447: "f32[3072]", getitem_2448: "f32[768, 3072]", getitem_2449: "f32[768]", getitem_2450: "f32[768]", getitem_2451: "f32[768]", getitem_2452: "f32[2304, 768]", getitem_2453: "f32[2304]", getitem_2454: "f32[768, 768]", getitem_2455: "f32[768]", getitem_2456: "f32[768]", getitem_2457: "f32[768]", getitem_2458: "f32[3072, 768]", getitem_2459: "f32[3072]", getitem_2460: "f32[768, 3072]", getitem_2461: "f32[768]", getitem_2462: "f32[768]", getitem_2463: "f32[768]", getitem_2464: "f32[2304, 768]", getitem_2465: "f32[2304]", getitem_2466: "f32[768, 768]", getitem_2467: "f32[768]", getitem_2468: "f32[768]", getitem_2469: "f32[768]", getitem_2470: "f32[3072, 768]", getitem_2471: "f32[3072]", getitem_2472: "f32[768, 3072]", getitem_2473: "f32[768]", getitem_2474: "f32[768]", getitem_2475: "f32[768]", getitem_2476: "f32[1000, 768]", getitem_2477: "f32[1000]", getitem_2478: "f32[1000, 768]", getitem_2479: "f32[1000]", getitem_1705: "f32[]", getitem_1706: "f32[]", getitem_1707: "f32[]", getitem_1708: "f32[]", getitem_1709: "f32[]", getitem_1710: "f32[]", getitem_1711: "f32[]", getitem_1712: "f32[]", getitem_1713: "f32[]", getitem_1714: "f32[]", getitem_1715: "f32[]", getitem_1716: "f32[]", getitem_1717: "f32[]", getitem_1718: "f32[]", getitem_1719: "f32[]", getitem_1720: "f32[]", getitem_1721: "f32[]", getitem_1722: "f32[]", getitem_1723: "f32[]", getitem_1724: "f32[]", getitem_1725: "f32[]", getitem_1726: "f32[]", getitem_1727: "f32[]", getitem_1728: "f32[]", getitem_1729: "f32[]", getitem_1730: "f32[]", getitem_1731: "f32[]", getitem_1732: "f32[]", getitem_1733: "f32[]", getitem_1734: "f32[]", getitem_1735: "f32[]", getitem_1736: "f32[]", getitem_1737: "f32[]", getitem_1738: "f32[]", getitem_1739: "f32[]", getitem_1740: "f32[]", getitem_1741: "f32[]", getitem_1742: "f32[]", getitem_1743: "f32[]", getitem_1744: "f32[]", getitem_1745: "f32[]", getitem_1746: "f32[]", getitem_1747: "f32[]", getitem_1748: "f32[]", getitem_1749: "f32[]", getitem_1750: "f32[]", getitem_1751: "f32[]", getitem_1752: "f32[]", getitem_1753: "f32[]", getitem_1754: "f32[]", getitem_1755: "f32[]", getitem_1756: "f32[]", getitem_1757: "f32[]", getitem_1758: "f32[]", getitem_1759: "f32[]", getitem_1760: "f32[]", getitem_1761: "f32[]", getitem_1762: "f32[]", getitem_1763: "f32[]", getitem_1764: "f32[]", getitem_1765: "f32[]", getitem_1766: "f32[]", getitem_1767: "f32[]", getitem_1768: "f32[]", getitem_1769: "f32[]", getitem_1770: "f32[]", getitem_1771: "f32[]", getitem_1772: "f32[]", getitem_1773: "f32[]", getitem_1774: "f32[]", getitem_1775: "f32[]", getitem_1776: "f32[]", getitem_1777: "f32[]", getitem_1778: "f32[]", getitem_1779: "f32[]", getitem_1780: "f32[]", getitem_1781: "f32[]", getitem_1782: "f32[]", getitem_1783: "f32[]", getitem_1784: "f32[]", getitem_1785: "f32[]", getitem_1786: "f32[]", getitem_1787: "f32[]", getitem_1788: "f32[]", getitem_1789: "f32[]", getitem_1790: "f32[]", getitem_1791: "f32[]", getitem_1792: "f32[]", getitem_1793: "f32[]", getitem_1794: "f32[]", getitem_1795: "f32[]", getitem_1796: "f32[]", getitem_1797: "f32[]", getitem_1798: "f32[]", getitem_1799: "f32[]", getitem_1800: "f32[]", getitem_1801: "f32[]", getitem_1802: "f32[]", getitem_1803: "f32[]", getitem_1804: "f32[]", getitem_1805: "f32[]", getitem_1806: "f32[]", getitem_1807: "f32[]", getitem_1808: "f32[]", getitem_1809: "f32[]", getitem_1810: "f32[]", getitem_1811: "f32[]", getitem_1812: "f32[]", getitem_1813: "f32[]", getitem_1814: "f32[]", getitem_1815: "f32[]", getitem_1816: "f32[]", getitem_1817: "f32[]", getitem_1818: "f32[]", getitem_1819: "f32[]", getitem_1820: "f32[]", getitem_1821: "f32[]", getitem_1822: "f32[]", getitem_1823: "f32[]", getitem_1824: "f32[]", getitem_1825: "f32[]", getitem_1826: "f32[]", getitem_1827: "f32[]", getitem_1828: "f32[]", getitem_1829: "f32[]", getitem_1830: "f32[]", getitem_1831: "f32[]", getitem_1832: "f32[]", getitem_1833: "f32[]", getitem_1834: "f32[]", getitem_1835: "f32[]", getitem_1836: "f32[]", getitem_1837: "f32[]", getitem_1838: "f32[]", getitem_1839: "f32[]", getitem_1840: "f32[]", getitem_1841: "f32[]", getitem_1842: "f32[]", getitem_1843: "f32[]", getitem_1844: "f32[]", getitem_1845: "f32[]", getitem_1846: "f32[]", getitem_1847: "f32[]", getitem_1848: "f32[]", getitem_1849: "f32[]", getitem_1850: "f32[]", getitem_1851: "f32[]", getitem_1852: "f32[]", getitem_1853: "f32[]", getitem_1854: "f32[]", getitem_1855: "f32[]", getitem_1856: "f32[]", getitem_1857: "f32[]", getitem_1858: "f32[]", getitem_1859: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[1, 198, 768]" = torch.ops.aten.full.default([1, 198, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[1, 1, 768]" = torch.ops.aten.full.default([1, 1, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[768, 3, 16, 16]" = torch.ops.aten.full.default([768, 3, 16, 16], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_75: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_77: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_79: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_85: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_87: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_95: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_97: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_99: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_105: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_107: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_109: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_110: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_112: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_113: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_114: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_115: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_117: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_119: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_120: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_121: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_122: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_123: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_124: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_125: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_126: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_127: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_128: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_129: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_130: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_131: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_132: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_133: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_134: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_135: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_136: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_137: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_138: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_139: "f32[2304, 768]" = torch.ops.aten.full.default([2304, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_140: "f32[2304]" = torch.ops.aten.full.default([2304], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_141: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_142: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_143: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_144: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_145: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_146: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_147: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_148: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_149: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_150: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_151: "f32[1000, 768]" = torch.ops.aten.full.default([1000, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_152: "f32[1000]" = torch.ops.aten.full.default([1000], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_153: "f32[1000, 768]" = torch.ops.aten.full.default([1000, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_154: "f32[1000]" = torch.ops.aten.full.default([1000], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg158_1, arg156_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_37, full_default_38, full_default_39, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_53, full_default_54, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_65, full_default_66, full_default_67, full_default_68, full_default_69, full_default_70, full_default_71, full_default_72, full_default_73, full_default_74, full_default_75, full_default_76, full_default_77, full_default_78, full_default_79, full_default_80, full_default_81, full_default_82, full_default_83, full_default_84, full_default_85, full_default_86, full_default_87, full_default_88, full_default_89, full_default_90, full_default_91, full_default_92, full_default_93, full_default_94, full_default_95, full_default_96, full_default_97, full_default_98, full_default_99, full_default_100, full_default_101, full_default_102, full_default_103, full_default_104, full_default_105, full_default_106, full_default_107, full_default_108, full_default_109, full_default_110, full_default_111, full_default_112, full_default_113, full_default_114, full_default_115, full_default_116, full_default_117, full_default_118, full_default_119, full_default_120, full_default_121, full_default_122, full_default_123, full_default_124, full_default_125, full_default_126, full_default_127, full_default_128, full_default_129, full_default_130, full_default_131, full_default_132, full_default_133, full_default_134, full_default_135, full_default_136, full_default_137, full_default_138, full_default_139, full_default_140, full_default_141, full_default_142, full_default_143, full_default_144, full_default_145, full_default_146, full_default_147, full_default_148, full_default_149, full_default_150, full_default_151, full_default_152, full_default_153, full_default_154], [getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309]);  arg158_1 = arg156_1 = arg314_1 = arg315_1 = arg316_1 = arg317_1 = arg318_1 = arg319_1 = arg320_1 = arg321_1 = arg322_1 = arg323_1 = arg324_1 = arg325_1 = arg326_1 = arg327_1 = arg328_1 = arg329_1 = arg330_1 = arg331_1 = arg332_1 = arg333_1 = arg334_1 = arg335_1 = arg336_1 = arg337_1 = arg338_1 = arg339_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = full_default_10 = full_default_11 = full_default_12 = full_default_13 = full_default_14 = full_default_15 = full_default_16 = full_default_17 = full_default_18 = full_default_19 = full_default_20 = full_default_21 = full_default_22 = full_default_23 = full_default_24 = full_default_25 = full_default_26 = full_default_27 = full_default_28 = full_default_29 = full_default_30 = full_default_31 = full_default_32 = full_default_33 = full_default_34 = full_default_35 = full_default_36 = full_default_37 = full_default_38 = full_default_39 = full_default_40 = full_default_41 = full_default_42 = full_default_43 = full_default_44 = full_default_45 = full_default_46 = full_default_47 = full_default_48 = full_default_49 = full_default_50 = full_default_51 = full_default_52 = full_default_53 = full_default_54 = full_default_55 = full_default_56 = full_default_57 = full_default_58 = full_default_59 = full_default_60 = full_default_61 = full_default_62 = full_default_63 = full_default_64 = full_default_65 = full_default_66 = full_default_67 = full_default_68 = full_default_69 = full_default_70 = full_default_71 = full_default_72 = full_default_73 = full_default_74 = full_default_75 = full_default_76 = full_default_77 = full_default_78 = full_default_79 = full_default_80 = full_default_81 = full_default_82 = full_default_83 = full_default_84 = full_default_85 = full_default_86 = full_default_87 = full_default_88 = full_default_89 = full_default_90 = full_default_91 = full_default_92 = full_default_93 = full_default_94 = full_default_95 = full_default_96 = full_default_97 = full_default_98 = full_default_99 = full_default_100 = full_default_101 = full_default_102 = full_default_103 = full_default_104 = full_default_105 = full_default_106 = full_default_107 = full_default_108 = full_default_109 = full_default_110 = full_default_111 = full_default_112 = full_default_113 = full_default_114 = full_default_115 = full_default_116 = full_default_117 = full_default_118 = full_default_119 = full_default_120 = full_default_121 = full_default_122 = full_default_123 = full_default_124 = full_default_125 = full_default_126 = full_default_127 = full_default_128 = full_default_129 = full_default_130 = full_default_131 = full_default_132 = full_default_133 = full_default_134 = full_default_135 = full_default_136 = full_default_137 = full_default_138 = full_default_139 = full_default_140 = full_default_141 = full_default_142 = full_default_143 = full_default_144 = full_default_145 = full_default_146 = full_default_147 = full_default_148 = full_default_149 = full_default_150 = full_default_151 = full_default_152 = full_default_153 = full_default_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = getitem_211 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = getitem_218 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = getitem_223 = getitem_224 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = getitem_244 = getitem_245 = getitem_246 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = getitem_251 = getitem_252 = getitem_253 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = getitem_259 = getitem_260 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = getitem_265 = getitem_266 = getitem_267 = getitem_268 = getitem_269 = getitem_270 = getitem_271 = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = getitem_277 = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = getitem_284 = getitem_285 = getitem_286 = getitem_287 = getitem_288 = getitem_289 = getitem_290 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = getitem_295 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = getitem_301 = getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = getitem_307 = getitem_308 = getitem_309 = None
        getitem: "f32[1, 1, 768]" = _foreach_addcmul_scalar[0]
        getitem_310: "f32[1, 198, 768]" = _foreach_addcmul_scalar[1]
        getitem_311: "f32[1, 1, 768]" = _foreach_addcmul_scalar[2]
        getitem_312: "f32[768, 3, 16, 16]" = _foreach_addcmul_scalar[3]
        getitem_313: "f32[768]" = _foreach_addcmul_scalar[4]
        getitem_314: "f32[768]" = _foreach_addcmul_scalar[5]
        getitem_315: "f32[768]" = _foreach_addcmul_scalar[6]
        getitem_316: "f32[2304, 768]" = _foreach_addcmul_scalar[7]
        getitem_317: "f32[2304]" = _foreach_addcmul_scalar[8]
        getitem_318: "f32[768, 768]" = _foreach_addcmul_scalar[9]
        getitem_319: "f32[768]" = _foreach_addcmul_scalar[10]
        getitem_320: "f32[768]" = _foreach_addcmul_scalar[11]
        getitem_321: "f32[768]" = _foreach_addcmul_scalar[12]
        getitem_322: "f32[3072, 768]" = _foreach_addcmul_scalar[13]
        getitem_323: "f32[3072]" = _foreach_addcmul_scalar[14]
        getitem_324: "f32[768, 3072]" = _foreach_addcmul_scalar[15]
        getitem_325: "f32[768]" = _foreach_addcmul_scalar[16]
        getitem_326: "f32[768]" = _foreach_addcmul_scalar[17]
        getitem_327: "f32[768]" = _foreach_addcmul_scalar[18]
        getitem_328: "f32[2304, 768]" = _foreach_addcmul_scalar[19]
        getitem_329: "f32[2304]" = _foreach_addcmul_scalar[20]
        getitem_330: "f32[768, 768]" = _foreach_addcmul_scalar[21]
        getitem_331: "f32[768]" = _foreach_addcmul_scalar[22]
        getitem_332: "f32[768]" = _foreach_addcmul_scalar[23]
        getitem_333: "f32[768]" = _foreach_addcmul_scalar[24]
        getitem_334: "f32[3072, 768]" = _foreach_addcmul_scalar[25]
        getitem_335: "f32[3072]" = _foreach_addcmul_scalar[26]
        getitem_336: "f32[768, 3072]" = _foreach_addcmul_scalar[27]
        getitem_337: "f32[768]" = _foreach_addcmul_scalar[28]
        getitem_338: "f32[768]" = _foreach_addcmul_scalar[29]
        getitem_339: "f32[768]" = _foreach_addcmul_scalar[30]
        getitem_340: "f32[2304, 768]" = _foreach_addcmul_scalar[31]
        getitem_341: "f32[2304]" = _foreach_addcmul_scalar[32]
        getitem_342: "f32[768, 768]" = _foreach_addcmul_scalar[33]
        getitem_343: "f32[768]" = _foreach_addcmul_scalar[34]
        getitem_344: "f32[768]" = _foreach_addcmul_scalar[35]
        getitem_345: "f32[768]" = _foreach_addcmul_scalar[36]
        getitem_346: "f32[3072, 768]" = _foreach_addcmul_scalar[37]
        getitem_347: "f32[3072]" = _foreach_addcmul_scalar[38]
        getitem_348: "f32[768, 3072]" = _foreach_addcmul_scalar[39]
        getitem_349: "f32[768]" = _foreach_addcmul_scalar[40]
        getitem_350: "f32[768]" = _foreach_addcmul_scalar[41]
        getitem_351: "f32[768]" = _foreach_addcmul_scalar[42]
        getitem_352: "f32[2304, 768]" = _foreach_addcmul_scalar[43]
        getitem_353: "f32[2304]" = _foreach_addcmul_scalar[44]
        getitem_354: "f32[768, 768]" = _foreach_addcmul_scalar[45]
        getitem_355: "f32[768]" = _foreach_addcmul_scalar[46]
        getitem_356: "f32[768]" = _foreach_addcmul_scalar[47]
        getitem_357: "f32[768]" = _foreach_addcmul_scalar[48]
        getitem_358: "f32[3072, 768]" = _foreach_addcmul_scalar[49]
        getitem_359: "f32[3072]" = _foreach_addcmul_scalar[50]
        getitem_360: "f32[768, 3072]" = _foreach_addcmul_scalar[51]
        getitem_361: "f32[768]" = _foreach_addcmul_scalar[52]
        getitem_362: "f32[768]" = _foreach_addcmul_scalar[53]
        getitem_363: "f32[768]" = _foreach_addcmul_scalar[54]
        getitem_364: "f32[2304, 768]" = _foreach_addcmul_scalar[55]
        getitem_365: "f32[2304]" = _foreach_addcmul_scalar[56]
        getitem_366: "f32[768, 768]" = _foreach_addcmul_scalar[57]
        getitem_367: "f32[768]" = _foreach_addcmul_scalar[58]
        getitem_368: "f32[768]" = _foreach_addcmul_scalar[59]
        getitem_369: "f32[768]" = _foreach_addcmul_scalar[60]
        getitem_370: "f32[3072, 768]" = _foreach_addcmul_scalar[61]
        getitem_371: "f32[3072]" = _foreach_addcmul_scalar[62]
        getitem_372: "f32[768, 3072]" = _foreach_addcmul_scalar[63]
        getitem_373: "f32[768]" = _foreach_addcmul_scalar[64]
        getitem_374: "f32[768]" = _foreach_addcmul_scalar[65]
        getitem_375: "f32[768]" = _foreach_addcmul_scalar[66]
        getitem_376: "f32[2304, 768]" = _foreach_addcmul_scalar[67]
        getitem_377: "f32[2304]" = _foreach_addcmul_scalar[68]
        getitem_378: "f32[768, 768]" = _foreach_addcmul_scalar[69]
        getitem_379: "f32[768]" = _foreach_addcmul_scalar[70]
        getitem_380: "f32[768]" = _foreach_addcmul_scalar[71]
        getitem_381: "f32[768]" = _foreach_addcmul_scalar[72]
        getitem_382: "f32[3072, 768]" = _foreach_addcmul_scalar[73]
        getitem_383: "f32[3072]" = _foreach_addcmul_scalar[74]
        getitem_384: "f32[768, 3072]" = _foreach_addcmul_scalar[75]
        getitem_385: "f32[768]" = _foreach_addcmul_scalar[76]
        getitem_386: "f32[768]" = _foreach_addcmul_scalar[77]
        getitem_387: "f32[768]" = _foreach_addcmul_scalar[78]
        getitem_388: "f32[2304, 768]" = _foreach_addcmul_scalar[79]
        getitem_389: "f32[2304]" = _foreach_addcmul_scalar[80]
        getitem_390: "f32[768, 768]" = _foreach_addcmul_scalar[81]
        getitem_391: "f32[768]" = _foreach_addcmul_scalar[82]
        getitem_392: "f32[768]" = _foreach_addcmul_scalar[83]
        getitem_393: "f32[768]" = _foreach_addcmul_scalar[84]
        getitem_394: "f32[3072, 768]" = _foreach_addcmul_scalar[85]
        getitem_395: "f32[3072]" = _foreach_addcmul_scalar[86]
        getitem_396: "f32[768, 3072]" = _foreach_addcmul_scalar[87]
        getitem_397: "f32[768]" = _foreach_addcmul_scalar[88]
        getitem_398: "f32[768]" = _foreach_addcmul_scalar[89]
        getitem_399: "f32[768]" = _foreach_addcmul_scalar[90]
        getitem_400: "f32[2304, 768]" = _foreach_addcmul_scalar[91]
        getitem_401: "f32[2304]" = _foreach_addcmul_scalar[92]
        getitem_402: "f32[768, 768]" = _foreach_addcmul_scalar[93]
        getitem_403: "f32[768]" = _foreach_addcmul_scalar[94]
        getitem_404: "f32[768]" = _foreach_addcmul_scalar[95]
        getitem_405: "f32[768]" = _foreach_addcmul_scalar[96]
        getitem_406: "f32[3072, 768]" = _foreach_addcmul_scalar[97]
        getitem_407: "f32[3072]" = _foreach_addcmul_scalar[98]
        getitem_408: "f32[768, 3072]" = _foreach_addcmul_scalar[99]
        getitem_409: "f32[768]" = _foreach_addcmul_scalar[100]
        getitem_410: "f32[768]" = _foreach_addcmul_scalar[101]
        getitem_411: "f32[768]" = _foreach_addcmul_scalar[102]
        getitem_412: "f32[2304, 768]" = _foreach_addcmul_scalar[103]
        getitem_413: "f32[2304]" = _foreach_addcmul_scalar[104]
        getitem_414: "f32[768, 768]" = _foreach_addcmul_scalar[105]
        getitem_415: "f32[768]" = _foreach_addcmul_scalar[106]
        getitem_416: "f32[768]" = _foreach_addcmul_scalar[107]
        getitem_417: "f32[768]" = _foreach_addcmul_scalar[108]
        getitem_418: "f32[3072, 768]" = _foreach_addcmul_scalar[109]
        getitem_419: "f32[3072]" = _foreach_addcmul_scalar[110]
        getitem_420: "f32[768, 3072]" = _foreach_addcmul_scalar[111]
        getitem_421: "f32[768]" = _foreach_addcmul_scalar[112]
        getitem_422: "f32[768]" = _foreach_addcmul_scalar[113]
        getitem_423: "f32[768]" = _foreach_addcmul_scalar[114]
        getitem_424: "f32[2304, 768]" = _foreach_addcmul_scalar[115]
        getitem_425: "f32[2304]" = _foreach_addcmul_scalar[116]
        getitem_426: "f32[768, 768]" = _foreach_addcmul_scalar[117]
        getitem_427: "f32[768]" = _foreach_addcmul_scalar[118]
        getitem_428: "f32[768]" = _foreach_addcmul_scalar[119]
        getitem_429: "f32[768]" = _foreach_addcmul_scalar[120]
        getitem_430: "f32[3072, 768]" = _foreach_addcmul_scalar[121]
        getitem_431: "f32[3072]" = _foreach_addcmul_scalar[122]
        getitem_432: "f32[768, 3072]" = _foreach_addcmul_scalar[123]
        getitem_433: "f32[768]" = _foreach_addcmul_scalar[124]
        getitem_434: "f32[768]" = _foreach_addcmul_scalar[125]
        getitem_435: "f32[768]" = _foreach_addcmul_scalar[126]
        getitem_436: "f32[2304, 768]" = _foreach_addcmul_scalar[127]
        getitem_437: "f32[2304]" = _foreach_addcmul_scalar[128]
        getitem_438: "f32[768, 768]" = _foreach_addcmul_scalar[129]
        getitem_439: "f32[768]" = _foreach_addcmul_scalar[130]
        getitem_440: "f32[768]" = _foreach_addcmul_scalar[131]
        getitem_441: "f32[768]" = _foreach_addcmul_scalar[132]
        getitem_442: "f32[3072, 768]" = _foreach_addcmul_scalar[133]
        getitem_443: "f32[3072]" = _foreach_addcmul_scalar[134]
        getitem_444: "f32[768, 3072]" = _foreach_addcmul_scalar[135]
        getitem_445: "f32[768]" = _foreach_addcmul_scalar[136]
        getitem_446: "f32[768]" = _foreach_addcmul_scalar[137]
        getitem_447: "f32[768]" = _foreach_addcmul_scalar[138]
        getitem_448: "f32[2304, 768]" = _foreach_addcmul_scalar[139]
        getitem_449: "f32[2304]" = _foreach_addcmul_scalar[140]
        getitem_450: "f32[768, 768]" = _foreach_addcmul_scalar[141]
        getitem_451: "f32[768]" = _foreach_addcmul_scalar[142]
        getitem_452: "f32[768]" = _foreach_addcmul_scalar[143]
        getitem_453: "f32[768]" = _foreach_addcmul_scalar[144]
        getitem_454: "f32[3072, 768]" = _foreach_addcmul_scalar[145]
        getitem_455: "f32[3072]" = _foreach_addcmul_scalar[146]
        getitem_456: "f32[768, 3072]" = _foreach_addcmul_scalar[147]
        getitem_457: "f32[768]" = _foreach_addcmul_scalar[148]
        getitem_458: "f32[768]" = _foreach_addcmul_scalar[149]
        getitem_459: "f32[768]" = _foreach_addcmul_scalar[150]
        getitem_460: "f32[1000, 768]" = _foreach_addcmul_scalar[151]
        getitem_461: "f32[1000]" = _foreach_addcmul_scalar[152]
        getitem_462: "f32[1000, 768]" = _foreach_addcmul_scalar[153]
        getitem_463: "f32[1000]" = _foreach_addcmul_scalar[154];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_2325, getitem_2326, getitem_2327, getitem_2328, getitem_2329, getitem_2330, getitem_2331, getitem_2332, getitem_2333, getitem_2334, getitem_2335, getitem_2336, getitem_2337, getitem_2338, getitem_2339, getitem_2340, getitem_2341, getitem_2342, getitem_2343, getitem_2344, getitem_2345, getitem_2346, getitem_2347, getitem_2348, getitem_2349, getitem_2350, getitem_2351, getitem_2352, getitem_2353, getitem_2354, getitem_2355, getitem_2356, getitem_2357, getitem_2358, getitem_2359, getitem_2360, getitem_2361, getitem_2362, getitem_2363, getitem_2364, getitem_2365, getitem_2366, getitem_2367, getitem_2368, getitem_2369, getitem_2370, getitem_2371, getitem_2372, getitem_2373, getitem_2374, getitem_2375, getitem_2376, getitem_2377, getitem_2378, getitem_2379, getitem_2380, getitem_2381, getitem_2382, getitem_2383, getitem_2384, getitem_2385, getitem_2386, getitem_2387, getitem_2388, getitem_2389, getitem_2390, getitem_2391, getitem_2392, getitem_2393, getitem_2394, getitem_2395, getitem_2396, getitem_2397, getitem_2398, getitem_2399, getitem_2400, getitem_2401, getitem_2402, getitem_2403, getitem_2404, getitem_2405, getitem_2406, getitem_2407, getitem_2408, getitem_2409, getitem_2410, getitem_2411, getitem_2412, getitem_2413, getitem_2414, getitem_2415, getitem_2416, getitem_2417, getitem_2418, getitem_2419, getitem_2420, getitem_2421, getitem_2422, getitem_2423, getitem_2424, getitem_2425, getitem_2426, getitem_2427, getitem_2428, getitem_2429, getitem_2430, getitem_2431, getitem_2432, getitem_2433, getitem_2434, getitem_2435, getitem_2436, getitem_2437, getitem_2438, getitem_2439, getitem_2440, getitem_2441, getitem_2442, getitem_2443, getitem_2444, getitem_2445, getitem_2446, getitem_2447, getitem_2448, getitem_2449, getitem_2450, getitem_2451, getitem_2452, getitem_2453, getitem_2454, getitem_2455, getitem_2456, getitem_2457, getitem_2458, getitem_2459, getitem_2460, getitem_2461, getitem_2462, getitem_2463, getitem_2464, getitem_2465, getitem_2466, getitem_2467, getitem_2468, getitem_2469, getitem_2470, getitem_2471, getitem_2472, getitem_2473, getitem_2474, getitem_2475, getitem_2476, getitem_2477, getitem_2478, getitem_2479], [getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859]);  getitem_2325 = getitem_2326 = getitem_2327 = getitem_2328 = getitem_2329 = getitem_2330 = getitem_2331 = getitem_2332 = getitem_2333 = getitem_2334 = getitem_2335 = getitem_2336 = getitem_2337 = getitem_2338 = getitem_2339 = getitem_2340 = getitem_2341 = getitem_2342 = getitem_2343 = getitem_2344 = getitem_2345 = getitem_2346 = getitem_2347 = getitem_2348 = getitem_2349 = getitem_2350 = getitem_2351 = getitem_2352 = getitem_2353 = getitem_2354 = getitem_2355 = getitem_2356 = getitem_2357 = getitem_2358 = getitem_2359 = getitem_2360 = getitem_2361 = getitem_2362 = getitem_2363 = getitem_2364 = getitem_2365 = getitem_2366 = getitem_2367 = getitem_2368 = getitem_2369 = getitem_2370 = getitem_2371 = getitem_2372 = getitem_2373 = getitem_2374 = getitem_2375 = getitem_2376 = getitem_2377 = getitem_2378 = getitem_2379 = getitem_2380 = getitem_2381 = getitem_2382 = getitem_2383 = getitem_2384 = getitem_2385 = getitem_2386 = getitem_2387 = getitem_2388 = getitem_2389 = getitem_2390 = getitem_2391 = getitem_2392 = getitem_2393 = getitem_2394 = getitem_2395 = getitem_2396 = getitem_2397 = getitem_2398 = getitem_2399 = getitem_2400 = getitem_2401 = getitem_2402 = getitem_2403 = getitem_2404 = getitem_2405 = getitem_2406 = getitem_2407 = getitem_2408 = getitem_2409 = getitem_2410 = getitem_2411 = getitem_2412 = getitem_2413 = getitem_2414 = getitem_2415 = getitem_2416 = getitem_2417 = getitem_2418 = getitem_2419 = getitem_2420 = getitem_2421 = getitem_2422 = getitem_2423 = getitem_2424 = getitem_2425 = getitem_2426 = getitem_2427 = getitem_2428 = getitem_2429 = getitem_2430 = getitem_2431 = getitem_2432 = getitem_2433 = getitem_2434 = getitem_2435 = getitem_2436 = getitem_2437 = getitem_2438 = getitem_2439 = getitem_2440 = getitem_2441 = getitem_2442 = getitem_2443 = getitem_2444 = getitem_2445 = getitem_2446 = getitem_2447 = getitem_2448 = getitem_2449 = getitem_2450 = getitem_2451 = getitem_2452 = getitem_2453 = getitem_2454 = getitem_2455 = getitem_2456 = getitem_2457 = getitem_2458 = getitem_2459 = getitem_2460 = getitem_2461 = getitem_2462 = getitem_2463 = getitem_2464 = getitem_2465 = getitem_2466 = getitem_2467 = getitem_2468 = getitem_2469 = getitem_2470 = getitem_2471 = getitem_2472 = getitem_2473 = getitem_2474 = getitem_2475 = getitem_2476 = getitem_2477 = getitem_2478 = getitem_2479 = getitem_1705 = getitem_1706 = getitem_1707 = getitem_1708 = getitem_1709 = getitem_1710 = getitem_1711 = getitem_1712 = getitem_1713 = getitem_1714 = getitem_1715 = getitem_1716 = getitem_1717 = getitem_1718 = getitem_1719 = getitem_1720 = getitem_1721 = getitem_1722 = getitem_1723 = getitem_1724 = getitem_1725 = getitem_1726 = getitem_1727 = getitem_1728 = getitem_1729 = getitem_1730 = getitem_1731 = getitem_1732 = getitem_1733 = getitem_1734 = getitem_1735 = getitem_1736 = getitem_1737 = getitem_1738 = getitem_1739 = getitem_1740 = getitem_1741 = getitem_1742 = getitem_1743 = getitem_1744 = getitem_1745 = getitem_1746 = getitem_1747 = getitem_1748 = getitem_1749 = getitem_1750 = getitem_1751 = getitem_1752 = getitem_1753 = getitem_1754 = getitem_1755 = getitem_1756 = getitem_1757 = getitem_1758 = getitem_1759 = getitem_1760 = getitem_1761 = getitem_1762 = getitem_1763 = getitem_1764 = getitem_1765 = getitem_1766 = getitem_1767 = getitem_1768 = getitem_1769 = getitem_1770 = getitem_1771 = getitem_1772 = getitem_1773 = getitem_1774 = getitem_1775 = getitem_1776 = getitem_1777 = getitem_1778 = getitem_1779 = getitem_1780 = getitem_1781 = getitem_1782 = getitem_1783 = getitem_1784 = getitem_1785 = getitem_1786 = getitem_1787 = getitem_1788 = getitem_1789 = getitem_1790 = getitem_1791 = getitem_1792 = getitem_1793 = getitem_1794 = getitem_1795 = getitem_1796 = getitem_1797 = getitem_1798 = getitem_1799 = getitem_1800 = getitem_1801 = getitem_1802 = getitem_1803 = getitem_1804 = getitem_1805 = getitem_1806 = getitem_1807 = getitem_1808 = getitem_1809 = getitem_1810 = getitem_1811 = getitem_1812 = getitem_1813 = getitem_1814 = getitem_1815 = getitem_1816 = getitem_1817 = getitem_1818 = getitem_1819 = getitem_1820 = getitem_1821 = getitem_1822 = getitem_1823 = getitem_1824 = getitem_1825 = getitem_1826 = getitem_1827 = getitem_1828 = getitem_1829 = getitem_1830 = getitem_1831 = getitem_1832 = getitem_1833 = getitem_1834 = getitem_1835 = getitem_1836 = getitem_1837 = getitem_1838 = getitem_1839 = getitem_1840 = getitem_1841 = getitem_1842 = getitem_1843 = getitem_1844 = getitem_1845 = getitem_1846 = getitem_1847 = getitem_1848 = getitem_1849 = getitem_1850 = getitem_1851 = getitem_1852 = getitem_1853 = getitem_1854 = getitem_1855 = getitem_1856 = getitem_1857 = getitem_1858 = getitem_1859 = None
        getitem_1860: "f32[1, 1, 768]" = _foreach_div_list[0]
        getitem_1861: "f32[1, 198, 768]" = _foreach_div_list[1]
        getitem_1862: "f32[1, 1, 768]" = _foreach_div_list[2]
        getitem_1863: "f32[768, 3, 16, 16]" = _foreach_div_list[3]
        getitem_1864: "f32[768]" = _foreach_div_list[4]
        getitem_1865: "f32[768]" = _foreach_div_list[5]
        getitem_1866: "f32[768]" = _foreach_div_list[6]
        getitem_1867: "f32[2304, 768]" = _foreach_div_list[7]
        getitem_1868: "f32[2304]" = _foreach_div_list[8]
        getitem_1869: "f32[768, 768]" = _foreach_div_list[9]
        getitem_1870: "f32[768]" = _foreach_div_list[10]
        getitem_1871: "f32[768]" = _foreach_div_list[11]
        getitem_1872: "f32[768]" = _foreach_div_list[12]
        getitem_1873: "f32[3072, 768]" = _foreach_div_list[13]
        getitem_1874: "f32[3072]" = _foreach_div_list[14]
        getitem_1875: "f32[768, 3072]" = _foreach_div_list[15]
        getitem_1876: "f32[768]" = _foreach_div_list[16]
        getitem_1877: "f32[768]" = _foreach_div_list[17]
        getitem_1878: "f32[768]" = _foreach_div_list[18]
        getitem_1879: "f32[2304, 768]" = _foreach_div_list[19]
        getitem_1880: "f32[2304]" = _foreach_div_list[20]
        getitem_1881: "f32[768, 768]" = _foreach_div_list[21]
        getitem_1882: "f32[768]" = _foreach_div_list[22]
        getitem_1883: "f32[768]" = _foreach_div_list[23]
        getitem_1884: "f32[768]" = _foreach_div_list[24]
        getitem_1885: "f32[3072, 768]" = _foreach_div_list[25]
        getitem_1886: "f32[3072]" = _foreach_div_list[26]
        getitem_1887: "f32[768, 3072]" = _foreach_div_list[27]
        getitem_1888: "f32[768]" = _foreach_div_list[28]
        getitem_1889: "f32[768]" = _foreach_div_list[29]
        getitem_1890: "f32[768]" = _foreach_div_list[30]
        getitem_1891: "f32[2304, 768]" = _foreach_div_list[31]
        getitem_1892: "f32[2304]" = _foreach_div_list[32]
        getitem_1893: "f32[768, 768]" = _foreach_div_list[33]
        getitem_1894: "f32[768]" = _foreach_div_list[34]
        getitem_1895: "f32[768]" = _foreach_div_list[35]
        getitem_1896: "f32[768]" = _foreach_div_list[36]
        getitem_1897: "f32[3072, 768]" = _foreach_div_list[37]
        getitem_1898: "f32[3072]" = _foreach_div_list[38]
        getitem_1899: "f32[768, 3072]" = _foreach_div_list[39]
        getitem_1900: "f32[768]" = _foreach_div_list[40]
        getitem_1901: "f32[768]" = _foreach_div_list[41]
        getitem_1902: "f32[768]" = _foreach_div_list[42]
        getitem_1903: "f32[2304, 768]" = _foreach_div_list[43]
        getitem_1904: "f32[2304]" = _foreach_div_list[44]
        getitem_1905: "f32[768, 768]" = _foreach_div_list[45]
        getitem_1906: "f32[768]" = _foreach_div_list[46]
        getitem_1907: "f32[768]" = _foreach_div_list[47]
        getitem_1908: "f32[768]" = _foreach_div_list[48]
        getitem_1909: "f32[3072, 768]" = _foreach_div_list[49]
        getitem_1910: "f32[3072]" = _foreach_div_list[50]
        getitem_1911: "f32[768, 3072]" = _foreach_div_list[51]
        getitem_1912: "f32[768]" = _foreach_div_list[52]
        getitem_1913: "f32[768]" = _foreach_div_list[53]
        getitem_1914: "f32[768]" = _foreach_div_list[54]
        getitem_1915: "f32[2304, 768]" = _foreach_div_list[55]
        getitem_1916: "f32[2304]" = _foreach_div_list[56]
        getitem_1917: "f32[768, 768]" = _foreach_div_list[57]
        getitem_1918: "f32[768]" = _foreach_div_list[58]
        getitem_1919: "f32[768]" = _foreach_div_list[59]
        getitem_1920: "f32[768]" = _foreach_div_list[60]
        getitem_1921: "f32[3072, 768]" = _foreach_div_list[61]
        getitem_1922: "f32[3072]" = _foreach_div_list[62]
        getitem_1923: "f32[768, 3072]" = _foreach_div_list[63]
        getitem_1924: "f32[768]" = _foreach_div_list[64]
        getitem_1925: "f32[768]" = _foreach_div_list[65]
        getitem_1926: "f32[768]" = _foreach_div_list[66]
        getitem_1927: "f32[2304, 768]" = _foreach_div_list[67]
        getitem_1928: "f32[2304]" = _foreach_div_list[68]
        getitem_1929: "f32[768, 768]" = _foreach_div_list[69]
        getitem_1930: "f32[768]" = _foreach_div_list[70]
        getitem_1931: "f32[768]" = _foreach_div_list[71]
        getitem_1932: "f32[768]" = _foreach_div_list[72]
        getitem_1933: "f32[3072, 768]" = _foreach_div_list[73]
        getitem_1934: "f32[3072]" = _foreach_div_list[74]
        getitem_1935: "f32[768, 3072]" = _foreach_div_list[75]
        getitem_1936: "f32[768]" = _foreach_div_list[76]
        getitem_1937: "f32[768]" = _foreach_div_list[77]
        getitem_1938: "f32[768]" = _foreach_div_list[78]
        getitem_1939: "f32[2304, 768]" = _foreach_div_list[79]
        getitem_1940: "f32[2304]" = _foreach_div_list[80]
        getitem_1941: "f32[768, 768]" = _foreach_div_list[81]
        getitem_1942: "f32[768]" = _foreach_div_list[82]
        getitem_1943: "f32[768]" = _foreach_div_list[83]
        getitem_1944: "f32[768]" = _foreach_div_list[84]
        getitem_1945: "f32[3072, 768]" = _foreach_div_list[85]
        getitem_1946: "f32[3072]" = _foreach_div_list[86]
        getitem_1947: "f32[768, 3072]" = _foreach_div_list[87]
        getitem_1948: "f32[768]" = _foreach_div_list[88]
        getitem_1949: "f32[768]" = _foreach_div_list[89]
        getitem_1950: "f32[768]" = _foreach_div_list[90]
        getitem_1951: "f32[2304, 768]" = _foreach_div_list[91]
        getitem_1952: "f32[2304]" = _foreach_div_list[92]
        getitem_1953: "f32[768, 768]" = _foreach_div_list[93]
        getitem_1954: "f32[768]" = _foreach_div_list[94]
        getitem_1955: "f32[768]" = _foreach_div_list[95]
        getitem_1956: "f32[768]" = _foreach_div_list[96]
        getitem_1957: "f32[3072, 768]" = _foreach_div_list[97]
        getitem_1958: "f32[3072]" = _foreach_div_list[98]
        getitem_1959: "f32[768, 3072]" = _foreach_div_list[99]
        getitem_1960: "f32[768]" = _foreach_div_list[100]
        getitem_1961: "f32[768]" = _foreach_div_list[101]
        getitem_1962: "f32[768]" = _foreach_div_list[102]
        getitem_1963: "f32[2304, 768]" = _foreach_div_list[103]
        getitem_1964: "f32[2304]" = _foreach_div_list[104]
        getitem_1965: "f32[768, 768]" = _foreach_div_list[105]
        getitem_1966: "f32[768]" = _foreach_div_list[106]
        getitem_1967: "f32[768]" = _foreach_div_list[107]
        getitem_1968: "f32[768]" = _foreach_div_list[108]
        getitem_1969: "f32[3072, 768]" = _foreach_div_list[109]
        getitem_1970: "f32[3072]" = _foreach_div_list[110]
        getitem_1971: "f32[768, 3072]" = _foreach_div_list[111]
        getitem_1972: "f32[768]" = _foreach_div_list[112]
        getitem_1973: "f32[768]" = _foreach_div_list[113]
        getitem_1974: "f32[768]" = _foreach_div_list[114]
        getitem_1975: "f32[2304, 768]" = _foreach_div_list[115]
        getitem_1976: "f32[2304]" = _foreach_div_list[116]
        getitem_1977: "f32[768, 768]" = _foreach_div_list[117]
        getitem_1978: "f32[768]" = _foreach_div_list[118]
        getitem_1979: "f32[768]" = _foreach_div_list[119]
        getitem_1980: "f32[768]" = _foreach_div_list[120]
        getitem_1981: "f32[3072, 768]" = _foreach_div_list[121]
        getitem_1982: "f32[3072]" = _foreach_div_list[122]
        getitem_1983: "f32[768, 3072]" = _foreach_div_list[123]
        getitem_1984: "f32[768]" = _foreach_div_list[124]
        getitem_1985: "f32[768]" = _foreach_div_list[125]
        getitem_1986: "f32[768]" = _foreach_div_list[126]
        getitem_1987: "f32[2304, 768]" = _foreach_div_list[127]
        getitem_1988: "f32[2304]" = _foreach_div_list[128]
        getitem_1989: "f32[768, 768]" = _foreach_div_list[129]
        getitem_1990: "f32[768]" = _foreach_div_list[130]
        getitem_1991: "f32[768]" = _foreach_div_list[131]
        getitem_1992: "f32[768]" = _foreach_div_list[132]
        getitem_1993: "f32[3072, 768]" = _foreach_div_list[133]
        getitem_1994: "f32[3072]" = _foreach_div_list[134]
        getitem_1995: "f32[768, 3072]" = _foreach_div_list[135]
        getitem_1996: "f32[768]" = _foreach_div_list[136]
        getitem_1997: "f32[768]" = _foreach_div_list[137]
        getitem_1998: "f32[768]" = _foreach_div_list[138]
        getitem_1999: "f32[2304, 768]" = _foreach_div_list[139]
        getitem_2000: "f32[2304]" = _foreach_div_list[140]
        getitem_2001: "f32[768, 768]" = _foreach_div_list[141]
        getitem_2002: "f32[768]" = _foreach_div_list[142]
        getitem_2003: "f32[768]" = _foreach_div_list[143]
        getitem_2004: "f32[768]" = _foreach_div_list[144]
        getitem_2005: "f32[3072, 768]" = _foreach_div_list[145]
        getitem_2006: "f32[3072]" = _foreach_div_list[146]
        getitem_2007: "f32[768, 3072]" = _foreach_div_list[147]
        getitem_2008: "f32[768]" = _foreach_div_list[148]
        getitem_2009: "f32[768]" = _foreach_div_list[149]
        getitem_2010: "f32[768]" = _foreach_div_list[150]
        getitem_2011: "f32[1000, 768]" = _foreach_div_list[151]
        getitem_2012: "f32[1000]" = _foreach_div_list[152]
        getitem_2013: "f32[1000, 768]" = _foreach_div_list[153]
        getitem_2014: "f32[1000]" = _foreach_div_list[154];  _foreach_div_list = None
        return (getitem, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014)


def _default_make_inputs():
    return [
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

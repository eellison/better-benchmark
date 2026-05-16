"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g77
Pattern hash: 5e0cdfa267ca
Shape hash: 4718abe3
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg171_1: "f32[64, 3, 3, 3]", arg169_1: "f32[64]", arg340_1: "f32[64]", arg341_1: "f32[64, 3, 1, 1]", arg342_1: "f32[64]", arg343_1: "f32[64]", arg344_1: "f32[96, 64, 3, 3]", arg345_1: "f32[96]", arg346_1: "f32[96]", arg347_1: "f32[96, 64, 1, 1]", arg348_1: "f32[96]", arg349_1: "f32[96]", arg350_1: "f32[96]", arg351_1: "f32[96]", arg352_1: "f32[96, 96, 3, 3]", arg353_1: "f32[96]", arg354_1: "f32[96]", arg355_1: "f32[96, 96, 1, 1]", arg356_1: "f32[96]", arg357_1: "f32[96]", arg358_1: "f32[192, 96, 3, 3]", arg359_1: "f32[192]", arg360_1: "f32[192]", arg361_1: "f32[192, 96, 1, 1]", arg362_1: "f32[192]", arg363_1: "f32[192]", arg364_1: "f32[192]", arg365_1: "f32[192]", arg366_1: "f32[192, 192, 3, 3]", arg367_1: "f32[192]", arg368_1: "f32[192]", arg369_1: "f32[192, 192, 1, 1]", arg370_1: "f32[192]", arg371_1: "f32[192]", arg372_1: "f32[192]", arg373_1: "f32[192]", arg374_1: "f32[192, 192, 3, 3]", arg375_1: "f32[192]", arg376_1: "f32[192]", arg377_1: "f32[192, 192, 1, 1]", arg378_1: "f32[192]", arg379_1: "f32[192]", arg380_1: "f32[192]", arg381_1: "f32[192]", arg382_1: "f32[192, 192, 3, 3]", arg383_1: "f32[192]", arg384_1: "f32[192]", arg385_1: "f32[192, 192, 1, 1]", arg386_1: "f32[192]", arg387_1: "f32[192]", arg388_1: "f32[384, 192, 3, 3]", arg389_1: "f32[384]", arg390_1: "f32[384]", arg391_1: "f32[384, 192, 1, 1]", arg392_1: "f32[384]", arg393_1: "f32[384]", arg394_1: "f32[384]", arg395_1: "f32[384]", arg396_1: "f32[384, 384, 3, 3]", arg397_1: "f32[384]", arg398_1: "f32[384]", arg399_1: "f32[384, 384, 1, 1]", arg400_1: "f32[384]", arg401_1: "f32[384]", arg402_1: "f32[384]", arg403_1: "f32[384]", arg404_1: "f32[384, 384, 3, 3]", arg405_1: "f32[384]", arg406_1: "f32[384]", arg407_1: "f32[384, 384, 1, 1]", arg408_1: "f32[384]", arg409_1: "f32[384]", arg410_1: "f32[384]", arg411_1: "f32[384]", arg412_1: "f32[384, 384, 3, 3]", arg413_1: "f32[384]", arg414_1: "f32[384]", arg415_1: "f32[384, 384, 1, 1]", arg416_1: "f32[384]", arg417_1: "f32[384]", arg418_1: "f32[384]", arg419_1: "f32[384]", arg420_1: "f32[384, 384, 3, 3]", arg421_1: "f32[384]", arg422_1: "f32[384]", arg423_1: "f32[384, 384, 1, 1]", arg424_1: "f32[384]", arg425_1: "f32[384]", arg426_1: "f32[384]", arg427_1: "f32[384]", arg428_1: "f32[384, 384, 3, 3]", arg429_1: "f32[384]", arg430_1: "f32[384]", arg431_1: "f32[384, 384, 1, 1]", arg432_1: "f32[384]", arg433_1: "f32[384]", arg434_1: "f32[384]", arg435_1: "f32[384]", arg436_1: "f32[384, 384, 3, 3]", arg437_1: "f32[384]", arg438_1: "f32[384]", arg439_1: "f32[384, 384, 1, 1]", arg440_1: "f32[384]", arg441_1: "f32[384]", arg442_1: "f32[384]", arg443_1: "f32[384]", arg444_1: "f32[384, 384, 3, 3]", arg445_1: "f32[384]", arg446_1: "f32[384]", arg447_1: "f32[384, 384, 1, 1]", arg448_1: "f32[384]", arg449_1: "f32[384]", arg450_1: "f32[384]", arg451_1: "f32[384]", arg452_1: "f32[384, 384, 3, 3]", arg453_1: "f32[384]", arg454_1: "f32[384]", arg455_1: "f32[384, 384, 1, 1]", arg456_1: "f32[384]", arg457_1: "f32[384]", arg458_1: "f32[384]", arg459_1: "f32[384]", arg460_1: "f32[384, 384, 3, 3]", arg461_1: "f32[384]", arg462_1: "f32[384]", arg463_1: "f32[384, 384, 1, 1]", arg464_1: "f32[384]", arg465_1: "f32[384]", arg466_1: "f32[384]", arg467_1: "f32[384]", arg468_1: "f32[384, 384, 3, 3]", arg469_1: "f32[384]", arg470_1: "f32[384]", arg471_1: "f32[384, 384, 1, 1]", arg472_1: "f32[384]", arg473_1: "f32[384]", arg474_1: "f32[384]", arg475_1: "f32[384]", arg476_1: "f32[384, 384, 3, 3]", arg477_1: "f32[384]", arg478_1: "f32[384]", arg479_1: "f32[384, 384, 1, 1]", arg480_1: "f32[384]", arg481_1: "f32[384]", arg482_1: "f32[384]", arg483_1: "f32[384]", arg484_1: "f32[384, 384, 3, 3]", arg485_1: "f32[384]", arg486_1: "f32[384]", arg487_1: "f32[384, 384, 1, 1]", arg488_1: "f32[384]", arg489_1: "f32[384]", arg490_1: "f32[384]", arg491_1: "f32[384]", arg492_1: "f32[384, 384, 3, 3]", arg493_1: "f32[384]", arg494_1: "f32[384]", arg495_1: "f32[384, 384, 1, 1]", arg496_1: "f32[384]", arg497_1: "f32[384]", arg498_1: "f32[1408, 384, 3, 3]", arg499_1: "f32[1408]", arg500_1: "f32[1408]", arg501_1: "f32[1408, 384, 1, 1]", arg502_1: "f32[1408]", arg503_1: "f32[1408]", arg504_1: "f32[1000, 1408]", arg505_1: "f32[1000]", getitem_168: "f32[64, 3, 3, 3]", getitem_169: "f32[64]", getitem_170: "f32[64]", getitem_171: "f32[64, 3, 1, 1]", getitem_172: "f32[64]", getitem_173: "f32[64]", getitem_174: "f32[96, 64, 3, 3]", getitem_175: "f32[96]", getitem_176: "f32[96]", getitem_177: "f32[96, 64, 1, 1]", getitem_178: "f32[96]", getitem_179: "f32[96]", getitem_180: "f32[96]", getitem_181: "f32[96]", getitem_182: "f32[96, 96, 3, 3]", getitem_183: "f32[96]", getitem_184: "f32[96]", getitem_185: "f32[96, 96, 1, 1]", getitem_186: "f32[96]", getitem_187: "f32[96]", getitem_188: "f32[192, 96, 3, 3]", getitem_189: "f32[192]", getitem_190: "f32[192]", getitem_191: "f32[192, 96, 1, 1]", getitem_192: "f32[192]", getitem_193: "f32[192]", getitem_194: "f32[192]", getitem_195: "f32[192]", getitem_196: "f32[192, 192, 3, 3]", getitem_197: "f32[192]", getitem_198: "f32[192]", getitem_199: "f32[192, 192, 1, 1]", getitem_200: "f32[192]", getitem_201: "f32[192]", getitem_202: "f32[192]", getitem_203: "f32[192]", getitem_204: "f32[192, 192, 3, 3]", getitem_205: "f32[192]", getitem_206: "f32[192]", getitem_207: "f32[192, 192, 1, 1]", getitem_208: "f32[192]", getitem_209: "f32[192]", getitem_210: "f32[192]", getitem_211: "f32[192]", getitem_212: "f32[192, 192, 3, 3]", getitem_213: "f32[192]", getitem_214: "f32[192]", getitem_215: "f32[192, 192, 1, 1]", getitem_216: "f32[192]", getitem_217: "f32[192]", getitem_218: "f32[384, 192, 3, 3]", getitem_219: "f32[384]", getitem_220: "f32[384]", getitem_221: "f32[384, 192, 1, 1]", getitem_222: "f32[384]", getitem_223: "f32[384]", getitem_224: "f32[384]", getitem_225: "f32[384]", getitem_226: "f32[384, 384, 3, 3]", getitem_227: "f32[384]", getitem_228: "f32[384]", getitem_229: "f32[384, 384, 1, 1]", getitem_230: "f32[384]", getitem_231: "f32[384]", getitem_232: "f32[384]", getitem_233: "f32[384]", getitem_234: "f32[384, 384, 3, 3]", getitem_235: "f32[384]", getitem_236: "f32[384]", getitem_237: "f32[384, 384, 1, 1]", getitem_238: "f32[384]", getitem_239: "f32[384]", getitem_240: "f32[384]", getitem_241: "f32[384]", getitem_242: "f32[384, 384, 3, 3]", getitem_243: "f32[384]", getitem_244: "f32[384]", getitem_245: "f32[384, 384, 1, 1]", getitem_246: "f32[384]", getitem_247: "f32[384]", getitem_248: "f32[384]", getitem_249: "f32[384]", getitem_250: "f32[384, 384, 3, 3]", getitem_251: "f32[384]", getitem_252: "f32[384]", getitem_253: "f32[384, 384, 1, 1]", getitem_254: "f32[384]", getitem_255: "f32[384]", getitem_256: "f32[384]", getitem_257: "f32[384]", getitem_258: "f32[384, 384, 3, 3]", getitem_259: "f32[384]", getitem_260: "f32[384]", getitem_261: "f32[384, 384, 1, 1]", getitem_262: "f32[384]", getitem_263: "f32[384]", getitem_264: "f32[384]", getitem_265: "f32[384]", getitem_266: "f32[384, 384, 3, 3]", getitem_267: "f32[384]", getitem_268: "f32[384]", getitem_269: "f32[384, 384, 1, 1]", getitem_270: "f32[384]", getitem_271: "f32[384]", getitem_272: "f32[384]", getitem_273: "f32[384]", getitem_274: "f32[384, 384, 3, 3]", getitem_275: "f32[384]", getitem_276: "f32[384]", getitem_277: "f32[384, 384, 1, 1]", getitem_278: "f32[384]", getitem_279: "f32[384]", getitem_280: "f32[384]", getitem_281: "f32[384]", getitem_282: "f32[384, 384, 3, 3]", getitem_283: "f32[384]", getitem_284: "f32[384]", getitem_285: "f32[384, 384, 1, 1]", getitem_286: "f32[384]", getitem_287: "f32[384]", getitem_288: "f32[384]", getitem_289: "f32[384]", getitem_290: "f32[384, 384, 3, 3]", getitem_291: "f32[384]", getitem_292: "f32[384]", getitem_293: "f32[384, 384, 1, 1]", getitem_294: "f32[384]", getitem_295: "f32[384]", getitem_296: "f32[384]", getitem_297: "f32[384]", getitem_298: "f32[384, 384, 3, 3]", getitem_299: "f32[384]", getitem_300: "f32[384]", getitem_301: "f32[384, 384, 1, 1]", getitem_302: "f32[384]", getitem_303: "f32[384]", getitem_304: "f32[384]", getitem_305: "f32[384]", getitem_306: "f32[384, 384, 3, 3]", getitem_307: "f32[384]", getitem_308: "f32[384]", getitem_309: "f32[384, 384, 1, 1]", getitem_310: "f32[384]", getitem_311: "f32[384]", getitem_312: "f32[384]", getitem_313: "f32[384]", getitem_314: "f32[384, 384, 3, 3]", getitem_315: "f32[384]", getitem_316: "f32[384]", getitem_317: "f32[384, 384, 1, 1]", getitem_318: "f32[384]", getitem_319: "f32[384]", getitem_320: "f32[384]", getitem_321: "f32[384]", getitem_322: "f32[384, 384, 3, 3]", getitem_323: "f32[384]", getitem_324: "f32[384]", getitem_325: "f32[384, 384, 1, 1]", getitem_326: "f32[384]", getitem_327: "f32[384]", getitem_328: "f32[1408, 384, 3, 3]", getitem_329: "f32[1408]", getitem_330: "f32[1408]", getitem_331: "f32[1408, 384, 1, 1]", getitem_332: "f32[1408]", getitem_333: "f32[1408]", getitem_334: "f32[1000, 1408]", getitem_335: "f32[1000]", getitem_2520: "f32[64, 3, 3, 3]", getitem_2521: "f32[64]", getitem_2522: "f32[64]", getitem_2523: "f32[64, 3, 1, 1]", getitem_2524: "f32[64]", getitem_2525: "f32[64]", getitem_2526: "f32[96, 64, 3, 3]", getitem_2527: "f32[96]", getitem_2528: "f32[96]", getitem_2529: "f32[96, 64, 1, 1]", getitem_2530: "f32[96]", getitem_2531: "f32[96]", getitem_2532: "f32[96]", getitem_2533: "f32[96]", getitem_2534: "f32[96, 96, 3, 3]", getitem_2535: "f32[96]", getitem_2536: "f32[96]", getitem_2537: "f32[96, 96, 1, 1]", getitem_2538: "f32[96]", getitem_2539: "f32[96]", getitem_2540: "f32[192, 96, 3, 3]", getitem_2541: "f32[192]", getitem_2542: "f32[192]", getitem_2543: "f32[192, 96, 1, 1]", getitem_2544: "f32[192]", getitem_2545: "f32[192]", getitem_2546: "f32[192]", getitem_2547: "f32[192]", getitem_2548: "f32[192, 192, 3, 3]", getitem_2549: "f32[192]", getitem_2550: "f32[192]", getitem_2551: "f32[192, 192, 1, 1]", getitem_2552: "f32[192]", getitem_2553: "f32[192]", getitem_2554: "f32[192]", getitem_2555: "f32[192]", getitem_2556: "f32[192, 192, 3, 3]", getitem_2557: "f32[192]", getitem_2558: "f32[192]", getitem_2559: "f32[192, 192, 1, 1]", getitem_2560: "f32[192]", getitem_2561: "f32[192]", getitem_2562: "f32[192]", getitem_2563: "f32[192]", getitem_2564: "f32[192, 192, 3, 3]", getitem_2565: "f32[192]", getitem_2566: "f32[192]", getitem_2567: "f32[192, 192, 1, 1]", getitem_2568: "f32[192]", getitem_2569: "f32[192]", getitem_2570: "f32[384, 192, 3, 3]", getitem_2571: "f32[384]", getitem_2572: "f32[384]", getitem_2573: "f32[384, 192, 1, 1]", getitem_2574: "f32[384]", getitem_2575: "f32[384]", getitem_2576: "f32[384]", getitem_2577: "f32[384]", getitem_2578: "f32[384, 384, 3, 3]", getitem_2579: "f32[384]", getitem_2580: "f32[384]", getitem_2581: "f32[384, 384, 1, 1]", getitem_2582: "f32[384]", getitem_2583: "f32[384]", getitem_2584: "f32[384]", getitem_2585: "f32[384]", getitem_2586: "f32[384, 384, 3, 3]", getitem_2587: "f32[384]", getitem_2588: "f32[384]", getitem_2589: "f32[384, 384, 1, 1]", getitem_2590: "f32[384]", getitem_2591: "f32[384]", getitem_2592: "f32[384]", getitem_2593: "f32[384]", getitem_2594: "f32[384, 384, 3, 3]", getitem_2595: "f32[384]", getitem_2596: "f32[384]", getitem_2597: "f32[384, 384, 1, 1]", getitem_2598: "f32[384]", getitem_2599: "f32[384]", getitem_2600: "f32[384]", getitem_2601: "f32[384]", getitem_2602: "f32[384, 384, 3, 3]", getitem_2603: "f32[384]", getitem_2604: "f32[384]", getitem_2605: "f32[384, 384, 1, 1]", getitem_2606: "f32[384]", getitem_2607: "f32[384]", getitem_2608: "f32[384]", getitem_2609: "f32[384]", getitem_2610: "f32[384, 384, 3, 3]", getitem_2611: "f32[384]", getitem_2612: "f32[384]", getitem_2613: "f32[384, 384, 1, 1]", getitem_2614: "f32[384]", getitem_2615: "f32[384]", getitem_2616: "f32[384]", getitem_2617: "f32[384]", getitem_2618: "f32[384, 384, 3, 3]", getitem_2619: "f32[384]", getitem_2620: "f32[384]", getitem_2621: "f32[384, 384, 1, 1]", getitem_2622: "f32[384]", getitem_2623: "f32[384]", getitem_2624: "f32[384]", getitem_2625: "f32[384]", getitem_2626: "f32[384, 384, 3, 3]", getitem_2627: "f32[384]", getitem_2628: "f32[384]", getitem_2629: "f32[384, 384, 1, 1]", getitem_2630: "f32[384]", getitem_2631: "f32[384]", getitem_2632: "f32[384]", getitem_2633: "f32[384]", getitem_2634: "f32[384, 384, 3, 3]", getitem_2635: "f32[384]", getitem_2636: "f32[384]", getitem_2637: "f32[384, 384, 1, 1]", getitem_2638: "f32[384]", getitem_2639: "f32[384]", getitem_2640: "f32[384]", getitem_2641: "f32[384]", getitem_2642: "f32[384, 384, 3, 3]", getitem_2643: "f32[384]", getitem_2644: "f32[384]", getitem_2645: "f32[384, 384, 1, 1]", getitem_2646: "f32[384]", getitem_2647: "f32[384]", getitem_2648: "f32[384]", getitem_2649: "f32[384]", getitem_2650: "f32[384, 384, 3, 3]", getitem_2651: "f32[384]", getitem_2652: "f32[384]", getitem_2653: "f32[384, 384, 1, 1]", getitem_2654: "f32[384]", getitem_2655: "f32[384]", getitem_2656: "f32[384]", getitem_2657: "f32[384]", getitem_2658: "f32[384, 384, 3, 3]", getitem_2659: "f32[384]", getitem_2660: "f32[384]", getitem_2661: "f32[384, 384, 1, 1]", getitem_2662: "f32[384]", getitem_2663: "f32[384]", getitem_2664: "f32[384]", getitem_2665: "f32[384]", getitem_2666: "f32[384, 384, 3, 3]", getitem_2667: "f32[384]", getitem_2668: "f32[384]", getitem_2669: "f32[384, 384, 1, 1]", getitem_2670: "f32[384]", getitem_2671: "f32[384]", getitem_2672: "f32[384]", getitem_2673: "f32[384]", getitem_2674: "f32[384, 384, 3, 3]", getitem_2675: "f32[384]", getitem_2676: "f32[384]", getitem_2677: "f32[384, 384, 1, 1]", getitem_2678: "f32[384]", getitem_2679: "f32[384]", getitem_2680: "f32[1408, 384, 3, 3]", getitem_2681: "f32[1408]", getitem_2682: "f32[1408]", getitem_2683: "f32[1408, 384, 1, 1]", getitem_2684: "f32[1408]", getitem_2685: "f32[1408]", getitem_2686: "f32[1000, 1408]", getitem_2687: "f32[1000]", getitem_1848: "f32[]", getitem_1849: "f32[]", getitem_1850: "f32[]", getitem_1851: "f32[]", getitem_1852: "f32[]", getitem_1853: "f32[]", getitem_1854: "f32[]", getitem_1855: "f32[]", getitem_1856: "f32[]", getitem_1857: "f32[]", getitem_1858: "f32[]", getitem_1859: "f32[]", getitem_1860: "f32[]", getitem_1861: "f32[]", getitem_1862: "f32[]", getitem_1863: "f32[]", getitem_1864: "f32[]", getitem_1865: "f32[]", getitem_1866: "f32[]", getitem_1867: "f32[]", getitem_1868: "f32[]", getitem_1869: "f32[]", getitem_1870: "f32[]", getitem_1871: "f32[]", getitem_1872: "f32[]", getitem_1873: "f32[]", getitem_1874: "f32[]", getitem_1875: "f32[]", getitem_1876: "f32[]", getitem_1877: "f32[]", getitem_1878: "f32[]", getitem_1879: "f32[]", getitem_1880: "f32[]", getitem_1881: "f32[]", getitem_1882: "f32[]", getitem_1883: "f32[]", getitem_1884: "f32[]", getitem_1885: "f32[]", getitem_1886: "f32[]", getitem_1887: "f32[]", getitem_1888: "f32[]", getitem_1889: "f32[]", getitem_1890: "f32[]", getitem_1891: "f32[]", getitem_1892: "f32[]", getitem_1893: "f32[]", getitem_1894: "f32[]", getitem_1895: "f32[]", getitem_1896: "f32[]", getitem_1897: "f32[]", getitem_1898: "f32[]", getitem_1899: "f32[]", getitem_1900: "f32[]", getitem_1901: "f32[]", getitem_1902: "f32[]", getitem_1903: "f32[]", getitem_1904: "f32[]", getitem_1905: "f32[]", getitem_1906: "f32[]", getitem_1907: "f32[]", getitem_1908: "f32[]", getitem_1909: "f32[]", getitem_1910: "f32[]", getitem_1911: "f32[]", getitem_1912: "f32[]", getitem_1913: "f32[]", getitem_1914: "f32[]", getitem_1915: "f32[]", getitem_1916: "f32[]", getitem_1917: "f32[]", getitem_1918: "f32[]", getitem_1919: "f32[]", getitem_1920: "f32[]", getitem_1921: "f32[]", getitem_1922: "f32[]", getitem_1923: "f32[]", getitem_1924: "f32[]", getitem_1925: "f32[]", getitem_1926: "f32[]", getitem_1927: "f32[]", getitem_1928: "f32[]", getitem_1929: "f32[]", getitem_1930: "f32[]", getitem_1931: "f32[]", getitem_1932: "f32[]", getitem_1933: "f32[]", getitem_1934: "f32[]", getitem_1935: "f32[]", getitem_1936: "f32[]", getitem_1937: "f32[]", getitem_1938: "f32[]", getitem_1939: "f32[]", getitem_1940: "f32[]", getitem_1941: "f32[]", getitem_1942: "f32[]", getitem_1943: "f32[]", getitem_1944: "f32[]", getitem_1945: "f32[]", getitem_1946: "f32[]", getitem_1947: "f32[]", getitem_1948: "f32[]", getitem_1949: "f32[]", getitem_1950: "f32[]", getitem_1951: "f32[]", getitem_1952: "f32[]", getitem_1953: "f32[]", getitem_1954: "f32[]", getitem_1955: "f32[]", getitem_1956: "f32[]", getitem_1957: "f32[]", getitem_1958: "f32[]", getitem_1959: "f32[]", getitem_1960: "f32[]", getitem_1961: "f32[]", getitem_1962: "f32[]", getitem_1963: "f32[]", getitem_1964: "f32[]", getitem_1965: "f32[]", getitem_1966: "f32[]", getitem_1967: "f32[]", getitem_1968: "f32[]", getitem_1969: "f32[]", getitem_1970: "f32[]", getitem_1971: "f32[]", getitem_1972: "f32[]", getitem_1973: "f32[]", getitem_1974: "f32[]", getitem_1975: "f32[]", getitem_1976: "f32[]", getitem_1977: "f32[]", getitem_1978: "f32[]", getitem_1979: "f32[]", getitem_1980: "f32[]", getitem_1981: "f32[]", getitem_1982: "f32[]", getitem_1983: "f32[]", getitem_1984: "f32[]", getitem_1985: "f32[]", getitem_1986: "f32[]", getitem_1987: "f32[]", getitem_1988: "f32[]", getitem_1989: "f32[]", getitem_1990: "f32[]", getitem_1991: "f32[]", getitem_1992: "f32[]", getitem_1993: "f32[]", getitem_1994: "f32[]", getitem_1995: "f32[]", getitem_1996: "f32[]", getitem_1997: "f32[]", getitem_1998: "f32[]", getitem_1999: "f32[]", getitem_2000: "f32[]", getitem_2001: "f32[]", getitem_2002: "f32[]", getitem_2003: "f32[]", getitem_2004: "f32[]", getitem_2005: "f32[]", getitem_2006: "f32[]", getitem_2007: "f32[]", getitem_2008: "f32[]", getitem_2009: "f32[]", getitem_2010: "f32[]", getitem_2011: "f32[]", getitem_2012: "f32[]", getitem_2013: "f32[]", getitem_2014: "f32[]", getitem_2015: "f32[]"):
        # No stacktrace found for following nodes
        full_default: "f32[64, 3, 3, 3]" = torch.ops.aten.full.default([64, 3, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[64, 3, 1, 1]" = torch.ops.aten.full.default([64, 3, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[64]" = torch.ops.aten.full.default([64], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[96, 64, 3, 3]" = torch.ops.aten.full.default([96, 64, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[96, 64, 1, 1]" = torch.ops.aten.full.default([96, 64, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[96, 96, 3, 3]" = torch.ops.aten.full.default([96, 96, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[96, 96, 1, 1]" = torch.ops.aten.full.default([96, 96, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[96]" = torch.ops.aten.full.default([96], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[192, 96, 3, 3]" = torch.ops.aten.full.default([192, 96, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[192, 96, 1, 1]" = torch.ops.aten.full.default([192, 96, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[192, 192, 3, 3]" = torch.ops.aten.full.default([192, 192, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[192, 192, 1, 1]" = torch.ops.aten.full.default([192, 192, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[192, 192, 3, 3]" = torch.ops.aten.full.default([192, 192, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[192, 192, 1, 1]" = torch.ops.aten.full.default([192, 192, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[192, 192, 3, 3]" = torch.ops.aten.full.default([192, 192, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[192, 192, 1, 1]" = torch.ops.aten.full.default([192, 192, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[192]" = torch.ops.aten.full.default([192], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[384, 192, 3, 3]" = torch.ops.aten.full.default([384, 192, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[384, 192, 1, 1]" = torch.ops.aten.full.default([384, 192, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_75: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_77: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_79: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_85: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_87: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_95: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_97: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_99: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_105: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_107: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_109: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_110: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_112: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_113: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_114: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_115: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_117: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_119: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_120: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_121: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_122: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_123: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_124: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_125: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_126: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_127: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_128: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_129: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_130: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_131: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_132: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_133: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_134: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_135: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_136: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_137: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_138: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_139: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_140: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_141: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_142: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_143: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_144: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_145: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_146: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_147: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_148: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_149: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_150: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_151: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_152: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_153: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_154: "f32[384, 384, 3, 3]" = torch.ops.aten.full.default([384, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_155: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_156: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_157: "f32[384, 384, 1, 1]" = torch.ops.aten.full.default([384, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_158: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_159: "f32[384]" = torch.ops.aten.full.default([384], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_160: "f32[1408, 384, 3, 3]" = torch.ops.aten.full.default([1408, 384, 3, 3], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_161: "f32[1408]" = torch.ops.aten.full.default([1408], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_162: "f32[1408]" = torch.ops.aten.full.default([1408], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_163: "f32[1408, 384, 1, 1]" = torch.ops.aten.full.default([1408, 384, 1, 1], 0.10000000149011612, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_164: "f32[1408]" = torch.ops.aten.full.default([1408], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_165: "f32[1408]" = torch.ops.aten.full.default([1408], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_166: "f32[1000, 1408]" = torch.ops.aten.full.default([1000, 1408], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_167: "f32[1000]" = torch.ops.aten.full.default([1000], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg171_1, arg169_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_37, full_default_38, full_default_39, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_53, full_default_54, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_65, full_default_66, full_default_67, full_default_68, full_default_69, full_default_70, full_default_71, full_default_72, full_default_73, full_default_74, full_default_75, full_default_76, full_default_77, full_default_78, full_default_79, full_default_80, full_default_81, full_default_82, full_default_83, full_default_84, full_default_85, full_default_86, full_default_87, full_default_88, full_default_89, full_default_90, full_default_91, full_default_92, full_default_93, full_default_94, full_default_95, full_default_96, full_default_97, full_default_98, full_default_99, full_default_100, full_default_101, full_default_102, full_default_103, full_default_104, full_default_105, full_default_106, full_default_107, full_default_108, full_default_109, full_default_110, full_default_111, full_default_112, full_default_113, full_default_114, full_default_115, full_default_116, full_default_117, full_default_118, full_default_119, full_default_120, full_default_121, full_default_122, full_default_123, full_default_124, full_default_125, full_default_126, full_default_127, full_default_128, full_default_129, full_default_130, full_default_131, full_default_132, full_default_133, full_default_134, full_default_135, full_default_136, full_default_137, full_default_138, full_default_139, full_default_140, full_default_141, full_default_142, full_default_143, full_default_144, full_default_145, full_default_146, full_default_147, full_default_148, full_default_149, full_default_150, full_default_151, full_default_152, full_default_153, full_default_154, full_default_155, full_default_156, full_default_157, full_default_158, full_default_159, full_default_160, full_default_161, full_default_162, full_default_163, full_default_164, full_default_165, full_default_166, full_default_167], [getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335]);  arg171_1 = arg169_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = full_default_10 = full_default_11 = full_default_12 = full_default_13 = full_default_14 = full_default_15 = full_default_16 = full_default_17 = full_default_18 = full_default_19 = full_default_20 = full_default_21 = full_default_22 = full_default_23 = full_default_24 = full_default_25 = full_default_26 = full_default_27 = full_default_28 = full_default_29 = full_default_30 = full_default_31 = full_default_32 = full_default_33 = full_default_34 = full_default_35 = full_default_36 = full_default_37 = full_default_38 = full_default_39 = full_default_40 = full_default_41 = full_default_42 = full_default_43 = full_default_44 = full_default_45 = full_default_46 = full_default_47 = full_default_48 = full_default_49 = full_default_50 = full_default_51 = full_default_52 = full_default_53 = full_default_54 = full_default_55 = full_default_56 = full_default_57 = full_default_58 = full_default_59 = full_default_60 = full_default_61 = full_default_62 = full_default_63 = full_default_64 = full_default_65 = full_default_66 = full_default_67 = full_default_68 = full_default_69 = full_default_70 = full_default_71 = full_default_72 = full_default_73 = full_default_74 = full_default_75 = full_default_76 = full_default_77 = full_default_78 = full_default_79 = full_default_80 = full_default_81 = full_default_82 = full_default_83 = full_default_84 = full_default_85 = full_default_86 = full_default_87 = full_default_88 = full_default_89 = full_default_90 = full_default_91 = full_default_92 = full_default_93 = full_default_94 = full_default_95 = full_default_96 = full_default_97 = full_default_98 = full_default_99 = full_default_100 = full_default_101 = full_default_102 = full_default_103 = full_default_104 = full_default_105 = full_default_106 = full_default_107 = full_default_108 = full_default_109 = full_default_110 = full_default_111 = full_default_112 = full_default_113 = full_default_114 = full_default_115 = full_default_116 = full_default_117 = full_default_118 = full_default_119 = full_default_120 = full_default_121 = full_default_122 = full_default_123 = full_default_124 = full_default_125 = full_default_126 = full_default_127 = full_default_128 = full_default_129 = full_default_130 = full_default_131 = full_default_132 = full_default_133 = full_default_134 = full_default_135 = full_default_136 = full_default_137 = full_default_138 = full_default_139 = full_default_140 = full_default_141 = full_default_142 = full_default_143 = full_default_144 = full_default_145 = full_default_146 = full_default_147 = full_default_148 = full_default_149 = full_default_150 = full_default_151 = full_default_152 = full_default_153 = full_default_154 = full_default_155 = full_default_156 = full_default_157 = full_default_158 = full_default_159 = full_default_160 = full_default_161 = full_default_162 = full_default_163 = full_default_164 = full_default_165 = full_default_166 = full_default_167 = getitem_168 = getitem_169 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = getitem_211 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = getitem_216 = getitem_217 = getitem_218 = getitem_219 = getitem_220 = getitem_221 = getitem_222 = getitem_223 = getitem_224 = getitem_225 = getitem_226 = getitem_227 = getitem_228 = getitem_229 = getitem_230 = getitem_231 = getitem_232 = getitem_233 = getitem_234 = getitem_235 = getitem_236 = getitem_237 = getitem_238 = getitem_239 = getitem_240 = getitem_241 = getitem_242 = getitem_243 = getitem_244 = getitem_245 = getitem_246 = getitem_247 = getitem_248 = getitem_249 = getitem_250 = getitem_251 = getitem_252 = getitem_253 = getitem_254 = getitem_255 = getitem_256 = getitem_257 = getitem_258 = getitem_259 = getitem_260 = getitem_261 = getitem_262 = getitem_263 = getitem_264 = getitem_265 = getitem_266 = getitem_267 = getitem_268 = getitem_269 = getitem_270 = getitem_271 = getitem_272 = getitem_273 = getitem_274 = getitem_275 = getitem_276 = getitem_277 = getitem_278 = getitem_279 = getitem_280 = getitem_281 = getitem_282 = getitem_283 = getitem_284 = getitem_285 = getitem_286 = getitem_287 = getitem_288 = getitem_289 = getitem_290 = getitem_291 = getitem_292 = getitem_293 = getitem_294 = getitem_295 = getitem_296 = getitem_297 = getitem_298 = getitem_299 = getitem_300 = getitem_301 = getitem_302 = getitem_303 = getitem_304 = getitem_305 = getitem_306 = getitem_307 = getitem_308 = getitem_309 = getitem_310 = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = getitem_317 = getitem_318 = getitem_319 = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = getitem_325 = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = getitem_331 = getitem_332 = getitem_333 = getitem_334 = getitem_335 = None
        getitem: "f32[64, 3, 3, 3]" = _foreach_addcmul_scalar[0]
        getitem_336: "f32[64]" = _foreach_addcmul_scalar[1]
        getitem_337: "f32[64]" = _foreach_addcmul_scalar[2]
        getitem_338: "f32[64, 3, 1, 1]" = _foreach_addcmul_scalar[3]
        getitem_339: "f32[64]" = _foreach_addcmul_scalar[4]
        getitem_340: "f32[64]" = _foreach_addcmul_scalar[5]
        getitem_341: "f32[96, 64, 3, 3]" = _foreach_addcmul_scalar[6]
        getitem_342: "f32[96]" = _foreach_addcmul_scalar[7]
        getitem_343: "f32[96]" = _foreach_addcmul_scalar[8]
        getitem_344: "f32[96, 64, 1, 1]" = _foreach_addcmul_scalar[9]
        getitem_345: "f32[96]" = _foreach_addcmul_scalar[10]
        getitem_346: "f32[96]" = _foreach_addcmul_scalar[11]
        getitem_347: "f32[96]" = _foreach_addcmul_scalar[12]
        getitem_348: "f32[96]" = _foreach_addcmul_scalar[13]
        getitem_349: "f32[96, 96, 3, 3]" = _foreach_addcmul_scalar[14]
        getitem_350: "f32[96]" = _foreach_addcmul_scalar[15]
        getitem_351: "f32[96]" = _foreach_addcmul_scalar[16]
        getitem_352: "f32[96, 96, 1, 1]" = _foreach_addcmul_scalar[17]
        getitem_353: "f32[96]" = _foreach_addcmul_scalar[18]
        getitem_354: "f32[96]" = _foreach_addcmul_scalar[19]
        getitem_355: "f32[192, 96, 3, 3]" = _foreach_addcmul_scalar[20]
        getitem_356: "f32[192]" = _foreach_addcmul_scalar[21]
        getitem_357: "f32[192]" = _foreach_addcmul_scalar[22]
        getitem_358: "f32[192, 96, 1, 1]" = _foreach_addcmul_scalar[23]
        getitem_359: "f32[192]" = _foreach_addcmul_scalar[24]
        getitem_360: "f32[192]" = _foreach_addcmul_scalar[25]
        getitem_361: "f32[192]" = _foreach_addcmul_scalar[26]
        getitem_362: "f32[192]" = _foreach_addcmul_scalar[27]
        getitem_363: "f32[192, 192, 3, 3]" = _foreach_addcmul_scalar[28]
        getitem_364: "f32[192]" = _foreach_addcmul_scalar[29]
        getitem_365: "f32[192]" = _foreach_addcmul_scalar[30]
        getitem_366: "f32[192, 192, 1, 1]" = _foreach_addcmul_scalar[31]
        getitem_367: "f32[192]" = _foreach_addcmul_scalar[32]
        getitem_368: "f32[192]" = _foreach_addcmul_scalar[33]
        getitem_369: "f32[192]" = _foreach_addcmul_scalar[34]
        getitem_370: "f32[192]" = _foreach_addcmul_scalar[35]
        getitem_371: "f32[192, 192, 3, 3]" = _foreach_addcmul_scalar[36]
        getitem_372: "f32[192]" = _foreach_addcmul_scalar[37]
        getitem_373: "f32[192]" = _foreach_addcmul_scalar[38]
        getitem_374: "f32[192, 192, 1, 1]" = _foreach_addcmul_scalar[39]
        getitem_375: "f32[192]" = _foreach_addcmul_scalar[40]
        getitem_376: "f32[192]" = _foreach_addcmul_scalar[41]
        getitem_377: "f32[192]" = _foreach_addcmul_scalar[42]
        getitem_378: "f32[192]" = _foreach_addcmul_scalar[43]
        getitem_379: "f32[192, 192, 3, 3]" = _foreach_addcmul_scalar[44]
        getitem_380: "f32[192]" = _foreach_addcmul_scalar[45]
        getitem_381: "f32[192]" = _foreach_addcmul_scalar[46]
        getitem_382: "f32[192, 192, 1, 1]" = _foreach_addcmul_scalar[47]
        getitem_383: "f32[192]" = _foreach_addcmul_scalar[48]
        getitem_384: "f32[192]" = _foreach_addcmul_scalar[49]
        getitem_385: "f32[384, 192, 3, 3]" = _foreach_addcmul_scalar[50]
        getitem_386: "f32[384]" = _foreach_addcmul_scalar[51]
        getitem_387: "f32[384]" = _foreach_addcmul_scalar[52]
        getitem_388: "f32[384, 192, 1, 1]" = _foreach_addcmul_scalar[53]
        getitem_389: "f32[384]" = _foreach_addcmul_scalar[54]
        getitem_390: "f32[384]" = _foreach_addcmul_scalar[55]
        getitem_391: "f32[384]" = _foreach_addcmul_scalar[56]
        getitem_392: "f32[384]" = _foreach_addcmul_scalar[57]
        getitem_393: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[58]
        getitem_394: "f32[384]" = _foreach_addcmul_scalar[59]
        getitem_395: "f32[384]" = _foreach_addcmul_scalar[60]
        getitem_396: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[61]
        getitem_397: "f32[384]" = _foreach_addcmul_scalar[62]
        getitem_398: "f32[384]" = _foreach_addcmul_scalar[63]
        getitem_399: "f32[384]" = _foreach_addcmul_scalar[64]
        getitem_400: "f32[384]" = _foreach_addcmul_scalar[65]
        getitem_401: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[66]
        getitem_402: "f32[384]" = _foreach_addcmul_scalar[67]
        getitem_403: "f32[384]" = _foreach_addcmul_scalar[68]
        getitem_404: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[69]
        getitem_405: "f32[384]" = _foreach_addcmul_scalar[70]
        getitem_406: "f32[384]" = _foreach_addcmul_scalar[71]
        getitem_407: "f32[384]" = _foreach_addcmul_scalar[72]
        getitem_408: "f32[384]" = _foreach_addcmul_scalar[73]
        getitem_409: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[74]
        getitem_410: "f32[384]" = _foreach_addcmul_scalar[75]
        getitem_411: "f32[384]" = _foreach_addcmul_scalar[76]
        getitem_412: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[77]
        getitem_413: "f32[384]" = _foreach_addcmul_scalar[78]
        getitem_414: "f32[384]" = _foreach_addcmul_scalar[79]
        getitem_415: "f32[384]" = _foreach_addcmul_scalar[80]
        getitem_416: "f32[384]" = _foreach_addcmul_scalar[81]
        getitem_417: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[82]
        getitem_418: "f32[384]" = _foreach_addcmul_scalar[83]
        getitem_419: "f32[384]" = _foreach_addcmul_scalar[84]
        getitem_420: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[85]
        getitem_421: "f32[384]" = _foreach_addcmul_scalar[86]
        getitem_422: "f32[384]" = _foreach_addcmul_scalar[87]
        getitem_423: "f32[384]" = _foreach_addcmul_scalar[88]
        getitem_424: "f32[384]" = _foreach_addcmul_scalar[89]
        getitem_425: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[90]
        getitem_426: "f32[384]" = _foreach_addcmul_scalar[91]
        getitem_427: "f32[384]" = _foreach_addcmul_scalar[92]
        getitem_428: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[93]
        getitem_429: "f32[384]" = _foreach_addcmul_scalar[94]
        getitem_430: "f32[384]" = _foreach_addcmul_scalar[95]
        getitem_431: "f32[384]" = _foreach_addcmul_scalar[96]
        getitem_432: "f32[384]" = _foreach_addcmul_scalar[97]
        getitem_433: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[98]
        getitem_434: "f32[384]" = _foreach_addcmul_scalar[99]
        getitem_435: "f32[384]" = _foreach_addcmul_scalar[100]
        getitem_436: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[101]
        getitem_437: "f32[384]" = _foreach_addcmul_scalar[102]
        getitem_438: "f32[384]" = _foreach_addcmul_scalar[103]
        getitem_439: "f32[384]" = _foreach_addcmul_scalar[104]
        getitem_440: "f32[384]" = _foreach_addcmul_scalar[105]
        getitem_441: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[106]
        getitem_442: "f32[384]" = _foreach_addcmul_scalar[107]
        getitem_443: "f32[384]" = _foreach_addcmul_scalar[108]
        getitem_444: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[109]
        getitem_445: "f32[384]" = _foreach_addcmul_scalar[110]
        getitem_446: "f32[384]" = _foreach_addcmul_scalar[111]
        getitem_447: "f32[384]" = _foreach_addcmul_scalar[112]
        getitem_448: "f32[384]" = _foreach_addcmul_scalar[113]
        getitem_449: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[114]
        getitem_450: "f32[384]" = _foreach_addcmul_scalar[115]
        getitem_451: "f32[384]" = _foreach_addcmul_scalar[116]
        getitem_452: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[117]
        getitem_453: "f32[384]" = _foreach_addcmul_scalar[118]
        getitem_454: "f32[384]" = _foreach_addcmul_scalar[119]
        getitem_455: "f32[384]" = _foreach_addcmul_scalar[120]
        getitem_456: "f32[384]" = _foreach_addcmul_scalar[121]
        getitem_457: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[122]
        getitem_458: "f32[384]" = _foreach_addcmul_scalar[123]
        getitem_459: "f32[384]" = _foreach_addcmul_scalar[124]
        getitem_460: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[125]
        getitem_461: "f32[384]" = _foreach_addcmul_scalar[126]
        getitem_462: "f32[384]" = _foreach_addcmul_scalar[127]
        getitem_463: "f32[384]" = _foreach_addcmul_scalar[128]
        getitem_464: "f32[384]" = _foreach_addcmul_scalar[129]
        getitem_465: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[130]
        getitem_466: "f32[384]" = _foreach_addcmul_scalar[131]
        getitem_467: "f32[384]" = _foreach_addcmul_scalar[132]
        getitem_468: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[133]
        getitem_469: "f32[384]" = _foreach_addcmul_scalar[134]
        getitem_470: "f32[384]" = _foreach_addcmul_scalar[135]
        getitem_471: "f32[384]" = _foreach_addcmul_scalar[136]
        getitem_472: "f32[384]" = _foreach_addcmul_scalar[137]
        getitem_473: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[138]
        getitem_474: "f32[384]" = _foreach_addcmul_scalar[139]
        getitem_475: "f32[384]" = _foreach_addcmul_scalar[140]
        getitem_476: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[141]
        getitem_477: "f32[384]" = _foreach_addcmul_scalar[142]
        getitem_478: "f32[384]" = _foreach_addcmul_scalar[143]
        getitem_479: "f32[384]" = _foreach_addcmul_scalar[144]
        getitem_480: "f32[384]" = _foreach_addcmul_scalar[145]
        getitem_481: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[146]
        getitem_482: "f32[384]" = _foreach_addcmul_scalar[147]
        getitem_483: "f32[384]" = _foreach_addcmul_scalar[148]
        getitem_484: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[149]
        getitem_485: "f32[384]" = _foreach_addcmul_scalar[150]
        getitem_486: "f32[384]" = _foreach_addcmul_scalar[151]
        getitem_487: "f32[384]" = _foreach_addcmul_scalar[152]
        getitem_488: "f32[384]" = _foreach_addcmul_scalar[153]
        getitem_489: "f32[384, 384, 3, 3]" = _foreach_addcmul_scalar[154]
        getitem_490: "f32[384]" = _foreach_addcmul_scalar[155]
        getitem_491: "f32[384]" = _foreach_addcmul_scalar[156]
        getitem_492: "f32[384, 384, 1, 1]" = _foreach_addcmul_scalar[157]
        getitem_493: "f32[384]" = _foreach_addcmul_scalar[158]
        getitem_494: "f32[384]" = _foreach_addcmul_scalar[159]
        getitem_495: "f32[1408, 384, 3, 3]" = _foreach_addcmul_scalar[160]
        getitem_496: "f32[1408]" = _foreach_addcmul_scalar[161]
        getitem_497: "f32[1408]" = _foreach_addcmul_scalar[162]
        getitem_498: "f32[1408, 384, 1, 1]" = _foreach_addcmul_scalar[163]
        getitem_499: "f32[1408]" = _foreach_addcmul_scalar[164]
        getitem_500: "f32[1408]" = _foreach_addcmul_scalar[165]
        getitem_501: "f32[1000, 1408]" = _foreach_addcmul_scalar[166]
        getitem_502: "f32[1000]" = _foreach_addcmul_scalar[167];  _foreach_addcmul_scalar = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677, getitem_2678, getitem_2679, getitem_2680, getitem_2681, getitem_2682, getitem_2683, getitem_2684, getitem_2685, getitem_2686, getitem_2687], [getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015]);  getitem_2520 = getitem_2521 = getitem_2522 = getitem_2523 = getitem_2524 = getitem_2525 = getitem_2526 = getitem_2527 = getitem_2528 = getitem_2529 = getitem_2530 = getitem_2531 = getitem_2532 = getitem_2533 = getitem_2534 = getitem_2535 = getitem_2536 = getitem_2537 = getitem_2538 = getitem_2539 = getitem_2540 = getitem_2541 = getitem_2542 = getitem_2543 = getitem_2544 = getitem_2545 = getitem_2546 = getitem_2547 = getitem_2548 = getitem_2549 = getitem_2550 = getitem_2551 = getitem_2552 = getitem_2553 = getitem_2554 = getitem_2555 = getitem_2556 = getitem_2557 = getitem_2558 = getitem_2559 = getitem_2560 = getitem_2561 = getitem_2562 = getitem_2563 = getitem_2564 = getitem_2565 = getitem_2566 = getitem_2567 = getitem_2568 = getitem_2569 = getitem_2570 = getitem_2571 = getitem_2572 = getitem_2573 = getitem_2574 = getitem_2575 = getitem_2576 = getitem_2577 = getitem_2578 = getitem_2579 = getitem_2580 = getitem_2581 = getitem_2582 = getitem_2583 = getitem_2584 = getitem_2585 = getitem_2586 = getitem_2587 = getitem_2588 = getitem_2589 = getitem_2590 = getitem_2591 = getitem_2592 = getitem_2593 = getitem_2594 = getitem_2595 = getitem_2596 = getitem_2597 = getitem_2598 = getitem_2599 = getitem_2600 = getitem_2601 = getitem_2602 = getitem_2603 = getitem_2604 = getitem_2605 = getitem_2606 = getitem_2607 = getitem_2608 = getitem_2609 = getitem_2610 = getitem_2611 = getitem_2612 = getitem_2613 = getitem_2614 = getitem_2615 = getitem_2616 = getitem_2617 = getitem_2618 = getitem_2619 = getitem_2620 = getitem_2621 = getitem_2622 = getitem_2623 = getitem_2624 = getitem_2625 = getitem_2626 = getitem_2627 = getitem_2628 = getitem_2629 = getitem_2630 = getitem_2631 = getitem_2632 = getitem_2633 = getitem_2634 = getitem_2635 = getitem_2636 = getitem_2637 = getitem_2638 = getitem_2639 = getitem_2640 = getitem_2641 = getitem_2642 = getitem_2643 = getitem_2644 = getitem_2645 = getitem_2646 = getitem_2647 = getitem_2648 = getitem_2649 = getitem_2650 = getitem_2651 = getitem_2652 = getitem_2653 = getitem_2654 = getitem_2655 = getitem_2656 = getitem_2657 = getitem_2658 = getitem_2659 = getitem_2660 = getitem_2661 = getitem_2662 = getitem_2663 = getitem_2664 = getitem_2665 = getitem_2666 = getitem_2667 = getitem_2668 = getitem_2669 = getitem_2670 = getitem_2671 = getitem_2672 = getitem_2673 = getitem_2674 = getitem_2675 = getitem_2676 = getitem_2677 = getitem_2678 = getitem_2679 = getitem_2680 = getitem_2681 = getitem_2682 = getitem_2683 = getitem_2684 = getitem_2685 = getitem_2686 = getitem_2687 = getitem_1848 = getitem_1849 = getitem_1850 = getitem_1851 = getitem_1852 = getitem_1853 = getitem_1854 = getitem_1855 = getitem_1856 = getitem_1857 = getitem_1858 = getitem_1859 = getitem_1860 = getitem_1861 = getitem_1862 = getitem_1863 = getitem_1864 = getitem_1865 = getitem_1866 = getitem_1867 = getitem_1868 = getitem_1869 = getitem_1870 = getitem_1871 = getitem_1872 = getitem_1873 = getitem_1874 = getitem_1875 = getitem_1876 = getitem_1877 = getitem_1878 = getitem_1879 = getitem_1880 = getitem_1881 = getitem_1882 = getitem_1883 = getitem_1884 = getitem_1885 = getitem_1886 = getitem_1887 = getitem_1888 = getitem_1889 = getitem_1890 = getitem_1891 = getitem_1892 = getitem_1893 = getitem_1894 = getitem_1895 = getitem_1896 = getitem_1897 = getitem_1898 = getitem_1899 = getitem_1900 = getitem_1901 = getitem_1902 = getitem_1903 = getitem_1904 = getitem_1905 = getitem_1906 = getitem_1907 = getitem_1908 = getitem_1909 = getitem_1910 = getitem_1911 = getitem_1912 = getitem_1913 = getitem_1914 = getitem_1915 = getitem_1916 = getitem_1917 = getitem_1918 = getitem_1919 = getitem_1920 = getitem_1921 = getitem_1922 = getitem_1923 = getitem_1924 = getitem_1925 = getitem_1926 = getitem_1927 = getitem_1928 = getitem_1929 = getitem_1930 = getitem_1931 = getitem_1932 = getitem_1933 = getitem_1934 = getitem_1935 = getitem_1936 = getitem_1937 = getitem_1938 = getitem_1939 = getitem_1940 = getitem_1941 = getitem_1942 = getitem_1943 = getitem_1944 = getitem_1945 = getitem_1946 = getitem_1947 = getitem_1948 = getitem_1949 = getitem_1950 = getitem_1951 = getitem_1952 = getitem_1953 = getitem_1954 = getitem_1955 = getitem_1956 = getitem_1957 = getitem_1958 = getitem_1959 = getitem_1960 = getitem_1961 = getitem_1962 = getitem_1963 = getitem_1964 = getitem_1965 = getitem_1966 = getitem_1967 = getitem_1968 = getitem_1969 = getitem_1970 = getitem_1971 = getitem_1972 = getitem_1973 = getitem_1974 = getitem_1975 = getitem_1976 = getitem_1977 = getitem_1978 = getitem_1979 = getitem_1980 = getitem_1981 = getitem_1982 = getitem_1983 = getitem_1984 = getitem_1985 = getitem_1986 = getitem_1987 = getitem_1988 = getitem_1989 = getitem_1990 = getitem_1991 = getitem_1992 = getitem_1993 = getitem_1994 = getitem_1995 = getitem_1996 = getitem_1997 = getitem_1998 = getitem_1999 = getitem_2000 = getitem_2001 = getitem_2002 = getitem_2003 = getitem_2004 = getitem_2005 = getitem_2006 = getitem_2007 = getitem_2008 = getitem_2009 = getitem_2010 = getitem_2011 = getitem_2012 = getitem_2013 = getitem_2014 = getitem_2015 = None
        getitem_2016: "f32[64, 3, 3, 3]" = _foreach_div_list[0]
        getitem_2017: "f32[64]" = _foreach_div_list[1]
        getitem_2018: "f32[64]" = _foreach_div_list[2]
        getitem_2019: "f32[64, 3, 1, 1]" = _foreach_div_list[3]
        getitem_2020: "f32[64]" = _foreach_div_list[4]
        getitem_2021: "f32[64]" = _foreach_div_list[5]
        getitem_2022: "f32[96, 64, 3, 3]" = _foreach_div_list[6]
        getitem_2023: "f32[96]" = _foreach_div_list[7]
        getitem_2024: "f32[96]" = _foreach_div_list[8]
        getitem_2025: "f32[96, 64, 1, 1]" = _foreach_div_list[9]
        getitem_2026: "f32[96]" = _foreach_div_list[10]
        getitem_2027: "f32[96]" = _foreach_div_list[11]
        getitem_2028: "f32[96]" = _foreach_div_list[12]
        getitem_2029: "f32[96]" = _foreach_div_list[13]
        getitem_2030: "f32[96, 96, 3, 3]" = _foreach_div_list[14]
        getitem_2031: "f32[96]" = _foreach_div_list[15]
        getitem_2032: "f32[96]" = _foreach_div_list[16]
        getitem_2033: "f32[96, 96, 1, 1]" = _foreach_div_list[17]
        getitem_2034: "f32[96]" = _foreach_div_list[18]
        getitem_2035: "f32[96]" = _foreach_div_list[19]
        getitem_2036: "f32[192, 96, 3, 3]" = _foreach_div_list[20]
        getitem_2037: "f32[192]" = _foreach_div_list[21]
        getitem_2038: "f32[192]" = _foreach_div_list[22]
        getitem_2039: "f32[192, 96, 1, 1]" = _foreach_div_list[23]
        getitem_2040: "f32[192]" = _foreach_div_list[24]
        getitem_2041: "f32[192]" = _foreach_div_list[25]
        getitem_2042: "f32[192]" = _foreach_div_list[26]
        getitem_2043: "f32[192]" = _foreach_div_list[27]
        getitem_2044: "f32[192, 192, 3, 3]" = _foreach_div_list[28]
        getitem_2045: "f32[192]" = _foreach_div_list[29]
        getitem_2046: "f32[192]" = _foreach_div_list[30]
        getitem_2047: "f32[192, 192, 1, 1]" = _foreach_div_list[31]
        getitem_2048: "f32[192]" = _foreach_div_list[32]
        getitem_2049: "f32[192]" = _foreach_div_list[33]
        getitem_2050: "f32[192]" = _foreach_div_list[34]
        getitem_2051: "f32[192]" = _foreach_div_list[35]
        getitem_2052: "f32[192, 192, 3, 3]" = _foreach_div_list[36]
        getitem_2053: "f32[192]" = _foreach_div_list[37]
        getitem_2054: "f32[192]" = _foreach_div_list[38]
        getitem_2055: "f32[192, 192, 1, 1]" = _foreach_div_list[39]
        getitem_2056: "f32[192]" = _foreach_div_list[40]
        getitem_2057: "f32[192]" = _foreach_div_list[41]
        getitem_2058: "f32[192]" = _foreach_div_list[42]
        getitem_2059: "f32[192]" = _foreach_div_list[43]
        getitem_2060: "f32[192, 192, 3, 3]" = _foreach_div_list[44]
        getitem_2061: "f32[192]" = _foreach_div_list[45]
        getitem_2062: "f32[192]" = _foreach_div_list[46]
        getitem_2063: "f32[192, 192, 1, 1]" = _foreach_div_list[47]
        getitem_2064: "f32[192]" = _foreach_div_list[48]
        getitem_2065: "f32[192]" = _foreach_div_list[49]
        getitem_2066: "f32[384, 192, 3, 3]" = _foreach_div_list[50]
        getitem_2067: "f32[384]" = _foreach_div_list[51]
        getitem_2068: "f32[384]" = _foreach_div_list[52]
        getitem_2069: "f32[384, 192, 1, 1]" = _foreach_div_list[53]
        getitem_2070: "f32[384]" = _foreach_div_list[54]
        getitem_2071: "f32[384]" = _foreach_div_list[55]
        getitem_2072: "f32[384]" = _foreach_div_list[56]
        getitem_2073: "f32[384]" = _foreach_div_list[57]
        getitem_2074: "f32[384, 384, 3, 3]" = _foreach_div_list[58]
        getitem_2075: "f32[384]" = _foreach_div_list[59]
        getitem_2076: "f32[384]" = _foreach_div_list[60]
        getitem_2077: "f32[384, 384, 1, 1]" = _foreach_div_list[61]
        getitem_2078: "f32[384]" = _foreach_div_list[62]
        getitem_2079: "f32[384]" = _foreach_div_list[63]
        getitem_2080: "f32[384]" = _foreach_div_list[64]
        getitem_2081: "f32[384]" = _foreach_div_list[65]
        getitem_2082: "f32[384, 384, 3, 3]" = _foreach_div_list[66]
        getitem_2083: "f32[384]" = _foreach_div_list[67]
        getitem_2084: "f32[384]" = _foreach_div_list[68]
        getitem_2085: "f32[384, 384, 1, 1]" = _foreach_div_list[69]
        getitem_2086: "f32[384]" = _foreach_div_list[70]
        getitem_2087: "f32[384]" = _foreach_div_list[71]
        getitem_2088: "f32[384]" = _foreach_div_list[72]
        getitem_2089: "f32[384]" = _foreach_div_list[73]
        getitem_2090: "f32[384, 384, 3, 3]" = _foreach_div_list[74]
        getitem_2091: "f32[384]" = _foreach_div_list[75]
        getitem_2092: "f32[384]" = _foreach_div_list[76]
        getitem_2093: "f32[384, 384, 1, 1]" = _foreach_div_list[77]
        getitem_2094: "f32[384]" = _foreach_div_list[78]
        getitem_2095: "f32[384]" = _foreach_div_list[79]
        getitem_2096: "f32[384]" = _foreach_div_list[80]
        getitem_2097: "f32[384]" = _foreach_div_list[81]
        getitem_2098: "f32[384, 384, 3, 3]" = _foreach_div_list[82]
        getitem_2099: "f32[384]" = _foreach_div_list[83]
        getitem_2100: "f32[384]" = _foreach_div_list[84]
        getitem_2101: "f32[384, 384, 1, 1]" = _foreach_div_list[85]
        getitem_2102: "f32[384]" = _foreach_div_list[86]
        getitem_2103: "f32[384]" = _foreach_div_list[87]
        getitem_2104: "f32[384]" = _foreach_div_list[88]
        getitem_2105: "f32[384]" = _foreach_div_list[89]
        getitem_2106: "f32[384, 384, 3, 3]" = _foreach_div_list[90]
        getitem_2107: "f32[384]" = _foreach_div_list[91]
        getitem_2108: "f32[384]" = _foreach_div_list[92]
        getitem_2109: "f32[384, 384, 1, 1]" = _foreach_div_list[93]
        getitem_2110: "f32[384]" = _foreach_div_list[94]
        getitem_2111: "f32[384]" = _foreach_div_list[95]
        getitem_2112: "f32[384]" = _foreach_div_list[96]
        getitem_2113: "f32[384]" = _foreach_div_list[97]
        getitem_2114: "f32[384, 384, 3, 3]" = _foreach_div_list[98]
        getitem_2115: "f32[384]" = _foreach_div_list[99]
        getitem_2116: "f32[384]" = _foreach_div_list[100]
        getitem_2117: "f32[384, 384, 1, 1]" = _foreach_div_list[101]
        getitem_2118: "f32[384]" = _foreach_div_list[102]
        getitem_2119: "f32[384]" = _foreach_div_list[103]
        getitem_2120: "f32[384]" = _foreach_div_list[104]
        getitem_2121: "f32[384]" = _foreach_div_list[105]
        getitem_2122: "f32[384, 384, 3, 3]" = _foreach_div_list[106]
        getitem_2123: "f32[384]" = _foreach_div_list[107]
        getitem_2124: "f32[384]" = _foreach_div_list[108]
        getitem_2125: "f32[384, 384, 1, 1]" = _foreach_div_list[109]
        getitem_2126: "f32[384]" = _foreach_div_list[110]
        getitem_2127: "f32[384]" = _foreach_div_list[111]
        getitem_2128: "f32[384]" = _foreach_div_list[112]
        getitem_2129: "f32[384]" = _foreach_div_list[113]
        getitem_2130: "f32[384, 384, 3, 3]" = _foreach_div_list[114]
        getitem_2131: "f32[384]" = _foreach_div_list[115]
        getitem_2132: "f32[384]" = _foreach_div_list[116]
        getitem_2133: "f32[384, 384, 1, 1]" = _foreach_div_list[117]
        getitem_2134: "f32[384]" = _foreach_div_list[118]
        getitem_2135: "f32[384]" = _foreach_div_list[119]
        getitem_2136: "f32[384]" = _foreach_div_list[120]
        getitem_2137: "f32[384]" = _foreach_div_list[121]
        getitem_2138: "f32[384, 384, 3, 3]" = _foreach_div_list[122]
        getitem_2139: "f32[384]" = _foreach_div_list[123]
        getitem_2140: "f32[384]" = _foreach_div_list[124]
        getitem_2141: "f32[384, 384, 1, 1]" = _foreach_div_list[125]
        getitem_2142: "f32[384]" = _foreach_div_list[126]
        getitem_2143: "f32[384]" = _foreach_div_list[127]
        getitem_2144: "f32[384]" = _foreach_div_list[128]
        getitem_2145: "f32[384]" = _foreach_div_list[129]
        getitem_2146: "f32[384, 384, 3, 3]" = _foreach_div_list[130]
        getitem_2147: "f32[384]" = _foreach_div_list[131]
        getitem_2148: "f32[384]" = _foreach_div_list[132]
        getitem_2149: "f32[384, 384, 1, 1]" = _foreach_div_list[133]
        getitem_2150: "f32[384]" = _foreach_div_list[134]
        getitem_2151: "f32[384]" = _foreach_div_list[135]
        getitem_2152: "f32[384]" = _foreach_div_list[136]
        getitem_2153: "f32[384]" = _foreach_div_list[137]
        getitem_2154: "f32[384, 384, 3, 3]" = _foreach_div_list[138]
        getitem_2155: "f32[384]" = _foreach_div_list[139]
        getitem_2156: "f32[384]" = _foreach_div_list[140]
        getitem_2157: "f32[384, 384, 1, 1]" = _foreach_div_list[141]
        getitem_2158: "f32[384]" = _foreach_div_list[142]
        getitem_2159: "f32[384]" = _foreach_div_list[143]
        getitem_2160: "f32[384]" = _foreach_div_list[144]
        getitem_2161: "f32[384]" = _foreach_div_list[145]
        getitem_2162: "f32[384, 384, 3, 3]" = _foreach_div_list[146]
        getitem_2163: "f32[384]" = _foreach_div_list[147]
        getitem_2164: "f32[384]" = _foreach_div_list[148]
        getitem_2165: "f32[384, 384, 1, 1]" = _foreach_div_list[149]
        getitem_2166: "f32[384]" = _foreach_div_list[150]
        getitem_2167: "f32[384]" = _foreach_div_list[151]
        getitem_2168: "f32[384]" = _foreach_div_list[152]
        getitem_2169: "f32[384]" = _foreach_div_list[153]
        getitem_2170: "f32[384, 384, 3, 3]" = _foreach_div_list[154]
        getitem_2171: "f32[384]" = _foreach_div_list[155]
        getitem_2172: "f32[384]" = _foreach_div_list[156]
        getitem_2173: "f32[384, 384, 1, 1]" = _foreach_div_list[157]
        getitem_2174: "f32[384]" = _foreach_div_list[158]
        getitem_2175: "f32[384]" = _foreach_div_list[159]
        getitem_2176: "f32[1408, 384, 3, 3]" = _foreach_div_list[160]
        getitem_2177: "f32[1408]" = _foreach_div_list[161]
        getitem_2178: "f32[1408]" = _foreach_div_list[162]
        getitem_2179: "f32[1408, 384, 1, 1]" = _foreach_div_list[163]
        getitem_2180: "f32[1408]" = _foreach_div_list[164]
        getitem_2181: "f32[1408]" = _foreach_div_list[165]
        getitem_2182: "f32[1000, 1408]" = _foreach_div_list[166]
        getitem_2183: "f32[1000]" = _foreach_div_list[167];  _foreach_div_list = None
        return (getitem, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059, getitem_2060, getitem_2061, getitem_2062, getitem_2063, getitem_2064, getitem_2065, getitem_2066, getitem_2067, getitem_2068, getitem_2069, getitem_2070, getitem_2071, getitem_2072, getitem_2073, getitem_2074, getitem_2075, getitem_2076, getitem_2077, getitem_2078, getitem_2079, getitem_2080, getitem_2081, getitem_2082, getitem_2083, getitem_2084, getitem_2085, getitem_2086, getitem_2087, getitem_2088, getitem_2089, getitem_2090, getitem_2091, getitem_2092, getitem_2093, getitem_2094, getitem_2095, getitem_2096, getitem_2097, getitem_2098, getitem_2099, getitem_2100, getitem_2101, getitem_2102, getitem_2103, getitem_2104, getitem_2105, getitem_2106, getitem_2107, getitem_2108, getitem_2109, getitem_2110, getitem_2111, getitem_2112, getitem_2113, getitem_2114, getitem_2115, getitem_2116, getitem_2117, getitem_2118, getitem_2119, getitem_2120, getitem_2121, getitem_2122, getitem_2123, getitem_2124, getitem_2125, getitem_2126, getitem_2127, getitem_2128, getitem_2129, getitem_2130, getitem_2131, getitem_2132, getitem_2133, getitem_2134, getitem_2135, getitem_2136, getitem_2137, getitem_2138, getitem_2139, getitem_2140, getitem_2141, getitem_2142, getitem_2143, getitem_2144, getitem_2145, getitem_2146, getitem_2147, getitem_2148, getitem_2149, getitem_2150, getitem_2151, getitem_2152, getitem_2153, getitem_2154, getitem_2155, getitem_2156, getitem_2157, getitem_2158, getitem_2159, getitem_2160, getitem_2161, getitem_2162, getitem_2163, getitem_2164, getitem_2165, getitem_2166, getitem_2167, getitem_2168, getitem_2169, getitem_2170, getitem_2171, getitem_2172, getitem_2173, getitem_2174, getitem_2175, getitem_2176, getitem_2177, getitem_2178, getitem_2179, getitem_2180, getitem_2181, getitem_2182, getitem_2183)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

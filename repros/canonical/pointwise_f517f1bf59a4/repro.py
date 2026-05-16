"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s2_g77
Pattern hash: f517f1bf59a4
Shape hash: 244d2bfa
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 1, 768]", arg1_1: "f32[1, 198, 768]", arg2_1: "f32[1, 1, 768]", arg3_1: "f32[768, 3, 16, 16]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[768]", arg7_1: "f32[2304, 768]", arg8_1: "f32[2304]", arg9_1: "f32[768, 768]", arg10_1: "f32[768]", arg11_1: "f32[768]", arg12_1: "f32[768]", arg13_1: "f32[3072, 768]", arg14_1: "f32[3072]", arg15_1: "f32[768, 3072]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[768]", arg19_1: "f32[2304, 768]", arg20_1: "f32[2304]", arg21_1: "f32[768, 768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[3072, 768]", arg26_1: "f32[3072]", arg27_1: "f32[768, 3072]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[2304, 768]", arg32_1: "f32[2304]", arg33_1: "f32[768, 768]", arg34_1: "f32[768]", arg35_1: "f32[768]", arg36_1: "f32[768]", arg37_1: "f32[3072, 768]", arg38_1: "f32[3072]", arg39_1: "f32[768, 3072]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[768]", arg43_1: "f32[2304, 768]", arg44_1: "f32[2304]", arg45_1: "f32[768, 768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[3072, 768]", arg50_1: "f32[3072]", arg51_1: "f32[768, 3072]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[2304, 768]", arg56_1: "f32[2304]", arg57_1: "f32[768, 768]", arg58_1: "f32[768]", arg59_1: "f32[768]", arg60_1: "f32[768]", arg61_1: "f32[3072, 768]", arg62_1: "f32[3072]", arg63_1: "f32[768, 3072]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[768]", arg67_1: "f32[2304, 768]", arg68_1: "f32[2304]", arg69_1: "f32[768, 768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[3072, 768]", arg74_1: "f32[3072]", arg75_1: "f32[768, 3072]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[2304, 768]", arg80_1: "f32[2304]", arg81_1: "f32[768, 768]", arg82_1: "f32[768]", arg83_1: "f32[768]", arg84_1: "f32[768]", arg85_1: "f32[3072, 768]", arg86_1: "f32[3072]", arg87_1: "f32[768, 3072]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[768]", arg91_1: "f32[2304, 768]", arg92_1: "f32[2304]", arg93_1: "f32[768, 768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[3072, 768]", arg98_1: "f32[3072]", arg99_1: "f32[768, 3072]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[2304, 768]", arg104_1: "f32[2304]", arg105_1: "f32[768, 768]", arg106_1: "f32[768]", arg107_1: "f32[768]", arg108_1: "f32[768]", arg109_1: "f32[3072, 768]", arg110_1: "f32[3072]", arg111_1: "f32[768, 3072]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[768]", arg115_1: "f32[2304, 768]", arg116_1: "f32[2304]", arg117_1: "f32[768, 768]", arg118_1: "f32[768]", arg119_1: "f32[768]", arg120_1: "f32[768]", arg121_1: "f32[3072, 768]", arg122_1: "f32[3072]", arg123_1: "f32[768, 3072]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[768]", arg127_1: "f32[2304, 768]", arg128_1: "f32[2304]", arg129_1: "f32[768, 768]", arg130_1: "f32[768]", arg131_1: "f32[768]", arg132_1: "f32[768]", arg133_1: "f32[3072, 768]", arg134_1: "f32[3072]", arg135_1: "f32[768, 3072]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[768]", arg139_1: "f32[2304, 768]", arg140_1: "f32[2304]", arg141_1: "f32[768, 768]", arg142_1: "f32[768]", arg143_1: "f32[768]", arg144_1: "f32[768]", arg145_1: "f32[3072, 768]", arg146_1: "f32[3072]", arg147_1: "f32[768, 3072]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[1000, 768]", arg152_1: "f32[1000]", arg153_1: "f32[1000, 768]", arg154_1: "f32[1000]", getitem_310: "f32[1, 1, 768]", getitem_311: "f32[1, 198, 768]", getitem_312: "f32[1, 1, 768]", getitem_313: "f32[768, 3, 16, 16]", getitem_314: "f32[768]", getitem_315: "f32[768]", getitem_316: "f32[768]", getitem_317: "f32[2304, 768]", getitem_318: "f32[2304]", getitem_319: "f32[768, 768]", getitem_320: "f32[768]", getitem_321: "f32[768]", getitem_322: "f32[768]", getitem_323: "f32[3072, 768]", getitem_324: "f32[3072]", getitem_325: "f32[768, 3072]", getitem_326: "f32[768]", getitem_327: "f32[768]", getitem_328: "f32[768]", getitem_329: "f32[2304, 768]", getitem_330: "f32[2304]", getitem_331: "f32[768, 768]", getitem_332: "f32[768]", getitem_333: "f32[768]", getitem_334: "f32[768]", getitem_335: "f32[3072, 768]", getitem_336: "f32[3072]", getitem_337: "f32[768, 3072]", getitem_338: "f32[768]", getitem_339: "f32[768]", getitem_340: "f32[768]", getitem_341: "f32[2304, 768]", getitem_342: "f32[2304]", getitem_343: "f32[768, 768]", getitem_344: "f32[768]", getitem_345: "f32[768]", getitem_346: "f32[768]", getitem_347: "f32[3072, 768]", getitem_348: "f32[3072]", getitem_349: "f32[768, 3072]", getitem_350: "f32[768]", getitem_351: "f32[768]", getitem_352: "f32[768]", getitem_353: "f32[2304, 768]", getitem_354: "f32[2304]", getitem_355: "f32[768, 768]", getitem_356: "f32[768]", getitem_357: "f32[768]", getitem_358: "f32[768]", getitem_359: "f32[3072, 768]", getitem_360: "f32[3072]", getitem_361: "f32[768, 3072]", getitem_362: "f32[768]", getitem_363: "f32[768]", getitem_364: "f32[768]", getitem_365: "f32[2304, 768]", getitem_366: "f32[2304]", getitem_367: "f32[768, 768]", getitem_368: "f32[768]", getitem_369: "f32[768]", getitem_370: "f32[768]", getitem_371: "f32[3072, 768]", getitem_372: "f32[3072]", getitem_373: "f32[768, 3072]", getitem_374: "f32[768]", getitem_375: "f32[768]", getitem_376: "f32[768]", getitem_377: "f32[2304, 768]", getitem_378: "f32[2304]", getitem_379: "f32[768, 768]", getitem_380: "f32[768]", getitem_381: "f32[768]", getitem_382: "f32[768]", getitem_383: "f32[3072, 768]", getitem_384: "f32[3072]", getitem_385: "f32[768, 3072]", getitem_386: "f32[768]", getitem_387: "f32[768]", getitem_388: "f32[768]", getitem_389: "f32[2304, 768]", getitem_390: "f32[2304]", getitem_391: "f32[768, 768]", getitem_392: "f32[768]", getitem_393: "f32[768]", getitem_394: "f32[768]", getitem_395: "f32[3072, 768]", getitem_396: "f32[3072]", getitem_397: "f32[768, 3072]", getitem_398: "f32[768]", getitem_399: "f32[768]", getitem_400: "f32[768]", getitem_401: "f32[2304, 768]", getitem_402: "f32[2304]", getitem_403: "f32[768, 768]", getitem_404: "f32[768]", getitem_405: "f32[768]", getitem_406: "f32[768]", getitem_407: "f32[3072, 768]", getitem_408: "f32[3072]", getitem_409: "f32[768, 3072]", getitem_410: "f32[768]", getitem_411: "f32[768]", getitem_412: "f32[768]", getitem_413: "f32[2304, 768]", getitem_414: "f32[2304]", getitem_415: "f32[768, 768]", getitem_416: "f32[768]", getitem_417: "f32[768]", getitem_418: "f32[768]", getitem_419: "f32[3072, 768]", getitem_420: "f32[3072]", getitem_421: "f32[768, 3072]", getitem_422: "f32[768]", getitem_423: "f32[768]", getitem_424: "f32[768]", getitem_425: "f32[2304, 768]", getitem_426: "f32[2304]", getitem_427: "f32[768, 768]", getitem_428: "f32[768]", getitem_429: "f32[768]", getitem_430: "f32[768]", getitem_431: "f32[3072, 768]", getitem_432: "f32[3072]", getitem_433: "f32[768, 3072]", getitem_434: "f32[768]", getitem_435: "f32[768]", getitem_436: "f32[768]", getitem_437: "f32[2304, 768]", getitem_438: "f32[2304]", getitem_439: "f32[768, 768]", getitem_440: "f32[768]", getitem_441: "f32[768]", getitem_442: "f32[768]", getitem_443: "f32[3072, 768]", getitem_444: "f32[3072]", getitem_445: "f32[768, 3072]", getitem_446: "f32[768]", getitem_447: "f32[768]", getitem_448: "f32[768]", getitem_449: "f32[2304, 768]", getitem_450: "f32[2304]", getitem_451: "f32[768, 768]", getitem_452: "f32[768]", getitem_453: "f32[768]", getitem_454: "f32[768]", getitem_455: "f32[3072, 768]", getitem_456: "f32[3072]", getitem_457: "f32[768, 3072]", getitem_458: "f32[768]", getitem_459: "f32[768]", getitem_460: "f32[768]", getitem_461: "f32[1000, 768]", getitem_462: "f32[1000]", getitem_463: "f32[1000, 768]", getitem_464: "f32[1000]", getitem_2480: "f32[1, 1, 768]", getitem_2481: "f32[1, 198, 768]", getitem_2482: "f32[1, 1, 768]", getitem_2483: "f32[768, 3, 16, 16]", getitem_2484: "f32[768]", getitem_2485: "f32[768]", getitem_2486: "f32[768]", getitem_2487: "f32[2304, 768]", getitem_2488: "f32[2304]", getitem_2489: "f32[768, 768]", getitem_2490: "f32[768]", getitem_2491: "f32[768]", getitem_2492: "f32[768]", getitem_2493: "f32[3072, 768]", getitem_2494: "f32[3072]", getitem_2495: "f32[768, 3072]", getitem_2496: "f32[768]", getitem_2497: "f32[768]", getitem_2498: "f32[768]", getitem_2499: "f32[2304, 768]", getitem_2500: "f32[2304]", getitem_2501: "f32[768, 768]", getitem_2502: "f32[768]", getitem_2503: "f32[768]", getitem_2504: "f32[768]", getitem_2505: "f32[3072, 768]", getitem_2506: "f32[3072]", getitem_2507: "f32[768, 3072]", getitem_2508: "f32[768]", getitem_2509: "f32[768]", getitem_2510: "f32[768]", getitem_2511: "f32[2304, 768]", getitem_2512: "f32[2304]", getitem_2513: "f32[768, 768]", getitem_2514: "f32[768]", getitem_2515: "f32[768]", getitem_2516: "f32[768]", getitem_2517: "f32[3072, 768]", getitem_2518: "f32[3072]", getitem_2519: "f32[768, 3072]", getitem_2520: "f32[768]", getitem_2521: "f32[768]", getitem_2522: "f32[768]", getitem_2523: "f32[2304, 768]", getitem_2524: "f32[2304]", getitem_2525: "f32[768, 768]", getitem_2526: "f32[768]", getitem_2527: "f32[768]", getitem_2528: "f32[768]", getitem_2529: "f32[3072, 768]", getitem_2530: "f32[3072]", getitem_2531: "f32[768, 3072]", getitem_2532: "f32[768]", getitem_2533: "f32[768]", getitem_2534: "f32[768]", getitem_2535: "f32[2304, 768]", getitem_2536: "f32[2304]", getitem_2537: "f32[768, 768]", getitem_2538: "f32[768]", getitem_2539: "f32[768]", getitem_2540: "f32[768]", getitem_2541: "f32[3072, 768]", getitem_2542: "f32[3072]", getitem_2543: "f32[768, 3072]", getitem_2544: "f32[768]", getitem_2545: "f32[768]", getitem_2546: "f32[768]", getitem_2547: "f32[2304, 768]", getitem_2548: "f32[2304]", getitem_2549: "f32[768, 768]", getitem_2550: "f32[768]", getitem_2551: "f32[768]", getitem_2552: "f32[768]", getitem_2553: "f32[3072, 768]", getitem_2554: "f32[3072]", getitem_2555: "f32[768, 3072]", getitem_2556: "f32[768]", getitem_2557: "f32[768]", getitem_2558: "f32[768]", getitem_2559: "f32[2304, 768]", getitem_2560: "f32[2304]", getitem_2561: "f32[768, 768]", getitem_2562: "f32[768]", getitem_2563: "f32[768]", getitem_2564: "f32[768]", getitem_2565: "f32[3072, 768]", getitem_2566: "f32[3072]", getitem_2567: "f32[768, 3072]", getitem_2568: "f32[768]", getitem_2569: "f32[768]", getitem_2570: "f32[768]", getitem_2571: "f32[2304, 768]", getitem_2572: "f32[2304]", getitem_2573: "f32[768, 768]", getitem_2574: "f32[768]", getitem_2575: "f32[768]", getitem_2576: "f32[768]", getitem_2577: "f32[3072, 768]", getitem_2578: "f32[3072]", getitem_2579: "f32[768, 3072]", getitem_2580: "f32[768]", getitem_2581: "f32[768]", getitem_2582: "f32[768]", getitem_2583: "f32[2304, 768]", getitem_2584: "f32[2304]", getitem_2585: "f32[768, 768]", getitem_2586: "f32[768]", getitem_2587: "f32[768]", getitem_2588: "f32[768]", getitem_2589: "f32[3072, 768]", getitem_2590: "f32[3072]", getitem_2591: "f32[768, 3072]", getitem_2592: "f32[768]", getitem_2593: "f32[768]", getitem_2594: "f32[768]", getitem_2595: "f32[2304, 768]", getitem_2596: "f32[2304]", getitem_2597: "f32[768, 768]", getitem_2598: "f32[768]", getitem_2599: "f32[768]", getitem_2600: "f32[768]", getitem_2601: "f32[3072, 768]", getitem_2602: "f32[3072]", getitem_2603: "f32[768, 3072]", getitem_2604: "f32[768]", getitem_2605: "f32[768]", getitem_2606: "f32[768]", getitem_2607: "f32[2304, 768]", getitem_2608: "f32[2304]", getitem_2609: "f32[768, 768]", getitem_2610: "f32[768]", getitem_2611: "f32[768]", getitem_2612: "f32[768]", getitem_2613: "f32[3072, 768]", getitem_2614: "f32[3072]", getitem_2615: "f32[768, 3072]", getitem_2616: "f32[768]", getitem_2617: "f32[768]", getitem_2618: "f32[768]", getitem_2619: "f32[2304, 768]", getitem_2620: "f32[2304]", getitem_2621: "f32[768, 768]", getitem_2622: "f32[768]", getitem_2623: "f32[768]", getitem_2624: "f32[768]", getitem_2625: "f32[3072, 768]", getitem_2626: "f32[3072]", getitem_2627: "f32[768, 3072]", getitem_2628: "f32[768]", getitem_2629: "f32[768]", getitem_2630: "f32[768]", getitem_2631: "f32[1000, 768]", getitem_2632: "f32[1000]", getitem_2633: "f32[1000, 768]", getitem_2634: "f32[1000]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1], [getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464], [getitem_2480, getitem_2481, getitem_2482, getitem_2483, getitem_2484, getitem_2485, getitem_2486, getitem_2487, getitem_2488, getitem_2489, getitem_2490, getitem_2491, getitem_2492, getitem_2493, getitem_2494, getitem_2495, getitem_2496, getitem_2497, getitem_2498, getitem_2499, getitem_2500, getitem_2501, getitem_2502, getitem_2503, getitem_2504, getitem_2505, getitem_2506, getitem_2507, getitem_2508, getitem_2509, getitem_2510, getitem_2511, getitem_2512, getitem_2513, getitem_2514, getitem_2515, getitem_2516, getitem_2517, getitem_2518, getitem_2519, getitem_2520, getitem_2521, getitem_2522, getitem_2523, getitem_2524, getitem_2525, getitem_2526, getitem_2527, getitem_2528, getitem_2529, getitem_2530, getitem_2531, getitem_2532, getitem_2533, getitem_2534, getitem_2535, getitem_2536, getitem_2537, getitem_2538, getitem_2539, getitem_2540, getitem_2541, getitem_2542, getitem_2543, getitem_2544, getitem_2545, getitem_2546, getitem_2547, getitem_2548, getitem_2549, getitem_2550, getitem_2551, getitem_2552, getitem_2553, getitem_2554, getitem_2555, getitem_2556, getitem_2557, getitem_2558, getitem_2559, getitem_2560, getitem_2561, getitem_2562, getitem_2563, getitem_2564, getitem_2565, getitem_2566, getitem_2567, getitem_2568, getitem_2569, getitem_2570, getitem_2571, getitem_2572, getitem_2573, getitem_2574, getitem_2575, getitem_2576, getitem_2577, getitem_2578, getitem_2579, getitem_2580, getitem_2581, getitem_2582, getitem_2583, getitem_2584, getitem_2585, getitem_2586, getitem_2587, getitem_2588, getitem_2589, getitem_2590, getitem_2591, getitem_2592, getitem_2593, getitem_2594, getitem_2595, getitem_2596, getitem_2597, getitem_2598, getitem_2599, getitem_2600, getitem_2601, getitem_2602, getitem_2603, getitem_2604, getitem_2605, getitem_2606, getitem_2607, getitem_2608, getitem_2609, getitem_2610, getitem_2611, getitem_2612, getitem_2613, getitem_2614, getitem_2615, getitem_2616, getitem_2617, getitem_2618, getitem_2619, getitem_2620, getitem_2621, getitem_2622, getitem_2623, getitem_2624, getitem_2625, getitem_2626, getitem_2627, getitem_2628, getitem_2629, getitem_2630, getitem_2631, getitem_2632, getitem_2633, getitem_2634]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = arg13_1 = arg14_1 = arg15_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = arg32_1 = arg33_1 = arg34_1 = arg35_1 = arg36_1 = arg37_1 = arg38_1 = arg39_1 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = arg50_1 = arg51_1 = arg52_1 = arg53_1 = arg54_1 = arg55_1 = arg56_1 = arg57_1 = arg58_1 = arg59_1 = arg60_1 = arg61_1 = arg62_1 = arg63_1 = arg64_1 = arg65_1 = arg66_1 = arg67_1 = arg68_1 = arg69_1 = arg70_1 = arg71_1 = arg72_1 = arg73_1 = arg74_1 = arg75_1 = arg76_1 = arg77_1 = arg78_1 = arg79_1 = arg80_1 = arg81_1 = arg82_1 = arg83_1 = arg84_1 = arg85_1 = arg86_1 = arg87_1 = arg88_1 = arg89_1 = arg90_1 = arg91_1 = arg92_1 = arg93_1 = arg94_1 = arg95_1 = arg96_1 = arg97_1 = arg98_1 = arg99_1 = arg100_1 = arg101_1 = arg102_1 = arg103_1 = arg104_1 = arg105_1 = arg106_1 = arg107_1 = arg108_1 = arg109_1 = arg110_1 = arg111_1 = arg112_1 = arg113_1 = arg114_1 = arg115_1 = arg116_1 = arg117_1 = arg118_1 = arg119_1 = arg120_1 = arg121_1 = arg122_1 = arg123_1 = arg124_1 = arg125_1 = arg126_1 = arg127_1 = arg128_1 = arg129_1 = arg130_1 = arg131_1 = arg132_1 = arg133_1 = arg134_1 = arg135_1 = arg136_1 = arg137_1 = arg138_1 = arg139_1 = arg140_1 = arg141_1 = arg142_1 = arg143_1 = arg144_1 = arg145_1 = arg146_1 = arg147_1 = arg148_1 = arg149_1 = arg150_1 = arg151_1 = arg152_1 = arg153_1 = arg154_1 = getitem_310 = getitem_311 = getitem_312 = getitem_313 = getitem_314 = getitem_315 = getitem_316 = getitem_317 = getitem_318 = getitem_319 = getitem_320 = getitem_321 = getitem_322 = getitem_323 = getitem_324 = getitem_325 = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = getitem_331 = getitem_332 = getitem_333 = getitem_334 = getitem_335 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = getitem_367 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = getitem_373 = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = getitem_409 = getitem_410 = getitem_411 = getitem_412 = getitem_413 = getitem_414 = getitem_415 = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = getitem_433 = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = getitem_439 = getitem_440 = getitem_441 = getitem_442 = getitem_443 = getitem_444 = getitem_445 = getitem_446 = getitem_447 = getitem_448 = getitem_449 = getitem_450 = getitem_451 = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = getitem_457 = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_2480 = getitem_2481 = getitem_2482 = getitem_2483 = getitem_2484 = getitem_2485 = getitem_2486 = getitem_2487 = getitem_2488 = getitem_2489 = getitem_2490 = getitem_2491 = getitem_2492 = getitem_2493 = getitem_2494 = getitem_2495 = getitem_2496 = getitem_2497 = getitem_2498 = getitem_2499 = getitem_2500 = getitem_2501 = getitem_2502 = getitem_2503 = getitem_2504 = getitem_2505 = getitem_2506 = getitem_2507 = getitem_2508 = getitem_2509 = getitem_2510 = getitem_2511 = getitem_2512 = getitem_2513 = getitem_2514 = getitem_2515 = getitem_2516 = getitem_2517 = getitem_2518 = getitem_2519 = getitem_2520 = getitem_2521 = getitem_2522 = getitem_2523 = getitem_2524 = getitem_2525 = getitem_2526 = getitem_2527 = getitem_2528 = getitem_2529 = getitem_2530 = getitem_2531 = getitem_2532 = getitem_2533 = getitem_2534 = getitem_2535 = getitem_2536 = getitem_2537 = getitem_2538 = getitem_2539 = getitem_2540 = getitem_2541 = getitem_2542 = getitem_2543 = getitem_2544 = getitem_2545 = getitem_2546 = getitem_2547 = getitem_2548 = getitem_2549 = getitem_2550 = getitem_2551 = getitem_2552 = getitem_2553 = getitem_2554 = getitem_2555 = getitem_2556 = getitem_2557 = getitem_2558 = getitem_2559 = getitem_2560 = getitem_2561 = getitem_2562 = getitem_2563 = getitem_2564 = getitem_2565 = getitem_2566 = getitem_2567 = getitem_2568 = getitem_2569 = getitem_2570 = getitem_2571 = getitem_2572 = getitem_2573 = getitem_2574 = getitem_2575 = getitem_2576 = getitem_2577 = getitem_2578 = getitem_2579 = getitem_2580 = getitem_2581 = getitem_2582 = getitem_2583 = getitem_2584 = getitem_2585 = getitem_2586 = getitem_2587 = getitem_2588 = getitem_2589 = getitem_2590 = getitem_2591 = getitem_2592 = getitem_2593 = getitem_2594 = getitem_2595 = getitem_2596 = getitem_2597 = getitem_2598 = getitem_2599 = getitem_2600 = getitem_2601 = getitem_2602 = getitem_2603 = getitem_2604 = getitem_2605 = getitem_2606 = getitem_2607 = getitem_2608 = getitem_2609 = getitem_2610 = getitem_2611 = getitem_2612 = getitem_2613 = getitem_2614 = getitem_2615 = getitem_2616 = getitem_2617 = getitem_2618 = getitem_2619 = getitem_2620 = getitem_2621 = getitem_2622 = getitem_2623 = getitem_2624 = getitem_2625 = getitem_2626 = getitem_2627 = getitem_2628 = getitem_2629 = getitem_2630 = getitem_2631 = getitem_2632 = getitem_2633 = getitem_2634 = None
        getitem: "f32[1, 1, 768]" = _foreach_addcdiv_scalar[0]
        getitem_2635: "f32[1, 198, 768]" = _foreach_addcdiv_scalar[1]
        getitem_2636: "f32[1, 1, 768]" = _foreach_addcdiv_scalar[2]
        getitem_2637: "f32[768, 3, 16, 16]" = _foreach_addcdiv_scalar[3]
        getitem_2638: "f32[768]" = _foreach_addcdiv_scalar[4]
        getitem_2639: "f32[768]" = _foreach_addcdiv_scalar[5]
        getitem_2640: "f32[768]" = _foreach_addcdiv_scalar[6]
        getitem_2641: "f32[2304, 768]" = _foreach_addcdiv_scalar[7]
        getitem_2642: "f32[2304]" = _foreach_addcdiv_scalar[8]
        getitem_2643: "f32[768, 768]" = _foreach_addcdiv_scalar[9]
        getitem_2644: "f32[768]" = _foreach_addcdiv_scalar[10]
        getitem_2645: "f32[768]" = _foreach_addcdiv_scalar[11]
        getitem_2646: "f32[768]" = _foreach_addcdiv_scalar[12]
        getitem_2647: "f32[3072, 768]" = _foreach_addcdiv_scalar[13]
        getitem_2648: "f32[3072]" = _foreach_addcdiv_scalar[14]
        getitem_2649: "f32[768, 3072]" = _foreach_addcdiv_scalar[15]
        getitem_2650: "f32[768]" = _foreach_addcdiv_scalar[16]
        getitem_2651: "f32[768]" = _foreach_addcdiv_scalar[17]
        getitem_2652: "f32[768]" = _foreach_addcdiv_scalar[18]
        getitem_2653: "f32[2304, 768]" = _foreach_addcdiv_scalar[19]
        getitem_2654: "f32[2304]" = _foreach_addcdiv_scalar[20]
        getitem_2655: "f32[768, 768]" = _foreach_addcdiv_scalar[21]
        getitem_2656: "f32[768]" = _foreach_addcdiv_scalar[22]
        getitem_2657: "f32[768]" = _foreach_addcdiv_scalar[23]
        getitem_2658: "f32[768]" = _foreach_addcdiv_scalar[24]
        getitem_2659: "f32[3072, 768]" = _foreach_addcdiv_scalar[25]
        getitem_2660: "f32[3072]" = _foreach_addcdiv_scalar[26]
        getitem_2661: "f32[768, 3072]" = _foreach_addcdiv_scalar[27]
        getitem_2662: "f32[768]" = _foreach_addcdiv_scalar[28]
        getitem_2663: "f32[768]" = _foreach_addcdiv_scalar[29]
        getitem_2664: "f32[768]" = _foreach_addcdiv_scalar[30]
        getitem_2665: "f32[2304, 768]" = _foreach_addcdiv_scalar[31]
        getitem_2666: "f32[2304]" = _foreach_addcdiv_scalar[32]
        getitem_2667: "f32[768, 768]" = _foreach_addcdiv_scalar[33]
        getitem_2668: "f32[768]" = _foreach_addcdiv_scalar[34]
        getitem_2669: "f32[768]" = _foreach_addcdiv_scalar[35]
        getitem_2670: "f32[768]" = _foreach_addcdiv_scalar[36]
        getitem_2671: "f32[3072, 768]" = _foreach_addcdiv_scalar[37]
        getitem_2672: "f32[3072]" = _foreach_addcdiv_scalar[38]
        getitem_2673: "f32[768, 3072]" = _foreach_addcdiv_scalar[39]
        getitem_2674: "f32[768]" = _foreach_addcdiv_scalar[40]
        getitem_2675: "f32[768]" = _foreach_addcdiv_scalar[41]
        getitem_2676: "f32[768]" = _foreach_addcdiv_scalar[42]
        getitem_2677: "f32[2304, 768]" = _foreach_addcdiv_scalar[43]
        getitem_2678: "f32[2304]" = _foreach_addcdiv_scalar[44]
        getitem_2679: "f32[768, 768]" = _foreach_addcdiv_scalar[45]
        getitem_2680: "f32[768]" = _foreach_addcdiv_scalar[46]
        getitem_2681: "f32[768]" = _foreach_addcdiv_scalar[47]
        getitem_2682: "f32[768]" = _foreach_addcdiv_scalar[48]
        getitem_2683: "f32[3072, 768]" = _foreach_addcdiv_scalar[49]
        getitem_2684: "f32[3072]" = _foreach_addcdiv_scalar[50]
        getitem_2685: "f32[768, 3072]" = _foreach_addcdiv_scalar[51]
        getitem_2686: "f32[768]" = _foreach_addcdiv_scalar[52]
        getitem_2687: "f32[768]" = _foreach_addcdiv_scalar[53]
        getitem_2688: "f32[768]" = _foreach_addcdiv_scalar[54]
        getitem_2689: "f32[2304, 768]" = _foreach_addcdiv_scalar[55]
        getitem_2690: "f32[2304]" = _foreach_addcdiv_scalar[56]
        getitem_2691: "f32[768, 768]" = _foreach_addcdiv_scalar[57]
        getitem_2692: "f32[768]" = _foreach_addcdiv_scalar[58]
        getitem_2693: "f32[768]" = _foreach_addcdiv_scalar[59]
        getitem_2694: "f32[768]" = _foreach_addcdiv_scalar[60]
        getitem_2695: "f32[3072, 768]" = _foreach_addcdiv_scalar[61]
        getitem_2696: "f32[3072]" = _foreach_addcdiv_scalar[62]
        getitem_2697: "f32[768, 3072]" = _foreach_addcdiv_scalar[63]
        getitem_2698: "f32[768]" = _foreach_addcdiv_scalar[64]
        getitem_2699: "f32[768]" = _foreach_addcdiv_scalar[65]
        getitem_2700: "f32[768]" = _foreach_addcdiv_scalar[66]
        getitem_2701: "f32[2304, 768]" = _foreach_addcdiv_scalar[67]
        getitem_2702: "f32[2304]" = _foreach_addcdiv_scalar[68]
        getitem_2703: "f32[768, 768]" = _foreach_addcdiv_scalar[69]
        getitem_2704: "f32[768]" = _foreach_addcdiv_scalar[70]
        getitem_2705: "f32[768]" = _foreach_addcdiv_scalar[71]
        getitem_2706: "f32[768]" = _foreach_addcdiv_scalar[72]
        getitem_2707: "f32[3072, 768]" = _foreach_addcdiv_scalar[73]
        getitem_2708: "f32[3072]" = _foreach_addcdiv_scalar[74]
        getitem_2709: "f32[768, 3072]" = _foreach_addcdiv_scalar[75]
        getitem_2710: "f32[768]" = _foreach_addcdiv_scalar[76]
        getitem_2711: "f32[768]" = _foreach_addcdiv_scalar[77]
        getitem_2712: "f32[768]" = _foreach_addcdiv_scalar[78]
        getitem_2713: "f32[2304, 768]" = _foreach_addcdiv_scalar[79]
        getitem_2714: "f32[2304]" = _foreach_addcdiv_scalar[80]
        getitem_2715: "f32[768, 768]" = _foreach_addcdiv_scalar[81]
        getitem_2716: "f32[768]" = _foreach_addcdiv_scalar[82]
        getitem_2717: "f32[768]" = _foreach_addcdiv_scalar[83]
        getitem_2718: "f32[768]" = _foreach_addcdiv_scalar[84]
        getitem_2719: "f32[3072, 768]" = _foreach_addcdiv_scalar[85]
        getitem_2720: "f32[3072]" = _foreach_addcdiv_scalar[86]
        getitem_2721: "f32[768, 3072]" = _foreach_addcdiv_scalar[87]
        getitem_2722: "f32[768]" = _foreach_addcdiv_scalar[88]
        getitem_2723: "f32[768]" = _foreach_addcdiv_scalar[89]
        getitem_2724: "f32[768]" = _foreach_addcdiv_scalar[90]
        getitem_2725: "f32[2304, 768]" = _foreach_addcdiv_scalar[91]
        getitem_2726: "f32[2304]" = _foreach_addcdiv_scalar[92]
        getitem_2727: "f32[768, 768]" = _foreach_addcdiv_scalar[93]
        getitem_2728: "f32[768]" = _foreach_addcdiv_scalar[94]
        getitem_2729: "f32[768]" = _foreach_addcdiv_scalar[95]
        getitem_2730: "f32[768]" = _foreach_addcdiv_scalar[96]
        getitem_2731: "f32[3072, 768]" = _foreach_addcdiv_scalar[97]
        getitem_2732: "f32[3072]" = _foreach_addcdiv_scalar[98]
        getitem_2733: "f32[768, 3072]" = _foreach_addcdiv_scalar[99]
        getitem_2734: "f32[768]" = _foreach_addcdiv_scalar[100]
        getitem_2735: "f32[768]" = _foreach_addcdiv_scalar[101]
        getitem_2736: "f32[768]" = _foreach_addcdiv_scalar[102]
        getitem_2737: "f32[2304, 768]" = _foreach_addcdiv_scalar[103]
        getitem_2738: "f32[2304]" = _foreach_addcdiv_scalar[104]
        getitem_2739: "f32[768, 768]" = _foreach_addcdiv_scalar[105]
        getitem_2740: "f32[768]" = _foreach_addcdiv_scalar[106]
        getitem_2741: "f32[768]" = _foreach_addcdiv_scalar[107]
        getitem_2742: "f32[768]" = _foreach_addcdiv_scalar[108]
        getitem_2743: "f32[3072, 768]" = _foreach_addcdiv_scalar[109]
        getitem_2744: "f32[3072]" = _foreach_addcdiv_scalar[110]
        getitem_2745: "f32[768, 3072]" = _foreach_addcdiv_scalar[111]
        getitem_2746: "f32[768]" = _foreach_addcdiv_scalar[112]
        getitem_2747: "f32[768]" = _foreach_addcdiv_scalar[113]
        getitem_2748: "f32[768]" = _foreach_addcdiv_scalar[114]
        getitem_2749: "f32[2304, 768]" = _foreach_addcdiv_scalar[115]
        getitem_2750: "f32[2304]" = _foreach_addcdiv_scalar[116]
        getitem_2751: "f32[768, 768]" = _foreach_addcdiv_scalar[117]
        getitem_2752: "f32[768]" = _foreach_addcdiv_scalar[118]
        getitem_2753: "f32[768]" = _foreach_addcdiv_scalar[119]
        getitem_2754: "f32[768]" = _foreach_addcdiv_scalar[120]
        getitem_2755: "f32[3072, 768]" = _foreach_addcdiv_scalar[121]
        getitem_2756: "f32[3072]" = _foreach_addcdiv_scalar[122]
        getitem_2757: "f32[768, 3072]" = _foreach_addcdiv_scalar[123]
        getitem_2758: "f32[768]" = _foreach_addcdiv_scalar[124]
        getitem_2759: "f32[768]" = _foreach_addcdiv_scalar[125]
        getitem_2760: "f32[768]" = _foreach_addcdiv_scalar[126]
        getitem_2761: "f32[2304, 768]" = _foreach_addcdiv_scalar[127]
        getitem_2762: "f32[2304]" = _foreach_addcdiv_scalar[128]
        getitem_2763: "f32[768, 768]" = _foreach_addcdiv_scalar[129]
        getitem_2764: "f32[768]" = _foreach_addcdiv_scalar[130]
        getitem_2765: "f32[768]" = _foreach_addcdiv_scalar[131]
        getitem_2766: "f32[768]" = _foreach_addcdiv_scalar[132]
        getitem_2767: "f32[3072, 768]" = _foreach_addcdiv_scalar[133]
        getitem_2768: "f32[3072]" = _foreach_addcdiv_scalar[134]
        getitem_2769: "f32[768, 3072]" = _foreach_addcdiv_scalar[135]
        getitem_2770: "f32[768]" = _foreach_addcdiv_scalar[136]
        getitem_2771: "f32[768]" = _foreach_addcdiv_scalar[137]
        getitem_2772: "f32[768]" = _foreach_addcdiv_scalar[138]
        getitem_2773: "f32[2304, 768]" = _foreach_addcdiv_scalar[139]
        getitem_2774: "f32[2304]" = _foreach_addcdiv_scalar[140]
        getitem_2775: "f32[768, 768]" = _foreach_addcdiv_scalar[141]
        getitem_2776: "f32[768]" = _foreach_addcdiv_scalar[142]
        getitem_2777: "f32[768]" = _foreach_addcdiv_scalar[143]
        getitem_2778: "f32[768]" = _foreach_addcdiv_scalar[144]
        getitem_2779: "f32[3072, 768]" = _foreach_addcdiv_scalar[145]
        getitem_2780: "f32[3072]" = _foreach_addcdiv_scalar[146]
        getitem_2781: "f32[768, 3072]" = _foreach_addcdiv_scalar[147]
        getitem_2782: "f32[768]" = _foreach_addcdiv_scalar[148]
        getitem_2783: "f32[768]" = _foreach_addcdiv_scalar[149]
        getitem_2784: "f32[768]" = _foreach_addcdiv_scalar[150]
        getitem_2785: "f32[1000, 768]" = _foreach_addcdiv_scalar[151]
        getitem_2786: "f32[1000]" = _foreach_addcdiv_scalar[152]
        getitem_2787: "f32[1000, 768]" = _foreach_addcdiv_scalar[153]
        getitem_2788: "f32[1000]" = _foreach_addcdiv_scalar[154];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_2635, getitem_2636, getitem_2637, getitem_2638, getitem_2639, getitem_2640, getitem_2641, getitem_2642, getitem_2643, getitem_2644, getitem_2645, getitem_2646, getitem_2647, getitem_2648, getitem_2649, getitem_2650, getitem_2651, getitem_2652, getitem_2653, getitem_2654, getitem_2655, getitem_2656, getitem_2657, getitem_2658, getitem_2659, getitem_2660, getitem_2661, getitem_2662, getitem_2663, getitem_2664, getitem_2665, getitem_2666, getitem_2667, getitem_2668, getitem_2669, getitem_2670, getitem_2671, getitem_2672, getitem_2673, getitem_2674, getitem_2675, getitem_2676, getitem_2677, getitem_2678, getitem_2679, getitem_2680, getitem_2681, getitem_2682, getitem_2683, getitem_2684, getitem_2685, getitem_2686, getitem_2687, getitem_2688, getitem_2689, getitem_2690, getitem_2691, getitem_2692, getitem_2693, getitem_2694, getitem_2695, getitem_2696, getitem_2697, getitem_2698, getitem_2699, getitem_2700, getitem_2701, getitem_2702, getitem_2703, getitem_2704, getitem_2705, getitem_2706, getitem_2707, getitem_2708, getitem_2709, getitem_2710, getitem_2711, getitem_2712, getitem_2713, getitem_2714, getitem_2715, getitem_2716, getitem_2717, getitem_2718, getitem_2719, getitem_2720, getitem_2721, getitem_2722, getitem_2723, getitem_2724, getitem_2725, getitem_2726, getitem_2727, getitem_2728, getitem_2729, getitem_2730, getitem_2731, getitem_2732, getitem_2733, getitem_2734, getitem_2735, getitem_2736, getitem_2737, getitem_2738, getitem_2739, getitem_2740, getitem_2741, getitem_2742, getitem_2743, getitem_2744, getitem_2745, getitem_2746, getitem_2747, getitem_2748, getitem_2749, getitem_2750, getitem_2751, getitem_2752, getitem_2753, getitem_2754, getitem_2755, getitem_2756, getitem_2757, getitem_2758, getitem_2759, getitem_2760, getitem_2761, getitem_2762, getitem_2763, getitem_2764, getitem_2765, getitem_2766, getitem_2767, getitem_2768, getitem_2769, getitem_2770, getitem_2771, getitem_2772, getitem_2773, getitem_2774, getitem_2775, getitem_2776, getitem_2777, getitem_2778, getitem_2779, getitem_2780, getitem_2781, getitem_2782, getitem_2783, getitem_2784, getitem_2785, getitem_2786, getitem_2787, getitem_2788)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

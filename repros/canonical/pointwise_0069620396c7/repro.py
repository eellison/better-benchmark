"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g77
Pattern hash: 0069620396c7
Shape hash: 8b13b173
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
    def forward(self, arg0_1: "f32[64, 3, 3, 3]", arg1_1: "f32[64]", arg2_1: "f32[64]", arg3_1: "f32[64, 3, 1, 1]", arg4_1: "f32[64]", arg5_1: "f32[64]", arg6_1: "f32[96, 64, 3, 3]", arg7_1: "f32[96]", arg8_1: "f32[96]", arg9_1: "f32[96, 64, 1, 1]", arg10_1: "f32[96]", arg11_1: "f32[96]", arg12_1: "f32[96]", arg13_1: "f32[96]", arg14_1: "f32[96, 96, 3, 3]", arg15_1: "f32[96]", arg16_1: "f32[96]", arg17_1: "f32[96, 96, 1, 1]", arg18_1: "f32[96]", arg19_1: "f32[96]", arg20_1: "f32[192, 96, 3, 3]", arg21_1: "f32[192]", arg22_1: "f32[192]", arg23_1: "f32[192, 96, 1, 1]", arg24_1: "f32[192]", arg25_1: "f32[192]", arg26_1: "f32[192]", arg27_1: "f32[192]", arg28_1: "f32[192, 192, 3, 3]", arg29_1: "f32[192]", arg30_1: "f32[192]", arg31_1: "f32[192, 192, 1, 1]", arg32_1: "f32[192]", arg33_1: "f32[192]", arg34_1: "f32[192]", arg35_1: "f32[192]", arg36_1: "f32[192, 192, 3, 3]", arg37_1: "f32[192]", arg38_1: "f32[192]", arg39_1: "f32[192, 192, 1, 1]", arg40_1: "f32[192]", arg41_1: "f32[192]", arg42_1: "f32[192]", arg43_1: "f32[192]", arg44_1: "f32[192, 192, 3, 3]", arg45_1: "f32[192]", arg46_1: "f32[192]", arg47_1: "f32[192, 192, 1, 1]", arg48_1: "f32[192]", arg49_1: "f32[192]", arg50_1: "f32[384, 192, 3, 3]", arg51_1: "f32[384]", arg52_1: "f32[384]", arg53_1: "f32[384, 192, 1, 1]", arg54_1: "f32[384]", arg55_1: "f32[384]", arg56_1: "f32[384]", arg57_1: "f32[384]", arg58_1: "f32[384, 384, 3, 3]", arg59_1: "f32[384]", arg60_1: "f32[384]", arg61_1: "f32[384, 384, 1, 1]", arg62_1: "f32[384]", arg63_1: "f32[384]", arg64_1: "f32[384]", arg65_1: "f32[384]", arg66_1: "f32[384, 384, 3, 3]", arg67_1: "f32[384]", arg68_1: "f32[384]", arg69_1: "f32[384, 384, 1, 1]", arg70_1: "f32[384]", arg71_1: "f32[384]", arg72_1: "f32[384]", arg73_1: "f32[384]", arg74_1: "f32[384, 384, 3, 3]", arg75_1: "f32[384]", arg76_1: "f32[384]", arg77_1: "f32[384, 384, 1, 1]", arg78_1: "f32[384]", arg79_1: "f32[384]", arg80_1: "f32[384]", arg81_1: "f32[384]", arg82_1: "f32[384, 384, 3, 3]", arg83_1: "f32[384]", arg84_1: "f32[384]", arg85_1: "f32[384, 384, 1, 1]", arg86_1: "f32[384]", arg87_1: "f32[384]", arg88_1: "f32[384]", arg89_1: "f32[384]", arg90_1: "f32[384, 384, 3, 3]", arg91_1: "f32[384]", arg92_1: "f32[384]", arg93_1: "f32[384, 384, 1, 1]", arg94_1: "f32[384]", arg95_1: "f32[384]", arg96_1: "f32[384]", arg97_1: "f32[384]", arg98_1: "f32[384, 384, 3, 3]", arg99_1: "f32[384]", arg100_1: "f32[384]", arg101_1: "f32[384, 384, 1, 1]", arg102_1: "f32[384]", arg103_1: "f32[384]", arg104_1: "f32[384]", arg105_1: "f32[384]", arg106_1: "f32[384, 384, 3, 3]", arg107_1: "f32[384]", arg108_1: "f32[384]", arg109_1: "f32[384, 384, 1, 1]", arg110_1: "f32[384]", arg111_1: "f32[384]", arg112_1: "f32[384]", arg113_1: "f32[384]", arg114_1: "f32[384, 384, 3, 3]", arg115_1: "f32[384]", arg116_1: "f32[384]", arg117_1: "f32[384, 384, 1, 1]", arg118_1: "f32[384]", arg119_1: "f32[384]", arg120_1: "f32[384]", arg121_1: "f32[384]", arg122_1: "f32[384, 384, 3, 3]", arg123_1: "f32[384]", arg124_1: "f32[384]", arg125_1: "f32[384, 384, 1, 1]", arg126_1: "f32[384]", arg127_1: "f32[384]", arg128_1: "f32[384]", arg129_1: "f32[384]", arg130_1: "f32[384, 384, 3, 3]", arg131_1: "f32[384]", arg132_1: "f32[384]", arg133_1: "f32[384, 384, 1, 1]", arg134_1: "f32[384]", arg135_1: "f32[384]", arg136_1: "f32[384]", arg137_1: "f32[384]", arg138_1: "f32[384, 384, 3, 3]", arg139_1: "f32[384]", arg140_1: "f32[384]", arg141_1: "f32[384, 384, 1, 1]", arg142_1: "f32[384]", arg143_1: "f32[384]", arg144_1: "f32[384]", arg145_1: "f32[384]", arg146_1: "f32[384, 384, 3, 3]", arg147_1: "f32[384]", arg148_1: "f32[384]", arg149_1: "f32[384, 384, 1, 1]", arg150_1: "f32[384]", arg151_1: "f32[384]", arg152_1: "f32[384]", arg153_1: "f32[384]", arg154_1: "f32[384, 384, 3, 3]", arg155_1: "f32[384]", arg156_1: "f32[384]", arg157_1: "f32[384, 384, 1, 1]", arg158_1: "f32[384]", arg159_1: "f32[384]", arg160_1: "f32[1408, 384, 3, 3]", arg161_1: "f32[1408]", arg162_1: "f32[1408]", arg163_1: "f32[1408, 384, 1, 1]", arg164_1: "f32[1408]", arg165_1: "f32[1408]", arg166_1: "f32[1000, 1408]", arg167_1: "f32[1000]", getitem_336: "f32[64, 3, 3, 3]", getitem_337: "f32[64]", getitem_338: "f32[64]", getitem_339: "f32[64, 3, 1, 1]", getitem_340: "f32[64]", getitem_341: "f32[64]", getitem_342: "f32[96, 64, 3, 3]", getitem_343: "f32[96]", getitem_344: "f32[96]", getitem_345: "f32[96, 64, 1, 1]", getitem_346: "f32[96]", getitem_347: "f32[96]", getitem_348: "f32[96]", getitem_349: "f32[96]", getitem_350: "f32[96, 96, 3, 3]", getitem_351: "f32[96]", getitem_352: "f32[96]", getitem_353: "f32[96, 96, 1, 1]", getitem_354: "f32[96]", getitem_355: "f32[96]", getitem_356: "f32[192, 96, 3, 3]", getitem_357: "f32[192]", getitem_358: "f32[192]", getitem_359: "f32[192, 96, 1, 1]", getitem_360: "f32[192]", getitem_361: "f32[192]", getitem_362: "f32[192]", getitem_363: "f32[192]", getitem_364: "f32[192, 192, 3, 3]", getitem_365: "f32[192]", getitem_366: "f32[192]", getitem_367: "f32[192, 192, 1, 1]", getitem_368: "f32[192]", getitem_369: "f32[192]", getitem_370: "f32[192]", getitem_371: "f32[192]", getitem_372: "f32[192, 192, 3, 3]", getitem_373: "f32[192]", getitem_374: "f32[192]", getitem_375: "f32[192, 192, 1, 1]", getitem_376: "f32[192]", getitem_377: "f32[192]", getitem_378: "f32[192]", getitem_379: "f32[192]", getitem_380: "f32[192, 192, 3, 3]", getitem_381: "f32[192]", getitem_382: "f32[192]", getitem_383: "f32[192, 192, 1, 1]", getitem_384: "f32[192]", getitem_385: "f32[192]", getitem_386: "f32[384, 192, 3, 3]", getitem_387: "f32[384]", getitem_388: "f32[384]", getitem_389: "f32[384, 192, 1, 1]", getitem_390: "f32[384]", getitem_391: "f32[384]", getitem_392: "f32[384]", getitem_393: "f32[384]", getitem_394: "f32[384, 384, 3, 3]", getitem_395: "f32[384]", getitem_396: "f32[384]", getitem_397: "f32[384, 384, 1, 1]", getitem_398: "f32[384]", getitem_399: "f32[384]", getitem_400: "f32[384]", getitem_401: "f32[384]", getitem_402: "f32[384, 384, 3, 3]", getitem_403: "f32[384]", getitem_404: "f32[384]", getitem_405: "f32[384, 384, 1, 1]", getitem_406: "f32[384]", getitem_407: "f32[384]", getitem_408: "f32[384]", getitem_409: "f32[384]", getitem_410: "f32[384, 384, 3, 3]", getitem_411: "f32[384]", getitem_412: "f32[384]", getitem_413: "f32[384, 384, 1, 1]", getitem_414: "f32[384]", getitem_415: "f32[384]", getitem_416: "f32[384]", getitem_417: "f32[384]", getitem_418: "f32[384, 384, 3, 3]", getitem_419: "f32[384]", getitem_420: "f32[384]", getitem_421: "f32[384, 384, 1, 1]", getitem_422: "f32[384]", getitem_423: "f32[384]", getitem_424: "f32[384]", getitem_425: "f32[384]", getitem_426: "f32[384, 384, 3, 3]", getitem_427: "f32[384]", getitem_428: "f32[384]", getitem_429: "f32[384, 384, 1, 1]", getitem_430: "f32[384]", getitem_431: "f32[384]", getitem_432: "f32[384]", getitem_433: "f32[384]", getitem_434: "f32[384, 384, 3, 3]", getitem_435: "f32[384]", getitem_436: "f32[384]", getitem_437: "f32[384, 384, 1, 1]", getitem_438: "f32[384]", getitem_439: "f32[384]", getitem_440: "f32[384]", getitem_441: "f32[384]", getitem_442: "f32[384, 384, 3, 3]", getitem_443: "f32[384]", getitem_444: "f32[384]", getitem_445: "f32[384, 384, 1, 1]", getitem_446: "f32[384]", getitem_447: "f32[384]", getitem_448: "f32[384]", getitem_449: "f32[384]", getitem_450: "f32[384, 384, 3, 3]", getitem_451: "f32[384]", getitem_452: "f32[384]", getitem_453: "f32[384, 384, 1, 1]", getitem_454: "f32[384]", getitem_455: "f32[384]", getitem_456: "f32[384]", getitem_457: "f32[384]", getitem_458: "f32[384, 384, 3, 3]", getitem_459: "f32[384]", getitem_460: "f32[384]", getitem_461: "f32[384, 384, 1, 1]", getitem_462: "f32[384]", getitem_463: "f32[384]", getitem_464: "f32[384]", getitem_465: "f32[384]", getitem_466: "f32[384, 384, 3, 3]", getitem_467: "f32[384]", getitem_468: "f32[384]", getitem_469: "f32[384, 384, 1, 1]", getitem_470: "f32[384]", getitem_471: "f32[384]", getitem_472: "f32[384]", getitem_473: "f32[384]", getitem_474: "f32[384, 384, 3, 3]", getitem_475: "f32[384]", getitem_476: "f32[384]", getitem_477: "f32[384, 384, 1, 1]", getitem_478: "f32[384]", getitem_479: "f32[384]", getitem_480: "f32[384]", getitem_481: "f32[384]", getitem_482: "f32[384, 384, 3, 3]", getitem_483: "f32[384]", getitem_484: "f32[384]", getitem_485: "f32[384, 384, 1, 1]", getitem_486: "f32[384]", getitem_487: "f32[384]", getitem_488: "f32[384]", getitem_489: "f32[384]", getitem_490: "f32[384, 384, 3, 3]", getitem_491: "f32[384]", getitem_492: "f32[384]", getitem_493: "f32[384, 384, 1, 1]", getitem_494: "f32[384]", getitem_495: "f32[384]", getitem_496: "f32[1408, 384, 3, 3]", getitem_497: "f32[1408]", getitem_498: "f32[1408]", getitem_499: "f32[1408, 384, 1, 1]", getitem_500: "f32[1408]", getitem_501: "f32[1408]", getitem_502: "f32[1000, 1408]", getitem_503: "f32[1000]", getitem_2688: "f32[64, 3, 3, 3]", getitem_2689: "f32[64]", getitem_2690: "f32[64]", getitem_2691: "f32[64, 3, 1, 1]", getitem_2692: "f32[64]", getitem_2693: "f32[64]", getitem_2694: "f32[96, 64, 3, 3]", getitem_2695: "f32[96]", getitem_2696: "f32[96]", getitem_2697: "f32[96, 64, 1, 1]", getitem_2698: "f32[96]", getitem_2699: "f32[96]", getitem_2700: "f32[96]", getitem_2701: "f32[96]", getitem_2702: "f32[96, 96, 3, 3]", getitem_2703: "f32[96]", getitem_2704: "f32[96]", getitem_2705: "f32[96, 96, 1, 1]", getitem_2706: "f32[96]", getitem_2707: "f32[96]", getitem_2708: "f32[192, 96, 3, 3]", getitem_2709: "f32[192]", getitem_2710: "f32[192]", getitem_2711: "f32[192, 96, 1, 1]", getitem_2712: "f32[192]", getitem_2713: "f32[192]", getitem_2714: "f32[192]", getitem_2715: "f32[192]", getitem_2716: "f32[192, 192, 3, 3]", getitem_2717: "f32[192]", getitem_2718: "f32[192]", getitem_2719: "f32[192, 192, 1, 1]", getitem_2720: "f32[192]", getitem_2721: "f32[192]", getitem_2722: "f32[192]", getitem_2723: "f32[192]", getitem_2724: "f32[192, 192, 3, 3]", getitem_2725: "f32[192]", getitem_2726: "f32[192]", getitem_2727: "f32[192, 192, 1, 1]", getitem_2728: "f32[192]", getitem_2729: "f32[192]", getitem_2730: "f32[192]", getitem_2731: "f32[192]", getitem_2732: "f32[192, 192, 3, 3]", getitem_2733: "f32[192]", getitem_2734: "f32[192]", getitem_2735: "f32[192, 192, 1, 1]", getitem_2736: "f32[192]", getitem_2737: "f32[192]", getitem_2738: "f32[384, 192, 3, 3]", getitem_2739: "f32[384]", getitem_2740: "f32[384]", getitem_2741: "f32[384, 192, 1, 1]", getitem_2742: "f32[384]", getitem_2743: "f32[384]", getitem_2744: "f32[384]", getitem_2745: "f32[384]", getitem_2746: "f32[384, 384, 3, 3]", getitem_2747: "f32[384]", getitem_2748: "f32[384]", getitem_2749: "f32[384, 384, 1, 1]", getitem_2750: "f32[384]", getitem_2751: "f32[384]", getitem_2752: "f32[384]", getitem_2753: "f32[384]", getitem_2754: "f32[384, 384, 3, 3]", getitem_2755: "f32[384]", getitem_2756: "f32[384]", getitem_2757: "f32[384, 384, 1, 1]", getitem_2758: "f32[384]", getitem_2759: "f32[384]", getitem_2760: "f32[384]", getitem_2761: "f32[384]", getitem_2762: "f32[384, 384, 3, 3]", getitem_2763: "f32[384]", getitem_2764: "f32[384]", getitem_2765: "f32[384, 384, 1, 1]", getitem_2766: "f32[384]", getitem_2767: "f32[384]", getitem_2768: "f32[384]", getitem_2769: "f32[384]", getitem_2770: "f32[384, 384, 3, 3]", getitem_2771: "f32[384]", getitem_2772: "f32[384]", getitem_2773: "f32[384, 384, 1, 1]", getitem_2774: "f32[384]", getitem_2775: "f32[384]", getitem_2776: "f32[384]", getitem_2777: "f32[384]", getitem_2778: "f32[384, 384, 3, 3]", getitem_2779: "f32[384]", getitem_2780: "f32[384]", getitem_2781: "f32[384, 384, 1, 1]", getitem_2782: "f32[384]", getitem_2783: "f32[384]", getitem_2784: "f32[384]", getitem_2785: "f32[384]", getitem_2786: "f32[384, 384, 3, 3]", getitem_2787: "f32[384]", getitem_2788: "f32[384]", getitem_2789: "f32[384, 384, 1, 1]", getitem_2790: "f32[384]", getitem_2791: "f32[384]", getitem_2792: "f32[384]", getitem_2793: "f32[384]", getitem_2794: "f32[384, 384, 3, 3]", getitem_2795: "f32[384]", getitem_2796: "f32[384]", getitem_2797: "f32[384, 384, 1, 1]", getitem_2798: "f32[384]", getitem_2799: "f32[384]", getitem_2800: "f32[384]", getitem_2801: "f32[384]", getitem_2802: "f32[384, 384, 3, 3]", getitem_2803: "f32[384]", getitem_2804: "f32[384]", getitem_2805: "f32[384, 384, 1, 1]", getitem_2806: "f32[384]", getitem_2807: "f32[384]", getitem_2808: "f32[384]", getitem_2809: "f32[384]", getitem_2810: "f32[384, 384, 3, 3]", getitem_2811: "f32[384]", getitem_2812: "f32[384]", getitem_2813: "f32[384, 384, 1, 1]", getitem_2814: "f32[384]", getitem_2815: "f32[384]", getitem_2816: "f32[384]", getitem_2817: "f32[384]", getitem_2818: "f32[384, 384, 3, 3]", getitem_2819: "f32[384]", getitem_2820: "f32[384]", getitem_2821: "f32[384, 384, 1, 1]", getitem_2822: "f32[384]", getitem_2823: "f32[384]", getitem_2824: "f32[384]", getitem_2825: "f32[384]", getitem_2826: "f32[384, 384, 3, 3]", getitem_2827: "f32[384]", getitem_2828: "f32[384]", getitem_2829: "f32[384, 384, 1, 1]", getitem_2830: "f32[384]", getitem_2831: "f32[384]", getitem_2832: "f32[384]", getitem_2833: "f32[384]", getitem_2834: "f32[384, 384, 3, 3]", getitem_2835: "f32[384]", getitem_2836: "f32[384]", getitem_2837: "f32[384, 384, 1, 1]", getitem_2838: "f32[384]", getitem_2839: "f32[384]", getitem_2840: "f32[384]", getitem_2841: "f32[384]", getitem_2842: "f32[384, 384, 3, 3]", getitem_2843: "f32[384]", getitem_2844: "f32[384]", getitem_2845: "f32[384, 384, 1, 1]", getitem_2846: "f32[384]", getitem_2847: "f32[384]", getitem_2848: "f32[1408, 384, 3, 3]", getitem_2849: "f32[1408]", getitem_2850: "f32[1408]", getitem_2851: "f32[1408, 384, 1, 1]", getitem_2852: "f32[1408]", getitem_2853: "f32[1408]", getitem_2854: "f32[1000, 1408]", getitem_2855: "f32[1000]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1], [getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431, getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503], [getitem_2688, getitem_2689, getitem_2690, getitem_2691, getitem_2692, getitem_2693, getitem_2694, getitem_2695, getitem_2696, getitem_2697, getitem_2698, getitem_2699, getitem_2700, getitem_2701, getitem_2702, getitem_2703, getitem_2704, getitem_2705, getitem_2706, getitem_2707, getitem_2708, getitem_2709, getitem_2710, getitem_2711, getitem_2712, getitem_2713, getitem_2714, getitem_2715, getitem_2716, getitem_2717, getitem_2718, getitem_2719, getitem_2720, getitem_2721, getitem_2722, getitem_2723, getitem_2724, getitem_2725, getitem_2726, getitem_2727, getitem_2728, getitem_2729, getitem_2730, getitem_2731, getitem_2732, getitem_2733, getitem_2734, getitem_2735, getitem_2736, getitem_2737, getitem_2738, getitem_2739, getitem_2740, getitem_2741, getitem_2742, getitem_2743, getitem_2744, getitem_2745, getitem_2746, getitem_2747, getitem_2748, getitem_2749, getitem_2750, getitem_2751, getitem_2752, getitem_2753, getitem_2754, getitem_2755, getitem_2756, getitem_2757, getitem_2758, getitem_2759, getitem_2760, getitem_2761, getitem_2762, getitem_2763, getitem_2764, getitem_2765, getitem_2766, getitem_2767, getitem_2768, getitem_2769, getitem_2770, getitem_2771, getitem_2772, getitem_2773, getitem_2774, getitem_2775, getitem_2776, getitem_2777, getitem_2778, getitem_2779, getitem_2780, getitem_2781, getitem_2782, getitem_2783, getitem_2784, getitem_2785, getitem_2786, getitem_2787, getitem_2788, getitem_2789, getitem_2790, getitem_2791, getitem_2792, getitem_2793, getitem_2794, getitem_2795, getitem_2796, getitem_2797, getitem_2798, getitem_2799, getitem_2800, getitem_2801, getitem_2802, getitem_2803, getitem_2804, getitem_2805, getitem_2806, getitem_2807, getitem_2808, getitem_2809, getitem_2810, getitem_2811, getitem_2812, getitem_2813, getitem_2814, getitem_2815, getitem_2816, getitem_2817, getitem_2818, getitem_2819, getitem_2820, getitem_2821, getitem_2822, getitem_2823, getitem_2824, getitem_2825, getitem_2826, getitem_2827, getitem_2828, getitem_2829, getitem_2830, getitem_2831, getitem_2832, getitem_2833, getitem_2834, getitem_2835, getitem_2836, getitem_2837, getitem_2838, getitem_2839, getitem_2840, getitem_2841, getitem_2842, getitem_2843, getitem_2844, getitem_2845, getitem_2846, getitem_2847, getitem_2848, getitem_2849, getitem_2850, getitem_2851, getitem_2852, getitem_2853, getitem_2854, getitem_2855]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = arg13_1 = arg14_1 = arg15_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = arg32_1 = arg33_1 = arg34_1 = arg35_1 = arg36_1 = arg37_1 = arg38_1 = arg39_1 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = arg50_1 = arg51_1 = arg52_1 = arg53_1 = arg54_1 = arg55_1 = arg56_1 = arg57_1 = arg58_1 = arg59_1 = arg60_1 = arg61_1 = arg62_1 = arg63_1 = arg64_1 = arg65_1 = arg66_1 = arg67_1 = arg68_1 = arg69_1 = arg70_1 = arg71_1 = arg72_1 = arg73_1 = arg74_1 = arg75_1 = arg76_1 = arg77_1 = arg78_1 = arg79_1 = arg80_1 = arg81_1 = arg82_1 = arg83_1 = arg84_1 = arg85_1 = arg86_1 = arg87_1 = arg88_1 = arg89_1 = arg90_1 = arg91_1 = arg92_1 = arg93_1 = arg94_1 = arg95_1 = arg96_1 = arg97_1 = arg98_1 = arg99_1 = arg100_1 = arg101_1 = arg102_1 = arg103_1 = arg104_1 = arg105_1 = arg106_1 = arg107_1 = arg108_1 = arg109_1 = arg110_1 = arg111_1 = arg112_1 = arg113_1 = arg114_1 = arg115_1 = arg116_1 = arg117_1 = arg118_1 = arg119_1 = arg120_1 = arg121_1 = arg122_1 = arg123_1 = arg124_1 = arg125_1 = arg126_1 = arg127_1 = arg128_1 = arg129_1 = arg130_1 = arg131_1 = arg132_1 = arg133_1 = arg134_1 = arg135_1 = arg136_1 = arg137_1 = arg138_1 = arg139_1 = arg140_1 = arg141_1 = arg142_1 = arg143_1 = arg144_1 = arg145_1 = arg146_1 = arg147_1 = arg148_1 = arg149_1 = arg150_1 = arg151_1 = arg152_1 = arg153_1 = arg154_1 = arg155_1 = arg156_1 = arg157_1 = arg158_1 = arg159_1 = arg160_1 = arg161_1 = arg162_1 = arg163_1 = arg164_1 = arg165_1 = arg166_1 = arg167_1 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = getitem_367 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = getitem_373 = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = getitem_409 = getitem_410 = getitem_411 = getitem_412 = getitem_413 = getitem_414 = getitem_415 = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = getitem_432 = getitem_433 = getitem_434 = getitem_435 = getitem_436 = getitem_437 = getitem_438 = getitem_439 = getitem_440 = getitem_441 = getitem_442 = getitem_443 = getitem_444 = getitem_445 = getitem_446 = getitem_447 = getitem_448 = getitem_449 = getitem_450 = getitem_451 = getitem_452 = getitem_453 = getitem_454 = getitem_455 = getitem_456 = getitem_457 = getitem_458 = getitem_459 = getitem_460 = getitem_461 = getitem_462 = getitem_463 = getitem_464 = getitem_465 = getitem_466 = getitem_467 = getitem_468 = getitem_469 = getitem_470 = getitem_471 = getitem_472 = getitem_473 = getitem_474 = getitem_475 = getitem_476 = getitem_477 = getitem_478 = getitem_479 = getitem_480 = getitem_481 = getitem_482 = getitem_483 = getitem_484 = getitem_485 = getitem_486 = getitem_487 = getitem_488 = getitem_489 = getitem_490 = getitem_491 = getitem_492 = getitem_493 = getitem_494 = getitem_495 = getitem_496 = getitem_497 = getitem_498 = getitem_499 = getitem_500 = getitem_501 = getitem_502 = getitem_503 = getitem_2688 = getitem_2689 = getitem_2690 = getitem_2691 = getitem_2692 = getitem_2693 = getitem_2694 = getitem_2695 = getitem_2696 = getitem_2697 = getitem_2698 = getitem_2699 = getitem_2700 = getitem_2701 = getitem_2702 = getitem_2703 = getitem_2704 = getitem_2705 = getitem_2706 = getitem_2707 = getitem_2708 = getitem_2709 = getitem_2710 = getitem_2711 = getitem_2712 = getitem_2713 = getitem_2714 = getitem_2715 = getitem_2716 = getitem_2717 = getitem_2718 = getitem_2719 = getitem_2720 = getitem_2721 = getitem_2722 = getitem_2723 = getitem_2724 = getitem_2725 = getitem_2726 = getitem_2727 = getitem_2728 = getitem_2729 = getitem_2730 = getitem_2731 = getitem_2732 = getitem_2733 = getitem_2734 = getitem_2735 = getitem_2736 = getitem_2737 = getitem_2738 = getitem_2739 = getitem_2740 = getitem_2741 = getitem_2742 = getitem_2743 = getitem_2744 = getitem_2745 = getitem_2746 = getitem_2747 = getitem_2748 = getitem_2749 = getitem_2750 = getitem_2751 = getitem_2752 = getitem_2753 = getitem_2754 = getitem_2755 = getitem_2756 = getitem_2757 = getitem_2758 = getitem_2759 = getitem_2760 = getitem_2761 = getitem_2762 = getitem_2763 = getitem_2764 = getitem_2765 = getitem_2766 = getitem_2767 = getitem_2768 = getitem_2769 = getitem_2770 = getitem_2771 = getitem_2772 = getitem_2773 = getitem_2774 = getitem_2775 = getitem_2776 = getitem_2777 = getitem_2778 = getitem_2779 = getitem_2780 = getitem_2781 = getitem_2782 = getitem_2783 = getitem_2784 = getitem_2785 = getitem_2786 = getitem_2787 = getitem_2788 = getitem_2789 = getitem_2790 = getitem_2791 = getitem_2792 = getitem_2793 = getitem_2794 = getitem_2795 = getitem_2796 = getitem_2797 = getitem_2798 = getitem_2799 = getitem_2800 = getitem_2801 = getitem_2802 = getitem_2803 = getitem_2804 = getitem_2805 = getitem_2806 = getitem_2807 = getitem_2808 = getitem_2809 = getitem_2810 = getitem_2811 = getitem_2812 = getitem_2813 = getitem_2814 = getitem_2815 = getitem_2816 = getitem_2817 = getitem_2818 = getitem_2819 = getitem_2820 = getitem_2821 = getitem_2822 = getitem_2823 = getitem_2824 = getitem_2825 = getitem_2826 = getitem_2827 = getitem_2828 = getitem_2829 = getitem_2830 = getitem_2831 = getitem_2832 = getitem_2833 = getitem_2834 = getitem_2835 = getitem_2836 = getitem_2837 = getitem_2838 = getitem_2839 = getitem_2840 = getitem_2841 = getitem_2842 = getitem_2843 = getitem_2844 = getitem_2845 = getitem_2846 = getitem_2847 = getitem_2848 = getitem_2849 = getitem_2850 = getitem_2851 = getitem_2852 = getitem_2853 = getitem_2854 = getitem_2855 = None
        getitem: "f32[64, 3, 3, 3]" = _foreach_addcdiv_scalar[0]
        getitem_2856: "f32[64]" = _foreach_addcdiv_scalar[1]
        getitem_2857: "f32[64]" = _foreach_addcdiv_scalar[2]
        getitem_2858: "f32[64, 3, 1, 1]" = _foreach_addcdiv_scalar[3]
        getitem_2859: "f32[64]" = _foreach_addcdiv_scalar[4]
        getitem_2860: "f32[64]" = _foreach_addcdiv_scalar[5]
        getitem_2861: "f32[96, 64, 3, 3]" = _foreach_addcdiv_scalar[6]
        getitem_2862: "f32[96]" = _foreach_addcdiv_scalar[7]
        getitem_2863: "f32[96]" = _foreach_addcdiv_scalar[8]
        getitem_2864: "f32[96, 64, 1, 1]" = _foreach_addcdiv_scalar[9]
        getitem_2865: "f32[96]" = _foreach_addcdiv_scalar[10]
        getitem_2866: "f32[96]" = _foreach_addcdiv_scalar[11]
        getitem_2867: "f32[96]" = _foreach_addcdiv_scalar[12]
        getitem_2868: "f32[96]" = _foreach_addcdiv_scalar[13]
        getitem_2869: "f32[96, 96, 3, 3]" = _foreach_addcdiv_scalar[14]
        getitem_2870: "f32[96]" = _foreach_addcdiv_scalar[15]
        getitem_2871: "f32[96]" = _foreach_addcdiv_scalar[16]
        getitem_2872: "f32[96, 96, 1, 1]" = _foreach_addcdiv_scalar[17]
        getitem_2873: "f32[96]" = _foreach_addcdiv_scalar[18]
        getitem_2874: "f32[96]" = _foreach_addcdiv_scalar[19]
        getitem_2875: "f32[192, 96, 3, 3]" = _foreach_addcdiv_scalar[20]
        getitem_2876: "f32[192]" = _foreach_addcdiv_scalar[21]
        getitem_2877: "f32[192]" = _foreach_addcdiv_scalar[22]
        getitem_2878: "f32[192, 96, 1, 1]" = _foreach_addcdiv_scalar[23]
        getitem_2879: "f32[192]" = _foreach_addcdiv_scalar[24]
        getitem_2880: "f32[192]" = _foreach_addcdiv_scalar[25]
        getitem_2881: "f32[192]" = _foreach_addcdiv_scalar[26]
        getitem_2882: "f32[192]" = _foreach_addcdiv_scalar[27]
        getitem_2883: "f32[192, 192, 3, 3]" = _foreach_addcdiv_scalar[28]
        getitem_2884: "f32[192]" = _foreach_addcdiv_scalar[29]
        getitem_2885: "f32[192]" = _foreach_addcdiv_scalar[30]
        getitem_2886: "f32[192, 192, 1, 1]" = _foreach_addcdiv_scalar[31]
        getitem_2887: "f32[192]" = _foreach_addcdiv_scalar[32]
        getitem_2888: "f32[192]" = _foreach_addcdiv_scalar[33]
        getitem_2889: "f32[192]" = _foreach_addcdiv_scalar[34]
        getitem_2890: "f32[192]" = _foreach_addcdiv_scalar[35]
        getitem_2891: "f32[192, 192, 3, 3]" = _foreach_addcdiv_scalar[36]
        getitem_2892: "f32[192]" = _foreach_addcdiv_scalar[37]
        getitem_2893: "f32[192]" = _foreach_addcdiv_scalar[38]
        getitem_2894: "f32[192, 192, 1, 1]" = _foreach_addcdiv_scalar[39]
        getitem_2895: "f32[192]" = _foreach_addcdiv_scalar[40]
        getitem_2896: "f32[192]" = _foreach_addcdiv_scalar[41]
        getitem_2897: "f32[192]" = _foreach_addcdiv_scalar[42]
        getitem_2898: "f32[192]" = _foreach_addcdiv_scalar[43]
        getitem_2899: "f32[192, 192, 3, 3]" = _foreach_addcdiv_scalar[44]
        getitem_2900: "f32[192]" = _foreach_addcdiv_scalar[45]
        getitem_2901: "f32[192]" = _foreach_addcdiv_scalar[46]
        getitem_2902: "f32[192, 192, 1, 1]" = _foreach_addcdiv_scalar[47]
        getitem_2903: "f32[192]" = _foreach_addcdiv_scalar[48]
        getitem_2904: "f32[192]" = _foreach_addcdiv_scalar[49]
        getitem_2905: "f32[384, 192, 3, 3]" = _foreach_addcdiv_scalar[50]
        getitem_2906: "f32[384]" = _foreach_addcdiv_scalar[51]
        getitem_2907: "f32[384]" = _foreach_addcdiv_scalar[52]
        getitem_2908: "f32[384, 192, 1, 1]" = _foreach_addcdiv_scalar[53]
        getitem_2909: "f32[384]" = _foreach_addcdiv_scalar[54]
        getitem_2910: "f32[384]" = _foreach_addcdiv_scalar[55]
        getitem_2911: "f32[384]" = _foreach_addcdiv_scalar[56]
        getitem_2912: "f32[384]" = _foreach_addcdiv_scalar[57]
        getitem_2913: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[58]
        getitem_2914: "f32[384]" = _foreach_addcdiv_scalar[59]
        getitem_2915: "f32[384]" = _foreach_addcdiv_scalar[60]
        getitem_2916: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[61]
        getitem_2917: "f32[384]" = _foreach_addcdiv_scalar[62]
        getitem_2918: "f32[384]" = _foreach_addcdiv_scalar[63]
        getitem_2919: "f32[384]" = _foreach_addcdiv_scalar[64]
        getitem_2920: "f32[384]" = _foreach_addcdiv_scalar[65]
        getitem_2921: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[66]
        getitem_2922: "f32[384]" = _foreach_addcdiv_scalar[67]
        getitem_2923: "f32[384]" = _foreach_addcdiv_scalar[68]
        getitem_2924: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[69]
        getitem_2925: "f32[384]" = _foreach_addcdiv_scalar[70]
        getitem_2926: "f32[384]" = _foreach_addcdiv_scalar[71]
        getitem_2927: "f32[384]" = _foreach_addcdiv_scalar[72]
        getitem_2928: "f32[384]" = _foreach_addcdiv_scalar[73]
        getitem_2929: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[74]
        getitem_2930: "f32[384]" = _foreach_addcdiv_scalar[75]
        getitem_2931: "f32[384]" = _foreach_addcdiv_scalar[76]
        getitem_2932: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[77]
        getitem_2933: "f32[384]" = _foreach_addcdiv_scalar[78]
        getitem_2934: "f32[384]" = _foreach_addcdiv_scalar[79]
        getitem_2935: "f32[384]" = _foreach_addcdiv_scalar[80]
        getitem_2936: "f32[384]" = _foreach_addcdiv_scalar[81]
        getitem_2937: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[82]
        getitem_2938: "f32[384]" = _foreach_addcdiv_scalar[83]
        getitem_2939: "f32[384]" = _foreach_addcdiv_scalar[84]
        getitem_2940: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[85]
        getitem_2941: "f32[384]" = _foreach_addcdiv_scalar[86]
        getitem_2942: "f32[384]" = _foreach_addcdiv_scalar[87]
        getitem_2943: "f32[384]" = _foreach_addcdiv_scalar[88]
        getitem_2944: "f32[384]" = _foreach_addcdiv_scalar[89]
        getitem_2945: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[90]
        getitem_2946: "f32[384]" = _foreach_addcdiv_scalar[91]
        getitem_2947: "f32[384]" = _foreach_addcdiv_scalar[92]
        getitem_2948: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[93]
        getitem_2949: "f32[384]" = _foreach_addcdiv_scalar[94]
        getitem_2950: "f32[384]" = _foreach_addcdiv_scalar[95]
        getitem_2951: "f32[384]" = _foreach_addcdiv_scalar[96]
        getitem_2952: "f32[384]" = _foreach_addcdiv_scalar[97]
        getitem_2953: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[98]
        getitem_2954: "f32[384]" = _foreach_addcdiv_scalar[99]
        getitem_2955: "f32[384]" = _foreach_addcdiv_scalar[100]
        getitem_2956: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[101]
        getitem_2957: "f32[384]" = _foreach_addcdiv_scalar[102]
        getitem_2958: "f32[384]" = _foreach_addcdiv_scalar[103]
        getitem_2959: "f32[384]" = _foreach_addcdiv_scalar[104]
        getitem_2960: "f32[384]" = _foreach_addcdiv_scalar[105]
        getitem_2961: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[106]
        getitem_2962: "f32[384]" = _foreach_addcdiv_scalar[107]
        getitem_2963: "f32[384]" = _foreach_addcdiv_scalar[108]
        getitem_2964: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[109]
        getitem_2965: "f32[384]" = _foreach_addcdiv_scalar[110]
        getitem_2966: "f32[384]" = _foreach_addcdiv_scalar[111]
        getitem_2967: "f32[384]" = _foreach_addcdiv_scalar[112]
        getitem_2968: "f32[384]" = _foreach_addcdiv_scalar[113]
        getitem_2969: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[114]
        getitem_2970: "f32[384]" = _foreach_addcdiv_scalar[115]
        getitem_2971: "f32[384]" = _foreach_addcdiv_scalar[116]
        getitem_2972: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[117]
        getitem_2973: "f32[384]" = _foreach_addcdiv_scalar[118]
        getitem_2974: "f32[384]" = _foreach_addcdiv_scalar[119]
        getitem_2975: "f32[384]" = _foreach_addcdiv_scalar[120]
        getitem_2976: "f32[384]" = _foreach_addcdiv_scalar[121]
        getitem_2977: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[122]
        getitem_2978: "f32[384]" = _foreach_addcdiv_scalar[123]
        getitem_2979: "f32[384]" = _foreach_addcdiv_scalar[124]
        getitem_2980: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[125]
        getitem_2981: "f32[384]" = _foreach_addcdiv_scalar[126]
        getitem_2982: "f32[384]" = _foreach_addcdiv_scalar[127]
        getitem_2983: "f32[384]" = _foreach_addcdiv_scalar[128]
        getitem_2984: "f32[384]" = _foreach_addcdiv_scalar[129]
        getitem_2985: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[130]
        getitem_2986: "f32[384]" = _foreach_addcdiv_scalar[131]
        getitem_2987: "f32[384]" = _foreach_addcdiv_scalar[132]
        getitem_2988: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[133]
        getitem_2989: "f32[384]" = _foreach_addcdiv_scalar[134]
        getitem_2990: "f32[384]" = _foreach_addcdiv_scalar[135]
        getitem_2991: "f32[384]" = _foreach_addcdiv_scalar[136]
        getitem_2992: "f32[384]" = _foreach_addcdiv_scalar[137]
        getitem_2993: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[138]
        getitem_2994: "f32[384]" = _foreach_addcdiv_scalar[139]
        getitem_2995: "f32[384]" = _foreach_addcdiv_scalar[140]
        getitem_2996: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[141]
        getitem_2997: "f32[384]" = _foreach_addcdiv_scalar[142]
        getitem_2998: "f32[384]" = _foreach_addcdiv_scalar[143]
        getitem_2999: "f32[384]" = _foreach_addcdiv_scalar[144]
        getitem_3000: "f32[384]" = _foreach_addcdiv_scalar[145]
        getitem_3001: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[146]
        getitem_3002: "f32[384]" = _foreach_addcdiv_scalar[147]
        getitem_3003: "f32[384]" = _foreach_addcdiv_scalar[148]
        getitem_3004: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[149]
        getitem_3005: "f32[384]" = _foreach_addcdiv_scalar[150]
        getitem_3006: "f32[384]" = _foreach_addcdiv_scalar[151]
        getitem_3007: "f32[384]" = _foreach_addcdiv_scalar[152]
        getitem_3008: "f32[384]" = _foreach_addcdiv_scalar[153]
        getitem_3009: "f32[384, 384, 3, 3]" = _foreach_addcdiv_scalar[154]
        getitem_3010: "f32[384]" = _foreach_addcdiv_scalar[155]
        getitem_3011: "f32[384]" = _foreach_addcdiv_scalar[156]
        getitem_3012: "f32[384, 384, 1, 1]" = _foreach_addcdiv_scalar[157]
        getitem_3013: "f32[384]" = _foreach_addcdiv_scalar[158]
        getitem_3014: "f32[384]" = _foreach_addcdiv_scalar[159]
        getitem_3015: "f32[1408, 384, 3, 3]" = _foreach_addcdiv_scalar[160]
        getitem_3016: "f32[1408]" = _foreach_addcdiv_scalar[161]
        getitem_3017: "f32[1408]" = _foreach_addcdiv_scalar[162]
        getitem_3018: "f32[1408, 384, 1, 1]" = _foreach_addcdiv_scalar[163]
        getitem_3019: "f32[1408]" = _foreach_addcdiv_scalar[164]
        getitem_3020: "f32[1408]" = _foreach_addcdiv_scalar[165]
        getitem_3021: "f32[1000, 1408]" = _foreach_addcdiv_scalar[166]
        getitem_3022: "f32[1000]" = _foreach_addcdiv_scalar[167];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_2856, getitem_2857, getitem_2858, getitem_2859, getitem_2860, getitem_2861, getitem_2862, getitem_2863, getitem_2864, getitem_2865, getitem_2866, getitem_2867, getitem_2868, getitem_2869, getitem_2870, getitem_2871, getitem_2872, getitem_2873, getitem_2874, getitem_2875, getitem_2876, getitem_2877, getitem_2878, getitem_2879, getitem_2880, getitem_2881, getitem_2882, getitem_2883, getitem_2884, getitem_2885, getitem_2886, getitem_2887, getitem_2888, getitem_2889, getitem_2890, getitem_2891, getitem_2892, getitem_2893, getitem_2894, getitem_2895, getitem_2896, getitem_2897, getitem_2898, getitem_2899, getitem_2900, getitem_2901, getitem_2902, getitem_2903, getitem_2904, getitem_2905, getitem_2906, getitem_2907, getitem_2908, getitem_2909, getitem_2910, getitem_2911, getitem_2912, getitem_2913, getitem_2914, getitem_2915, getitem_2916, getitem_2917, getitem_2918, getitem_2919, getitem_2920, getitem_2921, getitem_2922, getitem_2923, getitem_2924, getitem_2925, getitem_2926, getitem_2927, getitem_2928, getitem_2929, getitem_2930, getitem_2931, getitem_2932, getitem_2933, getitem_2934, getitem_2935, getitem_2936, getitem_2937, getitem_2938, getitem_2939, getitem_2940, getitem_2941, getitem_2942, getitem_2943, getitem_2944, getitem_2945, getitem_2946, getitem_2947, getitem_2948, getitem_2949, getitem_2950, getitem_2951, getitem_2952, getitem_2953, getitem_2954, getitem_2955, getitem_2956, getitem_2957, getitem_2958, getitem_2959, getitem_2960, getitem_2961, getitem_2962, getitem_2963, getitem_2964, getitem_2965, getitem_2966, getitem_2967, getitem_2968, getitem_2969, getitem_2970, getitem_2971, getitem_2972, getitem_2973, getitem_2974, getitem_2975, getitem_2976, getitem_2977, getitem_2978, getitem_2979, getitem_2980, getitem_2981, getitem_2982, getitem_2983, getitem_2984, getitem_2985, getitem_2986, getitem_2987, getitem_2988, getitem_2989, getitem_2990, getitem_2991, getitem_2992, getitem_2993, getitem_2994, getitem_2995, getitem_2996, getitem_2997, getitem_2998, getitem_2999, getitem_3000, getitem_3001, getitem_3002, getitem_3003, getitem_3004, getitem_3005, getitem_3006, getitem_3007, getitem_3008, getitem_3009, getitem_3010, getitem_3011, getitem_3012, getitem_3013, getitem_3014, getitem_3015, getitem_3016, getitem_3017, getitem_3018, getitem_3019, getitem_3020, getitem_3021, getitem_3022)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

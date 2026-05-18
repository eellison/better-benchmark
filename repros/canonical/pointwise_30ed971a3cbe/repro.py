"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s3_g70
Pattern hash: 30ed971a3cbe
Shape hash: 54322a09
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
    def forward(self, arg0_1: "f32[1, 1, 192]", getitem_2584: "f32[1, 1, 192]", arg1_1: "f32[1, 197, 192]", getitem_2585: "f32[1, 197, 192]", arg2_1: "f32[192, 3, 16, 16]", getitem_2586: "f32[192, 3, 16, 16]", arg3_1: "f32[192]", getitem_2587: "f32[192]", arg4_1: "f32[192]", getitem_2588: "f32[192]", arg5_1: "f32[192]", getitem_2589: "f32[192]", arg6_1: "f32[576, 192]", getitem_2590: "f32[576, 192]", arg7_1: "f32[576]", getitem_2591: "f32[576]", arg8_1: "f32[192, 192]", getitem_2592: "f32[192, 192]", arg9_1: "f32[192]", getitem_2593: "f32[192]", arg10_1: "f32[192]", getitem_2594: "f32[192]", arg11_1: "f32[192]", getitem_2595: "f32[192]", arg12_1: "f32[768, 192]", getitem_2596: "f32[768, 192]", arg13_1: "f32[768]", getitem_2597: "f32[768]", arg14_1: "f32[192, 768]", getitem_2598: "f32[192, 768]", arg15_1: "f32[192]", getitem_2599: "f32[192]", arg16_1: "f32[192]", getitem_2600: "f32[192]", arg17_1: "f32[192]", getitem_2601: "f32[192]", arg18_1: "f32[576, 192]", getitem_2602: "f32[576, 192]", arg19_1: "f32[576]", getitem_2603: "f32[576]", arg20_1: "f32[192, 192]", getitem_2604: "f32[192, 192]", arg21_1: "f32[192]", getitem_2605: "f32[192]", arg22_1: "f32[192]", getitem_2606: "f32[192]", arg23_1: "f32[192]", getitem_2607: "f32[192]", arg24_1: "f32[768, 192]", getitem_2608: "f32[768, 192]", arg25_1: "f32[768]", getitem_2609: "f32[768]", arg26_1: "f32[192, 768]", getitem_2610: "f32[192, 768]", arg27_1: "f32[192]", getitem_2611: "f32[192]", arg28_1: "f32[192]", getitem_2612: "f32[192]", arg29_1: "f32[192]", getitem_2613: "f32[192]", arg30_1: "f32[576, 192]", getitem_2614: "f32[576, 192]", arg31_1: "f32[576]", getitem_2615: "f32[576]", arg32_1: "f32[192, 192]", getitem_2616: "f32[192, 192]", arg33_1: "f32[192]", getitem_2617: "f32[192]", arg34_1: "f32[192]", getitem_2618: "f32[192]", arg35_1: "f32[192]", getitem_2619: "f32[192]", arg36_1: "f32[768, 192]", getitem_2620: "f32[768, 192]", arg37_1: "f32[768]", getitem_2621: "f32[768]", arg38_1: "f32[192, 768]", getitem_2622: "f32[192, 768]", arg39_1: "f32[192]", getitem_2623: "f32[192]", arg40_1: "f32[192]", getitem_2624: "f32[192]", arg41_1: "f32[192]", getitem_2625: "f32[192]", arg42_1: "f32[576, 192]", getitem_2626: "f32[576, 192]", arg43_1: "f32[576]", getitem_2627: "f32[576]", arg44_1: "f32[192, 192]", getitem_2628: "f32[192, 192]", arg45_1: "f32[192]", getitem_2629: "f32[192]", arg46_1: "f32[192]", getitem_2630: "f32[192]", arg47_1: "f32[192]", getitem_2631: "f32[192]", arg48_1: "f32[768, 192]", getitem_2632: "f32[768, 192]", arg49_1: "f32[768]", getitem_2633: "f32[768]", arg50_1: "f32[192, 768]", getitem_2634: "f32[192, 768]", arg51_1: "f32[192]", getitem_2635: "f32[192]", arg52_1: "f32[192]", getitem_2636: "f32[192]", arg53_1: "f32[192]", getitem_2637: "f32[192]", arg54_1: "f32[576, 192]", getitem_2638: "f32[576, 192]", arg55_1: "f32[576]", getitem_2639: "f32[576]", arg56_1: "f32[192, 192]", getitem_2640: "f32[192, 192]", arg57_1: "f32[192]", getitem_2641: "f32[192]", arg58_1: "f32[192]", getitem_2642: "f32[192]", arg59_1: "f32[192]", getitem_2643: "f32[192]", arg60_1: "f32[768, 192]", getitem_2644: "f32[768, 192]", arg61_1: "f32[768]", getitem_2645: "f32[768]", arg62_1: "f32[192, 768]", getitem_2646: "f32[192, 768]", arg63_1: "f32[192]", getitem_2647: "f32[192]", arg64_1: "f32[192]", getitem_2648: "f32[192]", arg65_1: "f32[192]", getitem_2649: "f32[192]", arg66_1: "f32[576, 192]", getitem_2650: "f32[576, 192]", arg67_1: "f32[576]", getitem_2651: "f32[576]", arg68_1: "f32[192, 192]", getitem_2652: "f32[192, 192]", arg69_1: "f32[192]", getitem_2653: "f32[192]", arg70_1: "f32[192]", getitem_2654: "f32[192]", arg71_1: "f32[192]", getitem_2655: "f32[192]", arg72_1: "f32[768, 192]", getitem_2656: "f32[768, 192]", arg73_1: "f32[768]", getitem_2657: "f32[768]", arg74_1: "f32[192, 768]", getitem_2658: "f32[192, 768]", arg75_1: "f32[192]", getitem_2659: "f32[192]", arg76_1: "f32[192]", getitem_2660: "f32[192]", arg77_1: "f32[192]", getitem_2661: "f32[192]", arg78_1: "f32[576, 192]", getitem_2662: "f32[576, 192]", arg79_1: "f32[576]", getitem_2663: "f32[576]", arg80_1: "f32[192, 192]", getitem_2664: "f32[192, 192]", arg81_1: "f32[192]", getitem_2665: "f32[192]", arg82_1: "f32[192]", getitem_2666: "f32[192]", arg83_1: "f32[192]", getitem_2667: "f32[192]", arg84_1: "f32[768, 192]", getitem_2668: "f32[768, 192]", arg85_1: "f32[768]", getitem_2669: "f32[768]", arg86_1: "f32[192, 768]", getitem_2670: "f32[192, 768]", arg87_1: "f32[192]", getitem_2671: "f32[192]", arg88_1: "f32[192]", getitem_2672: "f32[192]", arg89_1: "f32[192]", getitem_2673: "f32[192]", arg90_1: "f32[576, 192]", getitem_2674: "f32[576, 192]", arg91_1: "f32[576]", getitem_2675: "f32[576]", arg92_1: "f32[192, 192]", getitem_2676: "f32[192, 192]", arg93_1: "f32[192]", getitem_2677: "f32[192]", arg94_1: "f32[192]", getitem_2678: "f32[192]", arg95_1: "f32[192]", getitem_2679: "f32[192]", arg96_1: "f32[768, 192]", getitem_2680: "f32[768, 192]", arg97_1: "f32[768]", getitem_2681: "f32[768]", arg98_1: "f32[192, 768]", getitem_2682: "f32[192, 768]", arg99_1: "f32[192]", getitem_2683: "f32[192]", arg100_1: "f32[192]", getitem_2684: "f32[192]", arg101_1: "f32[192]", getitem_2685: "f32[192]", arg102_1: "f32[576, 192]", getitem_2686: "f32[576, 192]", arg103_1: "f32[576]", getitem_2687: "f32[576]", arg104_1: "f32[192, 192]", getitem_2688: "f32[192, 192]", arg105_1: "f32[192]", getitem_2689: "f32[192]", arg106_1: "f32[192]", getitem_2690: "f32[192]", arg107_1: "f32[192]", getitem_2691: "f32[192]", arg108_1: "f32[768, 192]", getitem_2692: "f32[768, 192]", arg109_1: "f32[768]", getitem_2693: "f32[768]", arg110_1: "f32[192, 768]", getitem_2694: "f32[192, 768]", arg111_1: "f32[192]", getitem_2695: "f32[192]", arg112_1: "f32[192]", getitem_2696: "f32[192]", arg113_1: "f32[192]", getitem_2697: "f32[192]", arg114_1: "f32[576, 192]", getitem_2698: "f32[576, 192]", arg115_1: "f32[576]", getitem_2699: "f32[576]", arg116_1: "f32[192, 192]", getitem_2700: "f32[192, 192]", arg117_1: "f32[192]", getitem_2701: "f32[192]", arg118_1: "f32[192]", getitem_2702: "f32[192]", arg119_1: "f32[192]", getitem_2703: "f32[192]", arg120_1: "f32[768, 192]", getitem_2704: "f32[768, 192]", arg121_1: "f32[768]", getitem_2705: "f32[768]", arg122_1: "f32[192, 768]", getitem_2706: "f32[192, 768]", arg123_1: "f32[192]", getitem_2707: "f32[192]", arg124_1: "f32[192]", getitem_2708: "f32[192]", arg125_1: "f32[192]", getitem_2709: "f32[192]", arg126_1: "f32[576, 192]", getitem_2710: "f32[576, 192]", arg127_1: "f32[576]", getitem_2711: "f32[576]", arg128_1: "f32[192, 192]", getitem_2712: "f32[192, 192]", arg129_1: "f32[192]", getitem_2713: "f32[192]", arg130_1: "f32[192]", getitem_2714: "f32[192]", arg131_1: "f32[192]", getitem_2715: "f32[192]", arg132_1: "f32[768, 192]", getitem_2716: "f32[768, 192]", arg133_1: "f32[768]", getitem_2717: "f32[768]", arg134_1: "f32[192, 768]", getitem_2718: "f32[192, 768]", arg135_1: "f32[192]", getitem_2719: "f32[192]", arg136_1: "f32[192]", getitem_2720: "f32[192]", arg137_1: "f32[192]", getitem_2721: "f32[192]", arg138_1: "f32[576, 192]", getitem_2722: "f32[576, 192]", arg139_1: "f32[576]", getitem_2723: "f32[576]", arg140_1: "f32[192, 192]", getitem_2724: "f32[192, 192]", arg141_1: "f32[192]", getitem_2725: "f32[192]", arg142_1: "f32[192]", getitem_2726: "f32[192]", arg143_1: "f32[192]", getitem_2727: "f32[192]", arg144_1: "f32[768, 192]", getitem_2728: "f32[768, 192]", arg145_1: "f32[768]", getitem_2729: "f32[768]", arg146_1: "f32[192, 768]", getitem_2730: "f32[192, 768]", arg147_1: "f32[192]", getitem_2731: "f32[192]", arg148_1: "f32[192]", getitem_2732: "f32[192]", arg149_1: "f32[192]", getitem_2733: "f32[192]", arg150_1: "f32[1000, 192]", getitem_2734: "f32[1000, 192]", arg151_1: "f32[1000]", getitem_2735: "f32[1000]", arg152_1: "f32[]", getitem_1: "f32[]", arg153_1: "f32[1, 197, 192]", getitem_305: "f32[1, 197, 192]", arg154_1: "f32[1, 197, 192]", getitem_609: "f32[1, 197, 192]", arg155_1: "f32[1, 1, 192]", getitem_304: "f32[1, 1, 192]", arg156_1: "f32[1, 1, 192]", getitem_608: "f32[1, 1, 192]", arg157_1: "f32[]", getitem: "f32[]", arg158_1: "f32[]", getitem_2: "f32[]", arg159_1: "f32[]", getitem_3: "f32[]", arg160_1: "f32[]", getitem_4: "f32[]", arg161_1: "f32[]", getitem_5: "f32[]", arg162_1: "f32[]", getitem_6: "f32[]", arg163_1: "f32[]", getitem_7: "f32[]", arg164_1: "f32[]", getitem_8: "f32[]", arg165_1: "f32[]", getitem_9: "f32[]", arg166_1: "f32[]", getitem_10: "f32[]", arg167_1: "f32[]", getitem_11: "f32[]", arg168_1: "f32[]", getitem_12: "f32[]", arg169_1: "f32[]", getitem_13: "f32[]", arg170_1: "f32[]", getitem_14: "f32[]", arg171_1: "f32[]", getitem_15: "f32[]", arg172_1: "f32[]", getitem_16: "f32[]", arg173_1: "f32[]", getitem_17: "f32[]", arg174_1: "f32[]", getitem_18: "f32[]", arg175_1: "f32[]", getitem_19: "f32[]", arg176_1: "f32[]", getitem_20: "f32[]", arg177_1: "f32[]", getitem_21: "f32[]", arg178_1: "f32[]", getitem_22: "f32[]", arg179_1: "f32[]", getitem_23: "f32[]", arg180_1: "f32[]", getitem_24: "f32[]", arg181_1: "f32[]", getitem_25: "f32[]", arg182_1: "f32[]", getitem_26: "f32[]", arg183_1: "f32[]", getitem_27: "f32[]", arg184_1: "f32[]", getitem_28: "f32[]", arg185_1: "f32[]", getitem_29: "f32[]", arg186_1: "f32[]", getitem_30: "f32[]", arg187_1: "f32[]", getitem_31: "f32[]", arg188_1: "f32[]", getitem_32: "f32[]", arg189_1: "f32[]", getitem_33: "f32[]", arg190_1: "f32[]", getitem_34: "f32[]", arg191_1: "f32[]", getitem_35: "f32[]", arg192_1: "f32[]", getitem_36: "f32[]", arg193_1: "f32[]", getitem_37: "f32[]", arg194_1: "f32[]", getitem_38: "f32[]", arg195_1: "f32[]", getitem_39: "f32[]", arg196_1: "f32[]", getitem_40: "f32[]", arg197_1: "f32[]", getitem_41: "f32[]", arg198_1: "f32[]", getitem_42: "f32[]", arg199_1: "f32[]", getitem_43: "f32[]", arg200_1: "f32[]", getitem_44: "f32[]", arg201_1: "f32[]", getitem_45: "f32[]", arg202_1: "f32[]", getitem_46: "f32[]", arg203_1: "f32[]", getitem_47: "f32[]", arg204_1: "f32[]", getitem_48: "f32[]", arg205_1: "f32[]", getitem_49: "f32[]", arg206_1: "f32[]", getitem_50: "f32[]", arg207_1: "f32[]", getitem_51: "f32[]", arg208_1: "f32[]", getitem_52: "f32[]", arg209_1: "f32[]", getitem_53: "f32[]", arg210_1: "f32[]", getitem_54: "f32[]", arg211_1: "f32[]", getitem_55: "f32[]", arg212_1: "f32[]", getitem_56: "f32[]", arg213_1: "f32[]", getitem_57: "f32[]", arg214_1: "f32[]", getitem_58: "f32[]", arg215_1: "f32[]", getitem_59: "f32[]", arg216_1: "f32[]", getitem_60: "f32[]", arg217_1: "f32[]", getitem_61: "f32[]", arg218_1: "f32[]", getitem_62: "f32[]", arg219_1: "f32[]", getitem_63: "f32[]", arg220_1: "f32[]", getitem_64: "f32[]", arg221_1: "f32[]", getitem_65: "f32[]", arg222_1: "f32[]", getitem_66: "f32[]", arg223_1: "f32[]", getitem_67: "f32[]", arg224_1: "f32[]", getitem_68: "f32[]", arg225_1: "f32[]", getitem_69: "f32[]", arg226_1: "f32[]", getitem_70: "f32[]", arg227_1: "f32[]", getitem_71: "f32[]", arg228_1: "f32[]", getitem_72: "f32[]", arg229_1: "f32[]", getitem_73: "f32[]", arg230_1: "f32[]", getitem_74: "f32[]", arg231_1: "f32[]", getitem_75: "f32[]", arg232_1: "f32[]", getitem_76: "f32[]", arg233_1: "f32[]", getitem_77: "f32[]", arg234_1: "f32[]", getitem_78: "f32[]", arg235_1: "f32[]", getitem_79: "f32[]", arg236_1: "f32[]", getitem_80: "f32[]", arg237_1: "f32[]", getitem_81: "f32[]", arg238_1: "f32[]", getitem_82: "f32[]", arg239_1: "f32[]", getitem_83: "f32[]", arg240_1: "f32[]", getitem_84: "f32[]", arg241_1: "f32[]", getitem_85: "f32[]", arg242_1: "f32[]", getitem_86: "f32[]", arg243_1: "f32[]", getitem_87: "f32[]", arg244_1: "f32[]", getitem_88: "f32[]", arg245_1: "f32[]", getitem_89: "f32[]", arg246_1: "f32[]", getitem_90: "f32[]", arg247_1: "f32[]", getitem_91: "f32[]", arg248_1: "f32[]", getitem_92: "f32[]", arg249_1: "f32[]", getitem_93: "f32[]", arg250_1: "f32[]", getitem_94: "f32[]", arg251_1: "f32[]", getitem_95: "f32[]", arg252_1: "f32[]", getitem_96: "f32[]", arg253_1: "f32[]", getitem_97: "f32[]", arg254_1: "f32[]", getitem_98: "f32[]", arg255_1: "f32[]", getitem_99: "f32[]", arg256_1: "f32[]", getitem_100: "f32[]", arg257_1: "f32[]", getitem_101: "f32[]", arg258_1: "f32[]", getitem_102: "f32[]", arg259_1: "f32[]", getitem_103: "f32[]", arg260_1: "f32[]", getitem_104: "f32[]", arg261_1: "f32[]", getitem_105: "f32[]", arg262_1: "f32[]", getitem_106: "f32[]", arg263_1: "f32[]", getitem_107: "f32[]", arg264_1: "f32[]", getitem_108: "f32[]", arg265_1: "f32[]", getitem_109: "f32[]", arg266_1: "f32[]", getitem_110: "f32[]", arg267_1: "f32[]", getitem_111: "f32[]", arg268_1: "f32[]", getitem_112: "f32[]", arg269_1: "f32[]", getitem_113: "f32[]", arg270_1: "f32[]", getitem_114: "f32[]", arg271_1: "f32[]", getitem_115: "f32[]", arg272_1: "f32[]", getitem_116: "f32[]", arg273_1: "f32[]", getitem_117: "f32[]", arg274_1: "f32[]", getitem_118: "f32[]", arg275_1: "f32[]", getitem_119: "f32[]", arg276_1: "f32[]", getitem_120: "f32[]", arg277_1: "f32[]", getitem_121: "f32[]", arg278_1: "f32[]", getitem_122: "f32[]", arg279_1: "f32[]", getitem_123: "f32[]", arg280_1: "f32[]", getitem_124: "f32[]", arg281_1: "f32[]", getitem_125: "f32[]", arg282_1: "f32[]", getitem_126: "f32[]", arg283_1: "f32[]", getitem_127: "f32[]", arg284_1: "f32[]", getitem_128: "f32[]", arg285_1: "f32[]", getitem_129: "f32[]", arg286_1: "f32[]", getitem_130: "f32[]", arg287_1: "f32[]", getitem_131: "f32[]", arg288_1: "f32[]", getitem_132: "f32[]", arg289_1: "f32[]", getitem_133: "f32[]", arg290_1: "f32[]", getitem_134: "f32[]", arg291_1: "f32[]", getitem_135: "f32[]", arg292_1: "f32[]", getitem_136: "f32[]", arg293_1: "f32[]", getitem_137: "f32[]", arg294_1: "f32[]", getitem_138: "f32[]", arg295_1: "f32[]", getitem_139: "f32[]", arg296_1: "f32[]", getitem_140: "f32[]", arg297_1: "f32[]", getitem_141: "f32[]", arg298_1: "f32[]", getitem_142: "f32[]", arg299_1: "f32[]", getitem_143: "f32[]", arg300_1: "f32[]", getitem_144: "f32[]", arg301_1: "f32[]", getitem_145: "f32[]", arg302_1: "f32[]", getitem_146: "f32[]", arg303_1: "f32[]", getitem_147: "f32[]", arg304_1: "f32[]", getitem_148: "f32[]", arg305_1: "f32[]", getitem_149: "f32[]", arg306_1: "f32[]", getitem_150: "f32[]", arg307_1: "f32[]", getitem_151: "f32[]", arg308_1: "f32[192, 3, 16, 16]", getitem_306: "f32[192, 3, 16, 16]", arg309_1: "f32[192]", getitem_307: "f32[192]", arg310_1: "f32[192]", getitem_308: "f32[192]", arg311_1: "f32[192]", getitem_309: "f32[192]", arg312_1: "f32[576, 192]", getitem_310: "f32[576, 192]", arg313_1: "f32[576]", getitem_311: "f32[576]", arg314_1: "f32[192, 192]", getitem_312: "f32[192, 192]", arg315_1: "f32[192]", getitem_313: "f32[192]", arg316_1: "f32[192]", getitem_314: "f32[192]", arg317_1: "f32[192]", getitem_315: "f32[192]", arg318_1: "f32[768, 192]", getitem_316: "f32[768, 192]", arg319_1: "f32[768]", getitem_317: "f32[768]", arg320_1: "f32[192, 768]", getitem_318: "f32[192, 768]", arg321_1: "f32[192]", getitem_319: "f32[192]", arg322_1: "f32[192]", getitem_320: "f32[192]", arg323_1: "f32[192]", getitem_321: "f32[192]", arg324_1: "f32[576, 192]", getitem_322: "f32[576, 192]", arg325_1: "f32[576]", getitem_323: "f32[576]", arg326_1: "f32[192, 192]", getitem_324: "f32[192, 192]", arg327_1: "f32[192]", getitem_325: "f32[192]", arg328_1: "f32[192]", getitem_326: "f32[192]", arg329_1: "f32[192]", getitem_327: "f32[192]", arg330_1: "f32[768, 192]", getitem_328: "f32[768, 192]", arg331_1: "f32[768]", getitem_329: "f32[768]", arg332_1: "f32[192, 768]", getitem_330: "f32[192, 768]", arg333_1: "f32[192]", getitem_331: "f32[192]", arg334_1: "f32[192]", getitem_332: "f32[192]", arg335_1: "f32[192]", getitem_333: "f32[192]", arg336_1: "f32[576, 192]", getitem_334: "f32[576, 192]", arg337_1: "f32[576]", getitem_335: "f32[576]", arg338_1: "f32[192, 192]", getitem_336: "f32[192, 192]", arg339_1: "f32[192]", getitem_337: "f32[192]", arg340_1: "f32[192]", getitem_338: "f32[192]", arg341_1: "f32[192]", getitem_339: "f32[192]", arg342_1: "f32[768, 192]", getitem_340: "f32[768, 192]", arg343_1: "f32[768]", getitem_341: "f32[768]", arg344_1: "f32[192, 768]", getitem_342: "f32[192, 768]", arg345_1: "f32[192]", getitem_343: "f32[192]", arg346_1: "f32[192]", getitem_344: "f32[192]", arg347_1: "f32[192]", getitem_345: "f32[192]", arg348_1: "f32[576, 192]", getitem_346: "f32[576, 192]", arg349_1: "f32[576]", getitem_347: "f32[576]", arg350_1: "f32[192, 192]", getitem_348: "f32[192, 192]", arg351_1: "f32[192]", getitem_349: "f32[192]", arg352_1: "f32[192]", getitem_350: "f32[192]", arg353_1: "f32[192]", getitem_351: "f32[192]", arg354_1: "f32[768, 192]", getitem_352: "f32[768, 192]", arg355_1: "f32[768]", getitem_353: "f32[768]", arg356_1: "f32[192, 768]", getitem_354: "f32[192, 768]", arg357_1: "f32[192]", getitem_355: "f32[192]", arg358_1: "f32[192]", getitem_356: "f32[192]", arg359_1: "f32[192]", getitem_357: "f32[192]", arg360_1: "f32[576, 192]", getitem_358: "f32[576, 192]", arg361_1: "f32[576]", getitem_359: "f32[576]", arg362_1: "f32[192, 192]", getitem_360: "f32[192, 192]", arg363_1: "f32[192]", getitem_361: "f32[192]", arg364_1: "f32[192]", getitem_362: "f32[192]", arg365_1: "f32[192]", getitem_363: "f32[192]", arg366_1: "f32[768, 192]", getitem_364: "f32[768, 192]", arg367_1: "f32[768]", getitem_365: "f32[768]", arg368_1: "f32[192, 768]", getitem_366: "f32[192, 768]", arg369_1: "f32[192]", getitem_367: "f32[192]", arg370_1: "f32[192]", getitem_368: "f32[192]", arg371_1: "f32[192]", getitem_369: "f32[192]", arg372_1: "f32[576, 192]", getitem_370: "f32[576, 192]", arg373_1: "f32[576]", getitem_371: "f32[576]", arg374_1: "f32[192, 192]", getitem_372: "f32[192, 192]", arg375_1: "f32[192]", getitem_373: "f32[192]", arg376_1: "f32[192]", getitem_374: "f32[192]", arg377_1: "f32[192]", getitem_375: "f32[192]", arg378_1: "f32[768, 192]", getitem_376: "f32[768, 192]", arg379_1: "f32[768]", getitem_377: "f32[768]", arg380_1: "f32[192, 768]", getitem_378: "f32[192, 768]", arg381_1: "f32[192]", getitem_379: "f32[192]", arg382_1: "f32[192]", getitem_380: "f32[192]", arg383_1: "f32[192]", getitem_381: "f32[192]", arg384_1: "f32[576, 192]", getitem_382: "f32[576, 192]", arg385_1: "f32[576]", getitem_383: "f32[576]", arg386_1: "f32[192, 192]", getitem_384: "f32[192, 192]", arg387_1: "f32[192]", getitem_385: "f32[192]", arg388_1: "f32[192]", getitem_386: "f32[192]", arg389_1: "f32[192]", getitem_387: "f32[192]", arg390_1: "f32[768, 192]", getitem_388: "f32[768, 192]", arg391_1: "f32[768]", getitem_389: "f32[768]", arg392_1: "f32[192, 768]", getitem_390: "f32[192, 768]", arg393_1: "f32[192]", getitem_391: "f32[192]", arg394_1: "f32[192]", getitem_392: "f32[192]", arg395_1: "f32[192]", getitem_393: "f32[192]", arg396_1: "f32[576, 192]", getitem_394: "f32[576, 192]", arg397_1: "f32[576]", getitem_395: "f32[576]", arg398_1: "f32[192, 192]", getitem_396: "f32[192, 192]", arg399_1: "f32[192]", getitem_397: "f32[192]", arg400_1: "f32[192]", getitem_398: "f32[192]", arg401_1: "f32[192]", getitem_399: "f32[192]", arg402_1: "f32[768, 192]", getitem_400: "f32[768, 192]", arg403_1: "f32[768]", getitem_401: "f32[768]", arg404_1: "f32[192, 768]", getitem_402: "f32[192, 768]", arg405_1: "f32[192]", getitem_403: "f32[192]", arg406_1: "f32[192]", getitem_404: "f32[192]", arg407_1: "f32[192]", getitem_405: "f32[192]", arg408_1: "f32[576, 192]", getitem_406: "f32[576, 192]", arg409_1: "f32[576]", getitem_407: "f32[576]", arg410_1: "f32[192, 192]", getitem_408: "f32[192, 192]", arg411_1: "f32[192]", getitem_409: "f32[192]", arg412_1: "f32[192]", getitem_410: "f32[192]", arg413_1: "f32[192]", getitem_411: "f32[192]", arg414_1: "f32[768, 192]", getitem_412: "f32[768, 192]", arg415_1: "f32[768]", getitem_413: "f32[768]", arg416_1: "f32[192, 768]", getitem_414: "f32[192, 768]", arg417_1: "f32[192]", getitem_415: "f32[192]", arg418_1: "f32[192]", getitem_416: "f32[192]", arg419_1: "f32[192]", getitem_417: "f32[192]", arg420_1: "f32[576, 192]", getitem_418: "f32[576, 192]", arg421_1: "f32[576]", getitem_419: "f32[576]", arg422_1: "f32[192, 192]", getitem_420: "f32[192, 192]", arg423_1: "f32[192]", getitem_421: "f32[192]", arg424_1: "f32[192]", getitem_422: "f32[192]", arg425_1: "f32[192]", getitem_423: "f32[192]", arg426_1: "f32[768, 192]", getitem_424: "f32[768, 192]", arg427_1: "f32[768]", getitem_425: "f32[768]", arg428_1: "f32[192, 768]", getitem_426: "f32[192, 768]", arg429_1: "f32[192]", getitem_427: "f32[192]", arg430_1: "f32[192]", getitem_428: "f32[192]", arg431_1: "f32[192]", getitem_429: "f32[192]", arg432_1: "f32[576, 192]", getitem_430: "f32[576, 192]", arg433_1: "f32[576]", getitem_431: "f32[576]", arg434_1: "f32[192, 192]", getitem_432: "f32[192, 192]", arg435_1: "f32[192]", getitem_433: "f32[192]", arg436_1: "f32[192]", getitem_434: "f32[192]", arg437_1: "f32[192]", getitem_435: "f32[192]", arg438_1: "f32[768, 192]", getitem_436: "f32[768, 192]", arg439_1: "f32[768]", getitem_437: "f32[768]", arg440_1: "f32[192, 768]", getitem_438: "f32[192, 768]", arg441_1: "f32[192]", getitem_439: "f32[192]", arg442_1: "f32[192]", getitem_440: "f32[192]", arg443_1: "f32[192]", getitem_441: "f32[192]", arg444_1: "f32[576, 192]", getitem_442: "f32[576, 192]", arg445_1: "f32[576]", getitem_443: "f32[576]", arg446_1: "f32[192, 192]", getitem_444: "f32[192, 192]", arg447_1: "f32[192]", getitem_445: "f32[192]", arg448_1: "f32[192]", getitem_446: "f32[192]", arg449_1: "f32[192]", getitem_447: "f32[192]", arg450_1: "f32[768, 192]", getitem_448: "f32[768, 192]", arg451_1: "f32[768]", getitem_449: "f32[768]", arg452_1: "f32[192, 768]", getitem_450: "f32[192, 768]", arg453_1: "f32[192]", getitem_451: "f32[192]", arg454_1: "f32[192]", getitem_452: "f32[192]", arg455_1: "f32[192]", getitem_453: "f32[192]", arg456_1: "f32[1000, 192]", getitem_454: "f32[1000, 192]", arg457_1: "f32[1000]", getitem_455: "f32[1000]", arg458_1: "f32[192, 3, 16, 16]", getitem_610: "f32[192, 3, 16, 16]", arg459_1: "f32[192]", getitem_611: "f32[192]", arg460_1: "f32[192]", getitem_612: "f32[192]", arg461_1: "f32[192]", getitem_613: "f32[192]", arg462_1: "f32[576, 192]", getitem_614: "f32[576, 192]", arg463_1: "f32[576]", getitem_615: "f32[576]", arg464_1: "f32[192, 192]", getitem_616: "f32[192, 192]", arg465_1: "f32[192]", getitem_617: "f32[192]", arg466_1: "f32[192]", getitem_618: "f32[192]", arg467_1: "f32[192]", getitem_619: "f32[192]", arg468_1: "f32[768, 192]", getitem_620: "f32[768, 192]", arg469_1: "f32[768]", getitem_621: "f32[768]", arg470_1: "f32[192, 768]", getitem_622: "f32[192, 768]", arg471_1: "f32[192]", getitem_623: "f32[192]", arg472_1: "f32[192]", getitem_624: "f32[192]", arg473_1: "f32[192]", getitem_625: "f32[192]", arg474_1: "f32[576, 192]", getitem_626: "f32[576, 192]", arg475_1: "f32[576]", getitem_627: "f32[576]", arg476_1: "f32[192, 192]", getitem_628: "f32[192, 192]", arg477_1: "f32[192]", getitem_629: "f32[192]", arg478_1: "f32[192]", getitem_630: "f32[192]", arg479_1: "f32[192]", getitem_631: "f32[192]", arg480_1: "f32[768, 192]", getitem_632: "f32[768, 192]", arg481_1: "f32[768]", getitem_633: "f32[768]", arg482_1: "f32[192, 768]", getitem_634: "f32[192, 768]", arg483_1: "f32[192]", getitem_635: "f32[192]", arg484_1: "f32[192]", getitem_636: "f32[192]", arg485_1: "f32[192]", getitem_637: "f32[192]", arg486_1: "f32[576, 192]", getitem_638: "f32[576, 192]", arg487_1: "f32[576]", getitem_639: "f32[576]", arg488_1: "f32[192, 192]", getitem_640: "f32[192, 192]", arg489_1: "f32[192]", getitem_641: "f32[192]", arg490_1: "f32[192]", getitem_642: "f32[192]", arg491_1: "f32[192]", getitem_643: "f32[192]", arg492_1: "f32[768, 192]", getitem_644: "f32[768, 192]", arg493_1: "f32[768]", getitem_645: "f32[768]", arg494_1: "f32[192, 768]", getitem_646: "f32[192, 768]", arg495_1: "f32[192]", getitem_647: "f32[192]", arg496_1: "f32[192]", getitem_648: "f32[192]", arg497_1: "f32[192]", getitem_649: "f32[192]", arg498_1: "f32[576, 192]", getitem_650: "f32[576, 192]", arg499_1: "f32[576]", getitem_651: "f32[576]", arg500_1: "f32[192, 192]", getitem_652: "f32[192, 192]", arg501_1: "f32[192]", getitem_653: "f32[192]", arg502_1: "f32[192]", getitem_654: "f32[192]", arg503_1: "f32[192]", getitem_655: "f32[192]", arg504_1: "f32[768, 192]", getitem_656: "f32[768, 192]", arg505_1: "f32[768]", getitem_657: "f32[768]", arg506_1: "f32[192, 768]", getitem_658: "f32[192, 768]", arg507_1: "f32[192]", getitem_659: "f32[192]", arg508_1: "f32[192]", getitem_660: "f32[192]", arg509_1: "f32[192]", getitem_661: "f32[192]", arg510_1: "f32[576, 192]", getitem_662: "f32[576, 192]", arg511_1: "f32[576]", getitem_663: "f32[576]", arg512_1: "f32[192, 192]", getitem_664: "f32[192, 192]", arg513_1: "f32[192]", getitem_665: "f32[192]", arg514_1: "f32[192]", getitem_666: "f32[192]", arg515_1: "f32[192]", getitem_667: "f32[192]", arg516_1: "f32[768, 192]", getitem_668: "f32[768, 192]", arg517_1: "f32[768]", getitem_669: "f32[768]", arg518_1: "f32[192, 768]", getitem_670: "f32[192, 768]", arg519_1: "f32[192]", getitem_671: "f32[192]", arg520_1: "f32[192]", getitem_672: "f32[192]", arg521_1: "f32[192]", getitem_673: "f32[192]", arg522_1: "f32[576, 192]", getitem_674: "f32[576, 192]", arg523_1: "f32[576]", getitem_675: "f32[576]", arg524_1: "f32[192, 192]", getitem_676: "f32[192, 192]", arg525_1: "f32[192]", getitem_677: "f32[192]", arg526_1: "f32[192]", getitem_678: "f32[192]", arg527_1: "f32[192]", getitem_679: "f32[192]", arg528_1: "f32[768, 192]", getitem_680: "f32[768, 192]", arg529_1: "f32[768]", getitem_681: "f32[768]", arg530_1: "f32[192, 768]", getitem_682: "f32[192, 768]", arg531_1: "f32[192]", getitem_683: "f32[192]", arg532_1: "f32[192]", getitem_684: "f32[192]", arg533_1: "f32[192]", getitem_685: "f32[192]", arg534_1: "f32[576, 192]", getitem_686: "f32[576, 192]", arg535_1: "f32[576]", getitem_687: "f32[576]", arg536_1: "f32[192, 192]", getitem_688: "f32[192, 192]", arg537_1: "f32[192]", getitem_689: "f32[192]", arg538_1: "f32[192]", getitem_690: "f32[192]", arg539_1: "f32[192]", getitem_691: "f32[192]", arg540_1: "f32[768, 192]", getitem_692: "f32[768, 192]", arg541_1: "f32[768]", getitem_693: "f32[768]", arg542_1: "f32[192, 768]", getitem_694: "f32[192, 768]", arg543_1: "f32[192]", getitem_695: "f32[192]", arg544_1: "f32[192]", getitem_696: "f32[192]", arg545_1: "f32[192]", getitem_697: "f32[192]", arg546_1: "f32[576, 192]", getitem_698: "f32[576, 192]", arg547_1: "f32[576]", getitem_699: "f32[576]", arg548_1: "f32[192, 192]", getitem_700: "f32[192, 192]", arg549_1: "f32[192]", getitem_701: "f32[192]", arg550_1: "f32[192]", getitem_702: "f32[192]", arg551_1: "f32[192]", getitem_703: "f32[192]", arg552_1: "f32[768, 192]", getitem_704: "f32[768, 192]", arg553_1: "f32[768]", getitem_705: "f32[768]", arg554_1: "f32[192, 768]", getitem_706: "f32[192, 768]", arg555_1: "f32[192]", getitem_707: "f32[192]", arg556_1: "f32[192]", getitem_708: "f32[192]", arg557_1: "f32[192]", getitem_709: "f32[192]", arg558_1: "f32[576, 192]", getitem_710: "f32[576, 192]", arg559_1: "f32[576]", getitem_711: "f32[576]", arg560_1: "f32[192, 192]", getitem_712: "f32[192, 192]", arg561_1: "f32[192]", getitem_713: "f32[192]", arg562_1: "f32[192]", getitem_714: "f32[192]", arg563_1: "f32[192]", getitem_715: "f32[192]", arg564_1: "f32[768, 192]", getitem_716: "f32[768, 192]", arg565_1: "f32[768]", getitem_717: "f32[768]", arg566_1: "f32[192, 768]", getitem_718: "f32[192, 768]", arg567_1: "f32[192]", getitem_719: "f32[192]", arg568_1: "f32[192]", getitem_720: "f32[192]", arg569_1: "f32[192]", getitem_721: "f32[192]", arg570_1: "f32[576, 192]", getitem_722: "f32[576, 192]", arg571_1: "f32[576]", getitem_723: "f32[576]", arg572_1: "f32[192, 192]", getitem_724: "f32[192, 192]", arg573_1: "f32[192]", getitem_725: "f32[192]", arg574_1: "f32[192]", getitem_726: "f32[192]", arg575_1: "f32[192]", getitem_727: "f32[192]", arg576_1: "f32[768, 192]", getitem_728: "f32[768, 192]", arg577_1: "f32[768]", getitem_729: "f32[768]", arg578_1: "f32[192, 768]", getitem_730: "f32[192, 768]", arg579_1: "f32[192]", getitem_731: "f32[192]", arg580_1: "f32[192]", getitem_732: "f32[192]", arg581_1: "f32[192]", getitem_733: "f32[192]", arg582_1: "f32[576, 192]", getitem_734: "f32[576, 192]", arg583_1: "f32[576]", getitem_735: "f32[576]", arg584_1: "f32[192, 192]", getitem_736: "f32[192, 192]", arg585_1: "f32[192]", getitem_737: "f32[192]", arg586_1: "f32[192]", getitem_738: "f32[192]", arg587_1: "f32[192]", getitem_739: "f32[192]", arg588_1: "f32[768, 192]", getitem_740: "f32[768, 192]", arg589_1: "f32[768]", getitem_741: "f32[768]", arg590_1: "f32[192, 768]", getitem_742: "f32[192, 768]", arg591_1: "f32[192]", getitem_743: "f32[192]", arg592_1: "f32[192]", getitem_744: "f32[192]", arg593_1: "f32[192]", getitem_745: "f32[192]", arg594_1: "f32[576, 192]", getitem_746: "f32[576, 192]", arg595_1: "f32[576]", getitem_747: "f32[576]", arg596_1: "f32[192, 192]", getitem_748: "f32[192, 192]", arg597_1: "f32[192]", getitem_749: "f32[192]", arg598_1: "f32[192]", getitem_750: "f32[192]", arg599_1: "f32[192]", getitem_751: "f32[192]", arg600_1: "f32[768, 192]", getitem_752: "f32[768, 192]", arg601_1: "f32[768]", getitem_753: "f32[768]", arg602_1: "f32[192, 768]", getitem_754: "f32[192, 768]", arg603_1: "f32[192]", getitem_755: "f32[192]", arg604_1: "f32[192]", getitem_756: "f32[192]", arg605_1: "f32[192]", getitem_757: "f32[192]", arg606_1: "f32[1000, 192]", getitem_758: "f32[1000, 192]", arg607_1: "f32[1000]", getitem_759: "f32[1000]"):
        # No stacktrace found for following nodes
        copy__default: "f32[1, 1, 192]" = torch.ops.aten.copy_.default(arg0_1, getitem_2584);  arg0_1 = getitem_2584 = None
        copy__default_1: "f32[1, 197, 192]" = torch.ops.aten.copy_.default(arg1_1, getitem_2585);  arg1_1 = getitem_2585 = None
        copy__default_2: "f32[192, 3, 16, 16]" = torch.ops.aten.copy_.default(arg2_1, getitem_2586);  arg2_1 = getitem_2586 = None
        copy__default_3: "f32[192]" = torch.ops.aten.copy_.default(arg3_1, getitem_2587);  arg3_1 = getitem_2587 = None
        copy__default_4: "f32[192]" = torch.ops.aten.copy_.default(arg4_1, getitem_2588);  arg4_1 = getitem_2588 = None
        copy__default_5: "f32[192]" = torch.ops.aten.copy_.default(arg5_1, getitem_2589);  arg5_1 = getitem_2589 = None
        copy__default_6: "f32[576, 192]" = torch.ops.aten.copy_.default(arg6_1, getitem_2590);  arg6_1 = getitem_2590 = None
        copy__default_7: "f32[576]" = torch.ops.aten.copy_.default(arg7_1, getitem_2591);  arg7_1 = getitem_2591 = None
        copy__default_8: "f32[192, 192]" = torch.ops.aten.copy_.default(arg8_1, getitem_2592);  arg8_1 = getitem_2592 = None
        copy__default_9: "f32[192]" = torch.ops.aten.copy_.default(arg9_1, getitem_2593);  arg9_1 = getitem_2593 = None
        copy__default_10: "f32[192]" = torch.ops.aten.copy_.default(arg10_1, getitem_2594);  arg10_1 = getitem_2594 = None
        copy__default_11: "f32[192]" = torch.ops.aten.copy_.default(arg11_1, getitem_2595);  arg11_1 = getitem_2595 = None
        copy__default_12: "f32[768, 192]" = torch.ops.aten.copy_.default(arg12_1, getitem_2596);  arg12_1 = getitem_2596 = None
        copy__default_13: "f32[768]" = torch.ops.aten.copy_.default(arg13_1, getitem_2597);  arg13_1 = getitem_2597 = None
        copy__default_14: "f32[192, 768]" = torch.ops.aten.copy_.default(arg14_1, getitem_2598);  arg14_1 = getitem_2598 = None
        copy__default_15: "f32[192]" = torch.ops.aten.copy_.default(arg15_1, getitem_2599);  arg15_1 = getitem_2599 = None
        copy__default_16: "f32[192]" = torch.ops.aten.copy_.default(arg16_1, getitem_2600);  arg16_1 = getitem_2600 = None
        copy__default_17: "f32[192]" = torch.ops.aten.copy_.default(arg17_1, getitem_2601);  arg17_1 = getitem_2601 = None
        copy__default_18: "f32[576, 192]" = torch.ops.aten.copy_.default(arg18_1, getitem_2602);  arg18_1 = getitem_2602 = None
        copy__default_19: "f32[576]" = torch.ops.aten.copy_.default(arg19_1, getitem_2603);  arg19_1 = getitem_2603 = None
        copy__default_20: "f32[192, 192]" = torch.ops.aten.copy_.default(arg20_1, getitem_2604);  arg20_1 = getitem_2604 = None
        copy__default_21: "f32[192]" = torch.ops.aten.copy_.default(arg21_1, getitem_2605);  arg21_1 = getitem_2605 = None
        copy__default_22: "f32[192]" = torch.ops.aten.copy_.default(arg22_1, getitem_2606);  arg22_1 = getitem_2606 = None
        copy__default_23: "f32[192]" = torch.ops.aten.copy_.default(arg23_1, getitem_2607);  arg23_1 = getitem_2607 = None
        copy__default_24: "f32[768, 192]" = torch.ops.aten.copy_.default(arg24_1, getitem_2608);  arg24_1 = getitem_2608 = None
        copy__default_25: "f32[768]" = torch.ops.aten.copy_.default(arg25_1, getitem_2609);  arg25_1 = getitem_2609 = None
        copy__default_26: "f32[192, 768]" = torch.ops.aten.copy_.default(arg26_1, getitem_2610);  arg26_1 = getitem_2610 = None
        copy__default_27: "f32[192]" = torch.ops.aten.copy_.default(arg27_1, getitem_2611);  arg27_1 = getitem_2611 = None
        copy__default_28: "f32[192]" = torch.ops.aten.copy_.default(arg28_1, getitem_2612);  arg28_1 = getitem_2612 = None
        copy__default_29: "f32[192]" = torch.ops.aten.copy_.default(arg29_1, getitem_2613);  arg29_1 = getitem_2613 = None
        copy__default_30: "f32[576, 192]" = torch.ops.aten.copy_.default(arg30_1, getitem_2614);  arg30_1 = getitem_2614 = None
        copy__default_31: "f32[576]" = torch.ops.aten.copy_.default(arg31_1, getitem_2615);  arg31_1 = getitem_2615 = None
        copy__default_32: "f32[192, 192]" = torch.ops.aten.copy_.default(arg32_1, getitem_2616);  arg32_1 = getitem_2616 = None
        copy__default_33: "f32[192]" = torch.ops.aten.copy_.default(arg33_1, getitem_2617);  arg33_1 = getitem_2617 = None
        copy__default_34: "f32[192]" = torch.ops.aten.copy_.default(arg34_1, getitem_2618);  arg34_1 = getitem_2618 = None
        copy__default_35: "f32[192]" = torch.ops.aten.copy_.default(arg35_1, getitem_2619);  arg35_1 = getitem_2619 = None
        copy__default_36: "f32[768, 192]" = torch.ops.aten.copy_.default(arg36_1, getitem_2620);  arg36_1 = getitem_2620 = None
        copy__default_37: "f32[768]" = torch.ops.aten.copy_.default(arg37_1, getitem_2621);  arg37_1 = getitem_2621 = None
        copy__default_38: "f32[192, 768]" = torch.ops.aten.copy_.default(arg38_1, getitem_2622);  arg38_1 = getitem_2622 = None
        copy__default_39: "f32[192]" = torch.ops.aten.copy_.default(arg39_1, getitem_2623);  arg39_1 = getitem_2623 = None
        copy__default_40: "f32[192]" = torch.ops.aten.copy_.default(arg40_1, getitem_2624);  arg40_1 = getitem_2624 = None
        copy__default_41: "f32[192]" = torch.ops.aten.copy_.default(arg41_1, getitem_2625);  arg41_1 = getitem_2625 = None
        copy__default_42: "f32[576, 192]" = torch.ops.aten.copy_.default(arg42_1, getitem_2626);  arg42_1 = getitem_2626 = None
        copy__default_43: "f32[576]" = torch.ops.aten.copy_.default(arg43_1, getitem_2627);  arg43_1 = getitem_2627 = None
        copy__default_44: "f32[192, 192]" = torch.ops.aten.copy_.default(arg44_1, getitem_2628);  arg44_1 = getitem_2628 = None
        copy__default_45: "f32[192]" = torch.ops.aten.copy_.default(arg45_1, getitem_2629);  arg45_1 = getitem_2629 = None
        copy__default_46: "f32[192]" = torch.ops.aten.copy_.default(arg46_1, getitem_2630);  arg46_1 = getitem_2630 = None
        copy__default_47: "f32[192]" = torch.ops.aten.copy_.default(arg47_1, getitem_2631);  arg47_1 = getitem_2631 = None
        copy__default_48: "f32[768, 192]" = torch.ops.aten.copy_.default(arg48_1, getitem_2632);  arg48_1 = getitem_2632 = None
        copy__default_49: "f32[768]" = torch.ops.aten.copy_.default(arg49_1, getitem_2633);  arg49_1 = getitem_2633 = None
        copy__default_50: "f32[192, 768]" = torch.ops.aten.copy_.default(arg50_1, getitem_2634);  arg50_1 = getitem_2634 = None
        copy__default_51: "f32[192]" = torch.ops.aten.copy_.default(arg51_1, getitem_2635);  arg51_1 = getitem_2635 = None
        copy__default_52: "f32[192]" = torch.ops.aten.copy_.default(arg52_1, getitem_2636);  arg52_1 = getitem_2636 = None
        copy__default_53: "f32[192]" = torch.ops.aten.copy_.default(arg53_1, getitem_2637);  arg53_1 = getitem_2637 = None
        copy__default_54: "f32[576, 192]" = torch.ops.aten.copy_.default(arg54_1, getitem_2638);  arg54_1 = getitem_2638 = None
        copy__default_55: "f32[576]" = torch.ops.aten.copy_.default(arg55_1, getitem_2639);  arg55_1 = getitem_2639 = None
        copy__default_56: "f32[192, 192]" = torch.ops.aten.copy_.default(arg56_1, getitem_2640);  arg56_1 = getitem_2640 = None
        copy__default_57: "f32[192]" = torch.ops.aten.copy_.default(arg57_1, getitem_2641);  arg57_1 = getitem_2641 = None
        copy__default_58: "f32[192]" = torch.ops.aten.copy_.default(arg58_1, getitem_2642);  arg58_1 = getitem_2642 = None
        copy__default_59: "f32[192]" = torch.ops.aten.copy_.default(arg59_1, getitem_2643);  arg59_1 = getitem_2643 = None
        copy__default_60: "f32[768, 192]" = torch.ops.aten.copy_.default(arg60_1, getitem_2644);  arg60_1 = getitem_2644 = None
        copy__default_61: "f32[768]" = torch.ops.aten.copy_.default(arg61_1, getitem_2645);  arg61_1 = getitem_2645 = None
        copy__default_62: "f32[192, 768]" = torch.ops.aten.copy_.default(arg62_1, getitem_2646);  arg62_1 = getitem_2646 = None
        copy__default_63: "f32[192]" = torch.ops.aten.copy_.default(arg63_1, getitem_2647);  arg63_1 = getitem_2647 = None
        copy__default_64: "f32[192]" = torch.ops.aten.copy_.default(arg64_1, getitem_2648);  arg64_1 = getitem_2648 = None
        copy__default_65: "f32[192]" = torch.ops.aten.copy_.default(arg65_1, getitem_2649);  arg65_1 = getitem_2649 = None
        copy__default_66: "f32[576, 192]" = torch.ops.aten.copy_.default(arg66_1, getitem_2650);  arg66_1 = getitem_2650 = None
        copy__default_67: "f32[576]" = torch.ops.aten.copy_.default(arg67_1, getitem_2651);  arg67_1 = getitem_2651 = None
        copy__default_68: "f32[192, 192]" = torch.ops.aten.copy_.default(arg68_1, getitem_2652);  arg68_1 = getitem_2652 = None
        copy__default_69: "f32[192]" = torch.ops.aten.copy_.default(arg69_1, getitem_2653);  arg69_1 = getitem_2653 = None
        copy__default_70: "f32[192]" = torch.ops.aten.copy_.default(arg70_1, getitem_2654);  arg70_1 = getitem_2654 = None
        copy__default_71: "f32[192]" = torch.ops.aten.copy_.default(arg71_1, getitem_2655);  arg71_1 = getitem_2655 = None
        copy__default_72: "f32[768, 192]" = torch.ops.aten.copy_.default(arg72_1, getitem_2656);  arg72_1 = getitem_2656 = None
        copy__default_73: "f32[768]" = torch.ops.aten.copy_.default(arg73_1, getitem_2657);  arg73_1 = getitem_2657 = None
        copy__default_74: "f32[192, 768]" = torch.ops.aten.copy_.default(arg74_1, getitem_2658);  arg74_1 = getitem_2658 = None
        copy__default_75: "f32[192]" = torch.ops.aten.copy_.default(arg75_1, getitem_2659);  arg75_1 = getitem_2659 = None
        copy__default_76: "f32[192]" = torch.ops.aten.copy_.default(arg76_1, getitem_2660);  arg76_1 = getitem_2660 = None
        copy__default_77: "f32[192]" = torch.ops.aten.copy_.default(arg77_1, getitem_2661);  arg77_1 = getitem_2661 = None
        copy__default_78: "f32[576, 192]" = torch.ops.aten.copy_.default(arg78_1, getitem_2662);  arg78_1 = getitem_2662 = None
        copy__default_79: "f32[576]" = torch.ops.aten.copy_.default(arg79_1, getitem_2663);  arg79_1 = getitem_2663 = None
        copy__default_80: "f32[192, 192]" = torch.ops.aten.copy_.default(arg80_1, getitem_2664);  arg80_1 = getitem_2664 = None
        copy__default_81: "f32[192]" = torch.ops.aten.copy_.default(arg81_1, getitem_2665);  arg81_1 = getitem_2665 = None
        copy__default_82: "f32[192]" = torch.ops.aten.copy_.default(arg82_1, getitem_2666);  arg82_1 = getitem_2666 = None
        copy__default_83: "f32[192]" = torch.ops.aten.copy_.default(arg83_1, getitem_2667);  arg83_1 = getitem_2667 = None
        copy__default_84: "f32[768, 192]" = torch.ops.aten.copy_.default(arg84_1, getitem_2668);  arg84_1 = getitem_2668 = None
        copy__default_85: "f32[768]" = torch.ops.aten.copy_.default(arg85_1, getitem_2669);  arg85_1 = getitem_2669 = None
        copy__default_86: "f32[192, 768]" = torch.ops.aten.copy_.default(arg86_1, getitem_2670);  arg86_1 = getitem_2670 = None
        copy__default_87: "f32[192]" = torch.ops.aten.copy_.default(arg87_1, getitem_2671);  arg87_1 = getitem_2671 = None
        copy__default_88: "f32[192]" = torch.ops.aten.copy_.default(arg88_1, getitem_2672);  arg88_1 = getitem_2672 = None
        copy__default_89: "f32[192]" = torch.ops.aten.copy_.default(arg89_1, getitem_2673);  arg89_1 = getitem_2673 = None
        copy__default_90: "f32[576, 192]" = torch.ops.aten.copy_.default(arg90_1, getitem_2674);  arg90_1 = getitem_2674 = None
        copy__default_91: "f32[576]" = torch.ops.aten.copy_.default(arg91_1, getitem_2675);  arg91_1 = getitem_2675 = None
        copy__default_92: "f32[192, 192]" = torch.ops.aten.copy_.default(arg92_1, getitem_2676);  arg92_1 = getitem_2676 = None
        copy__default_93: "f32[192]" = torch.ops.aten.copy_.default(arg93_1, getitem_2677);  arg93_1 = getitem_2677 = None
        copy__default_94: "f32[192]" = torch.ops.aten.copy_.default(arg94_1, getitem_2678);  arg94_1 = getitem_2678 = None
        copy__default_95: "f32[192]" = torch.ops.aten.copy_.default(arg95_1, getitem_2679);  arg95_1 = getitem_2679 = None
        copy__default_96: "f32[768, 192]" = torch.ops.aten.copy_.default(arg96_1, getitem_2680);  arg96_1 = getitem_2680 = None
        copy__default_97: "f32[768]" = torch.ops.aten.copy_.default(arg97_1, getitem_2681);  arg97_1 = getitem_2681 = None
        copy__default_98: "f32[192, 768]" = torch.ops.aten.copy_.default(arg98_1, getitem_2682);  arg98_1 = getitem_2682 = None
        copy__default_99: "f32[192]" = torch.ops.aten.copy_.default(arg99_1, getitem_2683);  arg99_1 = getitem_2683 = None
        copy__default_100: "f32[192]" = torch.ops.aten.copy_.default(arg100_1, getitem_2684);  arg100_1 = getitem_2684 = None
        copy__default_101: "f32[192]" = torch.ops.aten.copy_.default(arg101_1, getitem_2685);  arg101_1 = getitem_2685 = None
        copy__default_102: "f32[576, 192]" = torch.ops.aten.copy_.default(arg102_1, getitem_2686);  arg102_1 = getitem_2686 = None
        copy__default_103: "f32[576]" = torch.ops.aten.copy_.default(arg103_1, getitem_2687);  arg103_1 = getitem_2687 = None
        copy__default_104: "f32[192, 192]" = torch.ops.aten.copy_.default(arg104_1, getitem_2688);  arg104_1 = getitem_2688 = None
        copy__default_105: "f32[192]" = torch.ops.aten.copy_.default(arg105_1, getitem_2689);  arg105_1 = getitem_2689 = None
        copy__default_106: "f32[192]" = torch.ops.aten.copy_.default(arg106_1, getitem_2690);  arg106_1 = getitem_2690 = None
        copy__default_107: "f32[192]" = torch.ops.aten.copy_.default(arg107_1, getitem_2691);  arg107_1 = getitem_2691 = None
        copy__default_108: "f32[768, 192]" = torch.ops.aten.copy_.default(arg108_1, getitem_2692);  arg108_1 = getitem_2692 = None
        copy__default_109: "f32[768]" = torch.ops.aten.copy_.default(arg109_1, getitem_2693);  arg109_1 = getitem_2693 = None
        copy__default_110: "f32[192, 768]" = torch.ops.aten.copy_.default(arg110_1, getitem_2694);  arg110_1 = getitem_2694 = None
        copy__default_111: "f32[192]" = torch.ops.aten.copy_.default(arg111_1, getitem_2695);  arg111_1 = getitem_2695 = None
        copy__default_112: "f32[192]" = torch.ops.aten.copy_.default(arg112_1, getitem_2696);  arg112_1 = getitem_2696 = None
        copy__default_113: "f32[192]" = torch.ops.aten.copy_.default(arg113_1, getitem_2697);  arg113_1 = getitem_2697 = None
        copy__default_114: "f32[576, 192]" = torch.ops.aten.copy_.default(arg114_1, getitem_2698);  arg114_1 = getitem_2698 = None
        copy__default_115: "f32[576]" = torch.ops.aten.copy_.default(arg115_1, getitem_2699);  arg115_1 = getitem_2699 = None
        copy__default_116: "f32[192, 192]" = torch.ops.aten.copy_.default(arg116_1, getitem_2700);  arg116_1 = getitem_2700 = None
        copy__default_117: "f32[192]" = torch.ops.aten.copy_.default(arg117_1, getitem_2701);  arg117_1 = getitem_2701 = None
        copy__default_118: "f32[192]" = torch.ops.aten.copy_.default(arg118_1, getitem_2702);  arg118_1 = getitem_2702 = None
        copy__default_119: "f32[192]" = torch.ops.aten.copy_.default(arg119_1, getitem_2703);  arg119_1 = getitem_2703 = None
        copy__default_120: "f32[768, 192]" = torch.ops.aten.copy_.default(arg120_1, getitem_2704);  arg120_1 = getitem_2704 = None
        copy__default_121: "f32[768]" = torch.ops.aten.copy_.default(arg121_1, getitem_2705);  arg121_1 = getitem_2705 = None
        copy__default_122: "f32[192, 768]" = torch.ops.aten.copy_.default(arg122_1, getitem_2706);  arg122_1 = getitem_2706 = None
        copy__default_123: "f32[192]" = torch.ops.aten.copy_.default(arg123_1, getitem_2707);  arg123_1 = getitem_2707 = None
        copy__default_124: "f32[192]" = torch.ops.aten.copy_.default(arg124_1, getitem_2708);  arg124_1 = getitem_2708 = None
        copy__default_125: "f32[192]" = torch.ops.aten.copy_.default(arg125_1, getitem_2709);  arg125_1 = getitem_2709 = None
        copy__default_126: "f32[576, 192]" = torch.ops.aten.copy_.default(arg126_1, getitem_2710);  arg126_1 = getitem_2710 = None
        copy__default_127: "f32[576]" = torch.ops.aten.copy_.default(arg127_1, getitem_2711);  arg127_1 = getitem_2711 = None
        copy__default_128: "f32[192, 192]" = torch.ops.aten.copy_.default(arg128_1, getitem_2712);  arg128_1 = getitem_2712 = None
        copy__default_129: "f32[192]" = torch.ops.aten.copy_.default(arg129_1, getitem_2713);  arg129_1 = getitem_2713 = None
        copy__default_130: "f32[192]" = torch.ops.aten.copy_.default(arg130_1, getitem_2714);  arg130_1 = getitem_2714 = None
        copy__default_131: "f32[192]" = torch.ops.aten.copy_.default(arg131_1, getitem_2715);  arg131_1 = getitem_2715 = None
        copy__default_132: "f32[768, 192]" = torch.ops.aten.copy_.default(arg132_1, getitem_2716);  arg132_1 = getitem_2716 = None
        copy__default_133: "f32[768]" = torch.ops.aten.copy_.default(arg133_1, getitem_2717);  arg133_1 = getitem_2717 = None
        copy__default_134: "f32[192, 768]" = torch.ops.aten.copy_.default(arg134_1, getitem_2718);  arg134_1 = getitem_2718 = None
        copy__default_135: "f32[192]" = torch.ops.aten.copy_.default(arg135_1, getitem_2719);  arg135_1 = getitem_2719 = None
        copy__default_136: "f32[192]" = torch.ops.aten.copy_.default(arg136_1, getitem_2720);  arg136_1 = getitem_2720 = None
        copy__default_137: "f32[192]" = torch.ops.aten.copy_.default(arg137_1, getitem_2721);  arg137_1 = getitem_2721 = None
        copy__default_138: "f32[576, 192]" = torch.ops.aten.copy_.default(arg138_1, getitem_2722);  arg138_1 = getitem_2722 = None
        copy__default_139: "f32[576]" = torch.ops.aten.copy_.default(arg139_1, getitem_2723);  arg139_1 = getitem_2723 = None
        copy__default_140: "f32[192, 192]" = torch.ops.aten.copy_.default(arg140_1, getitem_2724);  arg140_1 = getitem_2724 = None
        copy__default_141: "f32[192]" = torch.ops.aten.copy_.default(arg141_1, getitem_2725);  arg141_1 = getitem_2725 = None
        copy__default_142: "f32[192]" = torch.ops.aten.copy_.default(arg142_1, getitem_2726);  arg142_1 = getitem_2726 = None
        copy__default_143: "f32[192]" = torch.ops.aten.copy_.default(arg143_1, getitem_2727);  arg143_1 = getitem_2727 = None
        copy__default_144: "f32[768, 192]" = torch.ops.aten.copy_.default(arg144_1, getitem_2728);  arg144_1 = getitem_2728 = None
        copy__default_145: "f32[768]" = torch.ops.aten.copy_.default(arg145_1, getitem_2729);  arg145_1 = getitem_2729 = None
        copy__default_146: "f32[192, 768]" = torch.ops.aten.copy_.default(arg146_1, getitem_2730);  arg146_1 = getitem_2730 = None
        copy__default_147: "f32[192]" = torch.ops.aten.copy_.default(arg147_1, getitem_2731);  arg147_1 = getitem_2731 = None
        copy__default_148: "f32[192]" = torch.ops.aten.copy_.default(arg148_1, getitem_2732);  arg148_1 = getitem_2732 = None
        copy__default_149: "f32[192]" = torch.ops.aten.copy_.default(arg149_1, getitem_2733);  arg149_1 = getitem_2733 = None
        copy__default_150: "f32[1000, 192]" = torch.ops.aten.copy_.default(arg150_1, getitem_2734);  arg150_1 = getitem_2734 = None
        copy__default_151: "f32[1000]" = torch.ops.aten.copy_.default(arg151_1, getitem_2735);  arg151_1 = getitem_2735 = None
        copy__default_152: "f32[]" = torch.ops.aten.copy_.default(arg152_1, getitem_1);  arg152_1 = getitem_1 = None
        copy__default_153: "f32[1, 197, 192]" = torch.ops.aten.copy_.default(arg153_1, getitem_305);  arg153_1 = getitem_305 = None
        copy__default_154: "f32[1, 197, 192]" = torch.ops.aten.copy_.default(arg154_1, getitem_609);  arg154_1 = getitem_609 = None
        copy__default_155: "f32[1, 1, 192]" = torch.ops.aten.copy_.default(arg155_1, getitem_304);  arg155_1 = getitem_304 = None
        copy__default_156: "f32[1, 1, 192]" = torch.ops.aten.copy_.default(arg156_1, getitem_608);  arg156_1 = getitem_608 = None
        copy__default_157: "f32[]" = torch.ops.aten.copy_.default(arg157_1, getitem);  arg157_1 = getitem = None
        copy__default_158: "f32[]" = torch.ops.aten.copy_.default(arg158_1, getitem_2);  arg158_1 = getitem_2 = None
        copy__default_159: "f32[]" = torch.ops.aten.copy_.default(arg159_1, getitem_3);  arg159_1 = getitem_3 = None
        copy__default_160: "f32[]" = torch.ops.aten.copy_.default(arg160_1, getitem_4);  arg160_1 = getitem_4 = None
        copy__default_161: "f32[]" = torch.ops.aten.copy_.default(arg161_1, getitem_5);  arg161_1 = getitem_5 = None
        copy__default_162: "f32[]" = torch.ops.aten.copy_.default(arg162_1, getitem_6);  arg162_1 = getitem_6 = None
        copy__default_163: "f32[]" = torch.ops.aten.copy_.default(arg163_1, getitem_7);  arg163_1 = getitem_7 = None
        copy__default_164: "f32[]" = torch.ops.aten.copy_.default(arg164_1, getitem_8);  arg164_1 = getitem_8 = None
        copy__default_165: "f32[]" = torch.ops.aten.copy_.default(arg165_1, getitem_9);  arg165_1 = getitem_9 = None
        copy__default_166: "f32[]" = torch.ops.aten.copy_.default(arg166_1, getitem_10);  arg166_1 = getitem_10 = None
        copy__default_167: "f32[]" = torch.ops.aten.copy_.default(arg167_1, getitem_11);  arg167_1 = getitem_11 = None
        copy__default_168: "f32[]" = torch.ops.aten.copy_.default(arg168_1, getitem_12);  arg168_1 = getitem_12 = None
        copy__default_169: "f32[]" = torch.ops.aten.copy_.default(arg169_1, getitem_13);  arg169_1 = getitem_13 = None
        copy__default_170: "f32[]" = torch.ops.aten.copy_.default(arg170_1, getitem_14);  arg170_1 = getitem_14 = None
        copy__default_171: "f32[]" = torch.ops.aten.copy_.default(arg171_1, getitem_15);  arg171_1 = getitem_15 = None
        copy__default_172: "f32[]" = torch.ops.aten.copy_.default(arg172_1, getitem_16);  arg172_1 = getitem_16 = None
        copy__default_173: "f32[]" = torch.ops.aten.copy_.default(arg173_1, getitem_17);  arg173_1 = getitem_17 = None
        copy__default_174: "f32[]" = torch.ops.aten.copy_.default(arg174_1, getitem_18);  arg174_1 = getitem_18 = None
        copy__default_175: "f32[]" = torch.ops.aten.copy_.default(arg175_1, getitem_19);  arg175_1 = getitem_19 = None
        copy__default_176: "f32[]" = torch.ops.aten.copy_.default(arg176_1, getitem_20);  arg176_1 = getitem_20 = None
        copy__default_177: "f32[]" = torch.ops.aten.copy_.default(arg177_1, getitem_21);  arg177_1 = getitem_21 = None
        copy__default_178: "f32[]" = torch.ops.aten.copy_.default(arg178_1, getitem_22);  arg178_1 = getitem_22 = None
        copy__default_179: "f32[]" = torch.ops.aten.copy_.default(arg179_1, getitem_23);  arg179_1 = getitem_23 = None
        copy__default_180: "f32[]" = torch.ops.aten.copy_.default(arg180_1, getitem_24);  arg180_1 = getitem_24 = None
        copy__default_181: "f32[]" = torch.ops.aten.copy_.default(arg181_1, getitem_25);  arg181_1 = getitem_25 = None
        copy__default_182: "f32[]" = torch.ops.aten.copy_.default(arg182_1, getitem_26);  arg182_1 = getitem_26 = None
        copy__default_183: "f32[]" = torch.ops.aten.copy_.default(arg183_1, getitem_27);  arg183_1 = getitem_27 = None
        copy__default_184: "f32[]" = torch.ops.aten.copy_.default(arg184_1, getitem_28);  arg184_1 = getitem_28 = None
        copy__default_185: "f32[]" = torch.ops.aten.copy_.default(arg185_1, getitem_29);  arg185_1 = getitem_29 = None
        copy__default_186: "f32[]" = torch.ops.aten.copy_.default(arg186_1, getitem_30);  arg186_1 = getitem_30 = None
        copy__default_187: "f32[]" = torch.ops.aten.copy_.default(arg187_1, getitem_31);  arg187_1 = getitem_31 = None
        copy__default_188: "f32[]" = torch.ops.aten.copy_.default(arg188_1, getitem_32);  arg188_1 = getitem_32 = None
        copy__default_189: "f32[]" = torch.ops.aten.copy_.default(arg189_1, getitem_33);  arg189_1 = getitem_33 = None
        copy__default_190: "f32[]" = torch.ops.aten.copy_.default(arg190_1, getitem_34);  arg190_1 = getitem_34 = None
        copy__default_191: "f32[]" = torch.ops.aten.copy_.default(arg191_1, getitem_35);  arg191_1 = getitem_35 = None
        copy__default_192: "f32[]" = torch.ops.aten.copy_.default(arg192_1, getitem_36);  arg192_1 = getitem_36 = None
        copy__default_193: "f32[]" = torch.ops.aten.copy_.default(arg193_1, getitem_37);  arg193_1 = getitem_37 = None
        copy__default_194: "f32[]" = torch.ops.aten.copy_.default(arg194_1, getitem_38);  arg194_1 = getitem_38 = None
        copy__default_195: "f32[]" = torch.ops.aten.copy_.default(arg195_1, getitem_39);  arg195_1 = getitem_39 = None
        copy__default_196: "f32[]" = torch.ops.aten.copy_.default(arg196_1, getitem_40);  arg196_1 = getitem_40 = None
        copy__default_197: "f32[]" = torch.ops.aten.copy_.default(arg197_1, getitem_41);  arg197_1 = getitem_41 = None
        copy__default_198: "f32[]" = torch.ops.aten.copy_.default(arg198_1, getitem_42);  arg198_1 = getitem_42 = None
        copy__default_199: "f32[]" = torch.ops.aten.copy_.default(arg199_1, getitem_43);  arg199_1 = getitem_43 = None
        copy__default_200: "f32[]" = torch.ops.aten.copy_.default(arg200_1, getitem_44);  arg200_1 = getitem_44 = None
        copy__default_201: "f32[]" = torch.ops.aten.copy_.default(arg201_1, getitem_45);  arg201_1 = getitem_45 = None
        copy__default_202: "f32[]" = torch.ops.aten.copy_.default(arg202_1, getitem_46);  arg202_1 = getitem_46 = None
        copy__default_203: "f32[]" = torch.ops.aten.copy_.default(arg203_1, getitem_47);  arg203_1 = getitem_47 = None
        copy__default_204: "f32[]" = torch.ops.aten.copy_.default(arg204_1, getitem_48);  arg204_1 = getitem_48 = None
        copy__default_205: "f32[]" = torch.ops.aten.copy_.default(arg205_1, getitem_49);  arg205_1 = getitem_49 = None
        copy__default_206: "f32[]" = torch.ops.aten.copy_.default(arg206_1, getitem_50);  arg206_1 = getitem_50 = None
        copy__default_207: "f32[]" = torch.ops.aten.copy_.default(arg207_1, getitem_51);  arg207_1 = getitem_51 = None
        copy__default_208: "f32[]" = torch.ops.aten.copy_.default(arg208_1, getitem_52);  arg208_1 = getitem_52 = None
        copy__default_209: "f32[]" = torch.ops.aten.copy_.default(arg209_1, getitem_53);  arg209_1 = getitem_53 = None
        copy__default_210: "f32[]" = torch.ops.aten.copy_.default(arg210_1, getitem_54);  arg210_1 = getitem_54 = None
        copy__default_211: "f32[]" = torch.ops.aten.copy_.default(arg211_1, getitem_55);  arg211_1 = getitem_55 = None
        copy__default_212: "f32[]" = torch.ops.aten.copy_.default(arg212_1, getitem_56);  arg212_1 = getitem_56 = None
        copy__default_213: "f32[]" = torch.ops.aten.copy_.default(arg213_1, getitem_57);  arg213_1 = getitem_57 = None
        copy__default_214: "f32[]" = torch.ops.aten.copy_.default(arg214_1, getitem_58);  arg214_1 = getitem_58 = None
        copy__default_215: "f32[]" = torch.ops.aten.copy_.default(arg215_1, getitem_59);  arg215_1 = getitem_59 = None
        copy__default_216: "f32[]" = torch.ops.aten.copy_.default(arg216_1, getitem_60);  arg216_1 = getitem_60 = None
        copy__default_217: "f32[]" = torch.ops.aten.copy_.default(arg217_1, getitem_61);  arg217_1 = getitem_61 = None
        copy__default_218: "f32[]" = torch.ops.aten.copy_.default(arg218_1, getitem_62);  arg218_1 = getitem_62 = None
        copy__default_219: "f32[]" = torch.ops.aten.copy_.default(arg219_1, getitem_63);  arg219_1 = getitem_63 = None
        copy__default_220: "f32[]" = torch.ops.aten.copy_.default(arg220_1, getitem_64);  arg220_1 = getitem_64 = None
        copy__default_221: "f32[]" = torch.ops.aten.copy_.default(arg221_1, getitem_65);  arg221_1 = getitem_65 = None
        copy__default_222: "f32[]" = torch.ops.aten.copy_.default(arg222_1, getitem_66);  arg222_1 = getitem_66 = None
        copy__default_223: "f32[]" = torch.ops.aten.copy_.default(arg223_1, getitem_67);  arg223_1 = getitem_67 = None
        copy__default_224: "f32[]" = torch.ops.aten.copy_.default(arg224_1, getitem_68);  arg224_1 = getitem_68 = None
        copy__default_225: "f32[]" = torch.ops.aten.copy_.default(arg225_1, getitem_69);  arg225_1 = getitem_69 = None
        copy__default_226: "f32[]" = torch.ops.aten.copy_.default(arg226_1, getitem_70);  arg226_1 = getitem_70 = None
        copy__default_227: "f32[]" = torch.ops.aten.copy_.default(arg227_1, getitem_71);  arg227_1 = getitem_71 = None
        copy__default_228: "f32[]" = torch.ops.aten.copy_.default(arg228_1, getitem_72);  arg228_1 = getitem_72 = None
        copy__default_229: "f32[]" = torch.ops.aten.copy_.default(arg229_1, getitem_73);  arg229_1 = getitem_73 = None
        copy__default_230: "f32[]" = torch.ops.aten.copy_.default(arg230_1, getitem_74);  arg230_1 = getitem_74 = None
        copy__default_231: "f32[]" = torch.ops.aten.copy_.default(arg231_1, getitem_75);  arg231_1 = getitem_75 = None
        copy__default_232: "f32[]" = torch.ops.aten.copy_.default(arg232_1, getitem_76);  arg232_1 = getitem_76 = None
        copy__default_233: "f32[]" = torch.ops.aten.copy_.default(arg233_1, getitem_77);  arg233_1 = getitem_77 = None
        copy__default_234: "f32[]" = torch.ops.aten.copy_.default(arg234_1, getitem_78);  arg234_1 = getitem_78 = None
        copy__default_235: "f32[]" = torch.ops.aten.copy_.default(arg235_1, getitem_79);  arg235_1 = getitem_79 = None
        copy__default_236: "f32[]" = torch.ops.aten.copy_.default(arg236_1, getitem_80);  arg236_1 = getitem_80 = None
        copy__default_237: "f32[]" = torch.ops.aten.copy_.default(arg237_1, getitem_81);  arg237_1 = getitem_81 = None
        copy__default_238: "f32[]" = torch.ops.aten.copy_.default(arg238_1, getitem_82);  arg238_1 = getitem_82 = None
        copy__default_239: "f32[]" = torch.ops.aten.copy_.default(arg239_1, getitem_83);  arg239_1 = getitem_83 = None
        copy__default_240: "f32[]" = torch.ops.aten.copy_.default(arg240_1, getitem_84);  arg240_1 = getitem_84 = None
        copy__default_241: "f32[]" = torch.ops.aten.copy_.default(arg241_1, getitem_85);  arg241_1 = getitem_85 = None
        copy__default_242: "f32[]" = torch.ops.aten.copy_.default(arg242_1, getitem_86);  arg242_1 = getitem_86 = None
        copy__default_243: "f32[]" = torch.ops.aten.copy_.default(arg243_1, getitem_87);  arg243_1 = getitem_87 = None
        copy__default_244: "f32[]" = torch.ops.aten.copy_.default(arg244_1, getitem_88);  arg244_1 = getitem_88 = None
        copy__default_245: "f32[]" = torch.ops.aten.copy_.default(arg245_1, getitem_89);  arg245_1 = getitem_89 = None
        copy__default_246: "f32[]" = torch.ops.aten.copy_.default(arg246_1, getitem_90);  arg246_1 = getitem_90 = None
        copy__default_247: "f32[]" = torch.ops.aten.copy_.default(arg247_1, getitem_91);  arg247_1 = getitem_91 = None
        copy__default_248: "f32[]" = torch.ops.aten.copy_.default(arg248_1, getitem_92);  arg248_1 = getitem_92 = None
        copy__default_249: "f32[]" = torch.ops.aten.copy_.default(arg249_1, getitem_93);  arg249_1 = getitem_93 = None
        copy__default_250: "f32[]" = torch.ops.aten.copy_.default(arg250_1, getitem_94);  arg250_1 = getitem_94 = None
        copy__default_251: "f32[]" = torch.ops.aten.copy_.default(arg251_1, getitem_95);  arg251_1 = getitem_95 = None
        copy__default_252: "f32[]" = torch.ops.aten.copy_.default(arg252_1, getitem_96);  arg252_1 = getitem_96 = None
        copy__default_253: "f32[]" = torch.ops.aten.copy_.default(arg253_1, getitem_97);  arg253_1 = getitem_97 = None
        copy__default_254: "f32[]" = torch.ops.aten.copy_.default(arg254_1, getitem_98);  arg254_1 = getitem_98 = None
        copy__default_255: "f32[]" = torch.ops.aten.copy_.default(arg255_1, getitem_99);  arg255_1 = getitem_99 = None
        copy__default_256: "f32[]" = torch.ops.aten.copy_.default(arg256_1, getitem_100);  arg256_1 = getitem_100 = None
        copy__default_257: "f32[]" = torch.ops.aten.copy_.default(arg257_1, getitem_101);  arg257_1 = getitem_101 = None
        copy__default_258: "f32[]" = torch.ops.aten.copy_.default(arg258_1, getitem_102);  arg258_1 = getitem_102 = None
        copy__default_259: "f32[]" = torch.ops.aten.copy_.default(arg259_1, getitem_103);  arg259_1 = getitem_103 = None
        copy__default_260: "f32[]" = torch.ops.aten.copy_.default(arg260_1, getitem_104);  arg260_1 = getitem_104 = None
        copy__default_261: "f32[]" = torch.ops.aten.copy_.default(arg261_1, getitem_105);  arg261_1 = getitem_105 = None
        copy__default_262: "f32[]" = torch.ops.aten.copy_.default(arg262_1, getitem_106);  arg262_1 = getitem_106 = None
        copy__default_263: "f32[]" = torch.ops.aten.copy_.default(arg263_1, getitem_107);  arg263_1 = getitem_107 = None
        copy__default_264: "f32[]" = torch.ops.aten.copy_.default(arg264_1, getitem_108);  arg264_1 = getitem_108 = None
        copy__default_265: "f32[]" = torch.ops.aten.copy_.default(arg265_1, getitem_109);  arg265_1 = getitem_109 = None
        copy__default_266: "f32[]" = torch.ops.aten.copy_.default(arg266_1, getitem_110);  arg266_1 = getitem_110 = None
        copy__default_267: "f32[]" = torch.ops.aten.copy_.default(arg267_1, getitem_111);  arg267_1 = getitem_111 = None
        copy__default_268: "f32[]" = torch.ops.aten.copy_.default(arg268_1, getitem_112);  arg268_1 = getitem_112 = None
        copy__default_269: "f32[]" = torch.ops.aten.copy_.default(arg269_1, getitem_113);  arg269_1 = getitem_113 = None
        copy__default_270: "f32[]" = torch.ops.aten.copy_.default(arg270_1, getitem_114);  arg270_1 = getitem_114 = None
        copy__default_271: "f32[]" = torch.ops.aten.copy_.default(arg271_1, getitem_115);  arg271_1 = getitem_115 = None
        copy__default_272: "f32[]" = torch.ops.aten.copy_.default(arg272_1, getitem_116);  arg272_1 = getitem_116 = None
        copy__default_273: "f32[]" = torch.ops.aten.copy_.default(arg273_1, getitem_117);  arg273_1 = getitem_117 = None
        copy__default_274: "f32[]" = torch.ops.aten.copy_.default(arg274_1, getitem_118);  arg274_1 = getitem_118 = None
        copy__default_275: "f32[]" = torch.ops.aten.copy_.default(arg275_1, getitem_119);  arg275_1 = getitem_119 = None
        copy__default_276: "f32[]" = torch.ops.aten.copy_.default(arg276_1, getitem_120);  arg276_1 = getitem_120 = None
        copy__default_277: "f32[]" = torch.ops.aten.copy_.default(arg277_1, getitem_121);  arg277_1 = getitem_121 = None
        copy__default_278: "f32[]" = torch.ops.aten.copy_.default(arg278_1, getitem_122);  arg278_1 = getitem_122 = None
        copy__default_279: "f32[]" = torch.ops.aten.copy_.default(arg279_1, getitem_123);  arg279_1 = getitem_123 = None
        copy__default_280: "f32[]" = torch.ops.aten.copy_.default(arg280_1, getitem_124);  arg280_1 = getitem_124 = None
        copy__default_281: "f32[]" = torch.ops.aten.copy_.default(arg281_1, getitem_125);  arg281_1 = getitem_125 = None
        copy__default_282: "f32[]" = torch.ops.aten.copy_.default(arg282_1, getitem_126);  arg282_1 = getitem_126 = None
        copy__default_283: "f32[]" = torch.ops.aten.copy_.default(arg283_1, getitem_127);  arg283_1 = getitem_127 = None
        copy__default_284: "f32[]" = torch.ops.aten.copy_.default(arg284_1, getitem_128);  arg284_1 = getitem_128 = None
        copy__default_285: "f32[]" = torch.ops.aten.copy_.default(arg285_1, getitem_129);  arg285_1 = getitem_129 = None
        copy__default_286: "f32[]" = torch.ops.aten.copy_.default(arg286_1, getitem_130);  arg286_1 = getitem_130 = None
        copy__default_287: "f32[]" = torch.ops.aten.copy_.default(arg287_1, getitem_131);  arg287_1 = getitem_131 = None
        copy__default_288: "f32[]" = torch.ops.aten.copy_.default(arg288_1, getitem_132);  arg288_1 = getitem_132 = None
        copy__default_289: "f32[]" = torch.ops.aten.copy_.default(arg289_1, getitem_133);  arg289_1 = getitem_133 = None
        copy__default_290: "f32[]" = torch.ops.aten.copy_.default(arg290_1, getitem_134);  arg290_1 = getitem_134 = None
        copy__default_291: "f32[]" = torch.ops.aten.copy_.default(arg291_1, getitem_135);  arg291_1 = getitem_135 = None
        copy__default_292: "f32[]" = torch.ops.aten.copy_.default(arg292_1, getitem_136);  arg292_1 = getitem_136 = None
        copy__default_293: "f32[]" = torch.ops.aten.copy_.default(arg293_1, getitem_137);  arg293_1 = getitem_137 = None
        copy__default_294: "f32[]" = torch.ops.aten.copy_.default(arg294_1, getitem_138);  arg294_1 = getitem_138 = None
        copy__default_295: "f32[]" = torch.ops.aten.copy_.default(arg295_1, getitem_139);  arg295_1 = getitem_139 = None
        copy__default_296: "f32[]" = torch.ops.aten.copy_.default(arg296_1, getitem_140);  arg296_1 = getitem_140 = None
        copy__default_297: "f32[]" = torch.ops.aten.copy_.default(arg297_1, getitem_141);  arg297_1 = getitem_141 = None
        copy__default_298: "f32[]" = torch.ops.aten.copy_.default(arg298_1, getitem_142);  arg298_1 = getitem_142 = None
        copy__default_299: "f32[]" = torch.ops.aten.copy_.default(arg299_1, getitem_143);  arg299_1 = getitem_143 = None
        copy__default_300: "f32[]" = torch.ops.aten.copy_.default(arg300_1, getitem_144);  arg300_1 = getitem_144 = None
        copy__default_301: "f32[]" = torch.ops.aten.copy_.default(arg301_1, getitem_145);  arg301_1 = getitem_145 = None
        copy__default_302: "f32[]" = torch.ops.aten.copy_.default(arg302_1, getitem_146);  arg302_1 = getitem_146 = None
        copy__default_303: "f32[]" = torch.ops.aten.copy_.default(arg303_1, getitem_147);  arg303_1 = getitem_147 = None
        copy__default_304: "f32[]" = torch.ops.aten.copy_.default(arg304_1, getitem_148);  arg304_1 = getitem_148 = None
        copy__default_305: "f32[]" = torch.ops.aten.copy_.default(arg305_1, getitem_149);  arg305_1 = getitem_149 = None
        copy__default_306: "f32[]" = torch.ops.aten.copy_.default(arg306_1, getitem_150);  arg306_1 = getitem_150 = None
        copy__default_307: "f32[]" = torch.ops.aten.copy_.default(arg307_1, getitem_151);  arg307_1 = getitem_151 = None
        copy__default_308: "f32[192, 3, 16, 16]" = torch.ops.aten.copy_.default(arg308_1, getitem_306);  arg308_1 = getitem_306 = None
        copy__default_309: "f32[192]" = torch.ops.aten.copy_.default(arg309_1, getitem_307);  arg309_1 = getitem_307 = None
        copy__default_310: "f32[192]" = torch.ops.aten.copy_.default(arg310_1, getitem_308);  arg310_1 = getitem_308 = None
        copy__default_311: "f32[192]" = torch.ops.aten.copy_.default(arg311_1, getitem_309);  arg311_1 = getitem_309 = None
        copy__default_312: "f32[576, 192]" = torch.ops.aten.copy_.default(arg312_1, getitem_310);  arg312_1 = getitem_310 = None
        copy__default_313: "f32[576]" = torch.ops.aten.copy_.default(arg313_1, getitem_311);  arg313_1 = getitem_311 = None
        copy__default_314: "f32[192, 192]" = torch.ops.aten.copy_.default(arg314_1, getitem_312);  arg314_1 = getitem_312 = None
        copy__default_315: "f32[192]" = torch.ops.aten.copy_.default(arg315_1, getitem_313);  arg315_1 = getitem_313 = None
        copy__default_316: "f32[192]" = torch.ops.aten.copy_.default(arg316_1, getitem_314);  arg316_1 = getitem_314 = None
        copy__default_317: "f32[192]" = torch.ops.aten.copy_.default(arg317_1, getitem_315);  arg317_1 = getitem_315 = None
        copy__default_318: "f32[768, 192]" = torch.ops.aten.copy_.default(arg318_1, getitem_316);  arg318_1 = getitem_316 = None
        copy__default_319: "f32[768]" = torch.ops.aten.copy_.default(arg319_1, getitem_317);  arg319_1 = getitem_317 = None
        copy__default_320: "f32[192, 768]" = torch.ops.aten.copy_.default(arg320_1, getitem_318);  arg320_1 = getitem_318 = None
        copy__default_321: "f32[192]" = torch.ops.aten.copy_.default(arg321_1, getitem_319);  arg321_1 = getitem_319 = None
        copy__default_322: "f32[192]" = torch.ops.aten.copy_.default(arg322_1, getitem_320);  arg322_1 = getitem_320 = None
        copy__default_323: "f32[192]" = torch.ops.aten.copy_.default(arg323_1, getitem_321);  arg323_1 = getitem_321 = None
        copy__default_324: "f32[576, 192]" = torch.ops.aten.copy_.default(arg324_1, getitem_322);  arg324_1 = getitem_322 = None
        copy__default_325: "f32[576]" = torch.ops.aten.copy_.default(arg325_1, getitem_323);  arg325_1 = getitem_323 = None
        copy__default_326: "f32[192, 192]" = torch.ops.aten.copy_.default(arg326_1, getitem_324);  arg326_1 = getitem_324 = None
        copy__default_327: "f32[192]" = torch.ops.aten.copy_.default(arg327_1, getitem_325);  arg327_1 = getitem_325 = None
        copy__default_328: "f32[192]" = torch.ops.aten.copy_.default(arg328_1, getitem_326);  arg328_1 = getitem_326 = None
        copy__default_329: "f32[192]" = torch.ops.aten.copy_.default(arg329_1, getitem_327);  arg329_1 = getitem_327 = None
        copy__default_330: "f32[768, 192]" = torch.ops.aten.copy_.default(arg330_1, getitem_328);  arg330_1 = getitem_328 = None
        copy__default_331: "f32[768]" = torch.ops.aten.copy_.default(arg331_1, getitem_329);  arg331_1 = getitem_329 = None
        copy__default_332: "f32[192, 768]" = torch.ops.aten.copy_.default(arg332_1, getitem_330);  arg332_1 = getitem_330 = None
        copy__default_333: "f32[192]" = torch.ops.aten.copy_.default(arg333_1, getitem_331);  arg333_1 = getitem_331 = None
        copy__default_334: "f32[192]" = torch.ops.aten.copy_.default(arg334_1, getitem_332);  arg334_1 = getitem_332 = None
        copy__default_335: "f32[192]" = torch.ops.aten.copy_.default(arg335_1, getitem_333);  arg335_1 = getitem_333 = None
        copy__default_336: "f32[576, 192]" = torch.ops.aten.copy_.default(arg336_1, getitem_334);  arg336_1 = getitem_334 = None
        copy__default_337: "f32[576]" = torch.ops.aten.copy_.default(arg337_1, getitem_335);  arg337_1 = getitem_335 = None
        copy__default_338: "f32[192, 192]" = torch.ops.aten.copy_.default(arg338_1, getitem_336);  arg338_1 = getitem_336 = None
        copy__default_339: "f32[192]" = torch.ops.aten.copy_.default(arg339_1, getitem_337);  arg339_1 = getitem_337 = None
        copy__default_340: "f32[192]" = torch.ops.aten.copy_.default(arg340_1, getitem_338);  arg340_1 = getitem_338 = None
        copy__default_341: "f32[192]" = torch.ops.aten.copy_.default(arg341_1, getitem_339);  arg341_1 = getitem_339 = None
        copy__default_342: "f32[768, 192]" = torch.ops.aten.copy_.default(arg342_1, getitem_340);  arg342_1 = getitem_340 = None
        copy__default_343: "f32[768]" = torch.ops.aten.copy_.default(arg343_1, getitem_341);  arg343_1 = getitem_341 = None
        copy__default_344: "f32[192, 768]" = torch.ops.aten.copy_.default(arg344_1, getitem_342);  arg344_1 = getitem_342 = None
        copy__default_345: "f32[192]" = torch.ops.aten.copy_.default(arg345_1, getitem_343);  arg345_1 = getitem_343 = None
        copy__default_346: "f32[192]" = torch.ops.aten.copy_.default(arg346_1, getitem_344);  arg346_1 = getitem_344 = None
        copy__default_347: "f32[192]" = torch.ops.aten.copy_.default(arg347_1, getitem_345);  arg347_1 = getitem_345 = None
        copy__default_348: "f32[576, 192]" = torch.ops.aten.copy_.default(arg348_1, getitem_346);  arg348_1 = getitem_346 = None
        copy__default_349: "f32[576]" = torch.ops.aten.copy_.default(arg349_1, getitem_347);  arg349_1 = getitem_347 = None
        copy__default_350: "f32[192, 192]" = torch.ops.aten.copy_.default(arg350_1, getitem_348);  arg350_1 = getitem_348 = None
        copy__default_351: "f32[192]" = torch.ops.aten.copy_.default(arg351_1, getitem_349);  arg351_1 = getitem_349 = None
        copy__default_352: "f32[192]" = torch.ops.aten.copy_.default(arg352_1, getitem_350);  arg352_1 = getitem_350 = None
        copy__default_353: "f32[192]" = torch.ops.aten.copy_.default(arg353_1, getitem_351);  arg353_1 = getitem_351 = None
        copy__default_354: "f32[768, 192]" = torch.ops.aten.copy_.default(arg354_1, getitem_352);  arg354_1 = getitem_352 = None
        copy__default_355: "f32[768]" = torch.ops.aten.copy_.default(arg355_1, getitem_353);  arg355_1 = getitem_353 = None
        copy__default_356: "f32[192, 768]" = torch.ops.aten.copy_.default(arg356_1, getitem_354);  arg356_1 = getitem_354 = None
        copy__default_357: "f32[192]" = torch.ops.aten.copy_.default(arg357_1, getitem_355);  arg357_1 = getitem_355 = None
        copy__default_358: "f32[192]" = torch.ops.aten.copy_.default(arg358_1, getitem_356);  arg358_1 = getitem_356 = None
        copy__default_359: "f32[192]" = torch.ops.aten.copy_.default(arg359_1, getitem_357);  arg359_1 = getitem_357 = None
        copy__default_360: "f32[576, 192]" = torch.ops.aten.copy_.default(arg360_1, getitem_358);  arg360_1 = getitem_358 = None
        copy__default_361: "f32[576]" = torch.ops.aten.copy_.default(arg361_1, getitem_359);  arg361_1 = getitem_359 = None
        copy__default_362: "f32[192, 192]" = torch.ops.aten.copy_.default(arg362_1, getitem_360);  arg362_1 = getitem_360 = None
        copy__default_363: "f32[192]" = torch.ops.aten.copy_.default(arg363_1, getitem_361);  arg363_1 = getitem_361 = None
        copy__default_364: "f32[192]" = torch.ops.aten.copy_.default(arg364_1, getitem_362);  arg364_1 = getitem_362 = None
        copy__default_365: "f32[192]" = torch.ops.aten.copy_.default(arg365_1, getitem_363);  arg365_1 = getitem_363 = None
        copy__default_366: "f32[768, 192]" = torch.ops.aten.copy_.default(arg366_1, getitem_364);  arg366_1 = getitem_364 = None
        copy__default_367: "f32[768]" = torch.ops.aten.copy_.default(arg367_1, getitem_365);  arg367_1 = getitem_365 = None
        copy__default_368: "f32[192, 768]" = torch.ops.aten.copy_.default(arg368_1, getitem_366);  arg368_1 = getitem_366 = None
        copy__default_369: "f32[192]" = torch.ops.aten.copy_.default(arg369_1, getitem_367);  arg369_1 = getitem_367 = None
        copy__default_370: "f32[192]" = torch.ops.aten.copy_.default(arg370_1, getitem_368);  arg370_1 = getitem_368 = None
        copy__default_371: "f32[192]" = torch.ops.aten.copy_.default(arg371_1, getitem_369);  arg371_1 = getitem_369 = None
        copy__default_372: "f32[576, 192]" = torch.ops.aten.copy_.default(arg372_1, getitem_370);  arg372_1 = getitem_370 = None
        copy__default_373: "f32[576]" = torch.ops.aten.copy_.default(arg373_1, getitem_371);  arg373_1 = getitem_371 = None
        copy__default_374: "f32[192, 192]" = torch.ops.aten.copy_.default(arg374_1, getitem_372);  arg374_1 = getitem_372 = None
        copy__default_375: "f32[192]" = torch.ops.aten.copy_.default(arg375_1, getitem_373);  arg375_1 = getitem_373 = None
        copy__default_376: "f32[192]" = torch.ops.aten.copy_.default(arg376_1, getitem_374);  arg376_1 = getitem_374 = None
        copy__default_377: "f32[192]" = torch.ops.aten.copy_.default(arg377_1, getitem_375);  arg377_1 = getitem_375 = None
        copy__default_378: "f32[768, 192]" = torch.ops.aten.copy_.default(arg378_1, getitem_376);  arg378_1 = getitem_376 = None
        copy__default_379: "f32[768]" = torch.ops.aten.copy_.default(arg379_1, getitem_377);  arg379_1 = getitem_377 = None
        copy__default_380: "f32[192, 768]" = torch.ops.aten.copy_.default(arg380_1, getitem_378);  arg380_1 = getitem_378 = None
        copy__default_381: "f32[192]" = torch.ops.aten.copy_.default(arg381_1, getitem_379);  arg381_1 = getitem_379 = None
        copy__default_382: "f32[192]" = torch.ops.aten.copy_.default(arg382_1, getitem_380);  arg382_1 = getitem_380 = None
        copy__default_383: "f32[192]" = torch.ops.aten.copy_.default(arg383_1, getitem_381);  arg383_1 = getitem_381 = None
        copy__default_384: "f32[576, 192]" = torch.ops.aten.copy_.default(arg384_1, getitem_382);  arg384_1 = getitem_382 = None
        copy__default_385: "f32[576]" = torch.ops.aten.copy_.default(arg385_1, getitem_383);  arg385_1 = getitem_383 = None
        copy__default_386: "f32[192, 192]" = torch.ops.aten.copy_.default(arg386_1, getitem_384);  arg386_1 = getitem_384 = None
        copy__default_387: "f32[192]" = torch.ops.aten.copy_.default(arg387_1, getitem_385);  arg387_1 = getitem_385 = None
        copy__default_388: "f32[192]" = torch.ops.aten.copy_.default(arg388_1, getitem_386);  arg388_1 = getitem_386 = None
        copy__default_389: "f32[192]" = torch.ops.aten.copy_.default(arg389_1, getitem_387);  arg389_1 = getitem_387 = None
        copy__default_390: "f32[768, 192]" = torch.ops.aten.copy_.default(arg390_1, getitem_388);  arg390_1 = getitem_388 = None
        copy__default_391: "f32[768]" = torch.ops.aten.copy_.default(arg391_1, getitem_389);  arg391_1 = getitem_389 = None
        copy__default_392: "f32[192, 768]" = torch.ops.aten.copy_.default(arg392_1, getitem_390);  arg392_1 = getitem_390 = None
        copy__default_393: "f32[192]" = torch.ops.aten.copy_.default(arg393_1, getitem_391);  arg393_1 = getitem_391 = None
        copy__default_394: "f32[192]" = torch.ops.aten.copy_.default(arg394_1, getitem_392);  arg394_1 = getitem_392 = None
        copy__default_395: "f32[192]" = torch.ops.aten.copy_.default(arg395_1, getitem_393);  arg395_1 = getitem_393 = None
        copy__default_396: "f32[576, 192]" = torch.ops.aten.copy_.default(arg396_1, getitem_394);  arg396_1 = getitem_394 = None
        copy__default_397: "f32[576]" = torch.ops.aten.copy_.default(arg397_1, getitem_395);  arg397_1 = getitem_395 = None
        copy__default_398: "f32[192, 192]" = torch.ops.aten.copy_.default(arg398_1, getitem_396);  arg398_1 = getitem_396 = None
        copy__default_399: "f32[192]" = torch.ops.aten.copy_.default(arg399_1, getitem_397);  arg399_1 = getitem_397 = None
        copy__default_400: "f32[192]" = torch.ops.aten.copy_.default(arg400_1, getitem_398);  arg400_1 = getitem_398 = None
        copy__default_401: "f32[192]" = torch.ops.aten.copy_.default(arg401_1, getitem_399);  arg401_1 = getitem_399 = None
        copy__default_402: "f32[768, 192]" = torch.ops.aten.copy_.default(arg402_1, getitem_400);  arg402_1 = getitem_400 = None
        copy__default_403: "f32[768]" = torch.ops.aten.copy_.default(arg403_1, getitem_401);  arg403_1 = getitem_401 = None
        copy__default_404: "f32[192, 768]" = torch.ops.aten.copy_.default(arg404_1, getitem_402);  arg404_1 = getitem_402 = None
        copy__default_405: "f32[192]" = torch.ops.aten.copy_.default(arg405_1, getitem_403);  arg405_1 = getitem_403 = None
        copy__default_406: "f32[192]" = torch.ops.aten.copy_.default(arg406_1, getitem_404);  arg406_1 = getitem_404 = None
        copy__default_407: "f32[192]" = torch.ops.aten.copy_.default(arg407_1, getitem_405);  arg407_1 = getitem_405 = None
        copy__default_408: "f32[576, 192]" = torch.ops.aten.copy_.default(arg408_1, getitem_406);  arg408_1 = getitem_406 = None
        copy__default_409: "f32[576]" = torch.ops.aten.copy_.default(arg409_1, getitem_407);  arg409_1 = getitem_407 = None
        copy__default_410: "f32[192, 192]" = torch.ops.aten.copy_.default(arg410_1, getitem_408);  arg410_1 = getitem_408 = None
        copy__default_411: "f32[192]" = torch.ops.aten.copy_.default(arg411_1, getitem_409);  arg411_1 = getitem_409 = None
        copy__default_412: "f32[192]" = torch.ops.aten.copy_.default(arg412_1, getitem_410);  arg412_1 = getitem_410 = None
        copy__default_413: "f32[192]" = torch.ops.aten.copy_.default(arg413_1, getitem_411);  arg413_1 = getitem_411 = None
        copy__default_414: "f32[768, 192]" = torch.ops.aten.copy_.default(arg414_1, getitem_412);  arg414_1 = getitem_412 = None
        copy__default_415: "f32[768]" = torch.ops.aten.copy_.default(arg415_1, getitem_413);  arg415_1 = getitem_413 = None
        copy__default_416: "f32[192, 768]" = torch.ops.aten.copy_.default(arg416_1, getitem_414);  arg416_1 = getitem_414 = None
        copy__default_417: "f32[192]" = torch.ops.aten.copy_.default(arg417_1, getitem_415);  arg417_1 = getitem_415 = None
        copy__default_418: "f32[192]" = torch.ops.aten.copy_.default(arg418_1, getitem_416);  arg418_1 = getitem_416 = None
        copy__default_419: "f32[192]" = torch.ops.aten.copy_.default(arg419_1, getitem_417);  arg419_1 = getitem_417 = None
        copy__default_420: "f32[576, 192]" = torch.ops.aten.copy_.default(arg420_1, getitem_418);  arg420_1 = getitem_418 = None
        copy__default_421: "f32[576]" = torch.ops.aten.copy_.default(arg421_1, getitem_419);  arg421_1 = getitem_419 = None
        copy__default_422: "f32[192, 192]" = torch.ops.aten.copy_.default(arg422_1, getitem_420);  arg422_1 = getitem_420 = None
        copy__default_423: "f32[192]" = torch.ops.aten.copy_.default(arg423_1, getitem_421);  arg423_1 = getitem_421 = None
        copy__default_424: "f32[192]" = torch.ops.aten.copy_.default(arg424_1, getitem_422);  arg424_1 = getitem_422 = None
        copy__default_425: "f32[192]" = torch.ops.aten.copy_.default(arg425_1, getitem_423);  arg425_1 = getitem_423 = None
        copy__default_426: "f32[768, 192]" = torch.ops.aten.copy_.default(arg426_1, getitem_424);  arg426_1 = getitem_424 = None
        copy__default_427: "f32[768]" = torch.ops.aten.copy_.default(arg427_1, getitem_425);  arg427_1 = getitem_425 = None
        copy__default_428: "f32[192, 768]" = torch.ops.aten.copy_.default(arg428_1, getitem_426);  arg428_1 = getitem_426 = None
        copy__default_429: "f32[192]" = torch.ops.aten.copy_.default(arg429_1, getitem_427);  arg429_1 = getitem_427 = None
        copy__default_430: "f32[192]" = torch.ops.aten.copy_.default(arg430_1, getitem_428);  arg430_1 = getitem_428 = None
        copy__default_431: "f32[192]" = torch.ops.aten.copy_.default(arg431_1, getitem_429);  arg431_1 = getitem_429 = None
        copy__default_432: "f32[576, 192]" = torch.ops.aten.copy_.default(arg432_1, getitem_430);  arg432_1 = getitem_430 = None
        copy__default_433: "f32[576]" = torch.ops.aten.copy_.default(arg433_1, getitem_431);  arg433_1 = getitem_431 = None
        copy__default_434: "f32[192, 192]" = torch.ops.aten.copy_.default(arg434_1, getitem_432);  arg434_1 = getitem_432 = None
        copy__default_435: "f32[192]" = torch.ops.aten.copy_.default(arg435_1, getitem_433);  arg435_1 = getitem_433 = None
        copy__default_436: "f32[192]" = torch.ops.aten.copy_.default(arg436_1, getitem_434);  arg436_1 = getitem_434 = None
        copy__default_437: "f32[192]" = torch.ops.aten.copy_.default(arg437_1, getitem_435);  arg437_1 = getitem_435 = None
        copy__default_438: "f32[768, 192]" = torch.ops.aten.copy_.default(arg438_1, getitem_436);  arg438_1 = getitem_436 = None
        copy__default_439: "f32[768]" = torch.ops.aten.copy_.default(arg439_1, getitem_437);  arg439_1 = getitem_437 = None
        copy__default_440: "f32[192, 768]" = torch.ops.aten.copy_.default(arg440_1, getitem_438);  arg440_1 = getitem_438 = None
        copy__default_441: "f32[192]" = torch.ops.aten.copy_.default(arg441_1, getitem_439);  arg441_1 = getitem_439 = None
        copy__default_442: "f32[192]" = torch.ops.aten.copy_.default(arg442_1, getitem_440);  arg442_1 = getitem_440 = None
        copy__default_443: "f32[192]" = torch.ops.aten.copy_.default(arg443_1, getitem_441);  arg443_1 = getitem_441 = None
        copy__default_444: "f32[576, 192]" = torch.ops.aten.copy_.default(arg444_1, getitem_442);  arg444_1 = getitem_442 = None
        copy__default_445: "f32[576]" = torch.ops.aten.copy_.default(arg445_1, getitem_443);  arg445_1 = getitem_443 = None
        copy__default_446: "f32[192, 192]" = torch.ops.aten.copy_.default(arg446_1, getitem_444);  arg446_1 = getitem_444 = None
        copy__default_447: "f32[192]" = torch.ops.aten.copy_.default(arg447_1, getitem_445);  arg447_1 = getitem_445 = None
        copy__default_448: "f32[192]" = torch.ops.aten.copy_.default(arg448_1, getitem_446);  arg448_1 = getitem_446 = None
        copy__default_449: "f32[192]" = torch.ops.aten.copy_.default(arg449_1, getitem_447);  arg449_1 = getitem_447 = None
        copy__default_450: "f32[768, 192]" = torch.ops.aten.copy_.default(arg450_1, getitem_448);  arg450_1 = getitem_448 = None
        copy__default_451: "f32[768]" = torch.ops.aten.copy_.default(arg451_1, getitem_449);  arg451_1 = getitem_449 = None
        copy__default_452: "f32[192, 768]" = torch.ops.aten.copy_.default(arg452_1, getitem_450);  arg452_1 = getitem_450 = None
        copy__default_453: "f32[192]" = torch.ops.aten.copy_.default(arg453_1, getitem_451);  arg453_1 = getitem_451 = None
        copy__default_454: "f32[192]" = torch.ops.aten.copy_.default(arg454_1, getitem_452);  arg454_1 = getitem_452 = None
        copy__default_455: "f32[192]" = torch.ops.aten.copy_.default(arg455_1, getitem_453);  arg455_1 = getitem_453 = None
        copy__default_456: "f32[1000, 192]" = torch.ops.aten.copy_.default(arg456_1, getitem_454);  arg456_1 = getitem_454 = None
        copy__default_457: "f32[1000]" = torch.ops.aten.copy_.default(arg457_1, getitem_455);  arg457_1 = getitem_455 = None
        copy__default_458: "f32[192, 3, 16, 16]" = torch.ops.aten.copy_.default(arg458_1, getitem_610);  arg458_1 = getitem_610 = None
        copy__default_459: "f32[192]" = torch.ops.aten.copy_.default(arg459_1, getitem_611);  arg459_1 = getitem_611 = None
        copy__default_460: "f32[192]" = torch.ops.aten.copy_.default(arg460_1, getitem_612);  arg460_1 = getitem_612 = None
        copy__default_461: "f32[192]" = torch.ops.aten.copy_.default(arg461_1, getitem_613);  arg461_1 = getitem_613 = None
        copy__default_462: "f32[576, 192]" = torch.ops.aten.copy_.default(arg462_1, getitem_614);  arg462_1 = getitem_614 = None
        copy__default_463: "f32[576]" = torch.ops.aten.copy_.default(arg463_1, getitem_615);  arg463_1 = getitem_615 = None
        copy__default_464: "f32[192, 192]" = torch.ops.aten.copy_.default(arg464_1, getitem_616);  arg464_1 = getitem_616 = None
        copy__default_465: "f32[192]" = torch.ops.aten.copy_.default(arg465_1, getitem_617);  arg465_1 = getitem_617 = None
        copy__default_466: "f32[192]" = torch.ops.aten.copy_.default(arg466_1, getitem_618);  arg466_1 = getitem_618 = None
        copy__default_467: "f32[192]" = torch.ops.aten.copy_.default(arg467_1, getitem_619);  arg467_1 = getitem_619 = None
        copy__default_468: "f32[768, 192]" = torch.ops.aten.copy_.default(arg468_1, getitem_620);  arg468_1 = getitem_620 = None
        copy__default_469: "f32[768]" = torch.ops.aten.copy_.default(arg469_1, getitem_621);  arg469_1 = getitem_621 = None
        copy__default_470: "f32[192, 768]" = torch.ops.aten.copy_.default(arg470_1, getitem_622);  arg470_1 = getitem_622 = None
        copy__default_471: "f32[192]" = torch.ops.aten.copy_.default(arg471_1, getitem_623);  arg471_1 = getitem_623 = None
        copy__default_472: "f32[192]" = torch.ops.aten.copy_.default(arg472_1, getitem_624);  arg472_1 = getitem_624 = None
        copy__default_473: "f32[192]" = torch.ops.aten.copy_.default(arg473_1, getitem_625);  arg473_1 = getitem_625 = None
        copy__default_474: "f32[576, 192]" = torch.ops.aten.copy_.default(arg474_1, getitem_626);  arg474_1 = getitem_626 = None
        copy__default_475: "f32[576]" = torch.ops.aten.copy_.default(arg475_1, getitem_627);  arg475_1 = getitem_627 = None
        copy__default_476: "f32[192, 192]" = torch.ops.aten.copy_.default(arg476_1, getitem_628);  arg476_1 = getitem_628 = None
        copy__default_477: "f32[192]" = torch.ops.aten.copy_.default(arg477_1, getitem_629);  arg477_1 = getitem_629 = None
        copy__default_478: "f32[192]" = torch.ops.aten.copy_.default(arg478_1, getitem_630);  arg478_1 = getitem_630 = None
        copy__default_479: "f32[192]" = torch.ops.aten.copy_.default(arg479_1, getitem_631);  arg479_1 = getitem_631 = None
        copy__default_480: "f32[768, 192]" = torch.ops.aten.copy_.default(arg480_1, getitem_632);  arg480_1 = getitem_632 = None
        copy__default_481: "f32[768]" = torch.ops.aten.copy_.default(arg481_1, getitem_633);  arg481_1 = getitem_633 = None
        copy__default_482: "f32[192, 768]" = torch.ops.aten.copy_.default(arg482_1, getitem_634);  arg482_1 = getitem_634 = None
        copy__default_483: "f32[192]" = torch.ops.aten.copy_.default(arg483_1, getitem_635);  arg483_1 = getitem_635 = None
        copy__default_484: "f32[192]" = torch.ops.aten.copy_.default(arg484_1, getitem_636);  arg484_1 = getitem_636 = None
        copy__default_485: "f32[192]" = torch.ops.aten.copy_.default(arg485_1, getitem_637);  arg485_1 = getitem_637 = None
        copy__default_486: "f32[576, 192]" = torch.ops.aten.copy_.default(arg486_1, getitem_638);  arg486_1 = getitem_638 = None
        copy__default_487: "f32[576]" = torch.ops.aten.copy_.default(arg487_1, getitem_639);  arg487_1 = getitem_639 = None
        copy__default_488: "f32[192, 192]" = torch.ops.aten.copy_.default(arg488_1, getitem_640);  arg488_1 = getitem_640 = None
        copy__default_489: "f32[192]" = torch.ops.aten.copy_.default(arg489_1, getitem_641);  arg489_1 = getitem_641 = None
        copy__default_490: "f32[192]" = torch.ops.aten.copy_.default(arg490_1, getitem_642);  arg490_1 = getitem_642 = None
        copy__default_491: "f32[192]" = torch.ops.aten.copy_.default(arg491_1, getitem_643);  arg491_1 = getitem_643 = None
        copy__default_492: "f32[768, 192]" = torch.ops.aten.copy_.default(arg492_1, getitem_644);  arg492_1 = getitem_644 = None
        copy__default_493: "f32[768]" = torch.ops.aten.copy_.default(arg493_1, getitem_645);  arg493_1 = getitem_645 = None
        copy__default_494: "f32[192, 768]" = torch.ops.aten.copy_.default(arg494_1, getitem_646);  arg494_1 = getitem_646 = None
        copy__default_495: "f32[192]" = torch.ops.aten.copy_.default(arg495_1, getitem_647);  arg495_1 = getitem_647 = None
        copy__default_496: "f32[192]" = torch.ops.aten.copy_.default(arg496_1, getitem_648);  arg496_1 = getitem_648 = None
        copy__default_497: "f32[192]" = torch.ops.aten.copy_.default(arg497_1, getitem_649);  arg497_1 = getitem_649 = None
        copy__default_498: "f32[576, 192]" = torch.ops.aten.copy_.default(arg498_1, getitem_650);  arg498_1 = getitem_650 = None
        copy__default_499: "f32[576]" = torch.ops.aten.copy_.default(arg499_1, getitem_651);  arg499_1 = getitem_651 = None
        copy__default_500: "f32[192, 192]" = torch.ops.aten.copy_.default(arg500_1, getitem_652);  arg500_1 = getitem_652 = None
        copy__default_501: "f32[192]" = torch.ops.aten.copy_.default(arg501_1, getitem_653);  arg501_1 = getitem_653 = None
        copy__default_502: "f32[192]" = torch.ops.aten.copy_.default(arg502_1, getitem_654);  arg502_1 = getitem_654 = None
        copy__default_503: "f32[192]" = torch.ops.aten.copy_.default(arg503_1, getitem_655);  arg503_1 = getitem_655 = None
        copy__default_504: "f32[768, 192]" = torch.ops.aten.copy_.default(arg504_1, getitem_656);  arg504_1 = getitem_656 = None
        copy__default_505: "f32[768]" = torch.ops.aten.copy_.default(arg505_1, getitem_657);  arg505_1 = getitem_657 = None
        copy__default_506: "f32[192, 768]" = torch.ops.aten.copy_.default(arg506_1, getitem_658);  arg506_1 = getitem_658 = None
        copy__default_507: "f32[192]" = torch.ops.aten.copy_.default(arg507_1, getitem_659);  arg507_1 = getitem_659 = None
        copy__default_508: "f32[192]" = torch.ops.aten.copy_.default(arg508_1, getitem_660);  arg508_1 = getitem_660 = None
        copy__default_509: "f32[192]" = torch.ops.aten.copy_.default(arg509_1, getitem_661);  arg509_1 = getitem_661 = None
        copy__default_510: "f32[576, 192]" = torch.ops.aten.copy_.default(arg510_1, getitem_662);  arg510_1 = getitem_662 = None
        copy__default_511: "f32[576]" = torch.ops.aten.copy_.default(arg511_1, getitem_663);  arg511_1 = getitem_663 = None
        copy__default_512: "f32[192, 192]" = torch.ops.aten.copy_.default(arg512_1, getitem_664);  arg512_1 = getitem_664 = None
        copy__default_513: "f32[192]" = torch.ops.aten.copy_.default(arg513_1, getitem_665);  arg513_1 = getitem_665 = None
        copy__default_514: "f32[192]" = torch.ops.aten.copy_.default(arg514_1, getitem_666);  arg514_1 = getitem_666 = None
        copy__default_515: "f32[192]" = torch.ops.aten.copy_.default(arg515_1, getitem_667);  arg515_1 = getitem_667 = None
        copy__default_516: "f32[768, 192]" = torch.ops.aten.copy_.default(arg516_1, getitem_668);  arg516_1 = getitem_668 = None
        copy__default_517: "f32[768]" = torch.ops.aten.copy_.default(arg517_1, getitem_669);  arg517_1 = getitem_669 = None
        copy__default_518: "f32[192, 768]" = torch.ops.aten.copy_.default(arg518_1, getitem_670);  arg518_1 = getitem_670 = None
        copy__default_519: "f32[192]" = torch.ops.aten.copy_.default(arg519_1, getitem_671);  arg519_1 = getitem_671 = None
        copy__default_520: "f32[192]" = torch.ops.aten.copy_.default(arg520_1, getitem_672);  arg520_1 = getitem_672 = None
        copy__default_521: "f32[192]" = torch.ops.aten.copy_.default(arg521_1, getitem_673);  arg521_1 = getitem_673 = None
        copy__default_522: "f32[576, 192]" = torch.ops.aten.copy_.default(arg522_1, getitem_674);  arg522_1 = getitem_674 = None
        copy__default_523: "f32[576]" = torch.ops.aten.copy_.default(arg523_1, getitem_675);  arg523_1 = getitem_675 = None
        copy__default_524: "f32[192, 192]" = torch.ops.aten.copy_.default(arg524_1, getitem_676);  arg524_1 = getitem_676 = None
        copy__default_525: "f32[192]" = torch.ops.aten.copy_.default(arg525_1, getitem_677);  arg525_1 = getitem_677 = None
        copy__default_526: "f32[192]" = torch.ops.aten.copy_.default(arg526_1, getitem_678);  arg526_1 = getitem_678 = None
        copy__default_527: "f32[192]" = torch.ops.aten.copy_.default(arg527_1, getitem_679);  arg527_1 = getitem_679 = None
        copy__default_528: "f32[768, 192]" = torch.ops.aten.copy_.default(arg528_1, getitem_680);  arg528_1 = getitem_680 = None
        copy__default_529: "f32[768]" = torch.ops.aten.copy_.default(arg529_1, getitem_681);  arg529_1 = getitem_681 = None
        copy__default_530: "f32[192, 768]" = torch.ops.aten.copy_.default(arg530_1, getitem_682);  arg530_1 = getitem_682 = None
        copy__default_531: "f32[192]" = torch.ops.aten.copy_.default(arg531_1, getitem_683);  arg531_1 = getitem_683 = None
        copy__default_532: "f32[192]" = torch.ops.aten.copy_.default(arg532_1, getitem_684);  arg532_1 = getitem_684 = None
        copy__default_533: "f32[192]" = torch.ops.aten.copy_.default(arg533_1, getitem_685);  arg533_1 = getitem_685 = None
        copy__default_534: "f32[576, 192]" = torch.ops.aten.copy_.default(arg534_1, getitem_686);  arg534_1 = getitem_686 = None
        copy__default_535: "f32[576]" = torch.ops.aten.copy_.default(arg535_1, getitem_687);  arg535_1 = getitem_687 = None
        copy__default_536: "f32[192, 192]" = torch.ops.aten.copy_.default(arg536_1, getitem_688);  arg536_1 = getitem_688 = None
        copy__default_537: "f32[192]" = torch.ops.aten.copy_.default(arg537_1, getitem_689);  arg537_1 = getitem_689 = None
        copy__default_538: "f32[192]" = torch.ops.aten.copy_.default(arg538_1, getitem_690);  arg538_1 = getitem_690 = None
        copy__default_539: "f32[192]" = torch.ops.aten.copy_.default(arg539_1, getitem_691);  arg539_1 = getitem_691 = None
        copy__default_540: "f32[768, 192]" = torch.ops.aten.copy_.default(arg540_1, getitem_692);  arg540_1 = getitem_692 = None
        copy__default_541: "f32[768]" = torch.ops.aten.copy_.default(arg541_1, getitem_693);  arg541_1 = getitem_693 = None
        copy__default_542: "f32[192, 768]" = torch.ops.aten.copy_.default(arg542_1, getitem_694);  arg542_1 = getitem_694 = None
        copy__default_543: "f32[192]" = torch.ops.aten.copy_.default(arg543_1, getitem_695);  arg543_1 = getitem_695 = None
        copy__default_544: "f32[192]" = torch.ops.aten.copy_.default(arg544_1, getitem_696);  arg544_1 = getitem_696 = None
        copy__default_545: "f32[192]" = torch.ops.aten.copy_.default(arg545_1, getitem_697);  arg545_1 = getitem_697 = None
        copy__default_546: "f32[576, 192]" = torch.ops.aten.copy_.default(arg546_1, getitem_698);  arg546_1 = getitem_698 = None
        copy__default_547: "f32[576]" = torch.ops.aten.copy_.default(arg547_1, getitem_699);  arg547_1 = getitem_699 = None
        copy__default_548: "f32[192, 192]" = torch.ops.aten.copy_.default(arg548_1, getitem_700);  arg548_1 = getitem_700 = None
        copy__default_549: "f32[192]" = torch.ops.aten.copy_.default(arg549_1, getitem_701);  arg549_1 = getitem_701 = None
        copy__default_550: "f32[192]" = torch.ops.aten.copy_.default(arg550_1, getitem_702);  arg550_1 = getitem_702 = None
        copy__default_551: "f32[192]" = torch.ops.aten.copy_.default(arg551_1, getitem_703);  arg551_1 = getitem_703 = None
        copy__default_552: "f32[768, 192]" = torch.ops.aten.copy_.default(arg552_1, getitem_704);  arg552_1 = getitem_704 = None
        copy__default_553: "f32[768]" = torch.ops.aten.copy_.default(arg553_1, getitem_705);  arg553_1 = getitem_705 = None
        copy__default_554: "f32[192, 768]" = torch.ops.aten.copy_.default(arg554_1, getitem_706);  arg554_1 = getitem_706 = None
        copy__default_555: "f32[192]" = torch.ops.aten.copy_.default(arg555_1, getitem_707);  arg555_1 = getitem_707 = None
        copy__default_556: "f32[192]" = torch.ops.aten.copy_.default(arg556_1, getitem_708);  arg556_1 = getitem_708 = None
        copy__default_557: "f32[192]" = torch.ops.aten.copy_.default(arg557_1, getitem_709);  arg557_1 = getitem_709 = None
        copy__default_558: "f32[576, 192]" = torch.ops.aten.copy_.default(arg558_1, getitem_710);  arg558_1 = getitem_710 = None
        copy__default_559: "f32[576]" = torch.ops.aten.copy_.default(arg559_1, getitem_711);  arg559_1 = getitem_711 = None
        copy__default_560: "f32[192, 192]" = torch.ops.aten.copy_.default(arg560_1, getitem_712);  arg560_1 = getitem_712 = None
        copy__default_561: "f32[192]" = torch.ops.aten.copy_.default(arg561_1, getitem_713);  arg561_1 = getitem_713 = None
        copy__default_562: "f32[192]" = torch.ops.aten.copy_.default(arg562_1, getitem_714);  arg562_1 = getitem_714 = None
        copy__default_563: "f32[192]" = torch.ops.aten.copy_.default(arg563_1, getitem_715);  arg563_1 = getitem_715 = None
        copy__default_564: "f32[768, 192]" = torch.ops.aten.copy_.default(arg564_1, getitem_716);  arg564_1 = getitem_716 = None
        copy__default_565: "f32[768]" = torch.ops.aten.copy_.default(arg565_1, getitem_717);  arg565_1 = getitem_717 = None
        copy__default_566: "f32[192, 768]" = torch.ops.aten.copy_.default(arg566_1, getitem_718);  arg566_1 = getitem_718 = None
        copy__default_567: "f32[192]" = torch.ops.aten.copy_.default(arg567_1, getitem_719);  arg567_1 = getitem_719 = None
        copy__default_568: "f32[192]" = torch.ops.aten.copy_.default(arg568_1, getitem_720);  arg568_1 = getitem_720 = None
        copy__default_569: "f32[192]" = torch.ops.aten.copy_.default(arg569_1, getitem_721);  arg569_1 = getitem_721 = None
        copy__default_570: "f32[576, 192]" = torch.ops.aten.copy_.default(arg570_1, getitem_722);  arg570_1 = getitem_722 = None
        copy__default_571: "f32[576]" = torch.ops.aten.copy_.default(arg571_1, getitem_723);  arg571_1 = getitem_723 = None
        copy__default_572: "f32[192, 192]" = torch.ops.aten.copy_.default(arg572_1, getitem_724);  arg572_1 = getitem_724 = None
        copy__default_573: "f32[192]" = torch.ops.aten.copy_.default(arg573_1, getitem_725);  arg573_1 = getitem_725 = None
        copy__default_574: "f32[192]" = torch.ops.aten.copy_.default(arg574_1, getitem_726);  arg574_1 = getitem_726 = None
        copy__default_575: "f32[192]" = torch.ops.aten.copy_.default(arg575_1, getitem_727);  arg575_1 = getitem_727 = None
        copy__default_576: "f32[768, 192]" = torch.ops.aten.copy_.default(arg576_1, getitem_728);  arg576_1 = getitem_728 = None
        copy__default_577: "f32[768]" = torch.ops.aten.copy_.default(arg577_1, getitem_729);  arg577_1 = getitem_729 = None
        copy__default_578: "f32[192, 768]" = torch.ops.aten.copy_.default(arg578_1, getitem_730);  arg578_1 = getitem_730 = None
        copy__default_579: "f32[192]" = torch.ops.aten.copy_.default(arg579_1, getitem_731);  arg579_1 = getitem_731 = None
        copy__default_580: "f32[192]" = torch.ops.aten.copy_.default(arg580_1, getitem_732);  arg580_1 = getitem_732 = None
        copy__default_581: "f32[192]" = torch.ops.aten.copy_.default(arg581_1, getitem_733);  arg581_1 = getitem_733 = None
        copy__default_582: "f32[576, 192]" = torch.ops.aten.copy_.default(arg582_1, getitem_734);  arg582_1 = getitem_734 = None
        copy__default_583: "f32[576]" = torch.ops.aten.copy_.default(arg583_1, getitem_735);  arg583_1 = getitem_735 = None
        copy__default_584: "f32[192, 192]" = torch.ops.aten.copy_.default(arg584_1, getitem_736);  arg584_1 = getitem_736 = None
        copy__default_585: "f32[192]" = torch.ops.aten.copy_.default(arg585_1, getitem_737);  arg585_1 = getitem_737 = None
        copy__default_586: "f32[192]" = torch.ops.aten.copy_.default(arg586_1, getitem_738);  arg586_1 = getitem_738 = None
        copy__default_587: "f32[192]" = torch.ops.aten.copy_.default(arg587_1, getitem_739);  arg587_1 = getitem_739 = None
        copy__default_588: "f32[768, 192]" = torch.ops.aten.copy_.default(arg588_1, getitem_740);  arg588_1 = getitem_740 = None
        copy__default_589: "f32[768]" = torch.ops.aten.copy_.default(arg589_1, getitem_741);  arg589_1 = getitem_741 = None
        copy__default_590: "f32[192, 768]" = torch.ops.aten.copy_.default(arg590_1, getitem_742);  arg590_1 = getitem_742 = None
        copy__default_591: "f32[192]" = torch.ops.aten.copy_.default(arg591_1, getitem_743);  arg591_1 = getitem_743 = None
        copy__default_592: "f32[192]" = torch.ops.aten.copy_.default(arg592_1, getitem_744);  arg592_1 = getitem_744 = None
        copy__default_593: "f32[192]" = torch.ops.aten.copy_.default(arg593_1, getitem_745);  arg593_1 = getitem_745 = None
        copy__default_594: "f32[576, 192]" = torch.ops.aten.copy_.default(arg594_1, getitem_746);  arg594_1 = getitem_746 = None
        copy__default_595: "f32[576]" = torch.ops.aten.copy_.default(arg595_1, getitem_747);  arg595_1 = getitem_747 = None
        copy__default_596: "f32[192, 192]" = torch.ops.aten.copy_.default(arg596_1, getitem_748);  arg596_1 = getitem_748 = None
        copy__default_597: "f32[192]" = torch.ops.aten.copy_.default(arg597_1, getitem_749);  arg597_1 = getitem_749 = None
        copy__default_598: "f32[192]" = torch.ops.aten.copy_.default(arg598_1, getitem_750);  arg598_1 = getitem_750 = None
        copy__default_599: "f32[192]" = torch.ops.aten.copy_.default(arg599_1, getitem_751);  arg599_1 = getitem_751 = None
        copy__default_600: "f32[768, 192]" = torch.ops.aten.copy_.default(arg600_1, getitem_752);  arg600_1 = getitem_752 = None
        copy__default_601: "f32[768]" = torch.ops.aten.copy_.default(arg601_1, getitem_753);  arg601_1 = getitem_753 = None
        copy__default_602: "f32[192, 768]" = torch.ops.aten.copy_.default(arg602_1, getitem_754);  arg602_1 = getitem_754 = None
        copy__default_603: "f32[192]" = torch.ops.aten.copy_.default(arg603_1, getitem_755);  arg603_1 = getitem_755 = None
        copy__default_604: "f32[192]" = torch.ops.aten.copy_.default(arg604_1, getitem_756);  arg604_1 = getitem_756 = None
        copy__default_605: "f32[192]" = torch.ops.aten.copy_.default(arg605_1, getitem_757);  arg605_1 = getitem_757 = None
        copy__default_606: "f32[1000, 192]" = torch.ops.aten.copy_.default(arg606_1, getitem_758);  arg606_1 = getitem_758 = None
        copy__default_607: "f32[1000]" = torch.ops.aten.copy_.default(arg607_1, getitem_759);  arg607_1 = getitem_759 = None
        return (copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182, copy__default_183, copy__default_184, copy__default_185, copy__default_186, copy__default_187, copy__default_188, copy__default_189, copy__default_190, copy__default_191, copy__default_192, copy__default_193, copy__default_194, copy__default_195, copy__default_196, copy__default_197, copy__default_198, copy__default_199, copy__default_200, copy__default_201, copy__default_202, copy__default_203, copy__default_204, copy__default_205, copy__default_206, copy__default_207, copy__default_208, copy__default_209, copy__default_210, copy__default_211, copy__default_212, copy__default_213, copy__default_214, copy__default_215, copy__default_216, copy__default_217, copy__default_218, copy__default_219, copy__default_220, copy__default_221, copy__default_222, copy__default_223, copy__default_224, copy__default_225, copy__default_226, copy__default_227, copy__default_228, copy__default_229, copy__default_230, copy__default_231, copy__default_232, copy__default_233, copy__default_234, copy__default_235, copy__default_236, copy__default_237, copy__default_238, copy__default_239, copy__default_240, copy__default_241, copy__default_242, copy__default_243, copy__default_244, copy__default_245, copy__default_246, copy__default_247, copy__default_248, copy__default_249, copy__default_250, copy__default_251, copy__default_252, copy__default_253, copy__default_254, copy__default_255, copy__default_256, copy__default_257, copy__default_258, copy__default_259, copy__default_260, copy__default_261, copy__default_262, copy__default_263, copy__default_264, copy__default_265, copy__default_266, copy__default_267, copy__default_268, copy__default_269, copy__default_270, copy__default_271, copy__default_272, copy__default_273, copy__default_274, copy__default_275, copy__default_276, copy__default_277, copy__default_278, copy__default_279, copy__default_280, copy__default_281, copy__default_282, copy__default_283, copy__default_284, copy__default_285, copy__default_286, copy__default_287, copy__default_288, copy__default_289, copy__default_290, copy__default_291, copy__default_292, copy__default_293, copy__default_294, copy__default_295, copy__default_296, copy__default_297, copy__default_298, copy__default_299, copy__default_300, copy__default_301, copy__default_302, copy__default_303, copy__default_304, copy__default_305, copy__default_306, copy__default_307, copy__default_308, copy__default_309, copy__default_310, copy__default_311, copy__default_312, copy__default_313, copy__default_314, copy__default_315, copy__default_316, copy__default_317, copy__default_318, copy__default_319, copy__default_320, copy__default_321, copy__default_322, copy__default_323, copy__default_324, copy__default_325, copy__default_326, copy__default_327, copy__default_328, copy__default_329, copy__default_330, copy__default_331, copy__default_332, copy__default_333, copy__default_334, copy__default_335, copy__default_336, copy__default_337, copy__default_338, copy__default_339, copy__default_340, copy__default_341, copy__default_342, copy__default_343, copy__default_344, copy__default_345, copy__default_346, copy__default_347, copy__default_348, copy__default_349, copy__default_350, copy__default_351, copy__default_352, copy__default_353, copy__default_354, copy__default_355, copy__default_356, copy__default_357, copy__default_358, copy__default_359, copy__default_360, copy__default_361, copy__default_362, copy__default_363, copy__default_364, copy__default_365, copy__default_366, copy__default_367, copy__default_368, copy__default_369, copy__default_370, copy__default_371, copy__default_372, copy__default_373, copy__default_374, copy__default_375, copy__default_376, copy__default_377, copy__default_378, copy__default_379, copy__default_380, copy__default_381, copy__default_382, copy__default_383, copy__default_384, copy__default_385, copy__default_386, copy__default_387, copy__default_388, copy__default_389, copy__default_390, copy__default_391, copy__default_392, copy__default_393, copy__default_394, copy__default_395, copy__default_396, copy__default_397, copy__default_398, copy__default_399, copy__default_400, copy__default_401, copy__default_402, copy__default_403, copy__default_404, copy__default_405, copy__default_406, copy__default_407, copy__default_408, copy__default_409, copy__default_410, copy__default_411, copy__default_412, copy__default_413, copy__default_414, copy__default_415, copy__default_416, copy__default_417, copy__default_418, copy__default_419, copy__default_420, copy__default_421, copy__default_422, copy__default_423, copy__default_424, copy__default_425, copy__default_426, copy__default_427, copy__default_428, copy__default_429, copy__default_430, copy__default_431, copy__default_432, copy__default_433, copy__default_434, copy__default_435, copy__default_436, copy__default_437, copy__default_438, copy__default_439, copy__default_440, copy__default_441, copy__default_442, copy__default_443, copy__default_444, copy__default_445, copy__default_446, copy__default_447, copy__default_448, copy__default_449, copy__default_450, copy__default_451, copy__default_452, copy__default_453, copy__default_454, copy__default_455, copy__default_456, copy__default_457, copy__default_458, copy__default_459, copy__default_460, copy__default_461, copy__default_462, copy__default_463, copy__default_464, copy__default_465, copy__default_466, copy__default_467, copy__default_468, copy__default_469, copy__default_470, copy__default_471, copy__default_472, copy__default_473, copy__default_474, copy__default_475, copy__default_476, copy__default_477, copy__default_478, copy__default_479, copy__default_480, copy__default_481, copy__default_482, copy__default_483, copy__default_484, copy__default_485, copy__default_486, copy__default_487, copy__default_488, copy__default_489, copy__default_490, copy__default_491, copy__default_492, copy__default_493, copy__default_494, copy__default_495, copy__default_496, copy__default_497, copy__default_498, copy__default_499, copy__default_500, copy__default_501, copy__default_502, copy__default_503, copy__default_504, copy__default_505, copy__default_506, copy__default_507, copy__default_508, copy__default_509, copy__default_510, copy__default_511, copy__default_512, copy__default_513, copy__default_514, copy__default_515, copy__default_516, copy__default_517, copy__default_518, copy__default_519, copy__default_520, copy__default_521, copy__default_522, copy__default_523, copy__default_524, copy__default_525, copy__default_526, copy__default_527, copy__default_528, copy__default_529, copy__default_530, copy__default_531, copy__default_532, copy__default_533, copy__default_534, copy__default_535, copy__default_536, copy__default_537, copy__default_538, copy__default_539, copy__default_540, copy__default_541, copy__default_542, copy__default_543, copy__default_544, copy__default_545, copy__default_546, copy__default_547, copy__default_548, copy__default_549, copy__default_550, copy__default_551, copy__default_552, copy__default_553, copy__default_554, copy__default_555, copy__default_556, copy__default_557, copy__default_558, copy__default_559, copy__default_560, copy__default_561, copy__default_562, copy__default_563, copy__default_564, copy__default_565, copy__default_566, copy__default_567, copy__default_568, copy__default_569, copy__default_570, copy__default_571, copy__default_572, copy__default_573, copy__default_574, copy__default_575, copy__default_576, copy__default_577, copy__default_578, copy__default_579, copy__default_580, copy__default_581, copy__default_582, copy__default_583, copy__default_584, copy__default_585, copy__default_586, copy__default_587, copy__default_588, copy__default_589, copy__default_590, copy__default_591, copy__default_592, copy__default_593, copy__default_594, copy__default_595, copy__default_596, copy__default_597, copy__default_598, copy__default_599, copy__default_600, copy__default_601, copy__default_602, copy__default_603, copy__default_604, copy__default_605, copy__default_606, copy__default_607)


def _default_make_inputs():
    return [
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([192, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 197, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1, 1, 192], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([192, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([192, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([192, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([192, 3, 16, 16], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576, 192], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([576], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192, 192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768, 192], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192, 768], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1000, 192], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    torch.randn([1000], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

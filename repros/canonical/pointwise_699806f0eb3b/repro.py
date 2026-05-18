"""
Standalone repro captured via capture_hook.
Label: tlparse_torchbench_s9_g77
Pattern hash: 699806f0eb3b
Shape hash: d0fb7c18
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
    def forward(self, arg0_1: "f32[768]", arg1_1: "f32[50, 768]", arg2_1: "f32[768, 512]", arg3_1: "f32[768, 3, 32, 32]", arg4_1: "f32[768]", arg5_1: "f32[768]", arg6_1: "f32[2304, 768]", arg7_1: "f32[2304]", arg8_1: "f32[768, 768]", arg9_1: "f32[768]", arg10_1: "f32[3072, 768]", arg11_1: "f32[3072]", arg12_1: "f32[768, 3072]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[768]", arg18_1: "f32[2304, 768]", arg19_1: "f32[2304]", arg20_1: "f32[768, 768]", arg21_1: "f32[768]", arg22_1: "f32[3072, 768]", arg23_1: "f32[3072]", arg24_1: "f32[768, 3072]", arg25_1: "f32[768]", arg26_1: "f32[768]", arg27_1: "f32[768]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[2304, 768]", arg31_1: "f32[2304]", arg32_1: "f32[768, 768]", arg33_1: "f32[768]", arg34_1: "f32[3072, 768]", arg35_1: "f32[3072]", arg36_1: "f32[768, 3072]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[768]", arg42_1: "f32[2304, 768]", arg43_1: "f32[2304]", arg44_1: "f32[768, 768]", arg45_1: "f32[768]", arg46_1: "f32[3072, 768]", arg47_1: "f32[3072]", arg48_1: "f32[768, 3072]", arg49_1: "f32[768]", arg50_1: "f32[768]", arg51_1: "f32[768]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[2304, 768]", arg55_1: "f32[2304]", arg56_1: "f32[768, 768]", arg57_1: "f32[768]", arg58_1: "f32[3072, 768]", arg59_1: "f32[3072]", arg60_1: "f32[768, 3072]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[768]", arg66_1: "f32[2304, 768]", arg67_1: "f32[2304]", arg68_1: "f32[768, 768]", arg69_1: "f32[768]", arg70_1: "f32[3072, 768]", arg71_1: "f32[3072]", arg72_1: "f32[768, 3072]", arg73_1: "f32[768]", arg74_1: "f32[768]", arg75_1: "f32[768]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[2304, 768]", arg79_1: "f32[2304]", arg80_1: "f32[768, 768]", arg81_1: "f32[768]", arg82_1: "f32[3072, 768]", arg83_1: "f32[3072]", arg84_1: "f32[768, 3072]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[768]", arg90_1: "f32[2304, 768]", arg91_1: "f32[2304]", arg92_1: "f32[768, 768]", arg93_1: "f32[768]", arg94_1: "f32[3072, 768]", arg95_1: "f32[3072]", arg96_1: "f32[768, 3072]", arg97_1: "f32[768]", arg98_1: "f32[768]", arg99_1: "f32[768]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[2304, 768]", arg103_1: "f32[2304]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[3072, 768]", arg107_1: "f32[3072]", arg108_1: "f32[768, 3072]", arg109_1: "f32[768]", arg110_1: "f32[768]", arg111_1: "f32[768]", arg112_1: "f32[768]", arg113_1: "f32[768]", arg114_1: "f32[2304, 768]", arg115_1: "f32[2304]", arg116_1: "f32[768, 768]", arg117_1: "f32[768]", arg118_1: "f32[3072, 768]", arg119_1: "f32[3072]", arg120_1: "f32[768, 3072]", arg121_1: "f32[768]", arg122_1: "f32[768]", arg123_1: "f32[768]", arg124_1: "f32[768]", arg125_1: "f32[768]", arg126_1: "f32[2304, 768]", arg127_1: "f32[2304]", arg128_1: "f32[768, 768]", arg129_1: "f32[768]", arg130_1: "f32[3072, 768]", arg131_1: "f32[3072]", arg132_1: "f32[768, 3072]", arg133_1: "f32[768]", arg134_1: "f32[768]", arg135_1: "f32[768]", arg136_1: "f32[768]", arg137_1: "f32[768]", arg138_1: "f32[2304, 768]", arg139_1: "f32[2304]", arg140_1: "f32[768, 768]", arg141_1: "f32[768]", arg142_1: "f32[3072, 768]", arg143_1: "f32[3072]", arg144_1: "f32[768, 3072]", arg145_1: "f32[768]", arg146_1: "f32[768]", arg147_1: "f32[768]", arg148_1: "f32[768]", arg149_1: "f32[768]", arg150_1: "f32[768]", arg151_1: "f32[768]", arg152_1: "f32[77, 512]", arg153_1: "f32[49408, 512]", arg154_1: "f32[1536, 512]", arg155_1: "f32[1536]", arg156_1: "f32[512, 512]", arg157_1: "f32[512]", arg158_1: "f32[2048, 512]", arg159_1: "f32[2048]", arg160_1: "f32[512, 2048]", arg161_1: "f32[512]", arg162_1: "f32[512]", arg163_1: "f32[512]", arg164_1: "f32[512]", arg165_1: "f32[512]", arg166_1: "f32[1536, 512]", arg167_1: "f32[1536]", arg168_1: "f32[512, 512]", arg169_1: "f32[512]", arg170_1: "f32[2048, 512]", arg171_1: "f32[2048]", arg172_1: "f32[512, 2048]", arg173_1: "f32[512]", arg174_1: "f32[512]", arg175_1: "f32[512]", arg176_1: "f32[512]", arg177_1: "f32[512]", arg178_1: "f32[1536, 512]", arg179_1: "f32[1536]", arg180_1: "f32[512, 512]", arg181_1: "f32[512]", arg182_1: "f32[2048, 512]", arg183_1: "f32[2048]", arg184_1: "f32[512, 2048]", arg185_1: "f32[512]", arg186_1: "f32[512]", arg187_1: "f32[512]", arg188_1: "f32[512]", arg189_1: "f32[512]", arg190_1: "f32[1536, 512]", arg191_1: "f32[1536]", arg192_1: "f32[512, 512]", arg193_1: "f32[512]", arg194_1: "f32[2048, 512]", arg195_1: "f32[2048]", arg196_1: "f32[512, 2048]", arg197_1: "f32[512]", arg198_1: "f32[512]", arg199_1: "f32[512]", arg200_1: "f32[512]", arg201_1: "f32[512]", arg202_1: "f32[1536, 512]", arg203_1: "f32[1536]", arg204_1: "f32[512, 512]", arg205_1: "f32[512]", arg206_1: "f32[2048, 512]", arg207_1: "f32[2048]", arg208_1: "f32[512, 2048]", arg209_1: "f32[512]", arg210_1: "f32[512]", arg211_1: "f32[512]", arg212_1: "f32[512]", arg213_1: "f32[512]", arg214_1: "f32[1536, 512]", arg215_1: "f32[1536]", arg216_1: "f32[512, 512]", arg217_1: "f32[512]", arg218_1: "f32[2048, 512]", arg219_1: "f32[2048]", arg220_1: "f32[512, 2048]", arg221_1: "f32[512]", arg222_1: "f32[512]", arg223_1: "f32[512]", arg224_1: "f32[512]", arg225_1: "f32[512]", arg226_1: "f32[1536, 512]", arg227_1: "f32[1536]", arg228_1: "f32[512, 512]", arg229_1: "f32[512]", arg230_1: "f32[2048, 512]", arg231_1: "f32[2048]", arg232_1: "f32[512, 2048]", arg233_1: "f32[512]", arg234_1: "f32[512]", arg235_1: "f32[512]", arg236_1: "f32[512]", arg237_1: "f32[512]", arg238_1: "f32[1536, 512]", arg239_1: "f32[1536]", arg240_1: "f32[512, 512]", arg241_1: "f32[512]", arg242_1: "f32[2048, 512]", arg243_1: "f32[2048]", arg244_1: "f32[512, 2048]", arg245_1: "f32[512]", arg246_1: "f32[512]", arg247_1: "f32[512]", arg248_1: "f32[512]", arg249_1: "f32[512]", arg250_1: "f32[1536, 512]", arg251_1: "f32[1536]", arg252_1: "f32[512, 512]", arg253_1: "f32[512]", arg254_1: "f32[2048, 512]", arg255_1: "f32[2048]", arg256_1: "f32[512, 2048]", arg257_1: "f32[512]", arg258_1: "f32[512]", arg259_1: "f32[512]", arg260_1: "f32[512]", arg261_1: "f32[512]", arg262_1: "f32[1536, 512]", arg263_1: "f32[1536]", arg264_1: "f32[512, 512]", arg265_1: "f32[512]", arg266_1: "f32[2048, 512]", arg267_1: "f32[2048]", arg268_1: "f32[512, 2048]", arg269_1: "f32[512]", arg270_1: "f32[512]", arg271_1: "f32[512]", arg272_1: "f32[512]", arg273_1: "f32[512]", arg274_1: "f32[1536, 512]", arg275_1: "f32[1536]", arg276_1: "f32[512, 512]", arg277_1: "f32[512]", arg278_1: "f32[2048, 512]", arg279_1: "f32[2048]", arg280_1: "f32[512, 2048]", arg281_1: "f32[512]", arg282_1: "f32[512]", arg283_1: "f32[512]", arg284_1: "f32[512]", arg285_1: "f32[512]", arg286_1: "f32[1536, 512]", arg287_1: "f32[1536]", arg288_1: "f32[512, 512]", arg289_1: "f32[512]", arg290_1: "f32[2048, 512]", arg291_1: "f32[2048]", arg292_1: "f32[512, 2048]", arg293_1: "f32[512]", arg294_1: "f32[512]", arg295_1: "f32[512]", arg296_1: "f32[512]", arg297_1: "f32[512]", arg298_1: "f32[512]", arg299_1: "f32[512]", arg300_1: "f32[512, 512]", getitem_602: "f32[768]", getitem_603: "f32[50, 768]", getitem_604: "f32[768, 512]", getitem_605: "f32[768, 3, 32, 32]", getitem_606: "f32[768]", getitem_607: "f32[768]", getitem_608: "f32[2304, 768]", getitem_609: "f32[2304]", getitem_610: "f32[768, 768]", getitem_611: "f32[768]", getitem_612: "f32[3072, 768]", getitem_613: "f32[3072]", getitem_614: "f32[768, 3072]", getitem_615: "f32[768]", getitem_616: "f32[768]", getitem_617: "f32[768]", getitem_618: "f32[768]", getitem_619: "f32[768]", getitem_620: "f32[2304, 768]", getitem_621: "f32[2304]", getitem_622: "f32[768, 768]", getitem_623: "f32[768]", getitem_624: "f32[3072, 768]", getitem_625: "f32[3072]", getitem_626: "f32[768, 3072]", getitem_627: "f32[768]", getitem_628: "f32[768]", getitem_629: "f32[768]", getitem_630: "f32[768]", getitem_631: "f32[768]", getitem_632: "f32[2304, 768]", getitem_633: "f32[2304]", getitem_634: "f32[768, 768]", getitem_635: "f32[768]", getitem_636: "f32[3072, 768]", getitem_637: "f32[3072]", getitem_638: "f32[768, 3072]", getitem_639: "f32[768]", getitem_640: "f32[768]", getitem_641: "f32[768]", getitem_642: "f32[768]", getitem_643: "f32[768]", getitem_644: "f32[2304, 768]", getitem_645: "f32[2304]", getitem_646: "f32[768, 768]", getitem_647: "f32[768]", getitem_648: "f32[3072, 768]", getitem_649: "f32[3072]", getitem_650: "f32[768, 3072]", getitem_651: "f32[768]", getitem_652: "f32[768]", getitem_653: "f32[768]", getitem_654: "f32[768]", getitem_655: "f32[768]", getitem_656: "f32[2304, 768]", getitem_657: "f32[2304]", getitem_658: "f32[768, 768]", getitem_659: "f32[768]", getitem_660: "f32[3072, 768]", getitem_661: "f32[3072]", getitem_662: "f32[768, 3072]", getitem_663: "f32[768]", getitem_664: "f32[768]", getitem_665: "f32[768]", getitem_666: "f32[768]", getitem_667: "f32[768]", getitem_668: "f32[2304, 768]", getitem_669: "f32[2304]", getitem_670: "f32[768, 768]", getitem_671: "f32[768]", getitem_672: "f32[3072, 768]", getitem_673: "f32[3072]", getitem_674: "f32[768, 3072]", getitem_675: "f32[768]", getitem_676: "f32[768]", getitem_677: "f32[768]", getitem_678: "f32[768]", getitem_679: "f32[768]", getitem_680: "f32[2304, 768]", getitem_681: "f32[2304]", getitem_682: "f32[768, 768]", getitem_683: "f32[768]", getitem_684: "f32[3072, 768]", getitem_685: "f32[3072]", getitem_686: "f32[768, 3072]", getitem_687: "f32[768]", getitem_688: "f32[768]", getitem_689: "f32[768]", getitem_690: "f32[768]", getitem_691: "f32[768]", getitem_692: "f32[2304, 768]", getitem_693: "f32[2304]", getitem_694: "f32[768, 768]", getitem_695: "f32[768]", getitem_696: "f32[3072, 768]", getitem_697: "f32[3072]", getitem_698: "f32[768, 3072]", getitem_699: "f32[768]", getitem_700: "f32[768]", getitem_701: "f32[768]", getitem_702: "f32[768]", getitem_703: "f32[768]", getitem_704: "f32[2304, 768]", getitem_705: "f32[2304]", getitem_706: "f32[768, 768]", getitem_707: "f32[768]", getitem_708: "f32[3072, 768]", getitem_709: "f32[3072]", getitem_710: "f32[768, 3072]", getitem_711: "f32[768]", getitem_712: "f32[768]", getitem_713: "f32[768]", getitem_714: "f32[768]", getitem_715: "f32[768]", getitem_716: "f32[2304, 768]", getitem_717: "f32[2304]", getitem_718: "f32[768, 768]", getitem_719: "f32[768]", getitem_720: "f32[3072, 768]", getitem_721: "f32[3072]", getitem_722: "f32[768, 3072]", getitem_723: "f32[768]", getitem_724: "f32[768]", getitem_725: "f32[768]", getitem_726: "f32[768]", getitem_727: "f32[768]", getitem_728: "f32[2304, 768]", getitem_729: "f32[2304]", getitem_730: "f32[768, 768]", getitem_731: "f32[768]", getitem_732: "f32[3072, 768]", getitem_733: "f32[3072]", getitem_734: "f32[768, 3072]", getitem_735: "f32[768]", getitem_736: "f32[768]", getitem_737: "f32[768]", getitem_738: "f32[768]", getitem_739: "f32[768]", getitem_740: "f32[2304, 768]", getitem_741: "f32[2304]", getitem_742: "f32[768, 768]", getitem_743: "f32[768]", getitem_744: "f32[3072, 768]", getitem_745: "f32[3072]", getitem_746: "f32[768, 3072]", getitem_747: "f32[768]", getitem_748: "f32[768]", getitem_749: "f32[768]", getitem_750: "f32[768]", getitem_751: "f32[768]", getitem_752: "f32[768]", getitem_753: "f32[768]", getitem_754: "f32[77, 512]", getitem_755: "f32[49408, 512]", getitem_756: "f32[1536, 512]", getitem_757: "f32[1536]", getitem_758: "f32[512, 512]", getitem_759: "f32[512]", getitem_760: "f32[2048, 512]", getitem_761: "f32[2048]", getitem_762: "f32[512, 2048]", getitem_763: "f32[512]", getitem_764: "f32[512]", getitem_765: "f32[512]", getitem_766: "f32[512]", getitem_767: "f32[512]", getitem_768: "f32[1536, 512]", getitem_769: "f32[1536]", getitem_770: "f32[512, 512]", getitem_771: "f32[512]", getitem_772: "f32[2048, 512]", getitem_773: "f32[2048]", getitem_774: "f32[512, 2048]", getitem_775: "f32[512]", getitem_776: "f32[512]", getitem_777: "f32[512]", getitem_778: "f32[512]", getitem_779: "f32[512]", getitem_780: "f32[1536, 512]", getitem_781: "f32[1536]", getitem_782: "f32[512, 512]", getitem_783: "f32[512]", getitem_784: "f32[2048, 512]", getitem_785: "f32[2048]", getitem_786: "f32[512, 2048]", getitem_787: "f32[512]", getitem_788: "f32[512]", getitem_789: "f32[512]", getitem_790: "f32[512]", getitem_791: "f32[512]", getitem_792: "f32[1536, 512]", getitem_793: "f32[1536]", getitem_794: "f32[512, 512]", getitem_795: "f32[512]", getitem_796: "f32[2048, 512]", getitem_797: "f32[2048]", getitem_798: "f32[512, 2048]", getitem_799: "f32[512]", getitem_800: "f32[512]", getitem_801: "f32[512]", getitem_802: "f32[512]", getitem_803: "f32[512]", getitem_804: "f32[1536, 512]", getitem_805: "f32[1536]", getitem_806: "f32[512, 512]", getitem_807: "f32[512]", getitem_808: "f32[2048, 512]", getitem_809: "f32[2048]", getitem_810: "f32[512, 2048]", getitem_811: "f32[512]", getitem_812: "f32[512]", getitem_813: "f32[512]", getitem_814: "f32[512]", getitem_815: "f32[512]", getitem_816: "f32[1536, 512]", getitem_817: "f32[1536]", getitem_818: "f32[512, 512]", getitem_819: "f32[512]", getitem_820: "f32[2048, 512]", getitem_821: "f32[2048]", getitem_822: "f32[512, 2048]", getitem_823: "f32[512]", getitem_824: "f32[512]", getitem_825: "f32[512]", getitem_826: "f32[512]", getitem_827: "f32[512]", getitem_828: "f32[1536, 512]", getitem_829: "f32[1536]", getitem_830: "f32[512, 512]", getitem_831: "f32[512]", getitem_832: "f32[2048, 512]", getitem_833: "f32[2048]", getitem_834: "f32[512, 2048]", getitem_835: "f32[512]", getitem_836: "f32[512]", getitem_837: "f32[512]", getitem_838: "f32[512]", getitem_839: "f32[512]", getitem_840: "f32[1536, 512]", getitem_841: "f32[1536]", getitem_842: "f32[512, 512]", getitem_843: "f32[512]", getitem_844: "f32[2048, 512]", getitem_845: "f32[2048]", getitem_846: "f32[512, 2048]", getitem_847: "f32[512]", getitem_848: "f32[512]", getitem_849: "f32[512]", getitem_850: "f32[512]", getitem_851: "f32[512]", getitem_852: "f32[1536, 512]", getitem_853: "f32[1536]", getitem_854: "f32[512, 512]", getitem_855: "f32[512]", getitem_856: "f32[2048, 512]", getitem_857: "f32[2048]", getitem_858: "f32[512, 2048]", getitem_859: "f32[512]", getitem_860: "f32[512]", getitem_861: "f32[512]", getitem_862: "f32[512]", getitem_863: "f32[512]", getitem_864: "f32[1536, 512]", getitem_865: "f32[1536]", getitem_866: "f32[512, 512]", getitem_867: "f32[512]", getitem_868: "f32[2048, 512]", getitem_869: "f32[2048]", getitem_870: "f32[512, 2048]", getitem_871: "f32[512]", getitem_872: "f32[512]", getitem_873: "f32[512]", getitem_874: "f32[512]", getitem_875: "f32[512]", getitem_876: "f32[1536, 512]", getitem_877: "f32[1536]", getitem_878: "f32[512, 512]", getitem_879: "f32[512]", getitem_880: "f32[2048, 512]", getitem_881: "f32[2048]", getitem_882: "f32[512, 2048]", getitem_883: "f32[512]", getitem_884: "f32[512]", getitem_885: "f32[512]", getitem_886: "f32[512]", getitem_887: "f32[512]", getitem_888: "f32[1536, 512]", getitem_889: "f32[1536]", getitem_890: "f32[512, 512]", getitem_891: "f32[512]", getitem_892: "f32[2048, 512]", getitem_893: "f32[2048]", getitem_894: "f32[512, 2048]", getitem_895: "f32[512]", getitem_896: "f32[512]", getitem_897: "f32[512]", getitem_898: "f32[512]", getitem_899: "f32[512]", getitem_900: "f32[512]", getitem_901: "f32[512]", getitem_902: "f32[512, 512]", getitem_4816: "f32[768]", getitem_4817: "f32[50, 768]", getitem_4818: "f32[768, 512]", getitem_4819: "f32[768, 3, 32, 32]", getitem_4820: "f32[768]", getitem_4821: "f32[768]", getitem_4822: "f32[2304, 768]", getitem_4823: "f32[2304]", getitem_4824: "f32[768, 768]", getitem_4825: "f32[768]", getitem_4826: "f32[3072, 768]", getitem_4827: "f32[3072]", getitem_4828: "f32[768, 3072]", getitem_4829: "f32[768]", getitem_4830: "f32[768]", getitem_4831: "f32[768]", getitem_4832: "f32[768]", getitem_4833: "f32[768]", getitem_4834: "f32[2304, 768]", getitem_4835: "f32[2304]", getitem_4836: "f32[768, 768]", getitem_4837: "f32[768]", getitem_4838: "f32[3072, 768]", getitem_4839: "f32[3072]", getitem_4840: "f32[768, 3072]", getitem_4841: "f32[768]", getitem_4842: "f32[768]", getitem_4843: "f32[768]", getitem_4844: "f32[768]", getitem_4845: "f32[768]", getitem_4846: "f32[2304, 768]", getitem_4847: "f32[2304]", getitem_4848: "f32[768, 768]", getitem_4849: "f32[768]", getitem_4850: "f32[3072, 768]", getitem_4851: "f32[3072]", getitem_4852: "f32[768, 3072]", getitem_4853: "f32[768]", getitem_4854: "f32[768]", getitem_4855: "f32[768]", getitem_4856: "f32[768]", getitem_4857: "f32[768]", getitem_4858: "f32[2304, 768]", getitem_4859: "f32[2304]", getitem_4860: "f32[768, 768]", getitem_4861: "f32[768]", getitem_4862: "f32[3072, 768]", getitem_4863: "f32[3072]", getitem_4864: "f32[768, 3072]", getitem_4865: "f32[768]", getitem_4866: "f32[768]", getitem_4867: "f32[768]", getitem_4868: "f32[768]", getitem_4869: "f32[768]", getitem_4870: "f32[2304, 768]", getitem_4871: "f32[2304]", getitem_4872: "f32[768, 768]", getitem_4873: "f32[768]", getitem_4874: "f32[3072, 768]", getitem_4875: "f32[3072]", getitem_4876: "f32[768, 3072]", getitem_4877: "f32[768]", getitem_4878: "f32[768]", getitem_4879: "f32[768]", getitem_4880: "f32[768]", getitem_4881: "f32[768]", getitem_4882: "f32[2304, 768]", getitem_4883: "f32[2304]", getitem_4884: "f32[768, 768]", getitem_4885: "f32[768]", getitem_4886: "f32[3072, 768]", getitem_4887: "f32[3072]", getitem_4888: "f32[768, 3072]", getitem_4889: "f32[768]", getitem_4890: "f32[768]", getitem_4891: "f32[768]", getitem_4892: "f32[768]", getitem_4893: "f32[768]", getitem_4894: "f32[2304, 768]", getitem_4895: "f32[2304]", getitem_4896: "f32[768, 768]", getitem_4897: "f32[768]", getitem_4898: "f32[3072, 768]", getitem_4899: "f32[3072]", getitem_4900: "f32[768, 3072]", getitem_4901: "f32[768]", getitem_4902: "f32[768]", getitem_4903: "f32[768]", getitem_4904: "f32[768]", getitem_4905: "f32[768]", getitem_4906: "f32[2304, 768]", getitem_4907: "f32[2304]", getitem_4908: "f32[768, 768]", getitem_4909: "f32[768]", getitem_4910: "f32[3072, 768]", getitem_4911: "f32[3072]", getitem_4912: "f32[768, 3072]", getitem_4913: "f32[768]", getitem_4914: "f32[768]", getitem_4915: "f32[768]", getitem_4916: "f32[768]", getitem_4917: "f32[768]", getitem_4918: "f32[2304, 768]", getitem_4919: "f32[2304]", getitem_4920: "f32[768, 768]", getitem_4921: "f32[768]", getitem_4922: "f32[3072, 768]", getitem_4923: "f32[3072]", getitem_4924: "f32[768, 3072]", getitem_4925: "f32[768]", getitem_4926: "f32[768]", getitem_4927: "f32[768]", getitem_4928: "f32[768]", getitem_4929: "f32[768]", getitem_4930: "f32[2304, 768]", getitem_4931: "f32[2304]", getitem_4932: "f32[768, 768]", getitem_4933: "f32[768]", getitem_4934: "f32[3072, 768]", getitem_4935: "f32[3072]", getitem_4936: "f32[768, 3072]", getitem_4937: "f32[768]", getitem_4938: "f32[768]", getitem_4939: "f32[768]", getitem_4940: "f32[768]", getitem_4941: "f32[768]", getitem_4942: "f32[2304, 768]", getitem_4943: "f32[2304]", getitem_4944: "f32[768, 768]", getitem_4945: "f32[768]", getitem_4946: "f32[3072, 768]", getitem_4947: "f32[3072]", getitem_4948: "f32[768, 3072]", getitem_4949: "f32[768]", getitem_4950: "f32[768]", getitem_4951: "f32[768]", getitem_4952: "f32[768]", getitem_4953: "f32[768]", getitem_4954: "f32[2304, 768]", getitem_4955: "f32[2304]", getitem_4956: "f32[768, 768]", getitem_4957: "f32[768]", getitem_4958: "f32[3072, 768]", getitem_4959: "f32[3072]", getitem_4960: "f32[768, 3072]", getitem_4961: "f32[768]", getitem_4962: "f32[768]", getitem_4963: "f32[768]", getitem_4964: "f32[768]", getitem_4965: "f32[768]", getitem_4966: "f32[768]", getitem_4967: "f32[768]", getitem_4968: "f32[77, 512]", getitem_4969: "f32[49408, 512]", getitem_4970: "f32[1536, 512]", getitem_4971: "f32[1536]", getitem_4972: "f32[512, 512]", getitem_4973: "f32[512]", getitem_4974: "f32[2048, 512]", getitem_4975: "f32[2048]", getitem_4976: "f32[512, 2048]", getitem_4977: "f32[512]", getitem_4978: "f32[512]", getitem_4979: "f32[512]", getitem_4980: "f32[512]", getitem_4981: "f32[512]", getitem_4982: "f32[1536, 512]", getitem_4983: "f32[1536]", getitem_4984: "f32[512, 512]", getitem_4985: "f32[512]", getitem_4986: "f32[2048, 512]", getitem_4987: "f32[2048]", getitem_4988: "f32[512, 2048]", getitem_4989: "f32[512]", getitem_4990: "f32[512]", getitem_4991: "f32[512]", getitem_4992: "f32[512]", getitem_4993: "f32[512]", getitem_4994: "f32[1536, 512]", getitem_4995: "f32[1536]", getitem_4996: "f32[512, 512]", getitem_4997: "f32[512]", getitem_4998: "f32[2048, 512]", getitem_4999: "f32[2048]", getitem_5000: "f32[512, 2048]", getitem_5001: "f32[512]", getitem_5002: "f32[512]", getitem_5003: "f32[512]", getitem_5004: "f32[512]", getitem_5005: "f32[512]", getitem_5006: "f32[1536, 512]", getitem_5007: "f32[1536]", getitem_5008: "f32[512, 512]", getitem_5009: "f32[512]", getitem_5010: "f32[2048, 512]", getitem_5011: "f32[2048]", getitem_5012: "f32[512, 2048]", getitem_5013: "f32[512]", getitem_5014: "f32[512]", getitem_5015: "f32[512]", getitem_5016: "f32[512]", getitem_5017: "f32[512]", getitem_5018: "f32[1536, 512]", getitem_5019: "f32[1536]", getitem_5020: "f32[512, 512]", getitem_5021: "f32[512]", getitem_5022: "f32[2048, 512]", getitem_5023: "f32[2048]", getitem_5024: "f32[512, 2048]", getitem_5025: "f32[512]", getitem_5026: "f32[512]", getitem_5027: "f32[512]", getitem_5028: "f32[512]", getitem_5029: "f32[512]", getitem_5030: "f32[1536, 512]", getitem_5031: "f32[1536]", getitem_5032: "f32[512, 512]", getitem_5033: "f32[512]", getitem_5034: "f32[2048, 512]", getitem_5035: "f32[2048]", getitem_5036: "f32[512, 2048]", getitem_5037: "f32[512]", getitem_5038: "f32[512]", getitem_5039: "f32[512]", getitem_5040: "f32[512]", getitem_5041: "f32[512]", getitem_5042: "f32[1536, 512]", getitem_5043: "f32[1536]", getitem_5044: "f32[512, 512]", getitem_5045: "f32[512]", getitem_5046: "f32[2048, 512]", getitem_5047: "f32[2048]", getitem_5048: "f32[512, 2048]", getitem_5049: "f32[512]", getitem_5050: "f32[512]", getitem_5051: "f32[512]", getitem_5052: "f32[512]", getitem_5053: "f32[512]", getitem_5054: "f32[1536, 512]", getitem_5055: "f32[1536]", getitem_5056: "f32[512, 512]", getitem_5057: "f32[512]", getitem_5058: "f32[2048, 512]", getitem_5059: "f32[2048]", getitem_5060: "f32[512, 2048]", getitem_5061: "f32[512]", getitem_5062: "f32[512]", getitem_5063: "f32[512]", getitem_5064: "f32[512]", getitem_5065: "f32[512]", getitem_5066: "f32[1536, 512]", getitem_5067: "f32[1536]", getitem_5068: "f32[512, 512]", getitem_5069: "f32[512]", getitem_5070: "f32[2048, 512]", getitem_5071: "f32[2048]", getitem_5072: "f32[512, 2048]", getitem_5073: "f32[512]", getitem_5074: "f32[512]", getitem_5075: "f32[512]", getitem_5076: "f32[512]", getitem_5077: "f32[512]", getitem_5078: "f32[1536, 512]", getitem_5079: "f32[1536]", getitem_5080: "f32[512, 512]", getitem_5081: "f32[512]", getitem_5082: "f32[2048, 512]", getitem_5083: "f32[2048]", getitem_5084: "f32[512, 2048]", getitem_5085: "f32[512]", getitem_5086: "f32[512]", getitem_5087: "f32[512]", getitem_5088: "f32[512]", getitem_5089: "f32[512]", getitem_5090: "f32[1536, 512]", getitem_5091: "f32[1536]", getitem_5092: "f32[512, 512]", getitem_5093: "f32[512]", getitem_5094: "f32[2048, 512]", getitem_5095: "f32[2048]", getitem_5096: "f32[512, 2048]", getitem_5097: "f32[512]", getitem_5098: "f32[512]", getitem_5099: "f32[512]", getitem_5100: "f32[512]", getitem_5101: "f32[512]", getitem_5102: "f32[1536, 512]", getitem_5103: "f32[1536]", getitem_5104: "f32[512, 512]", getitem_5105: "f32[512]", getitem_5106: "f32[2048, 512]", getitem_5107: "f32[2048]", getitem_5108: "f32[512, 2048]", getitem_5109: "f32[512]", getitem_5110: "f32[512]", getitem_5111: "f32[512]", getitem_5112: "f32[512]", getitem_5113: "f32[512]", getitem_5114: "f32[512]", getitem_5115: "f32[512]", getitem_5116: "f32[512, 512]"):
        # No stacktrace found for following nodes
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1], [getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902], [getitem_4816, getitem_4817, getitem_4818, getitem_4819, getitem_4820, getitem_4821, getitem_4822, getitem_4823, getitem_4824, getitem_4825, getitem_4826, getitem_4827, getitem_4828, getitem_4829, getitem_4830, getitem_4831, getitem_4832, getitem_4833, getitem_4834, getitem_4835, getitem_4836, getitem_4837, getitem_4838, getitem_4839, getitem_4840, getitem_4841, getitem_4842, getitem_4843, getitem_4844, getitem_4845, getitem_4846, getitem_4847, getitem_4848, getitem_4849, getitem_4850, getitem_4851, getitem_4852, getitem_4853, getitem_4854, getitem_4855, getitem_4856, getitem_4857, getitem_4858, getitem_4859, getitem_4860, getitem_4861, getitem_4862, getitem_4863, getitem_4864, getitem_4865, getitem_4866, getitem_4867, getitem_4868, getitem_4869, getitem_4870, getitem_4871, getitem_4872, getitem_4873, getitem_4874, getitem_4875, getitem_4876, getitem_4877, getitem_4878, getitem_4879, getitem_4880, getitem_4881, getitem_4882, getitem_4883, getitem_4884, getitem_4885, getitem_4886, getitem_4887, getitem_4888, getitem_4889, getitem_4890, getitem_4891, getitem_4892, getitem_4893, getitem_4894, getitem_4895, getitem_4896, getitem_4897, getitem_4898, getitem_4899, getitem_4900, getitem_4901, getitem_4902, getitem_4903, getitem_4904, getitem_4905, getitem_4906, getitem_4907, getitem_4908, getitem_4909, getitem_4910, getitem_4911, getitem_4912, getitem_4913, getitem_4914, getitem_4915, getitem_4916, getitem_4917, getitem_4918, getitem_4919, getitem_4920, getitem_4921, getitem_4922, getitem_4923, getitem_4924, getitem_4925, getitem_4926, getitem_4927, getitem_4928, getitem_4929, getitem_4930, getitem_4931, getitem_4932, getitem_4933, getitem_4934, getitem_4935, getitem_4936, getitem_4937, getitem_4938, getitem_4939, getitem_4940, getitem_4941, getitem_4942, getitem_4943, getitem_4944, getitem_4945, getitem_4946, getitem_4947, getitem_4948, getitem_4949, getitem_4950, getitem_4951, getitem_4952, getitem_4953, getitem_4954, getitem_4955, getitem_4956, getitem_4957, getitem_4958, getitem_4959, getitem_4960, getitem_4961, getitem_4962, getitem_4963, getitem_4964, getitem_4965, getitem_4966, getitem_4967, getitem_4968, getitem_4969, getitem_4970, getitem_4971, getitem_4972, getitem_4973, getitem_4974, getitem_4975, getitem_4976, getitem_4977, getitem_4978, getitem_4979, getitem_4980, getitem_4981, getitem_4982, getitem_4983, getitem_4984, getitem_4985, getitem_4986, getitem_4987, getitem_4988, getitem_4989, getitem_4990, getitem_4991, getitem_4992, getitem_4993, getitem_4994, getitem_4995, getitem_4996, getitem_4997, getitem_4998, getitem_4999, getitem_5000, getitem_5001, getitem_5002, getitem_5003, getitem_5004, getitem_5005, getitem_5006, getitem_5007, getitem_5008, getitem_5009, getitem_5010, getitem_5011, getitem_5012, getitem_5013, getitem_5014, getitem_5015, getitem_5016, getitem_5017, getitem_5018, getitem_5019, getitem_5020, getitem_5021, getitem_5022, getitem_5023, getitem_5024, getitem_5025, getitem_5026, getitem_5027, getitem_5028, getitem_5029, getitem_5030, getitem_5031, getitem_5032, getitem_5033, getitem_5034, getitem_5035, getitem_5036, getitem_5037, getitem_5038, getitem_5039, getitem_5040, getitem_5041, getitem_5042, getitem_5043, getitem_5044, getitem_5045, getitem_5046, getitem_5047, getitem_5048, getitem_5049, getitem_5050, getitem_5051, getitem_5052, getitem_5053, getitem_5054, getitem_5055, getitem_5056, getitem_5057, getitem_5058, getitem_5059, getitem_5060, getitem_5061, getitem_5062, getitem_5063, getitem_5064, getitem_5065, getitem_5066, getitem_5067, getitem_5068, getitem_5069, getitem_5070, getitem_5071, getitem_5072, getitem_5073, getitem_5074, getitem_5075, getitem_5076, getitem_5077, getitem_5078, getitem_5079, getitem_5080, getitem_5081, getitem_5082, getitem_5083, getitem_5084, getitem_5085, getitem_5086, getitem_5087, getitem_5088, getitem_5089, getitem_5090, getitem_5091, getitem_5092, getitem_5093, getitem_5094, getitem_5095, getitem_5096, getitem_5097, getitem_5098, getitem_5099, getitem_5100, getitem_5101, getitem_5102, getitem_5103, getitem_5104, getitem_5105, getitem_5106, getitem_5107, getitem_5108, getitem_5109, getitem_5110, getitem_5111, getitem_5112, getitem_5113, getitem_5114, getitem_5115, getitem_5116]);  arg0_1 = arg1_1 = arg2_1 = arg3_1 = arg4_1 = arg5_1 = arg6_1 = arg7_1 = arg8_1 = arg9_1 = arg10_1 = arg11_1 = arg12_1 = arg13_1 = arg14_1 = arg15_1 = arg16_1 = arg17_1 = arg18_1 = arg19_1 = arg20_1 = arg21_1 = arg22_1 = arg23_1 = arg24_1 = arg25_1 = arg26_1 = arg27_1 = arg28_1 = arg29_1 = arg30_1 = arg31_1 = arg32_1 = arg33_1 = arg34_1 = arg35_1 = arg36_1 = arg37_1 = arg38_1 = arg39_1 = arg40_1 = arg41_1 = arg42_1 = arg43_1 = arg44_1 = arg45_1 = arg46_1 = arg47_1 = arg48_1 = arg49_1 = arg50_1 = arg51_1 = arg52_1 = arg53_1 = arg54_1 = arg55_1 = arg56_1 = arg57_1 = arg58_1 = arg59_1 = arg60_1 = arg61_1 = arg62_1 = arg63_1 = arg64_1 = arg65_1 = arg66_1 = arg67_1 = arg68_1 = arg69_1 = arg70_1 = arg71_1 = arg72_1 = arg73_1 = arg74_1 = arg75_1 = arg76_1 = arg77_1 = arg78_1 = arg79_1 = arg80_1 = arg81_1 = arg82_1 = arg83_1 = arg84_1 = arg85_1 = arg86_1 = arg87_1 = arg88_1 = arg89_1 = arg90_1 = arg91_1 = arg92_1 = arg93_1 = arg94_1 = arg95_1 = arg96_1 = arg97_1 = arg98_1 = arg99_1 = arg100_1 = arg101_1 = arg102_1 = arg103_1 = arg104_1 = arg105_1 = arg106_1 = arg107_1 = arg108_1 = arg109_1 = arg110_1 = arg111_1 = arg112_1 = arg113_1 = arg114_1 = arg115_1 = arg116_1 = arg117_1 = arg118_1 = arg119_1 = arg120_1 = arg121_1 = arg122_1 = arg123_1 = arg124_1 = arg125_1 = arg126_1 = arg127_1 = arg128_1 = arg129_1 = arg130_1 = arg131_1 = arg132_1 = arg133_1 = arg134_1 = arg135_1 = arg136_1 = arg137_1 = arg138_1 = arg139_1 = arg140_1 = arg141_1 = arg142_1 = arg143_1 = arg144_1 = arg145_1 = arg146_1 = arg147_1 = arg148_1 = arg149_1 = arg150_1 = arg151_1 = arg152_1 = arg153_1 = arg154_1 = arg155_1 = arg156_1 = arg157_1 = arg158_1 = arg159_1 = arg160_1 = arg161_1 = arg162_1 = arg163_1 = arg164_1 = arg165_1 = arg166_1 = arg167_1 = arg168_1 = arg169_1 = arg170_1 = arg171_1 = arg172_1 = arg173_1 = arg174_1 = arg175_1 = arg176_1 = arg177_1 = arg178_1 = arg179_1 = arg180_1 = arg181_1 = arg182_1 = arg183_1 = arg184_1 = arg185_1 = arg186_1 = arg187_1 = arg188_1 = arg189_1 = arg190_1 = arg191_1 = arg192_1 = arg193_1 = arg194_1 = arg195_1 = arg196_1 = arg197_1 = arg198_1 = arg199_1 = arg200_1 = arg201_1 = arg202_1 = arg203_1 = arg204_1 = arg205_1 = arg206_1 = arg207_1 = arg208_1 = arg209_1 = arg210_1 = arg211_1 = arg212_1 = arg213_1 = arg214_1 = arg215_1 = arg216_1 = arg217_1 = arg218_1 = arg219_1 = arg220_1 = arg221_1 = arg222_1 = arg223_1 = arg224_1 = arg225_1 = arg226_1 = arg227_1 = arg228_1 = arg229_1 = arg230_1 = arg231_1 = arg232_1 = arg233_1 = arg234_1 = arg235_1 = arg236_1 = arg237_1 = arg238_1 = arg239_1 = arg240_1 = arg241_1 = arg242_1 = arg243_1 = arg244_1 = arg245_1 = arg246_1 = arg247_1 = arg248_1 = arg249_1 = arg250_1 = arg251_1 = arg252_1 = arg253_1 = arg254_1 = arg255_1 = arg256_1 = arg257_1 = arg258_1 = arg259_1 = arg260_1 = arg261_1 = arg262_1 = arg263_1 = arg264_1 = arg265_1 = arg266_1 = arg267_1 = arg268_1 = arg269_1 = arg270_1 = arg271_1 = arg272_1 = arg273_1 = arg274_1 = arg275_1 = arg276_1 = arg277_1 = arg278_1 = arg279_1 = arg280_1 = arg281_1 = arg282_1 = arg283_1 = arg284_1 = arg285_1 = arg286_1 = arg287_1 = arg288_1 = arg289_1 = arg290_1 = arg291_1 = arg292_1 = arg293_1 = arg294_1 = arg295_1 = arg296_1 = arg297_1 = arg298_1 = arg299_1 = arg300_1 = getitem_602 = getitem_603 = getitem_604 = getitem_605 = getitem_606 = getitem_607 = getitem_608 = getitem_609 = getitem_610 = getitem_611 = getitem_612 = getitem_613 = getitem_614 = getitem_615 = getitem_616 = getitem_617 = getitem_618 = getitem_619 = getitem_620 = getitem_621 = getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = getitem_646 = getitem_647 = getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = getitem_654 = getitem_655 = getitem_656 = getitem_657 = getitem_658 = getitem_659 = getitem_660 = getitem_661 = getitem_662 = getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = getitem_671 = getitem_672 = getitem_673 = getitem_674 = getitem_675 = getitem_676 = getitem_677 = getitem_678 = getitem_679 = getitem_680 = getitem_681 = getitem_682 = getitem_683 = getitem_684 = getitem_685 = getitem_686 = getitem_687 = getitem_688 = getitem_689 = getitem_690 = getitem_691 = getitem_692 = getitem_693 = getitem_694 = getitem_695 = getitem_696 = getitem_697 = getitem_698 = getitem_699 = getitem_700 = getitem_701 = getitem_702 = getitem_703 = getitem_704 = getitem_705 = getitem_706 = getitem_707 = getitem_708 = getitem_709 = getitem_710 = getitem_711 = getitem_712 = getitem_713 = getitem_714 = getitem_715 = getitem_716 = getitem_717 = getitem_718 = getitem_719 = getitem_720 = getitem_721 = getitem_722 = getitem_723 = getitem_724 = getitem_725 = getitem_726 = getitem_727 = getitem_728 = getitem_729 = getitem_730 = getitem_731 = getitem_732 = getitem_733 = getitem_734 = getitem_735 = getitem_736 = getitem_737 = getitem_738 = getitem_739 = getitem_740 = getitem_741 = getitem_742 = getitem_743 = getitem_744 = getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = getitem_752 = getitem_753 = getitem_754 = getitem_755 = getitem_756 = getitem_757 = getitem_758 = getitem_759 = getitem_760 = getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = getitem_768 = getitem_769 = getitem_770 = getitem_771 = getitem_772 = getitem_773 = getitem_774 = getitem_775 = getitem_776 = getitem_777 = getitem_778 = getitem_779 = getitem_780 = getitem_781 = getitem_782 = getitem_783 = getitem_784 = getitem_785 = getitem_786 = getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = getitem_794 = getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = getitem_802 = getitem_803 = getitem_804 = getitem_805 = getitem_806 = getitem_807 = getitem_808 = getitem_809 = getitem_810 = getitem_811 = getitem_812 = getitem_813 = getitem_814 = getitem_815 = getitem_816 = getitem_817 = getitem_818 = getitem_819 = getitem_820 = getitem_821 = getitem_822 = getitem_823 = getitem_824 = getitem_825 = getitem_826 = getitem_827 = getitem_828 = getitem_829 = getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = getitem_835 = getitem_836 = getitem_837 = getitem_838 = getitem_839 = getitem_840 = getitem_841 = getitem_842 = getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = getitem_849 = getitem_850 = getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = getitem_857 = getitem_858 = getitem_859 = getitem_860 = getitem_861 = getitem_862 = getitem_863 = getitem_864 = getitem_865 = getitem_866 = getitem_867 = getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = getitem_873 = getitem_874 = getitem_875 = getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = getitem_900 = getitem_901 = getitem_902 = getitem_4816 = getitem_4817 = getitem_4818 = getitem_4819 = getitem_4820 = getitem_4821 = getitem_4822 = getitem_4823 = getitem_4824 = getitem_4825 = getitem_4826 = getitem_4827 = getitem_4828 = getitem_4829 = getitem_4830 = getitem_4831 = getitem_4832 = getitem_4833 = getitem_4834 = getitem_4835 = getitem_4836 = getitem_4837 = getitem_4838 = getitem_4839 = getitem_4840 = getitem_4841 = getitem_4842 = getitem_4843 = getitem_4844 = getitem_4845 = getitem_4846 = getitem_4847 = getitem_4848 = getitem_4849 = getitem_4850 = getitem_4851 = getitem_4852 = getitem_4853 = getitem_4854 = getitem_4855 = getitem_4856 = getitem_4857 = getitem_4858 = getitem_4859 = getitem_4860 = getitem_4861 = getitem_4862 = getitem_4863 = getitem_4864 = getitem_4865 = getitem_4866 = getitem_4867 = getitem_4868 = getitem_4869 = getitem_4870 = getitem_4871 = getitem_4872 = getitem_4873 = getitem_4874 = getitem_4875 = getitem_4876 = getitem_4877 = getitem_4878 = getitem_4879 = getitem_4880 = getitem_4881 = getitem_4882 = getitem_4883 = getitem_4884 = getitem_4885 = getitem_4886 = getitem_4887 = getitem_4888 = getitem_4889 = getitem_4890 = getitem_4891 = getitem_4892 = getitem_4893 = getitem_4894 = getitem_4895 = getitem_4896 = getitem_4897 = getitem_4898 = getitem_4899 = getitem_4900 = getitem_4901 = getitem_4902 = getitem_4903 = getitem_4904 = getitem_4905 = getitem_4906 = getitem_4907 = getitem_4908 = getitem_4909 = getitem_4910 = getitem_4911 = getitem_4912 = getitem_4913 = getitem_4914 = getitem_4915 = getitem_4916 = getitem_4917 = getitem_4918 = getitem_4919 = getitem_4920 = getitem_4921 = getitem_4922 = getitem_4923 = getitem_4924 = getitem_4925 = getitem_4926 = getitem_4927 = getitem_4928 = getitem_4929 = getitem_4930 = getitem_4931 = getitem_4932 = getitem_4933 = getitem_4934 = getitem_4935 = getitem_4936 = getitem_4937 = getitem_4938 = getitem_4939 = getitem_4940 = getitem_4941 = getitem_4942 = getitem_4943 = getitem_4944 = getitem_4945 = getitem_4946 = getitem_4947 = getitem_4948 = getitem_4949 = getitem_4950 = getitem_4951 = getitem_4952 = getitem_4953 = getitem_4954 = getitem_4955 = getitem_4956 = getitem_4957 = getitem_4958 = getitem_4959 = getitem_4960 = getitem_4961 = getitem_4962 = getitem_4963 = getitem_4964 = getitem_4965 = getitem_4966 = getitem_4967 = getitem_4968 = getitem_4969 = getitem_4970 = getitem_4971 = getitem_4972 = getitem_4973 = getitem_4974 = getitem_4975 = getitem_4976 = getitem_4977 = getitem_4978 = getitem_4979 = getitem_4980 = getitem_4981 = getitem_4982 = getitem_4983 = getitem_4984 = getitem_4985 = getitem_4986 = getitem_4987 = getitem_4988 = getitem_4989 = getitem_4990 = getitem_4991 = getitem_4992 = getitem_4993 = getitem_4994 = getitem_4995 = getitem_4996 = getitem_4997 = getitem_4998 = getitem_4999 = getitem_5000 = getitem_5001 = getitem_5002 = getitem_5003 = getitem_5004 = getitem_5005 = getitem_5006 = getitem_5007 = getitem_5008 = getitem_5009 = getitem_5010 = getitem_5011 = getitem_5012 = getitem_5013 = getitem_5014 = getitem_5015 = getitem_5016 = getitem_5017 = getitem_5018 = getitem_5019 = getitem_5020 = getitem_5021 = getitem_5022 = getitem_5023 = getitem_5024 = getitem_5025 = getitem_5026 = getitem_5027 = getitem_5028 = getitem_5029 = getitem_5030 = getitem_5031 = getitem_5032 = getitem_5033 = getitem_5034 = getitem_5035 = getitem_5036 = getitem_5037 = getitem_5038 = getitem_5039 = getitem_5040 = getitem_5041 = getitem_5042 = getitem_5043 = getitem_5044 = getitem_5045 = getitem_5046 = getitem_5047 = getitem_5048 = getitem_5049 = getitem_5050 = getitem_5051 = getitem_5052 = getitem_5053 = getitem_5054 = getitem_5055 = getitem_5056 = getitem_5057 = getitem_5058 = getitem_5059 = getitem_5060 = getitem_5061 = getitem_5062 = getitem_5063 = getitem_5064 = getitem_5065 = getitem_5066 = getitem_5067 = getitem_5068 = getitem_5069 = getitem_5070 = getitem_5071 = getitem_5072 = getitem_5073 = getitem_5074 = getitem_5075 = getitem_5076 = getitem_5077 = getitem_5078 = getitem_5079 = getitem_5080 = getitem_5081 = getitem_5082 = getitem_5083 = getitem_5084 = getitem_5085 = getitem_5086 = getitem_5087 = getitem_5088 = getitem_5089 = getitem_5090 = getitem_5091 = getitem_5092 = getitem_5093 = getitem_5094 = getitem_5095 = getitem_5096 = getitem_5097 = getitem_5098 = getitem_5099 = getitem_5100 = getitem_5101 = getitem_5102 = getitem_5103 = getitem_5104 = getitem_5105 = getitem_5106 = getitem_5107 = getitem_5108 = getitem_5109 = getitem_5110 = getitem_5111 = getitem_5112 = getitem_5113 = getitem_5114 = getitem_5115 = getitem_5116 = None
        getitem: "f32[768]" = _foreach_addcdiv_scalar[0]
        getitem_5117: "f32[50, 768]" = _foreach_addcdiv_scalar[1]
        getitem_5118: "f32[768, 512]" = _foreach_addcdiv_scalar[2]
        getitem_5119: "f32[768, 3, 32, 32]" = _foreach_addcdiv_scalar[3]
        getitem_5120: "f32[768]" = _foreach_addcdiv_scalar[4]
        getitem_5121: "f32[768]" = _foreach_addcdiv_scalar[5]
        getitem_5122: "f32[2304, 768]" = _foreach_addcdiv_scalar[6]
        getitem_5123: "f32[2304]" = _foreach_addcdiv_scalar[7]
        getitem_5124: "f32[768, 768]" = _foreach_addcdiv_scalar[8]
        getitem_5125: "f32[768]" = _foreach_addcdiv_scalar[9]
        getitem_5126: "f32[3072, 768]" = _foreach_addcdiv_scalar[10]
        getitem_5127: "f32[3072]" = _foreach_addcdiv_scalar[11]
        getitem_5128: "f32[768, 3072]" = _foreach_addcdiv_scalar[12]
        getitem_5129: "f32[768]" = _foreach_addcdiv_scalar[13]
        getitem_5130: "f32[768]" = _foreach_addcdiv_scalar[14]
        getitem_5131: "f32[768]" = _foreach_addcdiv_scalar[15]
        getitem_5132: "f32[768]" = _foreach_addcdiv_scalar[16]
        getitem_5133: "f32[768]" = _foreach_addcdiv_scalar[17]
        getitem_5134: "f32[2304, 768]" = _foreach_addcdiv_scalar[18]
        getitem_5135: "f32[2304]" = _foreach_addcdiv_scalar[19]
        getitem_5136: "f32[768, 768]" = _foreach_addcdiv_scalar[20]
        getitem_5137: "f32[768]" = _foreach_addcdiv_scalar[21]
        getitem_5138: "f32[3072, 768]" = _foreach_addcdiv_scalar[22]
        getitem_5139: "f32[3072]" = _foreach_addcdiv_scalar[23]
        getitem_5140: "f32[768, 3072]" = _foreach_addcdiv_scalar[24]
        getitem_5141: "f32[768]" = _foreach_addcdiv_scalar[25]
        getitem_5142: "f32[768]" = _foreach_addcdiv_scalar[26]
        getitem_5143: "f32[768]" = _foreach_addcdiv_scalar[27]
        getitem_5144: "f32[768]" = _foreach_addcdiv_scalar[28]
        getitem_5145: "f32[768]" = _foreach_addcdiv_scalar[29]
        getitem_5146: "f32[2304, 768]" = _foreach_addcdiv_scalar[30]
        getitem_5147: "f32[2304]" = _foreach_addcdiv_scalar[31]
        getitem_5148: "f32[768, 768]" = _foreach_addcdiv_scalar[32]
        getitem_5149: "f32[768]" = _foreach_addcdiv_scalar[33]
        getitem_5150: "f32[3072, 768]" = _foreach_addcdiv_scalar[34]
        getitem_5151: "f32[3072]" = _foreach_addcdiv_scalar[35]
        getitem_5152: "f32[768, 3072]" = _foreach_addcdiv_scalar[36]
        getitem_5153: "f32[768]" = _foreach_addcdiv_scalar[37]
        getitem_5154: "f32[768]" = _foreach_addcdiv_scalar[38]
        getitem_5155: "f32[768]" = _foreach_addcdiv_scalar[39]
        getitem_5156: "f32[768]" = _foreach_addcdiv_scalar[40]
        getitem_5157: "f32[768]" = _foreach_addcdiv_scalar[41]
        getitem_5158: "f32[2304, 768]" = _foreach_addcdiv_scalar[42]
        getitem_5159: "f32[2304]" = _foreach_addcdiv_scalar[43]
        getitem_5160: "f32[768, 768]" = _foreach_addcdiv_scalar[44]
        getitem_5161: "f32[768]" = _foreach_addcdiv_scalar[45]
        getitem_5162: "f32[3072, 768]" = _foreach_addcdiv_scalar[46]
        getitem_5163: "f32[3072]" = _foreach_addcdiv_scalar[47]
        getitem_5164: "f32[768, 3072]" = _foreach_addcdiv_scalar[48]
        getitem_5165: "f32[768]" = _foreach_addcdiv_scalar[49]
        getitem_5166: "f32[768]" = _foreach_addcdiv_scalar[50]
        getitem_5167: "f32[768]" = _foreach_addcdiv_scalar[51]
        getitem_5168: "f32[768]" = _foreach_addcdiv_scalar[52]
        getitem_5169: "f32[768]" = _foreach_addcdiv_scalar[53]
        getitem_5170: "f32[2304, 768]" = _foreach_addcdiv_scalar[54]
        getitem_5171: "f32[2304]" = _foreach_addcdiv_scalar[55]
        getitem_5172: "f32[768, 768]" = _foreach_addcdiv_scalar[56]
        getitem_5173: "f32[768]" = _foreach_addcdiv_scalar[57]
        getitem_5174: "f32[3072, 768]" = _foreach_addcdiv_scalar[58]
        getitem_5175: "f32[3072]" = _foreach_addcdiv_scalar[59]
        getitem_5176: "f32[768, 3072]" = _foreach_addcdiv_scalar[60]
        getitem_5177: "f32[768]" = _foreach_addcdiv_scalar[61]
        getitem_5178: "f32[768]" = _foreach_addcdiv_scalar[62]
        getitem_5179: "f32[768]" = _foreach_addcdiv_scalar[63]
        getitem_5180: "f32[768]" = _foreach_addcdiv_scalar[64]
        getitem_5181: "f32[768]" = _foreach_addcdiv_scalar[65]
        getitem_5182: "f32[2304, 768]" = _foreach_addcdiv_scalar[66]
        getitem_5183: "f32[2304]" = _foreach_addcdiv_scalar[67]
        getitem_5184: "f32[768, 768]" = _foreach_addcdiv_scalar[68]
        getitem_5185: "f32[768]" = _foreach_addcdiv_scalar[69]
        getitem_5186: "f32[3072, 768]" = _foreach_addcdiv_scalar[70]
        getitem_5187: "f32[3072]" = _foreach_addcdiv_scalar[71]
        getitem_5188: "f32[768, 3072]" = _foreach_addcdiv_scalar[72]
        getitem_5189: "f32[768]" = _foreach_addcdiv_scalar[73]
        getitem_5190: "f32[768]" = _foreach_addcdiv_scalar[74]
        getitem_5191: "f32[768]" = _foreach_addcdiv_scalar[75]
        getitem_5192: "f32[768]" = _foreach_addcdiv_scalar[76]
        getitem_5193: "f32[768]" = _foreach_addcdiv_scalar[77]
        getitem_5194: "f32[2304, 768]" = _foreach_addcdiv_scalar[78]
        getitem_5195: "f32[2304]" = _foreach_addcdiv_scalar[79]
        getitem_5196: "f32[768, 768]" = _foreach_addcdiv_scalar[80]
        getitem_5197: "f32[768]" = _foreach_addcdiv_scalar[81]
        getitem_5198: "f32[3072, 768]" = _foreach_addcdiv_scalar[82]
        getitem_5199: "f32[3072]" = _foreach_addcdiv_scalar[83]
        getitem_5200: "f32[768, 3072]" = _foreach_addcdiv_scalar[84]
        getitem_5201: "f32[768]" = _foreach_addcdiv_scalar[85]
        getitem_5202: "f32[768]" = _foreach_addcdiv_scalar[86]
        getitem_5203: "f32[768]" = _foreach_addcdiv_scalar[87]
        getitem_5204: "f32[768]" = _foreach_addcdiv_scalar[88]
        getitem_5205: "f32[768]" = _foreach_addcdiv_scalar[89]
        getitem_5206: "f32[2304, 768]" = _foreach_addcdiv_scalar[90]
        getitem_5207: "f32[2304]" = _foreach_addcdiv_scalar[91]
        getitem_5208: "f32[768, 768]" = _foreach_addcdiv_scalar[92]
        getitem_5209: "f32[768]" = _foreach_addcdiv_scalar[93]
        getitem_5210: "f32[3072, 768]" = _foreach_addcdiv_scalar[94]
        getitem_5211: "f32[3072]" = _foreach_addcdiv_scalar[95]
        getitem_5212: "f32[768, 3072]" = _foreach_addcdiv_scalar[96]
        getitem_5213: "f32[768]" = _foreach_addcdiv_scalar[97]
        getitem_5214: "f32[768]" = _foreach_addcdiv_scalar[98]
        getitem_5215: "f32[768]" = _foreach_addcdiv_scalar[99]
        getitem_5216: "f32[768]" = _foreach_addcdiv_scalar[100]
        getitem_5217: "f32[768]" = _foreach_addcdiv_scalar[101]
        getitem_5218: "f32[2304, 768]" = _foreach_addcdiv_scalar[102]
        getitem_5219: "f32[2304]" = _foreach_addcdiv_scalar[103]
        getitem_5220: "f32[768, 768]" = _foreach_addcdiv_scalar[104]
        getitem_5221: "f32[768]" = _foreach_addcdiv_scalar[105]
        getitem_5222: "f32[3072, 768]" = _foreach_addcdiv_scalar[106]
        getitem_5223: "f32[3072]" = _foreach_addcdiv_scalar[107]
        getitem_5224: "f32[768, 3072]" = _foreach_addcdiv_scalar[108]
        getitem_5225: "f32[768]" = _foreach_addcdiv_scalar[109]
        getitem_5226: "f32[768]" = _foreach_addcdiv_scalar[110]
        getitem_5227: "f32[768]" = _foreach_addcdiv_scalar[111]
        getitem_5228: "f32[768]" = _foreach_addcdiv_scalar[112]
        getitem_5229: "f32[768]" = _foreach_addcdiv_scalar[113]
        getitem_5230: "f32[2304, 768]" = _foreach_addcdiv_scalar[114]
        getitem_5231: "f32[2304]" = _foreach_addcdiv_scalar[115]
        getitem_5232: "f32[768, 768]" = _foreach_addcdiv_scalar[116]
        getitem_5233: "f32[768]" = _foreach_addcdiv_scalar[117]
        getitem_5234: "f32[3072, 768]" = _foreach_addcdiv_scalar[118]
        getitem_5235: "f32[3072]" = _foreach_addcdiv_scalar[119]
        getitem_5236: "f32[768, 3072]" = _foreach_addcdiv_scalar[120]
        getitem_5237: "f32[768]" = _foreach_addcdiv_scalar[121]
        getitem_5238: "f32[768]" = _foreach_addcdiv_scalar[122]
        getitem_5239: "f32[768]" = _foreach_addcdiv_scalar[123]
        getitem_5240: "f32[768]" = _foreach_addcdiv_scalar[124]
        getitem_5241: "f32[768]" = _foreach_addcdiv_scalar[125]
        getitem_5242: "f32[2304, 768]" = _foreach_addcdiv_scalar[126]
        getitem_5243: "f32[2304]" = _foreach_addcdiv_scalar[127]
        getitem_5244: "f32[768, 768]" = _foreach_addcdiv_scalar[128]
        getitem_5245: "f32[768]" = _foreach_addcdiv_scalar[129]
        getitem_5246: "f32[3072, 768]" = _foreach_addcdiv_scalar[130]
        getitem_5247: "f32[3072]" = _foreach_addcdiv_scalar[131]
        getitem_5248: "f32[768, 3072]" = _foreach_addcdiv_scalar[132]
        getitem_5249: "f32[768]" = _foreach_addcdiv_scalar[133]
        getitem_5250: "f32[768]" = _foreach_addcdiv_scalar[134]
        getitem_5251: "f32[768]" = _foreach_addcdiv_scalar[135]
        getitem_5252: "f32[768]" = _foreach_addcdiv_scalar[136]
        getitem_5253: "f32[768]" = _foreach_addcdiv_scalar[137]
        getitem_5254: "f32[2304, 768]" = _foreach_addcdiv_scalar[138]
        getitem_5255: "f32[2304]" = _foreach_addcdiv_scalar[139]
        getitem_5256: "f32[768, 768]" = _foreach_addcdiv_scalar[140]
        getitem_5257: "f32[768]" = _foreach_addcdiv_scalar[141]
        getitem_5258: "f32[3072, 768]" = _foreach_addcdiv_scalar[142]
        getitem_5259: "f32[3072]" = _foreach_addcdiv_scalar[143]
        getitem_5260: "f32[768, 3072]" = _foreach_addcdiv_scalar[144]
        getitem_5261: "f32[768]" = _foreach_addcdiv_scalar[145]
        getitem_5262: "f32[768]" = _foreach_addcdiv_scalar[146]
        getitem_5263: "f32[768]" = _foreach_addcdiv_scalar[147]
        getitem_5264: "f32[768]" = _foreach_addcdiv_scalar[148]
        getitem_5265: "f32[768]" = _foreach_addcdiv_scalar[149]
        getitem_5266: "f32[768]" = _foreach_addcdiv_scalar[150]
        getitem_5267: "f32[768]" = _foreach_addcdiv_scalar[151]
        getitem_5268: "f32[77, 512]" = _foreach_addcdiv_scalar[152]
        getitem_5269: "f32[49408, 512]" = _foreach_addcdiv_scalar[153]
        getitem_5270: "f32[1536, 512]" = _foreach_addcdiv_scalar[154]
        getitem_5271: "f32[1536]" = _foreach_addcdiv_scalar[155]
        getitem_5272: "f32[512, 512]" = _foreach_addcdiv_scalar[156]
        getitem_5273: "f32[512]" = _foreach_addcdiv_scalar[157]
        getitem_5274: "f32[2048, 512]" = _foreach_addcdiv_scalar[158]
        getitem_5275: "f32[2048]" = _foreach_addcdiv_scalar[159]
        getitem_5276: "f32[512, 2048]" = _foreach_addcdiv_scalar[160]
        getitem_5277: "f32[512]" = _foreach_addcdiv_scalar[161]
        getitem_5278: "f32[512]" = _foreach_addcdiv_scalar[162]
        getitem_5279: "f32[512]" = _foreach_addcdiv_scalar[163]
        getitem_5280: "f32[512]" = _foreach_addcdiv_scalar[164]
        getitem_5281: "f32[512]" = _foreach_addcdiv_scalar[165]
        getitem_5282: "f32[1536, 512]" = _foreach_addcdiv_scalar[166]
        getitem_5283: "f32[1536]" = _foreach_addcdiv_scalar[167]
        getitem_5284: "f32[512, 512]" = _foreach_addcdiv_scalar[168]
        getitem_5285: "f32[512]" = _foreach_addcdiv_scalar[169]
        getitem_5286: "f32[2048, 512]" = _foreach_addcdiv_scalar[170]
        getitem_5287: "f32[2048]" = _foreach_addcdiv_scalar[171]
        getitem_5288: "f32[512, 2048]" = _foreach_addcdiv_scalar[172]
        getitem_5289: "f32[512]" = _foreach_addcdiv_scalar[173]
        getitem_5290: "f32[512]" = _foreach_addcdiv_scalar[174]
        getitem_5291: "f32[512]" = _foreach_addcdiv_scalar[175]
        getitem_5292: "f32[512]" = _foreach_addcdiv_scalar[176]
        getitem_5293: "f32[512]" = _foreach_addcdiv_scalar[177]
        getitem_5294: "f32[1536, 512]" = _foreach_addcdiv_scalar[178]
        getitem_5295: "f32[1536]" = _foreach_addcdiv_scalar[179]
        getitem_5296: "f32[512, 512]" = _foreach_addcdiv_scalar[180]
        getitem_5297: "f32[512]" = _foreach_addcdiv_scalar[181]
        getitem_5298: "f32[2048, 512]" = _foreach_addcdiv_scalar[182]
        getitem_5299: "f32[2048]" = _foreach_addcdiv_scalar[183]
        getitem_5300: "f32[512, 2048]" = _foreach_addcdiv_scalar[184]
        getitem_5301: "f32[512]" = _foreach_addcdiv_scalar[185]
        getitem_5302: "f32[512]" = _foreach_addcdiv_scalar[186]
        getitem_5303: "f32[512]" = _foreach_addcdiv_scalar[187]
        getitem_5304: "f32[512]" = _foreach_addcdiv_scalar[188]
        getitem_5305: "f32[512]" = _foreach_addcdiv_scalar[189]
        getitem_5306: "f32[1536, 512]" = _foreach_addcdiv_scalar[190]
        getitem_5307: "f32[1536]" = _foreach_addcdiv_scalar[191]
        getitem_5308: "f32[512, 512]" = _foreach_addcdiv_scalar[192]
        getitem_5309: "f32[512]" = _foreach_addcdiv_scalar[193]
        getitem_5310: "f32[2048, 512]" = _foreach_addcdiv_scalar[194]
        getitem_5311: "f32[2048]" = _foreach_addcdiv_scalar[195]
        getitem_5312: "f32[512, 2048]" = _foreach_addcdiv_scalar[196]
        getitem_5313: "f32[512]" = _foreach_addcdiv_scalar[197]
        getitem_5314: "f32[512]" = _foreach_addcdiv_scalar[198]
        getitem_5315: "f32[512]" = _foreach_addcdiv_scalar[199]
        getitem_5316: "f32[512]" = _foreach_addcdiv_scalar[200]
        getitem_5317: "f32[512]" = _foreach_addcdiv_scalar[201]
        getitem_5318: "f32[1536, 512]" = _foreach_addcdiv_scalar[202]
        getitem_5319: "f32[1536]" = _foreach_addcdiv_scalar[203]
        getitem_5320: "f32[512, 512]" = _foreach_addcdiv_scalar[204]
        getitem_5321: "f32[512]" = _foreach_addcdiv_scalar[205]
        getitem_5322: "f32[2048, 512]" = _foreach_addcdiv_scalar[206]
        getitem_5323: "f32[2048]" = _foreach_addcdiv_scalar[207]
        getitem_5324: "f32[512, 2048]" = _foreach_addcdiv_scalar[208]
        getitem_5325: "f32[512]" = _foreach_addcdiv_scalar[209]
        getitem_5326: "f32[512]" = _foreach_addcdiv_scalar[210]
        getitem_5327: "f32[512]" = _foreach_addcdiv_scalar[211]
        getitem_5328: "f32[512]" = _foreach_addcdiv_scalar[212]
        getitem_5329: "f32[512]" = _foreach_addcdiv_scalar[213]
        getitem_5330: "f32[1536, 512]" = _foreach_addcdiv_scalar[214]
        getitem_5331: "f32[1536]" = _foreach_addcdiv_scalar[215]
        getitem_5332: "f32[512, 512]" = _foreach_addcdiv_scalar[216]
        getitem_5333: "f32[512]" = _foreach_addcdiv_scalar[217]
        getitem_5334: "f32[2048, 512]" = _foreach_addcdiv_scalar[218]
        getitem_5335: "f32[2048]" = _foreach_addcdiv_scalar[219]
        getitem_5336: "f32[512, 2048]" = _foreach_addcdiv_scalar[220]
        getitem_5337: "f32[512]" = _foreach_addcdiv_scalar[221]
        getitem_5338: "f32[512]" = _foreach_addcdiv_scalar[222]
        getitem_5339: "f32[512]" = _foreach_addcdiv_scalar[223]
        getitem_5340: "f32[512]" = _foreach_addcdiv_scalar[224]
        getitem_5341: "f32[512]" = _foreach_addcdiv_scalar[225]
        getitem_5342: "f32[1536, 512]" = _foreach_addcdiv_scalar[226]
        getitem_5343: "f32[1536]" = _foreach_addcdiv_scalar[227]
        getitem_5344: "f32[512, 512]" = _foreach_addcdiv_scalar[228]
        getitem_5345: "f32[512]" = _foreach_addcdiv_scalar[229]
        getitem_5346: "f32[2048, 512]" = _foreach_addcdiv_scalar[230]
        getitem_5347: "f32[2048]" = _foreach_addcdiv_scalar[231]
        getitem_5348: "f32[512, 2048]" = _foreach_addcdiv_scalar[232]
        getitem_5349: "f32[512]" = _foreach_addcdiv_scalar[233]
        getitem_5350: "f32[512]" = _foreach_addcdiv_scalar[234]
        getitem_5351: "f32[512]" = _foreach_addcdiv_scalar[235]
        getitem_5352: "f32[512]" = _foreach_addcdiv_scalar[236]
        getitem_5353: "f32[512]" = _foreach_addcdiv_scalar[237]
        getitem_5354: "f32[1536, 512]" = _foreach_addcdiv_scalar[238]
        getitem_5355: "f32[1536]" = _foreach_addcdiv_scalar[239]
        getitem_5356: "f32[512, 512]" = _foreach_addcdiv_scalar[240]
        getitem_5357: "f32[512]" = _foreach_addcdiv_scalar[241]
        getitem_5358: "f32[2048, 512]" = _foreach_addcdiv_scalar[242]
        getitem_5359: "f32[2048]" = _foreach_addcdiv_scalar[243]
        getitem_5360: "f32[512, 2048]" = _foreach_addcdiv_scalar[244]
        getitem_5361: "f32[512]" = _foreach_addcdiv_scalar[245]
        getitem_5362: "f32[512]" = _foreach_addcdiv_scalar[246]
        getitem_5363: "f32[512]" = _foreach_addcdiv_scalar[247]
        getitem_5364: "f32[512]" = _foreach_addcdiv_scalar[248]
        getitem_5365: "f32[512]" = _foreach_addcdiv_scalar[249]
        getitem_5366: "f32[1536, 512]" = _foreach_addcdiv_scalar[250]
        getitem_5367: "f32[1536]" = _foreach_addcdiv_scalar[251]
        getitem_5368: "f32[512, 512]" = _foreach_addcdiv_scalar[252]
        getitem_5369: "f32[512]" = _foreach_addcdiv_scalar[253]
        getitem_5370: "f32[2048, 512]" = _foreach_addcdiv_scalar[254]
        getitem_5371: "f32[2048]" = _foreach_addcdiv_scalar[255]
        getitem_5372: "f32[512, 2048]" = _foreach_addcdiv_scalar[256]
        getitem_5373: "f32[512]" = _foreach_addcdiv_scalar[257]
        getitem_5374: "f32[512]" = _foreach_addcdiv_scalar[258]
        getitem_5375: "f32[512]" = _foreach_addcdiv_scalar[259]
        getitem_5376: "f32[512]" = _foreach_addcdiv_scalar[260]
        getitem_5377: "f32[512]" = _foreach_addcdiv_scalar[261]
        getitem_5378: "f32[1536, 512]" = _foreach_addcdiv_scalar[262]
        getitem_5379: "f32[1536]" = _foreach_addcdiv_scalar[263]
        getitem_5380: "f32[512, 512]" = _foreach_addcdiv_scalar[264]
        getitem_5381: "f32[512]" = _foreach_addcdiv_scalar[265]
        getitem_5382: "f32[2048, 512]" = _foreach_addcdiv_scalar[266]
        getitem_5383: "f32[2048]" = _foreach_addcdiv_scalar[267]
        getitem_5384: "f32[512, 2048]" = _foreach_addcdiv_scalar[268]
        getitem_5385: "f32[512]" = _foreach_addcdiv_scalar[269]
        getitem_5386: "f32[512]" = _foreach_addcdiv_scalar[270]
        getitem_5387: "f32[512]" = _foreach_addcdiv_scalar[271]
        getitem_5388: "f32[512]" = _foreach_addcdiv_scalar[272]
        getitem_5389: "f32[512]" = _foreach_addcdiv_scalar[273]
        getitem_5390: "f32[1536, 512]" = _foreach_addcdiv_scalar[274]
        getitem_5391: "f32[1536]" = _foreach_addcdiv_scalar[275]
        getitem_5392: "f32[512, 512]" = _foreach_addcdiv_scalar[276]
        getitem_5393: "f32[512]" = _foreach_addcdiv_scalar[277]
        getitem_5394: "f32[2048, 512]" = _foreach_addcdiv_scalar[278]
        getitem_5395: "f32[2048]" = _foreach_addcdiv_scalar[279]
        getitem_5396: "f32[512, 2048]" = _foreach_addcdiv_scalar[280]
        getitem_5397: "f32[512]" = _foreach_addcdiv_scalar[281]
        getitem_5398: "f32[512]" = _foreach_addcdiv_scalar[282]
        getitem_5399: "f32[512]" = _foreach_addcdiv_scalar[283]
        getitem_5400: "f32[512]" = _foreach_addcdiv_scalar[284]
        getitem_5401: "f32[512]" = _foreach_addcdiv_scalar[285]
        getitem_5402: "f32[1536, 512]" = _foreach_addcdiv_scalar[286]
        getitem_5403: "f32[1536]" = _foreach_addcdiv_scalar[287]
        getitem_5404: "f32[512, 512]" = _foreach_addcdiv_scalar[288]
        getitem_5405: "f32[512]" = _foreach_addcdiv_scalar[289]
        getitem_5406: "f32[2048, 512]" = _foreach_addcdiv_scalar[290]
        getitem_5407: "f32[2048]" = _foreach_addcdiv_scalar[291]
        getitem_5408: "f32[512, 2048]" = _foreach_addcdiv_scalar[292]
        getitem_5409: "f32[512]" = _foreach_addcdiv_scalar[293]
        getitem_5410: "f32[512]" = _foreach_addcdiv_scalar[294]
        getitem_5411: "f32[512]" = _foreach_addcdiv_scalar[295]
        getitem_5412: "f32[512]" = _foreach_addcdiv_scalar[296]
        getitem_5413: "f32[512]" = _foreach_addcdiv_scalar[297]
        getitem_5414: "f32[512]" = _foreach_addcdiv_scalar[298]
        getitem_5415: "f32[512]" = _foreach_addcdiv_scalar[299]
        getitem_5416: "f32[512, 512]" = _foreach_addcdiv_scalar[300];  _foreach_addcdiv_scalar = None
        return (getitem, getitem_5117, getitem_5118, getitem_5119, getitem_5120, getitem_5121, getitem_5122, getitem_5123, getitem_5124, getitem_5125, getitem_5126, getitem_5127, getitem_5128, getitem_5129, getitem_5130, getitem_5131, getitem_5132, getitem_5133, getitem_5134, getitem_5135, getitem_5136, getitem_5137, getitem_5138, getitem_5139, getitem_5140, getitem_5141, getitem_5142, getitem_5143, getitem_5144, getitem_5145, getitem_5146, getitem_5147, getitem_5148, getitem_5149, getitem_5150, getitem_5151, getitem_5152, getitem_5153, getitem_5154, getitem_5155, getitem_5156, getitem_5157, getitem_5158, getitem_5159, getitem_5160, getitem_5161, getitem_5162, getitem_5163, getitem_5164, getitem_5165, getitem_5166, getitem_5167, getitem_5168, getitem_5169, getitem_5170, getitem_5171, getitem_5172, getitem_5173, getitem_5174, getitem_5175, getitem_5176, getitem_5177, getitem_5178, getitem_5179, getitem_5180, getitem_5181, getitem_5182, getitem_5183, getitem_5184, getitem_5185, getitem_5186, getitem_5187, getitem_5188, getitem_5189, getitem_5190, getitem_5191, getitem_5192, getitem_5193, getitem_5194, getitem_5195, getitem_5196, getitem_5197, getitem_5198, getitem_5199, getitem_5200, getitem_5201, getitem_5202, getitem_5203, getitem_5204, getitem_5205, getitem_5206, getitem_5207, getitem_5208, getitem_5209, getitem_5210, getitem_5211, getitem_5212, getitem_5213, getitem_5214, getitem_5215, getitem_5216, getitem_5217, getitem_5218, getitem_5219, getitem_5220, getitem_5221, getitem_5222, getitem_5223, getitem_5224, getitem_5225, getitem_5226, getitem_5227, getitem_5228, getitem_5229, getitem_5230, getitem_5231, getitem_5232, getitem_5233, getitem_5234, getitem_5235, getitem_5236, getitem_5237, getitem_5238, getitem_5239, getitem_5240, getitem_5241, getitem_5242, getitem_5243, getitem_5244, getitem_5245, getitem_5246, getitem_5247, getitem_5248, getitem_5249, getitem_5250, getitem_5251, getitem_5252, getitem_5253, getitem_5254, getitem_5255, getitem_5256, getitem_5257, getitem_5258, getitem_5259, getitem_5260, getitem_5261, getitem_5262, getitem_5263, getitem_5264, getitem_5265, getitem_5266, getitem_5267, getitem_5268, getitem_5269, getitem_5270, getitem_5271, getitem_5272, getitem_5273, getitem_5274, getitem_5275, getitem_5276, getitem_5277, getitem_5278, getitem_5279, getitem_5280, getitem_5281, getitem_5282, getitem_5283, getitem_5284, getitem_5285, getitem_5286, getitem_5287, getitem_5288, getitem_5289, getitem_5290, getitem_5291, getitem_5292, getitem_5293, getitem_5294, getitem_5295, getitem_5296, getitem_5297, getitem_5298, getitem_5299, getitem_5300, getitem_5301, getitem_5302, getitem_5303, getitem_5304, getitem_5305, getitem_5306, getitem_5307, getitem_5308, getitem_5309, getitem_5310, getitem_5311, getitem_5312, getitem_5313, getitem_5314, getitem_5315, getitem_5316, getitem_5317, getitem_5318, getitem_5319, getitem_5320, getitem_5321, getitem_5322, getitem_5323, getitem_5324, getitem_5325, getitem_5326, getitem_5327, getitem_5328, getitem_5329, getitem_5330, getitem_5331, getitem_5332, getitem_5333, getitem_5334, getitem_5335, getitem_5336, getitem_5337, getitem_5338, getitem_5339, getitem_5340, getitem_5341, getitem_5342, getitem_5343, getitem_5344, getitem_5345, getitem_5346, getitem_5347, getitem_5348, getitem_5349, getitem_5350, getitem_5351, getitem_5352, getitem_5353, getitem_5354, getitem_5355, getitem_5356, getitem_5357, getitem_5358, getitem_5359, getitem_5360, getitem_5361, getitem_5362, getitem_5363, getitem_5364, getitem_5365, getitem_5366, getitem_5367, getitem_5368, getitem_5369, getitem_5370, getitem_5371, getitem_5372, getitem_5373, getitem_5374, getitem_5375, getitem_5376, getitem_5377, getitem_5378, getitem_5379, getitem_5380, getitem_5381, getitem_5382, getitem_5383, getitem_5384, getitem_5385, getitem_5386, getitem_5387, getitem_5388, getitem_5389, getitem_5390, getitem_5391, getitem_5392, getitem_5393, getitem_5394, getitem_5395, getitem_5396, getitem_5397, getitem_5398, getitem_5399, getitem_5400, getitem_5401, getitem_5402, getitem_5403, getitem_5404, getitem_5405, getitem_5406, getitem_5407, getitem_5408, getitem_5409, getitem_5410, getitem_5411, getitem_5412, getitem_5413, getitem_5414, getitem_5415, getitem_5416)


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
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

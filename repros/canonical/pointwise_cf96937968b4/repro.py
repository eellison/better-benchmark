"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s4_g59
Pattern hash: cf96937968b4
Shape hash: 69aba55a
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg135_1: "f32[32128, 512]", arg133_1: "f32[512, 512]", arg395_1: "f32[512, 512]", arg396_1: "f32[512, 512]", arg397_1: "f32[512, 512]", arg398_1: "f32[32, 8]", arg399_1: "f32[512]", arg400_1: "f32[2048, 512]", arg401_1: "f32[512, 2048]", arg402_1: "f32[512]", arg403_1: "f32[512, 512]", arg404_1: "f32[512, 512]", arg405_1: "f32[512, 512]", arg406_1: "f32[512, 512]", arg407_1: "f32[512]", arg408_1: "f32[2048, 512]", arg409_1: "f32[512, 2048]", arg410_1: "f32[512]", arg411_1: "f32[512, 512]", arg412_1: "f32[512, 512]", arg413_1: "f32[512, 512]", arg414_1: "f32[512, 512]", arg415_1: "f32[512]", arg416_1: "f32[2048, 512]", arg417_1: "f32[512, 2048]", arg418_1: "f32[512]", arg419_1: "f32[512, 512]", arg420_1: "f32[512, 512]", arg421_1: "f32[512, 512]", arg422_1: "f32[512, 512]", arg423_1: "f32[512]", arg424_1: "f32[2048, 512]", arg425_1: "f32[512, 2048]", arg426_1: "f32[512]", arg427_1: "f32[512, 512]", arg428_1: "f32[512, 512]", arg429_1: "f32[512, 512]", arg430_1: "f32[512, 512]", arg431_1: "f32[512]", arg432_1: "f32[2048, 512]", arg433_1: "f32[512, 2048]", arg434_1: "f32[512]", arg435_1: "f32[512, 512]", arg436_1: "f32[512, 512]", arg437_1: "f32[512, 512]", arg438_1: "f32[512, 512]", arg439_1: "f32[512]", arg440_1: "f32[2048, 512]", arg441_1: "f32[512, 2048]", arg442_1: "f32[512]", arg443_1: "f32[512]", arg444_1: "f32[512, 512]", arg445_1: "f32[512, 512]", arg446_1: "f32[512, 512]", arg447_1: "f32[512, 512]", arg448_1: "f32[32, 8]", arg449_1: "f32[512]", arg450_1: "f32[512, 512]", arg451_1: "f32[512, 512]", arg452_1: "f32[512, 512]", arg453_1: "f32[512, 512]", arg454_1: "f32[512]", arg455_1: "f32[2048, 512]", arg456_1: "f32[512, 2048]", arg457_1: "f32[512]", arg458_1: "f32[512, 512]", arg459_1: "f32[512, 512]", arg460_1: "f32[512, 512]", arg461_1: "f32[512, 512]", arg462_1: "f32[512]", arg463_1: "f32[512, 512]", arg464_1: "f32[512, 512]", arg465_1: "f32[512, 512]", arg466_1: "f32[512, 512]", arg467_1: "f32[512]", arg468_1: "f32[2048, 512]", arg469_1: "f32[512, 2048]", arg470_1: "f32[512]", arg471_1: "f32[512, 512]", arg472_1: "f32[512, 512]", arg473_1: "f32[512, 512]", arg474_1: "f32[512, 512]", arg475_1: "f32[512]", arg476_1: "f32[512, 512]", arg477_1: "f32[512, 512]", arg478_1: "f32[512, 512]", arg479_1: "f32[512, 512]", arg480_1: "f32[512]", arg481_1: "f32[2048, 512]", arg482_1: "f32[512, 2048]", arg483_1: "f32[512]", arg484_1: "f32[512, 512]", arg485_1: "f32[512, 512]", arg486_1: "f32[512, 512]", arg487_1: "f32[512, 512]", arg488_1: "f32[512]", arg489_1: "f32[512, 512]", arg490_1: "f32[512, 512]", arg491_1: "f32[512, 512]", arg492_1: "f32[512, 512]", arg493_1: "f32[512]", arg494_1: "f32[2048, 512]", arg495_1: "f32[512, 2048]", arg496_1: "f32[512]", arg497_1: "f32[512, 512]", arg498_1: "f32[512, 512]", arg499_1: "f32[512, 512]", arg500_1: "f32[512, 512]", arg501_1: "f32[512]", arg502_1: "f32[512, 512]", arg503_1: "f32[512, 512]", arg504_1: "f32[512, 512]", arg505_1: "f32[512, 512]", arg506_1: "f32[512]", arg507_1: "f32[2048, 512]", arg508_1: "f32[512, 2048]", arg509_1: "f32[512]", arg510_1: "f32[512, 512]", arg511_1: "f32[512, 512]", arg512_1: "f32[512, 512]", arg513_1: "f32[512, 512]", arg514_1: "f32[512]", arg515_1: "f32[512, 512]", arg516_1: "f32[512, 512]", arg517_1: "f32[512, 512]", arg518_1: "f32[512, 512]", arg519_1: "f32[512]", arg520_1: "f32[2048, 512]", arg521_1: "f32[512, 2048]", arg522_1: "f32[512]", arg523_1: "f32[512]", getitem_786: "f32[]", getitem_787: "f32[]", getitem_788: "f32[]", getitem_789: "f32[]", getitem_790: "f32[]", getitem_791: "f32[]", getitem_792: "f32[]", getitem_793: "f32[]", getitem_794: "f32[]", getitem_795: "f32[]", getitem_796: "f32[]", getitem_797: "f32[]", getitem_798: "f32[]", getitem_799: "f32[]", getitem_800: "f32[]", getitem_801: "f32[]", getitem_802: "f32[]", getitem_803: "f32[]", getitem_804: "f32[]", getitem_805: "f32[]", getitem_806: "f32[]", getitem_807: "f32[]", getitem_808: "f32[]", getitem_809: "f32[]", getitem_810: "f32[]", getitem_811: "f32[]", getitem_812: "f32[]", getitem_813: "f32[]", getitem_814: "f32[]", getitem_815: "f32[]", getitem_816: "f32[]", getitem_817: "f32[]", getitem_818: "f32[]", getitem_819: "f32[]", getitem_820: "f32[]", getitem_821: "f32[]", getitem_822: "f32[]", getitem_823: "f32[]", getitem_824: "f32[]", getitem_825: "f32[]", getitem_826: "f32[]", getitem_827: "f32[]", getitem_828: "f32[]", getitem_829: "f32[]", getitem_830: "f32[]", getitem_831: "f32[]", getitem_832: "f32[]", getitem_833: "f32[]", getitem_834: "f32[]", getitem_835: "f32[]", getitem_836: "f32[]", getitem_837: "f32[]", getitem_838: "f32[]", getitem_839: "f32[]", getitem_840: "f32[]", getitem_841: "f32[]", getitem_842: "f32[]", getitem_843: "f32[]", getitem_844: "f32[]", getitem_845: "f32[]", getitem_846: "f32[]", getitem_847: "f32[]", getitem_848: "f32[]", getitem_849: "f32[]", getitem_850: "f32[]", getitem_851: "f32[]", getitem_852: "f32[]", getitem_853: "f32[]", getitem_854: "f32[]", getitem_855: "f32[]", getitem_856: "f32[]", getitem_857: "f32[]", getitem_858: "f32[]", getitem_859: "f32[]", getitem_860: "f32[]", getitem_861: "f32[]", getitem_862: "f32[]", getitem_863: "f32[]", getitem_864: "f32[]", getitem_865: "f32[]", getitem_866: "f32[]", getitem_867: "f32[]", getitem_868: "f32[]", getitem_869: "f32[]", getitem_870: "f32[]", getitem_871: "f32[]", getitem_872: "f32[]", getitem_873: "f32[]", getitem_874: "f32[]", getitem_875: "f32[]", getitem_876: "f32[]", getitem_877: "f32[]", getitem_878: "f32[]", getitem_879: "f32[]", getitem_880: "f32[]", getitem_881: "f32[]", getitem_882: "f32[]", getitem_883: "f32[]", getitem_884: "f32[]", getitem_885: "f32[]", getitem_886: "f32[]", getitem_887: "f32[]", getitem_888: "f32[]", getitem_889: "f32[]", getitem_890: "f32[]", getitem_891: "f32[]", getitem_892: "f32[]", getitem_893: "f32[]", getitem_894: "f32[]", getitem_895: "f32[]", getitem_896: "f32[]", getitem_897: "f32[]", getitem_898: "f32[]", getitem_899: "f32[]", getitem_900: "f32[]", getitem_901: "f32[]", getitem_902: "f32[]", getitem_903: "f32[]", getitem_904: "f32[]", getitem_905: "f32[]", getitem_906: "f32[]", getitem_907: "f32[]", getitem_908: "f32[]", getitem_909: "f32[]", getitem_910: "f32[]", getitem_911: "f32[]", getitem_912: "f32[]", getitem_913: "f32[]", getitem_914: "f32[]", getitem_915: "f32[]", getitem_916: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg135_1, arg133_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1], 0.999);  arg135_1 = arg133_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = arg416_1 = arg417_1 = arg418_1 = arg419_1 = arg420_1 = arg421_1 = arg422_1 = arg423_1 = arg424_1 = arg425_1 = arg426_1 = arg427_1 = arg428_1 = arg429_1 = arg430_1 = arg431_1 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = arg506_1 = arg507_1 = arg508_1 = arg509_1 = arg510_1 = arg511_1 = arg512_1 = arg513_1 = arg514_1 = arg515_1 = arg516_1 = arg517_1 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = arg522_1 = arg523_1 = None
        getitem: "f32[32128, 512]" = _foreach_mul_scalar[0]
        getitem_1: "f32[512, 512]" = _foreach_mul_scalar[1]
        getitem_2: "f32[512, 512]" = _foreach_mul_scalar[2]
        getitem_3: "f32[512, 512]" = _foreach_mul_scalar[3]
        getitem_4: "f32[512, 512]" = _foreach_mul_scalar[4]
        getitem_5: "f32[32, 8]" = _foreach_mul_scalar[5]
        getitem_6: "f32[512]" = _foreach_mul_scalar[6]
        getitem_7: "f32[2048, 512]" = _foreach_mul_scalar[7]
        getitem_8: "f32[512, 2048]" = _foreach_mul_scalar[8]
        getitem_9: "f32[512]" = _foreach_mul_scalar[9]
        getitem_10: "f32[512, 512]" = _foreach_mul_scalar[10]
        getitem_11: "f32[512, 512]" = _foreach_mul_scalar[11]
        getitem_12: "f32[512, 512]" = _foreach_mul_scalar[12]
        getitem_13: "f32[512, 512]" = _foreach_mul_scalar[13]
        getitem_14: "f32[512]" = _foreach_mul_scalar[14]
        getitem_15: "f32[2048, 512]" = _foreach_mul_scalar[15]
        getitem_16: "f32[512, 2048]" = _foreach_mul_scalar[16]
        getitem_17: "f32[512]" = _foreach_mul_scalar[17]
        getitem_18: "f32[512, 512]" = _foreach_mul_scalar[18]
        getitem_19: "f32[512, 512]" = _foreach_mul_scalar[19]
        getitem_20: "f32[512, 512]" = _foreach_mul_scalar[20]
        getitem_21: "f32[512, 512]" = _foreach_mul_scalar[21]
        getitem_22: "f32[512]" = _foreach_mul_scalar[22]
        getitem_23: "f32[2048, 512]" = _foreach_mul_scalar[23]
        getitem_24: "f32[512, 2048]" = _foreach_mul_scalar[24]
        getitem_25: "f32[512]" = _foreach_mul_scalar[25]
        getitem_26: "f32[512, 512]" = _foreach_mul_scalar[26]
        getitem_27: "f32[512, 512]" = _foreach_mul_scalar[27]
        getitem_28: "f32[512, 512]" = _foreach_mul_scalar[28]
        getitem_29: "f32[512, 512]" = _foreach_mul_scalar[29]
        getitem_30: "f32[512]" = _foreach_mul_scalar[30]
        getitem_31: "f32[2048, 512]" = _foreach_mul_scalar[31]
        getitem_32: "f32[512, 2048]" = _foreach_mul_scalar[32]
        getitem_33: "f32[512]" = _foreach_mul_scalar[33]
        getitem_34: "f32[512, 512]" = _foreach_mul_scalar[34]
        getitem_35: "f32[512, 512]" = _foreach_mul_scalar[35]
        getitem_36: "f32[512, 512]" = _foreach_mul_scalar[36]
        getitem_37: "f32[512, 512]" = _foreach_mul_scalar[37]
        getitem_38: "f32[512]" = _foreach_mul_scalar[38]
        getitem_39: "f32[2048, 512]" = _foreach_mul_scalar[39]
        getitem_40: "f32[512, 2048]" = _foreach_mul_scalar[40]
        getitem_41: "f32[512]" = _foreach_mul_scalar[41]
        getitem_42: "f32[512, 512]" = _foreach_mul_scalar[42]
        getitem_43: "f32[512, 512]" = _foreach_mul_scalar[43]
        getitem_44: "f32[512, 512]" = _foreach_mul_scalar[44]
        getitem_45: "f32[512, 512]" = _foreach_mul_scalar[45]
        getitem_46: "f32[512]" = _foreach_mul_scalar[46]
        getitem_47: "f32[2048, 512]" = _foreach_mul_scalar[47]
        getitem_48: "f32[512, 2048]" = _foreach_mul_scalar[48]
        getitem_49: "f32[512]" = _foreach_mul_scalar[49]
        getitem_50: "f32[512]" = _foreach_mul_scalar[50]
        getitem_51: "f32[512, 512]" = _foreach_mul_scalar[51]
        getitem_52: "f32[512, 512]" = _foreach_mul_scalar[52]
        getitem_53: "f32[512, 512]" = _foreach_mul_scalar[53]
        getitem_54: "f32[512, 512]" = _foreach_mul_scalar[54]
        getitem_55: "f32[32, 8]" = _foreach_mul_scalar[55]
        getitem_56: "f32[512]" = _foreach_mul_scalar[56]
        getitem_57: "f32[512, 512]" = _foreach_mul_scalar[57]
        getitem_58: "f32[512, 512]" = _foreach_mul_scalar[58]
        getitem_59: "f32[512, 512]" = _foreach_mul_scalar[59]
        getitem_60: "f32[512, 512]" = _foreach_mul_scalar[60]
        getitem_61: "f32[512]" = _foreach_mul_scalar[61]
        getitem_62: "f32[2048, 512]" = _foreach_mul_scalar[62]
        getitem_63: "f32[512, 2048]" = _foreach_mul_scalar[63]
        getitem_64: "f32[512]" = _foreach_mul_scalar[64]
        getitem_65: "f32[512, 512]" = _foreach_mul_scalar[65]
        getitem_66: "f32[512, 512]" = _foreach_mul_scalar[66]
        getitem_67: "f32[512, 512]" = _foreach_mul_scalar[67]
        getitem_68: "f32[512, 512]" = _foreach_mul_scalar[68]
        getitem_69: "f32[512]" = _foreach_mul_scalar[69]
        getitem_70: "f32[512, 512]" = _foreach_mul_scalar[70]
        getitem_71: "f32[512, 512]" = _foreach_mul_scalar[71]
        getitem_72: "f32[512, 512]" = _foreach_mul_scalar[72]
        getitem_73: "f32[512, 512]" = _foreach_mul_scalar[73]
        getitem_74: "f32[512]" = _foreach_mul_scalar[74]
        getitem_75: "f32[2048, 512]" = _foreach_mul_scalar[75]
        getitem_76: "f32[512, 2048]" = _foreach_mul_scalar[76]
        getitem_77: "f32[512]" = _foreach_mul_scalar[77]
        getitem_78: "f32[512, 512]" = _foreach_mul_scalar[78]
        getitem_79: "f32[512, 512]" = _foreach_mul_scalar[79]
        getitem_80: "f32[512, 512]" = _foreach_mul_scalar[80]
        getitem_81: "f32[512, 512]" = _foreach_mul_scalar[81]
        getitem_82: "f32[512]" = _foreach_mul_scalar[82]
        getitem_83: "f32[512, 512]" = _foreach_mul_scalar[83]
        getitem_84: "f32[512, 512]" = _foreach_mul_scalar[84]
        getitem_85: "f32[512, 512]" = _foreach_mul_scalar[85]
        getitem_86: "f32[512, 512]" = _foreach_mul_scalar[86]
        getitem_87: "f32[512]" = _foreach_mul_scalar[87]
        getitem_88: "f32[2048, 512]" = _foreach_mul_scalar[88]
        getitem_89: "f32[512, 2048]" = _foreach_mul_scalar[89]
        getitem_90: "f32[512]" = _foreach_mul_scalar[90]
        getitem_91: "f32[512, 512]" = _foreach_mul_scalar[91]
        getitem_92: "f32[512, 512]" = _foreach_mul_scalar[92]
        getitem_93: "f32[512, 512]" = _foreach_mul_scalar[93]
        getitem_94: "f32[512, 512]" = _foreach_mul_scalar[94]
        getitem_95: "f32[512]" = _foreach_mul_scalar[95]
        getitem_96: "f32[512, 512]" = _foreach_mul_scalar[96]
        getitem_97: "f32[512, 512]" = _foreach_mul_scalar[97]
        getitem_98: "f32[512, 512]" = _foreach_mul_scalar[98]
        getitem_99: "f32[512, 512]" = _foreach_mul_scalar[99]
        getitem_100: "f32[512]" = _foreach_mul_scalar[100]
        getitem_101: "f32[2048, 512]" = _foreach_mul_scalar[101]
        getitem_102: "f32[512, 2048]" = _foreach_mul_scalar[102]
        getitem_103: "f32[512]" = _foreach_mul_scalar[103]
        getitem_104: "f32[512, 512]" = _foreach_mul_scalar[104]
        getitem_105: "f32[512, 512]" = _foreach_mul_scalar[105]
        getitem_106: "f32[512, 512]" = _foreach_mul_scalar[106]
        getitem_107: "f32[512, 512]" = _foreach_mul_scalar[107]
        getitem_108: "f32[512]" = _foreach_mul_scalar[108]
        getitem_109: "f32[512, 512]" = _foreach_mul_scalar[109]
        getitem_110: "f32[512, 512]" = _foreach_mul_scalar[110]
        getitem_111: "f32[512, 512]" = _foreach_mul_scalar[111]
        getitem_112: "f32[512, 512]" = _foreach_mul_scalar[112]
        getitem_113: "f32[512]" = _foreach_mul_scalar[113]
        getitem_114: "f32[2048, 512]" = _foreach_mul_scalar[114]
        getitem_115: "f32[512, 2048]" = _foreach_mul_scalar[115]
        getitem_116: "f32[512]" = _foreach_mul_scalar[116]
        getitem_117: "f32[512, 512]" = _foreach_mul_scalar[117]
        getitem_118: "f32[512, 512]" = _foreach_mul_scalar[118]
        getitem_119: "f32[512, 512]" = _foreach_mul_scalar[119]
        getitem_120: "f32[512, 512]" = _foreach_mul_scalar[120]
        getitem_121: "f32[512]" = _foreach_mul_scalar[121]
        getitem_122: "f32[512, 512]" = _foreach_mul_scalar[122]
        getitem_123: "f32[512, 512]" = _foreach_mul_scalar[123]
        getitem_124: "f32[512, 512]" = _foreach_mul_scalar[124]
        getitem_125: "f32[512, 512]" = _foreach_mul_scalar[125]
        getitem_126: "f32[512]" = _foreach_mul_scalar[126]
        getitem_127: "f32[2048, 512]" = _foreach_mul_scalar[127]
        getitem_128: "f32[512, 2048]" = _foreach_mul_scalar[128]
        getitem_129: "f32[512]" = _foreach_mul_scalar[129]
        getitem_130: "f32[512]" = _foreach_mul_scalar[130];  _foreach_mul_scalar = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916], 1);  getitem_786 = getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = getitem_794 = getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = getitem_802 = getitem_803 = getitem_804 = getitem_805 = getitem_806 = getitem_807 = getitem_808 = getitem_809 = getitem_810 = getitem_811 = getitem_812 = getitem_813 = getitem_814 = getitem_815 = getitem_816 = getitem_817 = getitem_818 = getitem_819 = getitem_820 = getitem_821 = getitem_822 = getitem_823 = getitem_824 = getitem_825 = getitem_826 = getitem_827 = getitem_828 = getitem_829 = getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = getitem_835 = getitem_836 = getitem_837 = getitem_838 = getitem_839 = getitem_840 = getitem_841 = getitem_842 = getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = getitem_849 = getitem_850 = getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = getitem_857 = getitem_858 = getitem_859 = getitem_860 = getitem_861 = getitem_862 = getitem_863 = getitem_864 = getitem_865 = getitem_866 = getitem_867 = getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = getitem_873 = getitem_874 = getitem_875 = getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = getitem_900 = getitem_901 = getitem_902 = getitem_903 = getitem_904 = getitem_905 = getitem_906 = getitem_907 = getitem_908 = getitem_909 = getitem_910 = getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = None
        getitem_917: "f32[]" = _foreach_sub_scalar[0]
        getitem_918: "f32[]" = _foreach_sub_scalar[1]
        getitem_919: "f32[]" = _foreach_sub_scalar[2]
        getitem_920: "f32[]" = _foreach_sub_scalar[3]
        getitem_921: "f32[]" = _foreach_sub_scalar[4]
        getitem_922: "f32[]" = _foreach_sub_scalar[5]
        getitem_923: "f32[]" = _foreach_sub_scalar[6]
        getitem_924: "f32[]" = _foreach_sub_scalar[7]
        getitem_925: "f32[]" = _foreach_sub_scalar[8]
        getitem_926: "f32[]" = _foreach_sub_scalar[9]
        getitem_927: "f32[]" = _foreach_sub_scalar[10]
        getitem_928: "f32[]" = _foreach_sub_scalar[11]
        getitem_929: "f32[]" = _foreach_sub_scalar[12]
        getitem_930: "f32[]" = _foreach_sub_scalar[13]
        getitem_931: "f32[]" = _foreach_sub_scalar[14]
        getitem_932: "f32[]" = _foreach_sub_scalar[15]
        getitem_933: "f32[]" = _foreach_sub_scalar[16]
        getitem_934: "f32[]" = _foreach_sub_scalar[17]
        getitem_935: "f32[]" = _foreach_sub_scalar[18]
        getitem_936: "f32[]" = _foreach_sub_scalar[19]
        getitem_937: "f32[]" = _foreach_sub_scalar[20]
        getitem_938: "f32[]" = _foreach_sub_scalar[21]
        getitem_939: "f32[]" = _foreach_sub_scalar[22]
        getitem_940: "f32[]" = _foreach_sub_scalar[23]
        getitem_941: "f32[]" = _foreach_sub_scalar[24]
        getitem_942: "f32[]" = _foreach_sub_scalar[25]
        getitem_943: "f32[]" = _foreach_sub_scalar[26]
        getitem_944: "f32[]" = _foreach_sub_scalar[27]
        getitem_945: "f32[]" = _foreach_sub_scalar[28]
        getitem_946: "f32[]" = _foreach_sub_scalar[29]
        getitem_947: "f32[]" = _foreach_sub_scalar[30]
        getitem_948: "f32[]" = _foreach_sub_scalar[31]
        getitem_949: "f32[]" = _foreach_sub_scalar[32]
        getitem_950: "f32[]" = _foreach_sub_scalar[33]
        getitem_951: "f32[]" = _foreach_sub_scalar[34]
        getitem_952: "f32[]" = _foreach_sub_scalar[35]
        getitem_953: "f32[]" = _foreach_sub_scalar[36]
        getitem_954: "f32[]" = _foreach_sub_scalar[37]
        getitem_955: "f32[]" = _foreach_sub_scalar[38]
        getitem_956: "f32[]" = _foreach_sub_scalar[39]
        getitem_957: "f32[]" = _foreach_sub_scalar[40]
        getitem_958: "f32[]" = _foreach_sub_scalar[41]
        getitem_959: "f32[]" = _foreach_sub_scalar[42]
        getitem_960: "f32[]" = _foreach_sub_scalar[43]
        getitem_961: "f32[]" = _foreach_sub_scalar[44]
        getitem_962: "f32[]" = _foreach_sub_scalar[45]
        getitem_963: "f32[]" = _foreach_sub_scalar[46]
        getitem_964: "f32[]" = _foreach_sub_scalar[47]
        getitem_965: "f32[]" = _foreach_sub_scalar[48]
        getitem_966: "f32[]" = _foreach_sub_scalar[49]
        getitem_967: "f32[]" = _foreach_sub_scalar[50]
        getitem_968: "f32[]" = _foreach_sub_scalar[51]
        getitem_969: "f32[]" = _foreach_sub_scalar[52]
        getitem_970: "f32[]" = _foreach_sub_scalar[53]
        getitem_971: "f32[]" = _foreach_sub_scalar[54]
        getitem_972: "f32[]" = _foreach_sub_scalar[55]
        getitem_973: "f32[]" = _foreach_sub_scalar[56]
        getitem_974: "f32[]" = _foreach_sub_scalar[57]
        getitem_975: "f32[]" = _foreach_sub_scalar[58]
        getitem_976: "f32[]" = _foreach_sub_scalar[59]
        getitem_977: "f32[]" = _foreach_sub_scalar[60]
        getitem_978: "f32[]" = _foreach_sub_scalar[61]
        getitem_979: "f32[]" = _foreach_sub_scalar[62]
        getitem_980: "f32[]" = _foreach_sub_scalar[63]
        getitem_981: "f32[]" = _foreach_sub_scalar[64]
        getitem_982: "f32[]" = _foreach_sub_scalar[65]
        getitem_983: "f32[]" = _foreach_sub_scalar[66]
        getitem_984: "f32[]" = _foreach_sub_scalar[67]
        getitem_985: "f32[]" = _foreach_sub_scalar[68]
        getitem_986: "f32[]" = _foreach_sub_scalar[69]
        getitem_987: "f32[]" = _foreach_sub_scalar[70]
        getitem_988: "f32[]" = _foreach_sub_scalar[71]
        getitem_989: "f32[]" = _foreach_sub_scalar[72]
        getitem_990: "f32[]" = _foreach_sub_scalar[73]
        getitem_991: "f32[]" = _foreach_sub_scalar[74]
        getitem_992: "f32[]" = _foreach_sub_scalar[75]
        getitem_993: "f32[]" = _foreach_sub_scalar[76]
        getitem_994: "f32[]" = _foreach_sub_scalar[77]
        getitem_995: "f32[]" = _foreach_sub_scalar[78]
        getitem_996: "f32[]" = _foreach_sub_scalar[79]
        getitem_997: "f32[]" = _foreach_sub_scalar[80]
        getitem_998: "f32[]" = _foreach_sub_scalar[81]
        getitem_999: "f32[]" = _foreach_sub_scalar[82]
        getitem_1000: "f32[]" = _foreach_sub_scalar[83]
        getitem_1001: "f32[]" = _foreach_sub_scalar[84]
        getitem_1002: "f32[]" = _foreach_sub_scalar[85]
        getitem_1003: "f32[]" = _foreach_sub_scalar[86]
        getitem_1004: "f32[]" = _foreach_sub_scalar[87]
        getitem_1005: "f32[]" = _foreach_sub_scalar[88]
        getitem_1006: "f32[]" = _foreach_sub_scalar[89]
        getitem_1007: "f32[]" = _foreach_sub_scalar[90]
        getitem_1008: "f32[]" = _foreach_sub_scalar[91]
        getitem_1009: "f32[]" = _foreach_sub_scalar[92]
        getitem_1010: "f32[]" = _foreach_sub_scalar[93]
        getitem_1011: "f32[]" = _foreach_sub_scalar[94]
        getitem_1012: "f32[]" = _foreach_sub_scalar[95]
        getitem_1013: "f32[]" = _foreach_sub_scalar[96]
        getitem_1014: "f32[]" = _foreach_sub_scalar[97]
        getitem_1015: "f32[]" = _foreach_sub_scalar[98]
        getitem_1016: "f32[]" = _foreach_sub_scalar[99]
        getitem_1017: "f32[]" = _foreach_sub_scalar[100]
        getitem_1018: "f32[]" = _foreach_sub_scalar[101]
        getitem_1019: "f32[]" = _foreach_sub_scalar[102]
        getitem_1020: "f32[]" = _foreach_sub_scalar[103]
        getitem_1021: "f32[]" = _foreach_sub_scalar[104]
        getitem_1022: "f32[]" = _foreach_sub_scalar[105]
        getitem_1023: "f32[]" = _foreach_sub_scalar[106]
        getitem_1024: "f32[]" = _foreach_sub_scalar[107]
        getitem_1025: "f32[]" = _foreach_sub_scalar[108]
        getitem_1026: "f32[]" = _foreach_sub_scalar[109]
        getitem_1027: "f32[]" = _foreach_sub_scalar[110]
        getitem_1028: "f32[]" = _foreach_sub_scalar[111]
        getitem_1029: "f32[]" = _foreach_sub_scalar[112]
        getitem_1030: "f32[]" = _foreach_sub_scalar[113]
        getitem_1031: "f32[]" = _foreach_sub_scalar[114]
        getitem_1032: "f32[]" = _foreach_sub_scalar[115]
        getitem_1033: "f32[]" = _foreach_sub_scalar[116]
        getitem_1034: "f32[]" = _foreach_sub_scalar[117]
        getitem_1035: "f32[]" = _foreach_sub_scalar[118]
        getitem_1036: "f32[]" = _foreach_sub_scalar[119]
        getitem_1037: "f32[]" = _foreach_sub_scalar[120]
        getitem_1038: "f32[]" = _foreach_sub_scalar[121]
        getitem_1039: "f32[]" = _foreach_sub_scalar[122]
        getitem_1040: "f32[]" = _foreach_sub_scalar[123]
        getitem_1041: "f32[]" = _foreach_sub_scalar[124]
        getitem_1042: "f32[]" = _foreach_sub_scalar[125]
        getitem_1043: "f32[]" = _foreach_sub_scalar[126]
        getitem_1044: "f32[]" = _foreach_sub_scalar[127]
        getitem_1045: "f32[]" = _foreach_sub_scalar[128]
        getitem_1046: "f32[]" = _foreach_sub_scalar[129]
        getitem_1047: "f32[]" = _foreach_sub_scalar[130];  _foreach_sub_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047)


def _default_make_inputs():
    return [
    torch.randn([32128, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([32, 8], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([2048, 512], dtype=torch.float32, device='cuda'),
    torch.randn([512, 2048], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([512], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

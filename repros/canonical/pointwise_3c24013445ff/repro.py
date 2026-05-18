"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: 3c24013445ff
Shape hash: 2ab5447f
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
    def forward(self, getitem_618: "f32[30522, 768]", getitem_619: "f32[512, 768]", getitem_620: "f32[1024, 768]", getitem_621: "f32[1024, 768]", getitem_622: "f32[1024, 768]", getitem_623: "f32[1024, 768]", getitem_624: "f32[2, 768]", getitem_625: "f32[768]", getitem_626: "f32[768]", getitem_627: "f32[768, 768]", getitem_628: "f32[768]", getitem_629: "f32[768, 768]", getitem_630: "f32[768]", getitem_631: "f32[768, 768]", getitem_632: "f32[768]", getitem_633: "f32[768, 768]", getitem_634: "f32[768]", getitem_635: "f32[768]", getitem_636: "f32[768]", getitem_637: "f32[3072, 768]", getitem_638: "f32[3072]", getitem_639: "f32[768, 3072]", getitem_640: "f32[768]", getitem_641: "f32[768]", getitem_642: "f32[768]", getitem_643: "f32[768, 768]", getitem_644: "f32[768]", getitem_645: "f32[768, 768]", getitem_646: "f32[768]", getitem_647: "f32[768, 768]", getitem_648: "f32[768]", getitem_649: "f32[768, 768]", getitem_650: "f32[768]", getitem_651: "f32[768]", getitem_652: "f32[768]", getitem_653: "f32[3072, 768]", getitem_654: "f32[3072]", getitem_655: "f32[768, 3072]", getitem_656: "f32[768]", getitem_657: "f32[768]", getitem_658: "f32[768]", getitem_659: "f32[768, 768]", getitem_660: "f32[768]", getitem_661: "f32[768, 768]", getitem_662: "f32[768]", getitem_663: "f32[768, 768]", getitem_664: "f32[768]", getitem_665: "f32[768, 768]", getitem_666: "f32[768]", getitem_667: "f32[768]", getitem_668: "f32[768]", getitem_669: "f32[3072, 768]", getitem_670: "f32[3072]", getitem_671: "f32[768, 3072]", getitem_672: "f32[768]", getitem_673: "f32[768]", getitem_674: "f32[768]", getitem_675: "f32[768, 768]", getitem_676: "f32[768]", getitem_677: "f32[768, 768]", getitem_678: "f32[768]", getitem_679: "f32[768, 768]", getitem_680: "f32[768]", getitem_681: "f32[768, 768]", getitem_682: "f32[768]", getitem_683: "f32[768]", getitem_684: "f32[768]", getitem_685: "f32[3072, 768]", getitem_686: "f32[3072]", getitem_687: "f32[768, 3072]", getitem_688: "f32[768]", getitem_689: "f32[768]", getitem_690: "f32[768]", getitem_691: "f32[768, 768]", getitem_692: "f32[768]", getitem_693: "f32[768, 768]", getitem_694: "f32[768]", getitem_695: "f32[768, 768]", getitem_696: "f32[768]", getitem_697: "f32[768, 768]", getitem_698: "f32[768]", getitem_699: "f32[768]", getitem_700: "f32[768]", getitem_701: "f32[3072, 768]", getitem_702: "f32[3072]", getitem_703: "f32[768, 3072]", getitem_704: "f32[768]", getitem_705: "f32[768]", getitem_706: "f32[768]", getitem_707: "f32[768, 768]", getitem_708: "f32[768]", getitem_709: "f32[768, 768]", getitem_710: "f32[768]", getitem_711: "f32[768, 768]", getitem_712: "f32[768]", getitem_713: "f32[768, 768]", getitem_714: "f32[768]", getitem_715: "f32[768]", getitem_716: "f32[768]", getitem_717: "f32[3072, 768]", getitem_718: "f32[3072]", getitem_719: "f32[768, 3072]", getitem_720: "f32[768]", getitem_721: "f32[768]", getitem_722: "f32[768]", getitem_723: "f32[768, 768]", getitem_724: "f32[768]", getitem_725: "f32[768, 768]", getitem_726: "f32[768]", getitem_727: "f32[768, 768]", getitem_728: "f32[768]", getitem_729: "f32[768, 768]", getitem_730: "f32[768]", getitem_731: "f32[768]", getitem_732: "f32[768]", getitem_733: "f32[3072, 768]", getitem_734: "f32[3072]", getitem_735: "f32[768, 3072]", getitem_736: "f32[768]", getitem_737: "f32[768]", getitem_738: "f32[768]", getitem_739: "f32[768, 768]", getitem_740: "f32[768]", getitem_741: "f32[768, 768]", getitem_742: "f32[768]", getitem_743: "f32[768, 768]", getitem_744: "f32[768]", getitem_745: "f32[768, 768]", getitem_746: "f32[768]", getitem_747: "f32[768]", getitem_748: "f32[768]", getitem_749: "f32[3072, 768]", getitem_750: "f32[3072]", getitem_751: "f32[768, 3072]", getitem_752: "f32[768]", getitem_753: "f32[768]", getitem_754: "f32[768]", getitem_755: "f32[768, 768]", getitem_756: "f32[768]", getitem_757: "f32[768, 768]", getitem_758: "f32[768]", getitem_759: "f32[768, 768]", getitem_760: "f32[768]", getitem_761: "f32[768, 768]", getitem_762: "f32[768]", getitem_763: "f32[768]", getitem_764: "f32[768]", getitem_765: "f32[3072, 768]", getitem_766: "f32[3072]", getitem_767: "f32[768, 3072]", getitem_768: "f32[768]", getitem_769: "f32[768]", getitem_770: "f32[768]", getitem_771: "f32[768, 768]", getitem_772: "f32[768]", getitem_773: "f32[768, 768]", getitem_774: "f32[768]", getitem_775: "f32[768, 768]", getitem_776: "f32[768]", getitem_777: "f32[768, 768]", getitem_778: "f32[768]", getitem_779: "f32[768]", getitem_780: "f32[768]", getitem_781: "f32[3072, 768]", getitem_782: "f32[3072]", getitem_783: "f32[768, 3072]", getitem_784: "f32[768]", getitem_785: "f32[768]", getitem_786: "f32[768]", getitem_787: "f32[768, 768]", getitem_788: "f32[768]", getitem_789: "f32[768, 768]", getitem_790: "f32[768]", getitem_791: "f32[768, 768]", getitem_792: "f32[768]", getitem_793: "f32[768, 768]", getitem_794: "f32[768]", getitem_795: "f32[768]", getitem_796: "f32[768]", getitem_797: "f32[3072, 768]", getitem_798: "f32[3072]", getitem_799: "f32[768, 3072]", getitem_800: "f32[768]", getitem_801: "f32[768]", getitem_802: "f32[768]", getitem_803: "f32[768, 768]", getitem_804: "f32[768]", getitem_805: "f32[768, 768]", getitem_806: "f32[768]", getitem_807: "f32[768, 768]", getitem_808: "f32[768]", getitem_809: "f32[768, 768]", getitem_810: "f32[768]", getitem_811: "f32[768]", getitem_812: "f32[768]", getitem_813: "f32[3072, 768]", getitem_814: "f32[3072]", getitem_815: "f32[768, 3072]", getitem_816: "f32[768]", getitem_817: "f32[768]", getitem_818: "f32[768]", getitem_819: "f32[30522]", getitem_820: "f32[768, 768]", getitem_821: "f32[768]", getitem_822: "f32[768]", getitem_823: "f32[768]", arg824_1: "f32[30522, 768]", arg825_1: "f32[512, 768]", arg826_1: "f32[1024, 768]", arg827_1: "f32[1024, 768]", arg828_1: "f32[1024, 768]", arg829_1: "f32[1024, 768]", arg830_1: "f32[2, 768]", arg831_1: "f32[768]", arg832_1: "f32[768]", arg833_1: "f32[768, 768]", arg834_1: "f32[768]", arg835_1: "f32[768, 768]", arg836_1: "f32[768]", arg837_1: "f32[768, 768]", arg838_1: "f32[768]", arg839_1: "f32[768, 768]", arg840_1: "f32[768]", arg841_1: "f32[768]", arg842_1: "f32[768]", arg843_1: "f32[3072, 768]", arg844_1: "f32[3072]", arg845_1: "f32[768, 3072]", arg846_1: "f32[768]", arg847_1: "f32[768]", arg848_1: "f32[768]", arg849_1: "f32[768, 768]", arg850_1: "f32[768]", arg851_1: "f32[768, 768]", arg852_1: "f32[768]", arg853_1: "f32[768, 768]", arg854_1: "f32[768]", arg855_1: "f32[768, 768]", arg856_1: "f32[768]", arg857_1: "f32[768]", arg858_1: "f32[768]", arg859_1: "f32[3072, 768]", arg860_1: "f32[3072]", arg861_1: "f32[768, 3072]", arg862_1: "f32[768]", arg863_1: "f32[768]", arg864_1: "f32[768]", arg865_1: "f32[768, 768]", arg866_1: "f32[768]", arg867_1: "f32[768, 768]", arg868_1: "f32[768]", arg869_1: "f32[768, 768]", arg870_1: "f32[768]", arg871_1: "f32[768, 768]", arg872_1: "f32[768]", arg873_1: "f32[768]", arg874_1: "f32[768]", arg875_1: "f32[3072, 768]", arg876_1: "f32[3072]", arg877_1: "f32[768, 3072]", arg878_1: "f32[768]", arg879_1: "f32[768]", arg880_1: "f32[768]", arg881_1: "f32[768, 768]", arg882_1: "f32[768]", arg883_1: "f32[768, 768]", arg884_1: "f32[768]", arg885_1: "f32[768, 768]", arg886_1: "f32[768]", arg887_1: "f32[768, 768]", arg888_1: "f32[768]", arg889_1: "f32[768]", arg890_1: "f32[768]", arg891_1: "f32[3072, 768]", arg892_1: "f32[3072]", arg893_1: "f32[768, 3072]", arg894_1: "f32[768]", arg895_1: "f32[768]", arg896_1: "f32[768]", arg897_1: "f32[768, 768]", arg898_1: "f32[768]", arg899_1: "f32[768, 768]", arg900_1: "f32[768]", arg901_1: "f32[768, 768]", arg902_1: "f32[768]", arg903_1: "f32[768, 768]", arg904_1: "f32[768]", arg905_1: "f32[768]", arg906_1: "f32[768]", arg907_1: "f32[3072, 768]", arg908_1: "f32[3072]", arg909_1: "f32[768, 3072]", arg910_1: "f32[768]", arg911_1: "f32[768]", arg912_1: "f32[768]", arg913_1: "f32[768, 768]", arg914_1: "f32[768]", arg915_1: "f32[768, 768]", arg916_1: "f32[768]", arg917_1: "f32[768, 768]", arg918_1: "f32[768]", arg919_1: "f32[768, 768]", arg920_1: "f32[768]", arg921_1: "f32[768]", arg922_1: "f32[768]", arg923_1: "f32[3072, 768]", arg924_1: "f32[3072]", arg925_1: "f32[768, 3072]", arg926_1: "f32[768]", arg927_1: "f32[768]", arg928_1: "f32[768]", arg929_1: "f32[768, 768]", arg930_1: "f32[768]", arg931_1: "f32[768, 768]", arg932_1: "f32[768]", arg933_1: "f32[768, 768]", arg934_1: "f32[768]", arg935_1: "f32[768, 768]", arg936_1: "f32[768]", arg937_1: "f32[768]", arg938_1: "f32[768]", arg939_1: "f32[3072, 768]", arg940_1: "f32[3072]", arg941_1: "f32[768, 3072]", arg942_1: "f32[768]", arg943_1: "f32[768]", arg944_1: "f32[768]", arg945_1: "f32[768, 768]", arg946_1: "f32[768]", arg947_1: "f32[768, 768]", arg948_1: "f32[768]", arg949_1: "f32[768, 768]", arg950_1: "f32[768]", arg951_1: "f32[768, 768]", arg952_1: "f32[768]", arg953_1: "f32[768]", arg954_1: "f32[768]", arg955_1: "f32[3072, 768]", arg956_1: "f32[3072]", arg957_1: "f32[768, 3072]", arg958_1: "f32[768]", arg959_1: "f32[768]", arg960_1: "f32[768]", arg961_1: "f32[768, 768]", arg962_1: "f32[768]", arg963_1: "f32[768, 768]", arg964_1: "f32[768]", arg965_1: "f32[768, 768]", arg966_1: "f32[768]", arg967_1: "f32[768, 768]", arg968_1: "f32[768]", arg969_1: "f32[768]", arg970_1: "f32[768]", arg971_1: "f32[3072, 768]", arg972_1: "f32[3072]", arg973_1: "f32[768, 3072]", arg974_1: "f32[768]", arg975_1: "f32[768]", arg976_1: "f32[768]", arg977_1: "f32[768, 768]", arg978_1: "f32[768]", arg979_1: "f32[768, 768]", arg980_1: "f32[768]", arg981_1: "f32[768, 768]", arg982_1: "f32[768]", arg983_1: "f32[768, 768]", arg984_1: "f32[768]", arg985_1: "f32[768]", arg986_1: "f32[768]", arg987_1: "f32[3072, 768]", arg988_1: "f32[3072]", arg989_1: "f32[768, 3072]", arg990_1: "f32[768]", arg991_1: "f32[768]", arg992_1: "f32[768]", arg993_1: "f32[768, 768]", arg994_1: "f32[768]", arg995_1: "f32[768, 768]", arg996_1: "f32[768]", arg997_1: "f32[768, 768]", arg998_1: "f32[768]", arg999_1: "f32[768, 768]", arg1000_1: "f32[768]", arg1001_1: "f32[768]", arg1002_1: "f32[768]", arg1003_1: "f32[3072, 768]", arg1004_1: "f32[3072]", arg1005_1: "f32[768, 3072]", arg1006_1: "f32[768]", arg1007_1: "f32[768]", arg1008_1: "f32[768]", arg1009_1: "f32[768, 768]", arg1010_1: "f32[768]", arg1011_1: "f32[768, 768]", arg1012_1: "f32[768]", arg1013_1: "f32[768, 768]", arg1014_1: "f32[768]", arg1015_1: "f32[768, 768]", arg1016_1: "f32[768]", arg1017_1: "f32[768]", arg1018_1: "f32[768]", arg1019_1: "f32[3072, 768]", arg1020_1: "f32[3072]", arg1021_1: "f32[768, 3072]", arg1022_1: "f32[768]", arg1023_1: "f32[768]", arg1024_1: "f32[768]", arg1025_1: "f32[30522]", arg1026_1: "f32[768, 768]", arg1027_1: "f32[768]", arg1028_1: "f32[768]", arg1029_1: "f32[768]", getitem: "f32[]", getitem_1: "f32[]", getitem_2: "f32[]", getitem_3: "f32[]", getitem_4: "f32[]", getitem_5: "f32[]", getitem_6: "f32[]", getitem_7: "f32[]", getitem_8: "f32[]", getitem_9: "f32[]", getitem_10: "f32[]", getitem_11: "f32[]", getitem_12: "f32[]", getitem_13: "f32[]", getitem_14: "f32[]", getitem_15: "f32[]", getitem_16: "f32[]", getitem_17: "f32[]", getitem_18: "f32[]", getitem_19: "f32[]", getitem_20: "f32[]", getitem_21: "f32[]", getitem_22: "f32[]", getitem_23: "f32[]", getitem_24: "f32[]", getitem_25: "f32[]", getitem_26: "f32[]", getitem_27: "f32[]", getitem_28: "f32[]", getitem_29: "f32[]", getitem_30: "f32[]", getitem_31: "f32[]", getitem_32: "f32[]", getitem_33: "f32[]", getitem_34: "f32[]", getitem_35: "f32[]", getitem_36: "f32[]", getitem_37: "f32[]", getitem_38: "f32[]", getitem_39: "f32[]", getitem_40: "f32[]", getitem_41: "f32[]", getitem_42: "f32[]", getitem_43: "f32[]", getitem_44: "f32[]", getitem_45: "f32[]", getitem_46: "f32[]", getitem_47: "f32[]", getitem_48: "f32[]", getitem_49: "f32[]", getitem_50: "f32[]", getitem_51: "f32[]", getitem_52: "f32[]", getitem_53: "f32[]", getitem_54: "f32[]", getitem_55: "f32[]", getitem_56: "f32[]", getitem_57: "f32[]", getitem_58: "f32[]", getitem_59: "f32[]", getitem_60: "f32[]", getitem_61: "f32[]", getitem_62: "f32[]", getitem_63: "f32[]", getitem_64: "f32[]", getitem_65: "f32[]", getitem_66: "f32[]", getitem_67: "f32[]", getitem_68: "f32[]", getitem_69: "f32[]", getitem_70: "f32[]", getitem_71: "f32[]", getitem_72: "f32[]", getitem_73: "f32[]", getitem_74: "f32[]", getitem_75: "f32[]", getitem_76: "f32[]", getitem_77: "f32[]", getitem_78: "f32[]", getitem_79: "f32[]", getitem_80: "f32[]", getitem_81: "f32[]", getitem_82: "f32[]", getitem_83: "f32[]", getitem_84: "f32[]", getitem_85: "f32[]", getitem_86: "f32[]", getitem_87: "f32[]", getitem_88: "f32[]", getitem_89: "f32[]", getitem_90: "f32[]", getitem_91: "f32[]", getitem_92: "f32[]", getitem_93: "f32[]", getitem_94: "f32[]", getitem_95: "f32[]", getitem_96: "f32[]", getitem_97: "f32[]", getitem_98: "f32[]", getitem_99: "f32[]", getitem_100: "f32[]", getitem_101: "f32[]", getitem_102: "f32[]", getitem_103: "f32[]", getitem_104: "f32[]", getitem_105: "f32[]", getitem_106: "f32[]", getitem_107: "f32[]", getitem_108: "f32[]", getitem_109: "f32[]", getitem_110: "f32[]", getitem_111: "f32[]", getitem_112: "f32[]", getitem_113: "f32[]", getitem_114: "f32[]", getitem_115: "f32[]", getitem_116: "f32[]", getitem_117: "f32[]", getitem_118: "f32[]", getitem_119: "f32[]", getitem_120: "f32[]", getitem_121: "f32[]", getitem_122: "f32[]", getitem_123: "f32[]", getitem_124: "f32[]", getitem_125: "f32[]", getitem_126: "f32[]", getitem_127: "f32[]", getitem_128: "f32[]", getitem_129: "f32[]", getitem_130: "f32[]", getitem_131: "f32[]", getitem_132: "f32[]", getitem_133: "f32[]", getitem_134: "f32[]", getitem_135: "f32[]", getitem_136: "f32[]", getitem_137: "f32[]", getitem_138: "f32[]", getitem_139: "f32[]", getitem_140: "f32[]", getitem_141: "f32[]", getitem_142: "f32[]", getitem_143: "f32[]", getitem_144: "f32[]", getitem_145: "f32[]", getitem_146: "f32[]", getitem_147: "f32[]", getitem_148: "f32[]", getitem_149: "f32[]", getitem_150: "f32[]", getitem_151: "f32[]", getitem_152: "f32[]", getitem_153: "f32[]", getitem_154: "f32[]", getitem_155: "f32[]", getitem_156: "f32[]", getitem_157: "f32[]", getitem_158: "f32[]", getitem_159: "f32[]", getitem_160: "f32[]", getitem_161: "f32[]", getitem_162: "f32[]", getitem_163: "f32[]", getitem_164: "f32[]", getitem_165: "f32[]", getitem_166: "f32[]", getitem_167: "f32[]", getitem_168: "f32[]", getitem_169: "f32[]", getitem_170: "f32[]", getitem_171: "f32[]", getitem_172: "f32[]", getitem_173: "f32[]", getitem_174: "f32[]", getitem_175: "f32[]", getitem_176: "f32[]", getitem_177: "f32[]", getitem_178: "f32[]", getitem_179: "f32[]", getitem_180: "f32[]", getitem_181: "f32[]", getitem_182: "f32[]", getitem_183: "f32[]", getitem_184: "f32[]", getitem_185: "f32[]", getitem_186: "f32[]", getitem_187: "f32[]", getitem_188: "f32[]", getitem_189: "f32[]", getitem_190: "f32[]", getitem_191: "f32[]", getitem_192: "f32[]", getitem_193: "f32[]", getitem_194: "f32[]", getitem_195: "f32[]", getitem_196: "f32[]", getitem_197: "f32[]", getitem_198: "f32[]", getitem_199: "f32[]", getitem_200: "f32[]", getitem_201: "f32[]", getitem_202: "f32[]", getitem_203: "f32[]", getitem_204: "f32[]", getitem_205: "f32[]", getitem_1648: "f32[]", getitem_1649: "f32[]", getitem_1650: "f32[]", getitem_1651: "f32[]", getitem_1652: "f32[]", getitem_1653: "f32[]", getitem_1654: "f32[]", getitem_1655: "f32[]", getitem_1656: "f32[]", getitem_1657: "f32[]", getitem_1658: "f32[]", getitem_1659: "f32[]", getitem_1660: "f32[]", getitem_1661: "f32[]", getitem_1662: "f32[]", getitem_1663: "f32[]", getitem_1664: "f32[]", getitem_1665: "f32[]", getitem_1666: "f32[]", getitem_1667: "f32[]", getitem_1668: "f32[]", getitem_1669: "f32[]", getitem_1670: "f32[]", getitem_1671: "f32[]", getitem_1672: "f32[]", getitem_1673: "f32[]", getitem_1674: "f32[]", getitem_1675: "f32[]", getitem_1676: "f32[]", getitem_1677: "f32[]", getitem_1678: "f32[]", getitem_1679: "f32[]", getitem_1680: "f32[]", getitem_1681: "f32[]", getitem_1682: "f32[]", getitem_1683: "f32[]", getitem_1684: "f32[]", getitem_1685: "f32[]", getitem_1686: "f32[]", getitem_1687: "f32[]", getitem_1688: "f32[]", getitem_1689: "f32[]", getitem_1690: "f32[]", getitem_1691: "f32[]", getitem_1692: "f32[]", getitem_1693: "f32[]", getitem_1694: "f32[]", getitem_1695: "f32[]", getitem_1696: "f32[]", getitem_1697: "f32[]", getitem_1698: "f32[]", getitem_1699: "f32[]", getitem_1700: "f32[]", getitem_1701: "f32[]", getitem_1702: "f32[]", getitem_1703: "f32[]", getitem_1704: "f32[]", getitem_1705: "f32[]", getitem_1706: "f32[]", getitem_1707: "f32[]", getitem_1708: "f32[]", getitem_1709: "f32[]", getitem_1710: "f32[]", getitem_1711: "f32[]", getitem_1712: "f32[]", getitem_1713: "f32[]", getitem_1714: "f32[]", getitem_1715: "f32[]", getitem_1716: "f32[]", getitem_1717: "f32[]", getitem_1718: "f32[]", getitem_1719: "f32[]", getitem_1720: "f32[]", getitem_1721: "f32[]", getitem_1722: "f32[]", getitem_1723: "f32[]", getitem_1724: "f32[]", getitem_1725: "f32[]", getitem_1726: "f32[]", getitem_1727: "f32[]", getitem_1728: "f32[]", getitem_1729: "f32[]", getitem_1730: "f32[]", getitem_1731: "f32[]", getitem_1732: "f32[]", getitem_1733: "f32[]", getitem_1734: "f32[]", getitem_1735: "f32[]", getitem_1736: "f32[]", getitem_1737: "f32[]", getitem_1738: "f32[]", getitem_1739: "f32[]", getitem_1740: "f32[]", getitem_1741: "f32[]", getitem_1742: "f32[]", getitem_1743: "f32[]", getitem_1744: "f32[]", getitem_1745: "f32[]", getitem_1746: "f32[]", getitem_1747: "f32[]", getitem_1748: "f32[]", getitem_1749: "f32[]", getitem_1750: "f32[]", getitem_1751: "f32[]", getitem_1752: "f32[]", getitem_1753: "f32[]", getitem_1754: "f32[]", getitem_1755: "f32[]", getitem_1756: "f32[]", getitem_1757: "f32[]", getitem_1758: "f32[]", getitem_1759: "f32[]", getitem_1760: "f32[]", getitem_1761: "f32[]", getitem_1762: "f32[]", getitem_1763: "f32[]", getitem_1764: "f32[]", getitem_1765: "f32[]", getitem_1766: "f32[]", getitem_1767: "f32[]", getitem_1768: "f32[]", getitem_1769: "f32[]", getitem_1770: "f32[]", getitem_1771: "f32[]", getitem_1772: "f32[]", getitem_1773: "f32[]", getitem_1774: "f32[]", getitem_1775: "f32[]", getitem_1776: "f32[]", getitem_1777: "f32[]", getitem_1778: "f32[]", getitem_1779: "f32[]", getitem_1780: "f32[]", getitem_1781: "f32[]", getitem_1782: "f32[]", getitem_1783: "f32[]", getitem_1784: "f32[]", getitem_1785: "f32[]", getitem_1786: "f32[]", getitem_1787: "f32[]", getitem_1788: "f32[]", getitem_1789: "f32[]", getitem_1790: "f32[]", getitem_1791: "f32[]", getitem_1792: "f32[]", getitem_1793: "f32[]", getitem_1794: "f32[]", getitem_1795: "f32[]", getitem_1796: "f32[]", getitem_1797: "f32[]", getitem_1798: "f32[]", getitem_1799: "f32[]", getitem_1800: "f32[]", getitem_1801: "f32[]", getitem_1802: "f32[]", getitem_1803: "f32[]", getitem_1804: "f32[]", getitem_1805: "f32[]", getitem_1806: "f32[]", getitem_1807: "f32[]", getitem_1808: "f32[]", getitem_1809: "f32[]", getitem_1810: "f32[]", getitem_1811: "f32[]", getitem_1812: "f32[]", getitem_1813: "f32[]", getitem_1814: "f32[]", getitem_1815: "f32[]", getitem_1816: "f32[]", getitem_1817: "f32[]", getitem_1818: "f32[]", getitem_1819: "f32[]", getitem_1820: "f32[]", getitem_1821: "f32[]", getitem_1822: "f32[]", getitem_1823: "f32[]", getitem_1824: "f32[]", getitem_1825: "f32[]", getitem_1826: "f32[]", getitem_1827: "f32[]", getitem_1828: "f32[]", getitem_1829: "f32[]", getitem_1830: "f32[]", getitem_1831: "f32[]", getitem_1832: "f32[]", getitem_1833: "f32[]", getitem_1834: "f32[]", getitem_1835: "f32[]", getitem_1836: "f32[]", getitem_1837: "f32[]", getitem_1838: "f32[]", getitem_1839: "f32[]", getitem_1840: "f32[]", getitem_1841: "f32[]", getitem_1842: "f32[]", getitem_1843: "f32[]", getitem_1844: "f32[]", getitem_1845: "f32[]", getitem_1846: "f32[]", getitem_1847: "f32[]", getitem_1848: "f32[]", getitem_1849: "f32[]", getitem_1850: "f32[]", getitem_1851: "f32[]", getitem_1852: "f32[]", getitem_1853: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647, getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755, getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823], [arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1], [arg824_1, arg825_1, arg826_1, arg827_1, arg828_1, arg829_1, arg830_1, arg831_1, arg832_1, arg833_1, arg834_1, arg835_1, arg836_1, arg837_1, arg838_1, arg839_1, arg840_1, arg841_1, arg842_1, arg843_1, arg844_1, arg845_1, arg846_1, arg847_1, arg848_1, arg849_1, arg850_1, arg851_1, arg852_1, arg853_1, arg854_1, arg855_1, arg856_1, arg857_1, arg858_1, arg859_1, arg860_1, arg861_1, arg862_1, arg863_1, arg864_1, arg865_1, arg866_1, arg867_1, arg868_1, arg869_1, arg870_1, arg871_1, arg872_1, arg873_1, arg874_1, arg875_1, arg876_1, arg877_1, arg878_1, arg879_1, arg880_1, arg881_1, arg882_1, arg883_1, arg884_1, arg885_1, arg886_1, arg887_1, arg888_1, arg889_1, arg890_1, arg891_1, arg892_1, arg893_1, arg894_1, arg895_1, arg896_1, arg897_1, arg898_1, arg899_1, arg900_1, arg901_1, arg902_1, arg903_1, arg904_1, arg905_1, arg906_1, arg907_1, arg908_1, arg909_1, arg910_1, arg911_1, arg912_1, arg913_1, arg914_1, arg915_1, arg916_1, arg917_1, arg918_1, arg919_1, arg920_1, arg921_1, arg922_1, arg923_1, arg924_1, arg925_1, arg926_1, arg927_1, arg928_1, arg929_1, arg930_1, arg931_1, arg932_1, arg933_1, arg934_1, arg935_1, arg936_1, arg937_1, arg938_1, arg939_1, arg940_1, arg941_1, arg942_1, arg943_1, arg944_1, arg945_1, arg946_1, arg947_1, arg948_1, arg949_1, arg950_1, arg951_1, arg952_1, arg953_1, arg954_1, arg955_1, arg956_1, arg957_1, arg958_1, arg959_1, arg960_1, arg961_1, arg962_1, arg963_1, arg964_1, arg965_1, arg966_1, arg967_1, arg968_1, arg969_1, arg970_1, arg971_1, arg972_1, arg973_1, arg974_1, arg975_1, arg976_1, arg977_1, arg978_1, arg979_1, arg980_1, arg981_1, arg982_1, arg983_1, arg984_1, arg985_1, arg986_1, arg987_1, arg988_1, arg989_1, arg990_1, arg991_1, arg992_1, arg993_1, arg994_1, arg995_1, arg996_1, arg997_1, arg998_1, arg999_1, arg1000_1, arg1001_1, arg1002_1, arg1003_1, arg1004_1, arg1005_1, arg1006_1, arg1007_1, arg1008_1, arg1009_1, arg1010_1, arg1011_1, arg1012_1, arg1013_1, arg1014_1, arg1015_1, arg1016_1, arg1017_1, arg1018_1, arg1019_1, arg1020_1, arg1021_1, arg1022_1, arg1023_1, arg1024_1, arg1025_1, arg1026_1, arg1027_1, arg1028_1, arg1029_1], 0.0010000000000000009);  getitem_618 = getitem_619 = getitem_620 = getitem_621 = getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = getitem_646 = getitem_647 = getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = getitem_654 = getitem_655 = getitem_656 = getitem_657 = getitem_658 = getitem_659 = getitem_660 = getitem_661 = getitem_662 = getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = getitem_671 = getitem_672 = getitem_673 = getitem_674 = getitem_675 = getitem_676 = getitem_677 = getitem_678 = getitem_679 = getitem_680 = getitem_681 = getitem_682 = getitem_683 = getitem_684 = getitem_685 = getitem_686 = getitem_687 = getitem_688 = getitem_689 = getitem_690 = getitem_691 = getitem_692 = getitem_693 = getitem_694 = getitem_695 = getitem_696 = getitem_697 = getitem_698 = getitem_699 = getitem_700 = getitem_701 = getitem_702 = getitem_703 = getitem_704 = getitem_705 = getitem_706 = getitem_707 = getitem_708 = getitem_709 = getitem_710 = getitem_711 = getitem_712 = getitem_713 = getitem_714 = getitem_715 = getitem_716 = getitem_717 = getitem_718 = getitem_719 = getitem_720 = getitem_721 = getitem_722 = getitem_723 = getitem_724 = getitem_725 = getitem_726 = getitem_727 = getitem_728 = getitem_729 = getitem_730 = getitem_731 = getitem_732 = getitem_733 = getitem_734 = getitem_735 = getitem_736 = getitem_737 = getitem_738 = getitem_739 = getitem_740 = getitem_741 = getitem_742 = getitem_743 = getitem_744 = getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = getitem_752 = getitem_753 = getitem_754 = getitem_755 = getitem_756 = getitem_757 = getitem_758 = getitem_759 = getitem_760 = getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = getitem_768 = getitem_769 = getitem_770 = getitem_771 = getitem_772 = getitem_773 = getitem_774 = getitem_775 = getitem_776 = getitem_777 = getitem_778 = getitem_779 = getitem_780 = getitem_781 = getitem_782 = getitem_783 = getitem_784 = getitem_785 = getitem_786 = getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = getitem_794 = getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = getitem_802 = getitem_803 = getitem_804 = getitem_805 = getitem_806 = getitem_807 = getitem_808 = getitem_809 = getitem_810 = getitem_811 = getitem_812 = getitem_813 = getitem_814 = getitem_815 = getitem_816 = getitem_817 = getitem_818 = getitem_819 = getitem_820 = getitem_821 = getitem_822 = getitem_823 = arg824_1 = arg825_1 = arg826_1 = arg827_1 = arg828_1 = arg829_1 = arg830_1 = arg831_1 = arg832_1 = arg833_1 = arg834_1 = arg835_1 = arg836_1 = arg837_1 = arg838_1 = arg839_1 = arg840_1 = arg841_1 = arg842_1 = arg843_1 = arg844_1 = arg845_1 = arg846_1 = arg847_1 = arg848_1 = arg849_1 = arg850_1 = arg851_1 = arg852_1 = arg853_1 = arg854_1 = arg855_1 = arg856_1 = arg857_1 = arg858_1 = arg859_1 = arg860_1 = arg861_1 = arg862_1 = arg863_1 = arg864_1 = arg865_1 = arg866_1 = arg867_1 = arg868_1 = arg869_1 = arg870_1 = arg871_1 = arg872_1 = arg873_1 = arg874_1 = arg875_1 = arg876_1 = arg877_1 = arg878_1 = arg879_1 = arg880_1 = arg881_1 = arg882_1 = arg883_1 = arg884_1 = arg885_1 = arg886_1 = arg887_1 = arg888_1 = arg889_1 = arg890_1 = arg891_1 = arg892_1 = arg893_1 = arg894_1 = arg895_1 = arg896_1 = arg897_1 = arg898_1 = arg899_1 = arg900_1 = arg901_1 = arg902_1 = arg903_1 = arg904_1 = arg905_1 = arg906_1 = arg907_1 = arg908_1 = arg909_1 = arg910_1 = arg911_1 = arg912_1 = arg913_1 = arg914_1 = arg915_1 = arg916_1 = arg917_1 = arg918_1 = arg919_1 = arg920_1 = arg921_1 = arg922_1 = arg923_1 = arg924_1 = arg925_1 = arg926_1 = arg927_1 = arg928_1 = arg929_1 = arg930_1 = arg931_1 = arg932_1 = arg933_1 = arg934_1 = arg935_1 = arg936_1 = arg937_1 = arg938_1 = arg939_1 = arg940_1 = arg941_1 = arg942_1 = arg943_1 = arg944_1 = arg945_1 = arg946_1 = arg947_1 = arg948_1 = arg949_1 = arg950_1 = arg951_1 = arg952_1 = arg953_1 = arg954_1 = arg955_1 = arg956_1 = arg957_1 = arg958_1 = arg959_1 = arg960_1 = arg961_1 = arg962_1 = arg963_1 = arg964_1 = arg965_1 = arg966_1 = arg967_1 = arg968_1 = arg969_1 = arg970_1 = arg971_1 = arg972_1 = arg973_1 = arg974_1 = arg975_1 = arg976_1 = arg977_1 = arg978_1 = arg979_1 = arg980_1 = arg981_1 = arg982_1 = arg983_1 = arg984_1 = arg985_1 = arg986_1 = arg987_1 = arg988_1 = arg989_1 = arg990_1 = arg991_1 = arg992_1 = arg993_1 = arg994_1 = arg995_1 = arg996_1 = arg997_1 = arg998_1 = arg999_1 = arg1000_1 = arg1001_1 = arg1002_1 = arg1003_1 = arg1004_1 = arg1005_1 = arg1006_1 = arg1007_1 = arg1008_1 = arg1009_1 = arg1010_1 = arg1011_1 = arg1012_1 = arg1013_1 = arg1014_1 = arg1015_1 = arg1016_1 = arg1017_1 = arg1018_1 = arg1019_1 = arg1020_1 = arg1021_1 = arg1022_1 = arg1023_1 = arg1024_1 = arg1025_1 = arg1026_1 = arg1027_1 = arg1028_1 = arg1029_1 = None
        getitem: "f32[30522, 768]" = _foreach_addcmul_scalar[0]
        getitem_824: "f32[512, 768]" = _foreach_addcmul_scalar[1]
        getitem_825: "f32[1024, 768]" = _foreach_addcmul_scalar[2]
        getitem_826: "f32[1024, 768]" = _foreach_addcmul_scalar[3]
        getitem_827: "f32[1024, 768]" = _foreach_addcmul_scalar[4]
        getitem_828: "f32[1024, 768]" = _foreach_addcmul_scalar[5]
        getitem_829: "f32[2, 768]" = _foreach_addcmul_scalar[6]
        getitem_830: "f32[768]" = _foreach_addcmul_scalar[7]
        getitem_831: "f32[768]" = _foreach_addcmul_scalar[8]
        getitem_832: "f32[768, 768]" = _foreach_addcmul_scalar[9]
        getitem_833: "f32[768]" = _foreach_addcmul_scalar[10]
        getitem_834: "f32[768, 768]" = _foreach_addcmul_scalar[11]
        getitem_835: "f32[768]" = _foreach_addcmul_scalar[12]
        getitem_836: "f32[768, 768]" = _foreach_addcmul_scalar[13]
        getitem_837: "f32[768]" = _foreach_addcmul_scalar[14]
        getitem_838: "f32[768, 768]" = _foreach_addcmul_scalar[15]
        getitem_839: "f32[768]" = _foreach_addcmul_scalar[16]
        getitem_840: "f32[768]" = _foreach_addcmul_scalar[17]
        getitem_841: "f32[768]" = _foreach_addcmul_scalar[18]
        getitem_842: "f32[3072, 768]" = _foreach_addcmul_scalar[19]
        getitem_843: "f32[3072]" = _foreach_addcmul_scalar[20]
        getitem_844: "f32[768, 3072]" = _foreach_addcmul_scalar[21]
        getitem_845: "f32[768]" = _foreach_addcmul_scalar[22]
        getitem_846: "f32[768]" = _foreach_addcmul_scalar[23]
        getitem_847: "f32[768]" = _foreach_addcmul_scalar[24]
        getitem_848: "f32[768, 768]" = _foreach_addcmul_scalar[25]
        getitem_849: "f32[768]" = _foreach_addcmul_scalar[26]
        getitem_850: "f32[768, 768]" = _foreach_addcmul_scalar[27]
        getitem_851: "f32[768]" = _foreach_addcmul_scalar[28]
        getitem_852: "f32[768, 768]" = _foreach_addcmul_scalar[29]
        getitem_853: "f32[768]" = _foreach_addcmul_scalar[30]
        getitem_854: "f32[768, 768]" = _foreach_addcmul_scalar[31]
        getitem_855: "f32[768]" = _foreach_addcmul_scalar[32]
        getitem_856: "f32[768]" = _foreach_addcmul_scalar[33]
        getitem_857: "f32[768]" = _foreach_addcmul_scalar[34]
        getitem_858: "f32[3072, 768]" = _foreach_addcmul_scalar[35]
        getitem_859: "f32[3072]" = _foreach_addcmul_scalar[36]
        getitem_860: "f32[768, 3072]" = _foreach_addcmul_scalar[37]
        getitem_861: "f32[768]" = _foreach_addcmul_scalar[38]
        getitem_862: "f32[768]" = _foreach_addcmul_scalar[39]
        getitem_863: "f32[768]" = _foreach_addcmul_scalar[40]
        getitem_864: "f32[768, 768]" = _foreach_addcmul_scalar[41]
        getitem_865: "f32[768]" = _foreach_addcmul_scalar[42]
        getitem_866: "f32[768, 768]" = _foreach_addcmul_scalar[43]
        getitem_867: "f32[768]" = _foreach_addcmul_scalar[44]
        getitem_868: "f32[768, 768]" = _foreach_addcmul_scalar[45]
        getitem_869: "f32[768]" = _foreach_addcmul_scalar[46]
        getitem_870: "f32[768, 768]" = _foreach_addcmul_scalar[47]
        getitem_871: "f32[768]" = _foreach_addcmul_scalar[48]
        getitem_872: "f32[768]" = _foreach_addcmul_scalar[49]
        getitem_873: "f32[768]" = _foreach_addcmul_scalar[50]
        getitem_874: "f32[3072, 768]" = _foreach_addcmul_scalar[51]
        getitem_875: "f32[3072]" = _foreach_addcmul_scalar[52]
        getitem_876: "f32[768, 3072]" = _foreach_addcmul_scalar[53]
        getitem_877: "f32[768]" = _foreach_addcmul_scalar[54]
        getitem_878: "f32[768]" = _foreach_addcmul_scalar[55]
        getitem_879: "f32[768]" = _foreach_addcmul_scalar[56]
        getitem_880: "f32[768, 768]" = _foreach_addcmul_scalar[57]
        getitem_881: "f32[768]" = _foreach_addcmul_scalar[58]
        getitem_882: "f32[768, 768]" = _foreach_addcmul_scalar[59]
        getitem_883: "f32[768]" = _foreach_addcmul_scalar[60]
        getitem_884: "f32[768, 768]" = _foreach_addcmul_scalar[61]
        getitem_885: "f32[768]" = _foreach_addcmul_scalar[62]
        getitem_886: "f32[768, 768]" = _foreach_addcmul_scalar[63]
        getitem_887: "f32[768]" = _foreach_addcmul_scalar[64]
        getitem_888: "f32[768]" = _foreach_addcmul_scalar[65]
        getitem_889: "f32[768]" = _foreach_addcmul_scalar[66]
        getitem_890: "f32[3072, 768]" = _foreach_addcmul_scalar[67]
        getitem_891: "f32[3072]" = _foreach_addcmul_scalar[68]
        getitem_892: "f32[768, 3072]" = _foreach_addcmul_scalar[69]
        getitem_893: "f32[768]" = _foreach_addcmul_scalar[70]
        getitem_894: "f32[768]" = _foreach_addcmul_scalar[71]
        getitem_895: "f32[768]" = _foreach_addcmul_scalar[72]
        getitem_896: "f32[768, 768]" = _foreach_addcmul_scalar[73]
        getitem_897: "f32[768]" = _foreach_addcmul_scalar[74]
        getitem_898: "f32[768, 768]" = _foreach_addcmul_scalar[75]
        getitem_899: "f32[768]" = _foreach_addcmul_scalar[76]
        getitem_900: "f32[768, 768]" = _foreach_addcmul_scalar[77]
        getitem_901: "f32[768]" = _foreach_addcmul_scalar[78]
        getitem_902: "f32[768, 768]" = _foreach_addcmul_scalar[79]
        getitem_903: "f32[768]" = _foreach_addcmul_scalar[80]
        getitem_904: "f32[768]" = _foreach_addcmul_scalar[81]
        getitem_905: "f32[768]" = _foreach_addcmul_scalar[82]
        getitem_906: "f32[3072, 768]" = _foreach_addcmul_scalar[83]
        getitem_907: "f32[3072]" = _foreach_addcmul_scalar[84]
        getitem_908: "f32[768, 3072]" = _foreach_addcmul_scalar[85]
        getitem_909: "f32[768]" = _foreach_addcmul_scalar[86]
        getitem_910: "f32[768]" = _foreach_addcmul_scalar[87]
        getitem_911: "f32[768]" = _foreach_addcmul_scalar[88]
        getitem_912: "f32[768, 768]" = _foreach_addcmul_scalar[89]
        getitem_913: "f32[768]" = _foreach_addcmul_scalar[90]
        getitem_914: "f32[768, 768]" = _foreach_addcmul_scalar[91]
        getitem_915: "f32[768]" = _foreach_addcmul_scalar[92]
        getitem_916: "f32[768, 768]" = _foreach_addcmul_scalar[93]
        getitem_917: "f32[768]" = _foreach_addcmul_scalar[94]
        getitem_918: "f32[768, 768]" = _foreach_addcmul_scalar[95]
        getitem_919: "f32[768]" = _foreach_addcmul_scalar[96]
        getitem_920: "f32[768]" = _foreach_addcmul_scalar[97]
        getitem_921: "f32[768]" = _foreach_addcmul_scalar[98]
        getitem_922: "f32[3072, 768]" = _foreach_addcmul_scalar[99]
        getitem_923: "f32[3072]" = _foreach_addcmul_scalar[100]
        getitem_924: "f32[768, 3072]" = _foreach_addcmul_scalar[101]
        getitem_925: "f32[768]" = _foreach_addcmul_scalar[102]
        getitem_926: "f32[768]" = _foreach_addcmul_scalar[103]
        getitem_927: "f32[768]" = _foreach_addcmul_scalar[104]
        getitem_928: "f32[768, 768]" = _foreach_addcmul_scalar[105]
        getitem_929: "f32[768]" = _foreach_addcmul_scalar[106]
        getitem_930: "f32[768, 768]" = _foreach_addcmul_scalar[107]
        getitem_931: "f32[768]" = _foreach_addcmul_scalar[108]
        getitem_932: "f32[768, 768]" = _foreach_addcmul_scalar[109]
        getitem_933: "f32[768]" = _foreach_addcmul_scalar[110]
        getitem_934: "f32[768, 768]" = _foreach_addcmul_scalar[111]
        getitem_935: "f32[768]" = _foreach_addcmul_scalar[112]
        getitem_936: "f32[768]" = _foreach_addcmul_scalar[113]
        getitem_937: "f32[768]" = _foreach_addcmul_scalar[114]
        getitem_938: "f32[3072, 768]" = _foreach_addcmul_scalar[115]
        getitem_939: "f32[3072]" = _foreach_addcmul_scalar[116]
        getitem_940: "f32[768, 3072]" = _foreach_addcmul_scalar[117]
        getitem_941: "f32[768]" = _foreach_addcmul_scalar[118]
        getitem_942: "f32[768]" = _foreach_addcmul_scalar[119]
        getitem_943: "f32[768]" = _foreach_addcmul_scalar[120]
        getitem_944: "f32[768, 768]" = _foreach_addcmul_scalar[121]
        getitem_945: "f32[768]" = _foreach_addcmul_scalar[122]
        getitem_946: "f32[768, 768]" = _foreach_addcmul_scalar[123]
        getitem_947: "f32[768]" = _foreach_addcmul_scalar[124]
        getitem_948: "f32[768, 768]" = _foreach_addcmul_scalar[125]
        getitem_949: "f32[768]" = _foreach_addcmul_scalar[126]
        getitem_950: "f32[768, 768]" = _foreach_addcmul_scalar[127]
        getitem_951: "f32[768]" = _foreach_addcmul_scalar[128]
        getitem_952: "f32[768]" = _foreach_addcmul_scalar[129]
        getitem_953: "f32[768]" = _foreach_addcmul_scalar[130]
        getitem_954: "f32[3072, 768]" = _foreach_addcmul_scalar[131]
        getitem_955: "f32[3072]" = _foreach_addcmul_scalar[132]
        getitem_956: "f32[768, 3072]" = _foreach_addcmul_scalar[133]
        getitem_957: "f32[768]" = _foreach_addcmul_scalar[134]
        getitem_958: "f32[768]" = _foreach_addcmul_scalar[135]
        getitem_959: "f32[768]" = _foreach_addcmul_scalar[136]
        getitem_960: "f32[768, 768]" = _foreach_addcmul_scalar[137]
        getitem_961: "f32[768]" = _foreach_addcmul_scalar[138]
        getitem_962: "f32[768, 768]" = _foreach_addcmul_scalar[139]
        getitem_963: "f32[768]" = _foreach_addcmul_scalar[140]
        getitem_964: "f32[768, 768]" = _foreach_addcmul_scalar[141]
        getitem_965: "f32[768]" = _foreach_addcmul_scalar[142]
        getitem_966: "f32[768, 768]" = _foreach_addcmul_scalar[143]
        getitem_967: "f32[768]" = _foreach_addcmul_scalar[144]
        getitem_968: "f32[768]" = _foreach_addcmul_scalar[145]
        getitem_969: "f32[768]" = _foreach_addcmul_scalar[146]
        getitem_970: "f32[3072, 768]" = _foreach_addcmul_scalar[147]
        getitem_971: "f32[3072]" = _foreach_addcmul_scalar[148]
        getitem_972: "f32[768, 3072]" = _foreach_addcmul_scalar[149]
        getitem_973: "f32[768]" = _foreach_addcmul_scalar[150]
        getitem_974: "f32[768]" = _foreach_addcmul_scalar[151]
        getitem_975: "f32[768]" = _foreach_addcmul_scalar[152]
        getitem_976: "f32[768, 768]" = _foreach_addcmul_scalar[153]
        getitem_977: "f32[768]" = _foreach_addcmul_scalar[154]
        getitem_978: "f32[768, 768]" = _foreach_addcmul_scalar[155]
        getitem_979: "f32[768]" = _foreach_addcmul_scalar[156]
        getitem_980: "f32[768, 768]" = _foreach_addcmul_scalar[157]
        getitem_981: "f32[768]" = _foreach_addcmul_scalar[158]
        getitem_982: "f32[768, 768]" = _foreach_addcmul_scalar[159]
        getitem_983: "f32[768]" = _foreach_addcmul_scalar[160]
        getitem_984: "f32[768]" = _foreach_addcmul_scalar[161]
        getitem_985: "f32[768]" = _foreach_addcmul_scalar[162]
        getitem_986: "f32[3072, 768]" = _foreach_addcmul_scalar[163]
        getitem_987: "f32[3072]" = _foreach_addcmul_scalar[164]
        getitem_988: "f32[768, 3072]" = _foreach_addcmul_scalar[165]
        getitem_989: "f32[768]" = _foreach_addcmul_scalar[166]
        getitem_990: "f32[768]" = _foreach_addcmul_scalar[167]
        getitem_991: "f32[768]" = _foreach_addcmul_scalar[168]
        getitem_992: "f32[768, 768]" = _foreach_addcmul_scalar[169]
        getitem_993: "f32[768]" = _foreach_addcmul_scalar[170]
        getitem_994: "f32[768, 768]" = _foreach_addcmul_scalar[171]
        getitem_995: "f32[768]" = _foreach_addcmul_scalar[172]
        getitem_996: "f32[768, 768]" = _foreach_addcmul_scalar[173]
        getitem_997: "f32[768]" = _foreach_addcmul_scalar[174]
        getitem_998: "f32[768, 768]" = _foreach_addcmul_scalar[175]
        getitem_999: "f32[768]" = _foreach_addcmul_scalar[176]
        getitem_1000: "f32[768]" = _foreach_addcmul_scalar[177]
        getitem_1001: "f32[768]" = _foreach_addcmul_scalar[178]
        getitem_1002: "f32[3072, 768]" = _foreach_addcmul_scalar[179]
        getitem_1003: "f32[3072]" = _foreach_addcmul_scalar[180]
        getitem_1004: "f32[768, 3072]" = _foreach_addcmul_scalar[181]
        getitem_1005: "f32[768]" = _foreach_addcmul_scalar[182]
        getitem_1006: "f32[768]" = _foreach_addcmul_scalar[183]
        getitem_1007: "f32[768]" = _foreach_addcmul_scalar[184]
        getitem_1008: "f32[768, 768]" = _foreach_addcmul_scalar[185]
        getitem_1009: "f32[768]" = _foreach_addcmul_scalar[186]
        getitem_1010: "f32[768, 768]" = _foreach_addcmul_scalar[187]
        getitem_1011: "f32[768]" = _foreach_addcmul_scalar[188]
        getitem_1012: "f32[768, 768]" = _foreach_addcmul_scalar[189]
        getitem_1013: "f32[768]" = _foreach_addcmul_scalar[190]
        getitem_1014: "f32[768, 768]" = _foreach_addcmul_scalar[191]
        getitem_1015: "f32[768]" = _foreach_addcmul_scalar[192]
        getitem_1016: "f32[768]" = _foreach_addcmul_scalar[193]
        getitem_1017: "f32[768]" = _foreach_addcmul_scalar[194]
        getitem_1018: "f32[3072, 768]" = _foreach_addcmul_scalar[195]
        getitem_1019: "f32[3072]" = _foreach_addcmul_scalar[196]
        getitem_1020: "f32[768, 3072]" = _foreach_addcmul_scalar[197]
        getitem_1021: "f32[768]" = _foreach_addcmul_scalar[198]
        getitem_1022: "f32[768]" = _foreach_addcmul_scalar[199]
        getitem_1023: "f32[768]" = _foreach_addcmul_scalar[200]
        getitem_1024: "f32[30522]" = _foreach_addcmul_scalar[201]
        getitem_1025: "f32[768, 768]" = _foreach_addcmul_scalar[202]
        getitem_1026: "f32[768]" = _foreach_addcmul_scalar[203]
        getitem_1027: "f32[768]" = _foreach_addcmul_scalar[204]
        getitem_1028: "f32[768]" = _foreach_addcmul_scalar[205];  _foreach_addcmul_scalar = None
        getitem_1029 = getitem
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem_1029, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205]);  getitem_1029 = getitem_1 = getitem_2 = getitem_3 = getitem_4 = getitem_5 = getitem_6 = getitem_7 = getitem_8 = getitem_9 = getitem_10 = getitem_11 = getitem_12 = getitem_13 = getitem_14 = getitem_15 = getitem_16 = getitem_17 = getitem_18 = getitem_19 = getitem_20 = getitem_21 = getitem_22 = getitem_23 = getitem_24 = getitem_25 = getitem_26 = getitem_27 = getitem_28 = getitem_29 = getitem_30 = getitem_31 = getitem_32 = getitem_33 = getitem_34 = getitem_35 = getitem_36 = getitem_37 = getitem_38 = getitem_39 = getitem_40 = getitem_41 = getitem_42 = getitem_43 = getitem_44 = getitem_45 = getitem_46 = getitem_47 = getitem_48 = getitem_49 = getitem_50 = getitem_51 = getitem_52 = getitem_53 = getitem_54 = getitem_55 = getitem_56 = getitem_57 = getitem_58 = getitem_59 = getitem_60 = getitem_61 = getitem_62 = getitem_63 = getitem_64 = getitem_65 = getitem_66 = getitem_67 = getitem_68 = getitem_69 = getitem_70 = getitem_71 = getitem_72 = getitem_73 = getitem_74 = getitem_75 = getitem_76 = getitem_77 = getitem_78 = getitem_79 = getitem_80 = getitem_81 = getitem_82 = getitem_83 = getitem_84 = getitem_85 = getitem_86 = getitem_87 = getitem_88 = getitem_89 = getitem_90 = getitem_91 = getitem_92 = getitem_93 = getitem_94 = getitem_95 = getitem_96 = getitem_97 = getitem_98 = getitem_99 = getitem_100 = getitem_101 = getitem_102 = getitem_103 = getitem_104 = getitem_105 = getitem_106 = getitem_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = getitem_205 = None
        getitem_206: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_207: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_208: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_209: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_210: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_211: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_212: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_213: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_214: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_215: "f32[]" = _foreach_pow_scalar_and_tensor[9]
        getitem_216: "f32[]" = _foreach_pow_scalar_and_tensor[10]
        getitem_217: "f32[]" = _foreach_pow_scalar_and_tensor[11]
        getitem_218: "f32[]" = _foreach_pow_scalar_and_tensor[12]
        getitem_219: "f32[]" = _foreach_pow_scalar_and_tensor[13]
        getitem_220: "f32[]" = _foreach_pow_scalar_and_tensor[14]
        getitem_221: "f32[]" = _foreach_pow_scalar_and_tensor[15]
        getitem_222: "f32[]" = _foreach_pow_scalar_and_tensor[16]
        getitem_223: "f32[]" = _foreach_pow_scalar_and_tensor[17]
        getitem_224: "f32[]" = _foreach_pow_scalar_and_tensor[18]
        getitem_225: "f32[]" = _foreach_pow_scalar_and_tensor[19]
        getitem_226: "f32[]" = _foreach_pow_scalar_and_tensor[20]
        getitem_227: "f32[]" = _foreach_pow_scalar_and_tensor[21]
        getitem_228: "f32[]" = _foreach_pow_scalar_and_tensor[22]
        getitem_229: "f32[]" = _foreach_pow_scalar_and_tensor[23]
        getitem_230: "f32[]" = _foreach_pow_scalar_and_tensor[24]
        getitem_231: "f32[]" = _foreach_pow_scalar_and_tensor[25]
        getitem_232: "f32[]" = _foreach_pow_scalar_and_tensor[26]
        getitem_233: "f32[]" = _foreach_pow_scalar_and_tensor[27]
        getitem_234: "f32[]" = _foreach_pow_scalar_and_tensor[28]
        getitem_235: "f32[]" = _foreach_pow_scalar_and_tensor[29]
        getitem_236: "f32[]" = _foreach_pow_scalar_and_tensor[30]
        getitem_237: "f32[]" = _foreach_pow_scalar_and_tensor[31]
        getitem_238: "f32[]" = _foreach_pow_scalar_and_tensor[32]
        getitem_239: "f32[]" = _foreach_pow_scalar_and_tensor[33]
        getitem_240: "f32[]" = _foreach_pow_scalar_and_tensor[34]
        getitem_241: "f32[]" = _foreach_pow_scalar_and_tensor[35]
        getitem_242: "f32[]" = _foreach_pow_scalar_and_tensor[36]
        getitem_243: "f32[]" = _foreach_pow_scalar_and_tensor[37]
        getitem_244: "f32[]" = _foreach_pow_scalar_and_tensor[38]
        getitem_245: "f32[]" = _foreach_pow_scalar_and_tensor[39]
        getitem_246: "f32[]" = _foreach_pow_scalar_and_tensor[40]
        getitem_247: "f32[]" = _foreach_pow_scalar_and_tensor[41]
        getitem_248: "f32[]" = _foreach_pow_scalar_and_tensor[42]
        getitem_249: "f32[]" = _foreach_pow_scalar_and_tensor[43]
        getitem_250: "f32[]" = _foreach_pow_scalar_and_tensor[44]
        getitem_251: "f32[]" = _foreach_pow_scalar_and_tensor[45]
        getitem_252: "f32[]" = _foreach_pow_scalar_and_tensor[46]
        getitem_253: "f32[]" = _foreach_pow_scalar_and_tensor[47]
        getitem_254: "f32[]" = _foreach_pow_scalar_and_tensor[48]
        getitem_255: "f32[]" = _foreach_pow_scalar_and_tensor[49]
        getitem_256: "f32[]" = _foreach_pow_scalar_and_tensor[50]
        getitem_257: "f32[]" = _foreach_pow_scalar_and_tensor[51]
        getitem_258: "f32[]" = _foreach_pow_scalar_and_tensor[52]
        getitem_259: "f32[]" = _foreach_pow_scalar_and_tensor[53]
        getitem_260: "f32[]" = _foreach_pow_scalar_and_tensor[54]
        getitem_261: "f32[]" = _foreach_pow_scalar_and_tensor[55]
        getitem_262: "f32[]" = _foreach_pow_scalar_and_tensor[56]
        getitem_263: "f32[]" = _foreach_pow_scalar_and_tensor[57]
        getitem_264: "f32[]" = _foreach_pow_scalar_and_tensor[58]
        getitem_265: "f32[]" = _foreach_pow_scalar_and_tensor[59]
        getitem_266: "f32[]" = _foreach_pow_scalar_and_tensor[60]
        getitem_267: "f32[]" = _foreach_pow_scalar_and_tensor[61]
        getitem_268: "f32[]" = _foreach_pow_scalar_and_tensor[62]
        getitem_269: "f32[]" = _foreach_pow_scalar_and_tensor[63]
        getitem_270: "f32[]" = _foreach_pow_scalar_and_tensor[64]
        getitem_271: "f32[]" = _foreach_pow_scalar_and_tensor[65]
        getitem_272: "f32[]" = _foreach_pow_scalar_and_tensor[66]
        getitem_273: "f32[]" = _foreach_pow_scalar_and_tensor[67]
        getitem_274: "f32[]" = _foreach_pow_scalar_and_tensor[68]
        getitem_275: "f32[]" = _foreach_pow_scalar_and_tensor[69]
        getitem_276: "f32[]" = _foreach_pow_scalar_and_tensor[70]
        getitem_277: "f32[]" = _foreach_pow_scalar_and_tensor[71]
        getitem_278: "f32[]" = _foreach_pow_scalar_and_tensor[72]
        getitem_279: "f32[]" = _foreach_pow_scalar_and_tensor[73]
        getitem_280: "f32[]" = _foreach_pow_scalar_and_tensor[74]
        getitem_281: "f32[]" = _foreach_pow_scalar_and_tensor[75]
        getitem_282: "f32[]" = _foreach_pow_scalar_and_tensor[76]
        getitem_283: "f32[]" = _foreach_pow_scalar_and_tensor[77]
        getitem_284: "f32[]" = _foreach_pow_scalar_and_tensor[78]
        getitem_285: "f32[]" = _foreach_pow_scalar_and_tensor[79]
        getitem_286: "f32[]" = _foreach_pow_scalar_and_tensor[80]
        getitem_287: "f32[]" = _foreach_pow_scalar_and_tensor[81]
        getitem_288: "f32[]" = _foreach_pow_scalar_and_tensor[82]
        getitem_289: "f32[]" = _foreach_pow_scalar_and_tensor[83]
        getitem_290: "f32[]" = _foreach_pow_scalar_and_tensor[84]
        getitem_291: "f32[]" = _foreach_pow_scalar_and_tensor[85]
        getitem_292: "f32[]" = _foreach_pow_scalar_and_tensor[86]
        getitem_293: "f32[]" = _foreach_pow_scalar_and_tensor[87]
        getitem_294: "f32[]" = _foreach_pow_scalar_and_tensor[88]
        getitem_295: "f32[]" = _foreach_pow_scalar_and_tensor[89]
        getitem_296: "f32[]" = _foreach_pow_scalar_and_tensor[90]
        getitem_297: "f32[]" = _foreach_pow_scalar_and_tensor[91]
        getitem_298: "f32[]" = _foreach_pow_scalar_and_tensor[92]
        getitem_299: "f32[]" = _foreach_pow_scalar_and_tensor[93]
        getitem_300: "f32[]" = _foreach_pow_scalar_and_tensor[94]
        getitem_301: "f32[]" = _foreach_pow_scalar_and_tensor[95]
        getitem_302: "f32[]" = _foreach_pow_scalar_and_tensor[96]
        getitem_303: "f32[]" = _foreach_pow_scalar_and_tensor[97]
        getitem_304: "f32[]" = _foreach_pow_scalar_and_tensor[98]
        getitem_305: "f32[]" = _foreach_pow_scalar_and_tensor[99]
        getitem_306: "f32[]" = _foreach_pow_scalar_and_tensor[100]
        getitem_307: "f32[]" = _foreach_pow_scalar_and_tensor[101]
        getitem_308: "f32[]" = _foreach_pow_scalar_and_tensor[102]
        getitem_309: "f32[]" = _foreach_pow_scalar_and_tensor[103]
        getitem_310: "f32[]" = _foreach_pow_scalar_and_tensor[104]
        getitem_311: "f32[]" = _foreach_pow_scalar_and_tensor[105]
        getitem_312: "f32[]" = _foreach_pow_scalar_and_tensor[106]
        getitem_313: "f32[]" = _foreach_pow_scalar_and_tensor[107]
        getitem_314: "f32[]" = _foreach_pow_scalar_and_tensor[108]
        getitem_315: "f32[]" = _foreach_pow_scalar_and_tensor[109]
        getitem_316: "f32[]" = _foreach_pow_scalar_and_tensor[110]
        getitem_317: "f32[]" = _foreach_pow_scalar_and_tensor[111]
        getitem_318: "f32[]" = _foreach_pow_scalar_and_tensor[112]
        getitem_319: "f32[]" = _foreach_pow_scalar_and_tensor[113]
        getitem_320: "f32[]" = _foreach_pow_scalar_and_tensor[114]
        getitem_321: "f32[]" = _foreach_pow_scalar_and_tensor[115]
        getitem_322: "f32[]" = _foreach_pow_scalar_and_tensor[116]
        getitem_323: "f32[]" = _foreach_pow_scalar_and_tensor[117]
        getitem_324: "f32[]" = _foreach_pow_scalar_and_tensor[118]
        getitem_325: "f32[]" = _foreach_pow_scalar_and_tensor[119]
        getitem_326: "f32[]" = _foreach_pow_scalar_and_tensor[120]
        getitem_327: "f32[]" = _foreach_pow_scalar_and_tensor[121]
        getitem_328: "f32[]" = _foreach_pow_scalar_and_tensor[122]
        getitem_329: "f32[]" = _foreach_pow_scalar_and_tensor[123]
        getitem_330: "f32[]" = _foreach_pow_scalar_and_tensor[124]
        getitem_331: "f32[]" = _foreach_pow_scalar_and_tensor[125]
        getitem_332: "f32[]" = _foreach_pow_scalar_and_tensor[126]
        getitem_333: "f32[]" = _foreach_pow_scalar_and_tensor[127]
        getitem_334: "f32[]" = _foreach_pow_scalar_and_tensor[128]
        getitem_335: "f32[]" = _foreach_pow_scalar_and_tensor[129]
        getitem_336: "f32[]" = _foreach_pow_scalar_and_tensor[130]
        getitem_337: "f32[]" = _foreach_pow_scalar_and_tensor[131]
        getitem_338: "f32[]" = _foreach_pow_scalar_and_tensor[132]
        getitem_339: "f32[]" = _foreach_pow_scalar_and_tensor[133]
        getitem_340: "f32[]" = _foreach_pow_scalar_and_tensor[134]
        getitem_341: "f32[]" = _foreach_pow_scalar_and_tensor[135]
        getitem_342: "f32[]" = _foreach_pow_scalar_and_tensor[136]
        getitem_343: "f32[]" = _foreach_pow_scalar_and_tensor[137]
        getitem_344: "f32[]" = _foreach_pow_scalar_and_tensor[138]
        getitem_345: "f32[]" = _foreach_pow_scalar_and_tensor[139]
        getitem_346: "f32[]" = _foreach_pow_scalar_and_tensor[140]
        getitem_347: "f32[]" = _foreach_pow_scalar_and_tensor[141]
        getitem_348: "f32[]" = _foreach_pow_scalar_and_tensor[142]
        getitem_349: "f32[]" = _foreach_pow_scalar_and_tensor[143]
        getitem_350: "f32[]" = _foreach_pow_scalar_and_tensor[144]
        getitem_351: "f32[]" = _foreach_pow_scalar_and_tensor[145]
        getitem_352: "f32[]" = _foreach_pow_scalar_and_tensor[146]
        getitem_353: "f32[]" = _foreach_pow_scalar_and_tensor[147]
        getitem_354: "f32[]" = _foreach_pow_scalar_and_tensor[148]
        getitem_355: "f32[]" = _foreach_pow_scalar_and_tensor[149]
        getitem_356: "f32[]" = _foreach_pow_scalar_and_tensor[150]
        getitem_357: "f32[]" = _foreach_pow_scalar_and_tensor[151]
        getitem_358: "f32[]" = _foreach_pow_scalar_and_tensor[152]
        getitem_359: "f32[]" = _foreach_pow_scalar_and_tensor[153]
        getitem_360: "f32[]" = _foreach_pow_scalar_and_tensor[154]
        getitem_361: "f32[]" = _foreach_pow_scalar_and_tensor[155]
        getitem_362: "f32[]" = _foreach_pow_scalar_and_tensor[156]
        getitem_363: "f32[]" = _foreach_pow_scalar_and_tensor[157]
        getitem_364: "f32[]" = _foreach_pow_scalar_and_tensor[158]
        getitem_365: "f32[]" = _foreach_pow_scalar_and_tensor[159]
        getitem_366: "f32[]" = _foreach_pow_scalar_and_tensor[160]
        getitem_367: "f32[]" = _foreach_pow_scalar_and_tensor[161]
        getitem_368: "f32[]" = _foreach_pow_scalar_and_tensor[162]
        getitem_369: "f32[]" = _foreach_pow_scalar_and_tensor[163]
        getitem_370: "f32[]" = _foreach_pow_scalar_and_tensor[164]
        getitem_371: "f32[]" = _foreach_pow_scalar_and_tensor[165]
        getitem_372: "f32[]" = _foreach_pow_scalar_and_tensor[166]
        getitem_373: "f32[]" = _foreach_pow_scalar_and_tensor[167]
        getitem_374: "f32[]" = _foreach_pow_scalar_and_tensor[168]
        getitem_375: "f32[]" = _foreach_pow_scalar_and_tensor[169]
        getitem_376: "f32[]" = _foreach_pow_scalar_and_tensor[170]
        getitem_377: "f32[]" = _foreach_pow_scalar_and_tensor[171]
        getitem_378: "f32[]" = _foreach_pow_scalar_and_tensor[172]
        getitem_379: "f32[]" = _foreach_pow_scalar_and_tensor[173]
        getitem_380: "f32[]" = _foreach_pow_scalar_and_tensor[174]
        getitem_381: "f32[]" = _foreach_pow_scalar_and_tensor[175]
        getitem_382: "f32[]" = _foreach_pow_scalar_and_tensor[176]
        getitem_383: "f32[]" = _foreach_pow_scalar_and_tensor[177]
        getitem_384: "f32[]" = _foreach_pow_scalar_and_tensor[178]
        getitem_385: "f32[]" = _foreach_pow_scalar_and_tensor[179]
        getitem_386: "f32[]" = _foreach_pow_scalar_and_tensor[180]
        getitem_387: "f32[]" = _foreach_pow_scalar_and_tensor[181]
        getitem_388: "f32[]" = _foreach_pow_scalar_and_tensor[182]
        getitem_389: "f32[]" = _foreach_pow_scalar_and_tensor[183]
        getitem_390: "f32[]" = _foreach_pow_scalar_and_tensor[184]
        getitem_391: "f32[]" = _foreach_pow_scalar_and_tensor[185]
        getitem_392: "f32[]" = _foreach_pow_scalar_and_tensor[186]
        getitem_393: "f32[]" = _foreach_pow_scalar_and_tensor[187]
        getitem_394: "f32[]" = _foreach_pow_scalar_and_tensor[188]
        getitem_395: "f32[]" = _foreach_pow_scalar_and_tensor[189]
        getitem_396: "f32[]" = _foreach_pow_scalar_and_tensor[190]
        getitem_397: "f32[]" = _foreach_pow_scalar_and_tensor[191]
        getitem_398: "f32[]" = _foreach_pow_scalar_and_tensor[192]
        getitem_399: "f32[]" = _foreach_pow_scalar_and_tensor[193]
        getitem_400: "f32[]" = _foreach_pow_scalar_and_tensor[194]
        getitem_401: "f32[]" = _foreach_pow_scalar_and_tensor[195]
        getitem_402: "f32[]" = _foreach_pow_scalar_and_tensor[196]
        getitem_403: "f32[]" = _foreach_pow_scalar_and_tensor[197]
        getitem_404: "f32[]" = _foreach_pow_scalar_and_tensor[198]
        getitem_405: "f32[]" = _foreach_pow_scalar_and_tensor[199]
        getitem_406: "f32[]" = _foreach_pow_scalar_and_tensor[200]
        getitem_407: "f32[]" = _foreach_pow_scalar_and_tensor[201]
        getitem_408: "f32[]" = _foreach_pow_scalar_and_tensor[202]
        getitem_409: "f32[]" = _foreach_pow_scalar_and_tensor[203]
        getitem_410: "f32[]" = _foreach_pow_scalar_and_tensor[204]
        getitem_411: "f32[]" = _foreach_pow_scalar_and_tensor[205];  _foreach_pow_scalar_and_tensor = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727, getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835, getitem_1836, getitem_1837, getitem_1838, getitem_1839, getitem_1840, getitem_1841, getitem_1842, getitem_1843, getitem_1844, getitem_1845, getitem_1846, getitem_1847, getitem_1848, getitem_1849, getitem_1850, getitem_1851, getitem_1852, getitem_1853]);  getitem_1648 = getitem_1649 = getitem_1650 = getitem_1651 = getitem_1652 = getitem_1653 = getitem_1654 = getitem_1655 = getitem_1656 = getitem_1657 = getitem_1658 = getitem_1659 = getitem_1660 = getitem_1661 = getitem_1662 = getitem_1663 = getitem_1664 = getitem_1665 = getitem_1666 = getitem_1667 = getitem_1668 = getitem_1669 = getitem_1670 = getitem_1671 = getitem_1672 = getitem_1673 = getitem_1674 = getitem_1675 = getitem_1676 = getitem_1677 = getitem_1678 = getitem_1679 = getitem_1680 = getitem_1681 = getitem_1682 = getitem_1683 = getitem_1684 = getitem_1685 = getitem_1686 = getitem_1687 = getitem_1688 = getitem_1689 = getitem_1690 = getitem_1691 = getitem_1692 = getitem_1693 = getitem_1694 = getitem_1695 = getitem_1696 = getitem_1697 = getitem_1698 = getitem_1699 = getitem_1700 = getitem_1701 = getitem_1702 = getitem_1703 = getitem_1704 = getitem_1705 = getitem_1706 = getitem_1707 = getitem_1708 = getitem_1709 = getitem_1710 = getitem_1711 = getitem_1712 = getitem_1713 = getitem_1714 = getitem_1715 = getitem_1716 = getitem_1717 = getitem_1718 = getitem_1719 = getitem_1720 = getitem_1721 = getitem_1722 = getitem_1723 = getitem_1724 = getitem_1725 = getitem_1726 = getitem_1727 = getitem_1728 = getitem_1729 = getitem_1730 = getitem_1731 = getitem_1732 = getitem_1733 = getitem_1734 = getitem_1735 = getitem_1736 = getitem_1737 = getitem_1738 = getitem_1739 = getitem_1740 = getitem_1741 = getitem_1742 = getitem_1743 = getitem_1744 = getitem_1745 = getitem_1746 = getitem_1747 = getitem_1748 = getitem_1749 = getitem_1750 = getitem_1751 = getitem_1752 = getitem_1753 = getitem_1754 = getitem_1755 = getitem_1756 = getitem_1757 = getitem_1758 = getitem_1759 = getitem_1760 = getitem_1761 = getitem_1762 = getitem_1763 = getitem_1764 = getitem_1765 = getitem_1766 = getitem_1767 = getitem_1768 = getitem_1769 = getitem_1770 = getitem_1771 = getitem_1772 = getitem_1773 = getitem_1774 = getitem_1775 = getitem_1776 = getitem_1777 = getitem_1778 = getitem_1779 = getitem_1780 = getitem_1781 = getitem_1782 = getitem_1783 = getitem_1784 = getitem_1785 = getitem_1786 = getitem_1787 = getitem_1788 = getitem_1789 = getitem_1790 = getitem_1791 = getitem_1792 = getitem_1793 = getitem_1794 = getitem_1795 = getitem_1796 = getitem_1797 = getitem_1798 = getitem_1799 = getitem_1800 = getitem_1801 = getitem_1802 = getitem_1803 = getitem_1804 = getitem_1805 = getitem_1806 = getitem_1807 = getitem_1808 = getitem_1809 = getitem_1810 = getitem_1811 = getitem_1812 = getitem_1813 = getitem_1814 = getitem_1815 = getitem_1816 = getitem_1817 = getitem_1818 = getitem_1819 = getitem_1820 = getitem_1821 = getitem_1822 = getitem_1823 = getitem_1824 = getitem_1825 = getitem_1826 = getitem_1827 = getitem_1828 = getitem_1829 = getitem_1830 = getitem_1831 = getitem_1832 = getitem_1833 = getitem_1834 = getitem_1835 = getitem_1836 = getitem_1837 = getitem_1838 = getitem_1839 = getitem_1840 = getitem_1841 = getitem_1842 = getitem_1843 = getitem_1844 = getitem_1845 = getitem_1846 = getitem_1847 = getitem_1848 = getitem_1849 = getitem_1850 = getitem_1851 = getitem_1852 = getitem_1853 = None
        getitem_1854: "f32[]" = _foreach_neg_default[0]
        getitem_1855: "f32[]" = _foreach_neg_default[1]
        getitem_1856: "f32[]" = _foreach_neg_default[2]
        getitem_1857: "f32[]" = _foreach_neg_default[3]
        getitem_1858: "f32[]" = _foreach_neg_default[4]
        getitem_1859: "f32[]" = _foreach_neg_default[5]
        getitem_1860: "f32[]" = _foreach_neg_default[6]
        getitem_1861: "f32[]" = _foreach_neg_default[7]
        getitem_1862: "f32[]" = _foreach_neg_default[8]
        getitem_1863: "f32[]" = _foreach_neg_default[9]
        getitem_1864: "f32[]" = _foreach_neg_default[10]
        getitem_1865: "f32[]" = _foreach_neg_default[11]
        getitem_1866: "f32[]" = _foreach_neg_default[12]
        getitem_1867: "f32[]" = _foreach_neg_default[13]
        getitem_1868: "f32[]" = _foreach_neg_default[14]
        getitem_1869: "f32[]" = _foreach_neg_default[15]
        getitem_1870: "f32[]" = _foreach_neg_default[16]
        getitem_1871: "f32[]" = _foreach_neg_default[17]
        getitem_1872: "f32[]" = _foreach_neg_default[18]
        getitem_1873: "f32[]" = _foreach_neg_default[19]
        getitem_1874: "f32[]" = _foreach_neg_default[20]
        getitem_1875: "f32[]" = _foreach_neg_default[21]
        getitem_1876: "f32[]" = _foreach_neg_default[22]
        getitem_1877: "f32[]" = _foreach_neg_default[23]
        getitem_1878: "f32[]" = _foreach_neg_default[24]
        getitem_1879: "f32[]" = _foreach_neg_default[25]
        getitem_1880: "f32[]" = _foreach_neg_default[26]
        getitem_1881: "f32[]" = _foreach_neg_default[27]
        getitem_1882: "f32[]" = _foreach_neg_default[28]
        getitem_1883: "f32[]" = _foreach_neg_default[29]
        getitem_1884: "f32[]" = _foreach_neg_default[30]
        getitem_1885: "f32[]" = _foreach_neg_default[31]
        getitem_1886: "f32[]" = _foreach_neg_default[32]
        getitem_1887: "f32[]" = _foreach_neg_default[33]
        getitem_1888: "f32[]" = _foreach_neg_default[34]
        getitem_1889: "f32[]" = _foreach_neg_default[35]
        getitem_1890: "f32[]" = _foreach_neg_default[36]
        getitem_1891: "f32[]" = _foreach_neg_default[37]
        getitem_1892: "f32[]" = _foreach_neg_default[38]
        getitem_1893: "f32[]" = _foreach_neg_default[39]
        getitem_1894: "f32[]" = _foreach_neg_default[40]
        getitem_1895: "f32[]" = _foreach_neg_default[41]
        getitem_1896: "f32[]" = _foreach_neg_default[42]
        getitem_1897: "f32[]" = _foreach_neg_default[43]
        getitem_1898: "f32[]" = _foreach_neg_default[44]
        getitem_1899: "f32[]" = _foreach_neg_default[45]
        getitem_1900: "f32[]" = _foreach_neg_default[46]
        getitem_1901: "f32[]" = _foreach_neg_default[47]
        getitem_1902: "f32[]" = _foreach_neg_default[48]
        getitem_1903: "f32[]" = _foreach_neg_default[49]
        getitem_1904: "f32[]" = _foreach_neg_default[50]
        getitem_1905: "f32[]" = _foreach_neg_default[51]
        getitem_1906: "f32[]" = _foreach_neg_default[52]
        getitem_1907: "f32[]" = _foreach_neg_default[53]
        getitem_1908: "f32[]" = _foreach_neg_default[54]
        getitem_1909: "f32[]" = _foreach_neg_default[55]
        getitem_1910: "f32[]" = _foreach_neg_default[56]
        getitem_1911: "f32[]" = _foreach_neg_default[57]
        getitem_1912: "f32[]" = _foreach_neg_default[58]
        getitem_1913: "f32[]" = _foreach_neg_default[59]
        getitem_1914: "f32[]" = _foreach_neg_default[60]
        getitem_1915: "f32[]" = _foreach_neg_default[61]
        getitem_1916: "f32[]" = _foreach_neg_default[62]
        getitem_1917: "f32[]" = _foreach_neg_default[63]
        getitem_1918: "f32[]" = _foreach_neg_default[64]
        getitem_1919: "f32[]" = _foreach_neg_default[65]
        getitem_1920: "f32[]" = _foreach_neg_default[66]
        getitem_1921: "f32[]" = _foreach_neg_default[67]
        getitem_1922: "f32[]" = _foreach_neg_default[68]
        getitem_1923: "f32[]" = _foreach_neg_default[69]
        getitem_1924: "f32[]" = _foreach_neg_default[70]
        getitem_1925: "f32[]" = _foreach_neg_default[71]
        getitem_1926: "f32[]" = _foreach_neg_default[72]
        getitem_1927: "f32[]" = _foreach_neg_default[73]
        getitem_1928: "f32[]" = _foreach_neg_default[74]
        getitem_1929: "f32[]" = _foreach_neg_default[75]
        getitem_1930: "f32[]" = _foreach_neg_default[76]
        getitem_1931: "f32[]" = _foreach_neg_default[77]
        getitem_1932: "f32[]" = _foreach_neg_default[78]
        getitem_1933: "f32[]" = _foreach_neg_default[79]
        getitem_1934: "f32[]" = _foreach_neg_default[80]
        getitem_1935: "f32[]" = _foreach_neg_default[81]
        getitem_1936: "f32[]" = _foreach_neg_default[82]
        getitem_1937: "f32[]" = _foreach_neg_default[83]
        getitem_1938: "f32[]" = _foreach_neg_default[84]
        getitem_1939: "f32[]" = _foreach_neg_default[85]
        getitem_1940: "f32[]" = _foreach_neg_default[86]
        getitem_1941: "f32[]" = _foreach_neg_default[87]
        getitem_1942: "f32[]" = _foreach_neg_default[88]
        getitem_1943: "f32[]" = _foreach_neg_default[89]
        getitem_1944: "f32[]" = _foreach_neg_default[90]
        getitem_1945: "f32[]" = _foreach_neg_default[91]
        getitem_1946: "f32[]" = _foreach_neg_default[92]
        getitem_1947: "f32[]" = _foreach_neg_default[93]
        getitem_1948: "f32[]" = _foreach_neg_default[94]
        getitem_1949: "f32[]" = _foreach_neg_default[95]
        getitem_1950: "f32[]" = _foreach_neg_default[96]
        getitem_1951: "f32[]" = _foreach_neg_default[97]
        getitem_1952: "f32[]" = _foreach_neg_default[98]
        getitem_1953: "f32[]" = _foreach_neg_default[99]
        getitem_1954: "f32[]" = _foreach_neg_default[100]
        getitem_1955: "f32[]" = _foreach_neg_default[101]
        getitem_1956: "f32[]" = _foreach_neg_default[102]
        getitem_1957: "f32[]" = _foreach_neg_default[103]
        getitem_1958: "f32[]" = _foreach_neg_default[104]
        getitem_1959: "f32[]" = _foreach_neg_default[105]
        getitem_1960: "f32[]" = _foreach_neg_default[106]
        getitem_1961: "f32[]" = _foreach_neg_default[107]
        getitem_1962: "f32[]" = _foreach_neg_default[108]
        getitem_1963: "f32[]" = _foreach_neg_default[109]
        getitem_1964: "f32[]" = _foreach_neg_default[110]
        getitem_1965: "f32[]" = _foreach_neg_default[111]
        getitem_1966: "f32[]" = _foreach_neg_default[112]
        getitem_1967: "f32[]" = _foreach_neg_default[113]
        getitem_1968: "f32[]" = _foreach_neg_default[114]
        getitem_1969: "f32[]" = _foreach_neg_default[115]
        getitem_1970: "f32[]" = _foreach_neg_default[116]
        getitem_1971: "f32[]" = _foreach_neg_default[117]
        getitem_1972: "f32[]" = _foreach_neg_default[118]
        getitem_1973: "f32[]" = _foreach_neg_default[119]
        getitem_1974: "f32[]" = _foreach_neg_default[120]
        getitem_1975: "f32[]" = _foreach_neg_default[121]
        getitem_1976: "f32[]" = _foreach_neg_default[122]
        getitem_1977: "f32[]" = _foreach_neg_default[123]
        getitem_1978: "f32[]" = _foreach_neg_default[124]
        getitem_1979: "f32[]" = _foreach_neg_default[125]
        getitem_1980: "f32[]" = _foreach_neg_default[126]
        getitem_1981: "f32[]" = _foreach_neg_default[127]
        getitem_1982: "f32[]" = _foreach_neg_default[128]
        getitem_1983: "f32[]" = _foreach_neg_default[129]
        getitem_1984: "f32[]" = _foreach_neg_default[130]
        getitem_1985: "f32[]" = _foreach_neg_default[131]
        getitem_1986: "f32[]" = _foreach_neg_default[132]
        getitem_1987: "f32[]" = _foreach_neg_default[133]
        getitem_1988: "f32[]" = _foreach_neg_default[134]
        getitem_1989: "f32[]" = _foreach_neg_default[135]
        getitem_1990: "f32[]" = _foreach_neg_default[136]
        getitem_1991: "f32[]" = _foreach_neg_default[137]
        getitem_1992: "f32[]" = _foreach_neg_default[138]
        getitem_1993: "f32[]" = _foreach_neg_default[139]
        getitem_1994: "f32[]" = _foreach_neg_default[140]
        getitem_1995: "f32[]" = _foreach_neg_default[141]
        getitem_1996: "f32[]" = _foreach_neg_default[142]
        getitem_1997: "f32[]" = _foreach_neg_default[143]
        getitem_1998: "f32[]" = _foreach_neg_default[144]
        getitem_1999: "f32[]" = _foreach_neg_default[145]
        getitem_2000: "f32[]" = _foreach_neg_default[146]
        getitem_2001: "f32[]" = _foreach_neg_default[147]
        getitem_2002: "f32[]" = _foreach_neg_default[148]
        getitem_2003: "f32[]" = _foreach_neg_default[149]
        getitem_2004: "f32[]" = _foreach_neg_default[150]
        getitem_2005: "f32[]" = _foreach_neg_default[151]
        getitem_2006: "f32[]" = _foreach_neg_default[152]
        getitem_2007: "f32[]" = _foreach_neg_default[153]
        getitem_2008: "f32[]" = _foreach_neg_default[154]
        getitem_2009: "f32[]" = _foreach_neg_default[155]
        getitem_2010: "f32[]" = _foreach_neg_default[156]
        getitem_2011: "f32[]" = _foreach_neg_default[157]
        getitem_2012: "f32[]" = _foreach_neg_default[158]
        getitem_2013: "f32[]" = _foreach_neg_default[159]
        getitem_2014: "f32[]" = _foreach_neg_default[160]
        getitem_2015: "f32[]" = _foreach_neg_default[161]
        getitem_2016: "f32[]" = _foreach_neg_default[162]
        getitem_2017: "f32[]" = _foreach_neg_default[163]
        getitem_2018: "f32[]" = _foreach_neg_default[164]
        getitem_2019: "f32[]" = _foreach_neg_default[165]
        getitem_2020: "f32[]" = _foreach_neg_default[166]
        getitem_2021: "f32[]" = _foreach_neg_default[167]
        getitem_2022: "f32[]" = _foreach_neg_default[168]
        getitem_2023: "f32[]" = _foreach_neg_default[169]
        getitem_2024: "f32[]" = _foreach_neg_default[170]
        getitem_2025: "f32[]" = _foreach_neg_default[171]
        getitem_2026: "f32[]" = _foreach_neg_default[172]
        getitem_2027: "f32[]" = _foreach_neg_default[173]
        getitem_2028: "f32[]" = _foreach_neg_default[174]
        getitem_2029: "f32[]" = _foreach_neg_default[175]
        getitem_2030: "f32[]" = _foreach_neg_default[176]
        getitem_2031: "f32[]" = _foreach_neg_default[177]
        getitem_2032: "f32[]" = _foreach_neg_default[178]
        getitem_2033: "f32[]" = _foreach_neg_default[179]
        getitem_2034: "f32[]" = _foreach_neg_default[180]
        getitem_2035: "f32[]" = _foreach_neg_default[181]
        getitem_2036: "f32[]" = _foreach_neg_default[182]
        getitem_2037: "f32[]" = _foreach_neg_default[183]
        getitem_2038: "f32[]" = _foreach_neg_default[184]
        getitem_2039: "f32[]" = _foreach_neg_default[185]
        getitem_2040: "f32[]" = _foreach_neg_default[186]
        getitem_2041: "f32[]" = _foreach_neg_default[187]
        getitem_2042: "f32[]" = _foreach_neg_default[188]
        getitem_2043: "f32[]" = _foreach_neg_default[189]
        getitem_2044: "f32[]" = _foreach_neg_default[190]
        getitem_2045: "f32[]" = _foreach_neg_default[191]
        getitem_2046: "f32[]" = _foreach_neg_default[192]
        getitem_2047: "f32[]" = _foreach_neg_default[193]
        getitem_2048: "f32[]" = _foreach_neg_default[194]
        getitem_2049: "f32[]" = _foreach_neg_default[195]
        getitem_2050: "f32[]" = _foreach_neg_default[196]
        getitem_2051: "f32[]" = _foreach_neg_default[197]
        getitem_2052: "f32[]" = _foreach_neg_default[198]
        getitem_2053: "f32[]" = _foreach_neg_default[199]
        getitem_2054: "f32[]" = _foreach_neg_default[200]
        getitem_2055: "f32[]" = _foreach_neg_default[201]
        getitem_2056: "f32[]" = _foreach_neg_default[202]
        getitem_2057: "f32[]" = _foreach_neg_default[203]
        getitem_2058: "f32[]" = _foreach_neg_default[204]
        getitem_2059: "f32[]" = _foreach_neg_default[205];  _foreach_neg_default = None
        return (getitem, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863, getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971, getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215, getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323, getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_1854, getitem_1855, getitem_1856, getitem_1857, getitem_1858, getitem_1859, getitem_1860, getitem_1861, getitem_1862, getitem_1863, getitem_1864, getitem_1865, getitem_1866, getitem_1867, getitem_1868, getitem_1869, getitem_1870, getitem_1871, getitem_1872, getitem_1873, getitem_1874, getitem_1875, getitem_1876, getitem_1877, getitem_1878, getitem_1879, getitem_1880, getitem_1881, getitem_1882, getitem_1883, getitem_1884, getitem_1885, getitem_1886, getitem_1887, getitem_1888, getitem_1889, getitem_1890, getitem_1891, getitem_1892, getitem_1893, getitem_1894, getitem_1895, getitem_1896, getitem_1897, getitem_1898, getitem_1899, getitem_1900, getitem_1901, getitem_1902, getitem_1903, getitem_1904, getitem_1905, getitem_1906, getitem_1907, getitem_1908, getitem_1909, getitem_1910, getitem_1911, getitem_1912, getitem_1913, getitem_1914, getitem_1915, getitem_1916, getitem_1917, getitem_1918, getitem_1919, getitem_1920, getitem_1921, getitem_1922, getitem_1923, getitem_1924, getitem_1925, getitem_1926, getitem_1927, getitem_1928, getitem_1929, getitem_1930, getitem_1931, getitem_1932, getitem_1933, getitem_1934, getitem_1935, getitem_1936, getitem_1937, getitem_1938, getitem_1939, getitem_1940, getitem_1941, getitem_1942, getitem_1943, getitem_1944, getitem_1945, getitem_1946, getitem_1947, getitem_1948, getitem_1949, getitem_1950, getitem_1951, getitem_1952, getitem_1953, getitem_1954, getitem_1955, getitem_1956, getitem_1957, getitem_1958, getitem_1959, getitem_1960, getitem_1961, getitem_1962, getitem_1963, getitem_1964, getitem_1965, getitem_1966, getitem_1967, getitem_1968, getitem_1969, getitem_1970, getitem_1971, getitem_1972, getitem_1973, getitem_1974, getitem_1975, getitem_1976, getitem_1977, getitem_1978, getitem_1979, getitem_1980, getitem_1981, getitem_1982, getitem_1983, getitem_1984, getitem_1985, getitem_1986, getitem_1987, getitem_1988, getitem_1989, getitem_1990, getitem_1991, getitem_1992, getitem_1993, getitem_1994, getitem_1995, getitem_1996, getitem_1997, getitem_1998, getitem_1999, getitem_2000, getitem_2001, getitem_2002, getitem_2003, getitem_2004, getitem_2005, getitem_2006, getitem_2007, getitem_2008, getitem_2009, getitem_2010, getitem_2011, getitem_2012, getitem_2013, getitem_2014, getitem_2015, getitem_2016, getitem_2017, getitem_2018, getitem_2019, getitem_2020, getitem_2021, getitem_2022, getitem_2023, getitem_2024, getitem_2025, getitem_2026, getitem_2027, getitem_2028, getitem_2029, getitem_2030, getitem_2031, getitem_2032, getitem_2033, getitem_2034, getitem_2035, getitem_2036, getitem_2037, getitem_2038, getitem_2039, getitem_2040, getitem_2041, getitem_2042, getitem_2043, getitem_2044, getitem_2045, getitem_2046, getitem_2047, getitem_2048, getitem_2049, getitem_2050, getitem_2051, getitem_2052, getitem_2053, getitem_2054, getitem_2055, getitem_2056, getitem_2057, getitem_2058, getitem_2059)


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
    torch.tensor(1),  # getitem_1029 (unknown shape)
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

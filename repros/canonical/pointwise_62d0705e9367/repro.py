"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s7_g77
Pattern hash: 62d0705e9367
Shape hash: 39a5f541
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg172_1: "f32[64, 3, 3, 3]", arg170_1: "f32[64]", arg506_1: "f32[64]", arg507_1: "f32[64, 3, 1, 1]", arg508_1: "f32[64]", arg509_1: "f32[64]", arg510_1: "f32[96, 64, 3, 3]", arg511_1: "f32[96]", arg512_1: "f32[96]", arg513_1: "f32[96, 64, 1, 1]", arg514_1: "f32[96]", arg515_1: "f32[96]", arg516_1: "f32[96]", arg517_1: "f32[96]", arg518_1: "f32[96, 96, 3, 3]", arg519_1: "f32[96]", arg520_1: "f32[96]", arg521_1: "f32[96, 96, 1, 1]", arg522_1: "f32[96]", arg523_1: "f32[96]", arg524_1: "f32[192, 96, 3, 3]", arg525_1: "f32[192]", arg526_1: "f32[192]", arg527_1: "f32[192, 96, 1, 1]", arg528_1: "f32[192]", arg529_1: "f32[192]", arg530_1: "f32[192]", arg531_1: "f32[192]", arg532_1: "f32[192, 192, 3, 3]", arg533_1: "f32[192]", arg534_1: "f32[192]", arg535_1: "f32[192, 192, 1, 1]", arg536_1: "f32[192]", arg537_1: "f32[192]", arg538_1: "f32[192]", arg539_1: "f32[192]", arg540_1: "f32[192, 192, 3, 3]", arg541_1: "f32[192]", arg542_1: "f32[192]", arg543_1: "f32[192, 192, 1, 1]", arg544_1: "f32[192]", arg545_1: "f32[192]", arg546_1: "f32[192]", arg547_1: "f32[192]", arg548_1: "f32[192, 192, 3, 3]", arg549_1: "f32[192]", arg550_1: "f32[192]", arg551_1: "f32[192, 192, 1, 1]", arg552_1: "f32[192]", arg553_1: "f32[192]", arg554_1: "f32[384, 192, 3, 3]", arg555_1: "f32[384]", arg556_1: "f32[384]", arg557_1: "f32[384, 192, 1, 1]", arg558_1: "f32[384]", arg559_1: "f32[384]", arg560_1: "f32[384]", arg561_1: "f32[384]", arg562_1: "f32[384, 384, 3, 3]", arg563_1: "f32[384]", arg564_1: "f32[384]", arg565_1: "f32[384, 384, 1, 1]", arg566_1: "f32[384]", arg567_1: "f32[384]", arg568_1: "f32[384]", arg569_1: "f32[384]", arg570_1: "f32[384, 384, 3, 3]", arg571_1: "f32[384]", arg572_1: "f32[384]", arg573_1: "f32[384, 384, 1, 1]", arg574_1: "f32[384]", arg575_1: "f32[384]", arg576_1: "f32[384]", arg577_1: "f32[384]", arg578_1: "f32[384, 384, 3, 3]", arg579_1: "f32[384]", arg580_1: "f32[384]", arg581_1: "f32[384, 384, 1, 1]", arg582_1: "f32[384]", arg583_1: "f32[384]", arg584_1: "f32[384]", arg585_1: "f32[384]", arg586_1: "f32[384, 384, 3, 3]", arg587_1: "f32[384]", arg588_1: "f32[384]", arg589_1: "f32[384, 384, 1, 1]", arg590_1: "f32[384]", arg591_1: "f32[384]", arg592_1: "f32[384]", arg593_1: "f32[384]", arg594_1: "f32[384, 384, 3, 3]", arg595_1: "f32[384]", arg596_1: "f32[384]", arg597_1: "f32[384, 384, 1, 1]", arg598_1: "f32[384]", arg599_1: "f32[384]", arg600_1: "f32[384]", arg601_1: "f32[384]", arg602_1: "f32[384, 384, 3, 3]", arg603_1: "f32[384]", arg604_1: "f32[384]", arg605_1: "f32[384, 384, 1, 1]", arg606_1: "f32[384]", arg607_1: "f32[384]", arg608_1: "f32[384]", arg609_1: "f32[384]", arg610_1: "f32[384, 384, 3, 3]", arg611_1: "f32[384]", arg612_1: "f32[384]", arg613_1: "f32[384, 384, 1, 1]", arg614_1: "f32[384]", arg615_1: "f32[384]", arg616_1: "f32[384]", arg617_1: "f32[384]", arg618_1: "f32[384, 384, 3, 3]", arg619_1: "f32[384]", arg620_1: "f32[384]", arg621_1: "f32[384, 384, 1, 1]", arg622_1: "f32[384]", arg623_1: "f32[384]", arg624_1: "f32[384]", arg625_1: "f32[384]", arg626_1: "f32[384, 384, 3, 3]", arg627_1: "f32[384]", arg628_1: "f32[384]", arg629_1: "f32[384, 384, 1, 1]", arg630_1: "f32[384]", arg631_1: "f32[384]", arg632_1: "f32[384]", arg633_1: "f32[384]", arg634_1: "f32[384, 384, 3, 3]", arg635_1: "f32[384]", arg636_1: "f32[384]", arg637_1: "f32[384, 384, 1, 1]", arg638_1: "f32[384]", arg639_1: "f32[384]", arg640_1: "f32[384]", arg641_1: "f32[384]", arg642_1: "f32[384, 384, 3, 3]", arg643_1: "f32[384]", arg644_1: "f32[384]", arg645_1: "f32[384, 384, 1, 1]", arg646_1: "f32[384]", arg647_1: "f32[384]", arg648_1: "f32[384]", arg649_1: "f32[384]", arg650_1: "f32[384, 384, 3, 3]", arg651_1: "f32[384]", arg652_1: "f32[384]", arg653_1: "f32[384, 384, 1, 1]", arg654_1: "f32[384]", arg655_1: "f32[384]", arg656_1: "f32[384]", arg657_1: "f32[384]", arg658_1: "f32[384, 384, 3, 3]", arg659_1: "f32[384]", arg660_1: "f32[384]", arg661_1: "f32[384, 384, 1, 1]", arg662_1: "f32[384]", arg663_1: "f32[384]", arg664_1: "f32[1408, 384, 3, 3]", arg665_1: "f32[1408]", arg666_1: "f32[1408]", arg667_1: "f32[1408, 384, 1, 1]", arg668_1: "f32[1408]", arg669_1: "f32[1408]", arg670_1: "f32[1000, 1408]", arg671_1: "f32[1000]", getitem_1008: "f32[]", getitem_1009: "f32[]", getitem_1010: "f32[]", getitem_1011: "f32[]", getitem_1012: "f32[]", getitem_1013: "f32[]", getitem_1014: "f32[]", getitem_1015: "f32[]", getitem_1016: "f32[]", getitem_1017: "f32[]", getitem_1018: "f32[]", getitem_1019: "f32[]", getitem_1020: "f32[]", getitem_1021: "f32[]", getitem_1022: "f32[]", getitem_1023: "f32[]", getitem_1024: "f32[]", getitem_1025: "f32[]", getitem_1026: "f32[]", getitem_1027: "f32[]", getitem_1028: "f32[]", getitem_1029: "f32[]", getitem_1030: "f32[]", getitem_1031: "f32[]", getitem_1032: "f32[]", getitem_1033: "f32[]", getitem_1034: "f32[]", getitem_1035: "f32[]", getitem_1036: "f32[]", getitem_1037: "f32[]", getitem_1038: "f32[]", getitem_1039: "f32[]", getitem_1040: "f32[]", getitem_1041: "f32[]", getitem_1042: "f32[]", getitem_1043: "f32[]", getitem_1044: "f32[]", getitem_1045: "f32[]", getitem_1046: "f32[]", getitem_1047: "f32[]", getitem_1048: "f32[]", getitem_1049: "f32[]", getitem_1050: "f32[]", getitem_1051: "f32[]", getitem_1052: "f32[]", getitem_1053: "f32[]", getitem_1054: "f32[]", getitem_1055: "f32[]", getitem_1056: "f32[]", getitem_1057: "f32[]", getitem_1058: "f32[]", getitem_1059: "f32[]", getitem_1060: "f32[]", getitem_1061: "f32[]", getitem_1062: "f32[]", getitem_1063: "f32[]", getitem_1064: "f32[]", getitem_1065: "f32[]", getitem_1066: "f32[]", getitem_1067: "f32[]", getitem_1068: "f32[]", getitem_1069: "f32[]", getitem_1070: "f32[]", getitem_1071: "f32[]", getitem_1072: "f32[]", getitem_1073: "f32[]", getitem_1074: "f32[]", getitem_1075: "f32[]", getitem_1076: "f32[]", getitem_1077: "f32[]", getitem_1078: "f32[]", getitem_1079: "f32[]", getitem_1080: "f32[]", getitem_1081: "f32[]", getitem_1082: "f32[]", getitem_1083: "f32[]", getitem_1084: "f32[]", getitem_1085: "f32[]", getitem_1086: "f32[]", getitem_1087: "f32[]", getitem_1088: "f32[]", getitem_1089: "f32[]", getitem_1090: "f32[]", getitem_1091: "f32[]", getitem_1092: "f32[]", getitem_1093: "f32[]", getitem_1094: "f32[]", getitem_1095: "f32[]", getitem_1096: "f32[]", getitem_1097: "f32[]", getitem_1098: "f32[]", getitem_1099: "f32[]", getitem_1100: "f32[]", getitem_1101: "f32[]", getitem_1102: "f32[]", getitem_1103: "f32[]", getitem_1104: "f32[]", getitem_1105: "f32[]", getitem_1106: "f32[]", getitem_1107: "f32[]", getitem_1108: "f32[]", getitem_1109: "f32[]", getitem_1110: "f32[]", getitem_1111: "f32[]", getitem_1112: "f32[]", getitem_1113: "f32[]", getitem_1114: "f32[]", getitem_1115: "f32[]", getitem_1116: "f32[]", getitem_1117: "f32[]", getitem_1118: "f32[]", getitem_1119: "f32[]", getitem_1120: "f32[]", getitem_1121: "f32[]", getitem_1122: "f32[]", getitem_1123: "f32[]", getitem_1124: "f32[]", getitem_1125: "f32[]", getitem_1126: "f32[]", getitem_1127: "f32[]", getitem_1128: "f32[]", getitem_1129: "f32[]", getitem_1130: "f32[]", getitem_1131: "f32[]", getitem_1132: "f32[]", getitem_1133: "f32[]", getitem_1134: "f32[]", getitem_1135: "f32[]", getitem_1136: "f32[]", getitem_1137: "f32[]", getitem_1138: "f32[]", getitem_1139: "f32[]", getitem_1140: "f32[]", getitem_1141: "f32[]", getitem_1142: "f32[]", getitem_1143: "f32[]", getitem_1144: "f32[]", getitem_1145: "f32[]", getitem_1146: "f32[]", getitem_1147: "f32[]", getitem_1148: "f32[]", getitem_1149: "f32[]", getitem_1150: "f32[]", getitem_1151: "f32[]", getitem_1152: "f32[]", getitem_1153: "f32[]", getitem_1154: "f32[]", getitem_1155: "f32[]", getitem_1156: "f32[]", getitem_1157: "f32[]", getitem_1158: "f32[]", getitem_1159: "f32[]", getitem_1160: "f32[]", getitem_1161: "f32[]", getitem_1162: "f32[]", getitem_1163: "f32[]", getitem_1164: "f32[]", getitem_1165: "f32[]", getitem_1166: "f32[]", getitem_1167: "f32[]", getitem_1168: "f32[]", getitem_1169: "f32[]", getitem_1170: "f32[]", getitem_1171: "f32[]", getitem_1172: "f32[]", getitem_1173: "f32[]", getitem_1174: "f32[]", getitem_1175: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg172_1, arg170_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1], 0.999);  arg172_1 = arg170_1 = arg506_1 = arg507_1 = arg508_1 = arg509_1 = arg510_1 = arg511_1 = arg512_1 = arg513_1 = arg514_1 = arg515_1 = arg516_1 = arg517_1 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = arg522_1 = arg523_1 = arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = arg540_1 = arg541_1 = arg542_1 = arg543_1 = arg544_1 = arg545_1 = arg546_1 = arg547_1 = arg548_1 = arg549_1 = arg550_1 = arg551_1 = arg552_1 = arg553_1 = arg554_1 = arg555_1 = arg556_1 = arg557_1 = arg558_1 = arg559_1 = arg560_1 = arg561_1 = arg562_1 = arg563_1 = arg564_1 = arg565_1 = arg566_1 = arg567_1 = arg568_1 = arg569_1 = arg570_1 = arg571_1 = arg572_1 = arg573_1 = arg574_1 = arg575_1 = arg576_1 = arg577_1 = arg578_1 = arg579_1 = arg580_1 = arg581_1 = arg582_1 = arg583_1 = arg584_1 = arg585_1 = arg586_1 = arg587_1 = arg588_1 = arg589_1 = arg590_1 = arg591_1 = arg592_1 = arg593_1 = arg594_1 = arg595_1 = arg596_1 = arg597_1 = arg598_1 = arg599_1 = arg600_1 = arg601_1 = arg602_1 = arg603_1 = arg604_1 = arg605_1 = arg606_1 = arg607_1 = arg608_1 = arg609_1 = arg610_1 = arg611_1 = arg612_1 = arg613_1 = arg614_1 = arg615_1 = arg616_1 = arg617_1 = arg618_1 = arg619_1 = arg620_1 = arg621_1 = arg622_1 = arg623_1 = arg624_1 = arg625_1 = arg626_1 = arg627_1 = arg628_1 = arg629_1 = arg630_1 = arg631_1 = arg632_1 = arg633_1 = arg634_1 = arg635_1 = arg636_1 = arg637_1 = arg638_1 = arg639_1 = arg640_1 = arg641_1 = arg642_1 = arg643_1 = arg644_1 = arg645_1 = arg646_1 = arg647_1 = arg648_1 = arg649_1 = arg650_1 = arg651_1 = arg652_1 = arg653_1 = arg654_1 = arg655_1 = arg656_1 = arg657_1 = arg658_1 = arg659_1 = arg660_1 = arg661_1 = arg662_1 = arg663_1 = arg664_1 = arg665_1 = arg666_1 = arg667_1 = arg668_1 = arg669_1 = arg670_1 = arg671_1 = None
        getitem: "f32[64, 3, 3, 3]" = _foreach_mul_scalar[0]
        getitem_1: "f32[64]" = _foreach_mul_scalar[1]
        getitem_2: "f32[64]" = _foreach_mul_scalar[2]
        getitem_3: "f32[64, 3, 1, 1]" = _foreach_mul_scalar[3]
        getitem_4: "f32[64]" = _foreach_mul_scalar[4]
        getitem_5: "f32[64]" = _foreach_mul_scalar[5]
        getitem_6: "f32[96, 64, 3, 3]" = _foreach_mul_scalar[6]
        getitem_7: "f32[96]" = _foreach_mul_scalar[7]
        getitem_8: "f32[96]" = _foreach_mul_scalar[8]
        getitem_9: "f32[96, 64, 1, 1]" = _foreach_mul_scalar[9]
        getitem_10: "f32[96]" = _foreach_mul_scalar[10]
        getitem_11: "f32[96]" = _foreach_mul_scalar[11]
        getitem_12: "f32[96]" = _foreach_mul_scalar[12]
        getitem_13: "f32[96]" = _foreach_mul_scalar[13]
        getitem_14: "f32[96, 96, 3, 3]" = _foreach_mul_scalar[14]
        getitem_15: "f32[96]" = _foreach_mul_scalar[15]
        getitem_16: "f32[96]" = _foreach_mul_scalar[16]
        getitem_17: "f32[96, 96, 1, 1]" = _foreach_mul_scalar[17]
        getitem_18: "f32[96]" = _foreach_mul_scalar[18]
        getitem_19: "f32[96]" = _foreach_mul_scalar[19]
        getitem_20: "f32[192, 96, 3, 3]" = _foreach_mul_scalar[20]
        getitem_21: "f32[192]" = _foreach_mul_scalar[21]
        getitem_22: "f32[192]" = _foreach_mul_scalar[22]
        getitem_23: "f32[192, 96, 1, 1]" = _foreach_mul_scalar[23]
        getitem_24: "f32[192]" = _foreach_mul_scalar[24]
        getitem_25: "f32[192]" = _foreach_mul_scalar[25]
        getitem_26: "f32[192]" = _foreach_mul_scalar[26]
        getitem_27: "f32[192]" = _foreach_mul_scalar[27]
        getitem_28: "f32[192, 192, 3, 3]" = _foreach_mul_scalar[28]
        getitem_29: "f32[192]" = _foreach_mul_scalar[29]
        getitem_30: "f32[192]" = _foreach_mul_scalar[30]
        getitem_31: "f32[192, 192, 1, 1]" = _foreach_mul_scalar[31]
        getitem_32: "f32[192]" = _foreach_mul_scalar[32]
        getitem_33: "f32[192]" = _foreach_mul_scalar[33]
        getitem_34: "f32[192]" = _foreach_mul_scalar[34]
        getitem_35: "f32[192]" = _foreach_mul_scalar[35]
        getitem_36: "f32[192, 192, 3, 3]" = _foreach_mul_scalar[36]
        getitem_37: "f32[192]" = _foreach_mul_scalar[37]
        getitem_38: "f32[192]" = _foreach_mul_scalar[38]
        getitem_39: "f32[192, 192, 1, 1]" = _foreach_mul_scalar[39]
        getitem_40: "f32[192]" = _foreach_mul_scalar[40]
        getitem_41: "f32[192]" = _foreach_mul_scalar[41]
        getitem_42: "f32[192]" = _foreach_mul_scalar[42]
        getitem_43: "f32[192]" = _foreach_mul_scalar[43]
        getitem_44: "f32[192, 192, 3, 3]" = _foreach_mul_scalar[44]
        getitem_45: "f32[192]" = _foreach_mul_scalar[45]
        getitem_46: "f32[192]" = _foreach_mul_scalar[46]
        getitem_47: "f32[192, 192, 1, 1]" = _foreach_mul_scalar[47]
        getitem_48: "f32[192]" = _foreach_mul_scalar[48]
        getitem_49: "f32[192]" = _foreach_mul_scalar[49]
        getitem_50: "f32[384, 192, 3, 3]" = _foreach_mul_scalar[50]
        getitem_51: "f32[384]" = _foreach_mul_scalar[51]
        getitem_52: "f32[384]" = _foreach_mul_scalar[52]
        getitem_53: "f32[384, 192, 1, 1]" = _foreach_mul_scalar[53]
        getitem_54: "f32[384]" = _foreach_mul_scalar[54]
        getitem_55: "f32[384]" = _foreach_mul_scalar[55]
        getitem_56: "f32[384]" = _foreach_mul_scalar[56]
        getitem_57: "f32[384]" = _foreach_mul_scalar[57]
        getitem_58: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[58]
        getitem_59: "f32[384]" = _foreach_mul_scalar[59]
        getitem_60: "f32[384]" = _foreach_mul_scalar[60]
        getitem_61: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[61]
        getitem_62: "f32[384]" = _foreach_mul_scalar[62]
        getitem_63: "f32[384]" = _foreach_mul_scalar[63]
        getitem_64: "f32[384]" = _foreach_mul_scalar[64]
        getitem_65: "f32[384]" = _foreach_mul_scalar[65]
        getitem_66: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[66]
        getitem_67: "f32[384]" = _foreach_mul_scalar[67]
        getitem_68: "f32[384]" = _foreach_mul_scalar[68]
        getitem_69: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[69]
        getitem_70: "f32[384]" = _foreach_mul_scalar[70]
        getitem_71: "f32[384]" = _foreach_mul_scalar[71]
        getitem_72: "f32[384]" = _foreach_mul_scalar[72]
        getitem_73: "f32[384]" = _foreach_mul_scalar[73]
        getitem_74: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[74]
        getitem_75: "f32[384]" = _foreach_mul_scalar[75]
        getitem_76: "f32[384]" = _foreach_mul_scalar[76]
        getitem_77: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[77]
        getitem_78: "f32[384]" = _foreach_mul_scalar[78]
        getitem_79: "f32[384]" = _foreach_mul_scalar[79]
        getitem_80: "f32[384]" = _foreach_mul_scalar[80]
        getitem_81: "f32[384]" = _foreach_mul_scalar[81]
        getitem_82: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[82]
        getitem_83: "f32[384]" = _foreach_mul_scalar[83]
        getitem_84: "f32[384]" = _foreach_mul_scalar[84]
        getitem_85: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[85]
        getitem_86: "f32[384]" = _foreach_mul_scalar[86]
        getitem_87: "f32[384]" = _foreach_mul_scalar[87]
        getitem_88: "f32[384]" = _foreach_mul_scalar[88]
        getitem_89: "f32[384]" = _foreach_mul_scalar[89]
        getitem_90: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[90]
        getitem_91: "f32[384]" = _foreach_mul_scalar[91]
        getitem_92: "f32[384]" = _foreach_mul_scalar[92]
        getitem_93: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[93]
        getitem_94: "f32[384]" = _foreach_mul_scalar[94]
        getitem_95: "f32[384]" = _foreach_mul_scalar[95]
        getitem_96: "f32[384]" = _foreach_mul_scalar[96]
        getitem_97: "f32[384]" = _foreach_mul_scalar[97]
        getitem_98: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[98]
        getitem_99: "f32[384]" = _foreach_mul_scalar[99]
        getitem_100: "f32[384]" = _foreach_mul_scalar[100]
        getitem_101: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[101]
        getitem_102: "f32[384]" = _foreach_mul_scalar[102]
        getitem_103: "f32[384]" = _foreach_mul_scalar[103]
        getitem_104: "f32[384]" = _foreach_mul_scalar[104]
        getitem_105: "f32[384]" = _foreach_mul_scalar[105]
        getitem_106: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[106]
        getitem_107: "f32[384]" = _foreach_mul_scalar[107]
        getitem_108: "f32[384]" = _foreach_mul_scalar[108]
        getitem_109: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[109]
        getitem_110: "f32[384]" = _foreach_mul_scalar[110]
        getitem_111: "f32[384]" = _foreach_mul_scalar[111]
        getitem_112: "f32[384]" = _foreach_mul_scalar[112]
        getitem_113: "f32[384]" = _foreach_mul_scalar[113]
        getitem_114: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[114]
        getitem_115: "f32[384]" = _foreach_mul_scalar[115]
        getitem_116: "f32[384]" = _foreach_mul_scalar[116]
        getitem_117: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[117]
        getitem_118: "f32[384]" = _foreach_mul_scalar[118]
        getitem_119: "f32[384]" = _foreach_mul_scalar[119]
        getitem_120: "f32[384]" = _foreach_mul_scalar[120]
        getitem_121: "f32[384]" = _foreach_mul_scalar[121]
        getitem_122: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[122]
        getitem_123: "f32[384]" = _foreach_mul_scalar[123]
        getitem_124: "f32[384]" = _foreach_mul_scalar[124]
        getitem_125: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[125]
        getitem_126: "f32[384]" = _foreach_mul_scalar[126]
        getitem_127: "f32[384]" = _foreach_mul_scalar[127]
        getitem_128: "f32[384]" = _foreach_mul_scalar[128]
        getitem_129: "f32[384]" = _foreach_mul_scalar[129]
        getitem_130: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[130]
        getitem_131: "f32[384]" = _foreach_mul_scalar[131]
        getitem_132: "f32[384]" = _foreach_mul_scalar[132]
        getitem_133: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[133]
        getitem_134: "f32[384]" = _foreach_mul_scalar[134]
        getitem_135: "f32[384]" = _foreach_mul_scalar[135]
        getitem_136: "f32[384]" = _foreach_mul_scalar[136]
        getitem_137: "f32[384]" = _foreach_mul_scalar[137]
        getitem_138: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[138]
        getitem_139: "f32[384]" = _foreach_mul_scalar[139]
        getitem_140: "f32[384]" = _foreach_mul_scalar[140]
        getitem_141: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[141]
        getitem_142: "f32[384]" = _foreach_mul_scalar[142]
        getitem_143: "f32[384]" = _foreach_mul_scalar[143]
        getitem_144: "f32[384]" = _foreach_mul_scalar[144]
        getitem_145: "f32[384]" = _foreach_mul_scalar[145]
        getitem_146: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[146]
        getitem_147: "f32[384]" = _foreach_mul_scalar[147]
        getitem_148: "f32[384]" = _foreach_mul_scalar[148]
        getitem_149: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[149]
        getitem_150: "f32[384]" = _foreach_mul_scalar[150]
        getitem_151: "f32[384]" = _foreach_mul_scalar[151]
        getitem_152: "f32[384]" = _foreach_mul_scalar[152]
        getitem_153: "f32[384]" = _foreach_mul_scalar[153]
        getitem_154: "f32[384, 384, 3, 3]" = _foreach_mul_scalar[154]
        getitem_155: "f32[384]" = _foreach_mul_scalar[155]
        getitem_156: "f32[384]" = _foreach_mul_scalar[156]
        getitem_157: "f32[384, 384, 1, 1]" = _foreach_mul_scalar[157]
        getitem_158: "f32[384]" = _foreach_mul_scalar[158]
        getitem_159: "f32[384]" = _foreach_mul_scalar[159]
        getitem_160: "f32[1408, 384, 3, 3]" = _foreach_mul_scalar[160]
        getitem_161: "f32[1408]" = _foreach_mul_scalar[161]
        getitem_162: "f32[1408]" = _foreach_mul_scalar[162]
        getitem_163: "f32[1408, 384, 1, 1]" = _foreach_mul_scalar[163]
        getitem_164: "f32[1408]" = _foreach_mul_scalar[164]
        getitem_165: "f32[1408]" = _foreach_mul_scalar[165]
        getitem_166: "f32[1000, 1408]" = _foreach_mul_scalar[166]
        getitem_167: "f32[1000]" = _foreach_mul_scalar[167];  _foreach_mul_scalar = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047, getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079, getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175], 1);  getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = getitem_1013 = getitem_1014 = getitem_1015 = getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = getitem_1023 = getitem_1024 = getitem_1025 = getitem_1026 = getitem_1027 = getitem_1028 = getitem_1029 = getitem_1030 = getitem_1031 = getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = getitem_1048 = getitem_1049 = getitem_1050 = getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = getitem_1056 = getitem_1057 = getitem_1058 = getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = getitem_1064 = getitem_1065 = getitem_1066 = getitem_1067 = getitem_1068 = getitem_1069 = getitem_1070 = getitem_1071 = getitem_1072 = getitem_1073 = getitem_1074 = getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = getitem_1080 = getitem_1081 = getitem_1082 = getitem_1083 = getitem_1084 = getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = getitem_1153 = getitem_1154 = getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = None
        getitem_1176: "f32[]" = _foreach_sub_scalar[0]
        getitem_1177: "f32[]" = _foreach_sub_scalar[1]
        getitem_1178: "f32[]" = _foreach_sub_scalar[2]
        getitem_1179: "f32[]" = _foreach_sub_scalar[3]
        getitem_1180: "f32[]" = _foreach_sub_scalar[4]
        getitem_1181: "f32[]" = _foreach_sub_scalar[5]
        getitem_1182: "f32[]" = _foreach_sub_scalar[6]
        getitem_1183: "f32[]" = _foreach_sub_scalar[7]
        getitem_1184: "f32[]" = _foreach_sub_scalar[8]
        getitem_1185: "f32[]" = _foreach_sub_scalar[9]
        getitem_1186: "f32[]" = _foreach_sub_scalar[10]
        getitem_1187: "f32[]" = _foreach_sub_scalar[11]
        getitem_1188: "f32[]" = _foreach_sub_scalar[12]
        getitem_1189: "f32[]" = _foreach_sub_scalar[13]
        getitem_1190: "f32[]" = _foreach_sub_scalar[14]
        getitem_1191: "f32[]" = _foreach_sub_scalar[15]
        getitem_1192: "f32[]" = _foreach_sub_scalar[16]
        getitem_1193: "f32[]" = _foreach_sub_scalar[17]
        getitem_1194: "f32[]" = _foreach_sub_scalar[18]
        getitem_1195: "f32[]" = _foreach_sub_scalar[19]
        getitem_1196: "f32[]" = _foreach_sub_scalar[20]
        getitem_1197: "f32[]" = _foreach_sub_scalar[21]
        getitem_1198: "f32[]" = _foreach_sub_scalar[22]
        getitem_1199: "f32[]" = _foreach_sub_scalar[23]
        getitem_1200: "f32[]" = _foreach_sub_scalar[24]
        getitem_1201: "f32[]" = _foreach_sub_scalar[25]
        getitem_1202: "f32[]" = _foreach_sub_scalar[26]
        getitem_1203: "f32[]" = _foreach_sub_scalar[27]
        getitem_1204: "f32[]" = _foreach_sub_scalar[28]
        getitem_1205: "f32[]" = _foreach_sub_scalar[29]
        getitem_1206: "f32[]" = _foreach_sub_scalar[30]
        getitem_1207: "f32[]" = _foreach_sub_scalar[31]
        getitem_1208: "f32[]" = _foreach_sub_scalar[32]
        getitem_1209: "f32[]" = _foreach_sub_scalar[33]
        getitem_1210: "f32[]" = _foreach_sub_scalar[34]
        getitem_1211: "f32[]" = _foreach_sub_scalar[35]
        getitem_1212: "f32[]" = _foreach_sub_scalar[36]
        getitem_1213: "f32[]" = _foreach_sub_scalar[37]
        getitem_1214: "f32[]" = _foreach_sub_scalar[38]
        getitem_1215: "f32[]" = _foreach_sub_scalar[39]
        getitem_1216: "f32[]" = _foreach_sub_scalar[40]
        getitem_1217: "f32[]" = _foreach_sub_scalar[41]
        getitem_1218: "f32[]" = _foreach_sub_scalar[42]
        getitem_1219: "f32[]" = _foreach_sub_scalar[43]
        getitem_1220: "f32[]" = _foreach_sub_scalar[44]
        getitem_1221: "f32[]" = _foreach_sub_scalar[45]
        getitem_1222: "f32[]" = _foreach_sub_scalar[46]
        getitem_1223: "f32[]" = _foreach_sub_scalar[47]
        getitem_1224: "f32[]" = _foreach_sub_scalar[48]
        getitem_1225: "f32[]" = _foreach_sub_scalar[49]
        getitem_1226: "f32[]" = _foreach_sub_scalar[50]
        getitem_1227: "f32[]" = _foreach_sub_scalar[51]
        getitem_1228: "f32[]" = _foreach_sub_scalar[52]
        getitem_1229: "f32[]" = _foreach_sub_scalar[53]
        getitem_1230: "f32[]" = _foreach_sub_scalar[54]
        getitem_1231: "f32[]" = _foreach_sub_scalar[55]
        getitem_1232: "f32[]" = _foreach_sub_scalar[56]
        getitem_1233: "f32[]" = _foreach_sub_scalar[57]
        getitem_1234: "f32[]" = _foreach_sub_scalar[58]
        getitem_1235: "f32[]" = _foreach_sub_scalar[59]
        getitem_1236: "f32[]" = _foreach_sub_scalar[60]
        getitem_1237: "f32[]" = _foreach_sub_scalar[61]
        getitem_1238: "f32[]" = _foreach_sub_scalar[62]
        getitem_1239: "f32[]" = _foreach_sub_scalar[63]
        getitem_1240: "f32[]" = _foreach_sub_scalar[64]
        getitem_1241: "f32[]" = _foreach_sub_scalar[65]
        getitem_1242: "f32[]" = _foreach_sub_scalar[66]
        getitem_1243: "f32[]" = _foreach_sub_scalar[67]
        getitem_1244: "f32[]" = _foreach_sub_scalar[68]
        getitem_1245: "f32[]" = _foreach_sub_scalar[69]
        getitem_1246: "f32[]" = _foreach_sub_scalar[70]
        getitem_1247: "f32[]" = _foreach_sub_scalar[71]
        getitem_1248: "f32[]" = _foreach_sub_scalar[72]
        getitem_1249: "f32[]" = _foreach_sub_scalar[73]
        getitem_1250: "f32[]" = _foreach_sub_scalar[74]
        getitem_1251: "f32[]" = _foreach_sub_scalar[75]
        getitem_1252: "f32[]" = _foreach_sub_scalar[76]
        getitem_1253: "f32[]" = _foreach_sub_scalar[77]
        getitem_1254: "f32[]" = _foreach_sub_scalar[78]
        getitem_1255: "f32[]" = _foreach_sub_scalar[79]
        getitem_1256: "f32[]" = _foreach_sub_scalar[80]
        getitem_1257: "f32[]" = _foreach_sub_scalar[81]
        getitem_1258: "f32[]" = _foreach_sub_scalar[82]
        getitem_1259: "f32[]" = _foreach_sub_scalar[83]
        getitem_1260: "f32[]" = _foreach_sub_scalar[84]
        getitem_1261: "f32[]" = _foreach_sub_scalar[85]
        getitem_1262: "f32[]" = _foreach_sub_scalar[86]
        getitem_1263: "f32[]" = _foreach_sub_scalar[87]
        getitem_1264: "f32[]" = _foreach_sub_scalar[88]
        getitem_1265: "f32[]" = _foreach_sub_scalar[89]
        getitem_1266: "f32[]" = _foreach_sub_scalar[90]
        getitem_1267: "f32[]" = _foreach_sub_scalar[91]
        getitem_1268: "f32[]" = _foreach_sub_scalar[92]
        getitem_1269: "f32[]" = _foreach_sub_scalar[93]
        getitem_1270: "f32[]" = _foreach_sub_scalar[94]
        getitem_1271: "f32[]" = _foreach_sub_scalar[95]
        getitem_1272: "f32[]" = _foreach_sub_scalar[96]
        getitem_1273: "f32[]" = _foreach_sub_scalar[97]
        getitem_1274: "f32[]" = _foreach_sub_scalar[98]
        getitem_1275: "f32[]" = _foreach_sub_scalar[99]
        getitem_1276: "f32[]" = _foreach_sub_scalar[100]
        getitem_1277: "f32[]" = _foreach_sub_scalar[101]
        getitem_1278: "f32[]" = _foreach_sub_scalar[102]
        getitem_1279: "f32[]" = _foreach_sub_scalar[103]
        getitem_1280: "f32[]" = _foreach_sub_scalar[104]
        getitem_1281: "f32[]" = _foreach_sub_scalar[105]
        getitem_1282: "f32[]" = _foreach_sub_scalar[106]
        getitem_1283: "f32[]" = _foreach_sub_scalar[107]
        getitem_1284: "f32[]" = _foreach_sub_scalar[108]
        getitem_1285: "f32[]" = _foreach_sub_scalar[109]
        getitem_1286: "f32[]" = _foreach_sub_scalar[110]
        getitem_1287: "f32[]" = _foreach_sub_scalar[111]
        getitem_1288: "f32[]" = _foreach_sub_scalar[112]
        getitem_1289: "f32[]" = _foreach_sub_scalar[113]
        getitem_1290: "f32[]" = _foreach_sub_scalar[114]
        getitem_1291: "f32[]" = _foreach_sub_scalar[115]
        getitem_1292: "f32[]" = _foreach_sub_scalar[116]
        getitem_1293: "f32[]" = _foreach_sub_scalar[117]
        getitem_1294: "f32[]" = _foreach_sub_scalar[118]
        getitem_1295: "f32[]" = _foreach_sub_scalar[119]
        getitem_1296: "f32[]" = _foreach_sub_scalar[120]
        getitem_1297: "f32[]" = _foreach_sub_scalar[121]
        getitem_1298: "f32[]" = _foreach_sub_scalar[122]
        getitem_1299: "f32[]" = _foreach_sub_scalar[123]
        getitem_1300: "f32[]" = _foreach_sub_scalar[124]
        getitem_1301: "f32[]" = _foreach_sub_scalar[125]
        getitem_1302: "f32[]" = _foreach_sub_scalar[126]
        getitem_1303: "f32[]" = _foreach_sub_scalar[127]
        getitem_1304: "f32[]" = _foreach_sub_scalar[128]
        getitem_1305: "f32[]" = _foreach_sub_scalar[129]
        getitem_1306: "f32[]" = _foreach_sub_scalar[130]
        getitem_1307: "f32[]" = _foreach_sub_scalar[131]
        getitem_1308: "f32[]" = _foreach_sub_scalar[132]
        getitem_1309: "f32[]" = _foreach_sub_scalar[133]
        getitem_1310: "f32[]" = _foreach_sub_scalar[134]
        getitem_1311: "f32[]" = _foreach_sub_scalar[135]
        getitem_1312: "f32[]" = _foreach_sub_scalar[136]
        getitem_1313: "f32[]" = _foreach_sub_scalar[137]
        getitem_1314: "f32[]" = _foreach_sub_scalar[138]
        getitem_1315: "f32[]" = _foreach_sub_scalar[139]
        getitem_1316: "f32[]" = _foreach_sub_scalar[140]
        getitem_1317: "f32[]" = _foreach_sub_scalar[141]
        getitem_1318: "f32[]" = _foreach_sub_scalar[142]
        getitem_1319: "f32[]" = _foreach_sub_scalar[143]
        getitem_1320: "f32[]" = _foreach_sub_scalar[144]
        getitem_1321: "f32[]" = _foreach_sub_scalar[145]
        getitem_1322: "f32[]" = _foreach_sub_scalar[146]
        getitem_1323: "f32[]" = _foreach_sub_scalar[147]
        getitem_1324: "f32[]" = _foreach_sub_scalar[148]
        getitem_1325: "f32[]" = _foreach_sub_scalar[149]
        getitem_1326: "f32[]" = _foreach_sub_scalar[150]
        getitem_1327: "f32[]" = _foreach_sub_scalar[151]
        getitem_1328: "f32[]" = _foreach_sub_scalar[152]
        getitem_1329: "f32[]" = _foreach_sub_scalar[153]
        getitem_1330: "f32[]" = _foreach_sub_scalar[154]
        getitem_1331: "f32[]" = _foreach_sub_scalar[155]
        getitem_1332: "f32[]" = _foreach_sub_scalar[156]
        getitem_1333: "f32[]" = _foreach_sub_scalar[157]
        getitem_1334: "f32[]" = _foreach_sub_scalar[158]
        getitem_1335: "f32[]" = _foreach_sub_scalar[159]
        getitem_1336: "f32[]" = _foreach_sub_scalar[160]
        getitem_1337: "f32[]" = _foreach_sub_scalar[161]
        getitem_1338: "f32[]" = _foreach_sub_scalar[162]
        getitem_1339: "f32[]" = _foreach_sub_scalar[163]
        getitem_1340: "f32[]" = _foreach_sub_scalar[164]
        getitem_1341: "f32[]" = _foreach_sub_scalar[165]
        getitem_1342: "f32[]" = _foreach_sub_scalar[166]
        getitem_1343: "f32[]" = _foreach_sub_scalar[167];  _foreach_sub_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_1176, getitem_1177, getitem_1178, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187, getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295, getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

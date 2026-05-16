"""
Standalone repro captured via capture_hook.
Label: tlparse_timm_s3_g70
Pattern hash: 94ba6449f80d
Shape hash: ab3c8ab4
"""
import sys
from pathlib import Path

import torch
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg157_1: "f32[]", arg152_1: "f32[]", arg158_1: "f32[]", arg159_1: "f32[]", arg160_1: "f32[]", arg161_1: "f32[]", arg162_1: "f32[]", arg163_1: "f32[]", arg164_1: "f32[]", arg165_1: "f32[]", arg166_1: "f32[]", arg167_1: "f32[]", arg168_1: "f32[]", arg169_1: "f32[]", arg170_1: "f32[]", arg171_1: "f32[]", arg172_1: "f32[]", arg173_1: "f32[]", arg174_1: "f32[]", arg175_1: "f32[]", arg176_1: "f32[]", arg177_1: "f32[]", arg178_1: "f32[]", arg179_1: "f32[]", arg180_1: "f32[]", arg181_1: "f32[]", arg182_1: "f32[]", arg183_1: "f32[]", arg184_1: "f32[]", arg185_1: "f32[]", arg186_1: "f32[]", arg187_1: "f32[]", arg188_1: "f32[]", arg189_1: "f32[]", arg190_1: "f32[]", arg191_1: "f32[]", arg192_1: "f32[]", arg193_1: "f32[]", arg194_1: "f32[]", arg195_1: "f32[]", arg196_1: "f32[]", arg197_1: "f32[]", arg198_1: "f32[]", arg199_1: "f32[]", arg200_1: "f32[]", arg201_1: "f32[]", arg202_1: "f32[]", arg203_1: "f32[]", arg204_1: "f32[]", arg205_1: "f32[]", arg206_1: "f32[]", arg207_1: "f32[]", arg208_1: "f32[]", arg209_1: "f32[]", arg210_1: "f32[]", arg211_1: "f32[]", arg212_1: "f32[]", arg213_1: "f32[]", arg214_1: "f32[]", arg215_1: "f32[]", arg216_1: "f32[]", arg217_1: "f32[]", arg218_1: "f32[]", arg219_1: "f32[]", arg220_1: "f32[]", arg221_1: "f32[]", arg222_1: "f32[]", arg223_1: "f32[]", arg224_1: "f32[]", arg225_1: "f32[]", arg226_1: "f32[]", arg227_1: "f32[]", arg228_1: "f32[]", arg229_1: "f32[]", arg230_1: "f32[]", arg231_1: "f32[]", arg232_1: "f32[]", arg233_1: "f32[]", arg234_1: "f32[]", arg235_1: "f32[]", arg236_1: "f32[]", arg237_1: "f32[]", arg238_1: "f32[]", arg239_1: "f32[]", arg240_1: "f32[]", arg241_1: "f32[]", arg242_1: "f32[]", arg243_1: "f32[]", arg244_1: "f32[]", arg245_1: "f32[]", arg246_1: "f32[]", arg247_1: "f32[]", arg248_1: "f32[]", arg249_1: "f32[]", arg250_1: "f32[]", arg251_1: "f32[]", arg252_1: "f32[]", arg253_1: "f32[]", arg254_1: "f32[]", arg255_1: "f32[]", arg256_1: "f32[]", arg257_1: "f32[]", arg258_1: "f32[]", arg259_1: "f32[]", arg260_1: "f32[]", arg261_1: "f32[]", arg262_1: "f32[]", arg263_1: "f32[]", arg264_1: "f32[]", arg265_1: "f32[]", arg266_1: "f32[]", arg267_1: "f32[]", arg268_1: "f32[]", arg269_1: "f32[]", arg270_1: "f32[]", arg271_1: "f32[]", arg272_1: "f32[]", arg273_1: "f32[]", arg274_1: "f32[]", arg275_1: "f32[]", arg276_1: "f32[]", arg277_1: "f32[]", arg278_1: "f32[]", arg279_1: "f32[]", arg280_1: "f32[]", arg281_1: "f32[]", arg282_1: "f32[]", arg283_1: "f32[]", arg284_1: "f32[]", arg285_1: "f32[]", arg286_1: "f32[]", arg287_1: "f32[]", arg288_1: "f32[]", arg289_1: "f32[]", arg290_1: "f32[]", arg291_1: "f32[]", arg292_1: "f32[]", arg293_1: "f32[]", arg294_1: "f32[]", arg295_1: "f32[]", arg296_1: "f32[]", arg297_1: "f32[]", arg298_1: "f32[]", arg299_1: "f32[]", arg300_1: "f32[]", arg301_1: "f32[]", arg302_1: "f32[]", arg303_1: "f32[]", arg304_1: "f32[]", arg305_1: "f32[]", arg306_1: "f32[]", arg307_1: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg157_1, arg152_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1], 1);  arg157_1 = arg152_1 = arg158_1 = arg159_1 = arg160_1 = arg161_1 = arg162_1 = arg163_1 = arg164_1 = arg165_1 = arg166_1 = arg167_1 = arg168_1 = arg169_1 = arg170_1 = arg171_1 = arg172_1 = arg173_1 = arg174_1 = arg175_1 = arg176_1 = arg177_1 = arg178_1 = arg179_1 = arg180_1 = arg181_1 = arg182_1 = arg183_1 = arg184_1 = arg185_1 = arg186_1 = arg187_1 = arg188_1 = arg189_1 = arg190_1 = arg191_1 = arg192_1 = arg193_1 = arg194_1 = arg195_1 = arg196_1 = arg197_1 = arg198_1 = arg199_1 = arg200_1 = arg201_1 = arg202_1 = arg203_1 = arg204_1 = arg205_1 = arg206_1 = arg207_1 = arg208_1 = arg209_1 = arg210_1 = arg211_1 = arg212_1 = arg213_1 = arg214_1 = arg215_1 = arg216_1 = arg217_1 = arg218_1 = arg219_1 = arg220_1 = arg221_1 = arg222_1 = arg223_1 = arg224_1 = arg225_1 = arg226_1 = arg227_1 = arg228_1 = arg229_1 = arg230_1 = arg231_1 = arg232_1 = arg233_1 = arg234_1 = arg235_1 = arg236_1 = arg237_1 = arg238_1 = arg239_1 = arg240_1 = arg241_1 = arg242_1 = arg243_1 = arg244_1 = arg245_1 = arg246_1 = arg247_1 = arg248_1 = arg249_1 = arg250_1 = arg251_1 = arg252_1 = arg253_1 = arg254_1 = arg255_1 = arg256_1 = arg257_1 = arg258_1 = arg259_1 = arg260_1 = arg261_1 = arg262_1 = arg263_1 = arg264_1 = arg265_1 = arg266_1 = arg267_1 = arg268_1 = arg269_1 = arg270_1 = arg271_1 = arg272_1 = arg273_1 = arg274_1 = arg275_1 = arg276_1 = arg277_1 = arg278_1 = arg279_1 = arg280_1 = arg281_1 = arg282_1 = arg283_1 = arg284_1 = arg285_1 = arg286_1 = arg287_1 = arg288_1 = arg289_1 = arg290_1 = arg291_1 = arg292_1 = arg293_1 = arg294_1 = arg295_1 = arg296_1 = arg297_1 = arg298_1 = arg299_1 = arg300_1 = arg301_1 = arg302_1 = arg303_1 = arg304_1 = arg305_1 = arg306_1 = arg307_1 = None
        getitem: "f32[]" = _foreach_add_scalar[0]
        getitem_1: "f32[]" = _foreach_add_scalar[1]
        getitem_2: "f32[]" = _foreach_add_scalar[2]
        getitem_3: "f32[]" = _foreach_add_scalar[3]
        getitem_4: "f32[]" = _foreach_add_scalar[4]
        getitem_5: "f32[]" = _foreach_add_scalar[5]
        getitem_6: "f32[]" = _foreach_add_scalar[6]
        getitem_7: "f32[]" = _foreach_add_scalar[7]
        getitem_8: "f32[]" = _foreach_add_scalar[8]
        getitem_9: "f32[]" = _foreach_add_scalar[9]
        getitem_10: "f32[]" = _foreach_add_scalar[10]
        getitem_11: "f32[]" = _foreach_add_scalar[11]
        getitem_12: "f32[]" = _foreach_add_scalar[12]
        getitem_13: "f32[]" = _foreach_add_scalar[13]
        getitem_14: "f32[]" = _foreach_add_scalar[14]
        getitem_15: "f32[]" = _foreach_add_scalar[15]
        getitem_16: "f32[]" = _foreach_add_scalar[16]
        getitem_17: "f32[]" = _foreach_add_scalar[17]
        getitem_18: "f32[]" = _foreach_add_scalar[18]
        getitem_19: "f32[]" = _foreach_add_scalar[19]
        getitem_20: "f32[]" = _foreach_add_scalar[20]
        getitem_21: "f32[]" = _foreach_add_scalar[21]
        getitem_22: "f32[]" = _foreach_add_scalar[22]
        getitem_23: "f32[]" = _foreach_add_scalar[23]
        getitem_24: "f32[]" = _foreach_add_scalar[24]
        getitem_25: "f32[]" = _foreach_add_scalar[25]
        getitem_26: "f32[]" = _foreach_add_scalar[26]
        getitem_27: "f32[]" = _foreach_add_scalar[27]
        getitem_28: "f32[]" = _foreach_add_scalar[28]
        getitem_29: "f32[]" = _foreach_add_scalar[29]
        getitem_30: "f32[]" = _foreach_add_scalar[30]
        getitem_31: "f32[]" = _foreach_add_scalar[31]
        getitem_32: "f32[]" = _foreach_add_scalar[32]
        getitem_33: "f32[]" = _foreach_add_scalar[33]
        getitem_34: "f32[]" = _foreach_add_scalar[34]
        getitem_35: "f32[]" = _foreach_add_scalar[35]
        getitem_36: "f32[]" = _foreach_add_scalar[36]
        getitem_37: "f32[]" = _foreach_add_scalar[37]
        getitem_38: "f32[]" = _foreach_add_scalar[38]
        getitem_39: "f32[]" = _foreach_add_scalar[39]
        getitem_40: "f32[]" = _foreach_add_scalar[40]
        getitem_41: "f32[]" = _foreach_add_scalar[41]
        getitem_42: "f32[]" = _foreach_add_scalar[42]
        getitem_43: "f32[]" = _foreach_add_scalar[43]
        getitem_44: "f32[]" = _foreach_add_scalar[44]
        getitem_45: "f32[]" = _foreach_add_scalar[45]
        getitem_46: "f32[]" = _foreach_add_scalar[46]
        getitem_47: "f32[]" = _foreach_add_scalar[47]
        getitem_48: "f32[]" = _foreach_add_scalar[48]
        getitem_49: "f32[]" = _foreach_add_scalar[49]
        getitem_50: "f32[]" = _foreach_add_scalar[50]
        getitem_51: "f32[]" = _foreach_add_scalar[51]
        getitem_52: "f32[]" = _foreach_add_scalar[52]
        getitem_53: "f32[]" = _foreach_add_scalar[53]
        getitem_54: "f32[]" = _foreach_add_scalar[54]
        getitem_55: "f32[]" = _foreach_add_scalar[55]
        getitem_56: "f32[]" = _foreach_add_scalar[56]
        getitem_57: "f32[]" = _foreach_add_scalar[57]
        getitem_58: "f32[]" = _foreach_add_scalar[58]
        getitem_59: "f32[]" = _foreach_add_scalar[59]
        getitem_60: "f32[]" = _foreach_add_scalar[60]
        getitem_61: "f32[]" = _foreach_add_scalar[61]
        getitem_62: "f32[]" = _foreach_add_scalar[62]
        getitem_63: "f32[]" = _foreach_add_scalar[63]
        getitem_64: "f32[]" = _foreach_add_scalar[64]
        getitem_65: "f32[]" = _foreach_add_scalar[65]
        getitem_66: "f32[]" = _foreach_add_scalar[66]
        getitem_67: "f32[]" = _foreach_add_scalar[67]
        getitem_68: "f32[]" = _foreach_add_scalar[68]
        getitem_69: "f32[]" = _foreach_add_scalar[69]
        getitem_70: "f32[]" = _foreach_add_scalar[70]
        getitem_71: "f32[]" = _foreach_add_scalar[71]
        getitem_72: "f32[]" = _foreach_add_scalar[72]
        getitem_73: "f32[]" = _foreach_add_scalar[73]
        getitem_74: "f32[]" = _foreach_add_scalar[74]
        getitem_75: "f32[]" = _foreach_add_scalar[75]
        getitem_76: "f32[]" = _foreach_add_scalar[76]
        getitem_77: "f32[]" = _foreach_add_scalar[77]
        getitem_78: "f32[]" = _foreach_add_scalar[78]
        getitem_79: "f32[]" = _foreach_add_scalar[79]
        getitem_80: "f32[]" = _foreach_add_scalar[80]
        getitem_81: "f32[]" = _foreach_add_scalar[81]
        getitem_82: "f32[]" = _foreach_add_scalar[82]
        getitem_83: "f32[]" = _foreach_add_scalar[83]
        getitem_84: "f32[]" = _foreach_add_scalar[84]
        getitem_85: "f32[]" = _foreach_add_scalar[85]
        getitem_86: "f32[]" = _foreach_add_scalar[86]
        getitem_87: "f32[]" = _foreach_add_scalar[87]
        getitem_88: "f32[]" = _foreach_add_scalar[88]
        getitem_89: "f32[]" = _foreach_add_scalar[89]
        getitem_90: "f32[]" = _foreach_add_scalar[90]
        getitem_91: "f32[]" = _foreach_add_scalar[91]
        getitem_92: "f32[]" = _foreach_add_scalar[92]
        getitem_93: "f32[]" = _foreach_add_scalar[93]
        getitem_94: "f32[]" = _foreach_add_scalar[94]
        getitem_95: "f32[]" = _foreach_add_scalar[95]
        getitem_96: "f32[]" = _foreach_add_scalar[96]
        getitem_97: "f32[]" = _foreach_add_scalar[97]
        getitem_98: "f32[]" = _foreach_add_scalar[98]
        getitem_99: "f32[]" = _foreach_add_scalar[99]
        getitem_100: "f32[]" = _foreach_add_scalar[100]
        getitem_101: "f32[]" = _foreach_add_scalar[101]
        getitem_102: "f32[]" = _foreach_add_scalar[102]
        getitem_103: "f32[]" = _foreach_add_scalar[103]
        getitem_104: "f32[]" = _foreach_add_scalar[104]
        getitem_105: "f32[]" = _foreach_add_scalar[105]
        getitem_106: "f32[]" = _foreach_add_scalar[106]
        getitem_107: "f32[]" = _foreach_add_scalar[107]
        getitem_108: "f32[]" = _foreach_add_scalar[108]
        getitem_109: "f32[]" = _foreach_add_scalar[109]
        getitem_110: "f32[]" = _foreach_add_scalar[110]
        getitem_111: "f32[]" = _foreach_add_scalar[111]
        getitem_112: "f32[]" = _foreach_add_scalar[112]
        getitem_113: "f32[]" = _foreach_add_scalar[113]
        getitem_114: "f32[]" = _foreach_add_scalar[114]
        getitem_115: "f32[]" = _foreach_add_scalar[115]
        getitem_116: "f32[]" = _foreach_add_scalar[116]
        getitem_117: "f32[]" = _foreach_add_scalar[117]
        getitem_118: "f32[]" = _foreach_add_scalar[118]
        getitem_119: "f32[]" = _foreach_add_scalar[119]
        getitem_120: "f32[]" = _foreach_add_scalar[120]
        getitem_121: "f32[]" = _foreach_add_scalar[121]
        getitem_122: "f32[]" = _foreach_add_scalar[122]
        getitem_123: "f32[]" = _foreach_add_scalar[123]
        getitem_124: "f32[]" = _foreach_add_scalar[124]
        getitem_125: "f32[]" = _foreach_add_scalar[125]
        getitem_126: "f32[]" = _foreach_add_scalar[126]
        getitem_127: "f32[]" = _foreach_add_scalar[127]
        getitem_128: "f32[]" = _foreach_add_scalar[128]
        getitem_129: "f32[]" = _foreach_add_scalar[129]
        getitem_130: "f32[]" = _foreach_add_scalar[130]
        getitem_131: "f32[]" = _foreach_add_scalar[131]
        getitem_132: "f32[]" = _foreach_add_scalar[132]
        getitem_133: "f32[]" = _foreach_add_scalar[133]
        getitem_134: "f32[]" = _foreach_add_scalar[134]
        getitem_135: "f32[]" = _foreach_add_scalar[135]
        getitem_136: "f32[]" = _foreach_add_scalar[136]
        getitem_137: "f32[]" = _foreach_add_scalar[137]
        getitem_138: "f32[]" = _foreach_add_scalar[138]
        getitem_139: "f32[]" = _foreach_add_scalar[139]
        getitem_140: "f32[]" = _foreach_add_scalar[140]
        getitem_141: "f32[]" = _foreach_add_scalar[141]
        getitem_142: "f32[]" = _foreach_add_scalar[142]
        getitem_143: "f32[]" = _foreach_add_scalar[143]
        getitem_144: "f32[]" = _foreach_add_scalar[144]
        getitem_145: "f32[]" = _foreach_add_scalar[145]
        getitem_146: "f32[]" = _foreach_add_scalar[146]
        getitem_147: "f32[]" = _foreach_add_scalar[147]
        getitem_148: "f32[]" = _foreach_add_scalar[148]
        getitem_149: "f32[]" = _foreach_add_scalar[149]
        getitem_150: "f32[]" = _foreach_add_scalar[150]
        getitem_151: "f32[]" = _foreach_add_scalar[151];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151)


def _default_make_inputs():
    return [
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

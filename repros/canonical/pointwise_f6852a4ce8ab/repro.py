"""
Standalone repro captured via capture_hook.
Label: tlparse_huggingface_s3_g60
Pattern hash: f6852a4ce8ab
Shape hash: a12294de
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
    def forward(self, arg211_1: "f32[]", arg206_1: "f32[]", arg212_1: "f32[]", arg213_1: "f32[]", arg214_1: "f32[]", arg215_1: "f32[]", arg216_1: "f32[]", arg217_1: "f32[]", arg218_1: "f32[]", arg219_1: "f32[]", arg220_1: "f32[]", arg221_1: "f32[]", arg222_1: "f32[]", arg223_1: "f32[]", arg224_1: "f32[]", arg225_1: "f32[]", arg226_1: "f32[]", arg227_1: "f32[]", arg228_1: "f32[]", arg229_1: "f32[]", arg230_1: "f32[]", arg231_1: "f32[]", arg232_1: "f32[]", arg233_1: "f32[]", arg234_1: "f32[]", arg235_1: "f32[]", arg236_1: "f32[]", arg237_1: "f32[]", arg238_1: "f32[]", arg239_1: "f32[]", arg240_1: "f32[]", arg241_1: "f32[]", arg242_1: "f32[]", arg243_1: "f32[]", arg244_1: "f32[]", arg245_1: "f32[]", arg246_1: "f32[]", arg247_1: "f32[]", arg248_1: "f32[]", arg249_1: "f32[]", arg250_1: "f32[]", arg251_1: "f32[]", arg252_1: "f32[]", arg253_1: "f32[]", arg254_1: "f32[]", arg255_1: "f32[]", arg256_1: "f32[]", arg257_1: "f32[]", arg258_1: "f32[]", arg259_1: "f32[]", arg260_1: "f32[]", arg261_1: "f32[]", arg262_1: "f32[]", arg263_1: "f32[]", arg264_1: "f32[]", arg265_1: "f32[]", arg266_1: "f32[]", arg267_1: "f32[]", arg268_1: "f32[]", arg269_1: "f32[]", arg270_1: "f32[]", arg271_1: "f32[]", arg272_1: "f32[]", arg273_1: "f32[]", arg274_1: "f32[]", arg275_1: "f32[]", arg276_1: "f32[]", arg277_1: "f32[]", arg278_1: "f32[]", arg279_1: "f32[]", arg280_1: "f32[]", arg281_1: "f32[]", arg282_1: "f32[]", arg283_1: "f32[]", arg284_1: "f32[]", arg285_1: "f32[]", arg286_1: "f32[]", arg287_1: "f32[]", arg288_1: "f32[]", arg289_1: "f32[]", arg290_1: "f32[]", arg291_1: "f32[]", arg292_1: "f32[]", arg293_1: "f32[]", arg294_1: "f32[]", arg295_1: "f32[]", arg296_1: "f32[]", arg297_1: "f32[]", arg298_1: "f32[]", arg299_1: "f32[]", arg300_1: "f32[]", arg301_1: "f32[]", arg302_1: "f32[]", arg303_1: "f32[]", arg304_1: "f32[]", arg305_1: "f32[]", arg306_1: "f32[]", arg307_1: "f32[]", arg308_1: "f32[]", arg309_1: "f32[]", arg310_1: "f32[]", arg311_1: "f32[]", arg312_1: "f32[]", arg313_1: "f32[]", arg314_1: "f32[]", arg315_1: "f32[]", arg316_1: "f32[]", arg317_1: "f32[]", arg318_1: "f32[]", arg319_1: "f32[]", arg320_1: "f32[]", arg321_1: "f32[]", arg322_1: "f32[]", arg323_1: "f32[]", arg324_1: "f32[]", arg325_1: "f32[]", arg326_1: "f32[]", arg327_1: "f32[]", arg328_1: "f32[]", arg329_1: "f32[]", arg330_1: "f32[]", arg331_1: "f32[]", arg332_1: "f32[]", arg333_1: "f32[]", arg334_1: "f32[]", arg335_1: "f32[]", arg336_1: "f32[]", arg337_1: "f32[]", arg338_1: "f32[]", arg339_1: "f32[]", arg340_1: "f32[]", arg341_1: "f32[]", arg342_1: "f32[]", arg343_1: "f32[]", arg344_1: "f32[]", arg345_1: "f32[]", arg346_1: "f32[]", arg347_1: "f32[]", arg348_1: "f32[]", arg349_1: "f32[]", arg350_1: "f32[]", arg351_1: "f32[]", arg352_1: "f32[]", arg353_1: "f32[]", arg354_1: "f32[]", arg355_1: "f32[]", arg356_1: "f32[]", arg357_1: "f32[]", arg358_1: "f32[]", arg359_1: "f32[]", arg360_1: "f32[]", arg361_1: "f32[]", arg362_1: "f32[]", arg363_1: "f32[]", arg364_1: "f32[]", arg365_1: "f32[]", arg366_1: "f32[]", arg367_1: "f32[]", arg368_1: "f32[]", arg369_1: "f32[]", arg370_1: "f32[]", arg371_1: "f32[]", arg372_1: "f32[]", arg373_1: "f32[]", arg374_1: "f32[]", arg375_1: "f32[]", arg376_1: "f32[]", arg377_1: "f32[]", arg378_1: "f32[]", arg379_1: "f32[]", arg380_1: "f32[]", arg381_1: "f32[]", arg382_1: "f32[]", arg383_1: "f32[]", arg384_1: "f32[]", arg385_1: "f32[]", arg386_1: "f32[]", arg387_1: "f32[]", arg388_1: "f32[]", arg389_1: "f32[]", arg390_1: "f32[]", arg391_1: "f32[]", arg392_1: "f32[]", arg393_1: "f32[]", arg394_1: "f32[]", arg395_1: "f32[]", arg396_1: "f32[]", arg397_1: "f32[]", arg398_1: "f32[]", arg399_1: "f32[]", arg400_1: "f32[]", arg401_1: "f32[]", arg402_1: "f32[]", arg403_1: "f32[]", arg404_1: "f32[]", arg405_1: "f32[]", arg406_1: "f32[]", arg407_1: "f32[]", arg408_1: "f32[]", arg409_1: "f32[]", arg410_1: "f32[]", arg411_1: "f32[]", arg412_1: "f32[]", arg413_1: "f32[]", arg414_1: "f32[]", arg415_1: "f32[]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg211_1, arg206_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1], 1);  arg211_1 = arg206_1 = arg212_1 = arg213_1 = arg214_1 = arg215_1 = arg216_1 = arg217_1 = arg218_1 = arg219_1 = arg220_1 = arg221_1 = arg222_1 = arg223_1 = arg224_1 = arg225_1 = arg226_1 = arg227_1 = arg228_1 = arg229_1 = arg230_1 = arg231_1 = arg232_1 = arg233_1 = arg234_1 = arg235_1 = arg236_1 = arg237_1 = arg238_1 = arg239_1 = arg240_1 = arg241_1 = arg242_1 = arg243_1 = arg244_1 = arg245_1 = arg246_1 = arg247_1 = arg248_1 = arg249_1 = arg250_1 = arg251_1 = arg252_1 = arg253_1 = arg254_1 = arg255_1 = arg256_1 = arg257_1 = arg258_1 = arg259_1 = arg260_1 = arg261_1 = arg262_1 = arg263_1 = arg264_1 = arg265_1 = arg266_1 = arg267_1 = arg268_1 = arg269_1 = arg270_1 = arg271_1 = arg272_1 = arg273_1 = arg274_1 = arg275_1 = arg276_1 = arg277_1 = arg278_1 = arg279_1 = arg280_1 = arg281_1 = arg282_1 = arg283_1 = arg284_1 = arg285_1 = arg286_1 = arg287_1 = arg288_1 = arg289_1 = arg290_1 = arg291_1 = arg292_1 = arg293_1 = arg294_1 = arg295_1 = arg296_1 = arg297_1 = arg298_1 = arg299_1 = arg300_1 = arg301_1 = arg302_1 = arg303_1 = arg304_1 = arg305_1 = arg306_1 = arg307_1 = arg308_1 = arg309_1 = arg310_1 = arg311_1 = arg312_1 = arg313_1 = arg314_1 = arg315_1 = arg316_1 = arg317_1 = arg318_1 = arg319_1 = arg320_1 = arg321_1 = arg322_1 = arg323_1 = arg324_1 = arg325_1 = arg326_1 = arg327_1 = arg328_1 = arg329_1 = arg330_1 = arg331_1 = arg332_1 = arg333_1 = arg334_1 = arg335_1 = arg336_1 = arg337_1 = arg338_1 = arg339_1 = arg340_1 = arg341_1 = arg342_1 = arg343_1 = arg344_1 = arg345_1 = arg346_1 = arg347_1 = arg348_1 = arg349_1 = arg350_1 = arg351_1 = arg352_1 = arg353_1 = arg354_1 = arg355_1 = arg356_1 = arg357_1 = arg358_1 = arg359_1 = arg360_1 = arg361_1 = arg362_1 = arg363_1 = arg364_1 = arg365_1 = arg366_1 = arg367_1 = arg368_1 = arg369_1 = arg370_1 = arg371_1 = arg372_1 = arg373_1 = arg374_1 = arg375_1 = arg376_1 = arg377_1 = arg378_1 = arg379_1 = arg380_1 = arg381_1 = arg382_1 = arg383_1 = arg384_1 = arg385_1 = arg386_1 = arg387_1 = arg388_1 = arg389_1 = arg390_1 = arg391_1 = arg392_1 = arg393_1 = arg394_1 = arg395_1 = arg396_1 = arg397_1 = arg398_1 = arg399_1 = arg400_1 = arg401_1 = arg402_1 = arg403_1 = arg404_1 = arg405_1 = arg406_1 = arg407_1 = arg408_1 = arg409_1 = arg410_1 = arg411_1 = arg412_1 = arg413_1 = arg414_1 = arg415_1 = None
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
        getitem_151: "f32[]" = _foreach_add_scalar[151]
        getitem_152: "f32[]" = _foreach_add_scalar[152]
        getitem_153: "f32[]" = _foreach_add_scalar[153]
        getitem_154: "f32[]" = _foreach_add_scalar[154]
        getitem_155: "f32[]" = _foreach_add_scalar[155]
        getitem_156: "f32[]" = _foreach_add_scalar[156]
        getitem_157: "f32[]" = _foreach_add_scalar[157]
        getitem_158: "f32[]" = _foreach_add_scalar[158]
        getitem_159: "f32[]" = _foreach_add_scalar[159]
        getitem_160: "f32[]" = _foreach_add_scalar[160]
        getitem_161: "f32[]" = _foreach_add_scalar[161]
        getitem_162: "f32[]" = _foreach_add_scalar[162]
        getitem_163: "f32[]" = _foreach_add_scalar[163]
        getitem_164: "f32[]" = _foreach_add_scalar[164]
        getitem_165: "f32[]" = _foreach_add_scalar[165]
        getitem_166: "f32[]" = _foreach_add_scalar[166]
        getitem_167: "f32[]" = _foreach_add_scalar[167]
        getitem_168: "f32[]" = _foreach_add_scalar[168]
        getitem_169: "f32[]" = _foreach_add_scalar[169]
        getitem_170: "f32[]" = _foreach_add_scalar[170]
        getitem_171: "f32[]" = _foreach_add_scalar[171]
        getitem_172: "f32[]" = _foreach_add_scalar[172]
        getitem_173: "f32[]" = _foreach_add_scalar[173]
        getitem_174: "f32[]" = _foreach_add_scalar[174]
        getitem_175: "f32[]" = _foreach_add_scalar[175]
        getitem_176: "f32[]" = _foreach_add_scalar[176]
        getitem_177: "f32[]" = _foreach_add_scalar[177]
        getitem_178: "f32[]" = _foreach_add_scalar[178]
        getitem_179: "f32[]" = _foreach_add_scalar[179]
        getitem_180: "f32[]" = _foreach_add_scalar[180]
        getitem_181: "f32[]" = _foreach_add_scalar[181]
        getitem_182: "f32[]" = _foreach_add_scalar[182]
        getitem_183: "f32[]" = _foreach_add_scalar[183]
        getitem_184: "f32[]" = _foreach_add_scalar[184]
        getitem_185: "f32[]" = _foreach_add_scalar[185]
        getitem_186: "f32[]" = _foreach_add_scalar[186]
        getitem_187: "f32[]" = _foreach_add_scalar[187]
        getitem_188: "f32[]" = _foreach_add_scalar[188]
        getitem_189: "f32[]" = _foreach_add_scalar[189]
        getitem_190: "f32[]" = _foreach_add_scalar[190]
        getitem_191: "f32[]" = _foreach_add_scalar[191]
        getitem_192: "f32[]" = _foreach_add_scalar[192]
        getitem_193: "f32[]" = _foreach_add_scalar[193]
        getitem_194: "f32[]" = _foreach_add_scalar[194]
        getitem_195: "f32[]" = _foreach_add_scalar[195]
        getitem_196: "f32[]" = _foreach_add_scalar[196]
        getitem_197: "f32[]" = _foreach_add_scalar[197]
        getitem_198: "f32[]" = _foreach_add_scalar[198]
        getitem_199: "f32[]" = _foreach_add_scalar[199]
        getitem_200: "f32[]" = _foreach_add_scalar[200]
        getitem_201: "f32[]" = _foreach_add_scalar[201]
        getitem_202: "f32[]" = _foreach_add_scalar[202]
        getitem_203: "f32[]" = _foreach_add_scalar[203]
        getitem_204: "f32[]" = _foreach_add_scalar[204]
        getitem_205: "f32[]" = _foreach_add_scalar[205];  _foreach_add_scalar = None
        return (getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107, getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205)


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
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
    torch.randn([], dtype=torch.float32, device='cuda'),
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

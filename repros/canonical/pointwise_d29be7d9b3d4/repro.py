"""
Standalone repro captured via capture_hook.
Label: inductor_huggingface_perf-2-5-linux.aws.a100_graph6
Pattern hash: d29be7d9b3d4
Shape hash: dc9f7717
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

class Repro(torch.nn.Module):
    def forward(self, arg113_1: "f32[]", arg108_1: "f32[]", arg114_1: "f32[]", arg115_1: "f32[]", arg116_1: "f32[]", arg117_1: "f32[]", arg118_1: "f32[]", arg119_1: "f32[]", arg120_1: "f32[]", arg121_1: "f32[]", arg122_1: "f32[]", arg123_1: "f32[]", arg124_1: "f32[]", arg125_1: "f32[]", arg126_1: "f32[]", arg127_1: "f32[]", arg128_1: "f32[]", arg129_1: "f32[]", arg130_1: "f32[]", arg131_1: "f32[]", arg132_1: "f32[]", arg133_1: "f32[]", arg134_1: "f32[]", arg135_1: "f32[]", arg136_1: "f32[]", arg137_1: "f32[]", arg138_1: "f32[]", arg139_1: "f32[]", arg140_1: "f32[]", arg141_1: "f32[]", arg142_1: "f32[]", arg143_1: "f32[]", arg144_1: "f32[]", arg145_1: "f32[]", arg146_1: "f32[]", arg147_1: "f32[]", arg148_1: "f32[]", arg149_1: "f32[]", arg150_1: "f32[]", arg151_1: "f32[]", arg152_1: "f32[]", arg153_1: "f32[]", arg154_1: "f32[]", arg155_1: "f32[]", arg156_1: "f32[]", arg157_1: "f32[]", arg158_1: "f32[]", arg159_1: "f32[]", arg160_1: "f32[]", arg161_1: "f32[]", arg162_1: "f32[]", arg163_1: "f32[]", arg164_1: "f32[]", arg165_1: "f32[]", arg166_1: "f32[]", arg167_1: "f32[]", arg168_1: "f32[]", arg169_1: "f32[]", arg170_1: "f32[]", arg171_1: "f32[]", arg172_1: "f32[]", arg173_1: "f32[]", arg174_1: "f32[]", arg175_1: "f32[]", arg176_1: "f32[]", arg177_1: "f32[]", arg178_1: "f32[]", arg179_1: "f32[]", arg180_1: "f32[]", arg181_1: "f32[]", arg182_1: "f32[]", arg183_1: "f32[]", arg184_1: "f32[]", arg185_1: "f32[]", arg186_1: "f32[]", arg187_1: "f32[]", arg188_1: "f32[]", arg189_1: "f32[]", arg190_1: "f32[]", arg191_1: "f32[]", arg192_1: "f32[]", arg193_1: "f32[]", arg194_1: "f32[]", arg195_1: "f32[]", arg196_1: "f32[]", arg197_1: "f32[]", arg198_1: "f32[]", arg199_1: "f32[]", arg200_1: "f32[]", arg201_1: "f32[]", arg202_1: "f32[]", arg203_1: "f32[]", arg204_1: "f32[]", arg205_1: "f32[]", arg206_1: "f32[]", arg207_1: "f32[]", arg208_1: "f32[]", arg209_1: "f32[]", arg210_1: "f32[]", arg211_1: "f32[]", arg212_1: "f32[]", arg213_1: "f32[]", arg214_1: "f32[]", arg215_1: "f32[]", arg216_1: "f32[]", arg217_1: "f32[]", arg218_1: "f32[]", arg219_1: "f32[]", arg432_1: "f32[32000, 768]", arg433_1: "f32[512, 768]", arg434_1: "f32[4, 768]", arg435_1: "f32[768]", arg436_1: "f32[768]", arg437_1: "f32[768, 768]", arg438_1: "f32[768]", arg439_1: "f32[768]", arg440_1: "f32[768]", arg441_1: "f32[3072, 768]", arg442_1: "f32[3072]", arg443_1: "f32[768, 3072]", arg444_1: "f32[768]", arg445_1: "f32[768]", arg446_1: "f32[768]", arg447_1: "f32[768]", arg448_1: "f32[768]", arg449_1: "f32[3072, 768]", arg450_1: "f32[3072]", arg451_1: "f32[768, 3072]", arg452_1: "f32[768]", arg453_1: "f32[768]", arg454_1: "f32[768]", arg455_1: "f32[768]", arg456_1: "f32[768]", arg457_1: "f32[3072, 768]", arg458_1: "f32[3072]", arg459_1: "f32[768, 3072]", arg460_1: "f32[768]", arg461_1: "f32[768]", arg462_1: "f32[768]", arg463_1: "f32[768]", arg464_1: "f32[768]", arg465_1: "f32[3072, 768]", arg466_1: "f32[3072]", arg467_1: "f32[768, 3072]", arg468_1: "f32[768]", arg469_1: "f32[768]", arg470_1: "f32[768]", arg471_1: "f32[768]", arg472_1: "f32[768]", arg473_1: "f32[3072, 768]", arg474_1: "f32[3072]", arg475_1: "f32[768, 3072]", arg476_1: "f32[768]", arg477_1: "f32[768]", arg478_1: "f32[768]", arg479_1: "f32[768]", arg480_1: "f32[768]", arg481_1: "f32[3072, 768]", arg482_1: "f32[3072]", arg483_1: "f32[768, 3072]", arg484_1: "f32[768]", arg485_1: "f32[768]", arg486_1: "f32[768]", arg487_1: "f32[768]", arg488_1: "f32[768]", arg489_1: "f32[3072, 768]", arg490_1: "f32[3072]", arg491_1: "f32[768, 3072]", arg492_1: "f32[768]", arg493_1: "f32[768]", arg494_1: "f32[768]", arg495_1: "f32[768]", arg496_1: "f32[768]", arg497_1: "f32[3072, 768]", arg498_1: "f32[3072]", arg499_1: "f32[768, 3072]", arg500_1: "f32[768]", arg501_1: "f32[768]", arg502_1: "f32[768]", arg503_1: "f32[768]", arg504_1: "f32[768]", arg505_1: "f32[3072, 768]", arg506_1: "f32[3072]", arg507_1: "f32[768, 3072]", arg508_1: "f32[768]", arg509_1: "f32[768]", arg510_1: "f32[768]", arg511_1: "f32[768]", arg512_1: "f32[768]", arg513_1: "f32[3072, 768]", arg514_1: "f32[3072]", arg515_1: "f32[768, 3072]", arg516_1: "f32[768]", arg517_1: "f32[768]", arg518_1: "f32[768]", arg519_1: "f32[768]", arg520_1: "f32[768]", arg521_1: "f32[3072, 768]", arg522_1: "f32[3072]", arg523_1: "f32[768, 3072]", arg524_1: "f32[768]", arg525_1: "f32[768]", arg526_1: "f32[768]", arg527_1: "f32[768]", arg528_1: "f32[768]", arg529_1: "f32[3072, 768]", arg530_1: "f32[3072]", arg531_1: "f32[768, 3072]", arg532_1: "f32[768]", arg533_1: "f32[768]", arg534_1: "f32[768]", arg535_1: "f32[32000]", arg536_1: "f32[768, 768]", arg537_1: "f32[768]", arg538_1: "f32[768]", arg539_1: "f32[768]", arg111_1: "f32[32000, 768]", arg109_1: "f32[512, 768]", arg220_1: "f32[4, 768]", arg221_1: "f32[768]", arg222_1: "f32[768]", arg223_1: "f32[768, 768]", arg224_1: "f32[768]", arg225_1: "f32[768]", arg226_1: "f32[768]", arg227_1: "f32[3072, 768]", arg228_1: "f32[3072]", arg229_1: "f32[768, 3072]", arg230_1: "f32[768]", arg231_1: "f32[768]", arg232_1: "f32[768]", arg233_1: "f32[768]", arg234_1: "f32[768]", arg235_1: "f32[3072, 768]", arg236_1: "f32[3072]", arg237_1: "f32[768, 3072]", arg238_1: "f32[768]", arg239_1: "f32[768]", arg240_1: "f32[768]", arg241_1: "f32[768]", arg242_1: "f32[768]", arg243_1: "f32[3072, 768]", arg244_1: "f32[3072]", arg245_1: "f32[768, 3072]", arg246_1: "f32[768]", arg247_1: "f32[768]", arg248_1: "f32[768]", arg249_1: "f32[768]", arg250_1: "f32[768]", arg251_1: "f32[3072, 768]", arg252_1: "f32[3072]", arg253_1: "f32[768, 3072]", arg254_1: "f32[768]", arg255_1: "f32[768]", arg256_1: "f32[768]", arg257_1: "f32[768]", arg258_1: "f32[768]", arg259_1: "f32[3072, 768]", arg260_1: "f32[3072]", arg261_1: "f32[768, 3072]", arg262_1: "f32[768]", arg263_1: "f32[768]", arg264_1: "f32[768]", arg265_1: "f32[768]", arg266_1: "f32[768]", arg267_1: "f32[3072, 768]", arg268_1: "f32[3072]", arg269_1: "f32[768, 3072]", arg270_1: "f32[768]", arg271_1: "f32[768]", arg272_1: "f32[768]", arg273_1: "f32[768]", arg274_1: "f32[768]", arg275_1: "f32[3072, 768]", arg276_1: "f32[3072]", arg277_1: "f32[768, 3072]", arg278_1: "f32[768]", arg279_1: "f32[768]", arg280_1: "f32[768]", arg281_1: "f32[768]", arg282_1: "f32[768]", arg283_1: "f32[3072, 768]", arg284_1: "f32[3072]", arg285_1: "f32[768, 3072]", arg286_1: "f32[768]", arg287_1: "f32[768]", arg288_1: "f32[768]", arg289_1: "f32[768]", arg290_1: "f32[768]", arg291_1: "f32[3072, 768]", arg292_1: "f32[3072]", arg293_1: "f32[768, 3072]", arg294_1: "f32[768]", arg295_1: "f32[768]", arg296_1: "f32[768]", arg297_1: "f32[768]", arg298_1: "f32[768]", arg299_1: "f32[3072, 768]", arg300_1: "f32[3072]", arg301_1: "f32[768, 3072]", arg302_1: "f32[768]", arg303_1: "f32[768]", arg304_1: "f32[768]", arg305_1: "f32[768]", arg306_1: "f32[768]", arg307_1: "f32[3072, 768]", arg308_1: "f32[3072]", arg309_1: "f32[768, 3072]", arg310_1: "f32[768]", arg311_1: "f32[768]", arg312_1: "f32[768]", arg313_1: "f32[768]", arg314_1: "f32[768]", arg315_1: "f32[3072, 768]", arg316_1: "f32[3072]", arg317_1: "f32[768, 3072]", arg318_1: "f32[768]", arg319_1: "f32[768]", arg320_1: "f32[768]", arg321_1: "f32[32000]", arg322_1: "f32[768, 768]", arg323_1: "f32[768]", arg324_1: "f32[768]", arg325_1: "f32[768]", arg112_1: "f32[32000, 768]", arg110_1: "f32[512, 768]", arg326_1: "f32[4, 768]", arg327_1: "f32[768]", arg328_1: "f32[768]", arg329_1: "f32[768, 768]", arg330_1: "f32[768]", arg331_1: "f32[768]", arg332_1: "f32[768]", arg333_1: "f32[3072, 768]", arg334_1: "f32[3072]", arg335_1: "f32[768, 3072]", arg336_1: "f32[768]", arg337_1: "f32[768]", arg338_1: "f32[768]", arg339_1: "f32[768]", arg340_1: "f32[768]", arg341_1: "f32[3072, 768]", arg342_1: "f32[3072]", arg343_1: "f32[768, 3072]", arg344_1: "f32[768]", arg345_1: "f32[768]", arg346_1: "f32[768]", arg347_1: "f32[768]", arg348_1: "f32[768]", arg349_1: "f32[3072, 768]", arg350_1: "f32[3072]", arg351_1: "f32[768, 3072]", arg352_1: "f32[768]", arg353_1: "f32[768]", arg354_1: "f32[768]", arg355_1: "f32[768]", arg356_1: "f32[768]", arg357_1: "f32[3072, 768]", arg358_1: "f32[3072]", arg359_1: "f32[768, 3072]", arg360_1: "f32[768]", arg361_1: "f32[768]", arg362_1: "f32[768]", arg363_1: "f32[768]", arg364_1: "f32[768]", arg365_1: "f32[3072, 768]", arg366_1: "f32[3072]", arg367_1: "f32[768, 3072]", arg368_1: "f32[768]", arg369_1: "f32[768]", arg370_1: "f32[768]", arg371_1: "f32[768]", arg372_1: "f32[768]", arg373_1: "f32[3072, 768]", arg374_1: "f32[3072]", arg375_1: "f32[768, 3072]", arg376_1: "f32[768]", arg377_1: "f32[768]", arg378_1: "f32[768]", arg379_1: "f32[768]", arg380_1: "f32[768]", arg381_1: "f32[3072, 768]", arg382_1: "f32[3072]", arg383_1: "f32[768, 3072]", arg384_1: "f32[768]", arg385_1: "f32[768]", arg386_1: "f32[768]", arg387_1: "f32[768]", arg388_1: "f32[768]", arg389_1: "f32[3072, 768]", arg390_1: "f32[3072]", arg391_1: "f32[768, 3072]", arg392_1: "f32[768]", arg393_1: "f32[768]", arg394_1: "f32[768]", arg395_1: "f32[768]", arg396_1: "f32[768]", arg397_1: "f32[3072, 768]", arg398_1: "f32[3072]", arg399_1: "f32[768, 3072]", arg400_1: "f32[768]", arg401_1: "f32[768]", arg402_1: "f32[768]", arg403_1: "f32[768]", arg404_1: "f32[768]", arg405_1: "f32[3072, 768]", arg406_1: "f32[3072]", arg407_1: "f32[768, 3072]", arg408_1: "f32[768]", arg409_1: "f32[768]", arg410_1: "f32[768]", arg411_1: "f32[768]", arg412_1: "f32[768]", arg413_1: "f32[3072, 768]", arg414_1: "f32[3072]", arg415_1: "f32[768, 3072]", arg416_1: "f32[768]", arg417_1: "f32[768]", arg418_1: "f32[768]", arg419_1: "f32[768]", arg420_1: "f32[768]", arg421_1: "f32[3072, 768]", arg422_1: "f32[3072]", arg423_1: "f32[768, 3072]", arg424_1: "f32[768]", arg425_1: "f32[768]", arg426_1: "f32[768]", arg427_1: "f32[32000]", arg428_1: "f32[768, 768]", arg429_1: "f32[768]", arg430_1: "f32[768]", arg431_1: "f32[768]", arg0_1: "f32[32000, 768]", arg1_1: "f32[512, 768]", arg2_1: "f32[4, 768]", arg3_1: "f32[768]", arg4_1: "f32[768]", arg5_1: "f32[768, 768]", arg6_1: "f32[768]", arg7_1: "f32[768]", arg8_1: "f32[768]", arg9_1: "f32[3072, 768]", arg10_1: "f32[3072]", arg11_1: "f32[768, 3072]", arg12_1: "f32[768]", arg13_1: "f32[768]", arg14_1: "f32[768]", arg15_1: "f32[768]", arg16_1: "f32[768]", arg17_1: "f32[3072, 768]", arg18_1: "f32[3072]", arg19_1: "f32[768, 3072]", arg20_1: "f32[768]", arg21_1: "f32[768]", arg22_1: "f32[768]", arg23_1: "f32[768]", arg24_1: "f32[768]", arg25_1: "f32[3072, 768]", arg26_1: "f32[3072]", arg27_1: "f32[768, 3072]", arg28_1: "f32[768]", arg29_1: "f32[768]", arg30_1: "f32[768]", arg31_1: "f32[768]", arg32_1: "f32[768]", arg33_1: "f32[3072, 768]", arg34_1: "f32[3072]", arg35_1: "f32[768, 3072]", arg36_1: "f32[768]", arg37_1: "f32[768]", arg38_1: "f32[768]", arg39_1: "f32[768]", arg40_1: "f32[768]", arg41_1: "f32[3072, 768]", arg42_1: "f32[3072]", arg43_1: "f32[768, 3072]", arg44_1: "f32[768]", arg45_1: "f32[768]", arg46_1: "f32[768]", arg47_1: "f32[768]", arg48_1: "f32[768]", arg49_1: "f32[3072, 768]", arg50_1: "f32[3072]", arg51_1: "f32[768, 3072]", arg52_1: "f32[768]", arg53_1: "f32[768]", arg54_1: "f32[768]", arg55_1: "f32[768]", arg56_1: "f32[768]", arg57_1: "f32[3072, 768]", arg58_1: "f32[3072]", arg59_1: "f32[768, 3072]", arg60_1: "f32[768]", arg61_1: "f32[768]", arg62_1: "f32[768]", arg63_1: "f32[768]", arg64_1: "f32[768]", arg65_1: "f32[3072, 768]", arg66_1: "f32[3072]", arg67_1: "f32[768, 3072]", arg68_1: "f32[768]", arg69_1: "f32[768]", arg70_1: "f32[768]", arg71_1: "f32[768]", arg72_1: "f32[768]", arg73_1: "f32[3072, 768]", arg74_1: "f32[3072]", arg75_1: "f32[768, 3072]", arg76_1: "f32[768]", arg77_1: "f32[768]", arg78_1: "f32[768]", arg79_1: "f32[768]", arg80_1: "f32[768]", arg81_1: "f32[3072, 768]", arg82_1: "f32[3072]", arg83_1: "f32[768, 3072]", arg84_1: "f32[768]", arg85_1: "f32[768]", arg86_1: "f32[768]", arg87_1: "f32[768]", arg88_1: "f32[768]", arg89_1: "f32[3072, 768]", arg90_1: "f32[3072]", arg91_1: "f32[768, 3072]", arg92_1: "f32[768]", arg93_1: "f32[768]", arg94_1: "f32[768]", arg95_1: "f32[768]", arg96_1: "f32[768]", arg97_1: "f32[3072, 768]", arg98_1: "f32[3072]", arg99_1: "f32[768, 3072]", arg100_1: "f32[768]", arg101_1: "f32[768]", arg102_1: "f32[768]", arg103_1: "f32[32000]", arg104_1: "f32[768, 768]", arg105_1: "f32[768]", arg106_1: "f32[768]", arg107_1: "f32[768]"):
        # No stacktrace found for following nodes
        _foreach_add_scalar = torch.ops.aten._foreach_add.Scalar([arg113_1, arg108_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1], 1)
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
        getitem_107: "f32[]" = _foreach_add_scalar[107];  _foreach_add_scalar = None
        _foreach_sub_list = torch.ops.aten._foreach_sub.List([arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1], [arg111_1, arg109_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1])
        getitem_108: "f32[32000, 768]" = _foreach_sub_list[0]
        getitem_109: "f32[512, 768]" = _foreach_sub_list[1]
        getitem_110: "f32[4, 768]" = _foreach_sub_list[2]
        getitem_111: "f32[768]" = _foreach_sub_list[3]
        getitem_112: "f32[768]" = _foreach_sub_list[4]
        getitem_113: "f32[768, 768]" = _foreach_sub_list[5]
        getitem_114: "f32[768]" = _foreach_sub_list[6]
        getitem_115: "f32[768]" = _foreach_sub_list[7]
        getitem_116: "f32[768]" = _foreach_sub_list[8]
        getitem_117: "f32[3072, 768]" = _foreach_sub_list[9]
        getitem_118: "f32[3072]" = _foreach_sub_list[10]
        getitem_119: "f32[768, 3072]" = _foreach_sub_list[11]
        getitem_120: "f32[768]" = _foreach_sub_list[12]
        getitem_121: "f32[768]" = _foreach_sub_list[13]
        getitem_122: "f32[768]" = _foreach_sub_list[14]
        getitem_123: "f32[768]" = _foreach_sub_list[15]
        getitem_124: "f32[768]" = _foreach_sub_list[16]
        getitem_125: "f32[3072, 768]" = _foreach_sub_list[17]
        getitem_126: "f32[3072]" = _foreach_sub_list[18]
        getitem_127: "f32[768, 3072]" = _foreach_sub_list[19]
        getitem_128: "f32[768]" = _foreach_sub_list[20]
        getitem_129: "f32[768]" = _foreach_sub_list[21]
        getitem_130: "f32[768]" = _foreach_sub_list[22]
        getitem_131: "f32[768]" = _foreach_sub_list[23]
        getitem_132: "f32[768]" = _foreach_sub_list[24]
        getitem_133: "f32[3072, 768]" = _foreach_sub_list[25]
        getitem_134: "f32[3072]" = _foreach_sub_list[26]
        getitem_135: "f32[768, 3072]" = _foreach_sub_list[27]
        getitem_136: "f32[768]" = _foreach_sub_list[28]
        getitem_137: "f32[768]" = _foreach_sub_list[29]
        getitem_138: "f32[768]" = _foreach_sub_list[30]
        getitem_139: "f32[768]" = _foreach_sub_list[31]
        getitem_140: "f32[768]" = _foreach_sub_list[32]
        getitem_141: "f32[3072, 768]" = _foreach_sub_list[33]
        getitem_142: "f32[3072]" = _foreach_sub_list[34]
        getitem_143: "f32[768, 3072]" = _foreach_sub_list[35]
        getitem_144: "f32[768]" = _foreach_sub_list[36]
        getitem_145: "f32[768]" = _foreach_sub_list[37]
        getitem_146: "f32[768]" = _foreach_sub_list[38]
        getitem_147: "f32[768]" = _foreach_sub_list[39]
        getitem_148: "f32[768]" = _foreach_sub_list[40]
        getitem_149: "f32[3072, 768]" = _foreach_sub_list[41]
        getitem_150: "f32[3072]" = _foreach_sub_list[42]
        getitem_151: "f32[768, 3072]" = _foreach_sub_list[43]
        getitem_152: "f32[768]" = _foreach_sub_list[44]
        getitem_153: "f32[768]" = _foreach_sub_list[45]
        getitem_154: "f32[768]" = _foreach_sub_list[46]
        getitem_155: "f32[768]" = _foreach_sub_list[47]
        getitem_156: "f32[768]" = _foreach_sub_list[48]
        getitem_157: "f32[3072, 768]" = _foreach_sub_list[49]
        getitem_158: "f32[3072]" = _foreach_sub_list[50]
        getitem_159: "f32[768, 3072]" = _foreach_sub_list[51]
        getitem_160: "f32[768]" = _foreach_sub_list[52]
        getitem_161: "f32[768]" = _foreach_sub_list[53]
        getitem_162: "f32[768]" = _foreach_sub_list[54]
        getitem_163: "f32[768]" = _foreach_sub_list[55]
        getitem_164: "f32[768]" = _foreach_sub_list[56]
        getitem_165: "f32[3072, 768]" = _foreach_sub_list[57]
        getitem_166: "f32[3072]" = _foreach_sub_list[58]
        getitem_167: "f32[768, 3072]" = _foreach_sub_list[59]
        getitem_168: "f32[768]" = _foreach_sub_list[60]
        getitem_169: "f32[768]" = _foreach_sub_list[61]
        getitem_170: "f32[768]" = _foreach_sub_list[62]
        getitem_171: "f32[768]" = _foreach_sub_list[63]
        getitem_172: "f32[768]" = _foreach_sub_list[64]
        getitem_173: "f32[3072, 768]" = _foreach_sub_list[65]
        getitem_174: "f32[3072]" = _foreach_sub_list[66]
        getitem_175: "f32[768, 3072]" = _foreach_sub_list[67]
        getitem_176: "f32[768]" = _foreach_sub_list[68]
        getitem_177: "f32[768]" = _foreach_sub_list[69]
        getitem_178: "f32[768]" = _foreach_sub_list[70]
        getitem_179: "f32[768]" = _foreach_sub_list[71]
        getitem_180: "f32[768]" = _foreach_sub_list[72]
        getitem_181: "f32[3072, 768]" = _foreach_sub_list[73]
        getitem_182: "f32[3072]" = _foreach_sub_list[74]
        getitem_183: "f32[768, 3072]" = _foreach_sub_list[75]
        getitem_184: "f32[768]" = _foreach_sub_list[76]
        getitem_185: "f32[768]" = _foreach_sub_list[77]
        getitem_186: "f32[768]" = _foreach_sub_list[78]
        getitem_187: "f32[768]" = _foreach_sub_list[79]
        getitem_188: "f32[768]" = _foreach_sub_list[80]
        getitem_189: "f32[3072, 768]" = _foreach_sub_list[81]
        getitem_190: "f32[3072]" = _foreach_sub_list[82]
        getitem_191: "f32[768, 3072]" = _foreach_sub_list[83]
        getitem_192: "f32[768]" = _foreach_sub_list[84]
        getitem_193: "f32[768]" = _foreach_sub_list[85]
        getitem_194: "f32[768]" = _foreach_sub_list[86]
        getitem_195: "f32[768]" = _foreach_sub_list[87]
        getitem_196: "f32[768]" = _foreach_sub_list[88]
        getitem_197: "f32[3072, 768]" = _foreach_sub_list[89]
        getitem_198: "f32[3072]" = _foreach_sub_list[90]
        getitem_199: "f32[768, 3072]" = _foreach_sub_list[91]
        getitem_200: "f32[768]" = _foreach_sub_list[92]
        getitem_201: "f32[768]" = _foreach_sub_list[93]
        getitem_202: "f32[768]" = _foreach_sub_list[94]
        getitem_203: "f32[768]" = _foreach_sub_list[95]
        getitem_204: "f32[768]" = _foreach_sub_list[96]
        getitem_205: "f32[3072, 768]" = _foreach_sub_list[97]
        getitem_206: "f32[3072]" = _foreach_sub_list[98]
        getitem_207: "f32[768, 3072]" = _foreach_sub_list[99]
        getitem_208: "f32[768]" = _foreach_sub_list[100]
        getitem_209: "f32[768]" = _foreach_sub_list[101]
        getitem_210: "f32[768]" = _foreach_sub_list[102]
        getitem_211: "f32[32000]" = _foreach_sub_list[103]
        getitem_212: "f32[768, 768]" = _foreach_sub_list[104]
        getitem_213: "f32[768]" = _foreach_sub_list[105]
        getitem_214: "f32[768]" = _foreach_sub_list[106]
        getitem_215: "f32[768]" = _foreach_sub_list[107];  _foreach_sub_list = None
        full_default: "f32[32000, 768]" = torch.ops.aten.full.default([32000, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_1: "f32[512, 768]" = torch.ops.aten.full.default([512, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_2: "f32[4, 768]" = torch.ops.aten.full.default([4, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_4: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_5: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_7: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_9: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_10: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_12: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_14: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_15: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_17: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_19: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_20: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_22: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_24: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_25: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_27: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_29: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_30: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_32: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_34: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_35: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_37: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_39: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_40: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_42: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_44: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_45: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_47: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_49: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_50: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_52: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_54: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_55: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_57: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_59: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_60: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_62: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_64: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_65: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_67: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_69: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_70: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_72: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_74: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_75: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_77: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_79: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_80: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_82: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_84: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_85: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_87: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_89: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_90: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_92: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_94: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_95: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_97: "f32[3072, 768]" = torch.ops.aten.full.default([3072, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[3072]" = torch.ops.aten.full.default([3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_99: "f32[768, 3072]" = torch.ops.aten.full.default([768, 3072], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_100: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_102: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[32000]" = torch.ops.aten.full.default([32000], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_104: "f32[768, 768]" = torch.ops.aten.full.default([768, 768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_105: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_107: "f32[768]" = torch.ops.aten.full.default([768], 0.09999999999999998, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        _foreach_addcmul_scalar = torch.ops.aten._foreach_addcmul.Scalar([arg111_1, arg109_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1], [full_default, full_default_1, full_default_2, full_default_3, full_default_4, full_default_5, full_default_6, full_default_7, full_default_8, full_default_9, full_default_10, full_default_11, full_default_12, full_default_13, full_default_14, full_default_15, full_default_16, full_default_17, full_default_18, full_default_19, full_default_20, full_default_21, full_default_22, full_default_23, full_default_24, full_default_25, full_default_26, full_default_27, full_default_28, full_default_29, full_default_30, full_default_31, full_default_32, full_default_33, full_default_34, full_default_35, full_default_36, full_default_37, full_default_38, full_default_39, full_default_40, full_default_41, full_default_42, full_default_43, full_default_44, full_default_45, full_default_46, full_default_47, full_default_48, full_default_49, full_default_50, full_default_51, full_default_52, full_default_53, full_default_54, full_default_55, full_default_56, full_default_57, full_default_58, full_default_59, full_default_60, full_default_61, full_default_62, full_default_63, full_default_64, full_default_65, full_default_66, full_default_67, full_default_68, full_default_69, full_default_70, full_default_71, full_default_72, full_default_73, full_default_74, full_default_75, full_default_76, full_default_77, full_default_78, full_default_79, full_default_80, full_default_81, full_default_82, full_default_83, full_default_84, full_default_85, full_default_86, full_default_87, full_default_88, full_default_89, full_default_90, full_default_91, full_default_92, full_default_93, full_default_94, full_default_95, full_default_96, full_default_97, full_default_98, full_default_99, full_default_100, full_default_101, full_default_102, full_default_103, full_default_104, full_default_105, full_default_106, full_default_107], [getitem_108, getitem_109, getitem_110, getitem_111, getitem_112, getitem_113, getitem_114, getitem_115, getitem_116, getitem_117, getitem_118, getitem_119, getitem_120, getitem_121, getitem_122, getitem_123, getitem_124, getitem_125, getitem_126, getitem_127, getitem_128, getitem_129, getitem_130, getitem_131, getitem_132, getitem_133, getitem_134, getitem_135, getitem_136, getitem_137, getitem_138, getitem_139, getitem_140, getitem_141, getitem_142, getitem_143, getitem_144, getitem_145, getitem_146, getitem_147, getitem_148, getitem_149, getitem_150, getitem_151, getitem_152, getitem_153, getitem_154, getitem_155, getitem_156, getitem_157, getitem_158, getitem_159, getitem_160, getitem_161, getitem_162, getitem_163, getitem_164, getitem_165, getitem_166, getitem_167, getitem_168, getitem_169, getitem_170, getitem_171, getitem_172, getitem_173, getitem_174, getitem_175, getitem_176, getitem_177, getitem_178, getitem_179, getitem_180, getitem_181, getitem_182, getitem_183, getitem_184, getitem_185, getitem_186, getitem_187, getitem_188, getitem_189, getitem_190, getitem_191, getitem_192, getitem_193, getitem_194, getitem_195, getitem_196, getitem_197, getitem_198, getitem_199, getitem_200, getitem_201, getitem_202, getitem_203, getitem_204, getitem_205, getitem_206, getitem_207, getitem_208, getitem_209, getitem_210, getitem_211, getitem_212, getitem_213, getitem_214, getitem_215]);  full_default = full_default_1 = full_default_2 = full_default_3 = full_default_4 = full_default_5 = full_default_6 = full_default_7 = full_default_8 = full_default_9 = full_default_10 = full_default_11 = full_default_12 = full_default_13 = full_default_14 = full_default_15 = full_default_16 = full_default_17 = full_default_18 = full_default_19 = full_default_20 = full_default_21 = full_default_22 = full_default_23 = full_default_24 = full_default_25 = full_default_26 = full_default_27 = full_default_28 = full_default_29 = full_default_30 = full_default_31 = full_default_32 = full_default_33 = full_default_34 = full_default_35 = full_default_36 = full_default_37 = full_default_38 = full_default_39 = full_default_40 = full_default_41 = full_default_42 = full_default_43 = full_default_44 = full_default_45 = full_default_46 = full_default_47 = full_default_48 = full_default_49 = full_default_50 = full_default_51 = full_default_52 = full_default_53 = full_default_54 = full_default_55 = full_default_56 = full_default_57 = full_default_58 = full_default_59 = full_default_60 = full_default_61 = full_default_62 = full_default_63 = full_default_64 = full_default_65 = full_default_66 = full_default_67 = full_default_68 = full_default_69 = full_default_70 = full_default_71 = full_default_72 = full_default_73 = full_default_74 = full_default_75 = full_default_76 = full_default_77 = full_default_78 = full_default_79 = full_default_80 = full_default_81 = full_default_82 = full_default_83 = full_default_84 = full_default_85 = full_default_86 = full_default_87 = full_default_88 = full_default_89 = full_default_90 = full_default_91 = full_default_92 = full_default_93 = full_default_94 = full_default_95 = full_default_96 = full_default_97 = full_default_98 = full_default_99 = full_default_100 = full_default_101 = full_default_102 = full_default_103 = full_default_104 = full_default_105 = full_default_106 = full_default_107 = getitem_108 = getitem_109 = getitem_110 = getitem_111 = getitem_112 = getitem_113 = getitem_114 = getitem_115 = getitem_116 = getitem_117 = getitem_118 = getitem_119 = getitem_120 = getitem_121 = getitem_122 = getitem_123 = getitem_124 = getitem_125 = getitem_126 = getitem_127 = getitem_128 = getitem_129 = getitem_130 = getitem_131 = getitem_132 = getitem_133 = getitem_134 = getitem_135 = getitem_136 = getitem_137 = getitem_138 = getitem_139 = getitem_140 = getitem_141 = getitem_142 = getitem_143 = getitem_144 = getitem_145 = getitem_146 = getitem_147 = getitem_148 = getitem_149 = getitem_150 = getitem_151 = getitem_152 = getitem_153 = getitem_154 = getitem_155 = getitem_156 = getitem_157 = getitem_158 = getitem_159 = getitem_160 = getitem_161 = getitem_162 = getitem_163 = getitem_164 = getitem_165 = getitem_166 = getitem_167 = getitem_168 = getitem_169 = getitem_170 = getitem_171 = getitem_172 = getitem_173 = getitem_174 = getitem_175 = getitem_176 = getitem_177 = getitem_178 = getitem_179 = getitem_180 = getitem_181 = getitem_182 = getitem_183 = getitem_184 = getitem_185 = getitem_186 = getitem_187 = getitem_188 = getitem_189 = getitem_190 = getitem_191 = getitem_192 = getitem_193 = getitem_194 = getitem_195 = getitem_196 = getitem_197 = getitem_198 = getitem_199 = getitem_200 = getitem_201 = getitem_202 = getitem_203 = getitem_204 = getitem_205 = getitem_206 = getitem_207 = getitem_208 = getitem_209 = getitem_210 = getitem_211 = getitem_212 = getitem_213 = getitem_214 = getitem_215 = None
        getitem_216: "f32[32000, 768]" = _foreach_addcmul_scalar[0]
        getitem_217: "f32[512, 768]" = _foreach_addcmul_scalar[1]
        getitem_218: "f32[4, 768]" = _foreach_addcmul_scalar[2]
        getitem_219: "f32[768]" = _foreach_addcmul_scalar[3]
        getitem_220: "f32[768]" = _foreach_addcmul_scalar[4]
        getitem_221: "f32[768, 768]" = _foreach_addcmul_scalar[5]
        getitem_222: "f32[768]" = _foreach_addcmul_scalar[6]
        getitem_223: "f32[768]" = _foreach_addcmul_scalar[7]
        getitem_224: "f32[768]" = _foreach_addcmul_scalar[8]
        getitem_225: "f32[3072, 768]" = _foreach_addcmul_scalar[9]
        getitem_226: "f32[3072]" = _foreach_addcmul_scalar[10]
        getitem_227: "f32[768, 3072]" = _foreach_addcmul_scalar[11]
        getitem_228: "f32[768]" = _foreach_addcmul_scalar[12]
        getitem_229: "f32[768]" = _foreach_addcmul_scalar[13]
        getitem_230: "f32[768]" = _foreach_addcmul_scalar[14]
        getitem_231: "f32[768]" = _foreach_addcmul_scalar[15]
        getitem_232: "f32[768]" = _foreach_addcmul_scalar[16]
        getitem_233: "f32[3072, 768]" = _foreach_addcmul_scalar[17]
        getitem_234: "f32[3072]" = _foreach_addcmul_scalar[18]
        getitem_235: "f32[768, 3072]" = _foreach_addcmul_scalar[19]
        getitem_236: "f32[768]" = _foreach_addcmul_scalar[20]
        getitem_237: "f32[768]" = _foreach_addcmul_scalar[21]
        getitem_238: "f32[768]" = _foreach_addcmul_scalar[22]
        getitem_239: "f32[768]" = _foreach_addcmul_scalar[23]
        getitem_240: "f32[768]" = _foreach_addcmul_scalar[24]
        getitem_241: "f32[3072, 768]" = _foreach_addcmul_scalar[25]
        getitem_242: "f32[3072]" = _foreach_addcmul_scalar[26]
        getitem_243: "f32[768, 3072]" = _foreach_addcmul_scalar[27]
        getitem_244: "f32[768]" = _foreach_addcmul_scalar[28]
        getitem_245: "f32[768]" = _foreach_addcmul_scalar[29]
        getitem_246: "f32[768]" = _foreach_addcmul_scalar[30]
        getitem_247: "f32[768]" = _foreach_addcmul_scalar[31]
        getitem_248: "f32[768]" = _foreach_addcmul_scalar[32]
        getitem_249: "f32[3072, 768]" = _foreach_addcmul_scalar[33]
        getitem_250: "f32[3072]" = _foreach_addcmul_scalar[34]
        getitem_251: "f32[768, 3072]" = _foreach_addcmul_scalar[35]
        getitem_252: "f32[768]" = _foreach_addcmul_scalar[36]
        getitem_253: "f32[768]" = _foreach_addcmul_scalar[37]
        getitem_254: "f32[768]" = _foreach_addcmul_scalar[38]
        getitem_255: "f32[768]" = _foreach_addcmul_scalar[39]
        getitem_256: "f32[768]" = _foreach_addcmul_scalar[40]
        getitem_257: "f32[3072, 768]" = _foreach_addcmul_scalar[41]
        getitem_258: "f32[3072]" = _foreach_addcmul_scalar[42]
        getitem_259: "f32[768, 3072]" = _foreach_addcmul_scalar[43]
        getitem_260: "f32[768]" = _foreach_addcmul_scalar[44]
        getitem_261: "f32[768]" = _foreach_addcmul_scalar[45]
        getitem_262: "f32[768]" = _foreach_addcmul_scalar[46]
        getitem_263: "f32[768]" = _foreach_addcmul_scalar[47]
        getitem_264: "f32[768]" = _foreach_addcmul_scalar[48]
        getitem_265: "f32[3072, 768]" = _foreach_addcmul_scalar[49]
        getitem_266: "f32[3072]" = _foreach_addcmul_scalar[50]
        getitem_267: "f32[768, 3072]" = _foreach_addcmul_scalar[51]
        getitem_268: "f32[768]" = _foreach_addcmul_scalar[52]
        getitem_269: "f32[768]" = _foreach_addcmul_scalar[53]
        getitem_270: "f32[768]" = _foreach_addcmul_scalar[54]
        getitem_271: "f32[768]" = _foreach_addcmul_scalar[55]
        getitem_272: "f32[768]" = _foreach_addcmul_scalar[56]
        getitem_273: "f32[3072, 768]" = _foreach_addcmul_scalar[57]
        getitem_274: "f32[3072]" = _foreach_addcmul_scalar[58]
        getitem_275: "f32[768, 3072]" = _foreach_addcmul_scalar[59]
        getitem_276: "f32[768]" = _foreach_addcmul_scalar[60]
        getitem_277: "f32[768]" = _foreach_addcmul_scalar[61]
        getitem_278: "f32[768]" = _foreach_addcmul_scalar[62]
        getitem_279: "f32[768]" = _foreach_addcmul_scalar[63]
        getitem_280: "f32[768]" = _foreach_addcmul_scalar[64]
        getitem_281: "f32[3072, 768]" = _foreach_addcmul_scalar[65]
        getitem_282: "f32[3072]" = _foreach_addcmul_scalar[66]
        getitem_283: "f32[768, 3072]" = _foreach_addcmul_scalar[67]
        getitem_284: "f32[768]" = _foreach_addcmul_scalar[68]
        getitem_285: "f32[768]" = _foreach_addcmul_scalar[69]
        getitem_286: "f32[768]" = _foreach_addcmul_scalar[70]
        getitem_287: "f32[768]" = _foreach_addcmul_scalar[71]
        getitem_288: "f32[768]" = _foreach_addcmul_scalar[72]
        getitem_289: "f32[3072, 768]" = _foreach_addcmul_scalar[73]
        getitem_290: "f32[3072]" = _foreach_addcmul_scalar[74]
        getitem_291: "f32[768, 3072]" = _foreach_addcmul_scalar[75]
        getitem_292: "f32[768]" = _foreach_addcmul_scalar[76]
        getitem_293: "f32[768]" = _foreach_addcmul_scalar[77]
        getitem_294: "f32[768]" = _foreach_addcmul_scalar[78]
        getitem_295: "f32[768]" = _foreach_addcmul_scalar[79]
        getitem_296: "f32[768]" = _foreach_addcmul_scalar[80]
        getitem_297: "f32[3072, 768]" = _foreach_addcmul_scalar[81]
        getitem_298: "f32[3072]" = _foreach_addcmul_scalar[82]
        getitem_299: "f32[768, 3072]" = _foreach_addcmul_scalar[83]
        getitem_300: "f32[768]" = _foreach_addcmul_scalar[84]
        getitem_301: "f32[768]" = _foreach_addcmul_scalar[85]
        getitem_302: "f32[768]" = _foreach_addcmul_scalar[86]
        getitem_303: "f32[768]" = _foreach_addcmul_scalar[87]
        getitem_304: "f32[768]" = _foreach_addcmul_scalar[88]
        getitem_305: "f32[3072, 768]" = _foreach_addcmul_scalar[89]
        getitem_306: "f32[3072]" = _foreach_addcmul_scalar[90]
        getitem_307: "f32[768, 3072]" = _foreach_addcmul_scalar[91]
        getitem_308: "f32[768]" = _foreach_addcmul_scalar[92]
        getitem_309: "f32[768]" = _foreach_addcmul_scalar[93]
        getitem_310: "f32[768]" = _foreach_addcmul_scalar[94]
        getitem_311: "f32[768]" = _foreach_addcmul_scalar[95]
        getitem_312: "f32[768]" = _foreach_addcmul_scalar[96]
        getitem_313: "f32[3072, 768]" = _foreach_addcmul_scalar[97]
        getitem_314: "f32[3072]" = _foreach_addcmul_scalar[98]
        getitem_315: "f32[768, 3072]" = _foreach_addcmul_scalar[99]
        getitem_316: "f32[768]" = _foreach_addcmul_scalar[100]
        getitem_317: "f32[768]" = _foreach_addcmul_scalar[101]
        getitem_318: "f32[768]" = _foreach_addcmul_scalar[102]
        getitem_319: "f32[32000]" = _foreach_addcmul_scalar[103]
        getitem_320: "f32[768, 768]" = _foreach_addcmul_scalar[104]
        getitem_321: "f32[768]" = _foreach_addcmul_scalar[105]
        getitem_322: "f32[768]" = _foreach_addcmul_scalar[106]
        getitem_323: "f32[768]" = _foreach_addcmul_scalar[107];  _foreach_addcmul_scalar = None
        _foreach_mul_scalar = torch.ops.aten._foreach_mul.Scalar([arg112_1, arg110_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1], 0.999)
        getitem_324: "f32[32000, 768]" = _foreach_mul_scalar[0]
        getitem_325: "f32[512, 768]" = _foreach_mul_scalar[1]
        getitem_326: "f32[4, 768]" = _foreach_mul_scalar[2]
        getitem_327: "f32[768]" = _foreach_mul_scalar[3]
        getitem_328: "f32[768]" = _foreach_mul_scalar[4]
        getitem_329: "f32[768, 768]" = _foreach_mul_scalar[5]
        getitem_330: "f32[768]" = _foreach_mul_scalar[6]
        getitem_331: "f32[768]" = _foreach_mul_scalar[7]
        getitem_332: "f32[768]" = _foreach_mul_scalar[8]
        getitem_333: "f32[3072, 768]" = _foreach_mul_scalar[9]
        getitem_334: "f32[3072]" = _foreach_mul_scalar[10]
        getitem_335: "f32[768, 3072]" = _foreach_mul_scalar[11]
        getitem_336: "f32[768]" = _foreach_mul_scalar[12]
        getitem_337: "f32[768]" = _foreach_mul_scalar[13]
        getitem_338: "f32[768]" = _foreach_mul_scalar[14]
        getitem_339: "f32[768]" = _foreach_mul_scalar[15]
        getitem_340: "f32[768]" = _foreach_mul_scalar[16]
        getitem_341: "f32[3072, 768]" = _foreach_mul_scalar[17]
        getitem_342: "f32[3072]" = _foreach_mul_scalar[18]
        getitem_343: "f32[768, 3072]" = _foreach_mul_scalar[19]
        getitem_344: "f32[768]" = _foreach_mul_scalar[20]
        getitem_345: "f32[768]" = _foreach_mul_scalar[21]
        getitem_346: "f32[768]" = _foreach_mul_scalar[22]
        getitem_347: "f32[768]" = _foreach_mul_scalar[23]
        getitem_348: "f32[768]" = _foreach_mul_scalar[24]
        getitem_349: "f32[3072, 768]" = _foreach_mul_scalar[25]
        getitem_350: "f32[3072]" = _foreach_mul_scalar[26]
        getitem_351: "f32[768, 3072]" = _foreach_mul_scalar[27]
        getitem_352: "f32[768]" = _foreach_mul_scalar[28]
        getitem_353: "f32[768]" = _foreach_mul_scalar[29]
        getitem_354: "f32[768]" = _foreach_mul_scalar[30]
        getitem_355: "f32[768]" = _foreach_mul_scalar[31]
        getitem_356: "f32[768]" = _foreach_mul_scalar[32]
        getitem_357: "f32[3072, 768]" = _foreach_mul_scalar[33]
        getitem_358: "f32[3072]" = _foreach_mul_scalar[34]
        getitem_359: "f32[768, 3072]" = _foreach_mul_scalar[35]
        getitem_360: "f32[768]" = _foreach_mul_scalar[36]
        getitem_361: "f32[768]" = _foreach_mul_scalar[37]
        getitem_362: "f32[768]" = _foreach_mul_scalar[38]
        getitem_363: "f32[768]" = _foreach_mul_scalar[39]
        getitem_364: "f32[768]" = _foreach_mul_scalar[40]
        getitem_365: "f32[3072, 768]" = _foreach_mul_scalar[41]
        getitem_366: "f32[3072]" = _foreach_mul_scalar[42]
        getitem_367: "f32[768, 3072]" = _foreach_mul_scalar[43]
        getitem_368: "f32[768]" = _foreach_mul_scalar[44]
        getitem_369: "f32[768]" = _foreach_mul_scalar[45]
        getitem_370: "f32[768]" = _foreach_mul_scalar[46]
        getitem_371: "f32[768]" = _foreach_mul_scalar[47]
        getitem_372: "f32[768]" = _foreach_mul_scalar[48]
        getitem_373: "f32[3072, 768]" = _foreach_mul_scalar[49]
        getitem_374: "f32[3072]" = _foreach_mul_scalar[50]
        getitem_375: "f32[768, 3072]" = _foreach_mul_scalar[51]
        getitem_376: "f32[768]" = _foreach_mul_scalar[52]
        getitem_377: "f32[768]" = _foreach_mul_scalar[53]
        getitem_378: "f32[768]" = _foreach_mul_scalar[54]
        getitem_379: "f32[768]" = _foreach_mul_scalar[55]
        getitem_380: "f32[768]" = _foreach_mul_scalar[56]
        getitem_381: "f32[3072, 768]" = _foreach_mul_scalar[57]
        getitem_382: "f32[3072]" = _foreach_mul_scalar[58]
        getitem_383: "f32[768, 3072]" = _foreach_mul_scalar[59]
        getitem_384: "f32[768]" = _foreach_mul_scalar[60]
        getitem_385: "f32[768]" = _foreach_mul_scalar[61]
        getitem_386: "f32[768]" = _foreach_mul_scalar[62]
        getitem_387: "f32[768]" = _foreach_mul_scalar[63]
        getitem_388: "f32[768]" = _foreach_mul_scalar[64]
        getitem_389: "f32[3072, 768]" = _foreach_mul_scalar[65]
        getitem_390: "f32[3072]" = _foreach_mul_scalar[66]
        getitem_391: "f32[768, 3072]" = _foreach_mul_scalar[67]
        getitem_392: "f32[768]" = _foreach_mul_scalar[68]
        getitem_393: "f32[768]" = _foreach_mul_scalar[69]
        getitem_394: "f32[768]" = _foreach_mul_scalar[70]
        getitem_395: "f32[768]" = _foreach_mul_scalar[71]
        getitem_396: "f32[768]" = _foreach_mul_scalar[72]
        getitem_397: "f32[3072, 768]" = _foreach_mul_scalar[73]
        getitem_398: "f32[3072]" = _foreach_mul_scalar[74]
        getitem_399: "f32[768, 3072]" = _foreach_mul_scalar[75]
        getitem_400: "f32[768]" = _foreach_mul_scalar[76]
        getitem_401: "f32[768]" = _foreach_mul_scalar[77]
        getitem_402: "f32[768]" = _foreach_mul_scalar[78]
        getitem_403: "f32[768]" = _foreach_mul_scalar[79]
        getitem_404: "f32[768]" = _foreach_mul_scalar[80]
        getitem_405: "f32[3072, 768]" = _foreach_mul_scalar[81]
        getitem_406: "f32[3072]" = _foreach_mul_scalar[82]
        getitem_407: "f32[768, 3072]" = _foreach_mul_scalar[83]
        getitem_408: "f32[768]" = _foreach_mul_scalar[84]
        getitem_409: "f32[768]" = _foreach_mul_scalar[85]
        getitem_410: "f32[768]" = _foreach_mul_scalar[86]
        getitem_411: "f32[768]" = _foreach_mul_scalar[87]
        getitem_412: "f32[768]" = _foreach_mul_scalar[88]
        getitem_413: "f32[3072, 768]" = _foreach_mul_scalar[89]
        getitem_414: "f32[3072]" = _foreach_mul_scalar[90]
        getitem_415: "f32[768, 3072]" = _foreach_mul_scalar[91]
        getitem_416: "f32[768]" = _foreach_mul_scalar[92]
        getitem_417: "f32[768]" = _foreach_mul_scalar[93]
        getitem_418: "f32[768]" = _foreach_mul_scalar[94]
        getitem_419: "f32[768]" = _foreach_mul_scalar[95]
        getitem_420: "f32[768]" = _foreach_mul_scalar[96]
        getitem_421: "f32[3072, 768]" = _foreach_mul_scalar[97]
        getitem_422: "f32[3072]" = _foreach_mul_scalar[98]
        getitem_423: "f32[768, 3072]" = _foreach_mul_scalar[99]
        getitem_424: "f32[768]" = _foreach_mul_scalar[100]
        getitem_425: "f32[768]" = _foreach_mul_scalar[101]
        getitem_426: "f32[768]" = _foreach_mul_scalar[102]
        getitem_427: "f32[32000]" = _foreach_mul_scalar[103]
        getitem_428: "f32[768, 768]" = _foreach_mul_scalar[104]
        getitem_429: "f32[768]" = _foreach_mul_scalar[105]
        getitem_430: "f32[768]" = _foreach_mul_scalar[106]
        getitem_431: "f32[768]" = _foreach_mul_scalar[107];  _foreach_mul_scalar = None
        _foreach_addcmul_scalar_1 = torch.ops.aten._foreach_addcmul.Scalar([getitem_324, getitem_325, getitem_326, getitem_327, getitem_328, getitem_329, getitem_330, getitem_331, getitem_332, getitem_333, getitem_334, getitem_335, getitem_336, getitem_337, getitem_338, getitem_339, getitem_340, getitem_341, getitem_342, getitem_343, getitem_344, getitem_345, getitem_346, getitem_347, getitem_348, getitem_349, getitem_350, getitem_351, getitem_352, getitem_353, getitem_354, getitem_355, getitem_356, getitem_357, getitem_358, getitem_359, getitem_360, getitem_361, getitem_362, getitem_363, getitem_364, getitem_365, getitem_366, getitem_367, getitem_368, getitem_369, getitem_370, getitem_371, getitem_372, getitem_373, getitem_374, getitem_375, getitem_376, getitem_377, getitem_378, getitem_379, getitem_380, getitem_381, getitem_382, getitem_383, getitem_384, getitem_385, getitem_386, getitem_387, getitem_388, getitem_389, getitem_390, getitem_391, getitem_392, getitem_393, getitem_394, getitem_395, getitem_396, getitem_397, getitem_398, getitem_399, getitem_400, getitem_401, getitem_402, getitem_403, getitem_404, getitem_405, getitem_406, getitem_407, getitem_408, getitem_409, getitem_410, getitem_411, getitem_412, getitem_413, getitem_414, getitem_415, getitem_416, getitem_417, getitem_418, getitem_419, getitem_420, getitem_421, getitem_422, getitem_423, getitem_424, getitem_425, getitem_426, getitem_427, getitem_428, getitem_429, getitem_430, getitem_431], [arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1], [arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1], 0.0010000000000000009);  getitem_324 = getitem_325 = getitem_326 = getitem_327 = getitem_328 = getitem_329 = getitem_330 = getitem_331 = getitem_332 = getitem_333 = getitem_334 = getitem_335 = getitem_336 = getitem_337 = getitem_338 = getitem_339 = getitem_340 = getitem_341 = getitem_342 = getitem_343 = getitem_344 = getitem_345 = getitem_346 = getitem_347 = getitem_348 = getitem_349 = getitem_350 = getitem_351 = getitem_352 = getitem_353 = getitem_354 = getitem_355 = getitem_356 = getitem_357 = getitem_358 = getitem_359 = getitem_360 = getitem_361 = getitem_362 = getitem_363 = getitem_364 = getitem_365 = getitem_366 = getitem_367 = getitem_368 = getitem_369 = getitem_370 = getitem_371 = getitem_372 = getitem_373 = getitem_374 = getitem_375 = getitem_376 = getitem_377 = getitem_378 = getitem_379 = getitem_380 = getitem_381 = getitem_382 = getitem_383 = getitem_384 = getitem_385 = getitem_386 = getitem_387 = getitem_388 = getitem_389 = getitem_390 = getitem_391 = getitem_392 = getitem_393 = getitem_394 = getitem_395 = getitem_396 = getitem_397 = getitem_398 = getitem_399 = getitem_400 = getitem_401 = getitem_402 = getitem_403 = getitem_404 = getitem_405 = getitem_406 = getitem_407 = getitem_408 = getitem_409 = getitem_410 = getitem_411 = getitem_412 = getitem_413 = getitem_414 = getitem_415 = getitem_416 = getitem_417 = getitem_418 = getitem_419 = getitem_420 = getitem_421 = getitem_422 = getitem_423 = getitem_424 = getitem_425 = getitem_426 = getitem_427 = getitem_428 = getitem_429 = getitem_430 = getitem_431 = arg432_1 = arg433_1 = arg434_1 = arg435_1 = arg436_1 = arg437_1 = arg438_1 = arg439_1 = arg440_1 = arg441_1 = arg442_1 = arg443_1 = arg444_1 = arg445_1 = arg446_1 = arg447_1 = arg448_1 = arg449_1 = arg450_1 = arg451_1 = arg452_1 = arg453_1 = arg454_1 = arg455_1 = arg456_1 = arg457_1 = arg458_1 = arg459_1 = arg460_1 = arg461_1 = arg462_1 = arg463_1 = arg464_1 = arg465_1 = arg466_1 = arg467_1 = arg468_1 = arg469_1 = arg470_1 = arg471_1 = arg472_1 = arg473_1 = arg474_1 = arg475_1 = arg476_1 = arg477_1 = arg478_1 = arg479_1 = arg480_1 = arg481_1 = arg482_1 = arg483_1 = arg484_1 = arg485_1 = arg486_1 = arg487_1 = arg488_1 = arg489_1 = arg490_1 = arg491_1 = arg492_1 = arg493_1 = arg494_1 = arg495_1 = arg496_1 = arg497_1 = arg498_1 = arg499_1 = arg500_1 = arg501_1 = arg502_1 = arg503_1 = arg504_1 = arg505_1 = arg506_1 = arg507_1 = arg508_1 = arg509_1 = arg510_1 = arg511_1 = arg512_1 = arg513_1 = arg514_1 = arg515_1 = arg516_1 = arg517_1 = arg518_1 = arg519_1 = arg520_1 = arg521_1 = arg522_1 = arg523_1 = arg524_1 = arg525_1 = arg526_1 = arg527_1 = arg528_1 = arg529_1 = arg530_1 = arg531_1 = arg532_1 = arg533_1 = arg534_1 = arg535_1 = arg536_1 = arg537_1 = arg538_1 = arg539_1 = None
        getitem_432: "f32[32000, 768]" = _foreach_addcmul_scalar_1[0]
        getitem_433: "f32[512, 768]" = _foreach_addcmul_scalar_1[1]
        getitem_434: "f32[4, 768]" = _foreach_addcmul_scalar_1[2]
        getitem_435: "f32[768]" = _foreach_addcmul_scalar_1[3]
        getitem_436: "f32[768]" = _foreach_addcmul_scalar_1[4]
        getitem_437: "f32[768, 768]" = _foreach_addcmul_scalar_1[5]
        getitem_438: "f32[768]" = _foreach_addcmul_scalar_1[6]
        getitem_439: "f32[768]" = _foreach_addcmul_scalar_1[7]
        getitem_440: "f32[768]" = _foreach_addcmul_scalar_1[8]
        getitem_441: "f32[3072, 768]" = _foreach_addcmul_scalar_1[9]
        getitem_442: "f32[3072]" = _foreach_addcmul_scalar_1[10]
        getitem_443: "f32[768, 3072]" = _foreach_addcmul_scalar_1[11]
        getitem_444: "f32[768]" = _foreach_addcmul_scalar_1[12]
        getitem_445: "f32[768]" = _foreach_addcmul_scalar_1[13]
        getitem_446: "f32[768]" = _foreach_addcmul_scalar_1[14]
        getitem_447: "f32[768]" = _foreach_addcmul_scalar_1[15]
        getitem_448: "f32[768]" = _foreach_addcmul_scalar_1[16]
        getitem_449: "f32[3072, 768]" = _foreach_addcmul_scalar_1[17]
        getitem_450: "f32[3072]" = _foreach_addcmul_scalar_1[18]
        getitem_451: "f32[768, 3072]" = _foreach_addcmul_scalar_1[19]
        getitem_452: "f32[768]" = _foreach_addcmul_scalar_1[20]
        getitem_453: "f32[768]" = _foreach_addcmul_scalar_1[21]
        getitem_454: "f32[768]" = _foreach_addcmul_scalar_1[22]
        getitem_455: "f32[768]" = _foreach_addcmul_scalar_1[23]
        getitem_456: "f32[768]" = _foreach_addcmul_scalar_1[24]
        getitem_457: "f32[3072, 768]" = _foreach_addcmul_scalar_1[25]
        getitem_458: "f32[3072]" = _foreach_addcmul_scalar_1[26]
        getitem_459: "f32[768, 3072]" = _foreach_addcmul_scalar_1[27]
        getitem_460: "f32[768]" = _foreach_addcmul_scalar_1[28]
        getitem_461: "f32[768]" = _foreach_addcmul_scalar_1[29]
        getitem_462: "f32[768]" = _foreach_addcmul_scalar_1[30]
        getitem_463: "f32[768]" = _foreach_addcmul_scalar_1[31]
        getitem_464: "f32[768]" = _foreach_addcmul_scalar_1[32]
        getitem_465: "f32[3072, 768]" = _foreach_addcmul_scalar_1[33]
        getitem_466: "f32[3072]" = _foreach_addcmul_scalar_1[34]
        getitem_467: "f32[768, 3072]" = _foreach_addcmul_scalar_1[35]
        getitem_468: "f32[768]" = _foreach_addcmul_scalar_1[36]
        getitem_469: "f32[768]" = _foreach_addcmul_scalar_1[37]
        getitem_470: "f32[768]" = _foreach_addcmul_scalar_1[38]
        getitem_471: "f32[768]" = _foreach_addcmul_scalar_1[39]
        getitem_472: "f32[768]" = _foreach_addcmul_scalar_1[40]
        getitem_473: "f32[3072, 768]" = _foreach_addcmul_scalar_1[41]
        getitem_474: "f32[3072]" = _foreach_addcmul_scalar_1[42]
        getitem_475: "f32[768, 3072]" = _foreach_addcmul_scalar_1[43]
        getitem_476: "f32[768]" = _foreach_addcmul_scalar_1[44]
        getitem_477: "f32[768]" = _foreach_addcmul_scalar_1[45]
        getitem_478: "f32[768]" = _foreach_addcmul_scalar_1[46]
        getitem_479: "f32[768]" = _foreach_addcmul_scalar_1[47]
        getitem_480: "f32[768]" = _foreach_addcmul_scalar_1[48]
        getitem_481: "f32[3072, 768]" = _foreach_addcmul_scalar_1[49]
        getitem_482: "f32[3072]" = _foreach_addcmul_scalar_1[50]
        getitem_483: "f32[768, 3072]" = _foreach_addcmul_scalar_1[51]
        getitem_484: "f32[768]" = _foreach_addcmul_scalar_1[52]
        getitem_485: "f32[768]" = _foreach_addcmul_scalar_1[53]
        getitem_486: "f32[768]" = _foreach_addcmul_scalar_1[54]
        getitem_487: "f32[768]" = _foreach_addcmul_scalar_1[55]
        getitem_488: "f32[768]" = _foreach_addcmul_scalar_1[56]
        getitem_489: "f32[3072, 768]" = _foreach_addcmul_scalar_1[57]
        getitem_490: "f32[3072]" = _foreach_addcmul_scalar_1[58]
        getitem_491: "f32[768, 3072]" = _foreach_addcmul_scalar_1[59]
        getitem_492: "f32[768]" = _foreach_addcmul_scalar_1[60]
        getitem_493: "f32[768]" = _foreach_addcmul_scalar_1[61]
        getitem_494: "f32[768]" = _foreach_addcmul_scalar_1[62]
        getitem_495: "f32[768]" = _foreach_addcmul_scalar_1[63]
        getitem_496: "f32[768]" = _foreach_addcmul_scalar_1[64]
        getitem_497: "f32[3072, 768]" = _foreach_addcmul_scalar_1[65]
        getitem_498: "f32[3072]" = _foreach_addcmul_scalar_1[66]
        getitem_499: "f32[768, 3072]" = _foreach_addcmul_scalar_1[67]
        getitem_500: "f32[768]" = _foreach_addcmul_scalar_1[68]
        getitem_501: "f32[768]" = _foreach_addcmul_scalar_1[69]
        getitem_502: "f32[768]" = _foreach_addcmul_scalar_1[70]
        getitem_503: "f32[768]" = _foreach_addcmul_scalar_1[71]
        getitem_504: "f32[768]" = _foreach_addcmul_scalar_1[72]
        getitem_505: "f32[3072, 768]" = _foreach_addcmul_scalar_1[73]
        getitem_506: "f32[3072]" = _foreach_addcmul_scalar_1[74]
        getitem_507: "f32[768, 3072]" = _foreach_addcmul_scalar_1[75]
        getitem_508: "f32[768]" = _foreach_addcmul_scalar_1[76]
        getitem_509: "f32[768]" = _foreach_addcmul_scalar_1[77]
        getitem_510: "f32[768]" = _foreach_addcmul_scalar_1[78]
        getitem_511: "f32[768]" = _foreach_addcmul_scalar_1[79]
        getitem_512: "f32[768]" = _foreach_addcmul_scalar_1[80]
        getitem_513: "f32[3072, 768]" = _foreach_addcmul_scalar_1[81]
        getitem_514: "f32[3072]" = _foreach_addcmul_scalar_1[82]
        getitem_515: "f32[768, 3072]" = _foreach_addcmul_scalar_1[83]
        getitem_516: "f32[768]" = _foreach_addcmul_scalar_1[84]
        getitem_517: "f32[768]" = _foreach_addcmul_scalar_1[85]
        getitem_518: "f32[768]" = _foreach_addcmul_scalar_1[86]
        getitem_519: "f32[768]" = _foreach_addcmul_scalar_1[87]
        getitem_520: "f32[768]" = _foreach_addcmul_scalar_1[88]
        getitem_521: "f32[3072, 768]" = _foreach_addcmul_scalar_1[89]
        getitem_522: "f32[3072]" = _foreach_addcmul_scalar_1[90]
        getitem_523: "f32[768, 3072]" = _foreach_addcmul_scalar_1[91]
        getitem_524: "f32[768]" = _foreach_addcmul_scalar_1[92]
        getitem_525: "f32[768]" = _foreach_addcmul_scalar_1[93]
        getitem_526: "f32[768]" = _foreach_addcmul_scalar_1[94]
        getitem_527: "f32[768]" = _foreach_addcmul_scalar_1[95]
        getitem_528: "f32[768]" = _foreach_addcmul_scalar_1[96]
        getitem_529: "f32[3072, 768]" = _foreach_addcmul_scalar_1[97]
        getitem_530: "f32[3072]" = _foreach_addcmul_scalar_1[98]
        getitem_531: "f32[768, 3072]" = _foreach_addcmul_scalar_1[99]
        getitem_532: "f32[768]" = _foreach_addcmul_scalar_1[100]
        getitem_533: "f32[768]" = _foreach_addcmul_scalar_1[101]
        getitem_534: "f32[768]" = _foreach_addcmul_scalar_1[102]
        getitem_535: "f32[32000]" = _foreach_addcmul_scalar_1[103]
        getitem_536: "f32[768, 768]" = _foreach_addcmul_scalar_1[104]
        getitem_537: "f32[768]" = _foreach_addcmul_scalar_1[105]
        getitem_538: "f32[768]" = _foreach_addcmul_scalar_1[106]
        getitem_539: "f32[768]" = _foreach_addcmul_scalar_1[107];  _foreach_addcmul_scalar_1 = None
        _foreach_pow_scalar_and_tensor = torch.ops.aten._foreach_pow.ScalarAndTensor(0.9, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107])
        getitem_540: "f32[]" = _foreach_pow_scalar_and_tensor[0]
        getitem_541: "f32[]" = _foreach_pow_scalar_and_tensor[1]
        getitem_542: "f32[]" = _foreach_pow_scalar_and_tensor[2]
        getitem_543: "f32[]" = _foreach_pow_scalar_and_tensor[3]
        getitem_544: "f32[]" = _foreach_pow_scalar_and_tensor[4]
        getitem_545: "f32[]" = _foreach_pow_scalar_and_tensor[5]
        getitem_546: "f32[]" = _foreach_pow_scalar_and_tensor[6]
        getitem_547: "f32[]" = _foreach_pow_scalar_and_tensor[7]
        getitem_548: "f32[]" = _foreach_pow_scalar_and_tensor[8]
        getitem_549: "f32[]" = _foreach_pow_scalar_and_tensor[9]
        getitem_550: "f32[]" = _foreach_pow_scalar_and_tensor[10]
        getitem_551: "f32[]" = _foreach_pow_scalar_and_tensor[11]
        getitem_552: "f32[]" = _foreach_pow_scalar_and_tensor[12]
        getitem_553: "f32[]" = _foreach_pow_scalar_and_tensor[13]
        getitem_554: "f32[]" = _foreach_pow_scalar_and_tensor[14]
        getitem_555: "f32[]" = _foreach_pow_scalar_and_tensor[15]
        getitem_556: "f32[]" = _foreach_pow_scalar_and_tensor[16]
        getitem_557: "f32[]" = _foreach_pow_scalar_and_tensor[17]
        getitem_558: "f32[]" = _foreach_pow_scalar_and_tensor[18]
        getitem_559: "f32[]" = _foreach_pow_scalar_and_tensor[19]
        getitem_560: "f32[]" = _foreach_pow_scalar_and_tensor[20]
        getitem_561: "f32[]" = _foreach_pow_scalar_and_tensor[21]
        getitem_562: "f32[]" = _foreach_pow_scalar_and_tensor[22]
        getitem_563: "f32[]" = _foreach_pow_scalar_and_tensor[23]
        getitem_564: "f32[]" = _foreach_pow_scalar_and_tensor[24]
        getitem_565: "f32[]" = _foreach_pow_scalar_and_tensor[25]
        getitem_566: "f32[]" = _foreach_pow_scalar_and_tensor[26]
        getitem_567: "f32[]" = _foreach_pow_scalar_and_tensor[27]
        getitem_568: "f32[]" = _foreach_pow_scalar_and_tensor[28]
        getitem_569: "f32[]" = _foreach_pow_scalar_and_tensor[29]
        getitem_570: "f32[]" = _foreach_pow_scalar_and_tensor[30]
        getitem_571: "f32[]" = _foreach_pow_scalar_and_tensor[31]
        getitem_572: "f32[]" = _foreach_pow_scalar_and_tensor[32]
        getitem_573: "f32[]" = _foreach_pow_scalar_and_tensor[33]
        getitem_574: "f32[]" = _foreach_pow_scalar_and_tensor[34]
        getitem_575: "f32[]" = _foreach_pow_scalar_and_tensor[35]
        getitem_576: "f32[]" = _foreach_pow_scalar_and_tensor[36]
        getitem_577: "f32[]" = _foreach_pow_scalar_and_tensor[37]
        getitem_578: "f32[]" = _foreach_pow_scalar_and_tensor[38]
        getitem_579: "f32[]" = _foreach_pow_scalar_and_tensor[39]
        getitem_580: "f32[]" = _foreach_pow_scalar_and_tensor[40]
        getitem_581: "f32[]" = _foreach_pow_scalar_and_tensor[41]
        getitem_582: "f32[]" = _foreach_pow_scalar_and_tensor[42]
        getitem_583: "f32[]" = _foreach_pow_scalar_and_tensor[43]
        getitem_584: "f32[]" = _foreach_pow_scalar_and_tensor[44]
        getitem_585: "f32[]" = _foreach_pow_scalar_and_tensor[45]
        getitem_586: "f32[]" = _foreach_pow_scalar_and_tensor[46]
        getitem_587: "f32[]" = _foreach_pow_scalar_and_tensor[47]
        getitem_588: "f32[]" = _foreach_pow_scalar_and_tensor[48]
        getitem_589: "f32[]" = _foreach_pow_scalar_and_tensor[49]
        getitem_590: "f32[]" = _foreach_pow_scalar_and_tensor[50]
        getitem_591: "f32[]" = _foreach_pow_scalar_and_tensor[51]
        getitem_592: "f32[]" = _foreach_pow_scalar_and_tensor[52]
        getitem_593: "f32[]" = _foreach_pow_scalar_and_tensor[53]
        getitem_594: "f32[]" = _foreach_pow_scalar_and_tensor[54]
        getitem_595: "f32[]" = _foreach_pow_scalar_and_tensor[55]
        getitem_596: "f32[]" = _foreach_pow_scalar_and_tensor[56]
        getitem_597: "f32[]" = _foreach_pow_scalar_and_tensor[57]
        getitem_598: "f32[]" = _foreach_pow_scalar_and_tensor[58]
        getitem_599: "f32[]" = _foreach_pow_scalar_and_tensor[59]
        getitem_600: "f32[]" = _foreach_pow_scalar_and_tensor[60]
        getitem_601: "f32[]" = _foreach_pow_scalar_and_tensor[61]
        getitem_602: "f32[]" = _foreach_pow_scalar_and_tensor[62]
        getitem_603: "f32[]" = _foreach_pow_scalar_and_tensor[63]
        getitem_604: "f32[]" = _foreach_pow_scalar_and_tensor[64]
        getitem_605: "f32[]" = _foreach_pow_scalar_and_tensor[65]
        getitem_606: "f32[]" = _foreach_pow_scalar_and_tensor[66]
        getitem_607: "f32[]" = _foreach_pow_scalar_and_tensor[67]
        getitem_608: "f32[]" = _foreach_pow_scalar_and_tensor[68]
        getitem_609: "f32[]" = _foreach_pow_scalar_and_tensor[69]
        getitem_610: "f32[]" = _foreach_pow_scalar_and_tensor[70]
        getitem_611: "f32[]" = _foreach_pow_scalar_and_tensor[71]
        getitem_612: "f32[]" = _foreach_pow_scalar_and_tensor[72]
        getitem_613: "f32[]" = _foreach_pow_scalar_and_tensor[73]
        getitem_614: "f32[]" = _foreach_pow_scalar_and_tensor[74]
        getitem_615: "f32[]" = _foreach_pow_scalar_and_tensor[75]
        getitem_616: "f32[]" = _foreach_pow_scalar_and_tensor[76]
        getitem_617: "f32[]" = _foreach_pow_scalar_and_tensor[77]
        getitem_618: "f32[]" = _foreach_pow_scalar_and_tensor[78]
        getitem_619: "f32[]" = _foreach_pow_scalar_and_tensor[79]
        getitem_620: "f32[]" = _foreach_pow_scalar_and_tensor[80]
        getitem_621: "f32[]" = _foreach_pow_scalar_and_tensor[81]
        getitem_622: "f32[]" = _foreach_pow_scalar_and_tensor[82]
        getitem_623: "f32[]" = _foreach_pow_scalar_and_tensor[83]
        getitem_624: "f32[]" = _foreach_pow_scalar_and_tensor[84]
        getitem_625: "f32[]" = _foreach_pow_scalar_and_tensor[85]
        getitem_626: "f32[]" = _foreach_pow_scalar_and_tensor[86]
        getitem_627: "f32[]" = _foreach_pow_scalar_and_tensor[87]
        getitem_628: "f32[]" = _foreach_pow_scalar_and_tensor[88]
        getitem_629: "f32[]" = _foreach_pow_scalar_and_tensor[89]
        getitem_630: "f32[]" = _foreach_pow_scalar_and_tensor[90]
        getitem_631: "f32[]" = _foreach_pow_scalar_and_tensor[91]
        getitem_632: "f32[]" = _foreach_pow_scalar_and_tensor[92]
        getitem_633: "f32[]" = _foreach_pow_scalar_and_tensor[93]
        getitem_634: "f32[]" = _foreach_pow_scalar_and_tensor[94]
        getitem_635: "f32[]" = _foreach_pow_scalar_and_tensor[95]
        getitem_636: "f32[]" = _foreach_pow_scalar_and_tensor[96]
        getitem_637: "f32[]" = _foreach_pow_scalar_and_tensor[97]
        getitem_638: "f32[]" = _foreach_pow_scalar_and_tensor[98]
        getitem_639: "f32[]" = _foreach_pow_scalar_and_tensor[99]
        getitem_640: "f32[]" = _foreach_pow_scalar_and_tensor[100]
        getitem_641: "f32[]" = _foreach_pow_scalar_and_tensor[101]
        getitem_642: "f32[]" = _foreach_pow_scalar_and_tensor[102]
        getitem_643: "f32[]" = _foreach_pow_scalar_and_tensor[103]
        getitem_644: "f32[]" = _foreach_pow_scalar_and_tensor[104]
        getitem_645: "f32[]" = _foreach_pow_scalar_and_tensor[105]
        getitem_646: "f32[]" = _foreach_pow_scalar_and_tensor[106]
        getitem_647: "f32[]" = _foreach_pow_scalar_and_tensor[107];  _foreach_pow_scalar_and_tensor = None
        _foreach_pow_scalar_and_tensor_1 = torch.ops.aten._foreach_pow.ScalarAndTensor(0.999, [getitem, getitem_1, getitem_2, getitem_3, getitem_4, getitem_5, getitem_6, getitem_7, getitem_8, getitem_9, getitem_10, getitem_11, getitem_12, getitem_13, getitem_14, getitem_15, getitem_16, getitem_17, getitem_18, getitem_19, getitem_20, getitem_21, getitem_22, getitem_23, getitem_24, getitem_25, getitem_26, getitem_27, getitem_28, getitem_29, getitem_30, getitem_31, getitem_32, getitem_33, getitem_34, getitem_35, getitem_36, getitem_37, getitem_38, getitem_39, getitem_40, getitem_41, getitem_42, getitem_43, getitem_44, getitem_45, getitem_46, getitem_47, getitem_48, getitem_49, getitem_50, getitem_51, getitem_52, getitem_53, getitem_54, getitem_55, getitem_56, getitem_57, getitem_58, getitem_59, getitem_60, getitem_61, getitem_62, getitem_63, getitem_64, getitem_65, getitem_66, getitem_67, getitem_68, getitem_69, getitem_70, getitem_71, getitem_72, getitem_73, getitem_74, getitem_75, getitem_76, getitem_77, getitem_78, getitem_79, getitem_80, getitem_81, getitem_82, getitem_83, getitem_84, getitem_85, getitem_86, getitem_87, getitem_88, getitem_89, getitem_90, getitem_91, getitem_92, getitem_93, getitem_94, getitem_95, getitem_96, getitem_97, getitem_98, getitem_99, getitem_100, getitem_101, getitem_102, getitem_103, getitem_104, getitem_105, getitem_106, getitem_107])
        getitem_648: "f32[]" = _foreach_pow_scalar_and_tensor_1[0]
        getitem_649: "f32[]" = _foreach_pow_scalar_and_tensor_1[1]
        getitem_650: "f32[]" = _foreach_pow_scalar_and_tensor_1[2]
        getitem_651: "f32[]" = _foreach_pow_scalar_and_tensor_1[3]
        getitem_652: "f32[]" = _foreach_pow_scalar_and_tensor_1[4]
        getitem_653: "f32[]" = _foreach_pow_scalar_and_tensor_1[5]
        getitem_654: "f32[]" = _foreach_pow_scalar_and_tensor_1[6]
        getitem_655: "f32[]" = _foreach_pow_scalar_and_tensor_1[7]
        getitem_656: "f32[]" = _foreach_pow_scalar_and_tensor_1[8]
        getitem_657: "f32[]" = _foreach_pow_scalar_and_tensor_1[9]
        getitem_658: "f32[]" = _foreach_pow_scalar_and_tensor_1[10]
        getitem_659: "f32[]" = _foreach_pow_scalar_and_tensor_1[11]
        getitem_660: "f32[]" = _foreach_pow_scalar_and_tensor_1[12]
        getitem_661: "f32[]" = _foreach_pow_scalar_and_tensor_1[13]
        getitem_662: "f32[]" = _foreach_pow_scalar_and_tensor_1[14]
        getitem_663: "f32[]" = _foreach_pow_scalar_and_tensor_1[15]
        getitem_664: "f32[]" = _foreach_pow_scalar_and_tensor_1[16]
        getitem_665: "f32[]" = _foreach_pow_scalar_and_tensor_1[17]
        getitem_666: "f32[]" = _foreach_pow_scalar_and_tensor_1[18]
        getitem_667: "f32[]" = _foreach_pow_scalar_and_tensor_1[19]
        getitem_668: "f32[]" = _foreach_pow_scalar_and_tensor_1[20]
        getitem_669: "f32[]" = _foreach_pow_scalar_and_tensor_1[21]
        getitem_670: "f32[]" = _foreach_pow_scalar_and_tensor_1[22]
        getitem_671: "f32[]" = _foreach_pow_scalar_and_tensor_1[23]
        getitem_672: "f32[]" = _foreach_pow_scalar_and_tensor_1[24]
        getitem_673: "f32[]" = _foreach_pow_scalar_and_tensor_1[25]
        getitem_674: "f32[]" = _foreach_pow_scalar_and_tensor_1[26]
        getitem_675: "f32[]" = _foreach_pow_scalar_and_tensor_1[27]
        getitem_676: "f32[]" = _foreach_pow_scalar_and_tensor_1[28]
        getitem_677: "f32[]" = _foreach_pow_scalar_and_tensor_1[29]
        getitem_678: "f32[]" = _foreach_pow_scalar_and_tensor_1[30]
        getitem_679: "f32[]" = _foreach_pow_scalar_and_tensor_1[31]
        getitem_680: "f32[]" = _foreach_pow_scalar_and_tensor_1[32]
        getitem_681: "f32[]" = _foreach_pow_scalar_and_tensor_1[33]
        getitem_682: "f32[]" = _foreach_pow_scalar_and_tensor_1[34]
        getitem_683: "f32[]" = _foreach_pow_scalar_and_tensor_1[35]
        getitem_684: "f32[]" = _foreach_pow_scalar_and_tensor_1[36]
        getitem_685: "f32[]" = _foreach_pow_scalar_and_tensor_1[37]
        getitem_686: "f32[]" = _foreach_pow_scalar_and_tensor_1[38]
        getitem_687: "f32[]" = _foreach_pow_scalar_and_tensor_1[39]
        getitem_688: "f32[]" = _foreach_pow_scalar_and_tensor_1[40]
        getitem_689: "f32[]" = _foreach_pow_scalar_and_tensor_1[41]
        getitem_690: "f32[]" = _foreach_pow_scalar_and_tensor_1[42]
        getitem_691: "f32[]" = _foreach_pow_scalar_and_tensor_1[43]
        getitem_692: "f32[]" = _foreach_pow_scalar_and_tensor_1[44]
        getitem_693: "f32[]" = _foreach_pow_scalar_and_tensor_1[45]
        getitem_694: "f32[]" = _foreach_pow_scalar_and_tensor_1[46]
        getitem_695: "f32[]" = _foreach_pow_scalar_and_tensor_1[47]
        getitem_696: "f32[]" = _foreach_pow_scalar_and_tensor_1[48]
        getitem_697: "f32[]" = _foreach_pow_scalar_and_tensor_1[49]
        getitem_698: "f32[]" = _foreach_pow_scalar_and_tensor_1[50]
        getitem_699: "f32[]" = _foreach_pow_scalar_and_tensor_1[51]
        getitem_700: "f32[]" = _foreach_pow_scalar_and_tensor_1[52]
        getitem_701: "f32[]" = _foreach_pow_scalar_and_tensor_1[53]
        getitem_702: "f32[]" = _foreach_pow_scalar_and_tensor_1[54]
        getitem_703: "f32[]" = _foreach_pow_scalar_and_tensor_1[55]
        getitem_704: "f32[]" = _foreach_pow_scalar_and_tensor_1[56]
        getitem_705: "f32[]" = _foreach_pow_scalar_and_tensor_1[57]
        getitem_706: "f32[]" = _foreach_pow_scalar_and_tensor_1[58]
        getitem_707: "f32[]" = _foreach_pow_scalar_and_tensor_1[59]
        getitem_708: "f32[]" = _foreach_pow_scalar_and_tensor_1[60]
        getitem_709: "f32[]" = _foreach_pow_scalar_and_tensor_1[61]
        getitem_710: "f32[]" = _foreach_pow_scalar_and_tensor_1[62]
        getitem_711: "f32[]" = _foreach_pow_scalar_and_tensor_1[63]
        getitem_712: "f32[]" = _foreach_pow_scalar_and_tensor_1[64]
        getitem_713: "f32[]" = _foreach_pow_scalar_and_tensor_1[65]
        getitem_714: "f32[]" = _foreach_pow_scalar_and_tensor_1[66]
        getitem_715: "f32[]" = _foreach_pow_scalar_and_tensor_1[67]
        getitem_716: "f32[]" = _foreach_pow_scalar_and_tensor_1[68]
        getitem_717: "f32[]" = _foreach_pow_scalar_and_tensor_1[69]
        getitem_718: "f32[]" = _foreach_pow_scalar_and_tensor_1[70]
        getitem_719: "f32[]" = _foreach_pow_scalar_and_tensor_1[71]
        getitem_720: "f32[]" = _foreach_pow_scalar_and_tensor_1[72]
        getitem_721: "f32[]" = _foreach_pow_scalar_and_tensor_1[73]
        getitem_722: "f32[]" = _foreach_pow_scalar_and_tensor_1[74]
        getitem_723: "f32[]" = _foreach_pow_scalar_and_tensor_1[75]
        getitem_724: "f32[]" = _foreach_pow_scalar_and_tensor_1[76]
        getitem_725: "f32[]" = _foreach_pow_scalar_and_tensor_1[77]
        getitem_726: "f32[]" = _foreach_pow_scalar_and_tensor_1[78]
        getitem_727: "f32[]" = _foreach_pow_scalar_and_tensor_1[79]
        getitem_728: "f32[]" = _foreach_pow_scalar_and_tensor_1[80]
        getitem_729: "f32[]" = _foreach_pow_scalar_and_tensor_1[81]
        getitem_730: "f32[]" = _foreach_pow_scalar_and_tensor_1[82]
        getitem_731: "f32[]" = _foreach_pow_scalar_and_tensor_1[83]
        getitem_732: "f32[]" = _foreach_pow_scalar_and_tensor_1[84]
        getitem_733: "f32[]" = _foreach_pow_scalar_and_tensor_1[85]
        getitem_734: "f32[]" = _foreach_pow_scalar_and_tensor_1[86]
        getitem_735: "f32[]" = _foreach_pow_scalar_and_tensor_1[87]
        getitem_736: "f32[]" = _foreach_pow_scalar_and_tensor_1[88]
        getitem_737: "f32[]" = _foreach_pow_scalar_and_tensor_1[89]
        getitem_738: "f32[]" = _foreach_pow_scalar_and_tensor_1[90]
        getitem_739: "f32[]" = _foreach_pow_scalar_and_tensor_1[91]
        getitem_740: "f32[]" = _foreach_pow_scalar_and_tensor_1[92]
        getitem_741: "f32[]" = _foreach_pow_scalar_and_tensor_1[93]
        getitem_742: "f32[]" = _foreach_pow_scalar_and_tensor_1[94]
        getitem_743: "f32[]" = _foreach_pow_scalar_and_tensor_1[95]
        getitem_744: "f32[]" = _foreach_pow_scalar_and_tensor_1[96]
        getitem_745: "f32[]" = _foreach_pow_scalar_and_tensor_1[97]
        getitem_746: "f32[]" = _foreach_pow_scalar_and_tensor_1[98]
        getitem_747: "f32[]" = _foreach_pow_scalar_and_tensor_1[99]
        getitem_748: "f32[]" = _foreach_pow_scalar_and_tensor_1[100]
        getitem_749: "f32[]" = _foreach_pow_scalar_and_tensor_1[101]
        getitem_750: "f32[]" = _foreach_pow_scalar_and_tensor_1[102]
        getitem_751: "f32[]" = _foreach_pow_scalar_and_tensor_1[103]
        getitem_752: "f32[]" = _foreach_pow_scalar_and_tensor_1[104]
        getitem_753: "f32[]" = _foreach_pow_scalar_and_tensor_1[105]
        getitem_754: "f32[]" = _foreach_pow_scalar_and_tensor_1[106]
        getitem_755: "f32[]" = _foreach_pow_scalar_and_tensor_1[107];  _foreach_pow_scalar_and_tensor_1 = None
        _foreach_sub_scalar = torch.ops.aten._foreach_sub.Scalar([getitem_540, getitem_541, getitem_542, getitem_543, getitem_544, getitem_545, getitem_546, getitem_547, getitem_548, getitem_549, getitem_550, getitem_551, getitem_552, getitem_553, getitem_554, getitem_555, getitem_556, getitem_557, getitem_558, getitem_559, getitem_560, getitem_561, getitem_562, getitem_563, getitem_564, getitem_565, getitem_566, getitem_567, getitem_568, getitem_569, getitem_570, getitem_571, getitem_572, getitem_573, getitem_574, getitem_575, getitem_576, getitem_577, getitem_578, getitem_579, getitem_580, getitem_581, getitem_582, getitem_583, getitem_584, getitem_585, getitem_586, getitem_587, getitem_588, getitem_589, getitem_590, getitem_591, getitem_592, getitem_593, getitem_594, getitem_595, getitem_596, getitem_597, getitem_598, getitem_599, getitem_600, getitem_601, getitem_602, getitem_603, getitem_604, getitem_605, getitem_606, getitem_607, getitem_608, getitem_609, getitem_610, getitem_611, getitem_612, getitem_613, getitem_614, getitem_615, getitem_616, getitem_617, getitem_618, getitem_619, getitem_620, getitem_621, getitem_622, getitem_623, getitem_624, getitem_625, getitem_626, getitem_627, getitem_628, getitem_629, getitem_630, getitem_631, getitem_632, getitem_633, getitem_634, getitem_635, getitem_636, getitem_637, getitem_638, getitem_639, getitem_640, getitem_641, getitem_642, getitem_643, getitem_644, getitem_645, getitem_646, getitem_647], 1);  getitem_540 = getitem_541 = getitem_542 = getitem_543 = getitem_544 = getitem_545 = getitem_546 = getitem_547 = getitem_548 = getitem_549 = getitem_550 = getitem_551 = getitem_552 = getitem_553 = getitem_554 = getitem_555 = getitem_556 = getitem_557 = getitem_558 = getitem_559 = getitem_560 = getitem_561 = getitem_562 = getitem_563 = getitem_564 = getitem_565 = getitem_566 = getitem_567 = getitem_568 = getitem_569 = getitem_570 = getitem_571 = getitem_572 = getitem_573 = getitem_574 = getitem_575 = getitem_576 = getitem_577 = getitem_578 = getitem_579 = getitem_580 = getitem_581 = getitem_582 = getitem_583 = getitem_584 = getitem_585 = getitem_586 = getitem_587 = getitem_588 = getitem_589 = getitem_590 = getitem_591 = getitem_592 = getitem_593 = getitem_594 = getitem_595 = getitem_596 = getitem_597 = getitem_598 = getitem_599 = getitem_600 = getitem_601 = getitem_602 = getitem_603 = getitem_604 = getitem_605 = getitem_606 = getitem_607 = getitem_608 = getitem_609 = getitem_610 = getitem_611 = getitem_612 = getitem_613 = getitem_614 = getitem_615 = getitem_616 = getitem_617 = getitem_618 = getitem_619 = getitem_620 = getitem_621 = getitem_622 = getitem_623 = getitem_624 = getitem_625 = getitem_626 = getitem_627 = getitem_628 = getitem_629 = getitem_630 = getitem_631 = getitem_632 = getitem_633 = getitem_634 = getitem_635 = getitem_636 = getitem_637 = getitem_638 = getitem_639 = getitem_640 = getitem_641 = getitem_642 = getitem_643 = getitem_644 = getitem_645 = getitem_646 = getitem_647 = None
        getitem_756: "f32[]" = _foreach_sub_scalar[0]
        getitem_757: "f32[]" = _foreach_sub_scalar[1]
        getitem_758: "f32[]" = _foreach_sub_scalar[2]
        getitem_759: "f32[]" = _foreach_sub_scalar[3]
        getitem_760: "f32[]" = _foreach_sub_scalar[4]
        getitem_761: "f32[]" = _foreach_sub_scalar[5]
        getitem_762: "f32[]" = _foreach_sub_scalar[6]
        getitem_763: "f32[]" = _foreach_sub_scalar[7]
        getitem_764: "f32[]" = _foreach_sub_scalar[8]
        getitem_765: "f32[]" = _foreach_sub_scalar[9]
        getitem_766: "f32[]" = _foreach_sub_scalar[10]
        getitem_767: "f32[]" = _foreach_sub_scalar[11]
        getitem_768: "f32[]" = _foreach_sub_scalar[12]
        getitem_769: "f32[]" = _foreach_sub_scalar[13]
        getitem_770: "f32[]" = _foreach_sub_scalar[14]
        getitem_771: "f32[]" = _foreach_sub_scalar[15]
        getitem_772: "f32[]" = _foreach_sub_scalar[16]
        getitem_773: "f32[]" = _foreach_sub_scalar[17]
        getitem_774: "f32[]" = _foreach_sub_scalar[18]
        getitem_775: "f32[]" = _foreach_sub_scalar[19]
        getitem_776: "f32[]" = _foreach_sub_scalar[20]
        getitem_777: "f32[]" = _foreach_sub_scalar[21]
        getitem_778: "f32[]" = _foreach_sub_scalar[22]
        getitem_779: "f32[]" = _foreach_sub_scalar[23]
        getitem_780: "f32[]" = _foreach_sub_scalar[24]
        getitem_781: "f32[]" = _foreach_sub_scalar[25]
        getitem_782: "f32[]" = _foreach_sub_scalar[26]
        getitem_783: "f32[]" = _foreach_sub_scalar[27]
        getitem_784: "f32[]" = _foreach_sub_scalar[28]
        getitem_785: "f32[]" = _foreach_sub_scalar[29]
        getitem_786: "f32[]" = _foreach_sub_scalar[30]
        getitem_787: "f32[]" = _foreach_sub_scalar[31]
        getitem_788: "f32[]" = _foreach_sub_scalar[32]
        getitem_789: "f32[]" = _foreach_sub_scalar[33]
        getitem_790: "f32[]" = _foreach_sub_scalar[34]
        getitem_791: "f32[]" = _foreach_sub_scalar[35]
        getitem_792: "f32[]" = _foreach_sub_scalar[36]
        getitem_793: "f32[]" = _foreach_sub_scalar[37]
        getitem_794: "f32[]" = _foreach_sub_scalar[38]
        getitem_795: "f32[]" = _foreach_sub_scalar[39]
        getitem_796: "f32[]" = _foreach_sub_scalar[40]
        getitem_797: "f32[]" = _foreach_sub_scalar[41]
        getitem_798: "f32[]" = _foreach_sub_scalar[42]
        getitem_799: "f32[]" = _foreach_sub_scalar[43]
        getitem_800: "f32[]" = _foreach_sub_scalar[44]
        getitem_801: "f32[]" = _foreach_sub_scalar[45]
        getitem_802: "f32[]" = _foreach_sub_scalar[46]
        getitem_803: "f32[]" = _foreach_sub_scalar[47]
        getitem_804: "f32[]" = _foreach_sub_scalar[48]
        getitem_805: "f32[]" = _foreach_sub_scalar[49]
        getitem_806: "f32[]" = _foreach_sub_scalar[50]
        getitem_807: "f32[]" = _foreach_sub_scalar[51]
        getitem_808: "f32[]" = _foreach_sub_scalar[52]
        getitem_809: "f32[]" = _foreach_sub_scalar[53]
        getitem_810: "f32[]" = _foreach_sub_scalar[54]
        getitem_811: "f32[]" = _foreach_sub_scalar[55]
        getitem_812: "f32[]" = _foreach_sub_scalar[56]
        getitem_813: "f32[]" = _foreach_sub_scalar[57]
        getitem_814: "f32[]" = _foreach_sub_scalar[58]
        getitem_815: "f32[]" = _foreach_sub_scalar[59]
        getitem_816: "f32[]" = _foreach_sub_scalar[60]
        getitem_817: "f32[]" = _foreach_sub_scalar[61]
        getitem_818: "f32[]" = _foreach_sub_scalar[62]
        getitem_819: "f32[]" = _foreach_sub_scalar[63]
        getitem_820: "f32[]" = _foreach_sub_scalar[64]
        getitem_821: "f32[]" = _foreach_sub_scalar[65]
        getitem_822: "f32[]" = _foreach_sub_scalar[66]
        getitem_823: "f32[]" = _foreach_sub_scalar[67]
        getitem_824: "f32[]" = _foreach_sub_scalar[68]
        getitem_825: "f32[]" = _foreach_sub_scalar[69]
        getitem_826: "f32[]" = _foreach_sub_scalar[70]
        getitem_827: "f32[]" = _foreach_sub_scalar[71]
        getitem_828: "f32[]" = _foreach_sub_scalar[72]
        getitem_829: "f32[]" = _foreach_sub_scalar[73]
        getitem_830: "f32[]" = _foreach_sub_scalar[74]
        getitem_831: "f32[]" = _foreach_sub_scalar[75]
        getitem_832: "f32[]" = _foreach_sub_scalar[76]
        getitem_833: "f32[]" = _foreach_sub_scalar[77]
        getitem_834: "f32[]" = _foreach_sub_scalar[78]
        getitem_835: "f32[]" = _foreach_sub_scalar[79]
        getitem_836: "f32[]" = _foreach_sub_scalar[80]
        getitem_837: "f32[]" = _foreach_sub_scalar[81]
        getitem_838: "f32[]" = _foreach_sub_scalar[82]
        getitem_839: "f32[]" = _foreach_sub_scalar[83]
        getitem_840: "f32[]" = _foreach_sub_scalar[84]
        getitem_841: "f32[]" = _foreach_sub_scalar[85]
        getitem_842: "f32[]" = _foreach_sub_scalar[86]
        getitem_843: "f32[]" = _foreach_sub_scalar[87]
        getitem_844: "f32[]" = _foreach_sub_scalar[88]
        getitem_845: "f32[]" = _foreach_sub_scalar[89]
        getitem_846: "f32[]" = _foreach_sub_scalar[90]
        getitem_847: "f32[]" = _foreach_sub_scalar[91]
        getitem_848: "f32[]" = _foreach_sub_scalar[92]
        getitem_849: "f32[]" = _foreach_sub_scalar[93]
        getitem_850: "f32[]" = _foreach_sub_scalar[94]
        getitem_851: "f32[]" = _foreach_sub_scalar[95]
        getitem_852: "f32[]" = _foreach_sub_scalar[96]
        getitem_853: "f32[]" = _foreach_sub_scalar[97]
        getitem_854: "f32[]" = _foreach_sub_scalar[98]
        getitem_855: "f32[]" = _foreach_sub_scalar[99]
        getitem_856: "f32[]" = _foreach_sub_scalar[100]
        getitem_857: "f32[]" = _foreach_sub_scalar[101]
        getitem_858: "f32[]" = _foreach_sub_scalar[102]
        getitem_859: "f32[]" = _foreach_sub_scalar[103]
        getitem_860: "f32[]" = _foreach_sub_scalar[104]
        getitem_861: "f32[]" = _foreach_sub_scalar[105]
        getitem_862: "f32[]" = _foreach_sub_scalar[106]
        getitem_863: "f32[]" = _foreach_sub_scalar[107];  _foreach_sub_scalar = None
        _foreach_sub_scalar_1 = torch.ops.aten._foreach_sub.Scalar([getitem_648, getitem_649, getitem_650, getitem_651, getitem_652, getitem_653, getitem_654, getitem_655, getitem_656, getitem_657, getitem_658, getitem_659, getitem_660, getitem_661, getitem_662, getitem_663, getitem_664, getitem_665, getitem_666, getitem_667, getitem_668, getitem_669, getitem_670, getitem_671, getitem_672, getitem_673, getitem_674, getitem_675, getitem_676, getitem_677, getitem_678, getitem_679, getitem_680, getitem_681, getitem_682, getitem_683, getitem_684, getitem_685, getitem_686, getitem_687, getitem_688, getitem_689, getitem_690, getitem_691, getitem_692, getitem_693, getitem_694, getitem_695, getitem_696, getitem_697, getitem_698, getitem_699, getitem_700, getitem_701, getitem_702, getitem_703, getitem_704, getitem_705, getitem_706, getitem_707, getitem_708, getitem_709, getitem_710, getitem_711, getitem_712, getitem_713, getitem_714, getitem_715, getitem_716, getitem_717, getitem_718, getitem_719, getitem_720, getitem_721, getitem_722, getitem_723, getitem_724, getitem_725, getitem_726, getitem_727, getitem_728, getitem_729, getitem_730, getitem_731, getitem_732, getitem_733, getitem_734, getitem_735, getitem_736, getitem_737, getitem_738, getitem_739, getitem_740, getitem_741, getitem_742, getitem_743, getitem_744, getitem_745, getitem_746, getitem_747, getitem_748, getitem_749, getitem_750, getitem_751, getitem_752, getitem_753, getitem_754, getitem_755], 1);  getitem_648 = getitem_649 = getitem_650 = getitem_651 = getitem_652 = getitem_653 = getitem_654 = getitem_655 = getitem_656 = getitem_657 = getitem_658 = getitem_659 = getitem_660 = getitem_661 = getitem_662 = getitem_663 = getitem_664 = getitem_665 = getitem_666 = getitem_667 = getitem_668 = getitem_669 = getitem_670 = getitem_671 = getitem_672 = getitem_673 = getitem_674 = getitem_675 = getitem_676 = getitem_677 = getitem_678 = getitem_679 = getitem_680 = getitem_681 = getitem_682 = getitem_683 = getitem_684 = getitem_685 = getitem_686 = getitem_687 = getitem_688 = getitem_689 = getitem_690 = getitem_691 = getitem_692 = getitem_693 = getitem_694 = getitem_695 = getitem_696 = getitem_697 = getitem_698 = getitem_699 = getitem_700 = getitem_701 = getitem_702 = getitem_703 = getitem_704 = getitem_705 = getitem_706 = getitem_707 = getitem_708 = getitem_709 = getitem_710 = getitem_711 = getitem_712 = getitem_713 = getitem_714 = getitem_715 = getitem_716 = getitem_717 = getitem_718 = getitem_719 = getitem_720 = getitem_721 = getitem_722 = getitem_723 = getitem_724 = getitem_725 = getitem_726 = getitem_727 = getitem_728 = getitem_729 = getitem_730 = getitem_731 = getitem_732 = getitem_733 = getitem_734 = getitem_735 = getitem_736 = getitem_737 = getitem_738 = getitem_739 = getitem_740 = getitem_741 = getitem_742 = getitem_743 = getitem_744 = getitem_745 = getitem_746 = getitem_747 = getitem_748 = getitem_749 = getitem_750 = getitem_751 = getitem_752 = getitem_753 = getitem_754 = getitem_755 = None
        getitem_864: "f32[]" = _foreach_sub_scalar_1[0]
        getitem_865: "f32[]" = _foreach_sub_scalar_1[1]
        getitem_866: "f32[]" = _foreach_sub_scalar_1[2]
        getitem_867: "f32[]" = _foreach_sub_scalar_1[3]
        getitem_868: "f32[]" = _foreach_sub_scalar_1[4]
        getitem_869: "f32[]" = _foreach_sub_scalar_1[5]
        getitem_870: "f32[]" = _foreach_sub_scalar_1[6]
        getitem_871: "f32[]" = _foreach_sub_scalar_1[7]
        getitem_872: "f32[]" = _foreach_sub_scalar_1[8]
        getitem_873: "f32[]" = _foreach_sub_scalar_1[9]
        getitem_874: "f32[]" = _foreach_sub_scalar_1[10]
        getitem_875: "f32[]" = _foreach_sub_scalar_1[11]
        getitem_876: "f32[]" = _foreach_sub_scalar_1[12]
        getitem_877: "f32[]" = _foreach_sub_scalar_1[13]
        getitem_878: "f32[]" = _foreach_sub_scalar_1[14]
        getitem_879: "f32[]" = _foreach_sub_scalar_1[15]
        getitem_880: "f32[]" = _foreach_sub_scalar_1[16]
        getitem_881: "f32[]" = _foreach_sub_scalar_1[17]
        getitem_882: "f32[]" = _foreach_sub_scalar_1[18]
        getitem_883: "f32[]" = _foreach_sub_scalar_1[19]
        getitem_884: "f32[]" = _foreach_sub_scalar_1[20]
        getitem_885: "f32[]" = _foreach_sub_scalar_1[21]
        getitem_886: "f32[]" = _foreach_sub_scalar_1[22]
        getitem_887: "f32[]" = _foreach_sub_scalar_1[23]
        getitem_888: "f32[]" = _foreach_sub_scalar_1[24]
        getitem_889: "f32[]" = _foreach_sub_scalar_1[25]
        getitem_890: "f32[]" = _foreach_sub_scalar_1[26]
        getitem_891: "f32[]" = _foreach_sub_scalar_1[27]
        getitem_892: "f32[]" = _foreach_sub_scalar_1[28]
        getitem_893: "f32[]" = _foreach_sub_scalar_1[29]
        getitem_894: "f32[]" = _foreach_sub_scalar_1[30]
        getitem_895: "f32[]" = _foreach_sub_scalar_1[31]
        getitem_896: "f32[]" = _foreach_sub_scalar_1[32]
        getitem_897: "f32[]" = _foreach_sub_scalar_1[33]
        getitem_898: "f32[]" = _foreach_sub_scalar_1[34]
        getitem_899: "f32[]" = _foreach_sub_scalar_1[35]
        getitem_900: "f32[]" = _foreach_sub_scalar_1[36]
        getitem_901: "f32[]" = _foreach_sub_scalar_1[37]
        getitem_902: "f32[]" = _foreach_sub_scalar_1[38]
        getitem_903: "f32[]" = _foreach_sub_scalar_1[39]
        getitem_904: "f32[]" = _foreach_sub_scalar_1[40]
        getitem_905: "f32[]" = _foreach_sub_scalar_1[41]
        getitem_906: "f32[]" = _foreach_sub_scalar_1[42]
        getitem_907: "f32[]" = _foreach_sub_scalar_1[43]
        getitem_908: "f32[]" = _foreach_sub_scalar_1[44]
        getitem_909: "f32[]" = _foreach_sub_scalar_1[45]
        getitem_910: "f32[]" = _foreach_sub_scalar_1[46]
        getitem_911: "f32[]" = _foreach_sub_scalar_1[47]
        getitem_912: "f32[]" = _foreach_sub_scalar_1[48]
        getitem_913: "f32[]" = _foreach_sub_scalar_1[49]
        getitem_914: "f32[]" = _foreach_sub_scalar_1[50]
        getitem_915: "f32[]" = _foreach_sub_scalar_1[51]
        getitem_916: "f32[]" = _foreach_sub_scalar_1[52]
        getitem_917: "f32[]" = _foreach_sub_scalar_1[53]
        getitem_918: "f32[]" = _foreach_sub_scalar_1[54]
        getitem_919: "f32[]" = _foreach_sub_scalar_1[55]
        getitem_920: "f32[]" = _foreach_sub_scalar_1[56]
        getitem_921: "f32[]" = _foreach_sub_scalar_1[57]
        getitem_922: "f32[]" = _foreach_sub_scalar_1[58]
        getitem_923: "f32[]" = _foreach_sub_scalar_1[59]
        getitem_924: "f32[]" = _foreach_sub_scalar_1[60]
        getitem_925: "f32[]" = _foreach_sub_scalar_1[61]
        getitem_926: "f32[]" = _foreach_sub_scalar_1[62]
        getitem_927: "f32[]" = _foreach_sub_scalar_1[63]
        getitem_928: "f32[]" = _foreach_sub_scalar_1[64]
        getitem_929: "f32[]" = _foreach_sub_scalar_1[65]
        getitem_930: "f32[]" = _foreach_sub_scalar_1[66]
        getitem_931: "f32[]" = _foreach_sub_scalar_1[67]
        getitem_932: "f32[]" = _foreach_sub_scalar_1[68]
        getitem_933: "f32[]" = _foreach_sub_scalar_1[69]
        getitem_934: "f32[]" = _foreach_sub_scalar_1[70]
        getitem_935: "f32[]" = _foreach_sub_scalar_1[71]
        getitem_936: "f32[]" = _foreach_sub_scalar_1[72]
        getitem_937: "f32[]" = _foreach_sub_scalar_1[73]
        getitem_938: "f32[]" = _foreach_sub_scalar_1[74]
        getitem_939: "f32[]" = _foreach_sub_scalar_1[75]
        getitem_940: "f32[]" = _foreach_sub_scalar_1[76]
        getitem_941: "f32[]" = _foreach_sub_scalar_1[77]
        getitem_942: "f32[]" = _foreach_sub_scalar_1[78]
        getitem_943: "f32[]" = _foreach_sub_scalar_1[79]
        getitem_944: "f32[]" = _foreach_sub_scalar_1[80]
        getitem_945: "f32[]" = _foreach_sub_scalar_1[81]
        getitem_946: "f32[]" = _foreach_sub_scalar_1[82]
        getitem_947: "f32[]" = _foreach_sub_scalar_1[83]
        getitem_948: "f32[]" = _foreach_sub_scalar_1[84]
        getitem_949: "f32[]" = _foreach_sub_scalar_1[85]
        getitem_950: "f32[]" = _foreach_sub_scalar_1[86]
        getitem_951: "f32[]" = _foreach_sub_scalar_1[87]
        getitem_952: "f32[]" = _foreach_sub_scalar_1[88]
        getitem_953: "f32[]" = _foreach_sub_scalar_1[89]
        getitem_954: "f32[]" = _foreach_sub_scalar_1[90]
        getitem_955: "f32[]" = _foreach_sub_scalar_1[91]
        getitem_956: "f32[]" = _foreach_sub_scalar_1[92]
        getitem_957: "f32[]" = _foreach_sub_scalar_1[93]
        getitem_958: "f32[]" = _foreach_sub_scalar_1[94]
        getitem_959: "f32[]" = _foreach_sub_scalar_1[95]
        getitem_960: "f32[]" = _foreach_sub_scalar_1[96]
        getitem_961: "f32[]" = _foreach_sub_scalar_1[97]
        getitem_962: "f32[]" = _foreach_sub_scalar_1[98]
        getitem_963: "f32[]" = _foreach_sub_scalar_1[99]
        getitem_964: "f32[]" = _foreach_sub_scalar_1[100]
        getitem_965: "f32[]" = _foreach_sub_scalar_1[101]
        getitem_966: "f32[]" = _foreach_sub_scalar_1[102]
        getitem_967: "f32[]" = _foreach_sub_scalar_1[103]
        getitem_968: "f32[]" = _foreach_sub_scalar_1[104]
        getitem_969: "f32[]" = _foreach_sub_scalar_1[105]
        getitem_970: "f32[]" = _foreach_sub_scalar_1[106]
        getitem_971: "f32[]" = _foreach_sub_scalar_1[107];  _foreach_sub_scalar_1 = None
        _foreach_neg_default = torch.ops.aten._foreach_neg.default([getitem_864, getitem_865, getitem_866, getitem_867, getitem_868, getitem_869, getitem_870, getitem_871, getitem_872, getitem_873, getitem_874, getitem_875, getitem_876, getitem_877, getitem_878, getitem_879, getitem_880, getitem_881, getitem_882, getitem_883, getitem_884, getitem_885, getitem_886, getitem_887, getitem_888, getitem_889, getitem_890, getitem_891, getitem_892, getitem_893, getitem_894, getitem_895, getitem_896, getitem_897, getitem_898, getitem_899, getitem_900, getitem_901, getitem_902, getitem_903, getitem_904, getitem_905, getitem_906, getitem_907, getitem_908, getitem_909, getitem_910, getitem_911, getitem_912, getitem_913, getitem_914, getitem_915, getitem_916, getitem_917, getitem_918, getitem_919, getitem_920, getitem_921, getitem_922, getitem_923, getitem_924, getitem_925, getitem_926, getitem_927, getitem_928, getitem_929, getitem_930, getitem_931, getitem_932, getitem_933, getitem_934, getitem_935, getitem_936, getitem_937, getitem_938, getitem_939, getitem_940, getitem_941, getitem_942, getitem_943, getitem_944, getitem_945, getitem_946, getitem_947, getitem_948, getitem_949, getitem_950, getitem_951, getitem_952, getitem_953, getitem_954, getitem_955, getitem_956, getitem_957, getitem_958, getitem_959, getitem_960, getitem_961, getitem_962, getitem_963, getitem_964, getitem_965, getitem_966, getitem_967, getitem_968, getitem_969, getitem_970, getitem_971]);  getitem_864 = getitem_865 = getitem_866 = getitem_867 = getitem_868 = getitem_869 = getitem_870 = getitem_871 = getitem_872 = getitem_873 = getitem_874 = getitem_875 = getitem_876 = getitem_877 = getitem_878 = getitem_879 = getitem_880 = getitem_881 = getitem_882 = getitem_883 = getitem_884 = getitem_885 = getitem_886 = getitem_887 = getitem_888 = getitem_889 = getitem_890 = getitem_891 = getitem_892 = getitem_893 = getitem_894 = getitem_895 = getitem_896 = getitem_897 = getitem_898 = getitem_899 = getitem_900 = getitem_901 = getitem_902 = getitem_903 = getitem_904 = getitem_905 = getitem_906 = getitem_907 = getitem_908 = getitem_909 = getitem_910 = getitem_911 = getitem_912 = getitem_913 = getitem_914 = getitem_915 = getitem_916 = getitem_917 = getitem_918 = getitem_919 = getitem_920 = getitem_921 = getitem_922 = getitem_923 = getitem_924 = getitem_925 = getitem_926 = getitem_927 = getitem_928 = getitem_929 = getitem_930 = getitem_931 = getitem_932 = getitem_933 = getitem_934 = getitem_935 = getitem_936 = getitem_937 = getitem_938 = getitem_939 = getitem_940 = getitem_941 = getitem_942 = getitem_943 = getitem_944 = getitem_945 = getitem_946 = getitem_947 = getitem_948 = getitem_949 = getitem_950 = getitem_951 = getitem_952 = getitem_953 = getitem_954 = getitem_955 = getitem_956 = getitem_957 = getitem_958 = getitem_959 = getitem_960 = getitem_961 = getitem_962 = getitem_963 = getitem_964 = getitem_965 = getitem_966 = getitem_967 = getitem_968 = getitem_969 = getitem_970 = getitem_971 = None
        getitem_972: "f32[]" = _foreach_neg_default[0]
        getitem_973: "f32[]" = _foreach_neg_default[1]
        getitem_974: "f32[]" = _foreach_neg_default[2]
        getitem_975: "f32[]" = _foreach_neg_default[3]
        getitem_976: "f32[]" = _foreach_neg_default[4]
        getitem_977: "f32[]" = _foreach_neg_default[5]
        getitem_978: "f32[]" = _foreach_neg_default[6]
        getitem_979: "f32[]" = _foreach_neg_default[7]
        getitem_980: "f32[]" = _foreach_neg_default[8]
        getitem_981: "f32[]" = _foreach_neg_default[9]
        getitem_982: "f32[]" = _foreach_neg_default[10]
        getitem_983: "f32[]" = _foreach_neg_default[11]
        getitem_984: "f32[]" = _foreach_neg_default[12]
        getitem_985: "f32[]" = _foreach_neg_default[13]
        getitem_986: "f32[]" = _foreach_neg_default[14]
        getitem_987: "f32[]" = _foreach_neg_default[15]
        getitem_988: "f32[]" = _foreach_neg_default[16]
        getitem_989: "f32[]" = _foreach_neg_default[17]
        getitem_990: "f32[]" = _foreach_neg_default[18]
        getitem_991: "f32[]" = _foreach_neg_default[19]
        getitem_992: "f32[]" = _foreach_neg_default[20]
        getitem_993: "f32[]" = _foreach_neg_default[21]
        getitem_994: "f32[]" = _foreach_neg_default[22]
        getitem_995: "f32[]" = _foreach_neg_default[23]
        getitem_996: "f32[]" = _foreach_neg_default[24]
        getitem_997: "f32[]" = _foreach_neg_default[25]
        getitem_998: "f32[]" = _foreach_neg_default[26]
        getitem_999: "f32[]" = _foreach_neg_default[27]
        getitem_1000: "f32[]" = _foreach_neg_default[28]
        getitem_1001: "f32[]" = _foreach_neg_default[29]
        getitem_1002: "f32[]" = _foreach_neg_default[30]
        getitem_1003: "f32[]" = _foreach_neg_default[31]
        getitem_1004: "f32[]" = _foreach_neg_default[32]
        getitem_1005: "f32[]" = _foreach_neg_default[33]
        getitem_1006: "f32[]" = _foreach_neg_default[34]
        getitem_1007: "f32[]" = _foreach_neg_default[35]
        getitem_1008: "f32[]" = _foreach_neg_default[36]
        getitem_1009: "f32[]" = _foreach_neg_default[37]
        getitem_1010: "f32[]" = _foreach_neg_default[38]
        getitem_1011: "f32[]" = _foreach_neg_default[39]
        getitem_1012: "f32[]" = _foreach_neg_default[40]
        getitem_1013: "f32[]" = _foreach_neg_default[41]
        getitem_1014: "f32[]" = _foreach_neg_default[42]
        getitem_1015: "f32[]" = _foreach_neg_default[43]
        getitem_1016: "f32[]" = _foreach_neg_default[44]
        getitem_1017: "f32[]" = _foreach_neg_default[45]
        getitem_1018: "f32[]" = _foreach_neg_default[46]
        getitem_1019: "f32[]" = _foreach_neg_default[47]
        getitem_1020: "f32[]" = _foreach_neg_default[48]
        getitem_1021: "f32[]" = _foreach_neg_default[49]
        getitem_1022: "f32[]" = _foreach_neg_default[50]
        getitem_1023: "f32[]" = _foreach_neg_default[51]
        getitem_1024: "f32[]" = _foreach_neg_default[52]
        getitem_1025: "f32[]" = _foreach_neg_default[53]
        getitem_1026: "f32[]" = _foreach_neg_default[54]
        getitem_1027: "f32[]" = _foreach_neg_default[55]
        getitem_1028: "f32[]" = _foreach_neg_default[56]
        getitem_1029: "f32[]" = _foreach_neg_default[57]
        getitem_1030: "f32[]" = _foreach_neg_default[58]
        getitem_1031: "f32[]" = _foreach_neg_default[59]
        getitem_1032: "f32[]" = _foreach_neg_default[60]
        getitem_1033: "f32[]" = _foreach_neg_default[61]
        getitem_1034: "f32[]" = _foreach_neg_default[62]
        getitem_1035: "f32[]" = _foreach_neg_default[63]
        getitem_1036: "f32[]" = _foreach_neg_default[64]
        getitem_1037: "f32[]" = _foreach_neg_default[65]
        getitem_1038: "f32[]" = _foreach_neg_default[66]
        getitem_1039: "f32[]" = _foreach_neg_default[67]
        getitem_1040: "f32[]" = _foreach_neg_default[68]
        getitem_1041: "f32[]" = _foreach_neg_default[69]
        getitem_1042: "f32[]" = _foreach_neg_default[70]
        getitem_1043: "f32[]" = _foreach_neg_default[71]
        getitem_1044: "f32[]" = _foreach_neg_default[72]
        getitem_1045: "f32[]" = _foreach_neg_default[73]
        getitem_1046: "f32[]" = _foreach_neg_default[74]
        getitem_1047: "f32[]" = _foreach_neg_default[75]
        getitem_1048: "f32[]" = _foreach_neg_default[76]
        getitem_1049: "f32[]" = _foreach_neg_default[77]
        getitem_1050: "f32[]" = _foreach_neg_default[78]
        getitem_1051: "f32[]" = _foreach_neg_default[79]
        getitem_1052: "f32[]" = _foreach_neg_default[80]
        getitem_1053: "f32[]" = _foreach_neg_default[81]
        getitem_1054: "f32[]" = _foreach_neg_default[82]
        getitem_1055: "f32[]" = _foreach_neg_default[83]
        getitem_1056: "f32[]" = _foreach_neg_default[84]
        getitem_1057: "f32[]" = _foreach_neg_default[85]
        getitem_1058: "f32[]" = _foreach_neg_default[86]
        getitem_1059: "f32[]" = _foreach_neg_default[87]
        getitem_1060: "f32[]" = _foreach_neg_default[88]
        getitem_1061: "f32[]" = _foreach_neg_default[89]
        getitem_1062: "f32[]" = _foreach_neg_default[90]
        getitem_1063: "f32[]" = _foreach_neg_default[91]
        getitem_1064: "f32[]" = _foreach_neg_default[92]
        getitem_1065: "f32[]" = _foreach_neg_default[93]
        getitem_1066: "f32[]" = _foreach_neg_default[94]
        getitem_1067: "f32[]" = _foreach_neg_default[95]
        getitem_1068: "f32[]" = _foreach_neg_default[96]
        getitem_1069: "f32[]" = _foreach_neg_default[97]
        getitem_1070: "f32[]" = _foreach_neg_default[98]
        getitem_1071: "f32[]" = _foreach_neg_default[99]
        getitem_1072: "f32[]" = _foreach_neg_default[100]
        getitem_1073: "f32[]" = _foreach_neg_default[101]
        getitem_1074: "f32[]" = _foreach_neg_default[102]
        getitem_1075: "f32[]" = _foreach_neg_default[103]
        getitem_1076: "f32[]" = _foreach_neg_default[104]
        getitem_1077: "f32[]" = _foreach_neg_default[105]
        getitem_1078: "f32[]" = _foreach_neg_default[106]
        getitem_1079: "f32[]" = _foreach_neg_default[107];  _foreach_neg_default = None
        _foreach_div_scalar = torch.ops.aten._foreach_div.Scalar([getitem_756, getitem_757, getitem_758, getitem_759, getitem_760, getitem_761, getitem_762, getitem_763, getitem_764, getitem_765, getitem_766, getitem_767, getitem_768, getitem_769, getitem_770, getitem_771, getitem_772, getitem_773, getitem_774, getitem_775, getitem_776, getitem_777, getitem_778, getitem_779, getitem_780, getitem_781, getitem_782, getitem_783, getitem_784, getitem_785, getitem_786, getitem_787, getitem_788, getitem_789, getitem_790, getitem_791, getitem_792, getitem_793, getitem_794, getitem_795, getitem_796, getitem_797, getitem_798, getitem_799, getitem_800, getitem_801, getitem_802, getitem_803, getitem_804, getitem_805, getitem_806, getitem_807, getitem_808, getitem_809, getitem_810, getitem_811, getitem_812, getitem_813, getitem_814, getitem_815, getitem_816, getitem_817, getitem_818, getitem_819, getitem_820, getitem_821, getitem_822, getitem_823, getitem_824, getitem_825, getitem_826, getitem_827, getitem_828, getitem_829, getitem_830, getitem_831, getitem_832, getitem_833, getitem_834, getitem_835, getitem_836, getitem_837, getitem_838, getitem_839, getitem_840, getitem_841, getitem_842, getitem_843, getitem_844, getitem_845, getitem_846, getitem_847, getitem_848, getitem_849, getitem_850, getitem_851, getitem_852, getitem_853, getitem_854, getitem_855, getitem_856, getitem_857, getitem_858, getitem_859, getitem_860, getitem_861, getitem_862, getitem_863], 0.01);  getitem_756 = getitem_757 = getitem_758 = getitem_759 = getitem_760 = getitem_761 = getitem_762 = getitem_763 = getitem_764 = getitem_765 = getitem_766 = getitem_767 = getitem_768 = getitem_769 = getitem_770 = getitem_771 = getitem_772 = getitem_773 = getitem_774 = getitem_775 = getitem_776 = getitem_777 = getitem_778 = getitem_779 = getitem_780 = getitem_781 = getitem_782 = getitem_783 = getitem_784 = getitem_785 = getitem_786 = getitem_787 = getitem_788 = getitem_789 = getitem_790 = getitem_791 = getitem_792 = getitem_793 = getitem_794 = getitem_795 = getitem_796 = getitem_797 = getitem_798 = getitem_799 = getitem_800 = getitem_801 = getitem_802 = getitem_803 = getitem_804 = getitem_805 = getitem_806 = getitem_807 = getitem_808 = getitem_809 = getitem_810 = getitem_811 = getitem_812 = getitem_813 = getitem_814 = getitem_815 = getitem_816 = getitem_817 = getitem_818 = getitem_819 = getitem_820 = getitem_821 = getitem_822 = getitem_823 = getitem_824 = getitem_825 = getitem_826 = getitem_827 = getitem_828 = getitem_829 = getitem_830 = getitem_831 = getitem_832 = getitem_833 = getitem_834 = getitem_835 = getitem_836 = getitem_837 = getitem_838 = getitem_839 = getitem_840 = getitem_841 = getitem_842 = getitem_843 = getitem_844 = getitem_845 = getitem_846 = getitem_847 = getitem_848 = getitem_849 = getitem_850 = getitem_851 = getitem_852 = getitem_853 = getitem_854 = getitem_855 = getitem_856 = getitem_857 = getitem_858 = getitem_859 = getitem_860 = getitem_861 = getitem_862 = getitem_863 = None
        getitem_1080: "f32[]" = _foreach_div_scalar[0]
        getitem_1081: "f32[]" = _foreach_div_scalar[1]
        getitem_1082: "f32[]" = _foreach_div_scalar[2]
        getitem_1083: "f32[]" = _foreach_div_scalar[3]
        getitem_1084: "f32[]" = _foreach_div_scalar[4]
        getitem_1085: "f32[]" = _foreach_div_scalar[5]
        getitem_1086: "f32[]" = _foreach_div_scalar[6]
        getitem_1087: "f32[]" = _foreach_div_scalar[7]
        getitem_1088: "f32[]" = _foreach_div_scalar[8]
        getitem_1089: "f32[]" = _foreach_div_scalar[9]
        getitem_1090: "f32[]" = _foreach_div_scalar[10]
        getitem_1091: "f32[]" = _foreach_div_scalar[11]
        getitem_1092: "f32[]" = _foreach_div_scalar[12]
        getitem_1093: "f32[]" = _foreach_div_scalar[13]
        getitem_1094: "f32[]" = _foreach_div_scalar[14]
        getitem_1095: "f32[]" = _foreach_div_scalar[15]
        getitem_1096: "f32[]" = _foreach_div_scalar[16]
        getitem_1097: "f32[]" = _foreach_div_scalar[17]
        getitem_1098: "f32[]" = _foreach_div_scalar[18]
        getitem_1099: "f32[]" = _foreach_div_scalar[19]
        getitem_1100: "f32[]" = _foreach_div_scalar[20]
        getitem_1101: "f32[]" = _foreach_div_scalar[21]
        getitem_1102: "f32[]" = _foreach_div_scalar[22]
        getitem_1103: "f32[]" = _foreach_div_scalar[23]
        getitem_1104: "f32[]" = _foreach_div_scalar[24]
        getitem_1105: "f32[]" = _foreach_div_scalar[25]
        getitem_1106: "f32[]" = _foreach_div_scalar[26]
        getitem_1107: "f32[]" = _foreach_div_scalar[27]
        getitem_1108: "f32[]" = _foreach_div_scalar[28]
        getitem_1109: "f32[]" = _foreach_div_scalar[29]
        getitem_1110: "f32[]" = _foreach_div_scalar[30]
        getitem_1111: "f32[]" = _foreach_div_scalar[31]
        getitem_1112: "f32[]" = _foreach_div_scalar[32]
        getitem_1113: "f32[]" = _foreach_div_scalar[33]
        getitem_1114: "f32[]" = _foreach_div_scalar[34]
        getitem_1115: "f32[]" = _foreach_div_scalar[35]
        getitem_1116: "f32[]" = _foreach_div_scalar[36]
        getitem_1117: "f32[]" = _foreach_div_scalar[37]
        getitem_1118: "f32[]" = _foreach_div_scalar[38]
        getitem_1119: "f32[]" = _foreach_div_scalar[39]
        getitem_1120: "f32[]" = _foreach_div_scalar[40]
        getitem_1121: "f32[]" = _foreach_div_scalar[41]
        getitem_1122: "f32[]" = _foreach_div_scalar[42]
        getitem_1123: "f32[]" = _foreach_div_scalar[43]
        getitem_1124: "f32[]" = _foreach_div_scalar[44]
        getitem_1125: "f32[]" = _foreach_div_scalar[45]
        getitem_1126: "f32[]" = _foreach_div_scalar[46]
        getitem_1127: "f32[]" = _foreach_div_scalar[47]
        getitem_1128: "f32[]" = _foreach_div_scalar[48]
        getitem_1129: "f32[]" = _foreach_div_scalar[49]
        getitem_1130: "f32[]" = _foreach_div_scalar[50]
        getitem_1131: "f32[]" = _foreach_div_scalar[51]
        getitem_1132: "f32[]" = _foreach_div_scalar[52]
        getitem_1133: "f32[]" = _foreach_div_scalar[53]
        getitem_1134: "f32[]" = _foreach_div_scalar[54]
        getitem_1135: "f32[]" = _foreach_div_scalar[55]
        getitem_1136: "f32[]" = _foreach_div_scalar[56]
        getitem_1137: "f32[]" = _foreach_div_scalar[57]
        getitem_1138: "f32[]" = _foreach_div_scalar[58]
        getitem_1139: "f32[]" = _foreach_div_scalar[59]
        getitem_1140: "f32[]" = _foreach_div_scalar[60]
        getitem_1141: "f32[]" = _foreach_div_scalar[61]
        getitem_1142: "f32[]" = _foreach_div_scalar[62]
        getitem_1143: "f32[]" = _foreach_div_scalar[63]
        getitem_1144: "f32[]" = _foreach_div_scalar[64]
        getitem_1145: "f32[]" = _foreach_div_scalar[65]
        getitem_1146: "f32[]" = _foreach_div_scalar[66]
        getitem_1147: "f32[]" = _foreach_div_scalar[67]
        getitem_1148: "f32[]" = _foreach_div_scalar[68]
        getitem_1149: "f32[]" = _foreach_div_scalar[69]
        getitem_1150: "f32[]" = _foreach_div_scalar[70]
        getitem_1151: "f32[]" = _foreach_div_scalar[71]
        getitem_1152: "f32[]" = _foreach_div_scalar[72]
        getitem_1153: "f32[]" = _foreach_div_scalar[73]
        getitem_1154: "f32[]" = _foreach_div_scalar[74]
        getitem_1155: "f32[]" = _foreach_div_scalar[75]
        getitem_1156: "f32[]" = _foreach_div_scalar[76]
        getitem_1157: "f32[]" = _foreach_div_scalar[77]
        getitem_1158: "f32[]" = _foreach_div_scalar[78]
        getitem_1159: "f32[]" = _foreach_div_scalar[79]
        getitem_1160: "f32[]" = _foreach_div_scalar[80]
        getitem_1161: "f32[]" = _foreach_div_scalar[81]
        getitem_1162: "f32[]" = _foreach_div_scalar[82]
        getitem_1163: "f32[]" = _foreach_div_scalar[83]
        getitem_1164: "f32[]" = _foreach_div_scalar[84]
        getitem_1165: "f32[]" = _foreach_div_scalar[85]
        getitem_1166: "f32[]" = _foreach_div_scalar[86]
        getitem_1167: "f32[]" = _foreach_div_scalar[87]
        getitem_1168: "f32[]" = _foreach_div_scalar[88]
        getitem_1169: "f32[]" = _foreach_div_scalar[89]
        getitem_1170: "f32[]" = _foreach_div_scalar[90]
        getitem_1171: "f32[]" = _foreach_div_scalar[91]
        getitem_1172: "f32[]" = _foreach_div_scalar[92]
        getitem_1173: "f32[]" = _foreach_div_scalar[93]
        getitem_1174: "f32[]" = _foreach_div_scalar[94]
        getitem_1175: "f32[]" = _foreach_div_scalar[95]
        getitem_1176: "f32[]" = _foreach_div_scalar[96]
        getitem_1177: "f32[]" = _foreach_div_scalar[97]
        getitem_1178: "f32[]" = _foreach_div_scalar[98]
        getitem_1179: "f32[]" = _foreach_div_scalar[99]
        getitem_1180: "f32[]" = _foreach_div_scalar[100]
        getitem_1181: "f32[]" = _foreach_div_scalar[101]
        getitem_1182: "f32[]" = _foreach_div_scalar[102]
        getitem_1183: "f32[]" = _foreach_div_scalar[103]
        getitem_1184: "f32[]" = _foreach_div_scalar[104]
        getitem_1185: "f32[]" = _foreach_div_scalar[105]
        getitem_1186: "f32[]" = _foreach_div_scalar[106]
        getitem_1187: "f32[]" = _foreach_div_scalar[107];  _foreach_div_scalar = None
        _foreach_reciprocal_default = torch.ops.aten._foreach_reciprocal.default([getitem_1080, getitem_1081, getitem_1082, getitem_1083, getitem_1084, getitem_1085, getitem_1086, getitem_1087, getitem_1088, getitem_1089, getitem_1090, getitem_1091, getitem_1092, getitem_1093, getitem_1094, getitem_1095, getitem_1096, getitem_1097, getitem_1098, getitem_1099, getitem_1100, getitem_1101, getitem_1102, getitem_1103, getitem_1104, getitem_1105, getitem_1106, getitem_1107, getitem_1108, getitem_1109, getitem_1110, getitem_1111, getitem_1112, getitem_1113, getitem_1114, getitem_1115, getitem_1116, getitem_1117, getitem_1118, getitem_1119, getitem_1120, getitem_1121, getitem_1122, getitem_1123, getitem_1124, getitem_1125, getitem_1126, getitem_1127, getitem_1128, getitem_1129, getitem_1130, getitem_1131, getitem_1132, getitem_1133, getitem_1134, getitem_1135, getitem_1136, getitem_1137, getitem_1138, getitem_1139, getitem_1140, getitem_1141, getitem_1142, getitem_1143, getitem_1144, getitem_1145, getitem_1146, getitem_1147, getitem_1148, getitem_1149, getitem_1150, getitem_1151, getitem_1152, getitem_1153, getitem_1154, getitem_1155, getitem_1156, getitem_1157, getitem_1158, getitem_1159, getitem_1160, getitem_1161, getitem_1162, getitem_1163, getitem_1164, getitem_1165, getitem_1166, getitem_1167, getitem_1168, getitem_1169, getitem_1170, getitem_1171, getitem_1172, getitem_1173, getitem_1174, getitem_1175, getitem_1176, getitem_1177, getitem_1178, getitem_1179, getitem_1180, getitem_1181, getitem_1182, getitem_1183, getitem_1184, getitem_1185, getitem_1186, getitem_1187]);  getitem_1080 = getitem_1081 = getitem_1082 = getitem_1083 = getitem_1084 = getitem_1085 = getitem_1086 = getitem_1087 = getitem_1088 = getitem_1089 = getitem_1090 = getitem_1091 = getitem_1092 = getitem_1093 = getitem_1094 = getitem_1095 = getitem_1096 = getitem_1097 = getitem_1098 = getitem_1099 = getitem_1100 = getitem_1101 = getitem_1102 = getitem_1103 = getitem_1104 = getitem_1105 = getitem_1106 = getitem_1107 = getitem_1108 = getitem_1109 = getitem_1110 = getitem_1111 = getitem_1112 = getitem_1113 = getitem_1114 = getitem_1115 = getitem_1116 = getitem_1117 = getitem_1118 = getitem_1119 = getitem_1120 = getitem_1121 = getitem_1122 = getitem_1123 = getitem_1124 = getitem_1125 = getitem_1126 = getitem_1127 = getitem_1128 = getitem_1129 = getitem_1130 = getitem_1131 = getitem_1132 = getitem_1133 = getitem_1134 = getitem_1135 = getitem_1136 = getitem_1137 = getitem_1138 = getitem_1139 = getitem_1140 = getitem_1141 = getitem_1142 = getitem_1143 = getitem_1144 = getitem_1145 = getitem_1146 = getitem_1147 = getitem_1148 = getitem_1149 = getitem_1150 = getitem_1151 = getitem_1152 = getitem_1153 = getitem_1154 = getitem_1155 = getitem_1156 = getitem_1157 = getitem_1158 = getitem_1159 = getitem_1160 = getitem_1161 = getitem_1162 = getitem_1163 = getitem_1164 = getitem_1165 = getitem_1166 = getitem_1167 = getitem_1168 = getitem_1169 = getitem_1170 = getitem_1171 = getitem_1172 = getitem_1173 = getitem_1174 = getitem_1175 = getitem_1176 = getitem_1177 = getitem_1178 = getitem_1179 = getitem_1180 = getitem_1181 = getitem_1182 = getitem_1183 = getitem_1184 = getitem_1185 = getitem_1186 = getitem_1187 = None
        getitem_1188: "f32[]" = _foreach_reciprocal_default[0]
        getitem_1189: "f32[]" = _foreach_reciprocal_default[1]
        getitem_1190: "f32[]" = _foreach_reciprocal_default[2]
        getitem_1191: "f32[]" = _foreach_reciprocal_default[3]
        getitem_1192: "f32[]" = _foreach_reciprocal_default[4]
        getitem_1193: "f32[]" = _foreach_reciprocal_default[5]
        getitem_1194: "f32[]" = _foreach_reciprocal_default[6]
        getitem_1195: "f32[]" = _foreach_reciprocal_default[7]
        getitem_1196: "f32[]" = _foreach_reciprocal_default[8]
        getitem_1197: "f32[]" = _foreach_reciprocal_default[9]
        getitem_1198: "f32[]" = _foreach_reciprocal_default[10]
        getitem_1199: "f32[]" = _foreach_reciprocal_default[11]
        getitem_1200: "f32[]" = _foreach_reciprocal_default[12]
        getitem_1201: "f32[]" = _foreach_reciprocal_default[13]
        getitem_1202: "f32[]" = _foreach_reciprocal_default[14]
        getitem_1203: "f32[]" = _foreach_reciprocal_default[15]
        getitem_1204: "f32[]" = _foreach_reciprocal_default[16]
        getitem_1205: "f32[]" = _foreach_reciprocal_default[17]
        getitem_1206: "f32[]" = _foreach_reciprocal_default[18]
        getitem_1207: "f32[]" = _foreach_reciprocal_default[19]
        getitem_1208: "f32[]" = _foreach_reciprocal_default[20]
        getitem_1209: "f32[]" = _foreach_reciprocal_default[21]
        getitem_1210: "f32[]" = _foreach_reciprocal_default[22]
        getitem_1211: "f32[]" = _foreach_reciprocal_default[23]
        getitem_1212: "f32[]" = _foreach_reciprocal_default[24]
        getitem_1213: "f32[]" = _foreach_reciprocal_default[25]
        getitem_1214: "f32[]" = _foreach_reciprocal_default[26]
        getitem_1215: "f32[]" = _foreach_reciprocal_default[27]
        getitem_1216: "f32[]" = _foreach_reciprocal_default[28]
        getitem_1217: "f32[]" = _foreach_reciprocal_default[29]
        getitem_1218: "f32[]" = _foreach_reciprocal_default[30]
        getitem_1219: "f32[]" = _foreach_reciprocal_default[31]
        getitem_1220: "f32[]" = _foreach_reciprocal_default[32]
        getitem_1221: "f32[]" = _foreach_reciprocal_default[33]
        getitem_1222: "f32[]" = _foreach_reciprocal_default[34]
        getitem_1223: "f32[]" = _foreach_reciprocal_default[35]
        getitem_1224: "f32[]" = _foreach_reciprocal_default[36]
        getitem_1225: "f32[]" = _foreach_reciprocal_default[37]
        getitem_1226: "f32[]" = _foreach_reciprocal_default[38]
        getitem_1227: "f32[]" = _foreach_reciprocal_default[39]
        getitem_1228: "f32[]" = _foreach_reciprocal_default[40]
        getitem_1229: "f32[]" = _foreach_reciprocal_default[41]
        getitem_1230: "f32[]" = _foreach_reciprocal_default[42]
        getitem_1231: "f32[]" = _foreach_reciprocal_default[43]
        getitem_1232: "f32[]" = _foreach_reciprocal_default[44]
        getitem_1233: "f32[]" = _foreach_reciprocal_default[45]
        getitem_1234: "f32[]" = _foreach_reciprocal_default[46]
        getitem_1235: "f32[]" = _foreach_reciprocal_default[47]
        getitem_1236: "f32[]" = _foreach_reciprocal_default[48]
        getitem_1237: "f32[]" = _foreach_reciprocal_default[49]
        getitem_1238: "f32[]" = _foreach_reciprocal_default[50]
        getitem_1239: "f32[]" = _foreach_reciprocal_default[51]
        getitem_1240: "f32[]" = _foreach_reciprocal_default[52]
        getitem_1241: "f32[]" = _foreach_reciprocal_default[53]
        getitem_1242: "f32[]" = _foreach_reciprocal_default[54]
        getitem_1243: "f32[]" = _foreach_reciprocal_default[55]
        getitem_1244: "f32[]" = _foreach_reciprocal_default[56]
        getitem_1245: "f32[]" = _foreach_reciprocal_default[57]
        getitem_1246: "f32[]" = _foreach_reciprocal_default[58]
        getitem_1247: "f32[]" = _foreach_reciprocal_default[59]
        getitem_1248: "f32[]" = _foreach_reciprocal_default[60]
        getitem_1249: "f32[]" = _foreach_reciprocal_default[61]
        getitem_1250: "f32[]" = _foreach_reciprocal_default[62]
        getitem_1251: "f32[]" = _foreach_reciprocal_default[63]
        getitem_1252: "f32[]" = _foreach_reciprocal_default[64]
        getitem_1253: "f32[]" = _foreach_reciprocal_default[65]
        getitem_1254: "f32[]" = _foreach_reciprocal_default[66]
        getitem_1255: "f32[]" = _foreach_reciprocal_default[67]
        getitem_1256: "f32[]" = _foreach_reciprocal_default[68]
        getitem_1257: "f32[]" = _foreach_reciprocal_default[69]
        getitem_1258: "f32[]" = _foreach_reciprocal_default[70]
        getitem_1259: "f32[]" = _foreach_reciprocal_default[71]
        getitem_1260: "f32[]" = _foreach_reciprocal_default[72]
        getitem_1261: "f32[]" = _foreach_reciprocal_default[73]
        getitem_1262: "f32[]" = _foreach_reciprocal_default[74]
        getitem_1263: "f32[]" = _foreach_reciprocal_default[75]
        getitem_1264: "f32[]" = _foreach_reciprocal_default[76]
        getitem_1265: "f32[]" = _foreach_reciprocal_default[77]
        getitem_1266: "f32[]" = _foreach_reciprocal_default[78]
        getitem_1267: "f32[]" = _foreach_reciprocal_default[79]
        getitem_1268: "f32[]" = _foreach_reciprocal_default[80]
        getitem_1269: "f32[]" = _foreach_reciprocal_default[81]
        getitem_1270: "f32[]" = _foreach_reciprocal_default[82]
        getitem_1271: "f32[]" = _foreach_reciprocal_default[83]
        getitem_1272: "f32[]" = _foreach_reciprocal_default[84]
        getitem_1273: "f32[]" = _foreach_reciprocal_default[85]
        getitem_1274: "f32[]" = _foreach_reciprocal_default[86]
        getitem_1275: "f32[]" = _foreach_reciprocal_default[87]
        getitem_1276: "f32[]" = _foreach_reciprocal_default[88]
        getitem_1277: "f32[]" = _foreach_reciprocal_default[89]
        getitem_1278: "f32[]" = _foreach_reciprocal_default[90]
        getitem_1279: "f32[]" = _foreach_reciprocal_default[91]
        getitem_1280: "f32[]" = _foreach_reciprocal_default[92]
        getitem_1281: "f32[]" = _foreach_reciprocal_default[93]
        getitem_1282: "f32[]" = _foreach_reciprocal_default[94]
        getitem_1283: "f32[]" = _foreach_reciprocal_default[95]
        getitem_1284: "f32[]" = _foreach_reciprocal_default[96]
        getitem_1285: "f32[]" = _foreach_reciprocal_default[97]
        getitem_1286: "f32[]" = _foreach_reciprocal_default[98]
        getitem_1287: "f32[]" = _foreach_reciprocal_default[99]
        getitem_1288: "f32[]" = _foreach_reciprocal_default[100]
        getitem_1289: "f32[]" = _foreach_reciprocal_default[101]
        getitem_1290: "f32[]" = _foreach_reciprocal_default[102]
        getitem_1291: "f32[]" = _foreach_reciprocal_default[103]
        getitem_1292: "f32[]" = _foreach_reciprocal_default[104]
        getitem_1293: "f32[]" = _foreach_reciprocal_default[105]
        getitem_1294: "f32[]" = _foreach_reciprocal_default[106]
        getitem_1295: "f32[]" = _foreach_reciprocal_default[107];  _foreach_reciprocal_default = None
        _foreach_sqrt_default = torch.ops.aten._foreach_sqrt.default([getitem_972, getitem_973, getitem_974, getitem_975, getitem_976, getitem_977, getitem_978, getitem_979, getitem_980, getitem_981, getitem_982, getitem_983, getitem_984, getitem_985, getitem_986, getitem_987, getitem_988, getitem_989, getitem_990, getitem_991, getitem_992, getitem_993, getitem_994, getitem_995, getitem_996, getitem_997, getitem_998, getitem_999, getitem_1000, getitem_1001, getitem_1002, getitem_1003, getitem_1004, getitem_1005, getitem_1006, getitem_1007, getitem_1008, getitem_1009, getitem_1010, getitem_1011, getitem_1012, getitem_1013, getitem_1014, getitem_1015, getitem_1016, getitem_1017, getitem_1018, getitem_1019, getitem_1020, getitem_1021, getitem_1022, getitem_1023, getitem_1024, getitem_1025, getitem_1026, getitem_1027, getitem_1028, getitem_1029, getitem_1030, getitem_1031, getitem_1032, getitem_1033, getitem_1034, getitem_1035, getitem_1036, getitem_1037, getitem_1038, getitem_1039, getitem_1040, getitem_1041, getitem_1042, getitem_1043, getitem_1044, getitem_1045, getitem_1046, getitem_1047, getitem_1048, getitem_1049, getitem_1050, getitem_1051, getitem_1052, getitem_1053, getitem_1054, getitem_1055, getitem_1056, getitem_1057, getitem_1058, getitem_1059, getitem_1060, getitem_1061, getitem_1062, getitem_1063, getitem_1064, getitem_1065, getitem_1066, getitem_1067, getitem_1068, getitem_1069, getitem_1070, getitem_1071, getitem_1072, getitem_1073, getitem_1074, getitem_1075, getitem_1076, getitem_1077, getitem_1078, getitem_1079]);  getitem_972 = getitem_973 = getitem_974 = getitem_975 = getitem_976 = getitem_977 = getitem_978 = getitem_979 = getitem_980 = getitem_981 = getitem_982 = getitem_983 = getitem_984 = getitem_985 = getitem_986 = getitem_987 = getitem_988 = getitem_989 = getitem_990 = getitem_991 = getitem_992 = getitem_993 = getitem_994 = getitem_995 = getitem_996 = getitem_997 = getitem_998 = getitem_999 = getitem_1000 = getitem_1001 = getitem_1002 = getitem_1003 = getitem_1004 = getitem_1005 = getitem_1006 = getitem_1007 = getitem_1008 = getitem_1009 = getitem_1010 = getitem_1011 = getitem_1012 = getitem_1013 = getitem_1014 = getitem_1015 = getitem_1016 = getitem_1017 = getitem_1018 = getitem_1019 = getitem_1020 = getitem_1021 = getitem_1022 = getitem_1023 = getitem_1024 = getitem_1025 = getitem_1026 = getitem_1027 = getitem_1028 = getitem_1029 = getitem_1030 = getitem_1031 = getitem_1032 = getitem_1033 = getitem_1034 = getitem_1035 = getitem_1036 = getitem_1037 = getitem_1038 = getitem_1039 = getitem_1040 = getitem_1041 = getitem_1042 = getitem_1043 = getitem_1044 = getitem_1045 = getitem_1046 = getitem_1047 = getitem_1048 = getitem_1049 = getitem_1050 = getitem_1051 = getitem_1052 = getitem_1053 = getitem_1054 = getitem_1055 = getitem_1056 = getitem_1057 = getitem_1058 = getitem_1059 = getitem_1060 = getitem_1061 = getitem_1062 = getitem_1063 = getitem_1064 = getitem_1065 = getitem_1066 = getitem_1067 = getitem_1068 = getitem_1069 = getitem_1070 = getitem_1071 = getitem_1072 = getitem_1073 = getitem_1074 = getitem_1075 = getitem_1076 = getitem_1077 = getitem_1078 = getitem_1079 = None
        getitem_1296: "f32[]" = _foreach_sqrt_default[0]
        getitem_1297: "f32[]" = _foreach_sqrt_default[1]
        getitem_1298: "f32[]" = _foreach_sqrt_default[2]
        getitem_1299: "f32[]" = _foreach_sqrt_default[3]
        getitem_1300: "f32[]" = _foreach_sqrt_default[4]
        getitem_1301: "f32[]" = _foreach_sqrt_default[5]
        getitem_1302: "f32[]" = _foreach_sqrt_default[6]
        getitem_1303: "f32[]" = _foreach_sqrt_default[7]
        getitem_1304: "f32[]" = _foreach_sqrt_default[8]
        getitem_1305: "f32[]" = _foreach_sqrt_default[9]
        getitem_1306: "f32[]" = _foreach_sqrt_default[10]
        getitem_1307: "f32[]" = _foreach_sqrt_default[11]
        getitem_1308: "f32[]" = _foreach_sqrt_default[12]
        getitem_1309: "f32[]" = _foreach_sqrt_default[13]
        getitem_1310: "f32[]" = _foreach_sqrt_default[14]
        getitem_1311: "f32[]" = _foreach_sqrt_default[15]
        getitem_1312: "f32[]" = _foreach_sqrt_default[16]
        getitem_1313: "f32[]" = _foreach_sqrt_default[17]
        getitem_1314: "f32[]" = _foreach_sqrt_default[18]
        getitem_1315: "f32[]" = _foreach_sqrt_default[19]
        getitem_1316: "f32[]" = _foreach_sqrt_default[20]
        getitem_1317: "f32[]" = _foreach_sqrt_default[21]
        getitem_1318: "f32[]" = _foreach_sqrt_default[22]
        getitem_1319: "f32[]" = _foreach_sqrt_default[23]
        getitem_1320: "f32[]" = _foreach_sqrt_default[24]
        getitem_1321: "f32[]" = _foreach_sqrt_default[25]
        getitem_1322: "f32[]" = _foreach_sqrt_default[26]
        getitem_1323: "f32[]" = _foreach_sqrt_default[27]
        getitem_1324: "f32[]" = _foreach_sqrt_default[28]
        getitem_1325: "f32[]" = _foreach_sqrt_default[29]
        getitem_1326: "f32[]" = _foreach_sqrt_default[30]
        getitem_1327: "f32[]" = _foreach_sqrt_default[31]
        getitem_1328: "f32[]" = _foreach_sqrt_default[32]
        getitem_1329: "f32[]" = _foreach_sqrt_default[33]
        getitem_1330: "f32[]" = _foreach_sqrt_default[34]
        getitem_1331: "f32[]" = _foreach_sqrt_default[35]
        getitem_1332: "f32[]" = _foreach_sqrt_default[36]
        getitem_1333: "f32[]" = _foreach_sqrt_default[37]
        getitem_1334: "f32[]" = _foreach_sqrt_default[38]
        getitem_1335: "f32[]" = _foreach_sqrt_default[39]
        getitem_1336: "f32[]" = _foreach_sqrt_default[40]
        getitem_1337: "f32[]" = _foreach_sqrt_default[41]
        getitem_1338: "f32[]" = _foreach_sqrt_default[42]
        getitem_1339: "f32[]" = _foreach_sqrt_default[43]
        getitem_1340: "f32[]" = _foreach_sqrt_default[44]
        getitem_1341: "f32[]" = _foreach_sqrt_default[45]
        getitem_1342: "f32[]" = _foreach_sqrt_default[46]
        getitem_1343: "f32[]" = _foreach_sqrt_default[47]
        getitem_1344: "f32[]" = _foreach_sqrt_default[48]
        getitem_1345: "f32[]" = _foreach_sqrt_default[49]
        getitem_1346: "f32[]" = _foreach_sqrt_default[50]
        getitem_1347: "f32[]" = _foreach_sqrt_default[51]
        getitem_1348: "f32[]" = _foreach_sqrt_default[52]
        getitem_1349: "f32[]" = _foreach_sqrt_default[53]
        getitem_1350: "f32[]" = _foreach_sqrt_default[54]
        getitem_1351: "f32[]" = _foreach_sqrt_default[55]
        getitem_1352: "f32[]" = _foreach_sqrt_default[56]
        getitem_1353: "f32[]" = _foreach_sqrt_default[57]
        getitem_1354: "f32[]" = _foreach_sqrt_default[58]
        getitem_1355: "f32[]" = _foreach_sqrt_default[59]
        getitem_1356: "f32[]" = _foreach_sqrt_default[60]
        getitem_1357: "f32[]" = _foreach_sqrt_default[61]
        getitem_1358: "f32[]" = _foreach_sqrt_default[62]
        getitem_1359: "f32[]" = _foreach_sqrt_default[63]
        getitem_1360: "f32[]" = _foreach_sqrt_default[64]
        getitem_1361: "f32[]" = _foreach_sqrt_default[65]
        getitem_1362: "f32[]" = _foreach_sqrt_default[66]
        getitem_1363: "f32[]" = _foreach_sqrt_default[67]
        getitem_1364: "f32[]" = _foreach_sqrt_default[68]
        getitem_1365: "f32[]" = _foreach_sqrt_default[69]
        getitem_1366: "f32[]" = _foreach_sqrt_default[70]
        getitem_1367: "f32[]" = _foreach_sqrt_default[71]
        getitem_1368: "f32[]" = _foreach_sqrt_default[72]
        getitem_1369: "f32[]" = _foreach_sqrt_default[73]
        getitem_1370: "f32[]" = _foreach_sqrt_default[74]
        getitem_1371: "f32[]" = _foreach_sqrt_default[75]
        getitem_1372: "f32[]" = _foreach_sqrt_default[76]
        getitem_1373: "f32[]" = _foreach_sqrt_default[77]
        getitem_1374: "f32[]" = _foreach_sqrt_default[78]
        getitem_1375: "f32[]" = _foreach_sqrt_default[79]
        getitem_1376: "f32[]" = _foreach_sqrt_default[80]
        getitem_1377: "f32[]" = _foreach_sqrt_default[81]
        getitem_1378: "f32[]" = _foreach_sqrt_default[82]
        getitem_1379: "f32[]" = _foreach_sqrt_default[83]
        getitem_1380: "f32[]" = _foreach_sqrt_default[84]
        getitem_1381: "f32[]" = _foreach_sqrt_default[85]
        getitem_1382: "f32[]" = _foreach_sqrt_default[86]
        getitem_1383: "f32[]" = _foreach_sqrt_default[87]
        getitem_1384: "f32[]" = _foreach_sqrt_default[88]
        getitem_1385: "f32[]" = _foreach_sqrt_default[89]
        getitem_1386: "f32[]" = _foreach_sqrt_default[90]
        getitem_1387: "f32[]" = _foreach_sqrt_default[91]
        getitem_1388: "f32[]" = _foreach_sqrt_default[92]
        getitem_1389: "f32[]" = _foreach_sqrt_default[93]
        getitem_1390: "f32[]" = _foreach_sqrt_default[94]
        getitem_1391: "f32[]" = _foreach_sqrt_default[95]
        getitem_1392: "f32[]" = _foreach_sqrt_default[96]
        getitem_1393: "f32[]" = _foreach_sqrt_default[97]
        getitem_1394: "f32[]" = _foreach_sqrt_default[98]
        getitem_1395: "f32[]" = _foreach_sqrt_default[99]
        getitem_1396: "f32[]" = _foreach_sqrt_default[100]
        getitem_1397: "f32[]" = _foreach_sqrt_default[101]
        getitem_1398: "f32[]" = _foreach_sqrt_default[102]
        getitem_1399: "f32[]" = _foreach_sqrt_default[103]
        getitem_1400: "f32[]" = _foreach_sqrt_default[104]
        getitem_1401: "f32[]" = _foreach_sqrt_default[105]
        getitem_1402: "f32[]" = _foreach_sqrt_default[106]
        getitem_1403: "f32[]" = _foreach_sqrt_default[107];  _foreach_sqrt_default = None
        _foreach_sqrt_default_1 = torch.ops.aten._foreach_sqrt.default([getitem_432, getitem_433, getitem_434, getitem_435, getitem_436, getitem_437, getitem_438, getitem_439, getitem_440, getitem_441, getitem_442, getitem_443, getitem_444, getitem_445, getitem_446, getitem_447, getitem_448, getitem_449, getitem_450, getitem_451, getitem_452, getitem_453, getitem_454, getitem_455, getitem_456, getitem_457, getitem_458, getitem_459, getitem_460, getitem_461, getitem_462, getitem_463, getitem_464, getitem_465, getitem_466, getitem_467, getitem_468, getitem_469, getitem_470, getitem_471, getitem_472, getitem_473, getitem_474, getitem_475, getitem_476, getitem_477, getitem_478, getitem_479, getitem_480, getitem_481, getitem_482, getitem_483, getitem_484, getitem_485, getitem_486, getitem_487, getitem_488, getitem_489, getitem_490, getitem_491, getitem_492, getitem_493, getitem_494, getitem_495, getitem_496, getitem_497, getitem_498, getitem_499, getitem_500, getitem_501, getitem_502, getitem_503, getitem_504, getitem_505, getitem_506, getitem_507, getitem_508, getitem_509, getitem_510, getitem_511, getitem_512, getitem_513, getitem_514, getitem_515, getitem_516, getitem_517, getitem_518, getitem_519, getitem_520, getitem_521, getitem_522, getitem_523, getitem_524, getitem_525, getitem_526, getitem_527, getitem_528, getitem_529, getitem_530, getitem_531, getitem_532, getitem_533, getitem_534, getitem_535, getitem_536, getitem_537, getitem_538, getitem_539])
        getitem_1404: "f32[32000, 768]" = _foreach_sqrt_default_1[0]
        getitem_1405: "f32[512, 768]" = _foreach_sqrt_default_1[1]
        getitem_1406: "f32[4, 768]" = _foreach_sqrt_default_1[2]
        getitem_1407: "f32[768]" = _foreach_sqrt_default_1[3]
        getitem_1408: "f32[768]" = _foreach_sqrt_default_1[4]
        getitem_1409: "f32[768, 768]" = _foreach_sqrt_default_1[5]
        getitem_1410: "f32[768]" = _foreach_sqrt_default_1[6]
        getitem_1411: "f32[768]" = _foreach_sqrt_default_1[7]
        getitem_1412: "f32[768]" = _foreach_sqrt_default_1[8]
        getitem_1413: "f32[3072, 768]" = _foreach_sqrt_default_1[9]
        getitem_1414: "f32[3072]" = _foreach_sqrt_default_1[10]
        getitem_1415: "f32[768, 3072]" = _foreach_sqrt_default_1[11]
        getitem_1416: "f32[768]" = _foreach_sqrt_default_1[12]
        getitem_1417: "f32[768]" = _foreach_sqrt_default_1[13]
        getitem_1418: "f32[768]" = _foreach_sqrt_default_1[14]
        getitem_1419: "f32[768]" = _foreach_sqrt_default_1[15]
        getitem_1420: "f32[768]" = _foreach_sqrt_default_1[16]
        getitem_1421: "f32[3072, 768]" = _foreach_sqrt_default_1[17]
        getitem_1422: "f32[3072]" = _foreach_sqrt_default_1[18]
        getitem_1423: "f32[768, 3072]" = _foreach_sqrt_default_1[19]
        getitem_1424: "f32[768]" = _foreach_sqrt_default_1[20]
        getitem_1425: "f32[768]" = _foreach_sqrt_default_1[21]
        getitem_1426: "f32[768]" = _foreach_sqrt_default_1[22]
        getitem_1427: "f32[768]" = _foreach_sqrt_default_1[23]
        getitem_1428: "f32[768]" = _foreach_sqrt_default_1[24]
        getitem_1429: "f32[3072, 768]" = _foreach_sqrt_default_1[25]
        getitem_1430: "f32[3072]" = _foreach_sqrt_default_1[26]
        getitem_1431: "f32[768, 3072]" = _foreach_sqrt_default_1[27]
        getitem_1432: "f32[768]" = _foreach_sqrt_default_1[28]
        getitem_1433: "f32[768]" = _foreach_sqrt_default_1[29]
        getitem_1434: "f32[768]" = _foreach_sqrt_default_1[30]
        getitem_1435: "f32[768]" = _foreach_sqrt_default_1[31]
        getitem_1436: "f32[768]" = _foreach_sqrt_default_1[32]
        getitem_1437: "f32[3072, 768]" = _foreach_sqrt_default_1[33]
        getitem_1438: "f32[3072]" = _foreach_sqrt_default_1[34]
        getitem_1439: "f32[768, 3072]" = _foreach_sqrt_default_1[35]
        getitem_1440: "f32[768]" = _foreach_sqrt_default_1[36]
        getitem_1441: "f32[768]" = _foreach_sqrt_default_1[37]
        getitem_1442: "f32[768]" = _foreach_sqrt_default_1[38]
        getitem_1443: "f32[768]" = _foreach_sqrt_default_1[39]
        getitem_1444: "f32[768]" = _foreach_sqrt_default_1[40]
        getitem_1445: "f32[3072, 768]" = _foreach_sqrt_default_1[41]
        getitem_1446: "f32[3072]" = _foreach_sqrt_default_1[42]
        getitem_1447: "f32[768, 3072]" = _foreach_sqrt_default_1[43]
        getitem_1448: "f32[768]" = _foreach_sqrt_default_1[44]
        getitem_1449: "f32[768]" = _foreach_sqrt_default_1[45]
        getitem_1450: "f32[768]" = _foreach_sqrt_default_1[46]
        getitem_1451: "f32[768]" = _foreach_sqrt_default_1[47]
        getitem_1452: "f32[768]" = _foreach_sqrt_default_1[48]
        getitem_1453: "f32[3072, 768]" = _foreach_sqrt_default_1[49]
        getitem_1454: "f32[3072]" = _foreach_sqrt_default_1[50]
        getitem_1455: "f32[768, 3072]" = _foreach_sqrt_default_1[51]
        getitem_1456: "f32[768]" = _foreach_sqrt_default_1[52]
        getitem_1457: "f32[768]" = _foreach_sqrt_default_1[53]
        getitem_1458: "f32[768]" = _foreach_sqrt_default_1[54]
        getitem_1459: "f32[768]" = _foreach_sqrt_default_1[55]
        getitem_1460: "f32[768]" = _foreach_sqrt_default_1[56]
        getitem_1461: "f32[3072, 768]" = _foreach_sqrt_default_1[57]
        getitem_1462: "f32[3072]" = _foreach_sqrt_default_1[58]
        getitem_1463: "f32[768, 3072]" = _foreach_sqrt_default_1[59]
        getitem_1464: "f32[768]" = _foreach_sqrt_default_1[60]
        getitem_1465: "f32[768]" = _foreach_sqrt_default_1[61]
        getitem_1466: "f32[768]" = _foreach_sqrt_default_1[62]
        getitem_1467: "f32[768]" = _foreach_sqrt_default_1[63]
        getitem_1468: "f32[768]" = _foreach_sqrt_default_1[64]
        getitem_1469: "f32[3072, 768]" = _foreach_sqrt_default_1[65]
        getitem_1470: "f32[3072]" = _foreach_sqrt_default_1[66]
        getitem_1471: "f32[768, 3072]" = _foreach_sqrt_default_1[67]
        getitem_1472: "f32[768]" = _foreach_sqrt_default_1[68]
        getitem_1473: "f32[768]" = _foreach_sqrt_default_1[69]
        getitem_1474: "f32[768]" = _foreach_sqrt_default_1[70]
        getitem_1475: "f32[768]" = _foreach_sqrt_default_1[71]
        getitem_1476: "f32[768]" = _foreach_sqrt_default_1[72]
        getitem_1477: "f32[3072, 768]" = _foreach_sqrt_default_1[73]
        getitem_1478: "f32[3072]" = _foreach_sqrt_default_1[74]
        getitem_1479: "f32[768, 3072]" = _foreach_sqrt_default_1[75]
        getitem_1480: "f32[768]" = _foreach_sqrt_default_1[76]
        getitem_1481: "f32[768]" = _foreach_sqrt_default_1[77]
        getitem_1482: "f32[768]" = _foreach_sqrt_default_1[78]
        getitem_1483: "f32[768]" = _foreach_sqrt_default_1[79]
        getitem_1484: "f32[768]" = _foreach_sqrt_default_1[80]
        getitem_1485: "f32[3072, 768]" = _foreach_sqrt_default_1[81]
        getitem_1486: "f32[3072]" = _foreach_sqrt_default_1[82]
        getitem_1487: "f32[768, 3072]" = _foreach_sqrt_default_1[83]
        getitem_1488: "f32[768]" = _foreach_sqrt_default_1[84]
        getitem_1489: "f32[768]" = _foreach_sqrt_default_1[85]
        getitem_1490: "f32[768]" = _foreach_sqrt_default_1[86]
        getitem_1491: "f32[768]" = _foreach_sqrt_default_1[87]
        getitem_1492: "f32[768]" = _foreach_sqrt_default_1[88]
        getitem_1493: "f32[3072, 768]" = _foreach_sqrt_default_1[89]
        getitem_1494: "f32[3072]" = _foreach_sqrt_default_1[90]
        getitem_1495: "f32[768, 3072]" = _foreach_sqrt_default_1[91]
        getitem_1496: "f32[768]" = _foreach_sqrt_default_1[92]
        getitem_1497: "f32[768]" = _foreach_sqrt_default_1[93]
        getitem_1498: "f32[768]" = _foreach_sqrt_default_1[94]
        getitem_1499: "f32[768]" = _foreach_sqrt_default_1[95]
        getitem_1500: "f32[768]" = _foreach_sqrt_default_1[96]
        getitem_1501: "f32[3072, 768]" = _foreach_sqrt_default_1[97]
        getitem_1502: "f32[3072]" = _foreach_sqrt_default_1[98]
        getitem_1503: "f32[768, 3072]" = _foreach_sqrt_default_1[99]
        getitem_1504: "f32[768]" = _foreach_sqrt_default_1[100]
        getitem_1505: "f32[768]" = _foreach_sqrt_default_1[101]
        getitem_1506: "f32[768]" = _foreach_sqrt_default_1[102]
        getitem_1507: "f32[32000]" = _foreach_sqrt_default_1[103]
        getitem_1508: "f32[768, 768]" = _foreach_sqrt_default_1[104]
        getitem_1509: "f32[768]" = _foreach_sqrt_default_1[105]
        getitem_1510: "f32[768]" = _foreach_sqrt_default_1[106]
        getitem_1511: "f32[768]" = _foreach_sqrt_default_1[107];  _foreach_sqrt_default_1 = None
        _foreach_div_list = torch.ops.aten._foreach_div.List([getitem_1404, getitem_1405, getitem_1406, getitem_1407, getitem_1408, getitem_1409, getitem_1410, getitem_1411, getitem_1412, getitem_1413, getitem_1414, getitem_1415, getitem_1416, getitem_1417, getitem_1418, getitem_1419, getitem_1420, getitem_1421, getitem_1422, getitem_1423, getitem_1424, getitem_1425, getitem_1426, getitem_1427, getitem_1428, getitem_1429, getitem_1430, getitem_1431, getitem_1432, getitem_1433, getitem_1434, getitem_1435, getitem_1436, getitem_1437, getitem_1438, getitem_1439, getitem_1440, getitem_1441, getitem_1442, getitem_1443, getitem_1444, getitem_1445, getitem_1446, getitem_1447, getitem_1448, getitem_1449, getitem_1450, getitem_1451, getitem_1452, getitem_1453, getitem_1454, getitem_1455, getitem_1456, getitem_1457, getitem_1458, getitem_1459, getitem_1460, getitem_1461, getitem_1462, getitem_1463, getitem_1464, getitem_1465, getitem_1466, getitem_1467, getitem_1468, getitem_1469, getitem_1470, getitem_1471, getitem_1472, getitem_1473, getitem_1474, getitem_1475, getitem_1476, getitem_1477, getitem_1478, getitem_1479, getitem_1480, getitem_1481, getitem_1482, getitem_1483, getitem_1484, getitem_1485, getitem_1486, getitem_1487, getitem_1488, getitem_1489, getitem_1490, getitem_1491, getitem_1492, getitem_1493, getitem_1494, getitem_1495, getitem_1496, getitem_1497, getitem_1498, getitem_1499, getitem_1500, getitem_1501, getitem_1502, getitem_1503, getitem_1504, getitem_1505, getitem_1506, getitem_1507, getitem_1508, getitem_1509, getitem_1510, getitem_1511], [getitem_1296, getitem_1297, getitem_1298, getitem_1299, getitem_1300, getitem_1301, getitem_1302, getitem_1303, getitem_1304, getitem_1305, getitem_1306, getitem_1307, getitem_1308, getitem_1309, getitem_1310, getitem_1311, getitem_1312, getitem_1313, getitem_1314, getitem_1315, getitem_1316, getitem_1317, getitem_1318, getitem_1319, getitem_1320, getitem_1321, getitem_1322, getitem_1323, getitem_1324, getitem_1325, getitem_1326, getitem_1327, getitem_1328, getitem_1329, getitem_1330, getitem_1331, getitem_1332, getitem_1333, getitem_1334, getitem_1335, getitem_1336, getitem_1337, getitem_1338, getitem_1339, getitem_1340, getitem_1341, getitem_1342, getitem_1343, getitem_1344, getitem_1345, getitem_1346, getitem_1347, getitem_1348, getitem_1349, getitem_1350, getitem_1351, getitem_1352, getitem_1353, getitem_1354, getitem_1355, getitem_1356, getitem_1357, getitem_1358, getitem_1359, getitem_1360, getitem_1361, getitem_1362, getitem_1363, getitem_1364, getitem_1365, getitem_1366, getitem_1367, getitem_1368, getitem_1369, getitem_1370, getitem_1371, getitem_1372, getitem_1373, getitem_1374, getitem_1375, getitem_1376, getitem_1377, getitem_1378, getitem_1379, getitem_1380, getitem_1381, getitem_1382, getitem_1383, getitem_1384, getitem_1385, getitem_1386, getitem_1387, getitem_1388, getitem_1389, getitem_1390, getitem_1391, getitem_1392, getitem_1393, getitem_1394, getitem_1395, getitem_1396, getitem_1397, getitem_1398, getitem_1399, getitem_1400, getitem_1401, getitem_1402, getitem_1403]);  getitem_1404 = getitem_1405 = getitem_1406 = getitem_1407 = getitem_1408 = getitem_1409 = getitem_1410 = getitem_1411 = getitem_1412 = getitem_1413 = getitem_1414 = getitem_1415 = getitem_1416 = getitem_1417 = getitem_1418 = getitem_1419 = getitem_1420 = getitem_1421 = getitem_1422 = getitem_1423 = getitem_1424 = getitem_1425 = getitem_1426 = getitem_1427 = getitem_1428 = getitem_1429 = getitem_1430 = getitem_1431 = getitem_1432 = getitem_1433 = getitem_1434 = getitem_1435 = getitem_1436 = getitem_1437 = getitem_1438 = getitem_1439 = getitem_1440 = getitem_1441 = getitem_1442 = getitem_1443 = getitem_1444 = getitem_1445 = getitem_1446 = getitem_1447 = getitem_1448 = getitem_1449 = getitem_1450 = getitem_1451 = getitem_1452 = getitem_1453 = getitem_1454 = getitem_1455 = getitem_1456 = getitem_1457 = getitem_1458 = getitem_1459 = getitem_1460 = getitem_1461 = getitem_1462 = getitem_1463 = getitem_1464 = getitem_1465 = getitem_1466 = getitem_1467 = getitem_1468 = getitem_1469 = getitem_1470 = getitem_1471 = getitem_1472 = getitem_1473 = getitem_1474 = getitem_1475 = getitem_1476 = getitem_1477 = getitem_1478 = getitem_1479 = getitem_1480 = getitem_1481 = getitem_1482 = getitem_1483 = getitem_1484 = getitem_1485 = getitem_1486 = getitem_1487 = getitem_1488 = getitem_1489 = getitem_1490 = getitem_1491 = getitem_1492 = getitem_1493 = getitem_1494 = getitem_1495 = getitem_1496 = getitem_1497 = getitem_1498 = getitem_1499 = getitem_1500 = getitem_1501 = getitem_1502 = getitem_1503 = getitem_1504 = getitem_1505 = getitem_1506 = getitem_1507 = getitem_1508 = getitem_1509 = getitem_1510 = getitem_1511 = getitem_1296 = getitem_1297 = getitem_1298 = getitem_1299 = getitem_1300 = getitem_1301 = getitem_1302 = getitem_1303 = getitem_1304 = getitem_1305 = getitem_1306 = getitem_1307 = getitem_1308 = getitem_1309 = getitem_1310 = getitem_1311 = getitem_1312 = getitem_1313 = getitem_1314 = getitem_1315 = getitem_1316 = getitem_1317 = getitem_1318 = getitem_1319 = getitem_1320 = getitem_1321 = getitem_1322 = getitem_1323 = getitem_1324 = getitem_1325 = getitem_1326 = getitem_1327 = getitem_1328 = getitem_1329 = getitem_1330 = getitem_1331 = getitem_1332 = getitem_1333 = getitem_1334 = getitem_1335 = getitem_1336 = getitem_1337 = getitem_1338 = getitem_1339 = getitem_1340 = getitem_1341 = getitem_1342 = getitem_1343 = getitem_1344 = getitem_1345 = getitem_1346 = getitem_1347 = getitem_1348 = getitem_1349 = getitem_1350 = getitem_1351 = getitem_1352 = getitem_1353 = getitem_1354 = getitem_1355 = getitem_1356 = getitem_1357 = getitem_1358 = getitem_1359 = getitem_1360 = getitem_1361 = getitem_1362 = getitem_1363 = getitem_1364 = getitem_1365 = getitem_1366 = getitem_1367 = getitem_1368 = getitem_1369 = getitem_1370 = getitem_1371 = getitem_1372 = getitem_1373 = getitem_1374 = getitem_1375 = getitem_1376 = getitem_1377 = getitem_1378 = getitem_1379 = getitem_1380 = getitem_1381 = getitem_1382 = getitem_1383 = getitem_1384 = getitem_1385 = getitem_1386 = getitem_1387 = getitem_1388 = getitem_1389 = getitem_1390 = getitem_1391 = getitem_1392 = getitem_1393 = getitem_1394 = getitem_1395 = getitem_1396 = getitem_1397 = getitem_1398 = getitem_1399 = getitem_1400 = getitem_1401 = getitem_1402 = getitem_1403 = None
        getitem_1512: "f32[32000, 768]" = _foreach_div_list[0]
        getitem_1513: "f32[512, 768]" = _foreach_div_list[1]
        getitem_1514: "f32[4, 768]" = _foreach_div_list[2]
        getitem_1515: "f32[768]" = _foreach_div_list[3]
        getitem_1516: "f32[768]" = _foreach_div_list[4]
        getitem_1517: "f32[768, 768]" = _foreach_div_list[5]
        getitem_1518: "f32[768]" = _foreach_div_list[6]
        getitem_1519: "f32[768]" = _foreach_div_list[7]
        getitem_1520: "f32[768]" = _foreach_div_list[8]
        getitem_1521: "f32[3072, 768]" = _foreach_div_list[9]
        getitem_1522: "f32[3072]" = _foreach_div_list[10]
        getitem_1523: "f32[768, 3072]" = _foreach_div_list[11]
        getitem_1524: "f32[768]" = _foreach_div_list[12]
        getitem_1525: "f32[768]" = _foreach_div_list[13]
        getitem_1526: "f32[768]" = _foreach_div_list[14]
        getitem_1527: "f32[768]" = _foreach_div_list[15]
        getitem_1528: "f32[768]" = _foreach_div_list[16]
        getitem_1529: "f32[3072, 768]" = _foreach_div_list[17]
        getitem_1530: "f32[3072]" = _foreach_div_list[18]
        getitem_1531: "f32[768, 3072]" = _foreach_div_list[19]
        getitem_1532: "f32[768]" = _foreach_div_list[20]
        getitem_1533: "f32[768]" = _foreach_div_list[21]
        getitem_1534: "f32[768]" = _foreach_div_list[22]
        getitem_1535: "f32[768]" = _foreach_div_list[23]
        getitem_1536: "f32[768]" = _foreach_div_list[24]
        getitem_1537: "f32[3072, 768]" = _foreach_div_list[25]
        getitem_1538: "f32[3072]" = _foreach_div_list[26]
        getitem_1539: "f32[768, 3072]" = _foreach_div_list[27]
        getitem_1540: "f32[768]" = _foreach_div_list[28]
        getitem_1541: "f32[768]" = _foreach_div_list[29]
        getitem_1542: "f32[768]" = _foreach_div_list[30]
        getitem_1543: "f32[768]" = _foreach_div_list[31]
        getitem_1544: "f32[768]" = _foreach_div_list[32]
        getitem_1545: "f32[3072, 768]" = _foreach_div_list[33]
        getitem_1546: "f32[3072]" = _foreach_div_list[34]
        getitem_1547: "f32[768, 3072]" = _foreach_div_list[35]
        getitem_1548: "f32[768]" = _foreach_div_list[36]
        getitem_1549: "f32[768]" = _foreach_div_list[37]
        getitem_1550: "f32[768]" = _foreach_div_list[38]
        getitem_1551: "f32[768]" = _foreach_div_list[39]
        getitem_1552: "f32[768]" = _foreach_div_list[40]
        getitem_1553: "f32[3072, 768]" = _foreach_div_list[41]
        getitem_1554: "f32[3072]" = _foreach_div_list[42]
        getitem_1555: "f32[768, 3072]" = _foreach_div_list[43]
        getitem_1556: "f32[768]" = _foreach_div_list[44]
        getitem_1557: "f32[768]" = _foreach_div_list[45]
        getitem_1558: "f32[768]" = _foreach_div_list[46]
        getitem_1559: "f32[768]" = _foreach_div_list[47]
        getitem_1560: "f32[768]" = _foreach_div_list[48]
        getitem_1561: "f32[3072, 768]" = _foreach_div_list[49]
        getitem_1562: "f32[3072]" = _foreach_div_list[50]
        getitem_1563: "f32[768, 3072]" = _foreach_div_list[51]
        getitem_1564: "f32[768]" = _foreach_div_list[52]
        getitem_1565: "f32[768]" = _foreach_div_list[53]
        getitem_1566: "f32[768]" = _foreach_div_list[54]
        getitem_1567: "f32[768]" = _foreach_div_list[55]
        getitem_1568: "f32[768]" = _foreach_div_list[56]
        getitem_1569: "f32[3072, 768]" = _foreach_div_list[57]
        getitem_1570: "f32[3072]" = _foreach_div_list[58]
        getitem_1571: "f32[768, 3072]" = _foreach_div_list[59]
        getitem_1572: "f32[768]" = _foreach_div_list[60]
        getitem_1573: "f32[768]" = _foreach_div_list[61]
        getitem_1574: "f32[768]" = _foreach_div_list[62]
        getitem_1575: "f32[768]" = _foreach_div_list[63]
        getitem_1576: "f32[768]" = _foreach_div_list[64]
        getitem_1577: "f32[3072, 768]" = _foreach_div_list[65]
        getitem_1578: "f32[3072]" = _foreach_div_list[66]
        getitem_1579: "f32[768, 3072]" = _foreach_div_list[67]
        getitem_1580: "f32[768]" = _foreach_div_list[68]
        getitem_1581: "f32[768]" = _foreach_div_list[69]
        getitem_1582: "f32[768]" = _foreach_div_list[70]
        getitem_1583: "f32[768]" = _foreach_div_list[71]
        getitem_1584: "f32[768]" = _foreach_div_list[72]
        getitem_1585: "f32[3072, 768]" = _foreach_div_list[73]
        getitem_1586: "f32[3072]" = _foreach_div_list[74]
        getitem_1587: "f32[768, 3072]" = _foreach_div_list[75]
        getitem_1588: "f32[768]" = _foreach_div_list[76]
        getitem_1589: "f32[768]" = _foreach_div_list[77]
        getitem_1590: "f32[768]" = _foreach_div_list[78]
        getitem_1591: "f32[768]" = _foreach_div_list[79]
        getitem_1592: "f32[768]" = _foreach_div_list[80]
        getitem_1593: "f32[3072, 768]" = _foreach_div_list[81]
        getitem_1594: "f32[3072]" = _foreach_div_list[82]
        getitem_1595: "f32[768, 3072]" = _foreach_div_list[83]
        getitem_1596: "f32[768]" = _foreach_div_list[84]
        getitem_1597: "f32[768]" = _foreach_div_list[85]
        getitem_1598: "f32[768]" = _foreach_div_list[86]
        getitem_1599: "f32[768]" = _foreach_div_list[87]
        getitem_1600: "f32[768]" = _foreach_div_list[88]
        getitem_1601: "f32[3072, 768]" = _foreach_div_list[89]
        getitem_1602: "f32[3072]" = _foreach_div_list[90]
        getitem_1603: "f32[768, 3072]" = _foreach_div_list[91]
        getitem_1604: "f32[768]" = _foreach_div_list[92]
        getitem_1605: "f32[768]" = _foreach_div_list[93]
        getitem_1606: "f32[768]" = _foreach_div_list[94]
        getitem_1607: "f32[768]" = _foreach_div_list[95]
        getitem_1608: "f32[768]" = _foreach_div_list[96]
        getitem_1609: "f32[3072, 768]" = _foreach_div_list[97]
        getitem_1610: "f32[3072]" = _foreach_div_list[98]
        getitem_1611: "f32[768, 3072]" = _foreach_div_list[99]
        getitem_1612: "f32[768]" = _foreach_div_list[100]
        getitem_1613: "f32[768]" = _foreach_div_list[101]
        getitem_1614: "f32[768]" = _foreach_div_list[102]
        getitem_1615: "f32[32000]" = _foreach_div_list[103]
        getitem_1616: "f32[768, 768]" = _foreach_div_list[104]
        getitem_1617: "f32[768]" = _foreach_div_list[105]
        getitem_1618: "f32[768]" = _foreach_div_list[106]
        getitem_1619: "f32[768]" = _foreach_div_list[107];  _foreach_div_list = None
        _foreach_add_scalar_1 = torch.ops.aten._foreach_add.Scalar([getitem_1512, getitem_1513, getitem_1514, getitem_1515, getitem_1516, getitem_1517, getitem_1518, getitem_1519, getitem_1520, getitem_1521, getitem_1522, getitem_1523, getitem_1524, getitem_1525, getitem_1526, getitem_1527, getitem_1528, getitem_1529, getitem_1530, getitem_1531, getitem_1532, getitem_1533, getitem_1534, getitem_1535, getitem_1536, getitem_1537, getitem_1538, getitem_1539, getitem_1540, getitem_1541, getitem_1542, getitem_1543, getitem_1544, getitem_1545, getitem_1546, getitem_1547, getitem_1548, getitem_1549, getitem_1550, getitem_1551, getitem_1552, getitem_1553, getitem_1554, getitem_1555, getitem_1556, getitem_1557, getitem_1558, getitem_1559, getitem_1560, getitem_1561, getitem_1562, getitem_1563, getitem_1564, getitem_1565, getitem_1566, getitem_1567, getitem_1568, getitem_1569, getitem_1570, getitem_1571, getitem_1572, getitem_1573, getitem_1574, getitem_1575, getitem_1576, getitem_1577, getitem_1578, getitem_1579, getitem_1580, getitem_1581, getitem_1582, getitem_1583, getitem_1584, getitem_1585, getitem_1586, getitem_1587, getitem_1588, getitem_1589, getitem_1590, getitem_1591, getitem_1592, getitem_1593, getitem_1594, getitem_1595, getitem_1596, getitem_1597, getitem_1598, getitem_1599, getitem_1600, getitem_1601, getitem_1602, getitem_1603, getitem_1604, getitem_1605, getitem_1606, getitem_1607, getitem_1608, getitem_1609, getitem_1610, getitem_1611, getitem_1612, getitem_1613, getitem_1614, getitem_1615, getitem_1616, getitem_1617, getitem_1618, getitem_1619], 1e-08);  getitem_1512 = getitem_1513 = getitem_1514 = getitem_1515 = getitem_1516 = getitem_1517 = getitem_1518 = getitem_1519 = getitem_1520 = getitem_1521 = getitem_1522 = getitem_1523 = getitem_1524 = getitem_1525 = getitem_1526 = getitem_1527 = getitem_1528 = getitem_1529 = getitem_1530 = getitem_1531 = getitem_1532 = getitem_1533 = getitem_1534 = getitem_1535 = getitem_1536 = getitem_1537 = getitem_1538 = getitem_1539 = getitem_1540 = getitem_1541 = getitem_1542 = getitem_1543 = getitem_1544 = getitem_1545 = getitem_1546 = getitem_1547 = getitem_1548 = getitem_1549 = getitem_1550 = getitem_1551 = getitem_1552 = getitem_1553 = getitem_1554 = getitem_1555 = getitem_1556 = getitem_1557 = getitem_1558 = getitem_1559 = getitem_1560 = getitem_1561 = getitem_1562 = getitem_1563 = getitem_1564 = getitem_1565 = getitem_1566 = getitem_1567 = getitem_1568 = getitem_1569 = getitem_1570 = getitem_1571 = getitem_1572 = getitem_1573 = getitem_1574 = getitem_1575 = getitem_1576 = getitem_1577 = getitem_1578 = getitem_1579 = getitem_1580 = getitem_1581 = getitem_1582 = getitem_1583 = getitem_1584 = getitem_1585 = getitem_1586 = getitem_1587 = getitem_1588 = getitem_1589 = getitem_1590 = getitem_1591 = getitem_1592 = getitem_1593 = getitem_1594 = getitem_1595 = getitem_1596 = getitem_1597 = getitem_1598 = getitem_1599 = getitem_1600 = getitem_1601 = getitem_1602 = getitem_1603 = getitem_1604 = getitem_1605 = getitem_1606 = getitem_1607 = getitem_1608 = getitem_1609 = getitem_1610 = getitem_1611 = getitem_1612 = getitem_1613 = getitem_1614 = getitem_1615 = getitem_1616 = getitem_1617 = getitem_1618 = getitem_1619 = None
        getitem_1620: "f32[32000, 768]" = _foreach_add_scalar_1[0]
        getitem_1621: "f32[512, 768]" = _foreach_add_scalar_1[1]
        getitem_1622: "f32[4, 768]" = _foreach_add_scalar_1[2]
        getitem_1623: "f32[768]" = _foreach_add_scalar_1[3]
        getitem_1624: "f32[768]" = _foreach_add_scalar_1[4]
        getitem_1625: "f32[768, 768]" = _foreach_add_scalar_1[5]
        getitem_1626: "f32[768]" = _foreach_add_scalar_1[6]
        getitem_1627: "f32[768]" = _foreach_add_scalar_1[7]
        getitem_1628: "f32[768]" = _foreach_add_scalar_1[8]
        getitem_1629: "f32[3072, 768]" = _foreach_add_scalar_1[9]
        getitem_1630: "f32[3072]" = _foreach_add_scalar_1[10]
        getitem_1631: "f32[768, 3072]" = _foreach_add_scalar_1[11]
        getitem_1632: "f32[768]" = _foreach_add_scalar_1[12]
        getitem_1633: "f32[768]" = _foreach_add_scalar_1[13]
        getitem_1634: "f32[768]" = _foreach_add_scalar_1[14]
        getitem_1635: "f32[768]" = _foreach_add_scalar_1[15]
        getitem_1636: "f32[768]" = _foreach_add_scalar_1[16]
        getitem_1637: "f32[3072, 768]" = _foreach_add_scalar_1[17]
        getitem_1638: "f32[3072]" = _foreach_add_scalar_1[18]
        getitem_1639: "f32[768, 3072]" = _foreach_add_scalar_1[19]
        getitem_1640: "f32[768]" = _foreach_add_scalar_1[20]
        getitem_1641: "f32[768]" = _foreach_add_scalar_1[21]
        getitem_1642: "f32[768]" = _foreach_add_scalar_1[22]
        getitem_1643: "f32[768]" = _foreach_add_scalar_1[23]
        getitem_1644: "f32[768]" = _foreach_add_scalar_1[24]
        getitem_1645: "f32[3072, 768]" = _foreach_add_scalar_1[25]
        getitem_1646: "f32[3072]" = _foreach_add_scalar_1[26]
        getitem_1647: "f32[768, 3072]" = _foreach_add_scalar_1[27]
        getitem_1648: "f32[768]" = _foreach_add_scalar_1[28]
        getitem_1649: "f32[768]" = _foreach_add_scalar_1[29]
        getitem_1650: "f32[768]" = _foreach_add_scalar_1[30]
        getitem_1651: "f32[768]" = _foreach_add_scalar_1[31]
        getitem_1652: "f32[768]" = _foreach_add_scalar_1[32]
        getitem_1653: "f32[3072, 768]" = _foreach_add_scalar_1[33]
        getitem_1654: "f32[3072]" = _foreach_add_scalar_1[34]
        getitem_1655: "f32[768, 3072]" = _foreach_add_scalar_1[35]
        getitem_1656: "f32[768]" = _foreach_add_scalar_1[36]
        getitem_1657: "f32[768]" = _foreach_add_scalar_1[37]
        getitem_1658: "f32[768]" = _foreach_add_scalar_1[38]
        getitem_1659: "f32[768]" = _foreach_add_scalar_1[39]
        getitem_1660: "f32[768]" = _foreach_add_scalar_1[40]
        getitem_1661: "f32[3072, 768]" = _foreach_add_scalar_1[41]
        getitem_1662: "f32[3072]" = _foreach_add_scalar_1[42]
        getitem_1663: "f32[768, 3072]" = _foreach_add_scalar_1[43]
        getitem_1664: "f32[768]" = _foreach_add_scalar_1[44]
        getitem_1665: "f32[768]" = _foreach_add_scalar_1[45]
        getitem_1666: "f32[768]" = _foreach_add_scalar_1[46]
        getitem_1667: "f32[768]" = _foreach_add_scalar_1[47]
        getitem_1668: "f32[768]" = _foreach_add_scalar_1[48]
        getitem_1669: "f32[3072, 768]" = _foreach_add_scalar_1[49]
        getitem_1670: "f32[3072]" = _foreach_add_scalar_1[50]
        getitem_1671: "f32[768, 3072]" = _foreach_add_scalar_1[51]
        getitem_1672: "f32[768]" = _foreach_add_scalar_1[52]
        getitem_1673: "f32[768]" = _foreach_add_scalar_1[53]
        getitem_1674: "f32[768]" = _foreach_add_scalar_1[54]
        getitem_1675: "f32[768]" = _foreach_add_scalar_1[55]
        getitem_1676: "f32[768]" = _foreach_add_scalar_1[56]
        getitem_1677: "f32[3072, 768]" = _foreach_add_scalar_1[57]
        getitem_1678: "f32[3072]" = _foreach_add_scalar_1[58]
        getitem_1679: "f32[768, 3072]" = _foreach_add_scalar_1[59]
        getitem_1680: "f32[768]" = _foreach_add_scalar_1[60]
        getitem_1681: "f32[768]" = _foreach_add_scalar_1[61]
        getitem_1682: "f32[768]" = _foreach_add_scalar_1[62]
        getitem_1683: "f32[768]" = _foreach_add_scalar_1[63]
        getitem_1684: "f32[768]" = _foreach_add_scalar_1[64]
        getitem_1685: "f32[3072, 768]" = _foreach_add_scalar_1[65]
        getitem_1686: "f32[3072]" = _foreach_add_scalar_1[66]
        getitem_1687: "f32[768, 3072]" = _foreach_add_scalar_1[67]
        getitem_1688: "f32[768]" = _foreach_add_scalar_1[68]
        getitem_1689: "f32[768]" = _foreach_add_scalar_1[69]
        getitem_1690: "f32[768]" = _foreach_add_scalar_1[70]
        getitem_1691: "f32[768]" = _foreach_add_scalar_1[71]
        getitem_1692: "f32[768]" = _foreach_add_scalar_1[72]
        getitem_1693: "f32[3072, 768]" = _foreach_add_scalar_1[73]
        getitem_1694: "f32[3072]" = _foreach_add_scalar_1[74]
        getitem_1695: "f32[768, 3072]" = _foreach_add_scalar_1[75]
        getitem_1696: "f32[768]" = _foreach_add_scalar_1[76]
        getitem_1697: "f32[768]" = _foreach_add_scalar_1[77]
        getitem_1698: "f32[768]" = _foreach_add_scalar_1[78]
        getitem_1699: "f32[768]" = _foreach_add_scalar_1[79]
        getitem_1700: "f32[768]" = _foreach_add_scalar_1[80]
        getitem_1701: "f32[3072, 768]" = _foreach_add_scalar_1[81]
        getitem_1702: "f32[3072]" = _foreach_add_scalar_1[82]
        getitem_1703: "f32[768, 3072]" = _foreach_add_scalar_1[83]
        getitem_1704: "f32[768]" = _foreach_add_scalar_1[84]
        getitem_1705: "f32[768]" = _foreach_add_scalar_1[85]
        getitem_1706: "f32[768]" = _foreach_add_scalar_1[86]
        getitem_1707: "f32[768]" = _foreach_add_scalar_1[87]
        getitem_1708: "f32[768]" = _foreach_add_scalar_1[88]
        getitem_1709: "f32[3072, 768]" = _foreach_add_scalar_1[89]
        getitem_1710: "f32[3072]" = _foreach_add_scalar_1[90]
        getitem_1711: "f32[768, 3072]" = _foreach_add_scalar_1[91]
        getitem_1712: "f32[768]" = _foreach_add_scalar_1[92]
        getitem_1713: "f32[768]" = _foreach_add_scalar_1[93]
        getitem_1714: "f32[768]" = _foreach_add_scalar_1[94]
        getitem_1715: "f32[768]" = _foreach_add_scalar_1[95]
        getitem_1716: "f32[768]" = _foreach_add_scalar_1[96]
        getitem_1717: "f32[3072, 768]" = _foreach_add_scalar_1[97]
        getitem_1718: "f32[3072]" = _foreach_add_scalar_1[98]
        getitem_1719: "f32[768, 3072]" = _foreach_add_scalar_1[99]
        getitem_1720: "f32[768]" = _foreach_add_scalar_1[100]
        getitem_1721: "f32[768]" = _foreach_add_scalar_1[101]
        getitem_1722: "f32[768]" = _foreach_add_scalar_1[102]
        getitem_1723: "f32[32000]" = _foreach_add_scalar_1[103]
        getitem_1724: "f32[768, 768]" = _foreach_add_scalar_1[104]
        getitem_1725: "f32[768]" = _foreach_add_scalar_1[105]
        getitem_1726: "f32[768]" = _foreach_add_scalar_1[106]
        getitem_1727: "f32[768]" = _foreach_add_scalar_1[107];  _foreach_add_scalar_1 = None
        _foreach_div_list_1 = torch.ops.aten._foreach_div.List([getitem_1620, getitem_1621, getitem_1622, getitem_1623, getitem_1624, getitem_1625, getitem_1626, getitem_1627, getitem_1628, getitem_1629, getitem_1630, getitem_1631, getitem_1632, getitem_1633, getitem_1634, getitem_1635, getitem_1636, getitem_1637, getitem_1638, getitem_1639, getitem_1640, getitem_1641, getitem_1642, getitem_1643, getitem_1644, getitem_1645, getitem_1646, getitem_1647, getitem_1648, getitem_1649, getitem_1650, getitem_1651, getitem_1652, getitem_1653, getitem_1654, getitem_1655, getitem_1656, getitem_1657, getitem_1658, getitem_1659, getitem_1660, getitem_1661, getitem_1662, getitem_1663, getitem_1664, getitem_1665, getitem_1666, getitem_1667, getitem_1668, getitem_1669, getitem_1670, getitem_1671, getitem_1672, getitem_1673, getitem_1674, getitem_1675, getitem_1676, getitem_1677, getitem_1678, getitem_1679, getitem_1680, getitem_1681, getitem_1682, getitem_1683, getitem_1684, getitem_1685, getitem_1686, getitem_1687, getitem_1688, getitem_1689, getitem_1690, getitem_1691, getitem_1692, getitem_1693, getitem_1694, getitem_1695, getitem_1696, getitem_1697, getitem_1698, getitem_1699, getitem_1700, getitem_1701, getitem_1702, getitem_1703, getitem_1704, getitem_1705, getitem_1706, getitem_1707, getitem_1708, getitem_1709, getitem_1710, getitem_1711, getitem_1712, getitem_1713, getitem_1714, getitem_1715, getitem_1716, getitem_1717, getitem_1718, getitem_1719, getitem_1720, getitem_1721, getitem_1722, getitem_1723, getitem_1724, getitem_1725, getitem_1726, getitem_1727], [getitem_1188, getitem_1189, getitem_1190, getitem_1191, getitem_1192, getitem_1193, getitem_1194, getitem_1195, getitem_1196, getitem_1197, getitem_1198, getitem_1199, getitem_1200, getitem_1201, getitem_1202, getitem_1203, getitem_1204, getitem_1205, getitem_1206, getitem_1207, getitem_1208, getitem_1209, getitem_1210, getitem_1211, getitem_1212, getitem_1213, getitem_1214, getitem_1215, getitem_1216, getitem_1217, getitem_1218, getitem_1219, getitem_1220, getitem_1221, getitem_1222, getitem_1223, getitem_1224, getitem_1225, getitem_1226, getitem_1227, getitem_1228, getitem_1229, getitem_1230, getitem_1231, getitem_1232, getitem_1233, getitem_1234, getitem_1235, getitem_1236, getitem_1237, getitem_1238, getitem_1239, getitem_1240, getitem_1241, getitem_1242, getitem_1243, getitem_1244, getitem_1245, getitem_1246, getitem_1247, getitem_1248, getitem_1249, getitem_1250, getitem_1251, getitem_1252, getitem_1253, getitem_1254, getitem_1255, getitem_1256, getitem_1257, getitem_1258, getitem_1259, getitem_1260, getitem_1261, getitem_1262, getitem_1263, getitem_1264, getitem_1265, getitem_1266, getitem_1267, getitem_1268, getitem_1269, getitem_1270, getitem_1271, getitem_1272, getitem_1273, getitem_1274, getitem_1275, getitem_1276, getitem_1277, getitem_1278, getitem_1279, getitem_1280, getitem_1281, getitem_1282, getitem_1283, getitem_1284, getitem_1285, getitem_1286, getitem_1287, getitem_1288, getitem_1289, getitem_1290, getitem_1291, getitem_1292, getitem_1293, getitem_1294, getitem_1295]);  getitem_1620 = getitem_1621 = getitem_1622 = getitem_1623 = getitem_1624 = getitem_1625 = getitem_1626 = getitem_1627 = getitem_1628 = getitem_1629 = getitem_1630 = getitem_1631 = getitem_1632 = getitem_1633 = getitem_1634 = getitem_1635 = getitem_1636 = getitem_1637 = getitem_1638 = getitem_1639 = getitem_1640 = getitem_1641 = getitem_1642 = getitem_1643 = getitem_1644 = getitem_1645 = getitem_1646 = getitem_1647 = getitem_1648 = getitem_1649 = getitem_1650 = getitem_1651 = getitem_1652 = getitem_1653 = getitem_1654 = getitem_1655 = getitem_1656 = getitem_1657 = getitem_1658 = getitem_1659 = getitem_1660 = getitem_1661 = getitem_1662 = getitem_1663 = getitem_1664 = getitem_1665 = getitem_1666 = getitem_1667 = getitem_1668 = getitem_1669 = getitem_1670 = getitem_1671 = getitem_1672 = getitem_1673 = getitem_1674 = getitem_1675 = getitem_1676 = getitem_1677 = getitem_1678 = getitem_1679 = getitem_1680 = getitem_1681 = getitem_1682 = getitem_1683 = getitem_1684 = getitem_1685 = getitem_1686 = getitem_1687 = getitem_1688 = getitem_1689 = getitem_1690 = getitem_1691 = getitem_1692 = getitem_1693 = getitem_1694 = getitem_1695 = getitem_1696 = getitem_1697 = getitem_1698 = getitem_1699 = getitem_1700 = getitem_1701 = getitem_1702 = getitem_1703 = getitem_1704 = getitem_1705 = getitem_1706 = getitem_1707 = getitem_1708 = getitem_1709 = getitem_1710 = getitem_1711 = getitem_1712 = getitem_1713 = getitem_1714 = getitem_1715 = getitem_1716 = getitem_1717 = getitem_1718 = getitem_1719 = getitem_1720 = getitem_1721 = getitem_1722 = getitem_1723 = getitem_1724 = getitem_1725 = getitem_1726 = getitem_1727 = getitem_1188 = getitem_1189 = getitem_1190 = getitem_1191 = getitem_1192 = getitem_1193 = getitem_1194 = getitem_1195 = getitem_1196 = getitem_1197 = getitem_1198 = getitem_1199 = getitem_1200 = getitem_1201 = getitem_1202 = getitem_1203 = getitem_1204 = getitem_1205 = getitem_1206 = getitem_1207 = getitem_1208 = getitem_1209 = getitem_1210 = getitem_1211 = getitem_1212 = getitem_1213 = getitem_1214 = getitem_1215 = getitem_1216 = getitem_1217 = getitem_1218 = getitem_1219 = getitem_1220 = getitem_1221 = getitem_1222 = getitem_1223 = getitem_1224 = getitem_1225 = getitem_1226 = getitem_1227 = getitem_1228 = getitem_1229 = getitem_1230 = getitem_1231 = getitem_1232 = getitem_1233 = getitem_1234 = getitem_1235 = getitem_1236 = getitem_1237 = getitem_1238 = getitem_1239 = getitem_1240 = getitem_1241 = getitem_1242 = getitem_1243 = getitem_1244 = getitem_1245 = getitem_1246 = getitem_1247 = getitem_1248 = getitem_1249 = getitem_1250 = getitem_1251 = getitem_1252 = getitem_1253 = getitem_1254 = getitem_1255 = getitem_1256 = getitem_1257 = getitem_1258 = getitem_1259 = getitem_1260 = getitem_1261 = getitem_1262 = getitem_1263 = getitem_1264 = getitem_1265 = getitem_1266 = getitem_1267 = getitem_1268 = getitem_1269 = getitem_1270 = getitem_1271 = getitem_1272 = getitem_1273 = getitem_1274 = getitem_1275 = getitem_1276 = getitem_1277 = getitem_1278 = getitem_1279 = getitem_1280 = getitem_1281 = getitem_1282 = getitem_1283 = getitem_1284 = getitem_1285 = getitem_1286 = getitem_1287 = getitem_1288 = getitem_1289 = getitem_1290 = getitem_1291 = getitem_1292 = getitem_1293 = getitem_1294 = getitem_1295 = None
        getitem_1728: "f32[32000, 768]" = _foreach_div_list_1[0]
        getitem_1729: "f32[512, 768]" = _foreach_div_list_1[1]
        getitem_1730: "f32[4, 768]" = _foreach_div_list_1[2]
        getitem_1731: "f32[768]" = _foreach_div_list_1[3]
        getitem_1732: "f32[768]" = _foreach_div_list_1[4]
        getitem_1733: "f32[768, 768]" = _foreach_div_list_1[5]
        getitem_1734: "f32[768]" = _foreach_div_list_1[6]
        getitem_1735: "f32[768]" = _foreach_div_list_1[7]
        getitem_1736: "f32[768]" = _foreach_div_list_1[8]
        getitem_1737: "f32[3072, 768]" = _foreach_div_list_1[9]
        getitem_1738: "f32[3072]" = _foreach_div_list_1[10]
        getitem_1739: "f32[768, 3072]" = _foreach_div_list_1[11]
        getitem_1740: "f32[768]" = _foreach_div_list_1[12]
        getitem_1741: "f32[768]" = _foreach_div_list_1[13]
        getitem_1742: "f32[768]" = _foreach_div_list_1[14]
        getitem_1743: "f32[768]" = _foreach_div_list_1[15]
        getitem_1744: "f32[768]" = _foreach_div_list_1[16]
        getitem_1745: "f32[3072, 768]" = _foreach_div_list_1[17]
        getitem_1746: "f32[3072]" = _foreach_div_list_1[18]
        getitem_1747: "f32[768, 3072]" = _foreach_div_list_1[19]
        getitem_1748: "f32[768]" = _foreach_div_list_1[20]
        getitem_1749: "f32[768]" = _foreach_div_list_1[21]
        getitem_1750: "f32[768]" = _foreach_div_list_1[22]
        getitem_1751: "f32[768]" = _foreach_div_list_1[23]
        getitem_1752: "f32[768]" = _foreach_div_list_1[24]
        getitem_1753: "f32[3072, 768]" = _foreach_div_list_1[25]
        getitem_1754: "f32[3072]" = _foreach_div_list_1[26]
        getitem_1755: "f32[768, 3072]" = _foreach_div_list_1[27]
        getitem_1756: "f32[768]" = _foreach_div_list_1[28]
        getitem_1757: "f32[768]" = _foreach_div_list_1[29]
        getitem_1758: "f32[768]" = _foreach_div_list_1[30]
        getitem_1759: "f32[768]" = _foreach_div_list_1[31]
        getitem_1760: "f32[768]" = _foreach_div_list_1[32]
        getitem_1761: "f32[3072, 768]" = _foreach_div_list_1[33]
        getitem_1762: "f32[3072]" = _foreach_div_list_1[34]
        getitem_1763: "f32[768, 3072]" = _foreach_div_list_1[35]
        getitem_1764: "f32[768]" = _foreach_div_list_1[36]
        getitem_1765: "f32[768]" = _foreach_div_list_1[37]
        getitem_1766: "f32[768]" = _foreach_div_list_1[38]
        getitem_1767: "f32[768]" = _foreach_div_list_1[39]
        getitem_1768: "f32[768]" = _foreach_div_list_1[40]
        getitem_1769: "f32[3072, 768]" = _foreach_div_list_1[41]
        getitem_1770: "f32[3072]" = _foreach_div_list_1[42]
        getitem_1771: "f32[768, 3072]" = _foreach_div_list_1[43]
        getitem_1772: "f32[768]" = _foreach_div_list_1[44]
        getitem_1773: "f32[768]" = _foreach_div_list_1[45]
        getitem_1774: "f32[768]" = _foreach_div_list_1[46]
        getitem_1775: "f32[768]" = _foreach_div_list_1[47]
        getitem_1776: "f32[768]" = _foreach_div_list_1[48]
        getitem_1777: "f32[3072, 768]" = _foreach_div_list_1[49]
        getitem_1778: "f32[3072]" = _foreach_div_list_1[50]
        getitem_1779: "f32[768, 3072]" = _foreach_div_list_1[51]
        getitem_1780: "f32[768]" = _foreach_div_list_1[52]
        getitem_1781: "f32[768]" = _foreach_div_list_1[53]
        getitem_1782: "f32[768]" = _foreach_div_list_1[54]
        getitem_1783: "f32[768]" = _foreach_div_list_1[55]
        getitem_1784: "f32[768]" = _foreach_div_list_1[56]
        getitem_1785: "f32[3072, 768]" = _foreach_div_list_1[57]
        getitem_1786: "f32[3072]" = _foreach_div_list_1[58]
        getitem_1787: "f32[768, 3072]" = _foreach_div_list_1[59]
        getitem_1788: "f32[768]" = _foreach_div_list_1[60]
        getitem_1789: "f32[768]" = _foreach_div_list_1[61]
        getitem_1790: "f32[768]" = _foreach_div_list_1[62]
        getitem_1791: "f32[768]" = _foreach_div_list_1[63]
        getitem_1792: "f32[768]" = _foreach_div_list_1[64]
        getitem_1793: "f32[3072, 768]" = _foreach_div_list_1[65]
        getitem_1794: "f32[3072]" = _foreach_div_list_1[66]
        getitem_1795: "f32[768, 3072]" = _foreach_div_list_1[67]
        getitem_1796: "f32[768]" = _foreach_div_list_1[68]
        getitem_1797: "f32[768]" = _foreach_div_list_1[69]
        getitem_1798: "f32[768]" = _foreach_div_list_1[70]
        getitem_1799: "f32[768]" = _foreach_div_list_1[71]
        getitem_1800: "f32[768]" = _foreach_div_list_1[72]
        getitem_1801: "f32[3072, 768]" = _foreach_div_list_1[73]
        getitem_1802: "f32[3072]" = _foreach_div_list_1[74]
        getitem_1803: "f32[768, 3072]" = _foreach_div_list_1[75]
        getitem_1804: "f32[768]" = _foreach_div_list_1[76]
        getitem_1805: "f32[768]" = _foreach_div_list_1[77]
        getitem_1806: "f32[768]" = _foreach_div_list_1[78]
        getitem_1807: "f32[768]" = _foreach_div_list_1[79]
        getitem_1808: "f32[768]" = _foreach_div_list_1[80]
        getitem_1809: "f32[3072, 768]" = _foreach_div_list_1[81]
        getitem_1810: "f32[3072]" = _foreach_div_list_1[82]
        getitem_1811: "f32[768, 3072]" = _foreach_div_list_1[83]
        getitem_1812: "f32[768]" = _foreach_div_list_1[84]
        getitem_1813: "f32[768]" = _foreach_div_list_1[85]
        getitem_1814: "f32[768]" = _foreach_div_list_1[86]
        getitem_1815: "f32[768]" = _foreach_div_list_1[87]
        getitem_1816: "f32[768]" = _foreach_div_list_1[88]
        getitem_1817: "f32[3072, 768]" = _foreach_div_list_1[89]
        getitem_1818: "f32[3072]" = _foreach_div_list_1[90]
        getitem_1819: "f32[768, 3072]" = _foreach_div_list_1[91]
        getitem_1820: "f32[768]" = _foreach_div_list_1[92]
        getitem_1821: "f32[768]" = _foreach_div_list_1[93]
        getitem_1822: "f32[768]" = _foreach_div_list_1[94]
        getitem_1823: "f32[768]" = _foreach_div_list_1[95]
        getitem_1824: "f32[768]" = _foreach_div_list_1[96]
        getitem_1825: "f32[3072, 768]" = _foreach_div_list_1[97]
        getitem_1826: "f32[3072]" = _foreach_div_list_1[98]
        getitem_1827: "f32[768, 3072]" = _foreach_div_list_1[99]
        getitem_1828: "f32[768]" = _foreach_div_list_1[100]
        getitem_1829: "f32[768]" = _foreach_div_list_1[101]
        getitem_1830: "f32[768]" = _foreach_div_list_1[102]
        getitem_1831: "f32[32000]" = _foreach_div_list_1[103]
        getitem_1832: "f32[768, 768]" = _foreach_div_list_1[104]
        getitem_1833: "f32[768]" = _foreach_div_list_1[105]
        getitem_1834: "f32[768]" = _foreach_div_list_1[106]
        getitem_1835: "f32[768]" = _foreach_div_list_1[107];  _foreach_div_list_1 = None
        _foreach_addcdiv_scalar = torch.ops.aten._foreach_addcdiv.Scalar([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1], [getitem_216, getitem_217, getitem_218, getitem_219, getitem_220, getitem_221, getitem_222, getitem_223, getitem_224, getitem_225, getitem_226, getitem_227, getitem_228, getitem_229, getitem_230, getitem_231, getitem_232, getitem_233, getitem_234, getitem_235, getitem_236, getitem_237, getitem_238, getitem_239, getitem_240, getitem_241, getitem_242, getitem_243, getitem_244, getitem_245, getitem_246, getitem_247, getitem_248, getitem_249, getitem_250, getitem_251, getitem_252, getitem_253, getitem_254, getitem_255, getitem_256, getitem_257, getitem_258, getitem_259, getitem_260, getitem_261, getitem_262, getitem_263, getitem_264, getitem_265, getitem_266, getitem_267, getitem_268, getitem_269, getitem_270, getitem_271, getitem_272, getitem_273, getitem_274, getitem_275, getitem_276, getitem_277, getitem_278, getitem_279, getitem_280, getitem_281, getitem_282, getitem_283, getitem_284, getitem_285, getitem_286, getitem_287, getitem_288, getitem_289, getitem_290, getitem_291, getitem_292, getitem_293, getitem_294, getitem_295, getitem_296, getitem_297, getitem_298, getitem_299, getitem_300, getitem_301, getitem_302, getitem_303, getitem_304, getitem_305, getitem_306, getitem_307, getitem_308, getitem_309, getitem_310, getitem_311, getitem_312, getitem_313, getitem_314, getitem_315, getitem_316, getitem_317, getitem_318, getitem_319, getitem_320, getitem_321, getitem_322, getitem_323], [getitem_1728, getitem_1729, getitem_1730, getitem_1731, getitem_1732, getitem_1733, getitem_1734, getitem_1735, getitem_1736, getitem_1737, getitem_1738, getitem_1739, getitem_1740, getitem_1741, getitem_1742, getitem_1743, getitem_1744, getitem_1745, getitem_1746, getitem_1747, getitem_1748, getitem_1749, getitem_1750, getitem_1751, getitem_1752, getitem_1753, getitem_1754, getitem_1755, getitem_1756, getitem_1757, getitem_1758, getitem_1759, getitem_1760, getitem_1761, getitem_1762, getitem_1763, getitem_1764, getitem_1765, getitem_1766, getitem_1767, getitem_1768, getitem_1769, getitem_1770, getitem_1771, getitem_1772, getitem_1773, getitem_1774, getitem_1775, getitem_1776, getitem_1777, getitem_1778, getitem_1779, getitem_1780, getitem_1781, getitem_1782, getitem_1783, getitem_1784, getitem_1785, getitem_1786, getitem_1787, getitem_1788, getitem_1789, getitem_1790, getitem_1791, getitem_1792, getitem_1793, getitem_1794, getitem_1795, getitem_1796, getitem_1797, getitem_1798, getitem_1799, getitem_1800, getitem_1801, getitem_1802, getitem_1803, getitem_1804, getitem_1805, getitem_1806, getitem_1807, getitem_1808, getitem_1809, getitem_1810, getitem_1811, getitem_1812, getitem_1813, getitem_1814, getitem_1815, getitem_1816, getitem_1817, getitem_1818, getitem_1819, getitem_1820, getitem_1821, getitem_1822, getitem_1823, getitem_1824, getitem_1825, getitem_1826, getitem_1827, getitem_1828, getitem_1829, getitem_1830, getitem_1831, getitem_1832, getitem_1833, getitem_1834, getitem_1835]);  getitem_1728 = getitem_1729 = getitem_1730 = getitem_1731 = getitem_1732 = getitem_1733 = getitem_1734 = getitem_1735 = getitem_1736 = getitem_1737 = getitem_1738 = getitem_1739 = getitem_1740 = getitem_1741 = getitem_1742 = getitem_1743 = getitem_1744 = getitem_1745 = getitem_1746 = getitem_1747 = getitem_1748 = getitem_1749 = getitem_1750 = getitem_1751 = getitem_1752 = getitem_1753 = getitem_1754 = getitem_1755 = getitem_1756 = getitem_1757 = getitem_1758 = getitem_1759 = getitem_1760 = getitem_1761 = getitem_1762 = getitem_1763 = getitem_1764 = getitem_1765 = getitem_1766 = getitem_1767 = getitem_1768 = getitem_1769 = getitem_1770 = getitem_1771 = getitem_1772 = getitem_1773 = getitem_1774 = getitem_1775 = getitem_1776 = getitem_1777 = getitem_1778 = getitem_1779 = getitem_1780 = getitem_1781 = getitem_1782 = getitem_1783 = getitem_1784 = getitem_1785 = getitem_1786 = getitem_1787 = getitem_1788 = getitem_1789 = getitem_1790 = getitem_1791 = getitem_1792 = getitem_1793 = getitem_1794 = getitem_1795 = getitem_1796 = getitem_1797 = getitem_1798 = getitem_1799 = getitem_1800 = getitem_1801 = getitem_1802 = getitem_1803 = getitem_1804 = getitem_1805 = getitem_1806 = getitem_1807 = getitem_1808 = getitem_1809 = getitem_1810 = getitem_1811 = getitem_1812 = getitem_1813 = getitem_1814 = getitem_1815 = getitem_1816 = getitem_1817 = getitem_1818 = getitem_1819 = getitem_1820 = getitem_1821 = getitem_1822 = getitem_1823 = getitem_1824 = getitem_1825 = getitem_1826 = getitem_1827 = getitem_1828 = getitem_1829 = getitem_1830 = getitem_1831 = getitem_1832 = getitem_1833 = getitem_1834 = getitem_1835 = None
        getitem_1836: "f32[32000, 768]" = _foreach_addcdiv_scalar[0]
        getitem_1837: "f32[512, 768]" = _foreach_addcdiv_scalar[1]
        getitem_1838: "f32[4, 768]" = _foreach_addcdiv_scalar[2]
        getitem_1839: "f32[768]" = _foreach_addcdiv_scalar[3]
        getitem_1840: "f32[768]" = _foreach_addcdiv_scalar[4]
        getitem_1841: "f32[768, 768]" = _foreach_addcdiv_scalar[5]
        getitem_1842: "f32[768]" = _foreach_addcdiv_scalar[6]
        getitem_1843: "f32[768]" = _foreach_addcdiv_scalar[7]
        getitem_1844: "f32[768]" = _foreach_addcdiv_scalar[8]
        getitem_1845: "f32[3072, 768]" = _foreach_addcdiv_scalar[9]
        getitem_1846: "f32[3072]" = _foreach_addcdiv_scalar[10]
        getitem_1847: "f32[768, 3072]" = _foreach_addcdiv_scalar[11]
        getitem_1848: "f32[768]" = _foreach_addcdiv_scalar[12]
        getitem_1849: "f32[768]" = _foreach_addcdiv_scalar[13]
        getitem_1850: "f32[768]" = _foreach_addcdiv_scalar[14]
        getitem_1851: "f32[768]" = _foreach_addcdiv_scalar[15]
        getitem_1852: "f32[768]" = _foreach_addcdiv_scalar[16]
        getitem_1853: "f32[3072, 768]" = _foreach_addcdiv_scalar[17]
        getitem_1854: "f32[3072]" = _foreach_addcdiv_scalar[18]
        getitem_1855: "f32[768, 3072]" = _foreach_addcdiv_scalar[19]
        getitem_1856: "f32[768]" = _foreach_addcdiv_scalar[20]
        getitem_1857: "f32[768]" = _foreach_addcdiv_scalar[21]
        getitem_1858: "f32[768]" = _foreach_addcdiv_scalar[22]
        getitem_1859: "f32[768]" = _foreach_addcdiv_scalar[23]
        getitem_1860: "f32[768]" = _foreach_addcdiv_scalar[24]
        getitem_1861: "f32[3072, 768]" = _foreach_addcdiv_scalar[25]
        getitem_1862: "f32[3072]" = _foreach_addcdiv_scalar[26]
        getitem_1863: "f32[768, 3072]" = _foreach_addcdiv_scalar[27]
        getitem_1864: "f32[768]" = _foreach_addcdiv_scalar[28]
        getitem_1865: "f32[768]" = _foreach_addcdiv_scalar[29]
        getitem_1866: "f32[768]" = _foreach_addcdiv_scalar[30]
        getitem_1867: "f32[768]" = _foreach_addcdiv_scalar[31]
        getitem_1868: "f32[768]" = _foreach_addcdiv_scalar[32]
        getitem_1869: "f32[3072, 768]" = _foreach_addcdiv_scalar[33]
        getitem_1870: "f32[3072]" = _foreach_addcdiv_scalar[34]
        getitem_1871: "f32[768, 3072]" = _foreach_addcdiv_scalar[35]
        getitem_1872: "f32[768]" = _foreach_addcdiv_scalar[36]
        getitem_1873: "f32[768]" = _foreach_addcdiv_scalar[37]
        getitem_1874: "f32[768]" = _foreach_addcdiv_scalar[38]
        getitem_1875: "f32[768]" = _foreach_addcdiv_scalar[39]
        getitem_1876: "f32[768]" = _foreach_addcdiv_scalar[40]
        getitem_1877: "f32[3072, 768]" = _foreach_addcdiv_scalar[41]
        getitem_1878: "f32[3072]" = _foreach_addcdiv_scalar[42]
        getitem_1879: "f32[768, 3072]" = _foreach_addcdiv_scalar[43]
        getitem_1880: "f32[768]" = _foreach_addcdiv_scalar[44]
        getitem_1881: "f32[768]" = _foreach_addcdiv_scalar[45]
        getitem_1882: "f32[768]" = _foreach_addcdiv_scalar[46]
        getitem_1883: "f32[768]" = _foreach_addcdiv_scalar[47]
        getitem_1884: "f32[768]" = _foreach_addcdiv_scalar[48]
        getitem_1885: "f32[3072, 768]" = _foreach_addcdiv_scalar[49]
        getitem_1886: "f32[3072]" = _foreach_addcdiv_scalar[50]
        getitem_1887: "f32[768, 3072]" = _foreach_addcdiv_scalar[51]
        getitem_1888: "f32[768]" = _foreach_addcdiv_scalar[52]
        getitem_1889: "f32[768]" = _foreach_addcdiv_scalar[53]
        getitem_1890: "f32[768]" = _foreach_addcdiv_scalar[54]
        getitem_1891: "f32[768]" = _foreach_addcdiv_scalar[55]
        getitem_1892: "f32[768]" = _foreach_addcdiv_scalar[56]
        getitem_1893: "f32[3072, 768]" = _foreach_addcdiv_scalar[57]
        getitem_1894: "f32[3072]" = _foreach_addcdiv_scalar[58]
        getitem_1895: "f32[768, 3072]" = _foreach_addcdiv_scalar[59]
        getitem_1896: "f32[768]" = _foreach_addcdiv_scalar[60]
        getitem_1897: "f32[768]" = _foreach_addcdiv_scalar[61]
        getitem_1898: "f32[768]" = _foreach_addcdiv_scalar[62]
        getitem_1899: "f32[768]" = _foreach_addcdiv_scalar[63]
        getitem_1900: "f32[768]" = _foreach_addcdiv_scalar[64]
        getitem_1901: "f32[3072, 768]" = _foreach_addcdiv_scalar[65]
        getitem_1902: "f32[3072]" = _foreach_addcdiv_scalar[66]
        getitem_1903: "f32[768, 3072]" = _foreach_addcdiv_scalar[67]
        getitem_1904: "f32[768]" = _foreach_addcdiv_scalar[68]
        getitem_1905: "f32[768]" = _foreach_addcdiv_scalar[69]
        getitem_1906: "f32[768]" = _foreach_addcdiv_scalar[70]
        getitem_1907: "f32[768]" = _foreach_addcdiv_scalar[71]
        getitem_1908: "f32[768]" = _foreach_addcdiv_scalar[72]
        getitem_1909: "f32[3072, 768]" = _foreach_addcdiv_scalar[73]
        getitem_1910: "f32[3072]" = _foreach_addcdiv_scalar[74]
        getitem_1911: "f32[768, 3072]" = _foreach_addcdiv_scalar[75]
        getitem_1912: "f32[768]" = _foreach_addcdiv_scalar[76]
        getitem_1913: "f32[768]" = _foreach_addcdiv_scalar[77]
        getitem_1914: "f32[768]" = _foreach_addcdiv_scalar[78]
        getitem_1915: "f32[768]" = _foreach_addcdiv_scalar[79]
        getitem_1916: "f32[768]" = _foreach_addcdiv_scalar[80]
        getitem_1917: "f32[3072, 768]" = _foreach_addcdiv_scalar[81]
        getitem_1918: "f32[3072]" = _foreach_addcdiv_scalar[82]
        getitem_1919: "f32[768, 3072]" = _foreach_addcdiv_scalar[83]
        getitem_1920: "f32[768]" = _foreach_addcdiv_scalar[84]
        getitem_1921: "f32[768]" = _foreach_addcdiv_scalar[85]
        getitem_1922: "f32[768]" = _foreach_addcdiv_scalar[86]
        getitem_1923: "f32[768]" = _foreach_addcdiv_scalar[87]
        getitem_1924: "f32[768]" = _foreach_addcdiv_scalar[88]
        getitem_1925: "f32[3072, 768]" = _foreach_addcdiv_scalar[89]
        getitem_1926: "f32[3072]" = _foreach_addcdiv_scalar[90]
        getitem_1927: "f32[768, 3072]" = _foreach_addcdiv_scalar[91]
        getitem_1928: "f32[768]" = _foreach_addcdiv_scalar[92]
        getitem_1929: "f32[768]" = _foreach_addcdiv_scalar[93]
        getitem_1930: "f32[768]" = _foreach_addcdiv_scalar[94]
        getitem_1931: "f32[768]" = _foreach_addcdiv_scalar[95]
        getitem_1932: "f32[768]" = _foreach_addcdiv_scalar[96]
        getitem_1933: "f32[3072, 768]" = _foreach_addcdiv_scalar[97]
        getitem_1934: "f32[3072]" = _foreach_addcdiv_scalar[98]
        getitem_1935: "f32[768, 3072]" = _foreach_addcdiv_scalar[99]
        getitem_1936: "f32[768]" = _foreach_addcdiv_scalar[100]
        getitem_1937: "f32[768]" = _foreach_addcdiv_scalar[101]
        getitem_1938: "f32[768]" = _foreach_addcdiv_scalar[102]
        getitem_1939: "f32[32000]" = _foreach_addcdiv_scalar[103]
        getitem_1940: "f32[768, 768]" = _foreach_addcdiv_scalar[104]
        getitem_1941: "f32[768]" = _foreach_addcdiv_scalar[105]
        getitem_1942: "f32[768]" = _foreach_addcdiv_scalar[106]
        getitem_1943: "f32[768]" = _foreach_addcdiv_scalar[107];  _foreach_addcdiv_scalar = None
        copy__default: "f32[32000, 768]" = torch.ops.aten.copy_.default(arg0_1, getitem_1836);  arg0_1 = getitem_1836 = None
        copy__default_1: "f32[512, 768]" = torch.ops.aten.copy_.default(arg1_1, getitem_1837);  arg1_1 = getitem_1837 = None
        copy__default_2: "f32[4, 768]" = torch.ops.aten.copy_.default(arg2_1, getitem_1838);  arg2_1 = getitem_1838 = None
        copy__default_3: "f32[768]" = torch.ops.aten.copy_.default(arg3_1, getitem_1839);  arg3_1 = getitem_1839 = None
        copy__default_4: "f32[768]" = torch.ops.aten.copy_.default(arg4_1, getitem_1840);  arg4_1 = getitem_1840 = None
        copy__default_5: "f32[768, 768]" = torch.ops.aten.copy_.default(arg5_1, getitem_1841);  arg5_1 = getitem_1841 = None
        copy__default_6: "f32[768]" = torch.ops.aten.copy_.default(arg6_1, getitem_1842);  arg6_1 = getitem_1842 = None
        copy__default_7: "f32[768]" = torch.ops.aten.copy_.default(arg7_1, getitem_1843);  arg7_1 = getitem_1843 = None
        copy__default_8: "f32[768]" = torch.ops.aten.copy_.default(arg8_1, getitem_1844);  arg8_1 = getitem_1844 = None
        copy__default_9: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg9_1, getitem_1845);  arg9_1 = getitem_1845 = None
        copy__default_10: "f32[3072]" = torch.ops.aten.copy_.default(arg10_1, getitem_1846);  arg10_1 = getitem_1846 = None
        copy__default_11: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg11_1, getitem_1847);  arg11_1 = getitem_1847 = None
        copy__default_12: "f32[768]" = torch.ops.aten.copy_.default(arg12_1, getitem_1848);  arg12_1 = getitem_1848 = None
        copy__default_13: "f32[768]" = torch.ops.aten.copy_.default(arg13_1, getitem_1849);  arg13_1 = getitem_1849 = None
        copy__default_14: "f32[768]" = torch.ops.aten.copy_.default(arg14_1, getitem_1850);  arg14_1 = getitem_1850 = None
        copy__default_15: "f32[768]" = torch.ops.aten.copy_.default(arg15_1, getitem_1851);  arg15_1 = getitem_1851 = None
        copy__default_16: "f32[768]" = torch.ops.aten.copy_.default(arg16_1, getitem_1852);  arg16_1 = getitem_1852 = None
        copy__default_17: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg17_1, getitem_1853);  arg17_1 = getitem_1853 = None
        copy__default_18: "f32[3072]" = torch.ops.aten.copy_.default(arg18_1, getitem_1854);  arg18_1 = getitem_1854 = None
        copy__default_19: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg19_1, getitem_1855);  arg19_1 = getitem_1855 = None
        copy__default_20: "f32[768]" = torch.ops.aten.copy_.default(arg20_1, getitem_1856);  arg20_1 = getitem_1856 = None
        copy__default_21: "f32[768]" = torch.ops.aten.copy_.default(arg21_1, getitem_1857);  arg21_1 = getitem_1857 = None
        copy__default_22: "f32[768]" = torch.ops.aten.copy_.default(arg22_1, getitem_1858);  arg22_1 = getitem_1858 = None
        copy__default_23: "f32[768]" = torch.ops.aten.copy_.default(arg23_1, getitem_1859);  arg23_1 = getitem_1859 = None
        copy__default_24: "f32[768]" = torch.ops.aten.copy_.default(arg24_1, getitem_1860);  arg24_1 = getitem_1860 = None
        copy__default_25: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg25_1, getitem_1861);  arg25_1 = getitem_1861 = None
        copy__default_26: "f32[3072]" = torch.ops.aten.copy_.default(arg26_1, getitem_1862);  arg26_1 = getitem_1862 = None
        copy__default_27: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg27_1, getitem_1863);  arg27_1 = getitem_1863 = None
        copy__default_28: "f32[768]" = torch.ops.aten.copy_.default(arg28_1, getitem_1864);  arg28_1 = getitem_1864 = None
        copy__default_29: "f32[768]" = torch.ops.aten.copy_.default(arg29_1, getitem_1865);  arg29_1 = getitem_1865 = None
        copy__default_30: "f32[768]" = torch.ops.aten.copy_.default(arg30_1, getitem_1866);  arg30_1 = getitem_1866 = None
        copy__default_31: "f32[768]" = torch.ops.aten.copy_.default(arg31_1, getitem_1867);  arg31_1 = getitem_1867 = None
        copy__default_32: "f32[768]" = torch.ops.aten.copy_.default(arg32_1, getitem_1868);  arg32_1 = getitem_1868 = None
        copy__default_33: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg33_1, getitem_1869);  arg33_1 = getitem_1869 = None
        copy__default_34: "f32[3072]" = torch.ops.aten.copy_.default(arg34_1, getitem_1870);  arg34_1 = getitem_1870 = None
        copy__default_35: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg35_1, getitem_1871);  arg35_1 = getitem_1871 = None
        copy__default_36: "f32[768]" = torch.ops.aten.copy_.default(arg36_1, getitem_1872);  arg36_1 = getitem_1872 = None
        copy__default_37: "f32[768]" = torch.ops.aten.copy_.default(arg37_1, getitem_1873);  arg37_1 = getitem_1873 = None
        copy__default_38: "f32[768]" = torch.ops.aten.copy_.default(arg38_1, getitem_1874);  arg38_1 = getitem_1874 = None
        copy__default_39: "f32[768]" = torch.ops.aten.copy_.default(arg39_1, getitem_1875);  arg39_1 = getitem_1875 = None
        copy__default_40: "f32[768]" = torch.ops.aten.copy_.default(arg40_1, getitem_1876);  arg40_1 = getitem_1876 = None
        copy__default_41: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg41_1, getitem_1877);  arg41_1 = getitem_1877 = None
        copy__default_42: "f32[3072]" = torch.ops.aten.copy_.default(arg42_1, getitem_1878);  arg42_1 = getitem_1878 = None
        copy__default_43: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg43_1, getitem_1879);  arg43_1 = getitem_1879 = None
        copy__default_44: "f32[768]" = torch.ops.aten.copy_.default(arg44_1, getitem_1880);  arg44_1 = getitem_1880 = None
        copy__default_45: "f32[768]" = torch.ops.aten.copy_.default(arg45_1, getitem_1881);  arg45_1 = getitem_1881 = None
        copy__default_46: "f32[768]" = torch.ops.aten.copy_.default(arg46_1, getitem_1882);  arg46_1 = getitem_1882 = None
        copy__default_47: "f32[768]" = torch.ops.aten.copy_.default(arg47_1, getitem_1883);  arg47_1 = getitem_1883 = None
        copy__default_48: "f32[768]" = torch.ops.aten.copy_.default(arg48_1, getitem_1884);  arg48_1 = getitem_1884 = None
        copy__default_49: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg49_1, getitem_1885);  arg49_1 = getitem_1885 = None
        copy__default_50: "f32[3072]" = torch.ops.aten.copy_.default(arg50_1, getitem_1886);  arg50_1 = getitem_1886 = None
        copy__default_51: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg51_1, getitem_1887);  arg51_1 = getitem_1887 = None
        copy__default_52: "f32[768]" = torch.ops.aten.copy_.default(arg52_1, getitem_1888);  arg52_1 = getitem_1888 = None
        copy__default_53: "f32[768]" = torch.ops.aten.copy_.default(arg53_1, getitem_1889);  arg53_1 = getitem_1889 = None
        copy__default_54: "f32[768]" = torch.ops.aten.copy_.default(arg54_1, getitem_1890);  arg54_1 = getitem_1890 = None
        copy__default_55: "f32[768]" = torch.ops.aten.copy_.default(arg55_1, getitem_1891);  arg55_1 = getitem_1891 = None
        copy__default_56: "f32[768]" = torch.ops.aten.copy_.default(arg56_1, getitem_1892);  arg56_1 = getitem_1892 = None
        copy__default_57: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg57_1, getitem_1893);  arg57_1 = getitem_1893 = None
        copy__default_58: "f32[3072]" = torch.ops.aten.copy_.default(arg58_1, getitem_1894);  arg58_1 = getitem_1894 = None
        copy__default_59: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg59_1, getitem_1895);  arg59_1 = getitem_1895 = None
        copy__default_60: "f32[768]" = torch.ops.aten.copy_.default(arg60_1, getitem_1896);  arg60_1 = getitem_1896 = None
        copy__default_61: "f32[768]" = torch.ops.aten.copy_.default(arg61_1, getitem_1897);  arg61_1 = getitem_1897 = None
        copy__default_62: "f32[768]" = torch.ops.aten.copy_.default(arg62_1, getitem_1898);  arg62_1 = getitem_1898 = None
        copy__default_63: "f32[768]" = torch.ops.aten.copy_.default(arg63_1, getitem_1899);  arg63_1 = getitem_1899 = None
        copy__default_64: "f32[768]" = torch.ops.aten.copy_.default(arg64_1, getitem_1900);  arg64_1 = getitem_1900 = None
        copy__default_65: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg65_1, getitem_1901);  arg65_1 = getitem_1901 = None
        copy__default_66: "f32[3072]" = torch.ops.aten.copy_.default(arg66_1, getitem_1902);  arg66_1 = getitem_1902 = None
        copy__default_67: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg67_1, getitem_1903);  arg67_1 = getitem_1903 = None
        copy__default_68: "f32[768]" = torch.ops.aten.copy_.default(arg68_1, getitem_1904);  arg68_1 = getitem_1904 = None
        copy__default_69: "f32[768]" = torch.ops.aten.copy_.default(arg69_1, getitem_1905);  arg69_1 = getitem_1905 = None
        copy__default_70: "f32[768]" = torch.ops.aten.copy_.default(arg70_1, getitem_1906);  arg70_1 = getitem_1906 = None
        copy__default_71: "f32[768]" = torch.ops.aten.copy_.default(arg71_1, getitem_1907);  arg71_1 = getitem_1907 = None
        copy__default_72: "f32[768]" = torch.ops.aten.copy_.default(arg72_1, getitem_1908);  arg72_1 = getitem_1908 = None
        copy__default_73: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg73_1, getitem_1909);  arg73_1 = getitem_1909 = None
        copy__default_74: "f32[3072]" = torch.ops.aten.copy_.default(arg74_1, getitem_1910);  arg74_1 = getitem_1910 = None
        copy__default_75: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg75_1, getitem_1911);  arg75_1 = getitem_1911 = None
        copy__default_76: "f32[768]" = torch.ops.aten.copy_.default(arg76_1, getitem_1912);  arg76_1 = getitem_1912 = None
        copy__default_77: "f32[768]" = torch.ops.aten.copy_.default(arg77_1, getitem_1913);  arg77_1 = getitem_1913 = None
        copy__default_78: "f32[768]" = torch.ops.aten.copy_.default(arg78_1, getitem_1914);  arg78_1 = getitem_1914 = None
        copy__default_79: "f32[768]" = torch.ops.aten.copy_.default(arg79_1, getitem_1915);  arg79_1 = getitem_1915 = None
        copy__default_80: "f32[768]" = torch.ops.aten.copy_.default(arg80_1, getitem_1916);  arg80_1 = getitem_1916 = None
        copy__default_81: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg81_1, getitem_1917);  arg81_1 = getitem_1917 = None
        copy__default_82: "f32[3072]" = torch.ops.aten.copy_.default(arg82_1, getitem_1918);  arg82_1 = getitem_1918 = None
        copy__default_83: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg83_1, getitem_1919);  arg83_1 = getitem_1919 = None
        copy__default_84: "f32[768]" = torch.ops.aten.copy_.default(arg84_1, getitem_1920);  arg84_1 = getitem_1920 = None
        copy__default_85: "f32[768]" = torch.ops.aten.copy_.default(arg85_1, getitem_1921);  arg85_1 = getitem_1921 = None
        copy__default_86: "f32[768]" = torch.ops.aten.copy_.default(arg86_1, getitem_1922);  arg86_1 = getitem_1922 = None
        copy__default_87: "f32[768]" = torch.ops.aten.copy_.default(arg87_1, getitem_1923);  arg87_1 = getitem_1923 = None
        copy__default_88: "f32[768]" = torch.ops.aten.copy_.default(arg88_1, getitem_1924);  arg88_1 = getitem_1924 = None
        copy__default_89: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg89_1, getitem_1925);  arg89_1 = getitem_1925 = None
        copy__default_90: "f32[3072]" = torch.ops.aten.copy_.default(arg90_1, getitem_1926);  arg90_1 = getitem_1926 = None
        copy__default_91: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg91_1, getitem_1927);  arg91_1 = getitem_1927 = None
        copy__default_92: "f32[768]" = torch.ops.aten.copy_.default(arg92_1, getitem_1928);  arg92_1 = getitem_1928 = None
        copy__default_93: "f32[768]" = torch.ops.aten.copy_.default(arg93_1, getitem_1929);  arg93_1 = getitem_1929 = None
        copy__default_94: "f32[768]" = torch.ops.aten.copy_.default(arg94_1, getitem_1930);  arg94_1 = getitem_1930 = None
        copy__default_95: "f32[768]" = torch.ops.aten.copy_.default(arg95_1, getitem_1931);  arg95_1 = getitem_1931 = None
        copy__default_96: "f32[768]" = torch.ops.aten.copy_.default(arg96_1, getitem_1932);  arg96_1 = getitem_1932 = None
        copy__default_97: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg97_1, getitem_1933);  arg97_1 = getitem_1933 = None
        copy__default_98: "f32[3072]" = torch.ops.aten.copy_.default(arg98_1, getitem_1934);  arg98_1 = getitem_1934 = None
        copy__default_99: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg99_1, getitem_1935);  arg99_1 = getitem_1935 = None
        copy__default_100: "f32[768]" = torch.ops.aten.copy_.default(arg100_1, getitem_1936);  arg100_1 = getitem_1936 = None
        copy__default_101: "f32[768]" = torch.ops.aten.copy_.default(arg101_1, getitem_1937);  arg101_1 = getitem_1937 = None
        copy__default_102: "f32[768]" = torch.ops.aten.copy_.default(arg102_1, getitem_1938);  arg102_1 = getitem_1938 = None
        copy__default_103: "f32[32000]" = torch.ops.aten.copy_.default(arg103_1, getitem_1939);  arg103_1 = getitem_1939 = None
        copy__default_104: "f32[768, 768]" = torch.ops.aten.copy_.default(arg104_1, getitem_1940);  arg104_1 = getitem_1940 = None
        copy__default_105: "f32[768]" = torch.ops.aten.copy_.default(arg105_1, getitem_1941);  arg105_1 = getitem_1941 = None
        copy__default_106: "f32[768]" = torch.ops.aten.copy_.default(arg106_1, getitem_1942);  arg106_1 = getitem_1942 = None
        copy__default_107: "f32[768]" = torch.ops.aten.copy_.default(arg107_1, getitem_1943);  arg107_1 = getitem_1943 = None
        copy__default_108: "f32[]" = torch.ops.aten.copy_.default(arg108_1, getitem_1);  arg108_1 = getitem_1 = None
        copy__default_109: "f32[512, 768]" = torch.ops.aten.copy_.default(arg109_1, getitem_217);  arg109_1 = getitem_217 = None
        copy__default_110: "f32[512, 768]" = torch.ops.aten.copy_.default(arg110_1, getitem_433);  arg110_1 = getitem_433 = None
        copy__default_111: "f32[32000, 768]" = torch.ops.aten.copy_.default(arg111_1, getitem_216);  arg111_1 = getitem_216 = None
        copy__default_112: "f32[32000, 768]" = torch.ops.aten.copy_.default(arg112_1, getitem_432);  arg112_1 = getitem_432 = None
        copy__default_113: "f32[]" = torch.ops.aten.copy_.default(arg113_1, getitem);  arg113_1 = getitem = None
        copy__default_114: "f32[]" = torch.ops.aten.copy_.default(arg114_1, getitem_2);  arg114_1 = getitem_2 = None
        copy__default_115: "f32[]" = torch.ops.aten.copy_.default(arg115_1, getitem_3);  arg115_1 = getitem_3 = None
        copy__default_116: "f32[]" = torch.ops.aten.copy_.default(arg116_1, getitem_4);  arg116_1 = getitem_4 = None
        copy__default_117: "f32[]" = torch.ops.aten.copy_.default(arg117_1, getitem_5);  arg117_1 = getitem_5 = None
        copy__default_118: "f32[]" = torch.ops.aten.copy_.default(arg118_1, getitem_6);  arg118_1 = getitem_6 = None
        copy__default_119: "f32[]" = torch.ops.aten.copy_.default(arg119_1, getitem_7);  arg119_1 = getitem_7 = None
        copy__default_120: "f32[]" = torch.ops.aten.copy_.default(arg120_1, getitem_8);  arg120_1 = getitem_8 = None
        copy__default_121: "f32[]" = torch.ops.aten.copy_.default(arg121_1, getitem_9);  arg121_1 = getitem_9 = None
        copy__default_122: "f32[]" = torch.ops.aten.copy_.default(arg122_1, getitem_10);  arg122_1 = getitem_10 = None
        copy__default_123: "f32[]" = torch.ops.aten.copy_.default(arg123_1, getitem_11);  arg123_1 = getitem_11 = None
        copy__default_124: "f32[]" = torch.ops.aten.copy_.default(arg124_1, getitem_12);  arg124_1 = getitem_12 = None
        copy__default_125: "f32[]" = torch.ops.aten.copy_.default(arg125_1, getitem_13);  arg125_1 = getitem_13 = None
        copy__default_126: "f32[]" = torch.ops.aten.copy_.default(arg126_1, getitem_14);  arg126_1 = getitem_14 = None
        copy__default_127: "f32[]" = torch.ops.aten.copy_.default(arg127_1, getitem_15);  arg127_1 = getitem_15 = None
        copy__default_128: "f32[]" = torch.ops.aten.copy_.default(arg128_1, getitem_16);  arg128_1 = getitem_16 = None
        copy__default_129: "f32[]" = torch.ops.aten.copy_.default(arg129_1, getitem_17);  arg129_1 = getitem_17 = None
        copy__default_130: "f32[]" = torch.ops.aten.copy_.default(arg130_1, getitem_18);  arg130_1 = getitem_18 = None
        copy__default_131: "f32[]" = torch.ops.aten.copy_.default(arg131_1, getitem_19);  arg131_1 = getitem_19 = None
        copy__default_132: "f32[]" = torch.ops.aten.copy_.default(arg132_1, getitem_20);  arg132_1 = getitem_20 = None
        copy__default_133: "f32[]" = torch.ops.aten.copy_.default(arg133_1, getitem_21);  arg133_1 = getitem_21 = None
        copy__default_134: "f32[]" = torch.ops.aten.copy_.default(arg134_1, getitem_22);  arg134_1 = getitem_22 = None
        copy__default_135: "f32[]" = torch.ops.aten.copy_.default(arg135_1, getitem_23);  arg135_1 = getitem_23 = None
        copy__default_136: "f32[]" = torch.ops.aten.copy_.default(arg136_1, getitem_24);  arg136_1 = getitem_24 = None
        copy__default_137: "f32[]" = torch.ops.aten.copy_.default(arg137_1, getitem_25);  arg137_1 = getitem_25 = None
        copy__default_138: "f32[]" = torch.ops.aten.copy_.default(arg138_1, getitem_26);  arg138_1 = getitem_26 = None
        copy__default_139: "f32[]" = torch.ops.aten.copy_.default(arg139_1, getitem_27);  arg139_1 = getitem_27 = None
        copy__default_140: "f32[]" = torch.ops.aten.copy_.default(arg140_1, getitem_28);  arg140_1 = getitem_28 = None
        copy__default_141: "f32[]" = torch.ops.aten.copy_.default(arg141_1, getitem_29);  arg141_1 = getitem_29 = None
        copy__default_142: "f32[]" = torch.ops.aten.copy_.default(arg142_1, getitem_30);  arg142_1 = getitem_30 = None
        copy__default_143: "f32[]" = torch.ops.aten.copy_.default(arg143_1, getitem_31);  arg143_1 = getitem_31 = None
        copy__default_144: "f32[]" = torch.ops.aten.copy_.default(arg144_1, getitem_32);  arg144_1 = getitem_32 = None
        copy__default_145: "f32[]" = torch.ops.aten.copy_.default(arg145_1, getitem_33);  arg145_1 = getitem_33 = None
        copy__default_146: "f32[]" = torch.ops.aten.copy_.default(arg146_1, getitem_34);  arg146_1 = getitem_34 = None
        copy__default_147: "f32[]" = torch.ops.aten.copy_.default(arg147_1, getitem_35);  arg147_1 = getitem_35 = None
        copy__default_148: "f32[]" = torch.ops.aten.copy_.default(arg148_1, getitem_36);  arg148_1 = getitem_36 = None
        copy__default_149: "f32[]" = torch.ops.aten.copy_.default(arg149_1, getitem_37);  arg149_1 = getitem_37 = None
        copy__default_150: "f32[]" = torch.ops.aten.copy_.default(arg150_1, getitem_38);  arg150_1 = getitem_38 = None
        copy__default_151: "f32[]" = torch.ops.aten.copy_.default(arg151_1, getitem_39);  arg151_1 = getitem_39 = None
        copy__default_152: "f32[]" = torch.ops.aten.copy_.default(arg152_1, getitem_40);  arg152_1 = getitem_40 = None
        copy__default_153: "f32[]" = torch.ops.aten.copy_.default(arg153_1, getitem_41);  arg153_1 = getitem_41 = None
        copy__default_154: "f32[]" = torch.ops.aten.copy_.default(arg154_1, getitem_42);  arg154_1 = getitem_42 = None
        copy__default_155: "f32[]" = torch.ops.aten.copy_.default(arg155_1, getitem_43);  arg155_1 = getitem_43 = None
        copy__default_156: "f32[]" = torch.ops.aten.copy_.default(arg156_1, getitem_44);  arg156_1 = getitem_44 = None
        copy__default_157: "f32[]" = torch.ops.aten.copy_.default(arg157_1, getitem_45);  arg157_1 = getitem_45 = None
        copy__default_158: "f32[]" = torch.ops.aten.copy_.default(arg158_1, getitem_46);  arg158_1 = getitem_46 = None
        copy__default_159: "f32[]" = torch.ops.aten.copy_.default(arg159_1, getitem_47);  arg159_1 = getitem_47 = None
        copy__default_160: "f32[]" = torch.ops.aten.copy_.default(arg160_1, getitem_48);  arg160_1 = getitem_48 = None
        copy__default_161: "f32[]" = torch.ops.aten.copy_.default(arg161_1, getitem_49);  arg161_1 = getitem_49 = None
        copy__default_162: "f32[]" = torch.ops.aten.copy_.default(arg162_1, getitem_50);  arg162_1 = getitem_50 = None
        copy__default_163: "f32[]" = torch.ops.aten.copy_.default(arg163_1, getitem_51);  arg163_1 = getitem_51 = None
        copy__default_164: "f32[]" = torch.ops.aten.copy_.default(arg164_1, getitem_52);  arg164_1 = getitem_52 = None
        copy__default_165: "f32[]" = torch.ops.aten.copy_.default(arg165_1, getitem_53);  arg165_1 = getitem_53 = None
        copy__default_166: "f32[]" = torch.ops.aten.copy_.default(arg166_1, getitem_54);  arg166_1 = getitem_54 = None
        copy__default_167: "f32[]" = torch.ops.aten.copy_.default(arg167_1, getitem_55);  arg167_1 = getitem_55 = None
        copy__default_168: "f32[]" = torch.ops.aten.copy_.default(arg168_1, getitem_56);  arg168_1 = getitem_56 = None
        copy__default_169: "f32[]" = torch.ops.aten.copy_.default(arg169_1, getitem_57);  arg169_1 = getitem_57 = None
        copy__default_170: "f32[]" = torch.ops.aten.copy_.default(arg170_1, getitem_58);  arg170_1 = getitem_58 = None
        copy__default_171: "f32[]" = torch.ops.aten.copy_.default(arg171_1, getitem_59);  arg171_1 = getitem_59 = None
        copy__default_172: "f32[]" = torch.ops.aten.copy_.default(arg172_1, getitem_60);  arg172_1 = getitem_60 = None
        copy__default_173: "f32[]" = torch.ops.aten.copy_.default(arg173_1, getitem_61);  arg173_1 = getitem_61 = None
        copy__default_174: "f32[]" = torch.ops.aten.copy_.default(arg174_1, getitem_62);  arg174_1 = getitem_62 = None
        copy__default_175: "f32[]" = torch.ops.aten.copy_.default(arg175_1, getitem_63);  arg175_1 = getitem_63 = None
        copy__default_176: "f32[]" = torch.ops.aten.copy_.default(arg176_1, getitem_64);  arg176_1 = getitem_64 = None
        copy__default_177: "f32[]" = torch.ops.aten.copy_.default(arg177_1, getitem_65);  arg177_1 = getitem_65 = None
        copy__default_178: "f32[]" = torch.ops.aten.copy_.default(arg178_1, getitem_66);  arg178_1 = getitem_66 = None
        copy__default_179: "f32[]" = torch.ops.aten.copy_.default(arg179_1, getitem_67);  arg179_1 = getitem_67 = None
        copy__default_180: "f32[]" = torch.ops.aten.copy_.default(arg180_1, getitem_68);  arg180_1 = getitem_68 = None
        copy__default_181: "f32[]" = torch.ops.aten.copy_.default(arg181_1, getitem_69);  arg181_1 = getitem_69 = None
        copy__default_182: "f32[]" = torch.ops.aten.copy_.default(arg182_1, getitem_70);  arg182_1 = getitem_70 = None
        copy__default_183: "f32[]" = torch.ops.aten.copy_.default(arg183_1, getitem_71);  arg183_1 = getitem_71 = None
        copy__default_184: "f32[]" = torch.ops.aten.copy_.default(arg184_1, getitem_72);  arg184_1 = getitem_72 = None
        copy__default_185: "f32[]" = torch.ops.aten.copy_.default(arg185_1, getitem_73);  arg185_1 = getitem_73 = None
        copy__default_186: "f32[]" = torch.ops.aten.copy_.default(arg186_1, getitem_74);  arg186_1 = getitem_74 = None
        copy__default_187: "f32[]" = torch.ops.aten.copy_.default(arg187_1, getitem_75);  arg187_1 = getitem_75 = None
        copy__default_188: "f32[]" = torch.ops.aten.copy_.default(arg188_1, getitem_76);  arg188_1 = getitem_76 = None
        copy__default_189: "f32[]" = torch.ops.aten.copy_.default(arg189_1, getitem_77);  arg189_1 = getitem_77 = None
        copy__default_190: "f32[]" = torch.ops.aten.copy_.default(arg190_1, getitem_78);  arg190_1 = getitem_78 = None
        copy__default_191: "f32[]" = torch.ops.aten.copy_.default(arg191_1, getitem_79);  arg191_1 = getitem_79 = None
        copy__default_192: "f32[]" = torch.ops.aten.copy_.default(arg192_1, getitem_80);  arg192_1 = getitem_80 = None
        copy__default_193: "f32[]" = torch.ops.aten.copy_.default(arg193_1, getitem_81);  arg193_1 = getitem_81 = None
        copy__default_194: "f32[]" = torch.ops.aten.copy_.default(arg194_1, getitem_82);  arg194_1 = getitem_82 = None
        copy__default_195: "f32[]" = torch.ops.aten.copy_.default(arg195_1, getitem_83);  arg195_1 = getitem_83 = None
        copy__default_196: "f32[]" = torch.ops.aten.copy_.default(arg196_1, getitem_84);  arg196_1 = getitem_84 = None
        copy__default_197: "f32[]" = torch.ops.aten.copy_.default(arg197_1, getitem_85);  arg197_1 = getitem_85 = None
        copy__default_198: "f32[]" = torch.ops.aten.copy_.default(arg198_1, getitem_86);  arg198_1 = getitem_86 = None
        copy__default_199: "f32[]" = torch.ops.aten.copy_.default(arg199_1, getitem_87);  arg199_1 = getitem_87 = None
        copy__default_200: "f32[]" = torch.ops.aten.copy_.default(arg200_1, getitem_88);  arg200_1 = getitem_88 = None
        copy__default_201: "f32[]" = torch.ops.aten.copy_.default(arg201_1, getitem_89);  arg201_1 = getitem_89 = None
        copy__default_202: "f32[]" = torch.ops.aten.copy_.default(arg202_1, getitem_90);  arg202_1 = getitem_90 = None
        copy__default_203: "f32[]" = torch.ops.aten.copy_.default(arg203_1, getitem_91);  arg203_1 = getitem_91 = None
        copy__default_204: "f32[]" = torch.ops.aten.copy_.default(arg204_1, getitem_92);  arg204_1 = getitem_92 = None
        copy__default_205: "f32[]" = torch.ops.aten.copy_.default(arg205_1, getitem_93);  arg205_1 = getitem_93 = None
        copy__default_206: "f32[]" = torch.ops.aten.copy_.default(arg206_1, getitem_94);  arg206_1 = getitem_94 = None
        copy__default_207: "f32[]" = torch.ops.aten.copy_.default(arg207_1, getitem_95);  arg207_1 = getitem_95 = None
        copy__default_208: "f32[]" = torch.ops.aten.copy_.default(arg208_1, getitem_96);  arg208_1 = getitem_96 = None
        copy__default_209: "f32[]" = torch.ops.aten.copy_.default(arg209_1, getitem_97);  arg209_1 = getitem_97 = None
        copy__default_210: "f32[]" = torch.ops.aten.copy_.default(arg210_1, getitem_98);  arg210_1 = getitem_98 = None
        copy__default_211: "f32[]" = torch.ops.aten.copy_.default(arg211_1, getitem_99);  arg211_1 = getitem_99 = None
        copy__default_212: "f32[]" = torch.ops.aten.copy_.default(arg212_1, getitem_100);  arg212_1 = getitem_100 = None
        copy__default_213: "f32[]" = torch.ops.aten.copy_.default(arg213_1, getitem_101);  arg213_1 = getitem_101 = None
        copy__default_214: "f32[]" = torch.ops.aten.copy_.default(arg214_1, getitem_102);  arg214_1 = getitem_102 = None
        copy__default_215: "f32[]" = torch.ops.aten.copy_.default(arg215_1, getitem_103);  arg215_1 = getitem_103 = None
        copy__default_216: "f32[]" = torch.ops.aten.copy_.default(arg216_1, getitem_104);  arg216_1 = getitem_104 = None
        copy__default_217: "f32[]" = torch.ops.aten.copy_.default(arg217_1, getitem_105);  arg217_1 = getitem_105 = None
        copy__default_218: "f32[]" = torch.ops.aten.copy_.default(arg218_1, getitem_106);  arg218_1 = getitem_106 = None
        copy__default_219: "f32[]" = torch.ops.aten.copy_.default(arg219_1, getitem_107);  arg219_1 = getitem_107 = None
        copy__default_220: "f32[4, 768]" = torch.ops.aten.copy_.default(arg220_1, getitem_218);  arg220_1 = getitem_218 = None
        copy__default_221: "f32[768]" = torch.ops.aten.copy_.default(arg221_1, getitem_219);  arg221_1 = getitem_219 = None
        copy__default_222: "f32[768]" = torch.ops.aten.copy_.default(arg222_1, getitem_220);  arg222_1 = getitem_220 = None
        copy__default_223: "f32[768, 768]" = torch.ops.aten.copy_.default(arg223_1, getitem_221);  arg223_1 = getitem_221 = None
        copy__default_224: "f32[768]" = torch.ops.aten.copy_.default(arg224_1, getitem_222);  arg224_1 = getitem_222 = None
        copy__default_225: "f32[768]" = torch.ops.aten.copy_.default(arg225_1, getitem_223);  arg225_1 = getitem_223 = None
        copy__default_226: "f32[768]" = torch.ops.aten.copy_.default(arg226_1, getitem_224);  arg226_1 = getitem_224 = None
        copy__default_227: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg227_1, getitem_225);  arg227_1 = getitem_225 = None
        copy__default_228: "f32[3072]" = torch.ops.aten.copy_.default(arg228_1, getitem_226);  arg228_1 = getitem_226 = None
        copy__default_229: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg229_1, getitem_227);  arg229_1 = getitem_227 = None
        copy__default_230: "f32[768]" = torch.ops.aten.copy_.default(arg230_1, getitem_228);  arg230_1 = getitem_228 = None
        copy__default_231: "f32[768]" = torch.ops.aten.copy_.default(arg231_1, getitem_229);  arg231_1 = getitem_229 = None
        copy__default_232: "f32[768]" = torch.ops.aten.copy_.default(arg232_1, getitem_230);  arg232_1 = getitem_230 = None
        copy__default_233: "f32[768]" = torch.ops.aten.copy_.default(arg233_1, getitem_231);  arg233_1 = getitem_231 = None
        copy__default_234: "f32[768]" = torch.ops.aten.copy_.default(arg234_1, getitem_232);  arg234_1 = getitem_232 = None
        copy__default_235: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg235_1, getitem_233);  arg235_1 = getitem_233 = None
        copy__default_236: "f32[3072]" = torch.ops.aten.copy_.default(arg236_1, getitem_234);  arg236_1 = getitem_234 = None
        copy__default_237: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg237_1, getitem_235);  arg237_1 = getitem_235 = None
        copy__default_238: "f32[768]" = torch.ops.aten.copy_.default(arg238_1, getitem_236);  arg238_1 = getitem_236 = None
        copy__default_239: "f32[768]" = torch.ops.aten.copy_.default(arg239_1, getitem_237);  arg239_1 = getitem_237 = None
        copy__default_240: "f32[768]" = torch.ops.aten.copy_.default(arg240_1, getitem_238);  arg240_1 = getitem_238 = None
        copy__default_241: "f32[768]" = torch.ops.aten.copy_.default(arg241_1, getitem_239);  arg241_1 = getitem_239 = None
        copy__default_242: "f32[768]" = torch.ops.aten.copy_.default(arg242_1, getitem_240);  arg242_1 = getitem_240 = None
        copy__default_243: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg243_1, getitem_241);  arg243_1 = getitem_241 = None
        copy__default_244: "f32[3072]" = torch.ops.aten.copy_.default(arg244_1, getitem_242);  arg244_1 = getitem_242 = None
        copy__default_245: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg245_1, getitem_243);  arg245_1 = getitem_243 = None
        copy__default_246: "f32[768]" = torch.ops.aten.copy_.default(arg246_1, getitem_244);  arg246_1 = getitem_244 = None
        copy__default_247: "f32[768]" = torch.ops.aten.copy_.default(arg247_1, getitem_245);  arg247_1 = getitem_245 = None
        copy__default_248: "f32[768]" = torch.ops.aten.copy_.default(arg248_1, getitem_246);  arg248_1 = getitem_246 = None
        copy__default_249: "f32[768]" = torch.ops.aten.copy_.default(arg249_1, getitem_247);  arg249_1 = getitem_247 = None
        copy__default_250: "f32[768]" = torch.ops.aten.copy_.default(arg250_1, getitem_248);  arg250_1 = getitem_248 = None
        copy__default_251: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg251_1, getitem_249);  arg251_1 = getitem_249 = None
        copy__default_252: "f32[3072]" = torch.ops.aten.copy_.default(arg252_1, getitem_250);  arg252_1 = getitem_250 = None
        copy__default_253: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg253_1, getitem_251);  arg253_1 = getitem_251 = None
        copy__default_254: "f32[768]" = torch.ops.aten.copy_.default(arg254_1, getitem_252);  arg254_1 = getitem_252 = None
        copy__default_255: "f32[768]" = torch.ops.aten.copy_.default(arg255_1, getitem_253);  arg255_1 = getitem_253 = None
        copy__default_256: "f32[768]" = torch.ops.aten.copy_.default(arg256_1, getitem_254);  arg256_1 = getitem_254 = None
        copy__default_257: "f32[768]" = torch.ops.aten.copy_.default(arg257_1, getitem_255);  arg257_1 = getitem_255 = None
        copy__default_258: "f32[768]" = torch.ops.aten.copy_.default(arg258_1, getitem_256);  arg258_1 = getitem_256 = None
        copy__default_259: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg259_1, getitem_257);  arg259_1 = getitem_257 = None
        copy__default_260: "f32[3072]" = torch.ops.aten.copy_.default(arg260_1, getitem_258);  arg260_1 = getitem_258 = None
        copy__default_261: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg261_1, getitem_259);  arg261_1 = getitem_259 = None
        copy__default_262: "f32[768]" = torch.ops.aten.copy_.default(arg262_1, getitem_260);  arg262_1 = getitem_260 = None
        copy__default_263: "f32[768]" = torch.ops.aten.copy_.default(arg263_1, getitem_261);  arg263_1 = getitem_261 = None
        copy__default_264: "f32[768]" = torch.ops.aten.copy_.default(arg264_1, getitem_262);  arg264_1 = getitem_262 = None
        copy__default_265: "f32[768]" = torch.ops.aten.copy_.default(arg265_1, getitem_263);  arg265_1 = getitem_263 = None
        copy__default_266: "f32[768]" = torch.ops.aten.copy_.default(arg266_1, getitem_264);  arg266_1 = getitem_264 = None
        copy__default_267: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg267_1, getitem_265);  arg267_1 = getitem_265 = None
        copy__default_268: "f32[3072]" = torch.ops.aten.copy_.default(arg268_1, getitem_266);  arg268_1 = getitem_266 = None
        copy__default_269: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg269_1, getitem_267);  arg269_1 = getitem_267 = None
        copy__default_270: "f32[768]" = torch.ops.aten.copy_.default(arg270_1, getitem_268);  arg270_1 = getitem_268 = None
        copy__default_271: "f32[768]" = torch.ops.aten.copy_.default(arg271_1, getitem_269);  arg271_1 = getitem_269 = None
        copy__default_272: "f32[768]" = torch.ops.aten.copy_.default(arg272_1, getitem_270);  arg272_1 = getitem_270 = None
        copy__default_273: "f32[768]" = torch.ops.aten.copy_.default(arg273_1, getitem_271);  arg273_1 = getitem_271 = None
        copy__default_274: "f32[768]" = torch.ops.aten.copy_.default(arg274_1, getitem_272);  arg274_1 = getitem_272 = None
        copy__default_275: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg275_1, getitem_273);  arg275_1 = getitem_273 = None
        copy__default_276: "f32[3072]" = torch.ops.aten.copy_.default(arg276_1, getitem_274);  arg276_1 = getitem_274 = None
        copy__default_277: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg277_1, getitem_275);  arg277_1 = getitem_275 = None
        copy__default_278: "f32[768]" = torch.ops.aten.copy_.default(arg278_1, getitem_276);  arg278_1 = getitem_276 = None
        copy__default_279: "f32[768]" = torch.ops.aten.copy_.default(arg279_1, getitem_277);  arg279_1 = getitem_277 = None
        copy__default_280: "f32[768]" = torch.ops.aten.copy_.default(arg280_1, getitem_278);  arg280_1 = getitem_278 = None
        copy__default_281: "f32[768]" = torch.ops.aten.copy_.default(arg281_1, getitem_279);  arg281_1 = getitem_279 = None
        copy__default_282: "f32[768]" = torch.ops.aten.copy_.default(arg282_1, getitem_280);  arg282_1 = getitem_280 = None
        copy__default_283: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg283_1, getitem_281);  arg283_1 = getitem_281 = None
        copy__default_284: "f32[3072]" = torch.ops.aten.copy_.default(arg284_1, getitem_282);  arg284_1 = getitem_282 = None
        copy__default_285: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg285_1, getitem_283);  arg285_1 = getitem_283 = None
        copy__default_286: "f32[768]" = torch.ops.aten.copy_.default(arg286_1, getitem_284);  arg286_1 = getitem_284 = None
        copy__default_287: "f32[768]" = torch.ops.aten.copy_.default(arg287_1, getitem_285);  arg287_1 = getitem_285 = None
        copy__default_288: "f32[768]" = torch.ops.aten.copy_.default(arg288_1, getitem_286);  arg288_1 = getitem_286 = None
        copy__default_289: "f32[768]" = torch.ops.aten.copy_.default(arg289_1, getitem_287);  arg289_1 = getitem_287 = None
        copy__default_290: "f32[768]" = torch.ops.aten.copy_.default(arg290_1, getitem_288);  arg290_1 = getitem_288 = None
        copy__default_291: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg291_1, getitem_289);  arg291_1 = getitem_289 = None
        copy__default_292: "f32[3072]" = torch.ops.aten.copy_.default(arg292_1, getitem_290);  arg292_1 = getitem_290 = None
        copy__default_293: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg293_1, getitem_291);  arg293_1 = getitem_291 = None
        copy__default_294: "f32[768]" = torch.ops.aten.copy_.default(arg294_1, getitem_292);  arg294_1 = getitem_292 = None
        copy__default_295: "f32[768]" = torch.ops.aten.copy_.default(arg295_1, getitem_293);  arg295_1 = getitem_293 = None
        copy__default_296: "f32[768]" = torch.ops.aten.copy_.default(arg296_1, getitem_294);  arg296_1 = getitem_294 = None
        copy__default_297: "f32[768]" = torch.ops.aten.copy_.default(arg297_1, getitem_295);  arg297_1 = getitem_295 = None
        copy__default_298: "f32[768]" = torch.ops.aten.copy_.default(arg298_1, getitem_296);  arg298_1 = getitem_296 = None
        copy__default_299: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg299_1, getitem_297);  arg299_1 = getitem_297 = None
        copy__default_300: "f32[3072]" = torch.ops.aten.copy_.default(arg300_1, getitem_298);  arg300_1 = getitem_298 = None
        copy__default_301: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg301_1, getitem_299);  arg301_1 = getitem_299 = None
        copy__default_302: "f32[768]" = torch.ops.aten.copy_.default(arg302_1, getitem_300);  arg302_1 = getitem_300 = None
        copy__default_303: "f32[768]" = torch.ops.aten.copy_.default(arg303_1, getitem_301);  arg303_1 = getitem_301 = None
        copy__default_304: "f32[768]" = torch.ops.aten.copy_.default(arg304_1, getitem_302);  arg304_1 = getitem_302 = None
        copy__default_305: "f32[768]" = torch.ops.aten.copy_.default(arg305_1, getitem_303);  arg305_1 = getitem_303 = None
        copy__default_306: "f32[768]" = torch.ops.aten.copy_.default(arg306_1, getitem_304);  arg306_1 = getitem_304 = None
        copy__default_307: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg307_1, getitem_305);  arg307_1 = getitem_305 = None
        copy__default_308: "f32[3072]" = torch.ops.aten.copy_.default(arg308_1, getitem_306);  arg308_1 = getitem_306 = None
        copy__default_309: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg309_1, getitem_307);  arg309_1 = getitem_307 = None
        copy__default_310: "f32[768]" = torch.ops.aten.copy_.default(arg310_1, getitem_308);  arg310_1 = getitem_308 = None
        copy__default_311: "f32[768]" = torch.ops.aten.copy_.default(arg311_1, getitem_309);  arg311_1 = getitem_309 = None
        copy__default_312: "f32[768]" = torch.ops.aten.copy_.default(arg312_1, getitem_310);  arg312_1 = getitem_310 = None
        copy__default_313: "f32[768]" = torch.ops.aten.copy_.default(arg313_1, getitem_311);  arg313_1 = getitem_311 = None
        copy__default_314: "f32[768]" = torch.ops.aten.copy_.default(arg314_1, getitem_312);  arg314_1 = getitem_312 = None
        copy__default_315: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg315_1, getitem_313);  arg315_1 = getitem_313 = None
        copy__default_316: "f32[3072]" = torch.ops.aten.copy_.default(arg316_1, getitem_314);  arg316_1 = getitem_314 = None
        copy__default_317: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg317_1, getitem_315);  arg317_1 = getitem_315 = None
        copy__default_318: "f32[768]" = torch.ops.aten.copy_.default(arg318_1, getitem_316);  arg318_1 = getitem_316 = None
        copy__default_319: "f32[768]" = torch.ops.aten.copy_.default(arg319_1, getitem_317);  arg319_1 = getitem_317 = None
        copy__default_320: "f32[768]" = torch.ops.aten.copy_.default(arg320_1, getitem_318);  arg320_1 = getitem_318 = None
        copy__default_321: "f32[32000]" = torch.ops.aten.copy_.default(arg321_1, getitem_319);  arg321_1 = getitem_319 = None
        copy__default_322: "f32[768, 768]" = torch.ops.aten.copy_.default(arg322_1, getitem_320);  arg322_1 = getitem_320 = None
        copy__default_323: "f32[768]" = torch.ops.aten.copy_.default(arg323_1, getitem_321);  arg323_1 = getitem_321 = None
        copy__default_324: "f32[768]" = torch.ops.aten.copy_.default(arg324_1, getitem_322);  arg324_1 = getitem_322 = None
        copy__default_325: "f32[768]" = torch.ops.aten.copy_.default(arg325_1, getitem_323);  arg325_1 = getitem_323 = None
        copy__default_326: "f32[4, 768]" = torch.ops.aten.copy_.default(arg326_1, getitem_434);  arg326_1 = getitem_434 = None
        copy__default_327: "f32[768]" = torch.ops.aten.copy_.default(arg327_1, getitem_435);  arg327_1 = getitem_435 = None
        copy__default_328: "f32[768]" = torch.ops.aten.copy_.default(arg328_1, getitem_436);  arg328_1 = getitem_436 = None
        copy__default_329: "f32[768, 768]" = torch.ops.aten.copy_.default(arg329_1, getitem_437);  arg329_1 = getitem_437 = None
        copy__default_330: "f32[768]" = torch.ops.aten.copy_.default(arg330_1, getitem_438);  arg330_1 = getitem_438 = None
        copy__default_331: "f32[768]" = torch.ops.aten.copy_.default(arg331_1, getitem_439);  arg331_1 = getitem_439 = None
        copy__default_332: "f32[768]" = torch.ops.aten.copy_.default(arg332_1, getitem_440);  arg332_1 = getitem_440 = None
        copy__default_333: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg333_1, getitem_441);  arg333_1 = getitem_441 = None
        copy__default_334: "f32[3072]" = torch.ops.aten.copy_.default(arg334_1, getitem_442);  arg334_1 = getitem_442 = None
        copy__default_335: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg335_1, getitem_443);  arg335_1 = getitem_443 = None
        copy__default_336: "f32[768]" = torch.ops.aten.copy_.default(arg336_1, getitem_444);  arg336_1 = getitem_444 = None
        copy__default_337: "f32[768]" = torch.ops.aten.copy_.default(arg337_1, getitem_445);  arg337_1 = getitem_445 = None
        copy__default_338: "f32[768]" = torch.ops.aten.copy_.default(arg338_1, getitem_446);  arg338_1 = getitem_446 = None
        copy__default_339: "f32[768]" = torch.ops.aten.copy_.default(arg339_1, getitem_447);  arg339_1 = getitem_447 = None
        copy__default_340: "f32[768]" = torch.ops.aten.copy_.default(arg340_1, getitem_448);  arg340_1 = getitem_448 = None
        copy__default_341: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg341_1, getitem_449);  arg341_1 = getitem_449 = None
        copy__default_342: "f32[3072]" = torch.ops.aten.copy_.default(arg342_1, getitem_450);  arg342_1 = getitem_450 = None
        copy__default_343: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg343_1, getitem_451);  arg343_1 = getitem_451 = None
        copy__default_344: "f32[768]" = torch.ops.aten.copy_.default(arg344_1, getitem_452);  arg344_1 = getitem_452 = None
        copy__default_345: "f32[768]" = torch.ops.aten.copy_.default(arg345_1, getitem_453);  arg345_1 = getitem_453 = None
        copy__default_346: "f32[768]" = torch.ops.aten.copy_.default(arg346_1, getitem_454);  arg346_1 = getitem_454 = None
        copy__default_347: "f32[768]" = torch.ops.aten.copy_.default(arg347_1, getitem_455);  arg347_1 = getitem_455 = None
        copy__default_348: "f32[768]" = torch.ops.aten.copy_.default(arg348_1, getitem_456);  arg348_1 = getitem_456 = None
        copy__default_349: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg349_1, getitem_457);  arg349_1 = getitem_457 = None
        copy__default_350: "f32[3072]" = torch.ops.aten.copy_.default(arg350_1, getitem_458);  arg350_1 = getitem_458 = None
        copy__default_351: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg351_1, getitem_459);  arg351_1 = getitem_459 = None
        copy__default_352: "f32[768]" = torch.ops.aten.copy_.default(arg352_1, getitem_460);  arg352_1 = getitem_460 = None
        copy__default_353: "f32[768]" = torch.ops.aten.copy_.default(arg353_1, getitem_461);  arg353_1 = getitem_461 = None
        copy__default_354: "f32[768]" = torch.ops.aten.copy_.default(arg354_1, getitem_462);  arg354_1 = getitem_462 = None
        copy__default_355: "f32[768]" = torch.ops.aten.copy_.default(arg355_1, getitem_463);  arg355_1 = getitem_463 = None
        copy__default_356: "f32[768]" = torch.ops.aten.copy_.default(arg356_1, getitem_464);  arg356_1 = getitem_464 = None
        copy__default_357: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg357_1, getitem_465);  arg357_1 = getitem_465 = None
        copy__default_358: "f32[3072]" = torch.ops.aten.copy_.default(arg358_1, getitem_466);  arg358_1 = getitem_466 = None
        copy__default_359: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg359_1, getitem_467);  arg359_1 = getitem_467 = None
        copy__default_360: "f32[768]" = torch.ops.aten.copy_.default(arg360_1, getitem_468);  arg360_1 = getitem_468 = None
        copy__default_361: "f32[768]" = torch.ops.aten.copy_.default(arg361_1, getitem_469);  arg361_1 = getitem_469 = None
        copy__default_362: "f32[768]" = torch.ops.aten.copy_.default(arg362_1, getitem_470);  arg362_1 = getitem_470 = None
        copy__default_363: "f32[768]" = torch.ops.aten.copy_.default(arg363_1, getitem_471);  arg363_1 = getitem_471 = None
        copy__default_364: "f32[768]" = torch.ops.aten.copy_.default(arg364_1, getitem_472);  arg364_1 = getitem_472 = None
        copy__default_365: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg365_1, getitem_473);  arg365_1 = getitem_473 = None
        copy__default_366: "f32[3072]" = torch.ops.aten.copy_.default(arg366_1, getitem_474);  arg366_1 = getitem_474 = None
        copy__default_367: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg367_1, getitem_475);  arg367_1 = getitem_475 = None
        copy__default_368: "f32[768]" = torch.ops.aten.copy_.default(arg368_1, getitem_476);  arg368_1 = getitem_476 = None
        copy__default_369: "f32[768]" = torch.ops.aten.copy_.default(arg369_1, getitem_477);  arg369_1 = getitem_477 = None
        copy__default_370: "f32[768]" = torch.ops.aten.copy_.default(arg370_1, getitem_478);  arg370_1 = getitem_478 = None
        copy__default_371: "f32[768]" = torch.ops.aten.copy_.default(arg371_1, getitem_479);  arg371_1 = getitem_479 = None
        copy__default_372: "f32[768]" = torch.ops.aten.copy_.default(arg372_1, getitem_480);  arg372_1 = getitem_480 = None
        copy__default_373: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg373_1, getitem_481);  arg373_1 = getitem_481 = None
        copy__default_374: "f32[3072]" = torch.ops.aten.copy_.default(arg374_1, getitem_482);  arg374_1 = getitem_482 = None
        copy__default_375: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg375_1, getitem_483);  arg375_1 = getitem_483 = None
        copy__default_376: "f32[768]" = torch.ops.aten.copy_.default(arg376_1, getitem_484);  arg376_1 = getitem_484 = None
        copy__default_377: "f32[768]" = torch.ops.aten.copy_.default(arg377_1, getitem_485);  arg377_1 = getitem_485 = None
        copy__default_378: "f32[768]" = torch.ops.aten.copy_.default(arg378_1, getitem_486);  arg378_1 = getitem_486 = None
        copy__default_379: "f32[768]" = torch.ops.aten.copy_.default(arg379_1, getitem_487);  arg379_1 = getitem_487 = None
        copy__default_380: "f32[768]" = torch.ops.aten.copy_.default(arg380_1, getitem_488);  arg380_1 = getitem_488 = None
        copy__default_381: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg381_1, getitem_489);  arg381_1 = getitem_489 = None
        copy__default_382: "f32[3072]" = torch.ops.aten.copy_.default(arg382_1, getitem_490);  arg382_1 = getitem_490 = None
        copy__default_383: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg383_1, getitem_491);  arg383_1 = getitem_491 = None
        copy__default_384: "f32[768]" = torch.ops.aten.copy_.default(arg384_1, getitem_492);  arg384_1 = getitem_492 = None
        copy__default_385: "f32[768]" = torch.ops.aten.copy_.default(arg385_1, getitem_493);  arg385_1 = getitem_493 = None
        copy__default_386: "f32[768]" = torch.ops.aten.copy_.default(arg386_1, getitem_494);  arg386_1 = getitem_494 = None
        copy__default_387: "f32[768]" = torch.ops.aten.copy_.default(arg387_1, getitem_495);  arg387_1 = getitem_495 = None
        copy__default_388: "f32[768]" = torch.ops.aten.copy_.default(arg388_1, getitem_496);  arg388_1 = getitem_496 = None
        copy__default_389: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg389_1, getitem_497);  arg389_1 = getitem_497 = None
        copy__default_390: "f32[3072]" = torch.ops.aten.copy_.default(arg390_1, getitem_498);  arg390_1 = getitem_498 = None
        copy__default_391: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg391_1, getitem_499);  arg391_1 = getitem_499 = None
        copy__default_392: "f32[768]" = torch.ops.aten.copy_.default(arg392_1, getitem_500);  arg392_1 = getitem_500 = None
        copy__default_393: "f32[768]" = torch.ops.aten.copy_.default(arg393_1, getitem_501);  arg393_1 = getitem_501 = None
        copy__default_394: "f32[768]" = torch.ops.aten.copy_.default(arg394_1, getitem_502);  arg394_1 = getitem_502 = None
        copy__default_395: "f32[768]" = torch.ops.aten.copy_.default(arg395_1, getitem_503);  arg395_1 = getitem_503 = None
        copy__default_396: "f32[768]" = torch.ops.aten.copy_.default(arg396_1, getitem_504);  arg396_1 = getitem_504 = None
        copy__default_397: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg397_1, getitem_505);  arg397_1 = getitem_505 = None
        copy__default_398: "f32[3072]" = torch.ops.aten.copy_.default(arg398_1, getitem_506);  arg398_1 = getitem_506 = None
        copy__default_399: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg399_1, getitem_507);  arg399_1 = getitem_507 = None
        copy__default_400: "f32[768]" = torch.ops.aten.copy_.default(arg400_1, getitem_508);  arg400_1 = getitem_508 = None
        copy__default_401: "f32[768]" = torch.ops.aten.copy_.default(arg401_1, getitem_509);  arg401_1 = getitem_509 = None
        copy__default_402: "f32[768]" = torch.ops.aten.copy_.default(arg402_1, getitem_510);  arg402_1 = getitem_510 = None
        copy__default_403: "f32[768]" = torch.ops.aten.copy_.default(arg403_1, getitem_511);  arg403_1 = getitem_511 = None
        copy__default_404: "f32[768]" = torch.ops.aten.copy_.default(arg404_1, getitem_512);  arg404_1 = getitem_512 = None
        copy__default_405: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg405_1, getitem_513);  arg405_1 = getitem_513 = None
        copy__default_406: "f32[3072]" = torch.ops.aten.copy_.default(arg406_1, getitem_514);  arg406_1 = getitem_514 = None
        copy__default_407: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg407_1, getitem_515);  arg407_1 = getitem_515 = None
        copy__default_408: "f32[768]" = torch.ops.aten.copy_.default(arg408_1, getitem_516);  arg408_1 = getitem_516 = None
        copy__default_409: "f32[768]" = torch.ops.aten.copy_.default(arg409_1, getitem_517);  arg409_1 = getitem_517 = None
        copy__default_410: "f32[768]" = torch.ops.aten.copy_.default(arg410_1, getitem_518);  arg410_1 = getitem_518 = None
        copy__default_411: "f32[768]" = torch.ops.aten.copy_.default(arg411_1, getitem_519);  arg411_1 = getitem_519 = None
        copy__default_412: "f32[768]" = torch.ops.aten.copy_.default(arg412_1, getitem_520);  arg412_1 = getitem_520 = None
        copy__default_413: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg413_1, getitem_521);  arg413_1 = getitem_521 = None
        copy__default_414: "f32[3072]" = torch.ops.aten.copy_.default(arg414_1, getitem_522);  arg414_1 = getitem_522 = None
        copy__default_415: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg415_1, getitem_523);  arg415_1 = getitem_523 = None
        copy__default_416: "f32[768]" = torch.ops.aten.copy_.default(arg416_1, getitem_524);  arg416_1 = getitem_524 = None
        copy__default_417: "f32[768]" = torch.ops.aten.copy_.default(arg417_1, getitem_525);  arg417_1 = getitem_525 = None
        copy__default_418: "f32[768]" = torch.ops.aten.copy_.default(arg418_1, getitem_526);  arg418_1 = getitem_526 = None
        copy__default_419: "f32[768]" = torch.ops.aten.copy_.default(arg419_1, getitem_527);  arg419_1 = getitem_527 = None
        copy__default_420: "f32[768]" = torch.ops.aten.copy_.default(arg420_1, getitem_528);  arg420_1 = getitem_528 = None
        copy__default_421: "f32[3072, 768]" = torch.ops.aten.copy_.default(arg421_1, getitem_529);  arg421_1 = getitem_529 = None
        copy__default_422: "f32[3072]" = torch.ops.aten.copy_.default(arg422_1, getitem_530);  arg422_1 = getitem_530 = None
        copy__default_423: "f32[768, 3072]" = torch.ops.aten.copy_.default(arg423_1, getitem_531);  arg423_1 = getitem_531 = None
        copy__default_424: "f32[768]" = torch.ops.aten.copy_.default(arg424_1, getitem_532);  arg424_1 = getitem_532 = None
        copy__default_425: "f32[768]" = torch.ops.aten.copy_.default(arg425_1, getitem_533);  arg425_1 = getitem_533 = None
        copy__default_426: "f32[768]" = torch.ops.aten.copy_.default(arg426_1, getitem_534);  arg426_1 = getitem_534 = None
        copy__default_427: "f32[32000]" = torch.ops.aten.copy_.default(arg427_1, getitem_535);  arg427_1 = getitem_535 = None
        copy__default_428: "f32[768, 768]" = torch.ops.aten.copy_.default(arg428_1, getitem_536);  arg428_1 = getitem_536 = None
        copy__default_429: "f32[768]" = torch.ops.aten.copy_.default(arg429_1, getitem_537);  arg429_1 = getitem_537 = None
        copy__default_430: "f32[768]" = torch.ops.aten.copy_.default(arg430_1, getitem_538);  arg430_1 = getitem_538 = None
        copy__default_431: "f32[768]" = torch.ops.aten.copy_.default(arg431_1, getitem_539);  arg431_1 = getitem_539 = None
        return (copy__default, copy__default_1, copy__default_2, copy__default_3, copy__default_4, copy__default_5, copy__default_6, copy__default_7, copy__default_8, copy__default_9, copy__default_10, copy__default_11, copy__default_12, copy__default_13, copy__default_14, copy__default_15, copy__default_16, copy__default_17, copy__default_18, copy__default_19, copy__default_20, copy__default_21, copy__default_22, copy__default_23, copy__default_24, copy__default_25, copy__default_26, copy__default_27, copy__default_28, copy__default_29, copy__default_30, copy__default_31, copy__default_32, copy__default_33, copy__default_34, copy__default_35, copy__default_36, copy__default_37, copy__default_38, copy__default_39, copy__default_40, copy__default_41, copy__default_42, copy__default_43, copy__default_44, copy__default_45, copy__default_46, copy__default_47, copy__default_48, copy__default_49, copy__default_50, copy__default_51, copy__default_52, copy__default_53, copy__default_54, copy__default_55, copy__default_56, copy__default_57, copy__default_58, copy__default_59, copy__default_60, copy__default_61, copy__default_62, copy__default_63, copy__default_64, copy__default_65, copy__default_66, copy__default_67, copy__default_68, copy__default_69, copy__default_70, copy__default_71, copy__default_72, copy__default_73, copy__default_74, copy__default_75, copy__default_76, copy__default_77, copy__default_78, copy__default_79, copy__default_80, copy__default_81, copy__default_82, copy__default_83, copy__default_84, copy__default_85, copy__default_86, copy__default_87, copy__default_88, copy__default_89, copy__default_90, copy__default_91, copy__default_92, copy__default_93, copy__default_94, copy__default_95, copy__default_96, copy__default_97, copy__default_98, copy__default_99, copy__default_100, copy__default_101, copy__default_102, copy__default_103, copy__default_104, copy__default_105, copy__default_106, copy__default_107, copy__default_108, copy__default_109, copy__default_110, copy__default_111, copy__default_112, copy__default_113, copy__default_114, copy__default_115, copy__default_116, copy__default_117, copy__default_118, copy__default_119, copy__default_120, copy__default_121, copy__default_122, copy__default_123, copy__default_124, copy__default_125, copy__default_126, copy__default_127, copy__default_128, copy__default_129, copy__default_130, copy__default_131, copy__default_132, copy__default_133, copy__default_134, copy__default_135, copy__default_136, copy__default_137, copy__default_138, copy__default_139, copy__default_140, copy__default_141, copy__default_142, copy__default_143, copy__default_144, copy__default_145, copy__default_146, copy__default_147, copy__default_148, copy__default_149, copy__default_150, copy__default_151, copy__default_152, copy__default_153, copy__default_154, copy__default_155, copy__default_156, copy__default_157, copy__default_158, copy__default_159, copy__default_160, copy__default_161, copy__default_162, copy__default_163, copy__default_164, copy__default_165, copy__default_166, copy__default_167, copy__default_168, copy__default_169, copy__default_170, copy__default_171, copy__default_172, copy__default_173, copy__default_174, copy__default_175, copy__default_176, copy__default_177, copy__default_178, copy__default_179, copy__default_180, copy__default_181, copy__default_182, copy__default_183, copy__default_184, copy__default_185, copy__default_186, copy__default_187, copy__default_188, copy__default_189, copy__default_190, copy__default_191, copy__default_192, copy__default_193, copy__default_194, copy__default_195, copy__default_196, copy__default_197, copy__default_198, copy__default_199, copy__default_200, copy__default_201, copy__default_202, copy__default_203, copy__default_204, copy__default_205, copy__default_206, copy__default_207, copy__default_208, copy__default_209, copy__default_210, copy__default_211, copy__default_212, copy__default_213, copy__default_214, copy__default_215, copy__default_216, copy__default_217, copy__default_218, copy__default_219, copy__default_220, copy__default_221, copy__default_222, copy__default_223, copy__default_224, copy__default_225, copy__default_226, copy__default_227, copy__default_228, copy__default_229, copy__default_230, copy__default_231, copy__default_232, copy__default_233, copy__default_234, copy__default_235, copy__default_236, copy__default_237, copy__default_238, copy__default_239, copy__default_240, copy__default_241, copy__default_242, copy__default_243, copy__default_244, copy__default_245, copy__default_246, copy__default_247, copy__default_248, copy__default_249, copy__default_250, copy__default_251, copy__default_252, copy__default_253, copy__default_254, copy__default_255, copy__default_256, copy__default_257, copy__default_258, copy__default_259, copy__default_260, copy__default_261, copy__default_262, copy__default_263, copy__default_264, copy__default_265, copy__default_266, copy__default_267, copy__default_268, copy__default_269, copy__default_270, copy__default_271, copy__default_272, copy__default_273, copy__default_274, copy__default_275, copy__default_276, copy__default_277, copy__default_278, copy__default_279, copy__default_280, copy__default_281, copy__default_282, copy__default_283, copy__default_284, copy__default_285, copy__default_286, copy__default_287, copy__default_288, copy__default_289, copy__default_290, copy__default_291, copy__default_292, copy__default_293, copy__default_294, copy__default_295, copy__default_296, copy__default_297, copy__default_298, copy__default_299, copy__default_300, copy__default_301, copy__default_302, copy__default_303, copy__default_304, copy__default_305, copy__default_306, copy__default_307, copy__default_308, copy__default_309, copy__default_310, copy__default_311, copy__default_312, copy__default_313, copy__default_314, copy__default_315, copy__default_316, copy__default_317, copy__default_318, copy__default_319, copy__default_320, copy__default_321, copy__default_322, copy__default_323, copy__default_324, copy__default_325, copy__default_326, copy__default_327, copy__default_328, copy__default_329, copy__default_330, copy__default_331, copy__default_332, copy__default_333, copy__default_334, copy__default_335, copy__default_336, copy__default_337, copy__default_338, copy__default_339, copy__default_340, copy__default_341, copy__default_342, copy__default_343, copy__default_344, copy__default_345, copy__default_346, copy__default_347, copy__default_348, copy__default_349, copy__default_350, copy__default_351, copy__default_352, copy__default_353, copy__default_354, copy__default_355, copy__default_356, copy__default_357, copy__default_358, copy__default_359, copy__default_360, copy__default_361, copy__default_362, copy__default_363, copy__default_364, copy__default_365, copy__default_366, copy__default_367, copy__default_368, copy__default_369, copy__default_370, copy__default_371, copy__default_372, copy__default_373, copy__default_374, copy__default_375, copy__default_376, copy__default_377, copy__default_378, copy__default_379, copy__default_380, copy__default_381, copy__default_382, copy__default_383, copy__default_384, copy__default_385, copy__default_386, copy__default_387, copy__default_388, copy__default_389, copy__default_390, copy__default_391, copy__default_392, copy__default_393, copy__default_394, copy__default_395, copy__default_396, copy__default_397, copy__default_398, copy__default_399, copy__default_400, copy__default_401, copy__default_402, copy__default_403, copy__default_404, copy__default_405, copy__default_406, copy__default_407, copy__default_408, copy__default_409, copy__default_410, copy__default_411, copy__default_412, copy__default_413, copy__default_414, copy__default_415, copy__default_416, copy__default_417, copy__default_418, copy__default_419, copy__default_420, copy__default_421, copy__default_422, copy__default_423, copy__default_424, copy__default_425, copy__default_426, copy__default_427, copy__default_428, copy__default_429, copy__default_430, copy__default_431)


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
    torch.randn([32000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000, 768], dtype=torch.float32, device='cuda'),
    torch.randn([512, 768], dtype=torch.float32, device='cuda'),
    torch.randn([4, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
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
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([3072, 768], dtype=torch.float32, device='cuda'),
    torch.randn([3072], dtype=torch.float32, device='cuda'),
    torch.randn([768, 3072], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([32000], dtype=torch.float32, device='cuda'),
    torch.randn([768, 768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    torch.randn([768], dtype=torch.float32, device='cuda'),
    ]


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

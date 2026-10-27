"""
Standalone repro captured via capture_hook.
Label: MobileBertForMaskedLM
Pattern hash: 3c57b3aa24a2
Shape hash: fdf2f4a6
"""
import sys
from pathlib import Path

import torch
import torch._inductor.inductor_prims  # noqa: F401
from math import inf, nan
from torch import device

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from repro_harness import benchmark_repro, make_inputs_from_config, load_shape_configs

_repro_version = 3
# Input shapes/strides/dtypes live in the sibling shapes.json (structured,
# one entry per point); forward()'s annotations document the default shapes
# inline. Default inputs = the first shapes.json point.

class Repro(torch.nn.Module):
    def forward(self, arg0_1: "i64[1, 512]", arg1_1: "f32[30522, 128]", arg2_1: "i64[256, 128]", arg3_1: "f32[512]", arg4_1: "f32[512, 384]", arg5_1: "f32[512, 512]", arg6_1: "f32[2, 512]", arg7_1: "f32[512]", arg8_1: "f32[512]", arg9_1: "f32[128]", arg10_1: "f32[128, 512]", arg11_1: "f32[128]", arg12_1: "f32[128]", arg13_1: "f32[128]", arg14_1: "f32[128, 512]", arg15_1: "f32[128]", arg16_1: "f32[128]", arg17_1: "f32[128]", arg18_1: "f32[128, 128]", arg19_1: "f32[128]", arg20_1: "f32[128, 128]", arg21_1: "f32[128]", arg22_1: "f32[128, 512]", arg23_1: "f32[128]", arg24_1: "f32[128, 128]", arg25_1: "f32[128]", arg26_1: "f32[128]", arg27_1: "f32[512]", arg28_1: "f32[512, 128]", arg29_1: "f32[128]", arg30_1: "f32[128, 512]", arg31_1: "f32[128]", arg32_1: "f32[128]", arg33_1: "f32[512]", arg34_1: "f32[512, 128]", arg35_1: "f32[128]", arg36_1: "f32[128, 512]", arg37_1: "f32[128]", arg38_1: "f32[128]", arg39_1: "f32[512]", arg40_1: "f32[512, 128]", arg41_1: "f32[128]", arg42_1: "f32[128, 512]", arg43_1: "f32[128]", arg44_1: "f32[128]", arg45_1: "f32[512]", arg46_1: "f32[512, 128]", arg47_1: "f32[128]", arg48_1: "f32[128, 512]", arg49_1: "f32[128]", arg50_1: "f32[128]", arg51_1: "f32[512]", arg52_1: "f32[512, 128]", arg53_1: "f32[512]", arg54_1: "f32[512]", arg55_1: "f32[128]", arg56_1: "f32[128, 512]", arg57_1: "f32[128]", arg58_1: "f32[128]", arg59_1: "f32[128]", arg60_1: "f32[128, 512]", arg61_1: "f32[128]", arg62_1: "f32[128]", arg63_1: "f32[128]", arg64_1: "f32[128, 128]", arg65_1: "f32[128]", arg66_1: "f32[128, 128]", arg67_1: "f32[128]", arg68_1: "f32[128, 512]", arg69_1: "f32[128]", arg70_1: "f32[128, 128]", arg71_1: "f32[128]", arg72_1: "f32[128]", arg73_1: "f32[512]", arg74_1: "f32[512, 128]", arg75_1: "f32[128]", arg76_1: "f32[128, 512]", arg77_1: "f32[128]", arg78_1: "f32[128]", arg79_1: "f32[512]", arg80_1: "f32[512, 128]", arg81_1: "f32[128]", arg82_1: "f32[128, 512]", arg83_1: "f32[128]", arg84_1: "f32[128]", arg85_1: "f32[512]", arg86_1: "f32[512, 128]", arg87_1: "f32[128]", arg88_1: "f32[128, 512]", arg89_1: "f32[128]", arg90_1: "f32[128]", arg91_1: "f32[512]", arg92_1: "f32[512, 128]", arg93_1: "f32[128]", arg94_1: "f32[128, 512]", arg95_1: "f32[128]", arg96_1: "f32[128]", arg97_1: "f32[512]", arg98_1: "f32[512, 128]", arg99_1: "f32[512]", arg100_1: "f32[512]", arg101_1: "f32[128]", arg102_1: "f32[128, 512]", arg103_1: "f32[128]", arg104_1: "f32[128]", arg105_1: "f32[128]", arg106_1: "f32[128, 512]", arg107_1: "f32[128]", arg108_1: "f32[128]", arg109_1: "f32[128]", arg110_1: "f32[128, 128]", arg111_1: "f32[128]", arg112_1: "f32[128, 128]", arg113_1: "f32[128]", arg114_1: "f32[128, 512]", arg115_1: "f32[128]", arg116_1: "f32[128, 128]", arg117_1: "f32[128]", arg118_1: "f32[128]", arg119_1: "f32[512]", arg120_1: "f32[512, 128]", arg121_1: "f32[128]", arg122_1: "f32[128, 512]", arg123_1: "f32[128]", arg124_1: "f32[128]", arg125_1: "f32[512]", arg126_1: "f32[512, 128]", arg127_1: "f32[128]", arg128_1: "f32[128, 512]", arg129_1: "f32[128]", arg130_1: "f32[128]", arg131_1: "f32[512]", arg132_1: "f32[512, 128]", arg133_1: "f32[128]", arg134_1: "f32[128, 512]", arg135_1: "f32[128]", arg136_1: "f32[128]", arg137_1: "f32[512]", arg138_1: "f32[512, 128]", arg139_1: "f32[128]", arg140_1: "f32[128, 512]", arg141_1: "f32[128]", arg142_1: "f32[128]", arg143_1: "f32[512]", arg144_1: "f32[512, 128]", arg145_1: "f32[512]", arg146_1: "f32[512]", arg147_1: "f32[128]", arg148_1: "f32[128, 512]", arg149_1: "f32[128]", arg150_1: "f32[128]", arg151_1: "f32[128]", arg152_1: "f32[128, 512]", arg153_1: "f32[128]", arg154_1: "f32[128]", arg155_1: "f32[128]", arg156_1: "f32[128, 128]", arg157_1: "f32[128]", arg158_1: "f32[128, 128]", arg159_1: "f32[128]", arg160_1: "f32[128, 512]", arg161_1: "f32[128]", arg162_1: "f32[128, 128]", arg163_1: "f32[128]", arg164_1: "f32[128]", arg165_1: "f32[512]", arg166_1: "f32[512, 128]", arg167_1: "f32[128]", arg168_1: "f32[128, 512]", arg169_1: "f32[128]", arg170_1: "f32[128]", arg171_1: "f32[512]", arg172_1: "f32[512, 128]", arg173_1: "f32[128]", arg174_1: "f32[128, 512]", arg175_1: "f32[128]", arg176_1: "f32[128]", arg177_1: "f32[512]", arg178_1: "f32[512, 128]", arg179_1: "f32[128]", arg180_1: "f32[128, 512]", arg181_1: "f32[128]", arg182_1: "f32[128]", arg183_1: "f32[512]", arg184_1: "f32[512, 128]", arg185_1: "f32[128]", arg186_1: "f32[128, 512]", arg187_1: "f32[128]", arg188_1: "f32[128]", arg189_1: "f32[512]", arg190_1: "f32[512, 128]", arg191_1: "f32[512]", arg192_1: "f32[512]", arg193_1: "f32[128]", arg194_1: "f32[128, 512]", arg195_1: "f32[128]", arg196_1: "f32[128]", arg197_1: "f32[128]", arg198_1: "f32[128, 512]", arg199_1: "f32[128]", arg200_1: "f32[128]", arg201_1: "f32[128]", arg202_1: "f32[128, 128]", arg203_1: "f32[128]", arg204_1: "f32[128, 128]", arg205_1: "f32[128]", arg206_1: "f32[128, 512]", arg207_1: "f32[128]", arg208_1: "f32[128, 128]", arg209_1: "f32[128]", arg210_1: "f32[128]", arg211_1: "f32[512]", arg212_1: "f32[512, 128]", arg213_1: "f32[128]", arg214_1: "f32[128, 512]", arg215_1: "f32[128]", arg216_1: "f32[128]", arg217_1: "f32[512]", arg218_1: "f32[512, 128]", arg219_1: "f32[128]", arg220_1: "f32[128, 512]", arg221_1: "f32[128]", arg222_1: "f32[128]", arg223_1: "f32[512]", arg224_1: "f32[512, 128]", arg225_1: "f32[128]", arg226_1: "f32[128, 512]", arg227_1: "f32[128]", arg228_1: "f32[128]", arg229_1: "f32[512]", arg230_1: "f32[512, 128]", arg231_1: "f32[128]", arg232_1: "f32[128, 512]", arg233_1: "f32[128]", arg234_1: "f32[128]", arg235_1: "f32[512]", arg236_1: "f32[512, 128]", arg237_1: "f32[512]", arg238_1: "f32[512]", arg239_1: "f32[128]", arg240_1: "f32[128, 512]", arg241_1: "f32[128]", arg242_1: "f32[128]", arg243_1: "f32[128]", arg244_1: "f32[128, 512]", arg245_1: "f32[128]", arg246_1: "f32[128]", arg247_1: "f32[128]", arg248_1: "f32[128, 128]", arg249_1: "f32[128]", arg250_1: "f32[128, 128]", arg251_1: "f32[128]", arg252_1: "f32[128, 512]", arg253_1: "f32[128]", arg254_1: "f32[128, 128]", arg255_1: "f32[128]", arg256_1: "f32[128]", arg257_1: "f32[512]", arg258_1: "f32[512, 128]", arg259_1: "f32[128]", arg260_1: "f32[128, 512]", arg261_1: "f32[128]", arg262_1: "f32[128]", arg263_1: "f32[512]", arg264_1: "f32[512, 128]", arg265_1: "f32[128]", arg266_1: "f32[128, 512]", arg267_1: "f32[128]", arg268_1: "f32[128]", arg269_1: "f32[512]", arg270_1: "f32[512, 128]", arg271_1: "f32[128]", arg272_1: "f32[128, 512]", arg273_1: "f32[128]", arg274_1: "f32[128]", arg275_1: "f32[512]", arg276_1: "f32[512, 128]", arg277_1: "f32[128]", arg278_1: "f32[128, 512]", arg279_1: "f32[128]", arg280_1: "f32[128]", arg281_1: "f32[512]", arg282_1: "f32[512, 128]", arg283_1: "f32[512]", arg284_1: "f32[512]", arg285_1: "f32[128]", arg286_1: "f32[128, 512]", arg287_1: "f32[128]", arg288_1: "f32[128]", arg289_1: "f32[128]", arg290_1: "f32[128, 512]", arg291_1: "f32[128]", arg292_1: "f32[128]", arg293_1: "f32[128]", arg294_1: "f32[128, 128]", arg295_1: "f32[128]", arg296_1: "f32[128, 128]", arg297_1: "f32[128]", arg298_1: "f32[128, 512]", arg299_1: "f32[128]", arg300_1: "f32[128, 128]", arg301_1: "f32[128]", arg302_1: "f32[128]", arg303_1: "f32[512]", arg304_1: "f32[512, 128]", arg305_1: "f32[128]", arg306_1: "f32[128, 512]", arg307_1: "f32[128]", arg308_1: "f32[128]", arg309_1: "f32[512]", arg310_1: "f32[512, 128]", arg311_1: "f32[128]", arg312_1: "f32[128, 512]", arg313_1: "f32[128]", arg314_1: "f32[128]", arg315_1: "f32[512]", arg316_1: "f32[512, 128]", arg317_1: "f32[128]", arg318_1: "f32[128, 512]", arg319_1: "f32[128]", arg320_1: "f32[128]", arg321_1: "f32[512]", arg322_1: "f32[512, 128]", arg323_1: "f32[128]", arg324_1: "f32[128, 512]", arg325_1: "f32[128]", arg326_1: "f32[128]", arg327_1: "f32[512]", arg328_1: "f32[512, 128]", arg329_1: "f32[512]", arg330_1: "f32[512]", arg331_1: "f32[128]", arg332_1: "f32[128, 512]", arg333_1: "f32[128]", arg334_1: "f32[128]", arg335_1: "f32[128]", arg336_1: "f32[128, 512]", arg337_1: "f32[128]", arg338_1: "f32[128]", arg339_1: "f32[128]", arg340_1: "f32[128, 128]", arg341_1: "f32[128]", arg342_1: "f32[128, 128]", arg343_1: "f32[128]", arg344_1: "f32[128, 512]", arg345_1: "f32[128]", arg346_1: "f32[128, 128]", arg347_1: "f32[128]", arg348_1: "f32[128]", arg349_1: "f32[512]", arg350_1: "f32[512, 128]", arg351_1: "f32[128]", arg352_1: "f32[128, 512]", arg353_1: "f32[128]", arg354_1: "f32[128]", arg355_1: "f32[512]", arg356_1: "f32[512, 128]", arg357_1: "f32[128]", arg358_1: "f32[128, 512]", arg359_1: "f32[128]", arg360_1: "f32[128]", arg361_1: "f32[512]", arg362_1: "f32[512, 128]", arg363_1: "f32[128]", arg364_1: "f32[128, 512]", arg365_1: "f32[128]", arg366_1: "f32[128]", arg367_1: "f32[512]", arg368_1: "f32[512, 128]", arg369_1: "f32[128]", arg370_1: "f32[128, 512]", arg371_1: "f32[128]", arg372_1: "f32[128]", arg373_1: "f32[512]", arg374_1: "f32[512, 128]", arg375_1: "f32[512]", arg376_1: "f32[512]", arg377_1: "f32[128]", arg378_1: "f32[128, 512]", arg379_1: "f32[128]", arg380_1: "f32[128]", arg381_1: "f32[128]", arg382_1: "f32[128, 512]", arg383_1: "f32[128]", arg384_1: "f32[128]", arg385_1: "f32[128]", arg386_1: "f32[128, 128]", arg387_1: "f32[128]", arg388_1: "f32[128, 128]", arg389_1: "f32[128]", arg390_1: "f32[128, 512]", arg391_1: "f32[128]", arg392_1: "f32[128, 128]", arg393_1: "f32[128]", arg394_1: "f32[128]", arg395_1: "f32[512]", arg396_1: "f32[512, 128]", arg397_1: "f32[128]", arg398_1: "f32[128, 512]", arg399_1: "f32[128]", arg400_1: "f32[128]", arg401_1: "f32[512]", arg402_1: "f32[512, 128]", arg403_1: "f32[128]", arg404_1: "f32[128, 512]", arg405_1: "f32[128]", arg406_1: "f32[128]", arg407_1: "f32[512]", arg408_1: "f32[512, 128]", arg409_1: "f32[128]", arg410_1: "f32[128, 512]", arg411_1: "f32[128]", arg412_1: "f32[128]", arg413_1: "f32[512]", arg414_1: "f32[512, 128]", arg415_1: "f32[128]", arg416_1: "f32[128, 512]", arg417_1: "f32[128]", arg418_1: "f32[128]", arg419_1: "f32[512]", arg420_1: "f32[512, 128]", arg421_1: "f32[512]", arg422_1: "f32[512]", arg423_1: "f32[128]", arg424_1: "f32[128, 512]", arg425_1: "f32[128]", arg426_1: "f32[128]", arg427_1: "f32[128]", arg428_1: "f32[128, 512]", arg429_1: "f32[128]", arg430_1: "f32[128]", arg431_1: "f32[128]", arg432_1: "f32[128, 128]", arg433_1: "f32[128]", arg434_1: "f32[128, 128]", arg435_1: "f32[128]", arg436_1: "f32[128, 512]", arg437_1: "f32[128]", arg438_1: "f32[128, 128]", arg439_1: "f32[128]", arg440_1: "f32[128]", arg441_1: "f32[512]", arg442_1: "f32[512, 128]", arg443_1: "f32[128]", arg444_1: "f32[128, 512]", arg445_1: "f32[128]", arg446_1: "f32[128]", arg447_1: "f32[512]", arg448_1: "f32[512, 128]", arg449_1: "f32[128]", arg450_1: "f32[128, 512]", arg451_1: "f32[128]", arg452_1: "f32[128]", arg453_1: "f32[512]", arg454_1: "f32[512, 128]", arg455_1: "f32[128]", arg456_1: "f32[128, 512]", arg457_1: "f32[128]", arg458_1: "f32[128]", arg459_1: "f32[512]", arg460_1: "f32[512, 128]", arg461_1: "f32[128]", arg462_1: "f32[128, 512]", arg463_1: "f32[128]", arg464_1: "f32[128]", arg465_1: "f32[512]", arg466_1: "f32[512, 128]", arg467_1: "f32[512]", arg468_1: "f32[512]", arg469_1: "f32[128]", arg470_1: "f32[128, 512]", arg471_1: "f32[128]", arg472_1: "f32[128]", arg473_1: "f32[128]", arg474_1: "f32[128, 512]", arg475_1: "f32[128]", arg476_1: "f32[128]", arg477_1: "f32[128]", arg478_1: "f32[128, 128]", arg479_1: "f32[128]", arg480_1: "f32[128, 128]", arg481_1: "f32[128]", arg482_1: "f32[128, 512]", arg483_1: "f32[128]", arg484_1: "f32[128, 128]", arg485_1: "f32[128]", arg486_1: "f32[128]", arg487_1: "f32[512]", arg488_1: "f32[512, 128]", arg489_1: "f32[128]", arg490_1: "f32[128, 512]", arg491_1: "f32[128]", arg492_1: "f32[128]", arg493_1: "f32[512]", arg494_1: "f32[512, 128]", arg495_1: "f32[128]", arg496_1: "f32[128, 512]", arg497_1: "f32[128]", arg498_1: "f32[128]", arg499_1: "f32[512]", arg500_1: "f32[512, 128]", arg501_1: "f32[128]", arg502_1: "f32[128, 512]", arg503_1: "f32[128]", arg504_1: "f32[128]", arg505_1: "f32[512]", arg506_1: "f32[512, 128]", arg507_1: "f32[128]", arg508_1: "f32[128, 512]", arg509_1: "f32[128]", arg510_1: "f32[128]", arg511_1: "f32[512]", arg512_1: "f32[512, 128]", arg513_1: "f32[512]", arg514_1: "f32[512]", arg515_1: "f32[128]", arg516_1: "f32[128, 512]", arg517_1: "f32[128]", arg518_1: "f32[128]", arg519_1: "f32[128]", arg520_1: "f32[128, 512]", arg521_1: "f32[128]", arg522_1: "f32[128]", arg523_1: "f32[128]", arg524_1: "f32[128, 128]", arg525_1: "f32[128]", arg526_1: "f32[128, 128]", arg527_1: "f32[128]", arg528_1: "f32[128, 512]", arg529_1: "f32[128]", arg530_1: "f32[128, 128]", arg531_1: "f32[128]", arg532_1: "f32[128]", arg533_1: "f32[512]", arg534_1: "f32[512, 128]", arg535_1: "f32[128]", arg536_1: "f32[128, 512]", arg537_1: "f32[128]", arg538_1: "f32[128]", arg539_1: "f32[512]", arg540_1: "f32[512, 128]", arg541_1: "f32[128]", arg542_1: "f32[128, 512]", arg543_1: "f32[128]", arg544_1: "f32[128]", arg545_1: "f32[512]", arg546_1: "f32[512, 128]", arg547_1: "f32[128]", arg548_1: "f32[128, 512]", arg549_1: "f32[128]", arg550_1: "f32[128]", arg551_1: "f32[512]", arg552_1: "f32[512, 128]", arg553_1: "f32[128]", arg554_1: "f32[128, 512]", arg555_1: "f32[128]", arg556_1: "f32[128]", arg557_1: "f32[512]", arg558_1: "f32[512, 128]", arg559_1: "f32[512]", arg560_1: "f32[512]", arg561_1: "f32[128]", arg562_1: "f32[128, 512]", arg563_1: "f32[128]", arg564_1: "f32[128]", arg565_1: "f32[128]", arg566_1: "f32[128, 512]", arg567_1: "f32[128]", arg568_1: "f32[128]", arg569_1: "f32[128]", arg570_1: "f32[128, 128]", arg571_1: "f32[128]", arg572_1: "f32[128, 128]", arg573_1: "f32[128]", arg574_1: "f32[128, 512]", arg575_1: "f32[128]", arg576_1: "f32[128, 128]", arg577_1: "f32[128]", arg578_1: "f32[128]", arg579_1: "f32[512]", arg580_1: "f32[512, 128]", arg581_1: "f32[128]", arg582_1: "f32[128, 512]", arg583_1: "f32[128]", arg584_1: "f32[128]", arg585_1: "f32[512]", arg586_1: "f32[512, 128]", arg587_1: "f32[128]", arg588_1: "f32[128, 512]", arg589_1: "f32[128]", arg590_1: "f32[128]", arg591_1: "f32[512]", arg592_1: "f32[512, 128]", arg593_1: "f32[128]", arg594_1: "f32[128, 512]", arg595_1: "f32[128]", arg596_1: "f32[128]", arg597_1: "f32[512]", arg598_1: "f32[512, 128]", arg599_1: "f32[128]", arg600_1: "f32[128, 512]", arg601_1: "f32[128]", arg602_1: "f32[128]", arg603_1: "f32[512]", arg604_1: "f32[512, 128]", arg605_1: "f32[512]", arg606_1: "f32[512]", arg607_1: "f32[128]", arg608_1: "f32[128, 512]", arg609_1: "f32[128]", arg610_1: "f32[128]", arg611_1: "f32[128]", arg612_1: "f32[128, 512]", arg613_1: "f32[128]", arg614_1: "f32[128]", arg615_1: "f32[128]", arg616_1: "f32[128, 128]", arg617_1: "f32[128]", arg618_1: "f32[128, 128]", arg619_1: "f32[128]", arg620_1: "f32[128, 512]", arg621_1: "f32[128]", arg622_1: "f32[128, 128]", arg623_1: "f32[128]", arg624_1: "f32[128]", arg625_1: "f32[512]", arg626_1: "f32[512, 128]", arg627_1: "f32[128]", arg628_1: "f32[128, 512]", arg629_1: "f32[128]", arg630_1: "f32[128]", arg631_1: "f32[512]", arg632_1: "f32[512, 128]", arg633_1: "f32[128]", arg634_1: "f32[128, 512]", arg635_1: "f32[128]", arg636_1: "f32[128]", arg637_1: "f32[512]", arg638_1: "f32[512, 128]", arg639_1: "f32[128]", arg640_1: "f32[128, 512]", arg641_1: "f32[128]", arg642_1: "f32[128]", arg643_1: "f32[512]", arg644_1: "f32[512, 128]", arg645_1: "f32[128]", arg646_1: "f32[128, 512]", arg647_1: "f32[128]", arg648_1: "f32[128]", arg649_1: "f32[512]", arg650_1: "f32[512, 128]", arg651_1: "f32[512]", arg652_1: "f32[512]", arg653_1: "f32[128]", arg654_1: "f32[128, 512]", arg655_1: "f32[128]", arg656_1: "f32[128]", arg657_1: "f32[128]", arg658_1: "f32[128, 512]", arg659_1: "f32[128]", arg660_1: "f32[128]", arg661_1: "f32[128]", arg662_1: "f32[128, 128]", arg663_1: "f32[128]", arg664_1: "f32[128, 128]", arg665_1: "f32[128]", arg666_1: "f32[128, 512]", arg667_1: "f32[128]", arg668_1: "f32[128, 128]", arg669_1: "f32[128]", arg670_1: "f32[128]", arg671_1: "f32[512]", arg672_1: "f32[512, 128]", arg673_1: "f32[128]", arg674_1: "f32[128, 512]", arg675_1: "f32[128]", arg676_1: "f32[128]", arg677_1: "f32[512]", arg678_1: "f32[512, 128]", arg679_1: "f32[128]", arg680_1: "f32[128, 512]", arg681_1: "f32[128]", arg682_1: "f32[128]", arg683_1: "f32[512]", arg684_1: "f32[512, 128]", arg685_1: "f32[128]", arg686_1: "f32[128, 512]", arg687_1: "f32[128]", arg688_1: "f32[128]", arg689_1: "f32[512]", arg690_1: "f32[512, 128]", arg691_1: "f32[128]", arg692_1: "f32[128, 512]", arg693_1: "f32[128]", arg694_1: "f32[128]", arg695_1: "f32[512]", arg696_1: "f32[512, 128]", arg697_1: "f32[512]", arg698_1: "f32[512]", arg699_1: "f32[128]", arg700_1: "f32[128, 512]", arg701_1: "f32[128]", arg702_1: "f32[128]", arg703_1: "f32[128]", arg704_1: "f32[128, 512]", arg705_1: "f32[128]", arg706_1: "f32[128]", arg707_1: "f32[128]", arg708_1: "f32[128, 128]", arg709_1: "f32[128]", arg710_1: "f32[128, 128]", arg711_1: "f32[128]", arg712_1: "f32[128, 512]", arg713_1: "f32[128]", arg714_1: "f32[128, 128]", arg715_1: "f32[128]", arg716_1: "f32[128]", arg717_1: "f32[512]", arg718_1: "f32[512, 128]", arg719_1: "f32[128]", arg720_1: "f32[128, 512]", arg721_1: "f32[128]", arg722_1: "f32[128]", arg723_1: "f32[512]", arg724_1: "f32[512, 128]", arg725_1: "f32[128]", arg726_1: "f32[128, 512]", arg727_1: "f32[128]", arg728_1: "f32[128]", arg729_1: "f32[512]", arg730_1: "f32[512, 128]", arg731_1: "f32[128]", arg732_1: "f32[128, 512]", arg733_1: "f32[128]", arg734_1: "f32[128]", arg735_1: "f32[512]", arg736_1: "f32[512, 128]", arg737_1: "f32[128]", arg738_1: "f32[128, 512]", arg739_1: "f32[128]", arg740_1: "f32[128]", arg741_1: "f32[512]", arg742_1: "f32[512, 128]", arg743_1: "f32[512]", arg744_1: "f32[512]", arg745_1: "f32[128]", arg746_1: "f32[128, 512]", arg747_1: "f32[128]", arg748_1: "f32[128]", arg749_1: "f32[128]", arg750_1: "f32[128, 512]", arg751_1: "f32[128]", arg752_1: "f32[128]", arg753_1: "f32[128]", arg754_1: "f32[128, 128]", arg755_1: "f32[128]", arg756_1: "f32[128, 128]", arg757_1: "f32[128]", arg758_1: "f32[128, 512]", arg759_1: "f32[128]", arg760_1: "f32[128, 128]", arg761_1: "f32[128]", arg762_1: "f32[128]", arg763_1: "f32[512]", arg764_1: "f32[512, 128]", arg765_1: "f32[128]", arg766_1: "f32[128, 512]", arg767_1: "f32[128]", arg768_1: "f32[128]", arg769_1: "f32[512]", arg770_1: "f32[512, 128]", arg771_1: "f32[128]", arg772_1: "f32[128, 512]", arg773_1: "f32[128]", arg774_1: "f32[128]", arg775_1: "f32[512]", arg776_1: "f32[512, 128]", arg777_1: "f32[128]", arg778_1: "f32[128, 512]", arg779_1: "f32[128]", arg780_1: "f32[128]", arg781_1: "f32[512]", arg782_1: "f32[512, 128]", arg783_1: "f32[128]", arg784_1: "f32[128, 512]", arg785_1: "f32[128]", arg786_1: "f32[128]", arg787_1: "f32[512]", arg788_1: "f32[512, 128]", arg789_1: "f32[512]", arg790_1: "f32[512]", arg791_1: "f32[128]", arg792_1: "f32[128, 512]", arg793_1: "f32[128]", arg794_1: "f32[128]", arg795_1: "f32[128]", arg796_1: "f32[128, 512]", arg797_1: "f32[128]", arg798_1: "f32[128]", arg799_1: "f32[128]", arg800_1: "f32[128, 128]", arg801_1: "f32[128]", arg802_1: "f32[128, 128]", arg803_1: "f32[128]", arg804_1: "f32[128, 512]", arg805_1: "f32[128]", arg806_1: "f32[128, 128]", arg807_1: "f32[128]", arg808_1: "f32[128]", arg809_1: "f32[512]", arg810_1: "f32[512, 128]", arg811_1: "f32[128]", arg812_1: "f32[128, 512]", arg813_1: "f32[128]", arg814_1: "f32[128]", arg815_1: "f32[512]", arg816_1: "f32[512, 128]", arg817_1: "f32[128]", arg818_1: "f32[128, 512]", arg819_1: "f32[128]", arg820_1: "f32[128]", arg821_1: "f32[512]", arg822_1: "f32[512, 128]", arg823_1: "f32[128]", arg824_1: "f32[128, 512]", arg825_1: "f32[128]", arg826_1: "f32[128]", arg827_1: "f32[512]", arg828_1: "f32[512, 128]", arg829_1: "f32[128]", arg830_1: "f32[128, 512]", arg831_1: "f32[128]", arg832_1: "f32[128]", arg833_1: "f32[512]", arg834_1: "f32[512, 128]", arg835_1: "f32[512]", arg836_1: "f32[512]", arg837_1: "f32[128]", arg838_1: "f32[128, 512]", arg839_1: "f32[128]", arg840_1: "f32[128]", arg841_1: "f32[128]", arg842_1: "f32[128, 512]", arg843_1: "f32[128]", arg844_1: "f32[128]", arg845_1: "f32[128]", arg846_1: "f32[128, 128]", arg847_1: "f32[128]", arg848_1: "f32[128, 128]", arg849_1: "f32[128]", arg850_1: "f32[128, 512]", arg851_1: "f32[128]", arg852_1: "f32[128, 128]", arg853_1: "f32[128]", arg854_1: "f32[128]", arg855_1: "f32[512]", arg856_1: "f32[512, 128]", arg857_1: "f32[128]", arg858_1: "f32[128, 512]", arg859_1: "f32[128]", arg860_1: "f32[128]", arg861_1: "f32[512]", arg862_1: "f32[512, 128]", arg863_1: "f32[128]", arg864_1: "f32[128, 512]", arg865_1: "f32[128]", arg866_1: "f32[128]", arg867_1: "f32[512]", arg868_1: "f32[512, 128]", arg869_1: "f32[128]", arg870_1: "f32[128, 512]", arg871_1: "f32[128]", arg872_1: "f32[128]", arg873_1: "f32[512]", arg874_1: "f32[512, 128]", arg875_1: "f32[128]", arg876_1: "f32[128, 512]", arg877_1: "f32[128]", arg878_1: "f32[128]", arg879_1: "f32[512]", arg880_1: "f32[512, 128]", arg881_1: "f32[512]", arg882_1: "f32[512]", arg883_1: "f32[128]", arg884_1: "f32[128, 512]", arg885_1: "f32[128]", arg886_1: "f32[128]", arg887_1: "f32[128]", arg888_1: "f32[128, 512]", arg889_1: "f32[128]", arg890_1: "f32[128]", arg891_1: "f32[128]", arg892_1: "f32[128, 128]", arg893_1: "f32[128]", arg894_1: "f32[128, 128]", arg895_1: "f32[128]", arg896_1: "f32[128, 512]", arg897_1: "f32[128]", arg898_1: "f32[128, 128]", arg899_1: "f32[128]", arg900_1: "f32[128]", arg901_1: "f32[512]", arg902_1: "f32[512, 128]", arg903_1: "f32[128]", arg904_1: "f32[128, 512]", arg905_1: "f32[128]", arg906_1: "f32[128]", arg907_1: "f32[512]", arg908_1: "f32[512, 128]", arg909_1: "f32[128]", arg910_1: "f32[128, 512]", arg911_1: "f32[128]", arg912_1: "f32[128]", arg913_1: "f32[512]", arg914_1: "f32[512, 128]", arg915_1: "f32[128]", arg916_1: "f32[128, 512]", arg917_1: "f32[128]", arg918_1: "f32[128]", arg919_1: "f32[512]", arg920_1: "f32[512, 128]", arg921_1: "f32[128]", arg922_1: "f32[128, 512]", arg923_1: "f32[128]", arg924_1: "f32[128]", arg925_1: "f32[512]", arg926_1: "f32[512, 128]", arg927_1: "f32[512]", arg928_1: "f32[512]", arg929_1: "f32[128]", arg930_1: "f32[128, 512]", arg931_1: "f32[128]", arg932_1: "f32[128]", arg933_1: "f32[128]", arg934_1: "f32[128, 512]", arg935_1: "f32[128]", arg936_1: "f32[128]", arg937_1: "f32[128]", arg938_1: "f32[128, 128]", arg939_1: "f32[128]", arg940_1: "f32[128, 128]", arg941_1: "f32[128]", arg942_1: "f32[128, 512]", arg943_1: "f32[128]", arg944_1: "f32[128, 128]", arg945_1: "f32[128]", arg946_1: "f32[128]", arg947_1: "f32[512]", arg948_1: "f32[512, 128]", arg949_1: "f32[128]", arg950_1: "f32[128, 512]", arg951_1: "f32[128]", arg952_1: "f32[128]", arg953_1: "f32[512]", arg954_1: "f32[512, 128]", arg955_1: "f32[128]", arg956_1: "f32[128, 512]", arg957_1: "f32[128]", arg958_1: "f32[128]", arg959_1: "f32[512]", arg960_1: "f32[512, 128]", arg961_1: "f32[128]", arg962_1: "f32[128, 512]", arg963_1: "f32[128]", arg964_1: "f32[128]", arg965_1: "f32[512]", arg966_1: "f32[512, 128]", arg967_1: "f32[128]", arg968_1: "f32[128, 512]", arg969_1: "f32[128]", arg970_1: "f32[128]", arg971_1: "f32[512]", arg972_1: "f32[512, 128]", arg973_1: "f32[512]", arg974_1: "f32[512]", arg975_1: "f32[128]", arg976_1: "f32[128, 512]", arg977_1: "f32[128]", arg978_1: "f32[128]", arg979_1: "f32[128]", arg980_1: "f32[128, 512]", arg981_1: "f32[128]", arg982_1: "f32[128]", arg983_1: "f32[128]", arg984_1: "f32[128, 128]", arg985_1: "f32[128]", arg986_1: "f32[128, 128]", arg987_1: "f32[128]", arg988_1: "f32[128, 512]", arg989_1: "f32[128]", arg990_1: "f32[128, 128]", arg991_1: "f32[128]", arg992_1: "f32[128]", arg993_1: "f32[512]", arg994_1: "f32[512, 128]", arg995_1: "f32[128]", arg996_1: "f32[128, 512]", arg997_1: "f32[128]", arg998_1: "f32[128]", arg999_1: "f32[512]", arg1000_1: "f32[512, 128]", arg1001_1: "f32[128]", arg1002_1: "f32[128, 512]", arg1003_1: "f32[128]", arg1004_1: "f32[128]", arg1005_1: "f32[512]", arg1006_1: "f32[512, 128]", arg1007_1: "f32[128]", arg1008_1: "f32[128, 512]", arg1009_1: "f32[128]", arg1010_1: "f32[128]", arg1011_1: "f32[512]", arg1012_1: "f32[512, 128]", arg1013_1: "f32[128]", arg1014_1: "f32[128, 512]", arg1015_1: "f32[128]", arg1016_1: "f32[128]", arg1017_1: "f32[512]", arg1018_1: "f32[512, 128]", arg1019_1: "f32[512]", arg1020_1: "f32[512]", arg1021_1: "f32[128]", arg1022_1: "f32[128, 512]", arg1023_1: "f32[128]", arg1024_1: "f32[128]", arg1025_1: "f32[128]", arg1026_1: "f32[128, 512]", arg1027_1: "f32[128]", arg1028_1: "f32[128]", arg1029_1: "f32[128]", arg1030_1: "f32[128, 128]", arg1031_1: "f32[128]", arg1032_1: "f32[128, 128]", arg1033_1: "f32[128]", arg1034_1: "f32[128, 512]", arg1035_1: "f32[128]", arg1036_1: "f32[128, 128]", arg1037_1: "f32[128]", arg1038_1: "f32[128]", arg1039_1: "f32[512]", arg1040_1: "f32[512, 128]", arg1041_1: "f32[128]", arg1042_1: "f32[128, 512]", arg1043_1: "f32[128]", arg1044_1: "f32[128]", arg1045_1: "f32[512]", arg1046_1: "f32[512, 128]", arg1047_1: "f32[128]", arg1048_1: "f32[128, 512]", arg1049_1: "f32[128]", arg1050_1: "f32[128]", arg1051_1: "f32[512]", arg1052_1: "f32[512, 128]", arg1053_1: "f32[128]", arg1054_1: "f32[128, 512]", arg1055_1: "f32[128]", arg1056_1: "f32[128]", arg1057_1: "f32[512]", arg1058_1: "f32[512, 128]", arg1059_1: "f32[128]", arg1060_1: "f32[128, 512]", arg1061_1: "f32[128]", arg1062_1: "f32[128]", arg1063_1: "f32[512]", arg1064_1: "f32[512, 128]", arg1065_1: "f32[512]", arg1066_1: "f32[512]", arg1067_1: "f32[128]", arg1068_1: "f32[128, 512]", arg1069_1: "f32[128]", arg1070_1: "f32[128]", arg1071_1: "f32[128]", arg1072_1: "f32[128, 512]", arg1073_1: "f32[128]", arg1074_1: "f32[128]", arg1075_1: "f32[128]", arg1076_1: "f32[128, 128]", arg1077_1: "f32[128]", arg1078_1: "f32[128, 128]", arg1079_1: "f32[128]", arg1080_1: "f32[128, 512]", arg1081_1: "f32[128]", arg1082_1: "f32[128, 128]", arg1083_1: "f32[128]", arg1084_1: "f32[128]", arg1085_1: "f32[512]", arg1086_1: "f32[512, 128]", arg1087_1: "f32[128]", arg1088_1: "f32[128, 512]", arg1089_1: "f32[128]", arg1090_1: "f32[128]", arg1091_1: "f32[512]", arg1092_1: "f32[512, 128]", arg1093_1: "f32[128]", arg1094_1: "f32[128, 512]", arg1095_1: "f32[128]", arg1096_1: "f32[128]", arg1097_1: "f32[512]", arg1098_1: "f32[512, 128]", arg1099_1: "f32[128]", arg1100_1: "f32[128, 512]", arg1101_1: "f32[128]", arg1102_1: "f32[128]", arg1103_1: "f32[512]", arg1104_1: "f32[512, 128]", arg1105_1: "f32[128]", arg1106_1: "f32[128, 512]", arg1107_1: "f32[128]", arg1108_1: "f32[128]", arg1109_1: "f32[512]", arg1110_1: "f32[512, 128]", arg1111_1: "f32[512]", arg1112_1: "f32[512]", arg1113_1: "f32[512]", arg1114_1: "f32[512, 512]", arg1115_1: "f32[512]", arg1116_1: "f32[512]", arg1117_1: "f32[384, 30522]", arg1118_1: "f32[30522]", arg1119_1: "i64[256, 128]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867, _shape_param_868, _shape_param_869, _shape_param_870, _shape_param_871, _shape_param_872, _shape_param_873, _shape_param_874, _shape_param_875, _shape_param_876, _shape_param_877, _shape_param_878, _shape_param_879, _shape_param_880, _shape_param_881, _shape_param_882, _shape_param_883, _shape_param_884, _shape_param_885, _shape_param_886, _shape_param_887, _shape_param_888, _shape_param_889, _shape_param_890, _shape_param_891, _shape_param_892, _shape_param_893, _shape_param_894, _shape_param_895, _shape_param_896, _shape_param_897, _shape_param_898, _shape_param_899, _shape_param_900, _shape_param_901, _shape_param_902, _shape_param_903, _shape_param_904, _shape_param_905, _shape_param_906, _shape_param_907, _shape_param_908, _shape_param_909, _shape_param_910, _shape_param_911, _shape_param_912, _shape_param_913, _shape_param_914, _shape_param_915, _shape_param_916, _shape_param_917, _shape_param_918, _shape_param_919, _shape_param_920, _shape_param_921, _shape_param_922, _shape_param_923, _shape_param_924, _shape_param_925, _shape_param_926, _shape_param_927, _shape_param_928, _shape_param_929, _shape_param_930, _shape_param_931, _shape_param_932, _shape_param_933, _shape_param_934, _shape_param_935, _shape_param_936, _shape_param_937, _shape_param_938, _shape_param_939, _shape_param_940, _shape_param_941, _shape_param_942, _shape_param_943, _shape_param_944, _shape_param_945, _shape_param_946, _shape_param_947, _shape_param_948, _shape_param_949, _shape_param_950, _shape_param_951, _shape_param_952, _shape_param_953, _shape_param_954, _shape_param_955, _shape_param_956, _shape_param_957, _shape_param_958, _shape_param_959, _shape_param_960, _shape_param_961, _shape_param_962, _shape_param_963, _shape_param_964, _shape_param_965, _shape_param_966, _shape_param_967, _shape_param_968, _shape_param_969, _shape_param_970, _shape_param_971, _shape_param_972, _shape_param_973, _shape_param_974, _shape_param_975, _shape_param_976, _shape_param_977, _shape_param_978, _shape_param_979, _shape_param_980, _shape_param_981, _shape_param_982, _shape_param_983, _shape_param_984, _shape_param_985, _shape_param_986, _shape_param_987, _shape_param_988, _shape_param_989, _shape_param_990, _shape_param_991, _shape_param_992, _shape_param_993, _shape_param_994, _shape_param_995, _shape_param_996, _shape_param_997, _shape_param_998, _shape_param_999, _shape_param_1000, _shape_param_1001, _shape_param_1002, _shape_param_1003, _shape_param_1004, _shape_param_1005, _shape_param_1006, _shape_param_1007, _shape_param_1008, _shape_param_1009, _shape_param_1010, _shape_param_1011, _shape_param_1012, _shape_param_1013, _shape_param_1014, _shape_param_1015, _shape_param_1016, _shape_param_1017, _shape_param_1018):
        # No stacktrace found for following nodes
        slice_1: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg0_1, 1, 0, 128);  arg0_1 = None
        full: "i64[256, 128]" = torch.ops.aten.full.default(_shape_param_0, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_0 = None
        embedding: "f32[256, 128, 128]" = torch.ops.aten.embedding.default(arg1_1, arg2_1, 0);  arg2_1 = None
        slice_2: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
        constant_pad_nd: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 0, 1, 0, 0], 0.0);  slice_2 = None
        slice_3: "f32[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
        constant_pad_nd_1: "f32[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_3, [0, 0, 1, 0, 0, 0], 0.0);  slice_3 = None
        cat: "f32[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
        convert_element_type: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.bfloat16);  arg3_1 = None
        convert_element_type_1: "bf16[512, 384]" = torch.ops.prims.convert_element_type.default(arg4_1, torch.bfloat16);  arg4_1 = None
        convert_element_type_2: "bf16[256, 128, 384]" = torch.ops.prims.convert_element_type.default(cat, torch.bfloat16);  cat = None
        view: "bf16[32768, 384]" = torch.ops.aten.view.default(convert_element_type_2, _shape_param_1);  convert_element_type_2 = _shape_param_1 = None
        permute: "bf16[384, 512]" = torch.ops.aten.permute.default(convert_element_type_1, [1, 0]);  convert_element_type_1 = None
        addmm: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type, view, permute);  convert_element_type = None
        view_1: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm, _shape_param_2);  _shape_param_2 = None
        embedding_1: "f32[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_1);  arg5_1 = slice_1 = None
        embedding_2: "f32[256, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full);  arg6_1 = None
        add: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = None
        add_1: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = None
        mul: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        add_2: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg8_1);  mul = arg8_1 = None
        convert_element_type_3: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg9_1, torch.bfloat16);  arg9_1 = None
        convert_element_type_4: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg10_1, torch.bfloat16);  arg10_1 = None
        convert_element_type_5: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_2, torch.bfloat16)
        view_2: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_5, _shape_param_3);  convert_element_type_5 = _shape_param_3 = None
        permute_1: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_4, [1, 0]);  convert_element_type_4 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_3, view_2, permute_1);  convert_element_type_3 = None
        view_3: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  _shape_param_4 = None
        mul_1: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, arg11_1);  view_3 = arg11_1 = None
        add_3: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, arg12_1);  mul_1 = arg12_1 = None
        convert_element_type_6: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.bfloat16);  arg13_1 = None
        convert_element_type_7: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg14_1, torch.bfloat16);  arg14_1 = None
        permute_2: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_7, [1, 0]);  convert_element_type_7 = None
        addmm_2: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_6, view_2, permute_2);  convert_element_type_6 = None
        view_4: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_2, _shape_param_5);  _shape_param_5 = None
        mul_2: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_4, arg15_1);  view_4 = arg15_1 = None
        add_4: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_2, arg16_1);  mul_2 = arg16_1 = None
        convert_element_type_8: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.bfloat16);  arg17_1 = None
        convert_element_type_9: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.bfloat16);  arg18_1 = None
        convert_element_type_10: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_4, torch.bfloat16);  add_4 = None
        view_5: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_10, _shape_param_6);  convert_element_type_10 = _shape_param_6 = None
        permute_3: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_9, [1, 0]);  convert_element_type_9 = None
        addmm_3: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_8, view_5, permute_3);  convert_element_type_8 = None
        view_6: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_3, _shape_param_7);  addmm_3 = _shape_param_7 = None
        view_7: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_6, _shape_param_8);  view_6 = _shape_param_8 = None
        permute_4: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None
        convert_element_type_11: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg19_1, torch.bfloat16);  arg19_1 = None
        convert_element_type_12: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg20_1, torch.bfloat16);  arg20_1 = None
        permute_5: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_12, [1, 0]);  convert_element_type_12 = None
        addmm_4: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_11, view_5, permute_5);  convert_element_type_11 = None
        view_8: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_4, _shape_param_9);  addmm_4 = _shape_param_9 = None
        view_9: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_8, _shape_param_10);  view_8 = _shape_param_10 = None
        permute_6: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_9, [0, 2, 1, 3]);  view_9 = None
        convert_element_type_13: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg21_1, torch.bfloat16);  arg21_1 = None
        convert_element_type_14: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.bfloat16);  arg22_1 = None
        permute_7: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_14, [1, 0]);  convert_element_type_14 = None
        addmm_5: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_13, view_2, permute_7);  convert_element_type_13 = None
        view_10: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_5, _shape_param_11);  addmm_5 = _shape_param_11 = None
        view_11: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_10, _shape_param_12);  view_10 = _shape_param_12 = None
        permute_8: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_11, [0, 2, 1, 3]);  view_11 = None
        mul_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_4, 0.4204482076268573);  permute_4 = None
        permute_9: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_6, [0, 1, 3, 2]);  permute_6 = None
        mul_4: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_9, 0.4204482076268573);  permute_9 = None
        expand: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_3, _shape_param_13);  mul_3 = _shape_param_13 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_12: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_14);  clone = _shape_param_14 = None
        expand_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_4, _shape_param_15);  mul_4 = _shape_param_15 = None
        clone_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_13: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_16);  clone_1 = _shape_param_16 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_12, view_13)
        view_14: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_17);  _shape_param_17 = None
        convert_element_type_15: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_14, torch.float32)
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_15, [-1], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_15, amax);  convert_element_type_15 = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = None
        convert_element_type_16: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_14, -inf);  view_14 = None
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_18, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_18 = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_1, convert_element_type_16);  convert_element_type_16 = None
        inductor_seeds: "i64[24]" = torch.ops.prims.inductor_seeds.default(24, device(type='cuda', index=0))
        inductor_lookup_seed: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 0)
        inductor_random: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_19, inductor_lookup_seed, 'rand');  _shape_param_19 = inductor_lookup_seed = None
        convert_element_type_17: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random, torch.bfloat16);  inductor_random = None
        gt: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_17, 0.1);  convert_element_type_17 = None
        mul_5: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt, where);  where = None
        mul_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_5, 1.1111111111111112);  mul_5 = None
        expand_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_6, _shape_param_20);  mul_6 = _shape_param_20 = None
        view_15: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_21);  expand_2 = _shape_param_21 = None
        expand_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_8, _shape_param_22);  permute_8 = _shape_param_22 = None
        clone_2: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_16: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_2, _shape_param_23);  clone_2 = _shape_param_23 = None
        bmm_1: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_15, view_16)
        view_17: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_24);  bmm_1 = _shape_param_24 = None
        permute_10: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_17, [0, 2, 1, 3]);  view_17 = None
        clone_3: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_10, memory_format = torch.contiguous_format);  permute_10 = None
        view_18: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_3, _shape_param_25);  clone_3 = _shape_param_25 = None
        convert_element_type_18: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.bfloat16);  arg23_1 = None
        convert_element_type_19: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg24_1, torch.bfloat16);  arg24_1 = None
        view_19: "bf16[32768, 128]" = torch.ops.aten.view.default(view_18, _shape_param_26);  view_18 = _shape_param_26 = None
        permute_11: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_19, [1, 0]);  convert_element_type_19 = None
        addmm_6: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_18, view_19, permute_11);  convert_element_type_18 = None
        view_20: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_6, _shape_param_27);  _shape_param_27 = None
        add_5: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_20, add_3);  view_20 = add_3 = None
        mul_7: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_5, arg25_1);  add_5 = arg25_1 = None
        add_6: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, arg26_1);  mul_7 = arg26_1 = None
        convert_element_type_20: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.bfloat16);  arg27_1 = None
        convert_element_type_21: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.bfloat16);  arg28_1 = None
        convert_element_type_22: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_6, torch.bfloat16)
        view_21: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_22, _shape_param_28);  convert_element_type_22 = _shape_param_28 = None
        permute_12: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_21, [1, 0]);  convert_element_type_21 = None
        addmm_7: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_20, view_21, permute_12);  convert_element_type_20 = None
        view_22: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_7, _shape_param_29);  addmm_7 = _shape_param_29 = None
        relu: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_22);  view_22 = None
        convert_element_type_23: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg29_1, torch.bfloat16);  arg29_1 = None
        convert_element_type_24: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg30_1, torch.bfloat16);  arg30_1 = None
        view_23: "bf16[32768, 512]" = torch.ops.aten.view.default(relu, _shape_param_30);  _shape_param_30 = None
        permute_13: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_24, [1, 0]);  convert_element_type_24 = None
        addmm_8: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_23, view_23, permute_13);  convert_element_type_23 = None
        view_24: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_8, _shape_param_31);  addmm_8 = _shape_param_31 = None
        add_7: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_24, add_6);  view_24 = add_6 = None
        mul_8: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_7, arg31_1);  arg31_1 = None
        add_8: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, arg32_1);  mul_8 = arg32_1 = None
        convert_element_type_25: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.bfloat16);  arg33_1 = None
        convert_element_type_26: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg34_1, torch.bfloat16);  arg34_1 = None
        convert_element_type_27: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_8, torch.bfloat16)
        view_25: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_27, _shape_param_32);  convert_element_type_27 = _shape_param_32 = None
        permute_14: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_26, [1, 0]);  convert_element_type_26 = None
        addmm_9: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_25, view_25, permute_14);  convert_element_type_25 = None
        view_26: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_9, _shape_param_33);  addmm_9 = _shape_param_33 = None
        relu_1: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_26);  view_26 = None
        convert_element_type_28: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg35_1, torch.bfloat16);  arg35_1 = None
        convert_element_type_29: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg36_1, torch.bfloat16);  arg36_1 = None
        view_27: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_1, _shape_param_34);  _shape_param_34 = None
        permute_15: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_29, [1, 0]);  convert_element_type_29 = None
        addmm_10: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_28, view_27, permute_15);  convert_element_type_28 = None
        view_28: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_10, _shape_param_35);  addmm_10 = _shape_param_35 = None
        add_9: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_28, add_8);  view_28 = add_8 = None
        mul_9: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, arg37_1);  arg37_1 = None
        add_10: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_9, arg38_1);  mul_9 = arg38_1 = None
        convert_element_type_30: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg39_1, torch.bfloat16);  arg39_1 = None
        convert_element_type_31: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg40_1, torch.bfloat16);  arg40_1 = None
        convert_element_type_32: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_10, torch.bfloat16)
        view_29: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_32, _shape_param_36);  convert_element_type_32 = _shape_param_36 = None
        permute_16: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_31, [1, 0]);  convert_element_type_31 = None
        addmm_11: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_30, view_29, permute_16);  convert_element_type_30 = None
        view_30: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_11, _shape_param_37);  addmm_11 = _shape_param_37 = None
        relu_2: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_30);  view_30 = None
        convert_element_type_33: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg41_1, torch.bfloat16);  arg41_1 = None
        convert_element_type_34: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.bfloat16);  arg42_1 = None
        view_31: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_2, _shape_param_38);  _shape_param_38 = None
        permute_17: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_34, [1, 0]);  convert_element_type_34 = None
        addmm_12: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_33, view_31, permute_17);  convert_element_type_33 = None
        view_32: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_12, _shape_param_39);  addmm_12 = _shape_param_39 = None
        add_11: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_32, add_10);  view_32 = add_10 = None
        mul_10: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_11, arg43_1);  arg43_1 = None
        add_12: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, arg44_1);  mul_10 = arg44_1 = None
        convert_element_type_35: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg45_1, torch.bfloat16);  arg45_1 = None
        convert_element_type_36: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg46_1, torch.bfloat16);  arg46_1 = None
        convert_element_type_37: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_12, torch.bfloat16)
        view_33: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_37, _shape_param_40);  convert_element_type_37 = _shape_param_40 = None
        permute_18: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_36, [1, 0]);  convert_element_type_36 = None
        addmm_13: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_35, view_33, permute_18);  convert_element_type_35 = None
        view_34: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_13, _shape_param_41);  addmm_13 = _shape_param_41 = None
        relu_3: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_34);  view_34 = None
        convert_element_type_38: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.bfloat16);  arg47_1 = None
        convert_element_type_39: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.bfloat16);  arg48_1 = None
        view_35: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_3, _shape_param_42);  _shape_param_42 = None
        permute_19: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_39, [1, 0]);  convert_element_type_39 = None
        addmm_14: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_38, view_35, permute_19);  convert_element_type_38 = None
        view_36: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_14, _shape_param_43);  addmm_14 = _shape_param_43 = None
        add_13: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_36, add_12);  view_36 = add_12 = None
        mul_11: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_13, arg49_1);  arg49_1 = None
        add_14: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, arg50_1);  mul_11 = arg50_1 = None
        convert_element_type_40: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg51_1, torch.bfloat16);  arg51_1 = None
        convert_element_type_41: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.bfloat16);  arg52_1 = None
        convert_element_type_42: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_14, torch.bfloat16);  add_14 = None
        view_37: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_42, _shape_param_44);  convert_element_type_42 = _shape_param_44 = None
        permute_20: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_41, [1, 0]);  convert_element_type_41 = None
        addmm_15: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_40, view_37, permute_20);  convert_element_type_40 = None
        view_38: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_15, _shape_param_45);  addmm_15 = _shape_param_45 = None
        add_15: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_38, add_2);  view_38 = add_2 = None
        mul_12: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_15, arg53_1);  arg53_1 = None
        add_16: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_12, arg54_1);  mul_12 = arg54_1 = None
        convert_element_type_43: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg55_1, torch.bfloat16);  arg55_1 = None
        convert_element_type_44: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg56_1, torch.bfloat16);  arg56_1 = None
        convert_element_type_45: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_16, torch.bfloat16)
        view_39: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_45, _shape_param_46);  convert_element_type_45 = _shape_param_46 = None
        permute_21: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_44, [1, 0]);  convert_element_type_44 = None
        addmm_16: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_43, view_39, permute_21);  convert_element_type_43 = None
        view_40: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_16, _shape_param_47);  _shape_param_47 = None
        mul_13: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_40, arg57_1);  view_40 = arg57_1 = None
        add_17: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_13, arg58_1);  mul_13 = arg58_1 = None
        convert_element_type_46: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg59_1, torch.bfloat16);  arg59_1 = None
        convert_element_type_47: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg60_1, torch.bfloat16);  arg60_1 = None
        permute_22: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_47, [1, 0]);  convert_element_type_47 = None
        addmm_17: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_46, view_39, permute_22);  convert_element_type_46 = None
        view_41: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_17, _shape_param_48);  _shape_param_48 = None
        mul_14: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_41, arg61_1);  view_41 = arg61_1 = None
        add_18: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, arg62_1);  mul_14 = arg62_1 = None
        convert_element_type_48: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.bfloat16);  arg63_1 = None
        convert_element_type_49: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg64_1, torch.bfloat16);  arg64_1 = None
        convert_element_type_50: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_18, torch.bfloat16);  add_18 = None
        view_42: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_50, _shape_param_49);  convert_element_type_50 = _shape_param_49 = None
        permute_23: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_49, [1, 0]);  convert_element_type_49 = None
        addmm_18: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_48, view_42, permute_23);  convert_element_type_48 = None
        view_43: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_18, _shape_param_50);  addmm_18 = _shape_param_50 = None
        view_44: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_43, _shape_param_51);  view_43 = _shape_param_51 = None
        permute_24: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_44, [0, 2, 1, 3]);  view_44 = None
        convert_element_type_51: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg65_1, torch.bfloat16);  arg65_1 = None
        convert_element_type_52: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg66_1, torch.bfloat16);  arg66_1 = None
        permute_25: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_52, [1, 0]);  convert_element_type_52 = None
        addmm_19: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_51, view_42, permute_25);  convert_element_type_51 = None
        view_45: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_19, _shape_param_52);  addmm_19 = _shape_param_52 = None
        view_46: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_45, _shape_param_53);  view_45 = _shape_param_53 = None
        permute_26: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        convert_element_type_53: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.bfloat16);  arg67_1 = None
        convert_element_type_54: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.bfloat16);  arg68_1 = None
        permute_27: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_54, [1, 0]);  convert_element_type_54 = None
        addmm_20: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_53, view_39, permute_27);  convert_element_type_53 = None
        view_47: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_20, _shape_param_54);  addmm_20 = _shape_param_54 = None
        view_48: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_47, _shape_param_55);  view_47 = _shape_param_55 = None
        permute_28: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_48, [0, 2, 1, 3]);  view_48 = None
        mul_15: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_24, 0.4204482076268573);  permute_24 = None
        permute_29: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_26, [0, 1, 3, 2]);  permute_26 = None
        mul_16: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_29, 0.4204482076268573);  permute_29 = None
        expand_4: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_15, _shape_param_56);  mul_15 = _shape_param_56 = None
        clone_4: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_49: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_4, _shape_param_57);  clone_4 = _shape_param_57 = None
        expand_5: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_16, _shape_param_58);  mul_16 = _shape_param_58 = None
        clone_5: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_50: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_59);  clone_5 = _shape_param_59 = None
        bmm_2: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_49, view_50)
        view_51: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_60);  bmm_2 = _shape_param_60 = None
        convert_element_type_55: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_51, torch.float32)
        amax_1: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_55, [-1], True)
        sub_1: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_55, amax_1);  convert_element_type_55 = amax_1 = None
        exp_1: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_56: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        eq_1: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_51, -inf);  view_51 = None
        logical_not_2: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        where_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_1, convert_element_type_56);  logical_not_3 = convert_element_type_56 = None
        inductor_lookup_seed_1: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 1)
        inductor_random_1: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_61, inductor_lookup_seed_1, 'rand');  _shape_param_61 = inductor_lookup_seed_1 = None
        convert_element_type_57: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_1, torch.bfloat16);  inductor_random_1 = None
        gt_1: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_57, 0.1);  convert_element_type_57 = None
        mul_17: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_1, where_1)
        mul_18: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_17, 1.1111111111111112);  mul_17 = None
        expand_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_18, _shape_param_62);  mul_18 = _shape_param_62 = None
        view_52: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_6, _shape_param_63);  expand_6 = _shape_param_63 = None
        expand_7: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_28, _shape_param_64);  permute_28 = _shape_param_64 = None
        clone_6: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_53: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_6, _shape_param_65);  clone_6 = _shape_param_65 = None
        bmm_3: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_52, view_53)
        view_54: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_3, _shape_param_66);  bmm_3 = _shape_param_66 = None
        permute_30: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_54, [0, 2, 1, 3]);  view_54 = None
        clone_7: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_30, memory_format = torch.contiguous_format);  permute_30 = None
        view_55: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_7, _shape_param_67);  clone_7 = _shape_param_67 = None
        convert_element_type_58: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg69_1, torch.bfloat16);  arg69_1 = None
        convert_element_type_59: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg70_1, torch.bfloat16);  arg70_1 = None
        view_56: "bf16[32768, 128]" = torch.ops.aten.view.default(view_55, _shape_param_68);  view_55 = _shape_param_68 = None
        permute_31: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_59, [1, 0]);  convert_element_type_59 = None
        addmm_21: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_58, view_56, permute_31);  convert_element_type_58 = None
        view_57: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_21, _shape_param_69);  _shape_param_69 = None
        add_19: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_57, add_17);  view_57 = add_17 = None
        mul_19: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_19, arg71_1);  add_19 = arg71_1 = None
        add_20: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, arg72_1);  mul_19 = arg72_1 = None
        convert_element_type_60: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.bfloat16);  arg73_1 = None
        convert_element_type_61: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg74_1, torch.bfloat16);  arg74_1 = None
        convert_element_type_62: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_20, torch.bfloat16)
        view_58: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_62, _shape_param_70);  convert_element_type_62 = _shape_param_70 = None
        permute_32: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_61, [1, 0]);  convert_element_type_61 = None
        addmm_22: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_60, view_58, permute_32);  convert_element_type_60 = None
        view_59: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_22, _shape_param_71);  addmm_22 = _shape_param_71 = None
        relu_4: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_59);  view_59 = None
        convert_element_type_63: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg75_1, torch.bfloat16);  arg75_1 = None
        convert_element_type_64: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg76_1, torch.bfloat16);  arg76_1 = None
        view_60: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_4, _shape_param_72);  _shape_param_72 = None
        permute_33: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_64, [1, 0]);  convert_element_type_64 = None
        addmm_23: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_63, view_60, permute_33);  convert_element_type_63 = None
        view_61: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_23, _shape_param_73);  addmm_23 = _shape_param_73 = None
        add_21: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_61, add_20);  view_61 = add_20 = None
        mul_20: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_21, arg77_1);  arg77_1 = None
        add_22: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_20, arg78_1);  mul_20 = arg78_1 = None
        convert_element_type_65: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg79_1, torch.bfloat16);  arg79_1 = None
        convert_element_type_66: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg80_1, torch.bfloat16);  arg80_1 = None
        convert_element_type_67: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_22, torch.bfloat16)
        view_62: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_67, _shape_param_74);  convert_element_type_67 = _shape_param_74 = None
        permute_34: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_66, [1, 0]);  convert_element_type_66 = None
        addmm_24: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_65, view_62, permute_34);  convert_element_type_65 = None
        view_63: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_24, _shape_param_75);  addmm_24 = _shape_param_75 = None
        relu_5: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_63);  view_63 = None
        convert_element_type_68: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg81_1, torch.bfloat16);  arg81_1 = None
        convert_element_type_69: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.bfloat16);  arg82_1 = None
        view_64: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_5, _shape_param_76);  _shape_param_76 = None
        permute_35: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_69, [1, 0]);  convert_element_type_69 = None
        addmm_25: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_68, view_64, permute_35);  convert_element_type_68 = None
        view_65: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_25, _shape_param_77);  addmm_25 = _shape_param_77 = None
        add_23: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_65, add_22);  view_65 = add_22 = None
        mul_21: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, arg83_1);  arg83_1 = None
        add_24: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, arg84_1);  mul_21 = arg84_1 = None
        convert_element_type_70: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg85_1, torch.bfloat16);  arg85_1 = None
        convert_element_type_71: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg86_1, torch.bfloat16);  arg86_1 = None
        convert_element_type_72: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_24, torch.bfloat16)
        view_66: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_72, _shape_param_78);  convert_element_type_72 = _shape_param_78 = None
        permute_36: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_71, [1, 0]);  convert_element_type_71 = None
        addmm_26: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_70, view_66, permute_36);  convert_element_type_70 = None
        view_67: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_26, _shape_param_79);  addmm_26 = _shape_param_79 = None
        relu_6: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_67);  view_67 = None
        convert_element_type_73: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.bfloat16);  arg87_1 = None
        convert_element_type_74: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.bfloat16);  arg88_1 = None
        view_68: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_6, _shape_param_80);  _shape_param_80 = None
        permute_37: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_74, [1, 0]);  convert_element_type_74 = None
        addmm_27: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_73, view_68, permute_37);  convert_element_type_73 = None
        view_69: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_27, _shape_param_81);  addmm_27 = _shape_param_81 = None
        add_25: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_69, add_24);  view_69 = add_24 = None
        mul_22: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, arg89_1);  arg89_1 = None
        add_26: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_22, arg90_1);  mul_22 = arg90_1 = None
        convert_element_type_75: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg91_1, torch.bfloat16);  arg91_1 = None
        convert_element_type_76: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.bfloat16);  arg92_1 = None
        convert_element_type_77: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_26, torch.bfloat16)
        view_70: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_77, _shape_param_82);  convert_element_type_77 = _shape_param_82 = None
        permute_38: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_76, [1, 0]);  convert_element_type_76 = None
        addmm_28: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_75, view_70, permute_38);  convert_element_type_75 = None
        view_71: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_28, _shape_param_83);  addmm_28 = _shape_param_83 = None
        relu_7: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_71);  view_71 = None
        convert_element_type_78: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.bfloat16);  arg93_1 = None
        convert_element_type_79: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg94_1, torch.bfloat16);  arg94_1 = None
        view_72: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_7, _shape_param_84);  _shape_param_84 = None
        permute_39: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_79, [1, 0]);  convert_element_type_79 = None
        addmm_29: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_78, view_72, permute_39);  convert_element_type_78 = None
        view_73: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_29, _shape_param_85);  addmm_29 = _shape_param_85 = None
        add_27: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_73, add_26);  view_73 = add_26 = None
        mul_23: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, arg95_1);  arg95_1 = None
        add_28: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_23, arg96_1);  mul_23 = arg96_1 = None
        convert_element_type_80: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.bfloat16);  arg97_1 = None
        convert_element_type_81: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.bfloat16);  arg98_1 = None
        convert_element_type_82: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_28, torch.bfloat16);  add_28 = None
        view_74: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_82, _shape_param_86);  convert_element_type_82 = _shape_param_86 = None
        permute_40: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_81, [1, 0]);  convert_element_type_81 = None
        addmm_30: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_80, view_74, permute_40);  convert_element_type_80 = None
        view_75: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_30, _shape_param_87);  addmm_30 = _shape_param_87 = None
        add_29: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_75, add_16);  view_75 = add_16 = None
        mul_24: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_29, arg99_1);  arg99_1 = None
        add_30: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_24, arg100_1);  mul_24 = arg100_1 = None
        convert_element_type_83: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg101_1, torch.bfloat16);  arg101_1 = None
        convert_element_type_84: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.bfloat16);  arg102_1 = None
        convert_element_type_85: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_30, torch.bfloat16)
        view_76: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_85, _shape_param_88);  convert_element_type_85 = _shape_param_88 = None
        permute_41: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_84, [1, 0]);  convert_element_type_84 = None
        addmm_31: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_83, view_76, permute_41);  convert_element_type_83 = None
        view_77: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_31, _shape_param_89);  _shape_param_89 = None
        mul_25: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_77, arg103_1);  view_77 = arg103_1 = None
        add_31: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_25, arg104_1);  mul_25 = arg104_1 = None
        convert_element_type_86: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg105_1, torch.bfloat16);  arg105_1 = None
        convert_element_type_87: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg106_1, torch.bfloat16);  arg106_1 = None
        permute_42: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_87, [1, 0]);  convert_element_type_87 = None
        addmm_32: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_86, view_76, permute_42);  convert_element_type_86 = None
        view_78: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_32, _shape_param_90);  _shape_param_90 = None
        mul_26: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_78, arg107_1);  view_78 = arg107_1 = None
        add_32: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, arg108_1);  mul_26 = arg108_1 = None
        convert_element_type_88: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg109_1, torch.bfloat16);  arg109_1 = None
        convert_element_type_89: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg110_1, torch.bfloat16);  arg110_1 = None
        convert_element_type_90: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_32, torch.bfloat16);  add_32 = None
        view_79: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_90, _shape_param_91);  convert_element_type_90 = _shape_param_91 = None
        permute_43: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_89, [1, 0]);  convert_element_type_89 = None
        addmm_33: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_88, view_79, permute_43);  convert_element_type_88 = None
        view_80: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_33, _shape_param_92);  addmm_33 = _shape_param_92 = None
        view_81: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_80, _shape_param_93);  view_80 = _shape_param_93 = None
        permute_44: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_81, [0, 2, 1, 3]);  view_81 = None
        convert_element_type_91: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg111_1, torch.bfloat16);  arg111_1 = None
        convert_element_type_92: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.bfloat16);  arg112_1 = None
        permute_45: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_92, [1, 0]);  convert_element_type_92 = None
        addmm_34: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_91, view_79, permute_45);  convert_element_type_91 = None
        view_82: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_34, _shape_param_94);  addmm_34 = _shape_param_94 = None
        view_83: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_82, _shape_param_95);  view_82 = _shape_param_95 = None
        permute_46: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_83, [0, 2, 1, 3]);  view_83 = None
        convert_element_type_93: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.bfloat16);  arg113_1 = None
        convert_element_type_94: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg114_1, torch.bfloat16);  arg114_1 = None
        permute_47: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_94, [1, 0]);  convert_element_type_94 = None
        addmm_35: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_93, view_76, permute_47);  convert_element_type_93 = None
        view_84: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_35, _shape_param_96);  addmm_35 = _shape_param_96 = None
        view_85: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_84, _shape_param_97);  view_84 = _shape_param_97 = None
        permute_48: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None
        mul_27: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_44, 0.4204482076268573);  permute_44 = None
        permute_49: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_46, [0, 1, 3, 2]);  permute_46 = None
        mul_28: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_49, 0.4204482076268573);  permute_49 = None
        expand_8: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_27, _shape_param_98);  mul_27 = _shape_param_98 = None
        clone_8: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_86: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_8, _shape_param_99);  clone_8 = _shape_param_99 = None
        expand_9: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_28, _shape_param_100);  mul_28 = _shape_param_100 = None
        clone_9: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_87: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_101);  clone_9 = _shape_param_101 = None
        bmm_4: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_86, view_87)
        view_88: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_102);  bmm_4 = _shape_param_102 = None
        convert_element_type_95: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_88, torch.float32)
        amax_2: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_95, [-1], True)
        sub_2: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_95, amax_2);  convert_element_type_95 = amax_2 = None
        exp_2: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_3: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_96: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        eq_2: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_88, -inf);  view_88 = None
        logical_not_4: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        where_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_1, convert_element_type_96);  logical_not_5 = convert_element_type_96 = None
        inductor_lookup_seed_2: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 2)
        inductor_random_2: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_103, inductor_lookup_seed_2, 'rand');  _shape_param_103 = inductor_lookup_seed_2 = None
        convert_element_type_97: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_2, torch.bfloat16);  inductor_random_2 = None
        gt_2: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_97, 0.1);  convert_element_type_97 = None
        mul_29: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_2, where_2)
        mul_30: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_29, 1.1111111111111112);  mul_29 = None
        expand_10: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_30, _shape_param_104);  mul_30 = _shape_param_104 = None
        view_89: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_105);  expand_10 = _shape_param_105 = None
        expand_11: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_48, _shape_param_106);  permute_48 = _shape_param_106 = None
        clone_10: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_90: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_10, _shape_param_107);  clone_10 = _shape_param_107 = None
        bmm_5: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_89, view_90)
        view_91: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_5, _shape_param_108);  bmm_5 = _shape_param_108 = None
        permute_50: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_91, [0, 2, 1, 3]);  view_91 = None
        clone_11: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_50, memory_format = torch.contiguous_format);  permute_50 = None
        view_92: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_11, _shape_param_109);  clone_11 = _shape_param_109 = None
        convert_element_type_98: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg115_1, torch.bfloat16);  arg115_1 = None
        convert_element_type_99: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg116_1, torch.bfloat16);  arg116_1 = None
        view_93: "bf16[32768, 128]" = torch.ops.aten.view.default(view_92, _shape_param_110);  view_92 = _shape_param_110 = None
        permute_51: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_99, [1, 0]);  convert_element_type_99 = None
        addmm_36: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_98, view_93, permute_51);  convert_element_type_98 = None
        view_94: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_36, _shape_param_111);  _shape_param_111 = None
        add_33: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_94, add_31);  view_94 = add_31 = None
        mul_31: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_33, arg117_1);  add_33 = arg117_1 = None
        add_34: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, arg118_1);  mul_31 = arg118_1 = None
        convert_element_type_100: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg119_1, torch.bfloat16);  arg119_1 = None
        convert_element_type_101: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg120_1, torch.bfloat16);  arg120_1 = None
        convert_element_type_102: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_34, torch.bfloat16)
        view_95: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_102, _shape_param_112);  convert_element_type_102 = _shape_param_112 = None
        permute_52: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_101, [1, 0]);  convert_element_type_101 = None
        addmm_37: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_100, view_95, permute_52);  convert_element_type_100 = None
        view_96: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_37, _shape_param_113);  addmm_37 = _shape_param_113 = None
        relu_8: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_96);  view_96 = None
        convert_element_type_103: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg121_1, torch.bfloat16);  arg121_1 = None
        convert_element_type_104: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.bfloat16);  arg122_1 = None
        view_97: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_8, _shape_param_114);  _shape_param_114 = None
        permute_53: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_104, [1, 0]);  convert_element_type_104 = None
        addmm_38: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_103, view_97, permute_53);  convert_element_type_103 = None
        view_98: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_38, _shape_param_115);  addmm_38 = _shape_param_115 = None
        add_35: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_98, add_34);  view_98 = add_34 = None
        mul_32: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_35, arg123_1);  arg123_1 = None
        add_36: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_32, arg124_1);  mul_32 = arg124_1 = None
        convert_element_type_105: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg125_1, torch.bfloat16);  arg125_1 = None
        convert_element_type_106: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg126_1, torch.bfloat16);  arg126_1 = None
        convert_element_type_107: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_36, torch.bfloat16)
        view_99: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_107, _shape_param_116);  convert_element_type_107 = _shape_param_116 = None
        permute_54: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_106, [1, 0]);  convert_element_type_106 = None
        addmm_39: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_105, view_99, permute_54);  convert_element_type_105 = None
        view_100: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_39, _shape_param_117);  addmm_39 = _shape_param_117 = None
        relu_9: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_100);  view_100 = None
        convert_element_type_108: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.bfloat16);  arg127_1 = None
        convert_element_type_109: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.bfloat16);  arg128_1 = None
        view_101: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_9, _shape_param_118);  _shape_param_118 = None
        permute_55: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_109, [1, 0]);  convert_element_type_109 = None
        addmm_40: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_108, view_101, permute_55);  convert_element_type_108 = None
        view_102: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_40, _shape_param_119);  addmm_40 = _shape_param_119 = None
        add_37: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_102, add_36);  view_102 = add_36 = None
        mul_33: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_37, arg129_1);  arg129_1 = None
        add_38: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_33, arg130_1);  mul_33 = arg130_1 = None
        convert_element_type_110: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg131_1, torch.bfloat16);  arg131_1 = None
        convert_element_type_111: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.bfloat16);  arg132_1 = None
        convert_element_type_112: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_38, torch.bfloat16)
        view_103: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_112, _shape_param_120);  convert_element_type_112 = _shape_param_120 = None
        permute_56: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_111, [1, 0]);  convert_element_type_111 = None
        addmm_41: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_110, view_103, permute_56);  convert_element_type_110 = None
        view_104: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_41, _shape_param_121);  addmm_41 = _shape_param_121 = None
        relu_10: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_104);  view_104 = None
        convert_element_type_113: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.bfloat16);  arg133_1 = None
        convert_element_type_114: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg134_1, torch.bfloat16);  arg134_1 = None
        view_105: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_10, _shape_param_122);  _shape_param_122 = None
        permute_57: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_114, [1, 0]);  convert_element_type_114 = None
        addmm_42: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_113, view_105, permute_57);  convert_element_type_113 = None
        view_106: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_42, _shape_param_123);  addmm_42 = _shape_param_123 = None
        add_39: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_106, add_38);  view_106 = add_38 = None
        mul_34: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_39, arg135_1);  arg135_1 = None
        add_40: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_34, arg136_1);  mul_34 = arg136_1 = None
        convert_element_type_115: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.bfloat16);  arg137_1 = None
        convert_element_type_116: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.bfloat16);  arg138_1 = None
        convert_element_type_117: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_40, torch.bfloat16)
        view_107: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_117, _shape_param_124);  convert_element_type_117 = _shape_param_124 = None
        permute_58: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_116, [1, 0]);  convert_element_type_116 = None
        addmm_43: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_115, view_107, permute_58);  convert_element_type_115 = None
        view_108: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_43, _shape_param_125);  addmm_43 = _shape_param_125 = None
        relu_11: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_108);  view_108 = None
        convert_element_type_118: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg139_1, torch.bfloat16);  arg139_1 = None
        convert_element_type_119: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg140_1, torch.bfloat16);  arg140_1 = None
        view_109: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_11, _shape_param_126);  _shape_param_126 = None
        permute_59: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_119, [1, 0]);  convert_element_type_119 = None
        addmm_44: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_118, view_109, permute_59);  convert_element_type_118 = None
        view_110: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_44, _shape_param_127);  addmm_44 = _shape_param_127 = None
        add_41: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_110, add_40);  view_110 = add_40 = None
        mul_35: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_41, arg141_1);  arg141_1 = None
        add_42: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, arg142_1);  mul_35 = arg142_1 = None
        convert_element_type_120: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.bfloat16);  arg143_1 = None
        convert_element_type_121: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg144_1, torch.bfloat16);  arg144_1 = None
        convert_element_type_122: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_42, torch.bfloat16);  add_42 = None
        view_111: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_122, _shape_param_128);  convert_element_type_122 = _shape_param_128 = None
        permute_60: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_121, [1, 0]);  convert_element_type_121 = None
        addmm_45: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_120, view_111, permute_60);  convert_element_type_120 = None
        view_112: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_45, _shape_param_129);  addmm_45 = _shape_param_129 = None
        add_43: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_112, add_30);  view_112 = add_30 = None
        mul_36: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_43, arg145_1);  arg145_1 = None
        add_44: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_36, arg146_1);  mul_36 = arg146_1 = None
        convert_element_type_123: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.bfloat16);  arg147_1 = None
        convert_element_type_124: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.bfloat16);  arg148_1 = None
        convert_element_type_125: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_44, torch.bfloat16)
        view_113: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_125, _shape_param_130);  convert_element_type_125 = _shape_param_130 = None
        permute_61: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_124, [1, 0]);  convert_element_type_124 = None
        addmm_46: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_123, view_113, permute_61);  convert_element_type_123 = None
        view_114: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_46, _shape_param_131);  _shape_param_131 = None
        mul_37: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_114, arg149_1);  view_114 = arg149_1 = None
        add_45: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, arg150_1);  mul_37 = arg150_1 = None
        convert_element_type_126: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg151_1, torch.bfloat16);  arg151_1 = None
        convert_element_type_127: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.bfloat16);  arg152_1 = None
        permute_62: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_127, [1, 0]);  convert_element_type_127 = None
        addmm_47: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_126, view_113, permute_62);  convert_element_type_126 = None
        view_115: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_47, _shape_param_132);  _shape_param_132 = None
        mul_38: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_115, arg153_1);  view_115 = arg153_1 = None
        add_46: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, arg154_1);  mul_38 = arg154_1 = None
        convert_element_type_128: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg155_1, torch.bfloat16);  arg155_1 = None
        convert_element_type_129: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg156_1, torch.bfloat16);  arg156_1 = None
        convert_element_type_130: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_46, torch.bfloat16);  add_46 = None
        view_116: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_130, _shape_param_133);  convert_element_type_130 = _shape_param_133 = None
        permute_63: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_129, [1, 0]);  convert_element_type_129 = None
        addmm_48: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_128, view_116, permute_63);  convert_element_type_128 = None
        view_117: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_48, _shape_param_134);  addmm_48 = _shape_param_134 = None
        view_118: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_117, _shape_param_135);  view_117 = _shape_param_135 = None
        permute_64: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_118, [0, 2, 1, 3]);  view_118 = None
        convert_element_type_131: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.bfloat16);  arg157_1 = None
        convert_element_type_132: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.bfloat16);  arg158_1 = None
        permute_65: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_132, [1, 0]);  convert_element_type_132 = None
        addmm_49: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_131, view_116, permute_65);  convert_element_type_131 = None
        view_119: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_49, _shape_param_136);  addmm_49 = _shape_param_136 = None
        view_120: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_119, _shape_param_137);  view_119 = _shape_param_137 = None
        permute_66: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None
        convert_element_type_133: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg159_1, torch.bfloat16);  arg159_1 = None
        convert_element_type_134: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg160_1, torch.bfloat16);  arg160_1 = None
        permute_67: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_134, [1, 0]);  convert_element_type_134 = None
        addmm_50: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_133, view_113, permute_67);  convert_element_type_133 = None
        view_121: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_50, _shape_param_138);  addmm_50 = _shape_param_138 = None
        view_122: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_121, _shape_param_139);  view_121 = _shape_param_139 = None
        permute_68: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_122, [0, 2, 1, 3]);  view_122 = None
        mul_39: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_64, 0.4204482076268573);  permute_64 = None
        permute_69: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_66, [0, 1, 3, 2]);  permute_66 = None
        mul_40: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_69, 0.4204482076268573);  permute_69 = None
        expand_12: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_39, _shape_param_140);  mul_39 = _shape_param_140 = None
        clone_12: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_123: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_12, _shape_param_141);  clone_12 = _shape_param_141 = None
        expand_13: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_40, _shape_param_142);  mul_40 = _shape_param_142 = None
        clone_13: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_124: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_143);  clone_13 = _shape_param_143 = None
        bmm_6: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_123, view_124)
        view_125: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_144);  bmm_6 = _shape_param_144 = None
        convert_element_type_135: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_125, torch.float32)
        amax_3: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_135, [-1], True)
        sub_3: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_135, amax_3);  convert_element_type_135 = amax_3 = None
        exp_3: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_136: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        eq_3: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_125, -inf);  view_125 = None
        logical_not_6: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        where_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_7, full_1, convert_element_type_136);  logical_not_7 = convert_element_type_136 = None
        inductor_lookup_seed_3: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 3)
        inductor_random_3: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_145, inductor_lookup_seed_3, 'rand');  _shape_param_145 = inductor_lookup_seed_3 = None
        convert_element_type_137: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_3, torch.bfloat16);  inductor_random_3 = None
        gt_3: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_137, 0.1);  convert_element_type_137 = None
        mul_41: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_3, where_3)
        mul_42: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_41, 1.1111111111111112);  mul_41 = None
        expand_14: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_42, _shape_param_146);  mul_42 = _shape_param_146 = None
        view_126: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_14, _shape_param_147);  expand_14 = _shape_param_147 = None
        expand_15: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_68, _shape_param_148);  permute_68 = _shape_param_148 = None
        clone_14: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_127: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_14, _shape_param_149);  clone_14 = _shape_param_149 = None
        bmm_7: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_126, view_127)
        view_128: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_7, _shape_param_150);  bmm_7 = _shape_param_150 = None
        permute_70: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_128, [0, 2, 1, 3]);  view_128 = None
        clone_15: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_70, memory_format = torch.contiguous_format);  permute_70 = None
        view_129: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_15, _shape_param_151);  clone_15 = _shape_param_151 = None
        convert_element_type_138: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg161_1, torch.bfloat16);  arg161_1 = None
        convert_element_type_139: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.bfloat16);  arg162_1 = None
        view_130: "bf16[32768, 128]" = torch.ops.aten.view.default(view_129, _shape_param_152);  view_129 = _shape_param_152 = None
        permute_71: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_139, [1, 0]);  convert_element_type_139 = None
        addmm_51: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_138, view_130, permute_71);  convert_element_type_138 = None
        view_131: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_51, _shape_param_153);  _shape_param_153 = None
        add_47: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_131, add_45);  view_131 = add_45 = None
        mul_43: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_47, arg163_1);  add_47 = arg163_1 = None
        add_48: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_43, arg164_1);  mul_43 = arg164_1 = None
        convert_element_type_140: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg165_1, torch.bfloat16);  arg165_1 = None
        convert_element_type_141: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg166_1, torch.bfloat16);  arg166_1 = None
        convert_element_type_142: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_48, torch.bfloat16)
        view_132: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_142, _shape_param_154);  convert_element_type_142 = _shape_param_154 = None
        permute_72: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_141, [1, 0]);  convert_element_type_141 = None
        addmm_52: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_140, view_132, permute_72);  convert_element_type_140 = None
        view_133: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_52, _shape_param_155);  addmm_52 = _shape_param_155 = None
        relu_12: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_133);  view_133 = None
        convert_element_type_143: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.bfloat16);  arg167_1 = None
        convert_element_type_144: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.bfloat16);  arg168_1 = None
        view_134: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_12, _shape_param_156);  _shape_param_156 = None
        permute_73: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_144, [1, 0]);  convert_element_type_144 = None
        addmm_53: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_143, view_134, permute_73);  convert_element_type_143 = None
        view_135: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_53, _shape_param_157);  addmm_53 = _shape_param_157 = None
        add_49: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_135, add_48);  view_135 = add_48 = None
        mul_44: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_49, arg169_1);  arg169_1 = None
        add_50: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_44, arg170_1);  mul_44 = arg170_1 = None
        convert_element_type_145: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg171_1, torch.bfloat16);  arg171_1 = None
        convert_element_type_146: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.bfloat16);  arg172_1 = None
        convert_element_type_147: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_50, torch.bfloat16)
        view_136: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_147, _shape_param_158);  convert_element_type_147 = _shape_param_158 = None
        permute_74: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_146, [1, 0]);  convert_element_type_146 = None
        addmm_54: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_145, view_136, permute_74);  convert_element_type_145 = None
        view_137: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_54, _shape_param_159);  addmm_54 = _shape_param_159 = None
        relu_13: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_137);  view_137 = None
        convert_element_type_148: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.bfloat16);  arg173_1 = None
        convert_element_type_149: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg174_1, torch.bfloat16);  arg174_1 = None
        view_138: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_13, _shape_param_160);  _shape_param_160 = None
        permute_75: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_149, [1, 0]);  convert_element_type_149 = None
        addmm_55: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_148, view_138, permute_75);  convert_element_type_148 = None
        view_139: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_55, _shape_param_161);  addmm_55 = _shape_param_161 = None
        add_51: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_139, add_50);  view_139 = add_50 = None
        mul_45: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_51, arg175_1);  arg175_1 = None
        add_52: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, arg176_1);  mul_45 = arg176_1 = None
        convert_element_type_150: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.bfloat16);  arg177_1 = None
        convert_element_type_151: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.bfloat16);  arg178_1 = None
        convert_element_type_152: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_52, torch.bfloat16)
        view_140: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_152, _shape_param_162);  convert_element_type_152 = _shape_param_162 = None
        permute_76: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_151, [1, 0]);  convert_element_type_151 = None
        addmm_56: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_150, view_140, permute_76);  convert_element_type_150 = None
        view_141: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_56, _shape_param_163);  addmm_56 = _shape_param_163 = None
        relu_14: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_141);  view_141 = None
        convert_element_type_153: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg179_1, torch.bfloat16);  arg179_1 = None
        convert_element_type_154: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg180_1, torch.bfloat16);  arg180_1 = None
        view_142: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_14, _shape_param_164);  _shape_param_164 = None
        permute_77: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_154, [1, 0]);  convert_element_type_154 = None
        addmm_57: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_153, view_142, permute_77);  convert_element_type_153 = None
        view_143: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_57, _shape_param_165);  addmm_57 = _shape_param_165 = None
        add_53: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_143, add_52);  view_143 = add_52 = None
        mul_46: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, arg181_1);  arg181_1 = None
        add_54: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, arg182_1);  mul_46 = arg182_1 = None
        convert_element_type_155: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.bfloat16);  arg183_1 = None
        convert_element_type_156: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg184_1, torch.bfloat16);  arg184_1 = None
        convert_element_type_157: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_54, torch.bfloat16)
        view_144: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_157, _shape_param_166);  convert_element_type_157 = _shape_param_166 = None
        permute_78: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_156, [1, 0]);  convert_element_type_156 = None
        addmm_58: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_155, view_144, permute_78);  convert_element_type_155 = None
        view_145: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_58, _shape_param_167);  addmm_58 = _shape_param_167 = None
        relu_15: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None
        convert_element_type_158: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg185_1, torch.bfloat16);  arg185_1 = None
        convert_element_type_159: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg186_1, torch.bfloat16);  arg186_1 = None
        view_146: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_15, _shape_param_168);  _shape_param_168 = None
        permute_79: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_159, [1, 0]);  convert_element_type_159 = None
        addmm_59: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_158, view_146, permute_79);  convert_element_type_158 = None
        view_147: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_59, _shape_param_169);  addmm_59 = _shape_param_169 = None
        add_55: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_54);  view_147 = add_54 = None
        mul_47: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, arg187_1);  arg187_1 = None
        add_56: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, arg188_1);  mul_47 = arg188_1 = None
        convert_element_type_160: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg189_1, torch.bfloat16);  arg189_1 = None
        convert_element_type_161: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg190_1, torch.bfloat16);  arg190_1 = None
        convert_element_type_162: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_56, torch.bfloat16);  add_56 = None
        view_148: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_162, _shape_param_170);  convert_element_type_162 = _shape_param_170 = None
        permute_80: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_161, [1, 0]);  convert_element_type_161 = None
        addmm_60: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_160, view_148, permute_80);  convert_element_type_160 = None
        view_149: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_60, _shape_param_171);  addmm_60 = _shape_param_171 = None
        add_57: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_149, add_44);  view_149 = add_44 = None
        mul_48: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_57, arg191_1);  arg191_1 = None
        add_58: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_48, arg192_1);  mul_48 = arg192_1 = None
        convert_element_type_163: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.bfloat16);  arg193_1 = None
        convert_element_type_164: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg194_1, torch.bfloat16);  arg194_1 = None
        convert_element_type_165: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_58, torch.bfloat16)
        view_150: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_165, _shape_param_172);  convert_element_type_165 = _shape_param_172 = None
        permute_81: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_164, [1, 0]);  convert_element_type_164 = None
        addmm_61: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_163, view_150, permute_81);  convert_element_type_163 = None
        view_151: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_61, _shape_param_173);  _shape_param_173 = None
        mul_49: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_151, arg195_1);  view_151 = arg195_1 = None
        add_59: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_49, arg196_1);  mul_49 = arg196_1 = None
        convert_element_type_166: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.bfloat16);  arg197_1 = None
        convert_element_type_167: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.bfloat16);  arg198_1 = None
        permute_82: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_167, [1, 0]);  convert_element_type_167 = None
        addmm_62: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_166, view_150, permute_82);  convert_element_type_166 = None
        view_152: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_62, _shape_param_174);  _shape_param_174 = None
        mul_50: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_152, arg199_1);  view_152 = arg199_1 = None
        add_60: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_50, arg200_1);  mul_50 = arg200_1 = None
        convert_element_type_168: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg201_1, torch.bfloat16);  arg201_1 = None
        convert_element_type_169: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.bfloat16);  arg202_1 = None
        convert_element_type_170: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_60, torch.bfloat16);  add_60 = None
        view_153: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_170, _shape_param_175);  convert_element_type_170 = _shape_param_175 = None
        permute_83: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_169, [1, 0]);  convert_element_type_169 = None
        addmm_63: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_168, view_153, permute_83);  convert_element_type_168 = None
        view_154: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_63, _shape_param_176);  addmm_63 = _shape_param_176 = None
        view_155: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_154, _shape_param_177);  view_154 = _shape_param_177 = None
        permute_84: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_155, [0, 2, 1, 3]);  view_155 = None
        convert_element_type_171: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.bfloat16);  arg203_1 = None
        convert_element_type_172: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg204_1, torch.bfloat16);  arg204_1 = None
        permute_85: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_172, [1, 0]);  convert_element_type_172 = None
        addmm_64: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_171, view_153, permute_85);  convert_element_type_171 = None
        view_156: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_64, _shape_param_178);  addmm_64 = _shape_param_178 = None
        view_157: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_156, _shape_param_179);  view_156 = _shape_param_179 = None
        permute_86: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_157, [0, 2, 1, 3]);  view_157 = None
        convert_element_type_173: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg205_1, torch.bfloat16);  arg205_1 = None
        convert_element_type_174: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg206_1, torch.bfloat16);  arg206_1 = None
        permute_87: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_174, [1, 0]);  convert_element_type_174 = None
        addmm_65: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_173, view_150, permute_87);  convert_element_type_173 = None
        view_158: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_65, _shape_param_180);  addmm_65 = _shape_param_180 = None
        view_159: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_158, _shape_param_181);  view_158 = _shape_param_181 = None
        permute_88: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_159, [0, 2, 1, 3]);  view_159 = None
        mul_51: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_84, 0.4204482076268573);  permute_84 = None
        permute_89: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_86, [0, 1, 3, 2]);  permute_86 = None
        mul_52: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_89, 0.4204482076268573);  permute_89 = None
        expand_16: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_51, _shape_param_182);  mul_51 = _shape_param_182 = None
        clone_16: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_160: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_16, _shape_param_183);  clone_16 = _shape_param_183 = None
        expand_17: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_52, _shape_param_184);  mul_52 = _shape_param_184 = None
        clone_17: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_161: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_185);  clone_17 = _shape_param_185 = None
        bmm_8: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_160, view_161)
        view_162: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_186);  bmm_8 = _shape_param_186 = None
        convert_element_type_175: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_162, torch.float32)
        amax_4: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_175, [-1], True)
        sub_4: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_175, amax_4);  convert_element_type_175 = amax_4 = None
        exp_4: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_5: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_176: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        eq_4: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_162, -inf);  view_162 = None
        logical_not_8: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        where_4: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_9, full_1, convert_element_type_176);  logical_not_9 = convert_element_type_176 = None
        inductor_lookup_seed_4: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 4)
        inductor_random_4: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_187, inductor_lookup_seed_4, 'rand');  _shape_param_187 = inductor_lookup_seed_4 = None
        convert_element_type_177: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_4, torch.bfloat16);  inductor_random_4 = None
        gt_4: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_177, 0.1);  convert_element_type_177 = None
        mul_53: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_4, where_4)
        mul_54: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_53, 1.1111111111111112);  mul_53 = None
        expand_18: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_54, _shape_param_188);  mul_54 = _shape_param_188 = None
        view_163: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_18, _shape_param_189);  expand_18 = _shape_param_189 = None
        expand_19: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_88, _shape_param_190);  permute_88 = _shape_param_190 = None
        clone_18: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_164: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_18, _shape_param_191);  clone_18 = _shape_param_191 = None
        bmm_9: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_163, view_164)
        view_165: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_9, _shape_param_192);  bmm_9 = _shape_param_192 = None
        permute_90: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None
        clone_19: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_90, memory_format = torch.contiguous_format);  permute_90 = None
        view_166: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_19, _shape_param_193);  clone_19 = _shape_param_193 = None
        convert_element_type_178: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.bfloat16);  arg207_1 = None
        convert_element_type_179: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.bfloat16);  arg208_1 = None
        view_167: "bf16[32768, 128]" = torch.ops.aten.view.default(view_166, _shape_param_194);  view_166 = _shape_param_194 = None
        permute_91: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_179, [1, 0]);  convert_element_type_179 = None
        addmm_66: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_178, view_167, permute_91);  convert_element_type_178 = None
        view_168: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_66, _shape_param_195);  _shape_param_195 = None
        add_61: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_168, add_59);  view_168 = add_59 = None
        mul_55: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_61, arg209_1);  add_61 = arg209_1 = None
        add_62: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, arg210_1);  mul_55 = arg210_1 = None
        convert_element_type_180: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg211_1, torch.bfloat16);  arg211_1 = None
        convert_element_type_181: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.bfloat16);  arg212_1 = None
        convert_element_type_182: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_62, torch.bfloat16)
        view_169: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_182, _shape_param_196);  convert_element_type_182 = _shape_param_196 = None
        permute_92: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_181, [1, 0]);  convert_element_type_181 = None
        addmm_67: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_180, view_169, permute_92);  convert_element_type_180 = None
        view_170: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_67, _shape_param_197);  addmm_67 = _shape_param_197 = None
        relu_16: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_170);  view_170 = None
        convert_element_type_183: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.bfloat16);  arg213_1 = None
        convert_element_type_184: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg214_1, torch.bfloat16);  arg214_1 = None
        view_171: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_16, _shape_param_198);  _shape_param_198 = None
        permute_93: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_184, [1, 0]);  convert_element_type_184 = None
        addmm_68: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_183, view_171, permute_93);  convert_element_type_183 = None
        view_172: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_68, _shape_param_199);  addmm_68 = _shape_param_199 = None
        add_63: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_172, add_62);  view_172 = add_62 = None
        mul_56: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_63, arg215_1);  arg215_1 = None
        add_64: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, arg216_1);  mul_56 = arg216_1 = None
        convert_element_type_185: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.bfloat16);  arg217_1 = None
        convert_element_type_186: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.bfloat16);  arg218_1 = None
        convert_element_type_187: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_64, torch.bfloat16)
        view_173: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_187, _shape_param_200);  convert_element_type_187 = _shape_param_200 = None
        permute_94: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_186, [1, 0]);  convert_element_type_186 = None
        addmm_69: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_185, view_173, permute_94);  convert_element_type_185 = None
        view_174: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_69, _shape_param_201);  addmm_69 = _shape_param_201 = None
        relu_17: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_174);  view_174 = None
        convert_element_type_188: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg219_1, torch.bfloat16);  arg219_1 = None
        convert_element_type_189: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg220_1, torch.bfloat16);  arg220_1 = None
        view_175: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_17, _shape_param_202);  _shape_param_202 = None
        permute_95: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_189, [1, 0]);  convert_element_type_189 = None
        addmm_70: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_188, view_175, permute_95);  convert_element_type_188 = None
        view_176: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_70, _shape_param_203);  addmm_70 = _shape_param_203 = None
        add_65: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_176, add_64);  view_176 = add_64 = None
        mul_57: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_65, arg221_1);  arg221_1 = None
        add_66: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_57, arg222_1);  mul_57 = arg222_1 = None
        convert_element_type_190: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.bfloat16);  arg223_1 = None
        convert_element_type_191: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg224_1, torch.bfloat16);  arg224_1 = None
        convert_element_type_192: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_66, torch.bfloat16)
        view_177: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_192, _shape_param_204);  convert_element_type_192 = _shape_param_204 = None
        permute_96: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_191, [1, 0]);  convert_element_type_191 = None
        addmm_71: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_190, view_177, permute_96);  convert_element_type_190 = None
        view_178: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_71, _shape_param_205);  addmm_71 = _shape_param_205 = None
        relu_18: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_178);  view_178 = None
        convert_element_type_193: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg225_1, torch.bfloat16);  arg225_1 = None
        convert_element_type_194: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg226_1, torch.bfloat16);  arg226_1 = None
        view_179: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_18, _shape_param_206);  _shape_param_206 = None
        permute_97: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_194, [1, 0]);  convert_element_type_194 = None
        addmm_72: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_193, view_179, permute_97);  convert_element_type_193 = None
        view_180: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_72, _shape_param_207);  addmm_72 = _shape_param_207 = None
        add_67: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_180, add_66);  view_180 = add_66 = None
        mul_58: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_67, arg227_1);  arg227_1 = None
        add_68: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_58, arg228_1);  mul_58 = arg228_1 = None
        convert_element_type_195: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg229_1, torch.bfloat16);  arg229_1 = None
        convert_element_type_196: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg230_1, torch.bfloat16);  arg230_1 = None
        convert_element_type_197: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_68, torch.bfloat16)
        view_181: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_197, _shape_param_208);  convert_element_type_197 = _shape_param_208 = None
        permute_98: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_196, [1, 0]);  convert_element_type_196 = None
        addmm_73: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_195, view_181, permute_98);  convert_element_type_195 = None
        view_182: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_73, _shape_param_209);  addmm_73 = _shape_param_209 = None
        relu_19: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_182);  view_182 = None
        convert_element_type_198: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg231_1, torch.bfloat16);  arg231_1 = None
        convert_element_type_199: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.bfloat16);  arg232_1 = None
        view_183: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_19, _shape_param_210);  _shape_param_210 = None
        permute_99: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_199, [1, 0]);  convert_element_type_199 = None
        addmm_74: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_198, view_183, permute_99);  convert_element_type_198 = None
        view_184: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_74, _shape_param_211);  addmm_74 = _shape_param_211 = None
        add_69: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_184, add_68);  view_184 = add_68 = None
        mul_59: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_69, arg233_1);  arg233_1 = None
        add_70: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, arg234_1);  mul_59 = arg234_1 = None
        convert_element_type_200: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg235_1, torch.bfloat16);  arg235_1 = None
        convert_element_type_201: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg236_1, torch.bfloat16);  arg236_1 = None
        convert_element_type_202: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_70, torch.bfloat16);  add_70 = None
        view_185: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_202, _shape_param_212);  convert_element_type_202 = _shape_param_212 = None
        permute_100: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_201, [1, 0]);  convert_element_type_201 = None
        addmm_75: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_200, view_185, permute_100);  convert_element_type_200 = None
        view_186: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_75, _shape_param_213);  addmm_75 = _shape_param_213 = None
        add_71: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_186, add_58);  view_186 = add_58 = None
        mul_60: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_71, arg237_1);  arg237_1 = None
        add_72: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_60, arg238_1);  mul_60 = arg238_1 = None
        convert_element_type_203: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg239_1, torch.bfloat16);  arg239_1 = None
        convert_element_type_204: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg240_1, torch.bfloat16);  arg240_1 = None
        convert_element_type_205: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_72, torch.bfloat16)
        view_187: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_205, _shape_param_214);  convert_element_type_205 = _shape_param_214 = None
        permute_101: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_204, [1, 0]);  convert_element_type_204 = None
        addmm_76: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_203, view_187, permute_101);  convert_element_type_203 = None
        view_188: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_76, _shape_param_215);  _shape_param_215 = None
        mul_61: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_188, arg241_1);  view_188 = arg241_1 = None
        add_73: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, arg242_1);  mul_61 = arg242_1 = None
        convert_element_type_206: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.bfloat16);  arg243_1 = None
        convert_element_type_207: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg244_1, torch.bfloat16);  arg244_1 = None
        permute_102: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_207, [1, 0]);  convert_element_type_207 = None
        addmm_77: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_206, view_187, permute_102);  convert_element_type_206 = None
        view_189: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_77, _shape_param_216);  _shape_param_216 = None
        mul_62: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_189, arg245_1);  view_189 = arg245_1 = None
        add_74: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_62, arg246_1);  mul_62 = arg246_1 = None
        convert_element_type_208: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.bfloat16);  arg247_1 = None
        convert_element_type_209: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.bfloat16);  arg248_1 = None
        convert_element_type_210: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_74, torch.bfloat16);  add_74 = None
        view_190: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_210, _shape_param_217);  convert_element_type_210 = _shape_param_217 = None
        permute_103: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_209, [1, 0]);  convert_element_type_209 = None
        addmm_78: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_208, view_190, permute_103);  convert_element_type_208 = None
        view_191: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_78, _shape_param_218);  addmm_78 = _shape_param_218 = None
        view_192: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_191, _shape_param_219);  view_191 = _shape_param_219 = None
        permute_104: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_192, [0, 2, 1, 3]);  view_192 = None
        convert_element_type_211: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg249_1, torch.bfloat16);  arg249_1 = None
        convert_element_type_212: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg250_1, torch.bfloat16);  arg250_1 = None
        permute_105: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_212, [1, 0]);  convert_element_type_212 = None
        addmm_79: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_211, view_190, permute_105);  convert_element_type_211 = None
        view_193: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_79, _shape_param_220);  addmm_79 = _shape_param_220 = None
        view_194: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_193, _shape_param_221);  view_193 = _shape_param_221 = None
        permute_106: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None
        convert_element_type_213: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg251_1, torch.bfloat16);  arg251_1 = None
        convert_element_type_214: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.bfloat16);  arg252_1 = None
        permute_107: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_214, [1, 0]);  convert_element_type_214 = None
        addmm_80: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_213, view_187, permute_107);  convert_element_type_213 = None
        view_195: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_80, _shape_param_222);  addmm_80 = _shape_param_222 = None
        view_196: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_195, _shape_param_223);  view_195 = _shape_param_223 = None
        permute_108: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_196, [0, 2, 1, 3]);  view_196 = None
        mul_63: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_104, 0.4204482076268573);  permute_104 = None
        permute_109: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_106, [0, 1, 3, 2]);  permute_106 = None
        mul_64: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_109, 0.4204482076268573);  permute_109 = None
        expand_20: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_63, _shape_param_224);  mul_63 = _shape_param_224 = None
        clone_20: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_197: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_20, _shape_param_225);  clone_20 = _shape_param_225 = None
        expand_21: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_64, _shape_param_226);  mul_64 = _shape_param_226 = None
        clone_21: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_198: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_227);  clone_21 = _shape_param_227 = None
        bmm_10: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_197, view_198)
        view_199: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_228);  bmm_10 = _shape_param_228 = None
        convert_element_type_215: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_199, torch.float32)
        amax_5: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_215, [-1], True)
        sub_5: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_215, amax_5);  convert_element_type_215 = amax_5 = None
        exp_5: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_6: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_216: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        eq_5: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_199, -inf);  view_199 = None
        logical_not_10: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        where_5: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_11, full_1, convert_element_type_216);  logical_not_11 = convert_element_type_216 = None
        inductor_lookup_seed_5: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 5)
        inductor_random_5: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_229, inductor_lookup_seed_5, 'rand');  _shape_param_229 = inductor_lookup_seed_5 = None
        convert_element_type_217: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_5, torch.bfloat16);  inductor_random_5 = None
        gt_5: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_217, 0.1);  convert_element_type_217 = None
        mul_65: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_5, where_5)
        mul_66: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_65, 1.1111111111111112);  mul_65 = None
        expand_22: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_66, _shape_param_230);  mul_66 = _shape_param_230 = None
        view_200: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_231);  expand_22 = _shape_param_231 = None
        expand_23: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_108, _shape_param_232);  permute_108 = _shape_param_232 = None
        clone_22: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_201: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_22, _shape_param_233);  clone_22 = _shape_param_233 = None
        bmm_11: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_200, view_201)
        view_202: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_11, _shape_param_234);  bmm_11 = _shape_param_234 = None
        permute_110: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_202, [0, 2, 1, 3]);  view_202 = None
        clone_23: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_110, memory_format = torch.contiguous_format);  permute_110 = None
        view_203: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_23, _shape_param_235);  clone_23 = _shape_param_235 = None
        convert_element_type_218: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.bfloat16);  arg253_1 = None
        convert_element_type_219: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg254_1, torch.bfloat16);  arg254_1 = None
        view_204: "bf16[32768, 128]" = torch.ops.aten.view.default(view_203, _shape_param_236);  view_203 = _shape_param_236 = None
        permute_111: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_219, [1, 0]);  convert_element_type_219 = None
        addmm_81: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_218, view_204, permute_111);  convert_element_type_218 = None
        view_205: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_81, _shape_param_237);  _shape_param_237 = None
        add_75: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_205, add_73);  view_205 = add_73 = None
        mul_67: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_75, arg255_1);  add_75 = arg255_1 = None
        add_76: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, arg256_1);  mul_67 = arg256_1 = None
        convert_element_type_220: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.bfloat16);  arg257_1 = None
        convert_element_type_221: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.bfloat16);  arg258_1 = None
        convert_element_type_222: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_76, torch.bfloat16)
        view_206: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_222, _shape_param_238);  convert_element_type_222 = _shape_param_238 = None
        permute_112: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_221, [1, 0]);  convert_element_type_221 = None
        addmm_82: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_220, view_206, permute_112);  convert_element_type_220 = None
        view_207: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_82, _shape_param_239);  addmm_82 = _shape_param_239 = None
        relu_20: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_207);  view_207 = None
        convert_element_type_223: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg259_1, torch.bfloat16);  arg259_1 = None
        convert_element_type_224: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg260_1, torch.bfloat16);  arg260_1 = None
        view_208: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_20, _shape_param_240);  _shape_param_240 = None
        permute_113: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_224, [1, 0]);  convert_element_type_224 = None
        addmm_83: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_223, view_208, permute_113);  convert_element_type_223 = None
        view_209: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_83, _shape_param_241);  addmm_83 = _shape_param_241 = None
        add_77: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_209, add_76);  view_209 = add_76 = None
        mul_68: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_77, arg261_1);  arg261_1 = None
        add_78: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, arg262_1);  mul_68 = arg262_1 = None
        convert_element_type_225: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.bfloat16);  arg263_1 = None
        convert_element_type_226: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg264_1, torch.bfloat16);  arg264_1 = None
        convert_element_type_227: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_78, torch.bfloat16)
        view_210: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_227, _shape_param_242);  convert_element_type_227 = _shape_param_242 = None
        permute_114: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_226, [1, 0]);  convert_element_type_226 = None
        addmm_84: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_225, view_210, permute_114);  convert_element_type_225 = None
        view_211: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_84, _shape_param_243);  addmm_84 = _shape_param_243 = None
        relu_21: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_211);  view_211 = None
        convert_element_type_228: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg265_1, torch.bfloat16);  arg265_1 = None
        convert_element_type_229: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg266_1, torch.bfloat16);  arg266_1 = None
        view_212: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_21, _shape_param_244);  _shape_param_244 = None
        permute_115: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_229, [1, 0]);  convert_element_type_229 = None
        addmm_85: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_228, view_212, permute_115);  convert_element_type_228 = None
        view_213: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_85, _shape_param_245);  addmm_85 = _shape_param_245 = None
        add_79: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_213, add_78);  view_213 = add_78 = None
        mul_69: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_79, arg267_1);  arg267_1 = None
        add_80: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, arg268_1);  mul_69 = arg268_1 = None
        convert_element_type_230: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg269_1, torch.bfloat16);  arg269_1 = None
        convert_element_type_231: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg270_1, torch.bfloat16);  arg270_1 = None
        convert_element_type_232: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_80, torch.bfloat16)
        view_214: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_232, _shape_param_246);  convert_element_type_232 = _shape_param_246 = None
        permute_116: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_231, [1, 0]);  convert_element_type_231 = None
        addmm_86: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_230, view_214, permute_116);  convert_element_type_230 = None
        view_215: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_86, _shape_param_247);  addmm_86 = _shape_param_247 = None
        relu_22: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_215);  view_215 = None
        convert_element_type_233: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg271_1, torch.bfloat16);  arg271_1 = None
        convert_element_type_234: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.bfloat16);  arg272_1 = None
        view_216: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_22, _shape_param_248);  _shape_param_248 = None
        permute_117: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_234, [1, 0]);  convert_element_type_234 = None
        addmm_87: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_233, view_216, permute_117);  convert_element_type_233 = None
        view_217: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_87, _shape_param_249);  addmm_87 = _shape_param_249 = None
        add_81: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_217, add_80);  view_217 = add_80 = None
        mul_70: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_81, arg273_1);  arg273_1 = None
        add_82: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_70, arg274_1);  mul_70 = arg274_1 = None
        convert_element_type_235: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg275_1, torch.bfloat16);  arg275_1 = None
        convert_element_type_236: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg276_1, torch.bfloat16);  arg276_1 = None
        convert_element_type_237: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_82, torch.bfloat16)
        view_218: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_237, _shape_param_250);  convert_element_type_237 = _shape_param_250 = None
        permute_118: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_236, [1, 0]);  convert_element_type_236 = None
        addmm_88: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_235, view_218, permute_118);  convert_element_type_235 = None
        view_219: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_88, _shape_param_251);  addmm_88 = _shape_param_251 = None
        relu_23: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_219);  view_219 = None
        convert_element_type_238: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.bfloat16);  arg277_1 = None
        convert_element_type_239: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.bfloat16);  arg278_1 = None
        view_220: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_23, _shape_param_252);  _shape_param_252 = None
        permute_119: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_239, [1, 0]);  convert_element_type_239 = None
        addmm_89: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_238, view_220, permute_119);  convert_element_type_238 = None
        view_221: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_89, _shape_param_253);  addmm_89 = _shape_param_253 = None
        add_83: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_221, add_82);  view_221 = add_82 = None
        mul_71: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, arg279_1);  arg279_1 = None
        add_84: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, arg280_1);  mul_71 = arg280_1 = None
        convert_element_type_240: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg281_1, torch.bfloat16);  arg281_1 = None
        convert_element_type_241: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.bfloat16);  arg282_1 = None
        convert_element_type_242: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_84, torch.bfloat16);  add_84 = None
        view_222: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_242, _shape_param_254);  convert_element_type_242 = _shape_param_254 = None
        permute_120: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_241, [1, 0]);  convert_element_type_241 = None
        addmm_90: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_240, view_222, permute_120);  convert_element_type_240 = None
        view_223: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_90, _shape_param_255);  addmm_90 = _shape_param_255 = None
        add_85: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_223, add_72);  view_223 = add_72 = None
        mul_72: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, arg283_1);  arg283_1 = None
        add_86: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_72, arg284_1);  mul_72 = arg284_1 = None
        convert_element_type_243: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg285_1, torch.bfloat16);  arg285_1 = None
        convert_element_type_244: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg286_1, torch.bfloat16);  arg286_1 = None
        convert_element_type_245: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_86, torch.bfloat16)
        view_224: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_245, _shape_param_256);  convert_element_type_245 = _shape_param_256 = None
        permute_121: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_244, [1, 0]);  convert_element_type_244 = None
        addmm_91: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_243, view_224, permute_121);  convert_element_type_243 = None
        view_225: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_91, _shape_param_257);  _shape_param_257 = None
        mul_73: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_225, arg287_1);  view_225 = arg287_1 = None
        add_87: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_73, arg288_1);  mul_73 = arg288_1 = None
        convert_element_type_246: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg289_1, torch.bfloat16);  arg289_1 = None
        convert_element_type_247: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg290_1, torch.bfloat16);  arg290_1 = None
        permute_122: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_247, [1, 0]);  convert_element_type_247 = None
        addmm_92: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_246, view_224, permute_122);  convert_element_type_246 = None
        view_226: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_92, _shape_param_258);  _shape_param_258 = None
        mul_74: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_226, arg291_1);  view_226 = arg291_1 = None
        add_88: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_74, arg292_1);  mul_74 = arg292_1 = None
        convert_element_type_248: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.bfloat16);  arg293_1 = None
        convert_element_type_249: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg294_1, torch.bfloat16);  arg294_1 = None
        convert_element_type_250: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_88, torch.bfloat16);  add_88 = None
        view_227: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_250, _shape_param_259);  convert_element_type_250 = _shape_param_259 = None
        permute_123: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_249, [1, 0]);  convert_element_type_249 = None
        addmm_93: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_248, view_227, permute_123);  convert_element_type_248 = None
        view_228: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_93, _shape_param_260);  addmm_93 = _shape_param_260 = None
        view_229: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_228, _shape_param_261);  view_228 = _shape_param_261 = None
        permute_124: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_229, [0, 2, 1, 3]);  view_229 = None
        convert_element_type_251: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg295_1, torch.bfloat16);  arg295_1 = None
        convert_element_type_252: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg296_1, torch.bfloat16);  arg296_1 = None
        permute_125: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_252, [1, 0]);  convert_element_type_252 = None
        addmm_94: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_251, view_227, permute_125);  convert_element_type_251 = None
        view_230: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_94, _shape_param_262);  addmm_94 = _shape_param_262 = None
        view_231: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_230, _shape_param_263);  view_230 = _shape_param_263 = None
        permute_126: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_231, [0, 2, 1, 3]);  view_231 = None
        convert_element_type_253: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.bfloat16);  arg297_1 = None
        convert_element_type_254: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.bfloat16);  arg298_1 = None
        permute_127: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_254, [1, 0]);  convert_element_type_254 = None
        addmm_95: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_253, view_224, permute_127);  convert_element_type_253 = None
        view_232: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_95, _shape_param_264);  addmm_95 = _shape_param_264 = None
        view_233: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_232, _shape_param_265);  view_232 = _shape_param_265 = None
        permute_128: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None
        mul_75: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_124, 0.4204482076268573);  permute_124 = None
        permute_129: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_126, [0, 1, 3, 2]);  permute_126 = None
        mul_76: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_129, 0.4204482076268573);  permute_129 = None
        expand_24: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_75, _shape_param_266);  mul_75 = _shape_param_266 = None
        clone_24: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_234: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_24, _shape_param_267);  clone_24 = _shape_param_267 = None
        expand_25: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_76, _shape_param_268);  mul_76 = _shape_param_268 = None
        clone_25: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_235: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_269);  clone_25 = _shape_param_269 = None
        bmm_12: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_234, view_235)
        view_236: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_270);  bmm_12 = _shape_param_270 = None
        convert_element_type_255: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_236, torch.float32)
        amax_6: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_255, [-1], True)
        sub_6: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_255, amax_6);  convert_element_type_255 = amax_6 = None
        exp_6: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_7: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_256: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        eq_6: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_236, -inf);  view_236 = None
        logical_not_12: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        where_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_13, full_1, convert_element_type_256);  logical_not_13 = convert_element_type_256 = None
        inductor_lookup_seed_6: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 6)
        inductor_random_6: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_271, inductor_lookup_seed_6, 'rand');  _shape_param_271 = inductor_lookup_seed_6 = None
        convert_element_type_257: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_6, torch.bfloat16);  inductor_random_6 = None
        gt_6: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_257, 0.1);  convert_element_type_257 = None
        mul_77: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_6, where_6)
        mul_78: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_77, 1.1111111111111112);  mul_77 = None
        expand_26: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_78, _shape_param_272);  mul_78 = _shape_param_272 = None
        view_237: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_26, _shape_param_273);  expand_26 = _shape_param_273 = None
        expand_27: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_128, _shape_param_274);  permute_128 = _shape_param_274 = None
        clone_26: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_238: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_26, _shape_param_275);  clone_26 = _shape_param_275 = None
        bmm_13: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_237, view_238)
        view_239: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_13, _shape_param_276);  bmm_13 = _shape_param_276 = None
        permute_130: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_239, [0, 2, 1, 3]);  view_239 = None
        clone_27: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_130, memory_format = torch.contiguous_format);  permute_130 = None
        view_240: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_27, _shape_param_277);  clone_27 = _shape_param_277 = None
        convert_element_type_258: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg299_1, torch.bfloat16);  arg299_1 = None
        convert_element_type_259: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg300_1, torch.bfloat16);  arg300_1 = None
        view_241: "bf16[32768, 128]" = torch.ops.aten.view.default(view_240, _shape_param_278);  view_240 = _shape_param_278 = None
        permute_131: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_259, [1, 0]);  convert_element_type_259 = None
        addmm_96: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_258, view_241, permute_131);  convert_element_type_258 = None
        view_242: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_96, _shape_param_279);  _shape_param_279 = None
        add_89: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_242, add_87);  view_242 = add_87 = None
        mul_79: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, arg301_1);  add_89 = arg301_1 = None
        add_90: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, arg302_1);  mul_79 = arg302_1 = None
        convert_element_type_260: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.bfloat16);  arg303_1 = None
        convert_element_type_261: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg304_1, torch.bfloat16);  arg304_1 = None
        convert_element_type_262: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_90, torch.bfloat16)
        view_243: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_262, _shape_param_280);  convert_element_type_262 = _shape_param_280 = None
        permute_132: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_261, [1, 0]);  convert_element_type_261 = None
        addmm_97: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_260, view_243, permute_132);  convert_element_type_260 = None
        view_244: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_97, _shape_param_281);  addmm_97 = _shape_param_281 = None
        relu_24: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_244);  view_244 = None
        convert_element_type_263: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg305_1, torch.bfloat16);  arg305_1 = None
        convert_element_type_264: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg306_1, torch.bfloat16);  arg306_1 = None
        view_245: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_24, _shape_param_282);  _shape_param_282 = None
        permute_133: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_264, [1, 0]);  convert_element_type_264 = None
        addmm_98: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_263, view_245, permute_133);  convert_element_type_263 = None
        view_246: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_98, _shape_param_283);  addmm_98 = _shape_param_283 = None
        add_91: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_246, add_90);  view_246 = add_90 = None
        mul_80: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_91, arg307_1);  arg307_1 = None
        add_92: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_80, arg308_1);  mul_80 = arg308_1 = None
        convert_element_type_265: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg309_1, torch.bfloat16);  arg309_1 = None
        convert_element_type_266: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg310_1, torch.bfloat16);  arg310_1 = None
        convert_element_type_267: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_92, torch.bfloat16)
        view_247: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_267, _shape_param_284);  convert_element_type_267 = _shape_param_284 = None
        permute_134: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_266, [1, 0]);  convert_element_type_266 = None
        addmm_99: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_265, view_247, permute_134);  convert_element_type_265 = None
        view_248: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_99, _shape_param_285);  addmm_99 = _shape_param_285 = None
        relu_25: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_248);  view_248 = None
        convert_element_type_268: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg311_1, torch.bfloat16);  arg311_1 = None
        convert_element_type_269: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.bfloat16);  arg312_1 = None
        view_249: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_25, _shape_param_286);  _shape_param_286 = None
        permute_135: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_269, [1, 0]);  convert_element_type_269 = None
        addmm_100: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_268, view_249, permute_135);  convert_element_type_268 = None
        view_250: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_100, _shape_param_287);  addmm_100 = _shape_param_287 = None
        add_93: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_250, add_92);  view_250 = add_92 = None
        mul_81: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_93, arg313_1);  arg313_1 = None
        add_94: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_81, arg314_1);  mul_81 = arg314_1 = None
        convert_element_type_270: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg315_1, torch.bfloat16);  arg315_1 = None
        convert_element_type_271: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg316_1, torch.bfloat16);  arg316_1 = None
        convert_element_type_272: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_94, torch.bfloat16)
        view_251: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_272, _shape_param_288);  convert_element_type_272 = _shape_param_288 = None
        permute_136: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_271, [1, 0]);  convert_element_type_271 = None
        addmm_101: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_270, view_251, permute_136);  convert_element_type_270 = None
        view_252: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_101, _shape_param_289);  addmm_101 = _shape_param_289 = None
        relu_26: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_252);  view_252 = None
        convert_element_type_273: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.bfloat16);  arg317_1 = None
        convert_element_type_274: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.bfloat16);  arg318_1 = None
        view_253: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_26, _shape_param_290);  _shape_param_290 = None
        permute_137: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_274, [1, 0]);  convert_element_type_274 = None
        addmm_102: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_273, view_253, permute_137);  convert_element_type_273 = None
        view_254: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_102, _shape_param_291);  addmm_102 = _shape_param_291 = None
        add_95: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_254, add_94);  view_254 = add_94 = None
        mul_82: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_95, arg319_1);  arg319_1 = None
        add_96: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_82, arg320_1);  mul_82 = arg320_1 = None
        convert_element_type_275: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg321_1, torch.bfloat16);  arg321_1 = None
        convert_element_type_276: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.bfloat16);  arg322_1 = None
        convert_element_type_277: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_96, torch.bfloat16)
        view_255: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_277, _shape_param_292);  convert_element_type_277 = _shape_param_292 = None
        permute_138: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_276, [1, 0]);  convert_element_type_276 = None
        addmm_103: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_275, view_255, permute_138);  convert_element_type_275 = None
        view_256: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_103, _shape_param_293);  addmm_103 = _shape_param_293 = None
        relu_27: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_256);  view_256 = None
        convert_element_type_278: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.bfloat16);  arg323_1 = None
        convert_element_type_279: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg324_1, torch.bfloat16);  arg324_1 = None
        view_257: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_27, _shape_param_294);  _shape_param_294 = None
        permute_139: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_279, [1, 0]);  convert_element_type_279 = None
        addmm_104: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_278, view_257, permute_139);  convert_element_type_278 = None
        view_258: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_104, _shape_param_295);  addmm_104 = _shape_param_295 = None
        add_97: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_258, add_96);  view_258 = add_96 = None
        mul_83: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_97, arg325_1);  arg325_1 = None
        add_98: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_83, arg326_1);  mul_83 = arg326_1 = None
        convert_element_type_280: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.bfloat16);  arg327_1 = None
        convert_element_type_281: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.bfloat16);  arg328_1 = None
        convert_element_type_282: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_98, torch.bfloat16);  add_98 = None
        view_259: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_282, _shape_param_296);  convert_element_type_282 = _shape_param_296 = None
        permute_140: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_281, [1, 0]);  convert_element_type_281 = None
        addmm_105: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_280, view_259, permute_140);  convert_element_type_280 = None
        view_260: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_105, _shape_param_297);  addmm_105 = _shape_param_297 = None
        add_99: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_260, add_86);  view_260 = add_86 = None
        mul_84: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_99, arg329_1);  arg329_1 = None
        add_100: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_84, arg330_1);  mul_84 = arg330_1 = None
        convert_element_type_283: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg331_1, torch.bfloat16);  arg331_1 = None
        convert_element_type_284: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.bfloat16);  arg332_1 = None
        convert_element_type_285: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_100, torch.bfloat16)
        view_261: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_285, _shape_param_298);  convert_element_type_285 = _shape_param_298 = None
        permute_141: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_284, [1, 0]);  convert_element_type_284 = None
        addmm_106: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_283, view_261, permute_141);  convert_element_type_283 = None
        view_262: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_106, _shape_param_299);  _shape_param_299 = None
        mul_85: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_262, arg333_1);  view_262 = arg333_1 = None
        add_101: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, arg334_1);  mul_85 = arg334_1 = None
        convert_element_type_286: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg335_1, torch.bfloat16);  arg335_1 = None
        convert_element_type_287: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg336_1, torch.bfloat16);  arg336_1 = None
        permute_142: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_287, [1, 0]);  convert_element_type_287 = None
        addmm_107: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_286, view_261, permute_142);  convert_element_type_286 = None
        view_263: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_107, _shape_param_300);  _shape_param_300 = None
        mul_86: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_263, arg337_1);  view_263 = arg337_1 = None
        add_102: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, arg338_1);  mul_86 = arg338_1 = None
        convert_element_type_288: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg339_1, torch.bfloat16);  arg339_1 = None
        convert_element_type_289: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg340_1, torch.bfloat16);  arg340_1 = None
        convert_element_type_290: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_102, torch.bfloat16);  add_102 = None
        view_264: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_290, _shape_param_301);  convert_element_type_290 = _shape_param_301 = None
        permute_143: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_289, [1, 0]);  convert_element_type_289 = None
        addmm_108: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_288, view_264, permute_143);  convert_element_type_288 = None
        view_265: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_108, _shape_param_302);  addmm_108 = _shape_param_302 = None
        view_266: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_265, _shape_param_303);  view_265 = _shape_param_303 = None
        permute_144: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_266, [0, 2, 1, 3]);  view_266 = None
        convert_element_type_291: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg341_1, torch.bfloat16);  arg341_1 = None
        convert_element_type_292: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.bfloat16);  arg342_1 = None
        permute_145: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_292, [1, 0]);  convert_element_type_292 = None
        addmm_109: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_291, view_264, permute_145);  convert_element_type_291 = None
        view_267: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_109, _shape_param_304);  addmm_109 = _shape_param_304 = None
        view_268: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_267, _shape_param_305);  view_267 = _shape_param_305 = None
        permute_146: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None
        convert_element_type_293: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.bfloat16);  arg343_1 = None
        convert_element_type_294: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg344_1, torch.bfloat16);  arg344_1 = None
        permute_147: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_294, [1, 0]);  convert_element_type_294 = None
        addmm_110: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_293, view_261, permute_147);  convert_element_type_293 = None
        view_269: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_110, _shape_param_306);  addmm_110 = _shape_param_306 = None
        view_270: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_269, _shape_param_307);  view_269 = _shape_param_307 = None
        permute_148: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_270, [0, 2, 1, 3]);  view_270 = None
        mul_87: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_144, 0.4204482076268573);  permute_144 = None
        permute_149: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_146, [0, 1, 3, 2]);  permute_146 = None
        mul_88: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_149, 0.4204482076268573);  permute_149 = None
        expand_28: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_87, _shape_param_308);  mul_87 = _shape_param_308 = None
        clone_28: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_271: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_28, _shape_param_309);  clone_28 = _shape_param_309 = None
        expand_29: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_88, _shape_param_310);  mul_88 = _shape_param_310 = None
        clone_29: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_272: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_311);  clone_29 = _shape_param_311 = None
        bmm_14: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_271, view_272)
        view_273: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_312);  bmm_14 = _shape_param_312 = None
        convert_element_type_295: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_273, torch.float32)
        amax_7: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_295, [-1], True)
        sub_7: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_295, amax_7);  convert_element_type_295 = amax_7 = None
        exp_7: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_8: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_296: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        eq_7: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_273, -inf);  view_273 = None
        logical_not_14: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        where_7: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_15, full_1, convert_element_type_296);  logical_not_15 = convert_element_type_296 = None
        inductor_lookup_seed_7: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 7)
        inductor_random_7: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_313, inductor_lookup_seed_7, 'rand');  _shape_param_313 = inductor_lookup_seed_7 = None
        convert_element_type_297: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_7, torch.bfloat16);  inductor_random_7 = None
        gt_7: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_297, 0.1);  convert_element_type_297 = None
        mul_89: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_7, where_7)
        mul_90: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_89, 1.1111111111111112);  mul_89 = None
        expand_30: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_90, _shape_param_314);  mul_90 = _shape_param_314 = None
        view_274: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_30, _shape_param_315);  expand_30 = _shape_param_315 = None
        expand_31: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_148, _shape_param_316);  permute_148 = _shape_param_316 = None
        clone_30: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_275: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_30, _shape_param_317);  clone_30 = _shape_param_317 = None
        bmm_15: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_274, view_275)
        view_276: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_15, _shape_param_318);  bmm_15 = _shape_param_318 = None
        permute_150: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_276, [0, 2, 1, 3]);  view_276 = None
        clone_31: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_150, memory_format = torch.contiguous_format);  permute_150 = None
        view_277: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_31, _shape_param_319);  clone_31 = _shape_param_319 = None
        convert_element_type_298: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg345_1, torch.bfloat16);  arg345_1 = None
        convert_element_type_299: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg346_1, torch.bfloat16);  arg346_1 = None
        view_278: "bf16[32768, 128]" = torch.ops.aten.view.default(view_277, _shape_param_320);  view_277 = _shape_param_320 = None
        permute_151: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_299, [1, 0]);  convert_element_type_299 = None
        addmm_111: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_298, view_278, permute_151);  convert_element_type_298 = None
        view_279: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_111, _shape_param_321);  _shape_param_321 = None
        add_103: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_101);  view_279 = add_101 = None
        mul_91: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_103, arg347_1);  add_103 = arg347_1 = None
        add_104: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, arg348_1);  mul_91 = arg348_1 = None
        convert_element_type_300: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg349_1, torch.bfloat16);  arg349_1 = None
        convert_element_type_301: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg350_1, torch.bfloat16);  arg350_1 = None
        convert_element_type_302: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_104, torch.bfloat16)
        view_280: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_302, _shape_param_322);  convert_element_type_302 = _shape_param_322 = None
        permute_152: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_301, [1, 0]);  convert_element_type_301 = None
        addmm_112: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_300, view_280, permute_152);  convert_element_type_300 = None
        view_281: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_112, _shape_param_323);  addmm_112 = _shape_param_323 = None
        relu_28: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_281);  view_281 = None
        convert_element_type_303: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg351_1, torch.bfloat16);  arg351_1 = None
        convert_element_type_304: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.bfloat16);  arg352_1 = None
        view_282: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_28, _shape_param_324);  _shape_param_324 = None
        permute_153: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_304, [1, 0]);  convert_element_type_304 = None
        addmm_113: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_303, view_282, permute_153);  convert_element_type_303 = None
        view_283: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_113, _shape_param_325);  addmm_113 = _shape_param_325 = None
        add_105: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_283, add_104);  view_283 = add_104 = None
        mul_92: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_105, arg353_1);  arg353_1 = None
        add_106: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_92, arg354_1);  mul_92 = arg354_1 = None
        convert_element_type_305: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg355_1, torch.bfloat16);  arg355_1 = None
        convert_element_type_306: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg356_1, torch.bfloat16);  arg356_1 = None
        convert_element_type_307: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_106, torch.bfloat16)
        view_284: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_307, _shape_param_326);  convert_element_type_307 = _shape_param_326 = None
        permute_154: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_306, [1, 0]);  convert_element_type_306 = None
        addmm_114: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_305, view_284, permute_154);  convert_element_type_305 = None
        view_285: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_114, _shape_param_327);  addmm_114 = _shape_param_327 = None
        relu_29: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_285);  view_285 = None
        convert_element_type_308: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.bfloat16);  arg357_1 = None
        convert_element_type_309: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.bfloat16);  arg358_1 = None
        view_286: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_29, _shape_param_328);  _shape_param_328 = None
        permute_155: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_309, [1, 0]);  convert_element_type_309 = None
        addmm_115: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_308, view_286, permute_155);  convert_element_type_308 = None
        view_287: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_115, _shape_param_329);  addmm_115 = _shape_param_329 = None
        add_107: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_287, add_106);  view_287 = add_106 = None
        mul_93: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_107, arg359_1);  arg359_1 = None
        add_108: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_93, arg360_1);  mul_93 = arg360_1 = None
        convert_element_type_310: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg361_1, torch.bfloat16);  arg361_1 = None
        convert_element_type_311: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.bfloat16);  arg362_1 = None
        convert_element_type_312: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_108, torch.bfloat16)
        view_288: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_312, _shape_param_330);  convert_element_type_312 = _shape_param_330 = None
        permute_156: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_311, [1, 0]);  convert_element_type_311 = None
        addmm_116: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_310, view_288, permute_156);  convert_element_type_310 = None
        view_289: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_116, _shape_param_331);  addmm_116 = _shape_param_331 = None
        relu_30: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_289);  view_289 = None
        convert_element_type_313: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.bfloat16);  arg363_1 = None
        convert_element_type_314: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg364_1, torch.bfloat16);  arg364_1 = None
        view_290: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_30, _shape_param_332);  _shape_param_332 = None
        permute_157: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_314, [1, 0]);  convert_element_type_314 = None
        addmm_117: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_313, view_290, permute_157);  convert_element_type_313 = None
        view_291: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_117, _shape_param_333);  addmm_117 = _shape_param_333 = None
        add_109: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_291, add_108);  view_291 = add_108 = None
        mul_94: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_109, arg365_1);  arg365_1 = None
        add_110: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_94, arg366_1);  mul_94 = arg366_1 = None
        convert_element_type_315: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg367_1, torch.bfloat16);  arg367_1 = None
        convert_element_type_316: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.bfloat16);  arg368_1 = None
        convert_element_type_317: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_110, torch.bfloat16)
        view_292: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_317, _shape_param_334);  convert_element_type_317 = _shape_param_334 = None
        permute_158: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_316, [1, 0]);  convert_element_type_316 = None
        addmm_118: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_315, view_292, permute_158);  convert_element_type_315 = None
        view_293: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_118, _shape_param_335);  addmm_118 = _shape_param_335 = None
        relu_31: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_293);  view_293 = None
        convert_element_type_318: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg369_1, torch.bfloat16);  arg369_1 = None
        convert_element_type_319: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg370_1, torch.bfloat16);  arg370_1 = None
        view_294: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_31, _shape_param_336);  _shape_param_336 = None
        permute_159: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_319, [1, 0]);  convert_element_type_319 = None
        addmm_119: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_318, view_294, permute_159);  convert_element_type_318 = None
        view_295: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_119, _shape_param_337);  addmm_119 = _shape_param_337 = None
        add_111: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_295, add_110);  view_295 = add_110 = None
        mul_95: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_111, arg371_1);  arg371_1 = None
        add_112: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, arg372_1);  mul_95 = arg372_1 = None
        convert_element_type_320: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg373_1, torch.bfloat16);  arg373_1 = None
        convert_element_type_321: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg374_1, torch.bfloat16);  arg374_1 = None
        convert_element_type_322: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_112, torch.bfloat16);  add_112 = None
        view_296: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_322, _shape_param_338);  convert_element_type_322 = _shape_param_338 = None
        permute_160: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_321, [1, 0]);  convert_element_type_321 = None
        addmm_120: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_320, view_296, permute_160);  convert_element_type_320 = None
        view_297: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_120, _shape_param_339);  addmm_120 = _shape_param_339 = None
        add_113: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_297, add_100);  view_297 = add_100 = None
        mul_96: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_113, arg375_1);  arg375_1 = None
        add_114: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_96, arg376_1);  mul_96 = arg376_1 = None
        convert_element_type_323: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.bfloat16);  arg377_1 = None
        convert_element_type_324: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.bfloat16);  arg378_1 = None
        convert_element_type_325: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_114, torch.bfloat16)
        view_298: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_325, _shape_param_340);  convert_element_type_325 = _shape_param_340 = None
        permute_161: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_324, [1, 0]);  convert_element_type_324 = None
        addmm_121: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_323, view_298, permute_161);  convert_element_type_323 = None
        view_299: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_121, _shape_param_341);  _shape_param_341 = None
        mul_97: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_299, arg379_1);  view_299 = arg379_1 = None
        add_115: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_97, arg380_1);  mul_97 = arg380_1 = None
        convert_element_type_326: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg381_1, torch.bfloat16);  arg381_1 = None
        convert_element_type_327: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg382_1, torch.bfloat16);  arg382_1 = None
        permute_162: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_327, [1, 0]);  convert_element_type_327 = None
        addmm_122: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_326, view_298, permute_162);  convert_element_type_326 = None
        view_300: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_122, _shape_param_342);  _shape_param_342 = None
        mul_98: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_300, arg383_1);  view_300 = arg383_1 = None
        add_116: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, arg384_1);  mul_98 = arg384_1 = None
        convert_element_type_328: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg385_1, torch.bfloat16);  arg385_1 = None
        convert_element_type_329: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg386_1, torch.bfloat16);  arg386_1 = None
        convert_element_type_330: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_116, torch.bfloat16);  add_116 = None
        view_301: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_330, _shape_param_343);  convert_element_type_330 = _shape_param_343 = None
        permute_163: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_329, [1, 0]);  convert_element_type_329 = None
        addmm_123: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_328, view_301, permute_163);  convert_element_type_328 = None
        view_302: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_123, _shape_param_344);  addmm_123 = _shape_param_344 = None
        view_303: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_302, _shape_param_345);  view_302 = _shape_param_345 = None
        permute_164: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_303, [0, 2, 1, 3]);  view_303 = None
        convert_element_type_331: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.bfloat16);  arg387_1 = None
        convert_element_type_332: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg388_1, torch.bfloat16);  arg388_1 = None
        permute_165: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_332, [1, 0]);  convert_element_type_332 = None
        addmm_124: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_331, view_301, permute_165);  convert_element_type_331 = None
        view_304: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_124, _shape_param_346);  addmm_124 = _shape_param_346 = None
        view_305: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_304, _shape_param_347);  view_304 = _shape_param_347 = None
        permute_166: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_305, [0, 2, 1, 3]);  view_305 = None
        convert_element_type_333: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg389_1, torch.bfloat16);  arg389_1 = None
        convert_element_type_334: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg390_1, torch.bfloat16);  arg390_1 = None
        permute_167: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_334, [1, 0]);  convert_element_type_334 = None
        addmm_125: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_333, view_298, permute_167);  convert_element_type_333 = None
        view_306: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_125, _shape_param_348);  addmm_125 = _shape_param_348 = None
        view_307: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_306, _shape_param_349);  view_306 = _shape_param_349 = None
        permute_168: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_307, [0, 2, 1, 3]);  view_307 = None
        mul_99: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_164, 0.4204482076268573);  permute_164 = None
        permute_169: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_166, [0, 1, 3, 2]);  permute_166 = None
        mul_100: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_169, 0.4204482076268573);  permute_169 = None
        expand_32: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_99, _shape_param_350);  mul_99 = _shape_param_350 = None
        clone_32: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_308: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_32, _shape_param_351);  clone_32 = _shape_param_351 = None
        expand_33: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_100, _shape_param_352);  mul_100 = _shape_param_352 = None
        clone_33: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_309: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_353);  clone_33 = _shape_param_353 = None
        bmm_16: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_308, view_309)
        view_310: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_354);  bmm_16 = _shape_param_354 = None
        convert_element_type_335: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_310, torch.float32)
        amax_8: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_335, [-1], True)
        sub_8: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_335, amax_8);  convert_element_type_335 = amax_8 = None
        exp_8: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_9: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_336: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        eq_8: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_310, -inf);  view_310 = None
        logical_not_16: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        where_8: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_17, full_1, convert_element_type_336);  logical_not_17 = convert_element_type_336 = None
        inductor_lookup_seed_8: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 8)
        inductor_random_8: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_355, inductor_lookup_seed_8, 'rand');  _shape_param_355 = inductor_lookup_seed_8 = None
        convert_element_type_337: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_8, torch.bfloat16);  inductor_random_8 = None
        gt_8: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_337, 0.1);  convert_element_type_337 = None
        mul_101: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_8, where_8)
        mul_102: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_101, 1.1111111111111112);  mul_101 = None
        expand_34: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_102, _shape_param_356);  mul_102 = _shape_param_356 = None
        view_311: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_34, _shape_param_357);  expand_34 = _shape_param_357 = None
        expand_35: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_168, _shape_param_358);  permute_168 = _shape_param_358 = None
        clone_34: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_312: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_34, _shape_param_359);  clone_34 = _shape_param_359 = None
        bmm_17: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_311, view_312)
        view_313: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_17, _shape_param_360);  bmm_17 = _shape_param_360 = None
        permute_170: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_313, [0, 2, 1, 3]);  view_313 = None
        clone_35: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None
        view_314: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_35, _shape_param_361);  clone_35 = _shape_param_361 = None
        convert_element_type_338: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg391_1, torch.bfloat16);  arg391_1 = None
        convert_element_type_339: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg392_1, torch.bfloat16);  arg392_1 = None
        view_315: "bf16[32768, 128]" = torch.ops.aten.view.default(view_314, _shape_param_362);  view_314 = _shape_param_362 = None
        permute_171: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_339, [1, 0]);  convert_element_type_339 = None
        addmm_126: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_338, view_315, permute_171);  convert_element_type_338 = None
        view_316: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_126, _shape_param_363);  _shape_param_363 = None
        add_117: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_316, add_115);  view_316 = add_115 = None
        mul_103: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, arg393_1);  add_117 = arg393_1 = None
        add_118: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_103, arg394_1);  mul_103 = arg394_1 = None
        convert_element_type_340: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg395_1, torch.bfloat16);  arg395_1 = None
        convert_element_type_341: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg396_1, torch.bfloat16);  arg396_1 = None
        convert_element_type_342: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_118, torch.bfloat16)
        view_317: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_342, _shape_param_364);  convert_element_type_342 = _shape_param_364 = None
        permute_172: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_341, [1, 0]);  convert_element_type_341 = None
        addmm_127: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_340, view_317, permute_172);  convert_element_type_340 = None
        view_318: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_127, _shape_param_365);  addmm_127 = _shape_param_365 = None
        relu_32: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_318);  view_318 = None
        convert_element_type_343: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg397_1, torch.bfloat16);  arg397_1 = None
        convert_element_type_344: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg398_1, torch.bfloat16);  arg398_1 = None
        view_319: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_32, _shape_param_366);  _shape_param_366 = None
        permute_173: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_344, [1, 0]);  convert_element_type_344 = None
        addmm_128: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_343, view_319, permute_173);  convert_element_type_343 = None
        view_320: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_128, _shape_param_367);  addmm_128 = _shape_param_367 = None
        add_119: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_320, add_118);  view_320 = add_118 = None
        mul_104: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, arg399_1);  arg399_1 = None
        add_120: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, arg400_1);  mul_104 = arg400_1 = None
        convert_element_type_345: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg401_1, torch.bfloat16);  arg401_1 = None
        convert_element_type_346: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg402_1, torch.bfloat16);  arg402_1 = None
        convert_element_type_347: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_120, torch.bfloat16)
        view_321: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_347, _shape_param_368);  convert_element_type_347 = _shape_param_368 = None
        permute_174: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_346, [1, 0]);  convert_element_type_346 = None
        addmm_129: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_345, view_321, permute_174);  convert_element_type_345 = None
        view_322: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_129, _shape_param_369);  addmm_129 = _shape_param_369 = None
        relu_33: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_322);  view_322 = None
        convert_element_type_348: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg403_1, torch.bfloat16);  arg403_1 = None
        convert_element_type_349: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg404_1, torch.bfloat16);  arg404_1 = None
        view_323: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_33, _shape_param_370);  _shape_param_370 = None
        permute_175: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_349, [1, 0]);  convert_element_type_349 = None
        addmm_130: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_348, view_323, permute_175);  convert_element_type_348 = None
        view_324: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_130, _shape_param_371);  addmm_130 = _shape_param_371 = None
        add_121: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_324, add_120);  view_324 = add_120 = None
        mul_105: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_121, arg405_1);  arg405_1 = None
        add_122: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_105, arg406_1);  mul_105 = arg406_1 = None
        convert_element_type_350: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg407_1, torch.bfloat16);  arg407_1 = None
        convert_element_type_351: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.bfloat16);  arg408_1 = None
        convert_element_type_352: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_122, torch.bfloat16)
        view_325: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_352, _shape_param_372);  convert_element_type_352 = _shape_param_372 = None
        permute_176: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_351, [1, 0]);  convert_element_type_351 = None
        addmm_131: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_350, view_325, permute_176);  convert_element_type_350 = None
        view_326: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_131, _shape_param_373);  addmm_131 = _shape_param_373 = None
        relu_34: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_326);  view_326 = None
        convert_element_type_353: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg409_1, torch.bfloat16);  arg409_1 = None
        convert_element_type_354: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg410_1, torch.bfloat16);  arg410_1 = None
        view_327: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_34, _shape_param_374);  _shape_param_374 = None
        permute_177: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_354, [1, 0]);  convert_element_type_354 = None
        addmm_132: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_353, view_327, permute_177);  convert_element_type_353 = None
        view_328: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_132, _shape_param_375);  addmm_132 = _shape_param_375 = None
        add_123: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_328, add_122);  view_328 = add_122 = None
        mul_106: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_123, arg411_1);  arg411_1 = None
        add_124: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, arg412_1);  mul_106 = arg412_1 = None
        convert_element_type_355: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg413_1, torch.bfloat16);  arg413_1 = None
        convert_element_type_356: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg414_1, torch.bfloat16);  arg414_1 = None
        convert_element_type_357: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_124, torch.bfloat16)
        view_329: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_357, _shape_param_376);  convert_element_type_357 = _shape_param_376 = None
        permute_178: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_356, [1, 0]);  convert_element_type_356 = None
        addmm_133: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_355, view_329, permute_178);  convert_element_type_355 = None
        view_330: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_133, _shape_param_377);  addmm_133 = _shape_param_377 = None
        relu_35: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_330);  view_330 = None
        convert_element_type_358: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg415_1, torch.bfloat16);  arg415_1 = None
        convert_element_type_359: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg416_1, torch.bfloat16);  arg416_1 = None
        view_331: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_35, _shape_param_378);  _shape_param_378 = None
        permute_179: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_359, [1, 0]);  convert_element_type_359 = None
        addmm_134: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_358, view_331, permute_179);  convert_element_type_358 = None
        view_332: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_134, _shape_param_379);  addmm_134 = _shape_param_379 = None
        add_125: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_332, add_124);  view_332 = add_124 = None
        mul_107: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_125, arg417_1);  arg417_1 = None
        add_126: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, arg418_1);  mul_107 = arg418_1 = None
        convert_element_type_360: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg419_1, torch.bfloat16);  arg419_1 = None
        convert_element_type_361: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg420_1, torch.bfloat16);  arg420_1 = None
        convert_element_type_362: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_126, torch.bfloat16);  add_126 = None
        view_333: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_362, _shape_param_380);  convert_element_type_362 = _shape_param_380 = None
        permute_180: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_361, [1, 0]);  convert_element_type_361 = None
        addmm_135: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_360, view_333, permute_180);  convert_element_type_360 = None
        view_334: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_135, _shape_param_381);  addmm_135 = _shape_param_381 = None
        add_127: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_334, add_114);  view_334 = add_114 = None
        mul_108: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_127, arg421_1);  arg421_1 = None
        add_128: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_108, arg422_1);  mul_108 = arg422_1 = None
        convert_element_type_363: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg423_1, torch.bfloat16);  arg423_1 = None
        convert_element_type_364: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg424_1, torch.bfloat16);  arg424_1 = None
        convert_element_type_365: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_128, torch.bfloat16)
        view_335: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_365, _shape_param_382);  convert_element_type_365 = _shape_param_382 = None
        permute_181: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_364, [1, 0]);  convert_element_type_364 = None
        addmm_136: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_363, view_335, permute_181);  convert_element_type_363 = None
        view_336: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_136, _shape_param_383);  _shape_param_383 = None
        mul_109: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_336, arg425_1);  view_336 = arg425_1 = None
        add_129: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, arg426_1);  mul_109 = arg426_1 = None
        convert_element_type_366: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg427_1, torch.bfloat16);  arg427_1 = None
        convert_element_type_367: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg428_1, torch.bfloat16);  arg428_1 = None
        permute_182: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_367, [1, 0]);  convert_element_type_367 = None
        addmm_137: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_366, view_335, permute_182);  convert_element_type_366 = None
        view_337: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_137, _shape_param_384);  _shape_param_384 = None
        mul_110: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_337, arg429_1);  view_337 = arg429_1 = None
        add_130: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_110, arg430_1);  mul_110 = arg430_1 = None
        convert_element_type_368: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg431_1, torch.bfloat16);  arg431_1 = None
        convert_element_type_369: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg432_1, torch.bfloat16);  arg432_1 = None
        convert_element_type_370: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_130, torch.bfloat16);  add_130 = None
        view_338: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_370, _shape_param_385);  convert_element_type_370 = _shape_param_385 = None
        permute_183: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_369, [1, 0]);  convert_element_type_369 = None
        addmm_138: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_368, view_338, permute_183);  convert_element_type_368 = None
        view_339: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_138, _shape_param_386);  addmm_138 = _shape_param_386 = None
        view_340: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_339, _shape_param_387);  view_339 = _shape_param_387 = None
        permute_184: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_340, [0, 2, 1, 3]);  view_340 = None
        convert_element_type_371: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg433_1, torch.bfloat16);  arg433_1 = None
        convert_element_type_372: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg434_1, torch.bfloat16);  arg434_1 = None
        permute_185: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_372, [1, 0]);  convert_element_type_372 = None
        addmm_139: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_371, view_338, permute_185);  convert_element_type_371 = None
        view_341: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_139, _shape_param_388);  addmm_139 = _shape_param_388 = None
        view_342: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_341, _shape_param_389);  view_341 = _shape_param_389 = None
        permute_186: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        convert_element_type_373: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg435_1, torch.bfloat16);  arg435_1 = None
        convert_element_type_374: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg436_1, torch.bfloat16);  arg436_1 = None
        permute_187: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_374, [1, 0]);  convert_element_type_374 = None
        addmm_140: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_373, view_335, permute_187);  convert_element_type_373 = None
        view_343: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_140, _shape_param_390);  addmm_140 = _shape_param_390 = None
        view_344: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_343, _shape_param_391);  view_343 = _shape_param_391 = None
        permute_188: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_344, [0, 2, 1, 3]);  view_344 = None
        mul_111: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_184, 0.4204482076268573);  permute_184 = None
        permute_189: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        mul_112: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_189, 0.4204482076268573);  permute_189 = None
        expand_36: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_111, _shape_param_392);  mul_111 = _shape_param_392 = None
        clone_36: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_345: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_36, _shape_param_393);  clone_36 = _shape_param_393 = None
        expand_37: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_112, _shape_param_394);  mul_112 = _shape_param_394 = None
        clone_37: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_346: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_395);  clone_37 = _shape_param_395 = None
        bmm_18: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_345, view_346)
        view_347: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_396);  bmm_18 = _shape_param_396 = None
        convert_element_type_375: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_347, torch.float32)
        amax_9: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_375, [-1], True)
        sub_9: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_375, amax_9);  convert_element_type_375 = amax_9 = None
        exp_9: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_10: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_376: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        eq_9: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_347, -inf);  view_347 = None
        logical_not_18: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        where_9: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_19, full_1, convert_element_type_376);  logical_not_19 = convert_element_type_376 = None
        inductor_lookup_seed_9: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 9)
        inductor_random_9: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_397, inductor_lookup_seed_9, 'rand');  _shape_param_397 = inductor_lookup_seed_9 = None
        convert_element_type_377: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_9, torch.bfloat16);  inductor_random_9 = None
        gt_9: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_377, 0.1);  convert_element_type_377 = None
        mul_113: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_9, where_9)
        mul_114: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_113, 1.1111111111111112);  mul_113 = None
        expand_38: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_114, _shape_param_398);  mul_114 = _shape_param_398 = None
        view_348: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_38, _shape_param_399);  expand_38 = _shape_param_399 = None
        expand_39: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_188, _shape_param_400);  permute_188 = _shape_param_400 = None
        clone_38: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_349: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_38, _shape_param_401);  clone_38 = _shape_param_401 = None
        bmm_19: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_348, view_349)
        view_350: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_19, _shape_param_402);  bmm_19 = _shape_param_402 = None
        permute_190: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_350, [0, 2, 1, 3]);  view_350 = None
        clone_39: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_190, memory_format = torch.contiguous_format);  permute_190 = None
        view_351: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_39, _shape_param_403);  clone_39 = _shape_param_403 = None
        convert_element_type_378: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.bfloat16);  arg437_1 = None
        convert_element_type_379: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg438_1, torch.bfloat16);  arg438_1 = None
        view_352: "bf16[32768, 128]" = torch.ops.aten.view.default(view_351, _shape_param_404);  view_351 = _shape_param_404 = None
        permute_191: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_379, [1, 0]);  convert_element_type_379 = None
        addmm_141: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_378, view_352, permute_191);  convert_element_type_378 = None
        view_353: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_141, _shape_param_405);  _shape_param_405 = None
        add_131: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_353, add_129);  view_353 = add_129 = None
        mul_115: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_131, arg439_1);  add_131 = arg439_1 = None
        add_132: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, arg440_1);  mul_115 = arg440_1 = None
        convert_element_type_380: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg441_1, torch.bfloat16);  arg441_1 = None
        convert_element_type_381: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg442_1, torch.bfloat16);  arg442_1 = None
        convert_element_type_382: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_132, torch.bfloat16)
        view_354: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_382, _shape_param_406);  convert_element_type_382 = _shape_param_406 = None
        permute_192: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_381, [1, 0]);  convert_element_type_381 = None
        addmm_142: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_380, view_354, permute_192);  convert_element_type_380 = None
        view_355: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_142, _shape_param_407);  addmm_142 = _shape_param_407 = None
        relu_36: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_355);  view_355 = None
        convert_element_type_383: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg443_1, torch.bfloat16);  arg443_1 = None
        convert_element_type_384: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg444_1, torch.bfloat16);  arg444_1 = None
        view_356: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_36, _shape_param_408);  _shape_param_408 = None
        permute_193: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_384, [1, 0]);  convert_element_type_384 = None
        addmm_143: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_383, view_356, permute_193);  convert_element_type_383 = None
        view_357: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_143, _shape_param_409);  addmm_143 = _shape_param_409 = None
        add_133: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_357, add_132);  view_357 = add_132 = None
        mul_116: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_133, arg445_1);  arg445_1 = None
        add_134: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_116, arg446_1);  mul_116 = arg446_1 = None
        convert_element_type_385: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg447_1, torch.bfloat16);  arg447_1 = None
        convert_element_type_386: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg448_1, torch.bfloat16);  arg448_1 = None
        convert_element_type_387: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_134, torch.bfloat16)
        view_358: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_387, _shape_param_410);  convert_element_type_387 = _shape_param_410 = None
        permute_194: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_386, [1, 0]);  convert_element_type_386 = None
        addmm_144: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_385, view_358, permute_194);  convert_element_type_385 = None
        view_359: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_144, _shape_param_411);  addmm_144 = _shape_param_411 = None
        relu_37: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_359);  view_359 = None
        convert_element_type_388: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg449_1, torch.bfloat16);  arg449_1 = None
        convert_element_type_389: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg450_1, torch.bfloat16);  arg450_1 = None
        view_360: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_37, _shape_param_412);  _shape_param_412 = None
        permute_195: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_389, [1, 0]);  convert_element_type_389 = None
        addmm_145: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_388, view_360, permute_195);  convert_element_type_388 = None
        view_361: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_145, _shape_param_413);  addmm_145 = _shape_param_413 = None
        add_135: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_361, add_134);  view_361 = add_134 = None
        mul_117: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_135, arg451_1);  arg451_1 = None
        add_136: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, arg452_1);  mul_117 = arg452_1 = None
        convert_element_type_390: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg453_1, torch.bfloat16);  arg453_1 = None
        convert_element_type_391: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg454_1, torch.bfloat16);  arg454_1 = None
        convert_element_type_392: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_136, torch.bfloat16)
        view_362: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_392, _shape_param_414);  convert_element_type_392 = _shape_param_414 = None
        permute_196: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_391, [1, 0]);  convert_element_type_391 = None
        addmm_146: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_390, view_362, permute_196);  convert_element_type_390 = None
        view_363: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_146, _shape_param_415);  addmm_146 = _shape_param_415 = None
        relu_38: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_363);  view_363 = None
        convert_element_type_393: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg455_1, torch.bfloat16);  arg455_1 = None
        convert_element_type_394: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg456_1, torch.bfloat16);  arg456_1 = None
        view_364: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_38, _shape_param_416);  _shape_param_416 = None
        permute_197: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_394, [1, 0]);  convert_element_type_394 = None
        addmm_147: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_393, view_364, permute_197);  convert_element_type_393 = None
        view_365: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_147, _shape_param_417);  addmm_147 = _shape_param_417 = None
        add_137: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_365, add_136);  view_365 = add_136 = None
        mul_118: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_137, arg457_1);  arg457_1 = None
        add_138: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, arg458_1);  mul_118 = arg458_1 = None
        convert_element_type_395: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg459_1, torch.bfloat16);  arg459_1 = None
        convert_element_type_396: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg460_1, torch.bfloat16);  arg460_1 = None
        convert_element_type_397: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_138, torch.bfloat16)
        view_366: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_397, _shape_param_418);  convert_element_type_397 = _shape_param_418 = None
        permute_198: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_396, [1, 0]);  convert_element_type_396 = None
        addmm_148: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_395, view_366, permute_198);  convert_element_type_395 = None
        view_367: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_148, _shape_param_419);  addmm_148 = _shape_param_419 = None
        relu_39: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_367);  view_367 = None
        convert_element_type_398: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg461_1, torch.bfloat16);  arg461_1 = None
        convert_element_type_399: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg462_1, torch.bfloat16);  arg462_1 = None
        view_368: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_39, _shape_param_420);  _shape_param_420 = None
        permute_199: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_399, [1, 0]);  convert_element_type_399 = None
        addmm_149: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_398, view_368, permute_199);  convert_element_type_398 = None
        view_369: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_149, _shape_param_421);  addmm_149 = _shape_param_421 = None
        add_139: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_369, add_138);  view_369 = add_138 = None
        mul_119: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_139, arg463_1);  arg463_1 = None
        add_140: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, arg464_1);  mul_119 = arg464_1 = None
        convert_element_type_400: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg465_1, torch.bfloat16);  arg465_1 = None
        convert_element_type_401: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg466_1, torch.bfloat16);  arg466_1 = None
        convert_element_type_402: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_140, torch.bfloat16);  add_140 = None
        view_370: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_402, _shape_param_422);  convert_element_type_402 = _shape_param_422 = None
        permute_200: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_401, [1, 0]);  convert_element_type_401 = None
        addmm_150: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_400, view_370, permute_200);  convert_element_type_400 = None
        view_371: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_150, _shape_param_423);  addmm_150 = _shape_param_423 = None
        add_141: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_371, add_128);  view_371 = add_128 = None
        mul_120: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_141, arg467_1);  arg467_1 = None
        add_142: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_120, arg468_1);  mul_120 = arg468_1 = None
        convert_element_type_403: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg469_1, torch.bfloat16);  arg469_1 = None
        convert_element_type_404: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg470_1, torch.bfloat16);  arg470_1 = None
        convert_element_type_405: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_142, torch.bfloat16)
        view_372: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_405, _shape_param_424);  convert_element_type_405 = _shape_param_424 = None
        permute_201: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_404, [1, 0]);  convert_element_type_404 = None
        addmm_151: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_403, view_372, permute_201);  convert_element_type_403 = None
        view_373: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_151, _shape_param_425);  _shape_param_425 = None
        mul_121: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_373, arg471_1);  view_373 = arg471_1 = None
        add_143: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_121, arg472_1);  mul_121 = arg472_1 = None
        convert_element_type_406: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg473_1, torch.bfloat16);  arg473_1 = None
        convert_element_type_407: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg474_1, torch.bfloat16);  arg474_1 = None
        permute_202: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_407, [1, 0]);  convert_element_type_407 = None
        addmm_152: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_406, view_372, permute_202);  convert_element_type_406 = None
        view_374: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_152, _shape_param_426);  _shape_param_426 = None
        mul_122: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_374, arg475_1);  view_374 = arg475_1 = None
        add_144: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_122, arg476_1);  mul_122 = arg476_1 = None
        convert_element_type_408: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg477_1, torch.bfloat16);  arg477_1 = None
        convert_element_type_409: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg478_1, torch.bfloat16);  arg478_1 = None
        convert_element_type_410: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_144, torch.bfloat16);  add_144 = None
        view_375: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_410, _shape_param_427);  convert_element_type_410 = _shape_param_427 = None
        permute_203: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_409, [1, 0]);  convert_element_type_409 = None
        addmm_153: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_408, view_375, permute_203);  convert_element_type_408 = None
        view_376: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_153, _shape_param_428);  addmm_153 = _shape_param_428 = None
        view_377: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_376, _shape_param_429);  view_376 = _shape_param_429 = None
        permute_204: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_377, [0, 2, 1, 3]);  view_377 = None
        convert_element_type_411: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg479_1, torch.bfloat16);  arg479_1 = None
        convert_element_type_412: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg480_1, torch.bfloat16);  arg480_1 = None
        permute_205: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_412, [1, 0]);  convert_element_type_412 = None
        addmm_154: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_411, view_375, permute_205);  convert_element_type_411 = None
        view_378: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_154, _shape_param_430);  addmm_154 = _shape_param_430 = None
        view_379: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_378, _shape_param_431);  view_378 = _shape_param_431 = None
        permute_206: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_379, [0, 2, 1, 3]);  view_379 = None
        convert_element_type_413: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg481_1, torch.bfloat16);  arg481_1 = None
        convert_element_type_414: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg482_1, torch.bfloat16);  arg482_1 = None
        permute_207: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_414, [1, 0]);  convert_element_type_414 = None
        addmm_155: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_413, view_372, permute_207);  convert_element_type_413 = None
        view_380: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_155, _shape_param_432);  addmm_155 = _shape_param_432 = None
        view_381: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_380, _shape_param_433);  view_380 = _shape_param_433 = None
        permute_208: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_381, [0, 2, 1, 3]);  view_381 = None
        mul_123: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_204, 0.4204482076268573);  permute_204 = None
        permute_209: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_206, [0, 1, 3, 2]);  permute_206 = None
        mul_124: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_209, 0.4204482076268573);  permute_209 = None
        expand_40: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_123, _shape_param_434);  mul_123 = _shape_param_434 = None
        clone_40: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_382: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_40, _shape_param_435);  clone_40 = _shape_param_435 = None
        expand_41: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_124, _shape_param_436);  mul_124 = _shape_param_436 = None
        clone_41: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_383: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_437);  clone_41 = _shape_param_437 = None
        bmm_20: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_382, view_383)
        view_384: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_438);  bmm_20 = _shape_param_438 = None
        convert_element_type_415: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_384, torch.float32)
        amax_10: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_415, [-1], True)
        sub_10: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_415, amax_10);  convert_element_type_415 = amax_10 = None
        exp_10: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_11: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_416: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        eq_10: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_384, -inf);  view_384 = None
        logical_not_20: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        where_10: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_21, full_1, convert_element_type_416);  logical_not_21 = convert_element_type_416 = None
        inductor_lookup_seed_10: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 10)
        inductor_random_10: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_439, inductor_lookup_seed_10, 'rand');  _shape_param_439 = inductor_lookup_seed_10 = None
        convert_element_type_417: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_10, torch.bfloat16);  inductor_random_10 = None
        gt_10: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_417, 0.1);  convert_element_type_417 = None
        mul_125: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_10, where_10)
        mul_126: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_125, 1.1111111111111112);  mul_125 = None
        expand_42: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_126, _shape_param_440);  mul_126 = _shape_param_440 = None
        view_385: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_42, _shape_param_441);  expand_42 = _shape_param_441 = None
        expand_43: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_208, _shape_param_442);  permute_208 = _shape_param_442 = None
        clone_42: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_386: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_42, _shape_param_443);  clone_42 = _shape_param_443 = None
        bmm_21: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_385, view_386)
        view_387: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_21, _shape_param_444);  bmm_21 = _shape_param_444 = None
        permute_210: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_387, [0, 2, 1, 3]);  view_387 = None
        clone_43: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_210, memory_format = torch.contiguous_format);  permute_210 = None
        view_388: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_43, _shape_param_445);  clone_43 = _shape_param_445 = None
        convert_element_type_418: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg483_1, torch.bfloat16);  arg483_1 = None
        convert_element_type_419: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg484_1, torch.bfloat16);  arg484_1 = None
        view_389: "bf16[32768, 128]" = torch.ops.aten.view.default(view_388, _shape_param_446);  view_388 = _shape_param_446 = None
        permute_211: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_419, [1, 0]);  convert_element_type_419 = None
        addmm_156: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_418, view_389, permute_211);  convert_element_type_418 = None
        view_390: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_156, _shape_param_447);  _shape_param_447 = None
        add_145: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_390, add_143);  view_390 = add_143 = None
        mul_127: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, arg485_1);  add_145 = arg485_1 = None
        add_146: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, arg486_1);  mul_127 = arg486_1 = None
        convert_element_type_420: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg487_1, torch.bfloat16);  arg487_1 = None
        convert_element_type_421: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg488_1, torch.bfloat16);  arg488_1 = None
        convert_element_type_422: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_146, torch.bfloat16)
        view_391: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_422, _shape_param_448);  convert_element_type_422 = _shape_param_448 = None
        permute_212: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_421, [1, 0]);  convert_element_type_421 = None
        addmm_157: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_420, view_391, permute_212);  convert_element_type_420 = None
        view_392: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_157, _shape_param_449);  addmm_157 = _shape_param_449 = None
        relu_40: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_392);  view_392 = None
        convert_element_type_423: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg489_1, torch.bfloat16);  arg489_1 = None
        convert_element_type_424: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg490_1, torch.bfloat16);  arg490_1 = None
        view_393: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_40, _shape_param_450);  _shape_param_450 = None
        permute_213: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_424, [1, 0]);  convert_element_type_424 = None
        addmm_158: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_423, view_393, permute_213);  convert_element_type_423 = None
        view_394: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_158, _shape_param_451);  addmm_158 = _shape_param_451 = None
        add_147: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_394, add_146);  view_394 = add_146 = None
        mul_128: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, arg491_1);  arg491_1 = None
        add_148: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, arg492_1);  mul_128 = arg492_1 = None
        convert_element_type_425: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg493_1, torch.bfloat16);  arg493_1 = None
        convert_element_type_426: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg494_1, torch.bfloat16);  arg494_1 = None
        convert_element_type_427: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_148, torch.bfloat16)
        view_395: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_427, _shape_param_452);  convert_element_type_427 = _shape_param_452 = None
        permute_214: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_426, [1, 0]);  convert_element_type_426 = None
        addmm_159: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_425, view_395, permute_214);  convert_element_type_425 = None
        view_396: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_159, _shape_param_453);  addmm_159 = _shape_param_453 = None
        relu_41: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_396);  view_396 = None
        convert_element_type_428: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg495_1, torch.bfloat16);  arg495_1 = None
        convert_element_type_429: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg496_1, torch.bfloat16);  arg496_1 = None
        view_397: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_41, _shape_param_454);  _shape_param_454 = None
        permute_215: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_429, [1, 0]);  convert_element_type_429 = None
        addmm_160: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_428, view_397, permute_215);  convert_element_type_428 = None
        view_398: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_160, _shape_param_455);  addmm_160 = _shape_param_455 = None
        add_149: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_398, add_148);  view_398 = add_148 = None
        mul_129: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, arg497_1);  arg497_1 = None
        add_150: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_129, arg498_1);  mul_129 = arg498_1 = None
        convert_element_type_430: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg499_1, torch.bfloat16);  arg499_1 = None
        convert_element_type_431: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg500_1, torch.bfloat16);  arg500_1 = None
        convert_element_type_432: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_150, torch.bfloat16)
        view_399: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_432, _shape_param_456);  convert_element_type_432 = _shape_param_456 = None
        permute_216: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_431, [1, 0]);  convert_element_type_431 = None
        addmm_161: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_430, view_399, permute_216);  convert_element_type_430 = None
        view_400: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_161, _shape_param_457);  addmm_161 = _shape_param_457 = None
        relu_42: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_400);  view_400 = None
        convert_element_type_433: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg501_1, torch.bfloat16);  arg501_1 = None
        convert_element_type_434: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg502_1, torch.bfloat16);  arg502_1 = None
        view_401: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_42, _shape_param_458);  _shape_param_458 = None
        permute_217: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_434, [1, 0]);  convert_element_type_434 = None
        addmm_162: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_433, view_401, permute_217);  convert_element_type_433 = None
        view_402: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_162, _shape_param_459);  addmm_162 = _shape_param_459 = None
        add_151: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_402, add_150);  view_402 = add_150 = None
        mul_130: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_151, arg503_1);  arg503_1 = None
        add_152: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_130, arg504_1);  mul_130 = arg504_1 = None
        convert_element_type_435: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg505_1, torch.bfloat16);  arg505_1 = None
        convert_element_type_436: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg506_1, torch.bfloat16);  arg506_1 = None
        convert_element_type_437: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_152, torch.bfloat16)
        view_403: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_437, _shape_param_460);  convert_element_type_437 = _shape_param_460 = None
        permute_218: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_436, [1, 0]);  convert_element_type_436 = None
        addmm_163: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_435, view_403, permute_218);  convert_element_type_435 = None
        view_404: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_163, _shape_param_461);  addmm_163 = _shape_param_461 = None
        relu_43: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_404);  view_404 = None
        convert_element_type_438: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg507_1, torch.bfloat16);  arg507_1 = None
        convert_element_type_439: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg508_1, torch.bfloat16);  arg508_1 = None
        view_405: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_43, _shape_param_462);  _shape_param_462 = None
        permute_219: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_439, [1, 0]);  convert_element_type_439 = None
        addmm_164: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_438, view_405, permute_219);  convert_element_type_438 = None
        view_406: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_164, _shape_param_463);  addmm_164 = _shape_param_463 = None
        add_153: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_406, add_152);  view_406 = add_152 = None
        mul_131: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_153, arg509_1);  arg509_1 = None
        add_154: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, arg510_1);  mul_131 = arg510_1 = None
        convert_element_type_440: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg511_1, torch.bfloat16);  arg511_1 = None
        convert_element_type_441: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg512_1, torch.bfloat16);  arg512_1 = None
        convert_element_type_442: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_154, torch.bfloat16);  add_154 = None
        view_407: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_442, _shape_param_464);  convert_element_type_442 = _shape_param_464 = None
        permute_220: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_441, [1, 0]);  convert_element_type_441 = None
        addmm_165: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_440, view_407, permute_220);  convert_element_type_440 = None
        view_408: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_165, _shape_param_465);  addmm_165 = _shape_param_465 = None
        add_155: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_408, add_142);  view_408 = add_142 = None
        mul_132: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_155, arg513_1);  arg513_1 = None
        add_156: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_132, arg514_1);  mul_132 = arg514_1 = None
        convert_element_type_443: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg515_1, torch.bfloat16);  arg515_1 = None
        convert_element_type_444: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg516_1, torch.bfloat16);  arg516_1 = None
        convert_element_type_445: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_156, torch.bfloat16)
        view_409: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_445, _shape_param_466);  convert_element_type_445 = _shape_param_466 = None
        permute_221: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_444, [1, 0]);  convert_element_type_444 = None
        addmm_166: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_443, view_409, permute_221);  convert_element_type_443 = None
        view_410: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_166, _shape_param_467);  _shape_param_467 = None
        mul_133: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_410, arg517_1);  view_410 = arg517_1 = None
        add_157: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_133, arg518_1);  mul_133 = arg518_1 = None
        convert_element_type_446: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg519_1, torch.bfloat16);  arg519_1 = None
        convert_element_type_447: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg520_1, torch.bfloat16);  arg520_1 = None
        permute_222: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_447, [1, 0]);  convert_element_type_447 = None
        addmm_167: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_446, view_409, permute_222);  convert_element_type_446 = None
        view_411: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_167, _shape_param_468);  _shape_param_468 = None
        mul_134: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_411, arg521_1);  view_411 = arg521_1 = None
        add_158: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, arg522_1);  mul_134 = arg522_1 = None
        convert_element_type_448: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg523_1, torch.bfloat16);  arg523_1 = None
        convert_element_type_449: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg524_1, torch.bfloat16);  arg524_1 = None
        convert_element_type_450: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_158, torch.bfloat16);  add_158 = None
        view_412: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_450, _shape_param_469);  convert_element_type_450 = _shape_param_469 = None
        permute_223: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_449, [1, 0]);  convert_element_type_449 = None
        addmm_168: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_448, view_412, permute_223);  convert_element_type_448 = None
        view_413: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_168, _shape_param_470);  addmm_168 = _shape_param_470 = None
        view_414: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_413, _shape_param_471);  view_413 = _shape_param_471 = None
        permute_224: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_414, [0, 2, 1, 3]);  view_414 = None
        convert_element_type_451: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg525_1, torch.bfloat16);  arg525_1 = None
        convert_element_type_452: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg526_1, torch.bfloat16);  arg526_1 = None
        permute_225: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_452, [1, 0]);  convert_element_type_452 = None
        addmm_169: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_451, view_412, permute_225);  convert_element_type_451 = None
        view_415: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_169, _shape_param_472);  addmm_169 = _shape_param_472 = None
        view_416: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_415, _shape_param_473);  view_415 = _shape_param_473 = None
        permute_226: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        convert_element_type_453: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg527_1, torch.bfloat16);  arg527_1 = None
        convert_element_type_454: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg528_1, torch.bfloat16);  arg528_1 = None
        permute_227: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_454, [1, 0]);  convert_element_type_454 = None
        addmm_170: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_453, view_409, permute_227);  convert_element_type_453 = None
        view_417: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_170, _shape_param_474);  addmm_170 = _shape_param_474 = None
        view_418: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_417, _shape_param_475);  view_417 = _shape_param_475 = None
        permute_228: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
        mul_135: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_224, 0.4204482076268573);  permute_224 = None
        permute_229: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_226, [0, 1, 3, 2]);  permute_226 = None
        mul_136: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_229, 0.4204482076268573);  permute_229 = None
        expand_44: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_135, _shape_param_476);  mul_135 = _shape_param_476 = None
        clone_44: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_419: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_44, _shape_param_477);  clone_44 = _shape_param_477 = None
        expand_45: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_136, _shape_param_478);  mul_136 = _shape_param_478 = None
        clone_45: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_420: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_479);  clone_45 = _shape_param_479 = None
        bmm_22: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_419, view_420)
        view_421: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_480);  bmm_22 = _shape_param_480 = None
        convert_element_type_455: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_421, torch.float32)
        amax_11: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_455, [-1], True)
        sub_11: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_455, amax_11);  convert_element_type_455 = amax_11 = None
        exp_11: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_12: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_456: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        eq_11: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_421, -inf);  view_421 = None
        logical_not_22: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        where_11: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_23, full_1, convert_element_type_456);  logical_not_23 = convert_element_type_456 = None
        inductor_lookup_seed_11: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 11)
        inductor_random_11: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_481, inductor_lookup_seed_11, 'rand');  _shape_param_481 = inductor_lookup_seed_11 = None
        convert_element_type_457: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_11, torch.bfloat16);  inductor_random_11 = None
        gt_11: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_457, 0.1);  convert_element_type_457 = None
        mul_137: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_11, where_11)
        mul_138: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_137, 1.1111111111111112);  mul_137 = None
        expand_46: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_138, _shape_param_482);  mul_138 = _shape_param_482 = None
        view_422: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_46, _shape_param_483);  expand_46 = _shape_param_483 = None
        expand_47: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_228, _shape_param_484);  permute_228 = _shape_param_484 = None
        clone_46: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_423: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_46, _shape_param_485);  clone_46 = _shape_param_485 = None
        bmm_23: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_422, view_423)
        view_424: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_23, _shape_param_486);  bmm_23 = _shape_param_486 = None
        permute_230: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_424, [0, 2, 1, 3]);  view_424 = None
        clone_47: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_230, memory_format = torch.contiguous_format);  permute_230 = None
        view_425: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_47, _shape_param_487);  clone_47 = _shape_param_487 = None
        convert_element_type_458: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg529_1, torch.bfloat16);  arg529_1 = None
        convert_element_type_459: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg530_1, torch.bfloat16);  arg530_1 = None
        view_426: "bf16[32768, 128]" = torch.ops.aten.view.default(view_425, _shape_param_488);  view_425 = _shape_param_488 = None
        permute_231: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_459, [1, 0]);  convert_element_type_459 = None
        addmm_171: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_458, view_426, permute_231);  convert_element_type_458 = None
        view_427: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_171, _shape_param_489);  _shape_param_489 = None
        add_159: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_157);  view_427 = add_157 = None
        mul_139: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_159, arg531_1);  add_159 = arg531_1 = None
        add_160: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, arg532_1);  mul_139 = arg532_1 = None
        convert_element_type_460: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg533_1, torch.bfloat16);  arg533_1 = None
        convert_element_type_461: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg534_1, torch.bfloat16);  arg534_1 = None
        convert_element_type_462: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_160, torch.bfloat16)
        view_428: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_462, _shape_param_490);  convert_element_type_462 = _shape_param_490 = None
        permute_232: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_461, [1, 0]);  convert_element_type_461 = None
        addmm_172: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_460, view_428, permute_232);  convert_element_type_460 = None
        view_429: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_172, _shape_param_491);  addmm_172 = _shape_param_491 = None
        relu_44: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None
        convert_element_type_463: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg535_1, torch.bfloat16);  arg535_1 = None
        convert_element_type_464: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg536_1, torch.bfloat16);  arg536_1 = None
        view_430: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_44, _shape_param_492);  _shape_param_492 = None
        permute_233: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_464, [1, 0]);  convert_element_type_464 = None
        addmm_173: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_463, view_430, permute_233);  convert_element_type_463 = None
        view_431: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_173, _shape_param_493);  addmm_173 = _shape_param_493 = None
        add_161: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_160);  view_431 = add_160 = None
        mul_140: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_161, arg537_1);  arg537_1 = None
        add_162: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_140, arg538_1);  mul_140 = arg538_1 = None
        convert_element_type_465: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg539_1, torch.bfloat16);  arg539_1 = None
        convert_element_type_466: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg540_1, torch.bfloat16);  arg540_1 = None
        convert_element_type_467: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_162, torch.bfloat16)
        view_432: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_467, _shape_param_494);  convert_element_type_467 = _shape_param_494 = None
        permute_234: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_466, [1, 0]);  convert_element_type_466 = None
        addmm_174: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_465, view_432, permute_234);  convert_element_type_465 = None
        view_433: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_174, _shape_param_495);  addmm_174 = _shape_param_495 = None
        relu_45: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None
        convert_element_type_468: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg541_1, torch.bfloat16);  arg541_1 = None
        convert_element_type_469: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg542_1, torch.bfloat16);  arg542_1 = None
        view_434: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_45, _shape_param_496);  _shape_param_496 = None
        permute_235: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_469, [1, 0]);  convert_element_type_469 = None
        addmm_175: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_468, view_434, permute_235);  convert_element_type_468 = None
        view_435: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_175, _shape_param_497);  addmm_175 = _shape_param_497 = None
        add_163: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_162);  view_435 = add_162 = None
        mul_141: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_163, arg543_1);  arg543_1 = None
        add_164: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, arg544_1);  mul_141 = arg544_1 = None
        convert_element_type_470: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg545_1, torch.bfloat16);  arg545_1 = None
        convert_element_type_471: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg546_1, torch.bfloat16);  arg546_1 = None
        convert_element_type_472: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_164, torch.bfloat16)
        view_436: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_472, _shape_param_498);  convert_element_type_472 = _shape_param_498 = None
        permute_236: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_471, [1, 0]);  convert_element_type_471 = None
        addmm_176: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_470, view_436, permute_236);  convert_element_type_470 = None
        view_437: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_176, _shape_param_499);  addmm_176 = _shape_param_499 = None
        relu_46: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None
        convert_element_type_473: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg547_1, torch.bfloat16);  arg547_1 = None
        convert_element_type_474: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg548_1, torch.bfloat16);  arg548_1 = None
        view_438: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_46, _shape_param_500);  _shape_param_500 = None
        permute_237: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_474, [1, 0]);  convert_element_type_474 = None
        addmm_177: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_473, view_438, permute_237);  convert_element_type_473 = None
        view_439: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_177, _shape_param_501);  addmm_177 = _shape_param_501 = None
        add_165: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_164);  view_439 = add_164 = None
        mul_142: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_165, arg549_1);  arg549_1 = None
        add_166: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_142, arg550_1);  mul_142 = arg550_1 = None
        convert_element_type_475: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg551_1, torch.bfloat16);  arg551_1 = None
        convert_element_type_476: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg552_1, torch.bfloat16);  arg552_1 = None
        convert_element_type_477: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_166, torch.bfloat16)
        view_440: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_477, _shape_param_502);  convert_element_type_477 = _shape_param_502 = None
        permute_238: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_476, [1, 0]);  convert_element_type_476 = None
        addmm_178: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_475, view_440, permute_238);  convert_element_type_475 = None
        view_441: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_178, _shape_param_503);  addmm_178 = _shape_param_503 = None
        relu_47: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_441);  view_441 = None
        convert_element_type_478: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg553_1, torch.bfloat16);  arg553_1 = None
        convert_element_type_479: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg554_1, torch.bfloat16);  arg554_1 = None
        view_442: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_47, _shape_param_504);  _shape_param_504 = None
        permute_239: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_479, [1, 0]);  convert_element_type_479 = None
        addmm_179: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_478, view_442, permute_239);  convert_element_type_478 = None
        view_443: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_179, _shape_param_505);  addmm_179 = _shape_param_505 = None
        add_167: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_443, add_166);  view_443 = add_166 = None
        mul_143: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_167, arg555_1);  arg555_1 = None
        add_168: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_143, arg556_1);  mul_143 = arg556_1 = None
        convert_element_type_480: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg557_1, torch.bfloat16);  arg557_1 = None
        convert_element_type_481: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg558_1, torch.bfloat16);  arg558_1 = None
        convert_element_type_482: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_168, torch.bfloat16);  add_168 = None
        view_444: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_482, _shape_param_506);  convert_element_type_482 = _shape_param_506 = None
        permute_240: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_481, [1, 0]);  convert_element_type_481 = None
        addmm_180: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_480, view_444, permute_240);  convert_element_type_480 = None
        view_445: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_180, _shape_param_507);  addmm_180 = _shape_param_507 = None
        add_169: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_445, add_156);  view_445 = add_156 = None
        mul_144: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_169, arg559_1);  arg559_1 = None
        add_170: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_144, arg560_1);  mul_144 = arg560_1 = None
        convert_element_type_483: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg561_1, torch.bfloat16);  arg561_1 = None
        convert_element_type_484: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg562_1, torch.bfloat16);  arg562_1 = None
        convert_element_type_485: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_170, torch.bfloat16)
        view_446: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_485, _shape_param_508);  convert_element_type_485 = _shape_param_508 = None
        permute_241: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_484, [1, 0]);  convert_element_type_484 = None
        addmm_181: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_483, view_446, permute_241);  convert_element_type_483 = None
        view_447: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_181, _shape_param_509);  _shape_param_509 = None
        mul_145: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_447, arg563_1);  view_447 = arg563_1 = None
        add_171: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_145, arg564_1);  mul_145 = arg564_1 = None
        convert_element_type_486: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg565_1, torch.bfloat16);  arg565_1 = None
        convert_element_type_487: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg566_1, torch.bfloat16);  arg566_1 = None
        permute_242: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_487, [1, 0]);  convert_element_type_487 = None
        addmm_182: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_486, view_446, permute_242);  convert_element_type_486 = None
        view_448: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_182, _shape_param_510);  _shape_param_510 = None
        mul_146: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_448, arg567_1);  view_448 = arg567_1 = None
        add_172: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_146, arg568_1);  mul_146 = arg568_1 = None
        convert_element_type_488: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg569_1, torch.bfloat16);  arg569_1 = None
        convert_element_type_489: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg570_1, torch.bfloat16);  arg570_1 = None
        convert_element_type_490: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_172, torch.bfloat16);  add_172 = None
        view_449: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_490, _shape_param_511);  convert_element_type_490 = _shape_param_511 = None
        permute_243: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_489, [1, 0]);  convert_element_type_489 = None
        addmm_183: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_488, view_449, permute_243);  convert_element_type_488 = None
        view_450: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_183, _shape_param_512);  addmm_183 = _shape_param_512 = None
        view_451: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_450, _shape_param_513);  view_450 = _shape_param_513 = None
        permute_244: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_451, [0, 2, 1, 3]);  view_451 = None
        convert_element_type_491: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg571_1, torch.bfloat16);  arg571_1 = None
        convert_element_type_492: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg572_1, torch.bfloat16);  arg572_1 = None
        permute_245: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_492, [1, 0]);  convert_element_type_492 = None
        addmm_184: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_491, view_449, permute_245);  convert_element_type_491 = None
        view_452: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_184, _shape_param_514);  addmm_184 = _shape_param_514 = None
        view_453: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_452, _shape_param_515);  view_452 = _shape_param_515 = None
        permute_246: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_453, [0, 2, 1, 3]);  view_453 = None
        convert_element_type_493: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg573_1, torch.bfloat16);  arg573_1 = None
        convert_element_type_494: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg574_1, torch.bfloat16);  arg574_1 = None
        permute_247: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_494, [1, 0]);  convert_element_type_494 = None
        addmm_185: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_493, view_446, permute_247);  convert_element_type_493 = None
        view_454: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_185, _shape_param_516);  addmm_185 = _shape_param_516 = None
        view_455: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_454, _shape_param_517);  view_454 = _shape_param_517 = None
        permute_248: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_455, [0, 2, 1, 3]);  view_455 = None
        mul_147: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_244, 0.4204482076268573);  permute_244 = None
        permute_249: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_246, [0, 1, 3, 2]);  permute_246 = None
        mul_148: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_249, 0.4204482076268573);  permute_249 = None
        expand_48: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_147, _shape_param_518);  mul_147 = _shape_param_518 = None
        clone_48: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_456: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_48, _shape_param_519);  clone_48 = _shape_param_519 = None
        expand_49: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_148, _shape_param_520);  mul_148 = _shape_param_520 = None
        clone_49: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_457: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_521);  clone_49 = _shape_param_521 = None
        bmm_24: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_456, view_457)
        view_458: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_522);  bmm_24 = _shape_param_522 = None
        convert_element_type_495: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_458, torch.float32)
        amax_12: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_495, [-1], True)
        sub_12: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_495, amax_12);  convert_element_type_495 = amax_12 = None
        exp_12: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_13: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_496: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        eq_12: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_458, -inf);  view_458 = None
        logical_not_24: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        where_12: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_25, full_1, convert_element_type_496);  logical_not_25 = convert_element_type_496 = None
        inductor_lookup_seed_12: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 12)
        inductor_random_12: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_523, inductor_lookup_seed_12, 'rand');  _shape_param_523 = inductor_lookup_seed_12 = None
        convert_element_type_497: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_12, torch.bfloat16);  inductor_random_12 = None
        gt_12: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_497, 0.1);  convert_element_type_497 = None
        mul_149: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_12, where_12)
        mul_150: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_149, 1.1111111111111112);  mul_149 = None
        expand_50: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_150, _shape_param_524);  mul_150 = _shape_param_524 = None
        view_459: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_50, _shape_param_525);  expand_50 = _shape_param_525 = None
        expand_51: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_248, _shape_param_526);  permute_248 = _shape_param_526 = None
        clone_50: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_460: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_50, _shape_param_527);  clone_50 = _shape_param_527 = None
        bmm_25: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_459, view_460)
        view_461: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_25, _shape_param_528);  bmm_25 = _shape_param_528 = None
        permute_250: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_461, [0, 2, 1, 3]);  view_461 = None
        clone_51: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_250, memory_format = torch.contiguous_format);  permute_250 = None
        view_462: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_51, _shape_param_529);  clone_51 = _shape_param_529 = None
        convert_element_type_498: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg575_1, torch.bfloat16);  arg575_1 = None
        convert_element_type_499: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg576_1, torch.bfloat16);  arg576_1 = None
        view_463: "bf16[32768, 128]" = torch.ops.aten.view.default(view_462, _shape_param_530);  view_462 = _shape_param_530 = None
        permute_251: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_499, [1, 0]);  convert_element_type_499 = None
        addmm_186: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_498, view_463, permute_251);  convert_element_type_498 = None
        view_464: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_186, _shape_param_531);  _shape_param_531 = None
        add_173: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_464, add_171);  view_464 = add_171 = None
        mul_151: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, arg577_1);  add_173 = arg577_1 = None
        add_174: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, arg578_1);  mul_151 = arg578_1 = None
        convert_element_type_500: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg579_1, torch.bfloat16);  arg579_1 = None
        convert_element_type_501: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg580_1, torch.bfloat16);  arg580_1 = None
        convert_element_type_502: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_174, torch.bfloat16)
        view_465: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_502, _shape_param_532);  convert_element_type_502 = _shape_param_532 = None
        permute_252: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_501, [1, 0]);  convert_element_type_501 = None
        addmm_187: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_500, view_465, permute_252);  convert_element_type_500 = None
        view_466: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_187, _shape_param_533);  addmm_187 = _shape_param_533 = None
        relu_48: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_466);  view_466 = None
        convert_element_type_503: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg581_1, torch.bfloat16);  arg581_1 = None
        convert_element_type_504: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg582_1, torch.bfloat16);  arg582_1 = None
        view_467: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_48, _shape_param_534);  _shape_param_534 = None
        permute_253: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_504, [1, 0]);  convert_element_type_504 = None
        addmm_188: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_503, view_467, permute_253);  convert_element_type_503 = None
        view_468: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_188, _shape_param_535);  addmm_188 = _shape_param_535 = None
        add_175: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_468, add_174);  view_468 = add_174 = None
        mul_152: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, arg583_1);  arg583_1 = None
        add_176: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_152, arg584_1);  mul_152 = arg584_1 = None
        convert_element_type_505: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg585_1, torch.bfloat16);  arg585_1 = None
        convert_element_type_506: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg586_1, torch.bfloat16);  arg586_1 = None
        convert_element_type_507: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_176, torch.bfloat16)
        view_469: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_507, _shape_param_536);  convert_element_type_507 = _shape_param_536 = None
        permute_254: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_506, [1, 0]);  convert_element_type_506 = None
        addmm_189: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_505, view_469, permute_254);  convert_element_type_505 = None
        view_470: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_189, _shape_param_537);  addmm_189 = _shape_param_537 = None
        relu_49: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_470);  view_470 = None
        convert_element_type_508: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg587_1, torch.bfloat16);  arg587_1 = None
        convert_element_type_509: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg588_1, torch.bfloat16);  arg588_1 = None
        view_471: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_49, _shape_param_538);  _shape_param_538 = None
        permute_255: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_509, [1, 0]);  convert_element_type_509 = None
        addmm_190: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_508, view_471, permute_255);  convert_element_type_508 = None
        view_472: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_190, _shape_param_539);  addmm_190 = _shape_param_539 = None
        add_177: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_472, add_176);  view_472 = add_176 = None
        mul_153: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, arg589_1);  arg589_1 = None
        add_178: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_153, arg590_1);  mul_153 = arg590_1 = None
        convert_element_type_510: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg591_1, torch.bfloat16);  arg591_1 = None
        convert_element_type_511: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg592_1, torch.bfloat16);  arg592_1 = None
        convert_element_type_512: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_178, torch.bfloat16)
        view_473: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_512, _shape_param_540);  convert_element_type_512 = _shape_param_540 = None
        permute_256: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_511, [1, 0]);  convert_element_type_511 = None
        addmm_191: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_510, view_473, permute_256);  convert_element_type_510 = None
        view_474: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_191, _shape_param_541);  addmm_191 = _shape_param_541 = None
        relu_50: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_474);  view_474 = None
        convert_element_type_513: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg593_1, torch.bfloat16);  arg593_1 = None
        convert_element_type_514: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg594_1, torch.bfloat16);  arg594_1 = None
        view_475: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_50, _shape_param_542);  _shape_param_542 = None
        permute_257: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_514, [1, 0]);  convert_element_type_514 = None
        addmm_192: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_513, view_475, permute_257);  convert_element_type_513 = None
        view_476: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_192, _shape_param_543);  addmm_192 = _shape_param_543 = None
        add_179: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_476, add_178);  view_476 = add_178 = None
        mul_154: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, arg595_1);  arg595_1 = None
        add_180: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_154, arg596_1);  mul_154 = arg596_1 = None
        convert_element_type_515: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg597_1, torch.bfloat16);  arg597_1 = None
        convert_element_type_516: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg598_1, torch.bfloat16);  arg598_1 = None
        convert_element_type_517: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_180, torch.bfloat16)
        view_477: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_517, _shape_param_544);  convert_element_type_517 = _shape_param_544 = None
        permute_258: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_516, [1, 0]);  convert_element_type_516 = None
        addmm_193: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_515, view_477, permute_258);  convert_element_type_515 = None
        view_478: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_193, _shape_param_545);  addmm_193 = _shape_param_545 = None
        relu_51: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_478);  view_478 = None
        convert_element_type_518: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg599_1, torch.bfloat16);  arg599_1 = None
        convert_element_type_519: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg600_1, torch.bfloat16);  arg600_1 = None
        view_479: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_51, _shape_param_546);  _shape_param_546 = None
        permute_259: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_519, [1, 0]);  convert_element_type_519 = None
        addmm_194: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_518, view_479, permute_259);  convert_element_type_518 = None
        view_480: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_194, _shape_param_547);  addmm_194 = _shape_param_547 = None
        add_181: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_480, add_180);  view_480 = add_180 = None
        mul_155: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_181, arg601_1);  arg601_1 = None
        add_182: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, arg602_1);  mul_155 = arg602_1 = None
        convert_element_type_520: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg603_1, torch.bfloat16);  arg603_1 = None
        convert_element_type_521: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg604_1, torch.bfloat16);  arg604_1 = None
        convert_element_type_522: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_182, torch.bfloat16);  add_182 = None
        view_481: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_522, _shape_param_548);  convert_element_type_522 = _shape_param_548 = None
        permute_260: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_521, [1, 0]);  convert_element_type_521 = None
        addmm_195: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_520, view_481, permute_260);  convert_element_type_520 = None
        view_482: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_195, _shape_param_549);  addmm_195 = _shape_param_549 = None
        add_183: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_482, add_170);  view_482 = add_170 = None
        mul_156: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_183, arg605_1);  arg605_1 = None
        add_184: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_156, arg606_1);  mul_156 = arg606_1 = None
        convert_element_type_523: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg607_1, torch.bfloat16);  arg607_1 = None
        convert_element_type_524: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg608_1, torch.bfloat16);  arg608_1 = None
        convert_element_type_525: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_184, torch.bfloat16)
        view_483: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_525, _shape_param_550);  convert_element_type_525 = _shape_param_550 = None
        permute_261: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_524, [1, 0]);  convert_element_type_524 = None
        addmm_196: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_523, view_483, permute_261);  convert_element_type_523 = None
        view_484: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_196, _shape_param_551);  _shape_param_551 = None
        mul_157: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_484, arg609_1);  view_484 = arg609_1 = None
        add_185: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, arg610_1);  mul_157 = arg610_1 = None
        convert_element_type_526: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg611_1, torch.bfloat16);  arg611_1 = None
        convert_element_type_527: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg612_1, torch.bfloat16);  arg612_1 = None
        permute_262: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_527, [1, 0]);  convert_element_type_527 = None
        addmm_197: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_526, view_483, permute_262);  convert_element_type_526 = None
        view_485: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_197, _shape_param_552);  _shape_param_552 = None
        mul_158: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_485, arg613_1);  view_485 = arg613_1 = None
        add_186: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, arg614_1);  mul_158 = arg614_1 = None
        convert_element_type_528: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg615_1, torch.bfloat16);  arg615_1 = None
        convert_element_type_529: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg616_1, torch.bfloat16);  arg616_1 = None
        convert_element_type_530: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_186, torch.bfloat16);  add_186 = None
        view_486: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_530, _shape_param_553);  convert_element_type_530 = _shape_param_553 = None
        permute_263: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_529, [1, 0]);  convert_element_type_529 = None
        addmm_198: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_528, view_486, permute_263);  convert_element_type_528 = None
        view_487: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_198, _shape_param_554);  addmm_198 = _shape_param_554 = None
        view_488: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_487, _shape_param_555);  view_487 = _shape_param_555 = None
        permute_264: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None
        convert_element_type_531: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg617_1, torch.bfloat16);  arg617_1 = None
        convert_element_type_532: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg618_1, torch.bfloat16);  arg618_1 = None
        permute_265: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_532, [1, 0]);  convert_element_type_532 = None
        addmm_199: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_531, view_486, permute_265);  convert_element_type_531 = None
        view_489: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_199, _shape_param_556);  addmm_199 = _shape_param_556 = None
        view_490: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_489, _shape_param_557);  view_489 = _shape_param_557 = None
        permute_266: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        convert_element_type_533: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg619_1, torch.bfloat16);  arg619_1 = None
        convert_element_type_534: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg620_1, torch.bfloat16);  arg620_1 = None
        permute_267: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_534, [1, 0]);  convert_element_type_534 = None
        addmm_200: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_533, view_483, permute_267);  convert_element_type_533 = None
        view_491: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_200, _shape_param_558);  addmm_200 = _shape_param_558 = None
        view_492: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_491, _shape_param_559);  view_491 = _shape_param_559 = None
        permute_268: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_492, [0, 2, 1, 3]);  view_492 = None
        mul_159: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_264, 0.4204482076268573);  permute_264 = None
        permute_269: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_266, [0, 1, 3, 2]);  permute_266 = None
        mul_160: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_269, 0.4204482076268573);  permute_269 = None
        expand_52: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_159, _shape_param_560);  mul_159 = _shape_param_560 = None
        clone_52: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_493: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_52, _shape_param_561);  clone_52 = _shape_param_561 = None
        expand_53: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_160, _shape_param_562);  mul_160 = _shape_param_562 = None
        clone_53: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_494: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_563);  clone_53 = _shape_param_563 = None
        bmm_26: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_493, view_494)
        view_495: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_564);  bmm_26 = _shape_param_564 = None
        convert_element_type_535: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_495, torch.float32)
        amax_13: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_535, [-1], True)
        sub_13: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_535, amax_13);  convert_element_type_535 = amax_13 = None
        exp_13: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_14: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_536: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        eq_13: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_495, -inf);  view_495 = None
        logical_not_26: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        where_13: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_27, full_1, convert_element_type_536);  logical_not_27 = convert_element_type_536 = None
        inductor_lookup_seed_13: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 13)
        inductor_random_13: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_565, inductor_lookup_seed_13, 'rand');  _shape_param_565 = inductor_lookup_seed_13 = None
        convert_element_type_537: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_13, torch.bfloat16);  inductor_random_13 = None
        gt_13: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_537, 0.1);  convert_element_type_537 = None
        mul_161: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_13, where_13)
        mul_162: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_161, 1.1111111111111112);  mul_161 = None
        expand_54: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_162, _shape_param_566);  mul_162 = _shape_param_566 = None
        view_496: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_54, _shape_param_567);  expand_54 = _shape_param_567 = None
        expand_55: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_268, _shape_param_568);  permute_268 = _shape_param_568 = None
        clone_54: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_497: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_54, _shape_param_569);  clone_54 = _shape_param_569 = None
        bmm_27: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_496, view_497)
        view_498: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_27, _shape_param_570);  bmm_27 = _shape_param_570 = None
        permute_270: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_55: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_270, memory_format = torch.contiguous_format);  permute_270 = None
        view_499: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_55, _shape_param_571);  clone_55 = _shape_param_571 = None
        convert_element_type_538: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg621_1, torch.bfloat16);  arg621_1 = None
        convert_element_type_539: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg622_1, torch.bfloat16);  arg622_1 = None
        view_500: "bf16[32768, 128]" = torch.ops.aten.view.default(view_499, _shape_param_572);  view_499 = _shape_param_572 = None
        permute_271: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_539, [1, 0]);  convert_element_type_539 = None
        addmm_201: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_538, view_500, permute_271);  convert_element_type_538 = None
        view_501: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_201, _shape_param_573);  _shape_param_573 = None
        add_187: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_501, add_185);  view_501 = add_185 = None
        mul_163: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_187, arg623_1);  add_187 = arg623_1 = None
        add_188: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_163, arg624_1);  mul_163 = arg624_1 = None
        convert_element_type_540: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg625_1, torch.bfloat16);  arg625_1 = None
        convert_element_type_541: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg626_1, torch.bfloat16);  arg626_1 = None
        convert_element_type_542: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_188, torch.bfloat16)
        view_502: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_542, _shape_param_574);  convert_element_type_542 = _shape_param_574 = None
        permute_272: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_541, [1, 0]);  convert_element_type_541 = None
        addmm_202: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_540, view_502, permute_272);  convert_element_type_540 = None
        view_503: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_202, _shape_param_575);  addmm_202 = _shape_param_575 = None
        relu_52: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_503);  view_503 = None
        convert_element_type_543: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg627_1, torch.bfloat16);  arg627_1 = None
        convert_element_type_544: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg628_1, torch.bfloat16);  arg628_1 = None
        view_504: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_52, _shape_param_576);  _shape_param_576 = None
        permute_273: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_544, [1, 0]);  convert_element_type_544 = None
        addmm_203: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_543, view_504, permute_273);  convert_element_type_543 = None
        view_505: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_203, _shape_param_577);  addmm_203 = _shape_param_577 = None
        add_189: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_505, add_188);  view_505 = add_188 = None
        mul_164: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_189, arg629_1);  arg629_1 = None
        add_190: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_164, arg630_1);  mul_164 = arg630_1 = None
        convert_element_type_545: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg631_1, torch.bfloat16);  arg631_1 = None
        convert_element_type_546: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg632_1, torch.bfloat16);  arg632_1 = None
        convert_element_type_547: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_190, torch.bfloat16)
        view_506: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_547, _shape_param_578);  convert_element_type_547 = _shape_param_578 = None
        permute_274: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_546, [1, 0]);  convert_element_type_546 = None
        addmm_204: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_545, view_506, permute_274);  convert_element_type_545 = None
        view_507: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_204, _shape_param_579);  addmm_204 = _shape_param_579 = None
        relu_53: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_507);  view_507 = None
        convert_element_type_548: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg633_1, torch.bfloat16);  arg633_1 = None
        convert_element_type_549: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg634_1, torch.bfloat16);  arg634_1 = None
        view_508: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_53, _shape_param_580);  _shape_param_580 = None
        permute_275: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_549, [1, 0]);  convert_element_type_549 = None
        addmm_205: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_548, view_508, permute_275);  convert_element_type_548 = None
        view_509: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_205, _shape_param_581);  addmm_205 = _shape_param_581 = None
        add_191: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_509, add_190);  view_509 = add_190 = None
        mul_165: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_191, arg635_1);  arg635_1 = None
        add_192: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, arg636_1);  mul_165 = arg636_1 = None
        convert_element_type_550: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg637_1, torch.bfloat16);  arg637_1 = None
        convert_element_type_551: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg638_1, torch.bfloat16);  arg638_1 = None
        convert_element_type_552: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_192, torch.bfloat16)
        view_510: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_552, _shape_param_582);  convert_element_type_552 = _shape_param_582 = None
        permute_276: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_551, [1, 0]);  convert_element_type_551 = None
        addmm_206: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_550, view_510, permute_276);  convert_element_type_550 = None
        view_511: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_206, _shape_param_583);  addmm_206 = _shape_param_583 = None
        relu_54: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_511);  view_511 = None
        convert_element_type_553: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg639_1, torch.bfloat16);  arg639_1 = None
        convert_element_type_554: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg640_1, torch.bfloat16);  arg640_1 = None
        view_512: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_54, _shape_param_584);  _shape_param_584 = None
        permute_277: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_554, [1, 0]);  convert_element_type_554 = None
        addmm_207: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_553, view_512, permute_277);  convert_element_type_553 = None
        view_513: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_207, _shape_param_585);  addmm_207 = _shape_param_585 = None
        add_193: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_513, add_192);  view_513 = add_192 = None
        mul_166: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_193, arg641_1);  arg641_1 = None
        add_194: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, arg642_1);  mul_166 = arg642_1 = None
        convert_element_type_555: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg643_1, torch.bfloat16);  arg643_1 = None
        convert_element_type_556: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg644_1, torch.bfloat16);  arg644_1 = None
        convert_element_type_557: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_194, torch.bfloat16)
        view_514: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_557, _shape_param_586);  convert_element_type_557 = _shape_param_586 = None
        permute_278: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_556, [1, 0]);  convert_element_type_556 = None
        addmm_208: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_555, view_514, permute_278);  convert_element_type_555 = None
        view_515: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_208, _shape_param_587);  addmm_208 = _shape_param_587 = None
        relu_55: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_515);  view_515 = None
        convert_element_type_558: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg645_1, torch.bfloat16);  arg645_1 = None
        convert_element_type_559: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg646_1, torch.bfloat16);  arg646_1 = None
        view_516: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_55, _shape_param_588);  _shape_param_588 = None
        permute_279: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_559, [1, 0]);  convert_element_type_559 = None
        addmm_209: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_558, view_516, permute_279);  convert_element_type_558 = None
        view_517: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_209, _shape_param_589);  addmm_209 = _shape_param_589 = None
        add_195: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_517, add_194);  view_517 = add_194 = None
        mul_167: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_195, arg647_1);  arg647_1 = None
        add_196: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, arg648_1);  mul_167 = arg648_1 = None
        convert_element_type_560: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg649_1, torch.bfloat16);  arg649_1 = None
        convert_element_type_561: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg650_1, torch.bfloat16);  arg650_1 = None
        convert_element_type_562: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_196, torch.bfloat16);  add_196 = None
        view_518: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_562, _shape_param_590);  convert_element_type_562 = _shape_param_590 = None
        permute_280: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_561, [1, 0]);  convert_element_type_561 = None
        addmm_210: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_560, view_518, permute_280);  convert_element_type_560 = None
        view_519: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_210, _shape_param_591);  addmm_210 = _shape_param_591 = None
        add_197: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_519, add_184);  view_519 = add_184 = None
        mul_168: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_197, arg651_1);  arg651_1 = None
        add_198: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_168, arg652_1);  mul_168 = arg652_1 = None
        convert_element_type_563: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg653_1, torch.bfloat16);  arg653_1 = None
        convert_element_type_564: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg654_1, torch.bfloat16);  arg654_1 = None
        convert_element_type_565: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_198, torch.bfloat16)
        view_520: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_565, _shape_param_592);  convert_element_type_565 = _shape_param_592 = None
        permute_281: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_564, [1, 0]);  convert_element_type_564 = None
        addmm_211: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_563, view_520, permute_281);  convert_element_type_563 = None
        view_521: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_211, _shape_param_593);  _shape_param_593 = None
        mul_169: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_521, arg655_1);  view_521 = arg655_1 = None
        add_199: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_169, arg656_1);  mul_169 = arg656_1 = None
        convert_element_type_566: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg657_1, torch.bfloat16);  arg657_1 = None
        convert_element_type_567: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg658_1, torch.bfloat16);  arg658_1 = None
        permute_282: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_567, [1, 0]);  convert_element_type_567 = None
        addmm_212: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_566, view_520, permute_282);  convert_element_type_566 = None
        view_522: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_212, _shape_param_594);  _shape_param_594 = None
        mul_170: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_522, arg659_1);  view_522 = arg659_1 = None
        add_200: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_170, arg660_1);  mul_170 = arg660_1 = None
        convert_element_type_568: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg661_1, torch.bfloat16);  arg661_1 = None
        convert_element_type_569: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg662_1, torch.bfloat16);  arg662_1 = None
        convert_element_type_570: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_200, torch.bfloat16);  add_200 = None
        view_523: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_570, _shape_param_595);  convert_element_type_570 = _shape_param_595 = None
        permute_283: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_569, [1, 0]);  convert_element_type_569 = None
        addmm_213: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_568, view_523, permute_283);  convert_element_type_568 = None
        view_524: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_213, _shape_param_596);  addmm_213 = _shape_param_596 = None
        view_525: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_524, _shape_param_597);  view_524 = _shape_param_597 = None
        permute_284: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_525, [0, 2, 1, 3]);  view_525 = None
        convert_element_type_571: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg663_1, torch.bfloat16);  arg663_1 = None
        convert_element_type_572: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg664_1, torch.bfloat16);  arg664_1 = None
        permute_285: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_572, [1, 0]);  convert_element_type_572 = None
        addmm_214: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_571, view_523, permute_285);  convert_element_type_571 = None
        view_526: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_214, _shape_param_598);  addmm_214 = _shape_param_598 = None
        view_527: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_526, _shape_param_599);  view_526 = _shape_param_599 = None
        permute_286: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None
        convert_element_type_573: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg665_1, torch.bfloat16);  arg665_1 = None
        convert_element_type_574: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg666_1, torch.bfloat16);  arg666_1 = None
        permute_287: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_574, [1, 0]);  convert_element_type_574 = None
        addmm_215: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_573, view_520, permute_287);  convert_element_type_573 = None
        view_528: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_215, _shape_param_600);  addmm_215 = _shape_param_600 = None
        view_529: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_528, _shape_param_601);  view_528 = _shape_param_601 = None
        permute_288: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_529, [0, 2, 1, 3]);  view_529 = None
        mul_171: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_284, 0.4204482076268573);  permute_284 = None
        permute_289: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_286, [0, 1, 3, 2]);  permute_286 = None
        mul_172: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_289, 0.4204482076268573);  permute_289 = None
        expand_56: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_171, _shape_param_602);  mul_171 = _shape_param_602 = None
        clone_56: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_530: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_56, _shape_param_603);  clone_56 = _shape_param_603 = None
        expand_57: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_172, _shape_param_604);  mul_172 = _shape_param_604 = None
        clone_57: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_531: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_605);  clone_57 = _shape_param_605 = None
        bmm_28: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_530, view_531)
        view_532: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_606);  bmm_28 = _shape_param_606 = None
        convert_element_type_575: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32)
        amax_14: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_575, [-1], True)
        sub_14: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_575, amax_14);  convert_element_type_575 = amax_14 = None
        exp_14: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_15: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_576: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        eq_14: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_532, -inf);  view_532 = None
        logical_not_28: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        where_14: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_29, full_1, convert_element_type_576);  logical_not_29 = convert_element_type_576 = None
        inductor_lookup_seed_14: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 14)
        inductor_random_14: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_607, inductor_lookup_seed_14, 'rand');  _shape_param_607 = inductor_lookup_seed_14 = None
        convert_element_type_577: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_14, torch.bfloat16);  inductor_random_14 = None
        gt_14: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_577, 0.1);  convert_element_type_577 = None
        mul_173: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_14, where_14)
        mul_174: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_173, 1.1111111111111112);  mul_173 = None
        expand_58: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_174, _shape_param_608);  mul_174 = _shape_param_608 = None
        view_533: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_58, _shape_param_609);  expand_58 = _shape_param_609 = None
        expand_59: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_288, _shape_param_610);  permute_288 = _shape_param_610 = None
        clone_58: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_534: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_58, _shape_param_611);  clone_58 = _shape_param_611 = None
        bmm_29: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_533, view_534)
        view_535: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_29, _shape_param_612);  bmm_29 = _shape_param_612 = None
        permute_290: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_535, [0, 2, 1, 3]);  view_535 = None
        clone_59: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_290, memory_format = torch.contiguous_format);  permute_290 = None
        view_536: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_59, _shape_param_613);  clone_59 = _shape_param_613 = None
        convert_element_type_578: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg667_1, torch.bfloat16);  arg667_1 = None
        convert_element_type_579: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg668_1, torch.bfloat16);  arg668_1 = None
        view_537: "bf16[32768, 128]" = torch.ops.aten.view.default(view_536, _shape_param_614);  view_536 = _shape_param_614 = None
        permute_291: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_579, [1, 0]);  convert_element_type_579 = None
        addmm_216: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_578, view_537, permute_291);  convert_element_type_578 = None
        view_538: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_216, _shape_param_615);  _shape_param_615 = None
        add_201: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_538, add_199);  view_538 = add_199 = None
        mul_175: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_201, arg669_1);  add_201 = arg669_1 = None
        add_202: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, arg670_1);  mul_175 = arg670_1 = None
        convert_element_type_580: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg671_1, torch.bfloat16);  arg671_1 = None
        convert_element_type_581: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg672_1, torch.bfloat16);  arg672_1 = None
        convert_element_type_582: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_202, torch.bfloat16)
        view_539: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_582, _shape_param_616);  convert_element_type_582 = _shape_param_616 = None
        permute_292: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_581, [1, 0]);  convert_element_type_581 = None
        addmm_217: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_580, view_539, permute_292);  convert_element_type_580 = None
        view_540: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_217, _shape_param_617);  addmm_217 = _shape_param_617 = None
        relu_56: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_540);  view_540 = None
        convert_element_type_583: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg673_1, torch.bfloat16);  arg673_1 = None
        convert_element_type_584: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg674_1, torch.bfloat16);  arg674_1 = None
        view_541: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_56, _shape_param_618);  _shape_param_618 = None
        permute_293: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_584, [1, 0]);  convert_element_type_584 = None
        addmm_218: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_583, view_541, permute_293);  convert_element_type_583 = None
        view_542: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_218, _shape_param_619);  addmm_218 = _shape_param_619 = None
        add_203: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_542, add_202);  view_542 = add_202 = None
        mul_176: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, arg675_1);  arg675_1 = None
        add_204: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, arg676_1);  mul_176 = arg676_1 = None
        convert_element_type_585: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg677_1, torch.bfloat16);  arg677_1 = None
        convert_element_type_586: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg678_1, torch.bfloat16);  arg678_1 = None
        convert_element_type_587: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_204, torch.bfloat16)
        view_543: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_587, _shape_param_620);  convert_element_type_587 = _shape_param_620 = None
        permute_294: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_586, [1, 0]);  convert_element_type_586 = None
        addmm_219: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_585, view_543, permute_294);  convert_element_type_585 = None
        view_544: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_219, _shape_param_621);  addmm_219 = _shape_param_621 = None
        relu_57: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_544);  view_544 = None
        convert_element_type_588: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg679_1, torch.bfloat16);  arg679_1 = None
        convert_element_type_589: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg680_1, torch.bfloat16);  arg680_1 = None
        view_545: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_57, _shape_param_622);  _shape_param_622 = None
        permute_295: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_589, [1, 0]);  convert_element_type_589 = None
        addmm_220: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_588, view_545, permute_295);  convert_element_type_588 = None
        view_546: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_220, _shape_param_623);  addmm_220 = _shape_param_623 = None
        add_205: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_546, add_204);  view_546 = add_204 = None
        mul_177: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, arg681_1);  arg681_1 = None
        add_206: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_177, arg682_1);  mul_177 = arg682_1 = None
        convert_element_type_590: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg683_1, torch.bfloat16);  arg683_1 = None
        convert_element_type_591: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg684_1, torch.bfloat16);  arg684_1 = None
        convert_element_type_592: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_206, torch.bfloat16)
        view_547: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_592, _shape_param_624);  convert_element_type_592 = _shape_param_624 = None
        permute_296: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_591, [1, 0]);  convert_element_type_591 = None
        addmm_221: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_590, view_547, permute_296);  convert_element_type_590 = None
        view_548: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_221, _shape_param_625);  addmm_221 = _shape_param_625 = None
        relu_58: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_548);  view_548 = None
        convert_element_type_593: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg685_1, torch.bfloat16);  arg685_1 = None
        convert_element_type_594: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg686_1, torch.bfloat16);  arg686_1 = None
        view_549: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_58, _shape_param_626);  _shape_param_626 = None
        permute_297: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_594, [1, 0]);  convert_element_type_594 = None
        addmm_222: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_593, view_549, permute_297);  convert_element_type_593 = None
        view_550: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_222, _shape_param_627);  addmm_222 = _shape_param_627 = None
        add_207: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_550, add_206);  view_550 = add_206 = None
        mul_178: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, arg687_1);  arg687_1 = None
        add_208: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, arg688_1);  mul_178 = arg688_1 = None
        convert_element_type_595: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg689_1, torch.bfloat16);  arg689_1 = None
        convert_element_type_596: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg690_1, torch.bfloat16);  arg690_1 = None
        convert_element_type_597: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_208, torch.bfloat16)
        view_551: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_597, _shape_param_628);  convert_element_type_597 = _shape_param_628 = None
        permute_298: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_596, [1, 0]);  convert_element_type_596 = None
        addmm_223: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_595, view_551, permute_298);  convert_element_type_595 = None
        view_552: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_223, _shape_param_629);  addmm_223 = _shape_param_629 = None
        relu_59: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_552);  view_552 = None
        convert_element_type_598: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg691_1, torch.bfloat16);  arg691_1 = None
        convert_element_type_599: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg692_1, torch.bfloat16);  arg692_1 = None
        view_553: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_59, _shape_param_630);  _shape_param_630 = None
        permute_299: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_599, [1, 0]);  convert_element_type_599 = None
        addmm_224: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_598, view_553, permute_299);  convert_element_type_598 = None
        view_554: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_224, _shape_param_631);  addmm_224 = _shape_param_631 = None
        add_209: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_554, add_208);  view_554 = add_208 = None
        mul_179: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, arg693_1);  arg693_1 = None
        add_210: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, arg694_1);  mul_179 = arg694_1 = None
        convert_element_type_600: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg695_1, torch.bfloat16);  arg695_1 = None
        convert_element_type_601: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg696_1, torch.bfloat16);  arg696_1 = None
        convert_element_type_602: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_210, torch.bfloat16);  add_210 = None
        view_555: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_602, _shape_param_632);  convert_element_type_602 = _shape_param_632 = None
        permute_300: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_601, [1, 0]);  convert_element_type_601 = None
        addmm_225: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_600, view_555, permute_300);  convert_element_type_600 = None
        view_556: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_225, _shape_param_633);  addmm_225 = _shape_param_633 = None
        add_211: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_556, add_198);  view_556 = add_198 = None
        mul_180: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_211, arg697_1);  arg697_1 = None
        add_212: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_180, arg698_1);  mul_180 = arg698_1 = None
        convert_element_type_603: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg699_1, torch.bfloat16);  arg699_1 = None
        convert_element_type_604: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg700_1, torch.bfloat16);  arg700_1 = None
        convert_element_type_605: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_212, torch.bfloat16)
        view_557: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_605, _shape_param_634);  convert_element_type_605 = _shape_param_634 = None
        permute_301: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_604, [1, 0]);  convert_element_type_604 = None
        addmm_226: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_603, view_557, permute_301);  convert_element_type_603 = None
        view_558: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_226, _shape_param_635);  _shape_param_635 = None
        mul_181: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_558, arg701_1);  view_558 = arg701_1 = None
        add_213: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, arg702_1);  mul_181 = arg702_1 = None
        convert_element_type_606: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg703_1, torch.bfloat16);  arg703_1 = None
        convert_element_type_607: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg704_1, torch.bfloat16);  arg704_1 = None
        permute_302: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_607, [1, 0]);  convert_element_type_607 = None
        addmm_227: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_606, view_557, permute_302);  convert_element_type_606 = None
        view_559: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_227, _shape_param_636);  _shape_param_636 = None
        mul_182: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_559, arg705_1);  view_559 = arg705_1 = None
        add_214: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_182, arg706_1);  mul_182 = arg706_1 = None
        convert_element_type_608: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg707_1, torch.bfloat16);  arg707_1 = None
        convert_element_type_609: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg708_1, torch.bfloat16);  arg708_1 = None
        convert_element_type_610: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_214, torch.bfloat16);  add_214 = None
        view_560: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_610, _shape_param_637);  convert_element_type_610 = _shape_param_637 = None
        permute_303: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_609, [1, 0]);  convert_element_type_609 = None
        addmm_228: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_608, view_560, permute_303);  convert_element_type_608 = None
        view_561: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_228, _shape_param_638);  addmm_228 = _shape_param_638 = None
        view_562: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_561, _shape_param_639);  view_561 = _shape_param_639 = None
        permute_304: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None
        convert_element_type_611: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg709_1, torch.bfloat16);  arg709_1 = None
        convert_element_type_612: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg710_1, torch.bfloat16);  arg710_1 = None
        permute_305: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_612, [1, 0]);  convert_element_type_612 = None
        addmm_229: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_611, view_560, permute_305);  convert_element_type_611 = None
        view_563: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_229, _shape_param_640);  addmm_229 = _shape_param_640 = None
        view_564: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_563, _shape_param_641);  view_563 = _shape_param_641 = None
        permute_306: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_564, [0, 2, 1, 3]);  view_564 = None
        convert_element_type_613: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg711_1, torch.bfloat16);  arg711_1 = None
        convert_element_type_614: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg712_1, torch.bfloat16);  arg712_1 = None
        permute_307: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_614, [1, 0]);  convert_element_type_614 = None
        addmm_230: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_613, view_557, permute_307);  convert_element_type_613 = None
        view_565: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_230, _shape_param_642);  addmm_230 = _shape_param_642 = None
        view_566: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_565, _shape_param_643);  view_565 = _shape_param_643 = None
        permute_308: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
        mul_183: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_304, 0.4204482076268573);  permute_304 = None
        permute_309: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_306, [0, 1, 3, 2]);  permute_306 = None
        mul_184: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_309, 0.4204482076268573);  permute_309 = None
        expand_60: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_183, _shape_param_644);  mul_183 = _shape_param_644 = None
        clone_60: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_567: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_60, _shape_param_645);  clone_60 = _shape_param_645 = None
        expand_61: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_184, _shape_param_646);  mul_184 = _shape_param_646 = None
        clone_61: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_568: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_647);  clone_61 = _shape_param_647 = None
        bmm_30: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_567, view_568)
        view_569: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_648);  bmm_30 = _shape_param_648 = None
        convert_element_type_615: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_569, torch.float32)
        amax_15: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_615, [-1], True)
        sub_15: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_615, amax_15);  convert_element_type_615 = amax_15 = None
        exp_15: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_16: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_616: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        eq_15: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_569, -inf);  view_569 = None
        logical_not_30: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        where_15: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_31, full_1, convert_element_type_616);  logical_not_31 = convert_element_type_616 = None
        inductor_lookup_seed_15: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 15)
        inductor_random_15: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_649, inductor_lookup_seed_15, 'rand');  _shape_param_649 = inductor_lookup_seed_15 = None
        convert_element_type_617: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_15, torch.bfloat16);  inductor_random_15 = None
        gt_15: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_617, 0.1);  convert_element_type_617 = None
        mul_185: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_15, where_15)
        mul_186: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_185, 1.1111111111111112);  mul_185 = None
        expand_62: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_186, _shape_param_650);  mul_186 = _shape_param_650 = None
        view_570: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_62, _shape_param_651);  expand_62 = _shape_param_651 = None
        expand_63: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_308, _shape_param_652);  permute_308 = _shape_param_652 = None
        clone_62: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_571: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_62, _shape_param_653);  clone_62 = _shape_param_653 = None
        bmm_31: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_570, view_571)
        view_572: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_31, _shape_param_654);  bmm_31 = _shape_param_654 = None
        permute_310: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_572, [0, 2, 1, 3]);  view_572 = None
        clone_63: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_310, memory_format = torch.contiguous_format);  permute_310 = None
        view_573: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_63, _shape_param_655);  clone_63 = _shape_param_655 = None
        convert_element_type_618: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg713_1, torch.bfloat16);  arg713_1 = None
        convert_element_type_619: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg714_1, torch.bfloat16);  arg714_1 = None
        view_574: "bf16[32768, 128]" = torch.ops.aten.view.default(view_573, _shape_param_656);  view_573 = _shape_param_656 = None
        permute_311: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_619, [1, 0]);  convert_element_type_619 = None
        addmm_231: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_618, view_574, permute_311);  convert_element_type_618 = None
        view_575: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_231, _shape_param_657);  _shape_param_657 = None
        add_215: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_575, add_213);  view_575 = add_213 = None
        mul_187: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_215, arg715_1);  add_215 = arg715_1 = None
        add_216: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, arg716_1);  mul_187 = arg716_1 = None
        convert_element_type_620: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg717_1, torch.bfloat16);  arg717_1 = None
        convert_element_type_621: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg718_1, torch.bfloat16);  arg718_1 = None
        convert_element_type_622: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_216, torch.bfloat16)
        view_576: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_622, _shape_param_658);  convert_element_type_622 = _shape_param_658 = None
        permute_312: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_621, [1, 0]);  convert_element_type_621 = None
        addmm_232: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_620, view_576, permute_312);  convert_element_type_620 = None
        view_577: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_232, _shape_param_659);  addmm_232 = _shape_param_659 = None
        relu_60: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_577);  view_577 = None
        convert_element_type_623: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg719_1, torch.bfloat16);  arg719_1 = None
        convert_element_type_624: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg720_1, torch.bfloat16);  arg720_1 = None
        view_578: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_60, _shape_param_660);  _shape_param_660 = None
        permute_313: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_624, [1, 0]);  convert_element_type_624 = None
        addmm_233: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_623, view_578, permute_313);  convert_element_type_623 = None
        view_579: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_233, _shape_param_661);  addmm_233 = _shape_param_661 = None
        add_217: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_579, add_216);  view_579 = add_216 = None
        mul_188: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_217, arg721_1);  arg721_1 = None
        add_218: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_188, arg722_1);  mul_188 = arg722_1 = None
        convert_element_type_625: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg723_1, torch.bfloat16);  arg723_1 = None
        convert_element_type_626: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg724_1, torch.bfloat16);  arg724_1 = None
        convert_element_type_627: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_218, torch.bfloat16)
        view_580: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_627, _shape_param_662);  convert_element_type_627 = _shape_param_662 = None
        permute_314: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_626, [1, 0]);  convert_element_type_626 = None
        addmm_234: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_625, view_580, permute_314);  convert_element_type_625 = None
        view_581: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_234, _shape_param_663);  addmm_234 = _shape_param_663 = None
        relu_61: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_581);  view_581 = None
        convert_element_type_628: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg725_1, torch.bfloat16);  arg725_1 = None
        convert_element_type_629: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg726_1, torch.bfloat16);  arg726_1 = None
        view_582: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_61, _shape_param_664);  _shape_param_664 = None
        permute_315: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_629, [1, 0]);  convert_element_type_629 = None
        addmm_235: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_628, view_582, permute_315);  convert_element_type_628 = None
        view_583: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_235, _shape_param_665);  addmm_235 = _shape_param_665 = None
        add_219: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_583, add_218);  view_583 = add_218 = None
        mul_189: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_219, arg727_1);  arg727_1 = None
        add_220: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, arg728_1);  mul_189 = arg728_1 = None
        convert_element_type_630: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg729_1, torch.bfloat16);  arg729_1 = None
        convert_element_type_631: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg730_1, torch.bfloat16);  arg730_1 = None
        convert_element_type_632: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_220, torch.bfloat16)
        view_584: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_632, _shape_param_666);  convert_element_type_632 = _shape_param_666 = None
        permute_316: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_631, [1, 0]);  convert_element_type_631 = None
        addmm_236: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_630, view_584, permute_316);  convert_element_type_630 = None
        view_585: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_236, _shape_param_667);  addmm_236 = _shape_param_667 = None
        relu_62: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None
        convert_element_type_633: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg731_1, torch.bfloat16);  arg731_1 = None
        convert_element_type_634: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg732_1, torch.bfloat16);  arg732_1 = None
        view_586: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_62, _shape_param_668);  _shape_param_668 = None
        permute_317: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_634, [1, 0]);  convert_element_type_634 = None
        addmm_237: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_633, view_586, permute_317);  convert_element_type_633 = None
        view_587: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_237, _shape_param_669);  addmm_237 = _shape_param_669 = None
        add_221: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_220);  view_587 = add_220 = None
        mul_190: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_221, arg733_1);  arg733_1 = None
        add_222: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_190, arg734_1);  mul_190 = arg734_1 = None
        convert_element_type_635: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg735_1, torch.bfloat16);  arg735_1 = None
        convert_element_type_636: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg736_1, torch.bfloat16);  arg736_1 = None
        convert_element_type_637: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_222, torch.bfloat16)
        view_588: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_637, _shape_param_670);  convert_element_type_637 = _shape_param_670 = None
        permute_318: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_636, [1, 0]);  convert_element_type_636 = None
        addmm_238: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_635, view_588, permute_318);  convert_element_type_635 = None
        view_589: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_238, _shape_param_671);  addmm_238 = _shape_param_671 = None
        relu_63: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None
        convert_element_type_638: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg737_1, torch.bfloat16);  arg737_1 = None
        convert_element_type_639: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg738_1, torch.bfloat16);  arg738_1 = None
        view_590: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_63, _shape_param_672);  _shape_param_672 = None
        permute_319: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_639, [1, 0]);  convert_element_type_639 = None
        addmm_239: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_638, view_590, permute_319);  convert_element_type_638 = None
        view_591: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_239, _shape_param_673);  addmm_239 = _shape_param_673 = None
        add_223: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_222);  view_591 = add_222 = None
        mul_191: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_223, arg739_1);  arg739_1 = None
        add_224: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, arg740_1);  mul_191 = arg740_1 = None
        convert_element_type_640: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg741_1, torch.bfloat16);  arg741_1 = None
        convert_element_type_641: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg742_1, torch.bfloat16);  arg742_1 = None
        convert_element_type_642: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_224, torch.bfloat16);  add_224 = None
        view_592: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_642, _shape_param_674);  convert_element_type_642 = _shape_param_674 = None
        permute_320: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_641, [1, 0]);  convert_element_type_641 = None
        addmm_240: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_640, view_592, permute_320);  convert_element_type_640 = None
        view_593: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_240, _shape_param_675);  addmm_240 = _shape_param_675 = None
        add_225: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_593, add_212);  view_593 = add_212 = None
        mul_192: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_225, arg743_1);  arg743_1 = None
        add_226: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_192, arg744_1);  mul_192 = arg744_1 = None
        convert_element_type_643: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg745_1, torch.bfloat16);  arg745_1 = None
        convert_element_type_644: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg746_1, torch.bfloat16);  arg746_1 = None
        convert_element_type_645: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_226, torch.bfloat16)
        view_594: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_645, _shape_param_676);  convert_element_type_645 = _shape_param_676 = None
        permute_321: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_644, [1, 0]);  convert_element_type_644 = None
        addmm_241: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_643, view_594, permute_321);  convert_element_type_643 = None
        view_595: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_241, _shape_param_677);  _shape_param_677 = None
        mul_193: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_595, arg747_1);  view_595 = arg747_1 = None
        add_227: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_193, arg748_1);  mul_193 = arg748_1 = None
        convert_element_type_646: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg749_1, torch.bfloat16);  arg749_1 = None
        convert_element_type_647: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg750_1, torch.bfloat16);  arg750_1 = None
        permute_322: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_647, [1, 0]);  convert_element_type_647 = None
        addmm_242: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_646, view_594, permute_322);  convert_element_type_646 = None
        view_596: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_242, _shape_param_678);  _shape_param_678 = None
        mul_194: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_596, arg751_1);  view_596 = arg751_1 = None
        add_228: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_194, arg752_1);  mul_194 = arg752_1 = None
        convert_element_type_648: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg753_1, torch.bfloat16);  arg753_1 = None
        convert_element_type_649: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg754_1, torch.bfloat16);  arg754_1 = None
        convert_element_type_650: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_228, torch.bfloat16);  add_228 = None
        view_597: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_650, _shape_param_679);  convert_element_type_650 = _shape_param_679 = None
        permute_323: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_649, [1, 0]);  convert_element_type_649 = None
        addmm_243: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_648, view_597, permute_323);  convert_element_type_648 = None
        view_598: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_243, _shape_param_680);  addmm_243 = _shape_param_680 = None
        view_599: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_598, _shape_param_681);  view_598 = _shape_param_681 = None
        permute_324: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_599, [0, 2, 1, 3]);  view_599 = None
        convert_element_type_651: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg755_1, torch.bfloat16);  arg755_1 = None
        convert_element_type_652: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg756_1, torch.bfloat16);  arg756_1 = None
        permute_325: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_652, [1, 0]);  convert_element_type_652 = None
        addmm_244: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_651, view_597, permute_325);  convert_element_type_651 = None
        view_600: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_244, _shape_param_682);  addmm_244 = _shape_param_682 = None
        view_601: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_600, _shape_param_683);  view_600 = _shape_param_683 = None
        permute_326: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_601, [0, 2, 1, 3]);  view_601 = None
        convert_element_type_653: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg757_1, torch.bfloat16);  arg757_1 = None
        convert_element_type_654: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg758_1, torch.bfloat16);  arg758_1 = None
        permute_327: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_654, [1, 0]);  convert_element_type_654 = None
        addmm_245: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_653, view_594, permute_327);  convert_element_type_653 = None
        view_602: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_245, _shape_param_684);  addmm_245 = _shape_param_684 = None
        view_603: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_602, _shape_param_685);  view_602 = _shape_param_685 = None
        permute_328: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_603, [0, 2, 1, 3]);  view_603 = None
        mul_195: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_324, 0.4204482076268573);  permute_324 = None
        permute_329: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_326, [0, 1, 3, 2]);  permute_326 = None
        mul_196: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_329, 0.4204482076268573);  permute_329 = None
        expand_64: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_195, _shape_param_686);  mul_195 = _shape_param_686 = None
        clone_64: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_604: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_64, _shape_param_687);  clone_64 = _shape_param_687 = None
        expand_65: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_196, _shape_param_688);  mul_196 = _shape_param_688 = None
        clone_65: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_605: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_689);  clone_65 = _shape_param_689 = None
        bmm_32: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_604, view_605)
        view_606: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_690);  bmm_32 = _shape_param_690 = None
        convert_element_type_655: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_606, torch.float32)
        amax_16: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_655, [-1], True)
        sub_16: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_655, amax_16);  convert_element_type_655 = amax_16 = None
        exp_16: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_17: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_656: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        eq_16: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_606, -inf);  view_606 = None
        logical_not_32: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        where_16: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_33, full_1, convert_element_type_656);  logical_not_33 = convert_element_type_656 = None
        inductor_lookup_seed_16: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 16)
        inductor_random_16: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_691, inductor_lookup_seed_16, 'rand');  _shape_param_691 = inductor_lookup_seed_16 = None
        convert_element_type_657: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_16, torch.bfloat16);  inductor_random_16 = None
        gt_16: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_657, 0.1);  convert_element_type_657 = None
        mul_197: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_16, where_16)
        mul_198: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_197, 1.1111111111111112);  mul_197 = None
        expand_66: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_198, _shape_param_692);  mul_198 = _shape_param_692 = None
        view_607: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_66, _shape_param_693);  expand_66 = _shape_param_693 = None
        expand_67: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_328, _shape_param_694);  permute_328 = _shape_param_694 = None
        clone_66: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_608: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_66, _shape_param_695);  clone_66 = _shape_param_695 = None
        bmm_33: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_607, view_608)
        view_609: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_33, _shape_param_696);  bmm_33 = _shape_param_696 = None
        permute_330: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_609, [0, 2, 1, 3]);  view_609 = None
        clone_67: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_330, memory_format = torch.contiguous_format);  permute_330 = None
        view_610: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_67, _shape_param_697);  clone_67 = _shape_param_697 = None
        convert_element_type_658: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg759_1, torch.bfloat16);  arg759_1 = None
        convert_element_type_659: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg760_1, torch.bfloat16);  arg760_1 = None
        view_611: "bf16[32768, 128]" = torch.ops.aten.view.default(view_610, _shape_param_698);  view_610 = _shape_param_698 = None
        permute_331: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_659, [1, 0]);  convert_element_type_659 = None
        addmm_246: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_658, view_611, permute_331);  convert_element_type_658 = None
        view_612: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_246, _shape_param_699);  _shape_param_699 = None
        add_229: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_612, add_227);  view_612 = add_227 = None
        mul_199: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_229, arg761_1);  add_229 = arg761_1 = None
        add_230: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_199, arg762_1);  mul_199 = arg762_1 = None
        convert_element_type_660: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg763_1, torch.bfloat16);  arg763_1 = None
        convert_element_type_661: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg764_1, torch.bfloat16);  arg764_1 = None
        convert_element_type_662: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_230, torch.bfloat16)
        view_613: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_662, _shape_param_700);  convert_element_type_662 = _shape_param_700 = None
        permute_332: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_661, [1, 0]);  convert_element_type_661 = None
        addmm_247: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_660, view_613, permute_332);  convert_element_type_660 = None
        view_614: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_247, _shape_param_701);  addmm_247 = _shape_param_701 = None
        relu_64: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_614);  view_614 = None
        convert_element_type_663: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg765_1, torch.bfloat16);  arg765_1 = None
        convert_element_type_664: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg766_1, torch.bfloat16);  arg766_1 = None
        view_615: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_64, _shape_param_702);  _shape_param_702 = None
        permute_333: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_664, [1, 0]);  convert_element_type_664 = None
        addmm_248: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_663, view_615, permute_333);  convert_element_type_663 = None
        view_616: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_248, _shape_param_703);  addmm_248 = _shape_param_703 = None
        add_231: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_616, add_230);  view_616 = add_230 = None
        mul_200: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_231, arg767_1);  arg767_1 = None
        add_232: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_200, arg768_1);  mul_200 = arg768_1 = None
        convert_element_type_665: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg769_1, torch.bfloat16);  arg769_1 = None
        convert_element_type_666: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg770_1, torch.bfloat16);  arg770_1 = None
        convert_element_type_667: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_232, torch.bfloat16)
        view_617: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_667, _shape_param_704);  convert_element_type_667 = _shape_param_704 = None
        permute_334: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_666, [1, 0]);  convert_element_type_666 = None
        addmm_249: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_665, view_617, permute_334);  convert_element_type_665 = None
        view_618: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_249, _shape_param_705);  addmm_249 = _shape_param_705 = None
        relu_65: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_618);  view_618 = None
        convert_element_type_668: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg771_1, torch.bfloat16);  arg771_1 = None
        convert_element_type_669: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg772_1, torch.bfloat16);  arg772_1 = None
        view_619: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_65, _shape_param_706);  _shape_param_706 = None
        permute_335: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_669, [1, 0]);  convert_element_type_669 = None
        addmm_250: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_668, view_619, permute_335);  convert_element_type_668 = None
        view_620: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_250, _shape_param_707);  addmm_250 = _shape_param_707 = None
        add_233: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_620, add_232);  view_620 = add_232 = None
        mul_201: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, arg773_1);  arg773_1 = None
        add_234: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_201, arg774_1);  mul_201 = arg774_1 = None
        convert_element_type_670: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg775_1, torch.bfloat16);  arg775_1 = None
        convert_element_type_671: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg776_1, torch.bfloat16);  arg776_1 = None
        convert_element_type_672: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_234, torch.bfloat16)
        view_621: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_672, _shape_param_708);  convert_element_type_672 = _shape_param_708 = None
        permute_336: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_671, [1, 0]);  convert_element_type_671 = None
        addmm_251: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_670, view_621, permute_336);  convert_element_type_670 = None
        view_622: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_251, _shape_param_709);  addmm_251 = _shape_param_709 = None
        relu_66: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_622);  view_622 = None
        convert_element_type_673: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg777_1, torch.bfloat16);  arg777_1 = None
        convert_element_type_674: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg778_1, torch.bfloat16);  arg778_1 = None
        view_623: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_66, _shape_param_710);  _shape_param_710 = None
        permute_337: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_674, [1, 0]);  convert_element_type_674 = None
        addmm_252: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_673, view_623, permute_337);  convert_element_type_673 = None
        view_624: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_252, _shape_param_711);  addmm_252 = _shape_param_711 = None
        add_235: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_624, add_234);  view_624 = add_234 = None
        mul_202: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, arg779_1);  arg779_1 = None
        add_236: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_202, arg780_1);  mul_202 = arg780_1 = None
        convert_element_type_675: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg781_1, torch.bfloat16);  arg781_1 = None
        convert_element_type_676: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg782_1, torch.bfloat16);  arg782_1 = None
        convert_element_type_677: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_236, torch.bfloat16)
        view_625: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_677, _shape_param_712);  convert_element_type_677 = _shape_param_712 = None
        permute_338: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_676, [1, 0]);  convert_element_type_676 = None
        addmm_253: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_675, view_625, permute_338);  convert_element_type_675 = None
        view_626: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_253, _shape_param_713);  addmm_253 = _shape_param_713 = None
        relu_67: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_626);  view_626 = None
        convert_element_type_678: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg783_1, torch.bfloat16);  arg783_1 = None
        convert_element_type_679: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg784_1, torch.bfloat16);  arg784_1 = None
        view_627: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_67, _shape_param_714);  _shape_param_714 = None
        permute_339: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_679, [1, 0]);  convert_element_type_679 = None
        addmm_254: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_678, view_627, permute_339);  convert_element_type_678 = None
        view_628: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_254, _shape_param_715);  addmm_254 = _shape_param_715 = None
        add_237: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_628, add_236);  view_628 = add_236 = None
        mul_203: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, arg785_1);  arg785_1 = None
        add_238: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_203, arg786_1);  mul_203 = arg786_1 = None
        convert_element_type_680: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg787_1, torch.bfloat16);  arg787_1 = None
        convert_element_type_681: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg788_1, torch.bfloat16);  arg788_1 = None
        convert_element_type_682: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_238, torch.bfloat16);  add_238 = None
        view_629: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_682, _shape_param_716);  convert_element_type_682 = _shape_param_716 = None
        permute_340: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_681, [1, 0]);  convert_element_type_681 = None
        addmm_255: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_680, view_629, permute_340);  convert_element_type_680 = None
        view_630: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_255, _shape_param_717);  addmm_255 = _shape_param_717 = None
        add_239: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_630, add_226);  view_630 = add_226 = None
        mul_204: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_239, arg789_1);  arg789_1 = None
        add_240: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_204, arg790_1);  mul_204 = arg790_1 = None
        convert_element_type_683: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg791_1, torch.bfloat16);  arg791_1 = None
        convert_element_type_684: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg792_1, torch.bfloat16);  arg792_1 = None
        convert_element_type_685: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_240, torch.bfloat16)
        view_631: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_685, _shape_param_718);  convert_element_type_685 = _shape_param_718 = None
        permute_341: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_684, [1, 0]);  convert_element_type_684 = None
        addmm_256: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_683, view_631, permute_341);  convert_element_type_683 = None
        view_632: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_256, _shape_param_719);  _shape_param_719 = None
        mul_205: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_632, arg793_1);  view_632 = arg793_1 = None
        add_241: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_205, arg794_1);  mul_205 = arg794_1 = None
        convert_element_type_686: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg795_1, torch.bfloat16);  arg795_1 = None
        convert_element_type_687: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg796_1, torch.bfloat16);  arg796_1 = None
        permute_342: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_687, [1, 0]);  convert_element_type_687 = None
        addmm_257: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_686, view_631, permute_342);  convert_element_type_686 = None
        view_633: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_257, _shape_param_720);  _shape_param_720 = None
        mul_206: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_633, arg797_1);  view_633 = arg797_1 = None
        add_242: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_206, arg798_1);  mul_206 = arg798_1 = None
        convert_element_type_688: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg799_1, torch.bfloat16);  arg799_1 = None
        convert_element_type_689: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg800_1, torch.bfloat16);  arg800_1 = None
        convert_element_type_690: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_242, torch.bfloat16);  add_242 = None
        view_634: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_690, _shape_param_721);  convert_element_type_690 = _shape_param_721 = None
        permute_343: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_689, [1, 0]);  convert_element_type_689 = None
        addmm_258: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_688, view_634, permute_343);  convert_element_type_688 = None
        view_635: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_258, _shape_param_722);  addmm_258 = _shape_param_722 = None
        view_636: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_635, _shape_param_723);  view_635 = _shape_param_723 = None
        permute_344: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None
        convert_element_type_691: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg801_1, torch.bfloat16);  arg801_1 = None
        convert_element_type_692: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg802_1, torch.bfloat16);  arg802_1 = None
        permute_345: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_692, [1, 0]);  convert_element_type_692 = None
        addmm_259: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_691, view_634, permute_345);  convert_element_type_691 = None
        view_637: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_259, _shape_param_724);  addmm_259 = _shape_param_724 = None
        view_638: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_637, _shape_param_725);  view_637 = _shape_param_725 = None
        permute_346: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_638, [0, 2, 1, 3]);  view_638 = None
        convert_element_type_693: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg803_1, torch.bfloat16);  arg803_1 = None
        convert_element_type_694: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg804_1, torch.bfloat16);  arg804_1 = None
        permute_347: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_694, [1, 0]);  convert_element_type_694 = None
        addmm_260: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_693, view_631, permute_347);  convert_element_type_693 = None
        view_639: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_260, _shape_param_726);  addmm_260 = _shape_param_726 = None
        view_640: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_639, _shape_param_727);  view_639 = _shape_param_727 = None
        permute_348: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_640, [0, 2, 1, 3]);  view_640 = None
        mul_207: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_344, 0.4204482076268573);  permute_344 = None
        permute_349: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_346, [0, 1, 3, 2]);  permute_346 = None
        mul_208: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_349, 0.4204482076268573);  permute_349 = None
        expand_68: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_207, _shape_param_728);  mul_207 = _shape_param_728 = None
        clone_68: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_641: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_68, _shape_param_729);  clone_68 = _shape_param_729 = None
        expand_69: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_208, _shape_param_730);  mul_208 = _shape_param_730 = None
        clone_69: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_642: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_731);  clone_69 = _shape_param_731 = None
        bmm_34: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_641, view_642)
        view_643: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_732);  bmm_34 = _shape_param_732 = None
        convert_element_type_695: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_643, torch.float32)
        amax_17: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_695, [-1], True)
        sub_17: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_695, amax_17);  convert_element_type_695 = amax_17 = None
        exp_17: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_18: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_696: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        eq_17: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_643, -inf);  view_643 = None
        logical_not_34: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        where_17: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_35, full_1, convert_element_type_696);  logical_not_35 = convert_element_type_696 = None
        inductor_lookup_seed_17: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 17)
        inductor_random_17: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_733, inductor_lookup_seed_17, 'rand');  _shape_param_733 = inductor_lookup_seed_17 = None
        convert_element_type_697: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_17, torch.bfloat16);  inductor_random_17 = None
        gt_17: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_697, 0.1);  convert_element_type_697 = None
        mul_209: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_17, where_17)
        mul_210: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_209, 1.1111111111111112);  mul_209 = None
        expand_70: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_210, _shape_param_734);  mul_210 = _shape_param_734 = None
        view_644: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_70, _shape_param_735);  expand_70 = _shape_param_735 = None
        expand_71: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_348, _shape_param_736);  permute_348 = _shape_param_736 = None
        clone_70: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_645: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_70, _shape_param_737);  clone_70 = _shape_param_737 = None
        bmm_35: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_644, view_645)
        view_646: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_35, _shape_param_738);  bmm_35 = _shape_param_738 = None
        permute_350: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None
        clone_71: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_350, memory_format = torch.contiguous_format);  permute_350 = None
        view_647: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_71, _shape_param_739);  clone_71 = _shape_param_739 = None
        convert_element_type_698: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg805_1, torch.bfloat16);  arg805_1 = None
        convert_element_type_699: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg806_1, torch.bfloat16);  arg806_1 = None
        view_648: "bf16[32768, 128]" = torch.ops.aten.view.default(view_647, _shape_param_740);  view_647 = _shape_param_740 = None
        permute_351: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_699, [1, 0]);  convert_element_type_699 = None
        addmm_261: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_698, view_648, permute_351);  convert_element_type_698 = None
        view_649: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_261, _shape_param_741);  _shape_param_741 = None
        add_243: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_649, add_241);  view_649 = add_241 = None
        mul_211: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_243, arg807_1);  add_243 = arg807_1 = None
        add_244: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_211, arg808_1);  mul_211 = arg808_1 = None
        convert_element_type_700: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg809_1, torch.bfloat16);  arg809_1 = None
        convert_element_type_701: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg810_1, torch.bfloat16);  arg810_1 = None
        convert_element_type_702: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_244, torch.bfloat16)
        view_650: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_702, _shape_param_742);  convert_element_type_702 = _shape_param_742 = None
        permute_352: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_701, [1, 0]);  convert_element_type_701 = None
        addmm_262: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_700, view_650, permute_352);  convert_element_type_700 = None
        view_651: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_262, _shape_param_743);  addmm_262 = _shape_param_743 = None
        relu_68: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_651);  view_651 = None
        convert_element_type_703: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg811_1, torch.bfloat16);  arg811_1 = None
        convert_element_type_704: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg812_1, torch.bfloat16);  arg812_1 = None
        view_652: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_68, _shape_param_744);  _shape_param_744 = None
        permute_353: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_704, [1, 0]);  convert_element_type_704 = None
        addmm_263: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_703, view_652, permute_353);  convert_element_type_703 = None
        view_653: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_263, _shape_param_745);  addmm_263 = _shape_param_745 = None
        add_245: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_653, add_244);  view_653 = add_244 = None
        mul_212: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_245, arg813_1);  arg813_1 = None
        add_246: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_212, arg814_1);  mul_212 = arg814_1 = None
        convert_element_type_705: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg815_1, torch.bfloat16);  arg815_1 = None
        convert_element_type_706: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg816_1, torch.bfloat16);  arg816_1 = None
        convert_element_type_707: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_246, torch.bfloat16)
        view_654: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_707, _shape_param_746);  convert_element_type_707 = _shape_param_746 = None
        permute_354: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_706, [1, 0]);  convert_element_type_706 = None
        addmm_264: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_705, view_654, permute_354);  convert_element_type_705 = None
        view_655: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_264, _shape_param_747);  addmm_264 = _shape_param_747 = None
        relu_69: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_655);  view_655 = None
        convert_element_type_708: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg817_1, torch.bfloat16);  arg817_1 = None
        convert_element_type_709: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg818_1, torch.bfloat16);  arg818_1 = None
        view_656: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_69, _shape_param_748);  _shape_param_748 = None
        permute_355: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_709, [1, 0]);  convert_element_type_709 = None
        addmm_265: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_708, view_656, permute_355);  convert_element_type_708 = None
        view_657: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_265, _shape_param_749);  addmm_265 = _shape_param_749 = None
        add_247: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_657, add_246);  view_657 = add_246 = None
        mul_213: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_247, arg819_1);  arg819_1 = None
        add_248: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_213, arg820_1);  mul_213 = arg820_1 = None
        convert_element_type_710: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg821_1, torch.bfloat16);  arg821_1 = None
        convert_element_type_711: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg822_1, torch.bfloat16);  arg822_1 = None
        convert_element_type_712: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_248, torch.bfloat16)
        view_658: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_712, _shape_param_750);  convert_element_type_712 = _shape_param_750 = None
        permute_356: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_711, [1, 0]);  convert_element_type_711 = None
        addmm_266: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_710, view_658, permute_356);  convert_element_type_710 = None
        view_659: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_266, _shape_param_751);  addmm_266 = _shape_param_751 = None
        relu_70: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_659);  view_659 = None
        convert_element_type_713: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg823_1, torch.bfloat16);  arg823_1 = None
        convert_element_type_714: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg824_1, torch.bfloat16);  arg824_1 = None
        view_660: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_70, _shape_param_752);  _shape_param_752 = None
        permute_357: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_714, [1, 0]);  convert_element_type_714 = None
        addmm_267: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_713, view_660, permute_357);  convert_element_type_713 = None
        view_661: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_267, _shape_param_753);  addmm_267 = _shape_param_753 = None
        add_249: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_661, add_248);  view_661 = add_248 = None
        mul_214: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_249, arg825_1);  arg825_1 = None
        add_250: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_214, arg826_1);  mul_214 = arg826_1 = None
        convert_element_type_715: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg827_1, torch.bfloat16);  arg827_1 = None
        convert_element_type_716: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg828_1, torch.bfloat16);  arg828_1 = None
        convert_element_type_717: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_250, torch.bfloat16)
        view_662: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_717, _shape_param_754);  convert_element_type_717 = _shape_param_754 = None
        permute_358: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_716, [1, 0]);  convert_element_type_716 = None
        addmm_268: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_715, view_662, permute_358);  convert_element_type_715 = None
        view_663: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_268, _shape_param_755);  addmm_268 = _shape_param_755 = None
        relu_71: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_663);  view_663 = None
        convert_element_type_718: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg829_1, torch.bfloat16);  arg829_1 = None
        convert_element_type_719: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg830_1, torch.bfloat16);  arg830_1 = None
        view_664: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_71, _shape_param_756);  _shape_param_756 = None
        permute_359: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_719, [1, 0]);  convert_element_type_719 = None
        addmm_269: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_718, view_664, permute_359);  convert_element_type_718 = None
        view_665: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_269, _shape_param_757);  addmm_269 = _shape_param_757 = None
        add_251: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_665, add_250);  view_665 = add_250 = None
        mul_215: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_251, arg831_1);  arg831_1 = None
        add_252: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_215, arg832_1);  mul_215 = arg832_1 = None
        convert_element_type_720: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg833_1, torch.bfloat16);  arg833_1 = None
        convert_element_type_721: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg834_1, torch.bfloat16);  arg834_1 = None
        convert_element_type_722: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_252, torch.bfloat16);  add_252 = None
        view_666: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_722, _shape_param_758);  convert_element_type_722 = _shape_param_758 = None
        permute_360: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_721, [1, 0]);  convert_element_type_721 = None
        addmm_270: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_720, view_666, permute_360);  convert_element_type_720 = None
        view_667: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_270, _shape_param_759);  addmm_270 = _shape_param_759 = None
        add_253: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_667, add_240);  view_667 = add_240 = None
        mul_216: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_253, arg835_1);  arg835_1 = None
        add_254: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_216, arg836_1);  mul_216 = arg836_1 = None
        convert_element_type_723: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg837_1, torch.bfloat16);  arg837_1 = None
        convert_element_type_724: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg838_1, torch.bfloat16);  arg838_1 = None
        convert_element_type_725: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_254, torch.bfloat16)
        view_668: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_725, _shape_param_760);  convert_element_type_725 = _shape_param_760 = None
        permute_361: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_724, [1, 0]);  convert_element_type_724 = None
        addmm_271: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_723, view_668, permute_361);  convert_element_type_723 = None
        view_669: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_271, _shape_param_761);  _shape_param_761 = None
        mul_217: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_669, arg839_1);  view_669 = arg839_1 = None
        add_255: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_217, arg840_1);  mul_217 = arg840_1 = None
        convert_element_type_726: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg841_1, torch.bfloat16);  arg841_1 = None
        convert_element_type_727: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg842_1, torch.bfloat16);  arg842_1 = None
        permute_362: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_727, [1, 0]);  convert_element_type_727 = None
        addmm_272: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_726, view_668, permute_362);  convert_element_type_726 = None
        view_670: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_272, _shape_param_762);  _shape_param_762 = None
        mul_218: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_670, arg843_1);  view_670 = arg843_1 = None
        add_256: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_218, arg844_1);  mul_218 = arg844_1 = None
        convert_element_type_728: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg845_1, torch.bfloat16);  arg845_1 = None
        convert_element_type_729: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg846_1, torch.bfloat16);  arg846_1 = None
        convert_element_type_730: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_256, torch.bfloat16);  add_256 = None
        view_671: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_730, _shape_param_763);  convert_element_type_730 = _shape_param_763 = None
        permute_363: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_729, [1, 0]);  convert_element_type_729 = None
        addmm_273: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_728, view_671, permute_363);  convert_element_type_728 = None
        view_672: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_273, _shape_param_764);  addmm_273 = _shape_param_764 = None
        view_673: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_672, _shape_param_765);  view_672 = _shape_param_765 = None
        permute_364: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_673, [0, 2, 1, 3]);  view_673 = None
        convert_element_type_731: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg847_1, torch.bfloat16);  arg847_1 = None
        convert_element_type_732: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg848_1, torch.bfloat16);  arg848_1 = None
        permute_365: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_732, [1, 0]);  convert_element_type_732 = None
        addmm_274: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_731, view_671, permute_365);  convert_element_type_731 = None
        view_674: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_274, _shape_param_766);  addmm_274 = _shape_param_766 = None
        view_675: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_674, _shape_param_767);  view_674 = _shape_param_767 = None
        permute_366: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None
        convert_element_type_733: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg849_1, torch.bfloat16);  arg849_1 = None
        convert_element_type_734: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg850_1, torch.bfloat16);  arg850_1 = None
        permute_367: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_734, [1, 0]);  convert_element_type_734 = None
        addmm_275: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_733, view_668, permute_367);  convert_element_type_733 = None
        view_676: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_275, _shape_param_768);  addmm_275 = _shape_param_768 = None
        view_677: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_676, _shape_param_769);  view_676 = _shape_param_769 = None
        permute_368: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_677, [0, 2, 1, 3]);  view_677 = None
        mul_219: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_364, 0.4204482076268573);  permute_364 = None
        permute_369: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_366, [0, 1, 3, 2]);  permute_366 = None
        mul_220: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_369, 0.4204482076268573);  permute_369 = None
        expand_72: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_219, _shape_param_770);  mul_219 = _shape_param_770 = None
        clone_72: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_678: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_72, _shape_param_771);  clone_72 = _shape_param_771 = None
        expand_73: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_220, _shape_param_772);  mul_220 = _shape_param_772 = None
        clone_73: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_679: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_773);  clone_73 = _shape_param_773 = None
        bmm_36: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_678, view_679)
        view_680: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_774);  bmm_36 = _shape_param_774 = None
        convert_element_type_735: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_680, torch.float32)
        amax_18: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_735, [-1], True)
        sub_18: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_735, amax_18);  convert_element_type_735 = amax_18 = None
        exp_18: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_19: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_736: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        eq_18: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_680, -inf);  view_680 = None
        logical_not_36: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        where_18: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_37, full_1, convert_element_type_736);  logical_not_37 = convert_element_type_736 = None
        inductor_lookup_seed_18: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 18)
        inductor_random_18: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_775, inductor_lookup_seed_18, 'rand');  _shape_param_775 = inductor_lookup_seed_18 = None
        convert_element_type_737: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_18, torch.bfloat16);  inductor_random_18 = None
        gt_18: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_737, 0.1);  convert_element_type_737 = None
        mul_221: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_18, where_18)
        mul_222: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_221, 1.1111111111111112);  mul_221 = None
        expand_74: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_222, _shape_param_776);  mul_222 = _shape_param_776 = None
        view_681: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_74, _shape_param_777);  expand_74 = _shape_param_777 = None
        expand_75: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_368, _shape_param_778);  permute_368 = _shape_param_778 = None
        clone_74: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_682: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_74, _shape_param_779);  clone_74 = _shape_param_779 = None
        bmm_37: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_681, view_682)
        view_683: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_37, _shape_param_780);  bmm_37 = _shape_param_780 = None
        permute_370: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_683, [0, 2, 1, 3]);  view_683 = None
        clone_75: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_370, memory_format = torch.contiguous_format);  permute_370 = None
        view_684: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_75, _shape_param_781);  clone_75 = _shape_param_781 = None
        convert_element_type_738: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg851_1, torch.bfloat16);  arg851_1 = None
        convert_element_type_739: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg852_1, torch.bfloat16);  arg852_1 = None
        view_685: "bf16[32768, 128]" = torch.ops.aten.view.default(view_684, _shape_param_782);  view_684 = _shape_param_782 = None
        permute_371: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_739, [1, 0]);  convert_element_type_739 = None
        addmm_276: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_738, view_685, permute_371);  convert_element_type_738 = None
        view_686: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_276, _shape_param_783);  _shape_param_783 = None
        add_257: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_686, add_255);  view_686 = add_255 = None
        mul_223: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_257, arg853_1);  add_257 = arg853_1 = None
        add_258: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_223, arg854_1);  mul_223 = arg854_1 = None
        convert_element_type_740: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg855_1, torch.bfloat16);  arg855_1 = None
        convert_element_type_741: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg856_1, torch.bfloat16);  arg856_1 = None
        convert_element_type_742: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_258, torch.bfloat16)
        view_687: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_742, _shape_param_784);  convert_element_type_742 = _shape_param_784 = None
        permute_372: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_741, [1, 0]);  convert_element_type_741 = None
        addmm_277: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_740, view_687, permute_372);  convert_element_type_740 = None
        view_688: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_277, _shape_param_785);  addmm_277 = _shape_param_785 = None
        relu_72: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_688);  view_688 = None
        convert_element_type_743: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg857_1, torch.bfloat16);  arg857_1 = None
        convert_element_type_744: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg858_1, torch.bfloat16);  arg858_1 = None
        view_689: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_72, _shape_param_786);  _shape_param_786 = None
        permute_373: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_744, [1, 0]);  convert_element_type_744 = None
        addmm_278: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_743, view_689, permute_373);  convert_element_type_743 = None
        view_690: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_278, _shape_param_787);  addmm_278 = _shape_param_787 = None
        add_259: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_690, add_258);  view_690 = add_258 = None
        mul_224: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_259, arg859_1);  arg859_1 = None
        add_260: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_224, arg860_1);  mul_224 = arg860_1 = None
        convert_element_type_745: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg861_1, torch.bfloat16);  arg861_1 = None
        convert_element_type_746: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg862_1, torch.bfloat16);  arg862_1 = None
        convert_element_type_747: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_260, torch.bfloat16)
        view_691: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_747, _shape_param_788);  convert_element_type_747 = _shape_param_788 = None
        permute_374: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_746, [1, 0]);  convert_element_type_746 = None
        addmm_279: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_745, view_691, permute_374);  convert_element_type_745 = None
        view_692: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_279, _shape_param_789);  addmm_279 = _shape_param_789 = None
        relu_73: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_692);  view_692 = None
        convert_element_type_748: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg863_1, torch.bfloat16);  arg863_1 = None
        convert_element_type_749: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg864_1, torch.bfloat16);  arg864_1 = None
        view_693: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_73, _shape_param_790);  _shape_param_790 = None
        permute_375: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_749, [1, 0]);  convert_element_type_749 = None
        addmm_280: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_748, view_693, permute_375);  convert_element_type_748 = None
        view_694: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_280, _shape_param_791);  addmm_280 = _shape_param_791 = None
        add_261: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_694, add_260);  view_694 = add_260 = None
        mul_225: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_261, arg865_1);  arg865_1 = None
        add_262: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_225, arg866_1);  mul_225 = arg866_1 = None
        convert_element_type_750: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg867_1, torch.bfloat16);  arg867_1 = None
        convert_element_type_751: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg868_1, torch.bfloat16);  arg868_1 = None
        convert_element_type_752: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_262, torch.bfloat16)
        view_695: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_752, _shape_param_792);  convert_element_type_752 = _shape_param_792 = None
        permute_376: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_751, [1, 0]);  convert_element_type_751 = None
        addmm_281: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_750, view_695, permute_376);  convert_element_type_750 = None
        view_696: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_281, _shape_param_793);  addmm_281 = _shape_param_793 = None
        relu_74: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_696);  view_696 = None
        convert_element_type_753: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg869_1, torch.bfloat16);  arg869_1 = None
        convert_element_type_754: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg870_1, torch.bfloat16);  arg870_1 = None
        view_697: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_74, _shape_param_794);  _shape_param_794 = None
        permute_377: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_754, [1, 0]);  convert_element_type_754 = None
        addmm_282: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_753, view_697, permute_377);  convert_element_type_753 = None
        view_698: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_282, _shape_param_795);  addmm_282 = _shape_param_795 = None
        add_263: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_698, add_262);  view_698 = add_262 = None
        mul_226: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, arg871_1);  arg871_1 = None
        add_264: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_226, arg872_1);  mul_226 = arg872_1 = None
        convert_element_type_755: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg873_1, torch.bfloat16);  arg873_1 = None
        convert_element_type_756: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg874_1, torch.bfloat16);  arg874_1 = None
        convert_element_type_757: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_264, torch.bfloat16)
        view_699: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_757, _shape_param_796);  convert_element_type_757 = _shape_param_796 = None
        permute_378: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_756, [1, 0]);  convert_element_type_756 = None
        addmm_283: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_755, view_699, permute_378);  convert_element_type_755 = None
        view_700: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_283, _shape_param_797);  addmm_283 = _shape_param_797 = None
        relu_75: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_700);  view_700 = None
        convert_element_type_758: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg875_1, torch.bfloat16);  arg875_1 = None
        convert_element_type_759: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg876_1, torch.bfloat16);  arg876_1 = None
        view_701: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_75, _shape_param_798);  _shape_param_798 = None
        permute_379: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_759, [1, 0]);  convert_element_type_759 = None
        addmm_284: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_758, view_701, permute_379);  convert_element_type_758 = None
        view_702: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_284, _shape_param_799);  addmm_284 = _shape_param_799 = None
        add_265: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_702, add_264);  view_702 = add_264 = None
        mul_227: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, arg877_1);  arg877_1 = None
        add_266: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_227, arg878_1);  mul_227 = arg878_1 = None
        convert_element_type_760: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg879_1, torch.bfloat16);  arg879_1 = None
        convert_element_type_761: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg880_1, torch.bfloat16);  arg880_1 = None
        convert_element_type_762: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_266, torch.bfloat16);  add_266 = None
        view_703: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_762, _shape_param_800);  convert_element_type_762 = _shape_param_800 = None
        permute_380: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_761, [1, 0]);  convert_element_type_761 = None
        addmm_285: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_760, view_703, permute_380);  convert_element_type_760 = None
        view_704: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_285, _shape_param_801);  addmm_285 = _shape_param_801 = None
        add_267: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_704, add_254);  view_704 = add_254 = None
        mul_228: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_267, arg881_1);  arg881_1 = None
        add_268: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_228, arg882_1);  mul_228 = arg882_1 = None
        convert_element_type_763: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg883_1, torch.bfloat16);  arg883_1 = None
        convert_element_type_764: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg884_1, torch.bfloat16);  arg884_1 = None
        convert_element_type_765: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_268, torch.bfloat16)
        view_705: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_765, _shape_param_802);  convert_element_type_765 = _shape_param_802 = None
        permute_381: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_764, [1, 0]);  convert_element_type_764 = None
        addmm_286: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_763, view_705, permute_381);  convert_element_type_763 = None
        view_706: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_286, _shape_param_803);  _shape_param_803 = None
        mul_229: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_706, arg885_1);  view_706 = arg885_1 = None
        add_269: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_229, arg886_1);  mul_229 = arg886_1 = None
        convert_element_type_766: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg887_1, torch.bfloat16);  arg887_1 = None
        convert_element_type_767: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg888_1, torch.bfloat16);  arg888_1 = None
        permute_382: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_767, [1, 0]);  convert_element_type_767 = None
        addmm_287: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_766, view_705, permute_382);  convert_element_type_766 = None
        view_707: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_287, _shape_param_804);  _shape_param_804 = None
        mul_230: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_707, arg889_1);  view_707 = arg889_1 = None
        add_270: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_230, arg890_1);  mul_230 = arg890_1 = None
        convert_element_type_768: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg891_1, torch.bfloat16);  arg891_1 = None
        convert_element_type_769: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg892_1, torch.bfloat16);  arg892_1 = None
        convert_element_type_770: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_270, torch.bfloat16);  add_270 = None
        view_708: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_770, _shape_param_805);  convert_element_type_770 = _shape_param_805 = None
        permute_383: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_769, [1, 0]);  convert_element_type_769 = None
        addmm_288: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_768, view_708, permute_383);  convert_element_type_768 = None
        view_709: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_288, _shape_param_806);  addmm_288 = _shape_param_806 = None
        view_710: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_709, _shape_param_807);  view_709 = _shape_param_807 = None
        permute_384: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None
        convert_element_type_771: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg893_1, torch.bfloat16);  arg893_1 = None
        convert_element_type_772: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg894_1, torch.bfloat16);  arg894_1 = None
        permute_385: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_772, [1, 0]);  convert_element_type_772 = None
        addmm_289: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_771, view_708, permute_385);  convert_element_type_771 = None
        view_711: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_289, _shape_param_808);  addmm_289 = _shape_param_808 = None
        view_712: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_711, _shape_param_809);  view_711 = _shape_param_809 = None
        permute_386: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_712, [0, 2, 1, 3]);  view_712 = None
        convert_element_type_773: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg895_1, torch.bfloat16);  arg895_1 = None
        convert_element_type_774: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg896_1, torch.bfloat16);  arg896_1 = None
        permute_387: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_774, [1, 0]);  convert_element_type_774 = None
        addmm_290: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_773, view_705, permute_387);  convert_element_type_773 = None
        view_713: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_290, _shape_param_810);  addmm_290 = _shape_param_810 = None
        view_714: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_713, _shape_param_811);  view_713 = _shape_param_811 = None
        permute_388: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_714, [0, 2, 1, 3]);  view_714 = None
        mul_231: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_384, 0.4204482076268573);  permute_384 = None
        permute_389: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_386, [0, 1, 3, 2]);  permute_386 = None
        mul_232: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_389, 0.4204482076268573);  permute_389 = None
        expand_76: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_231, _shape_param_812);  mul_231 = _shape_param_812 = None
        clone_76: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_715: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_76, _shape_param_813);  clone_76 = _shape_param_813 = None
        expand_77: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_232, _shape_param_814);  mul_232 = _shape_param_814 = None
        clone_77: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_716: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_815);  clone_77 = _shape_param_815 = None
        bmm_38: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_715, view_716)
        view_717: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_816);  bmm_38 = _shape_param_816 = None
        convert_element_type_775: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_717, torch.float32)
        amax_19: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_775, [-1], True)
        sub_19: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_775, amax_19);  convert_element_type_775 = amax_19 = None
        exp_19: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_20: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_776: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        eq_19: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_717, -inf);  view_717 = None
        logical_not_38: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        where_19: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_39, full_1, convert_element_type_776);  logical_not_39 = convert_element_type_776 = None
        inductor_lookup_seed_19: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 19)
        inductor_random_19: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_817, inductor_lookup_seed_19, 'rand');  _shape_param_817 = inductor_lookup_seed_19 = None
        convert_element_type_777: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_19, torch.bfloat16);  inductor_random_19 = None
        gt_19: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_777, 0.1);  convert_element_type_777 = None
        mul_233: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_19, where_19)
        mul_234: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_233, 1.1111111111111112);  mul_233 = None
        expand_78: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_234, _shape_param_818);  mul_234 = _shape_param_818 = None
        view_718: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_78, _shape_param_819);  expand_78 = _shape_param_819 = None
        expand_79: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_388, _shape_param_820);  permute_388 = _shape_param_820 = None
        clone_78: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_719: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_78, _shape_param_821);  clone_78 = _shape_param_821 = None
        bmm_39: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_718, view_719)
        view_720: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_39, _shape_param_822);  bmm_39 = _shape_param_822 = None
        permute_390: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_720, [0, 2, 1, 3]);  view_720 = None
        clone_79: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_390, memory_format = torch.contiguous_format);  permute_390 = None
        view_721: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_79, _shape_param_823);  clone_79 = _shape_param_823 = None
        convert_element_type_778: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg897_1, torch.bfloat16);  arg897_1 = None
        convert_element_type_779: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg898_1, torch.bfloat16);  arg898_1 = None
        view_722: "bf16[32768, 128]" = torch.ops.aten.view.default(view_721, _shape_param_824);  view_721 = _shape_param_824 = None
        permute_391: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_779, [1, 0]);  convert_element_type_779 = None
        addmm_291: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_778, view_722, permute_391);  convert_element_type_778 = None
        view_723: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_291, _shape_param_825);  _shape_param_825 = None
        add_271: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_723, add_269);  view_723 = add_269 = None
        mul_235: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_271, arg899_1);  add_271 = arg899_1 = None
        add_272: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_235, arg900_1);  mul_235 = arg900_1 = None
        convert_element_type_780: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg901_1, torch.bfloat16);  arg901_1 = None
        convert_element_type_781: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg902_1, torch.bfloat16);  arg902_1 = None
        convert_element_type_782: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_272, torch.bfloat16)
        view_724: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_782, _shape_param_826);  convert_element_type_782 = _shape_param_826 = None
        permute_392: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_781, [1, 0]);  convert_element_type_781 = None
        addmm_292: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_780, view_724, permute_392);  convert_element_type_780 = None
        view_725: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_292, _shape_param_827);  addmm_292 = _shape_param_827 = None
        relu_76: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_725);  view_725 = None
        convert_element_type_783: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg903_1, torch.bfloat16);  arg903_1 = None
        convert_element_type_784: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg904_1, torch.bfloat16);  arg904_1 = None
        view_726: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_76, _shape_param_828);  _shape_param_828 = None
        permute_393: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_784, [1, 0]);  convert_element_type_784 = None
        addmm_293: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_783, view_726, permute_393);  convert_element_type_783 = None
        view_727: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_293, _shape_param_829);  addmm_293 = _shape_param_829 = None
        add_273: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_727, add_272);  view_727 = add_272 = None
        mul_236: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_273, arg905_1);  arg905_1 = None
        add_274: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_236, arg906_1);  mul_236 = arg906_1 = None
        convert_element_type_785: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg907_1, torch.bfloat16);  arg907_1 = None
        convert_element_type_786: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg908_1, torch.bfloat16);  arg908_1 = None
        convert_element_type_787: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_274, torch.bfloat16)
        view_728: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_787, _shape_param_830);  convert_element_type_787 = _shape_param_830 = None
        permute_394: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_786, [1, 0]);  convert_element_type_786 = None
        addmm_294: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_785, view_728, permute_394);  convert_element_type_785 = None
        view_729: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_294, _shape_param_831);  addmm_294 = _shape_param_831 = None
        relu_77: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_729);  view_729 = None
        convert_element_type_788: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg909_1, torch.bfloat16);  arg909_1 = None
        convert_element_type_789: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg910_1, torch.bfloat16);  arg910_1 = None
        view_730: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_77, _shape_param_832);  _shape_param_832 = None
        permute_395: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_789, [1, 0]);  convert_element_type_789 = None
        addmm_295: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_788, view_730, permute_395);  convert_element_type_788 = None
        view_731: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_295, _shape_param_833);  addmm_295 = _shape_param_833 = None
        add_275: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_731, add_274);  view_731 = add_274 = None
        mul_237: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_275, arg911_1);  arg911_1 = None
        add_276: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_237, arg912_1);  mul_237 = arg912_1 = None
        convert_element_type_790: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg913_1, torch.bfloat16);  arg913_1 = None
        convert_element_type_791: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg914_1, torch.bfloat16);  arg914_1 = None
        convert_element_type_792: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_276, torch.bfloat16)
        view_732: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_792, _shape_param_834);  convert_element_type_792 = _shape_param_834 = None
        permute_396: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_791, [1, 0]);  convert_element_type_791 = None
        addmm_296: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_790, view_732, permute_396);  convert_element_type_790 = None
        view_733: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_296, _shape_param_835);  addmm_296 = _shape_param_835 = None
        relu_78: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_733);  view_733 = None
        convert_element_type_793: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg915_1, torch.bfloat16);  arg915_1 = None
        convert_element_type_794: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg916_1, torch.bfloat16);  arg916_1 = None
        view_734: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_78, _shape_param_836);  _shape_param_836 = None
        permute_397: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_794, [1, 0]);  convert_element_type_794 = None
        addmm_297: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_793, view_734, permute_397);  convert_element_type_793 = None
        view_735: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_297, _shape_param_837);  addmm_297 = _shape_param_837 = None
        add_277: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_735, add_276);  view_735 = add_276 = None
        mul_238: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_277, arg917_1);  arg917_1 = None
        add_278: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_238, arg918_1);  mul_238 = arg918_1 = None
        convert_element_type_795: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg919_1, torch.bfloat16);  arg919_1 = None
        convert_element_type_796: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg920_1, torch.bfloat16);  arg920_1 = None
        convert_element_type_797: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_278, torch.bfloat16)
        view_736: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_797, _shape_param_838);  convert_element_type_797 = _shape_param_838 = None
        permute_398: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_796, [1, 0]);  convert_element_type_796 = None
        addmm_298: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_795, view_736, permute_398);  convert_element_type_795 = None
        view_737: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_298, _shape_param_839);  addmm_298 = _shape_param_839 = None
        relu_79: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_737);  view_737 = None
        convert_element_type_798: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg921_1, torch.bfloat16);  arg921_1 = None
        convert_element_type_799: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg922_1, torch.bfloat16);  arg922_1 = None
        view_738: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_79, _shape_param_840);  _shape_param_840 = None
        permute_399: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_799, [1, 0]);  convert_element_type_799 = None
        addmm_299: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_798, view_738, permute_399);  convert_element_type_798 = None
        view_739: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_299, _shape_param_841);  addmm_299 = _shape_param_841 = None
        add_279: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_739, add_278);  view_739 = add_278 = None
        mul_239: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_279, arg923_1);  arg923_1 = None
        add_280: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_239, arg924_1);  mul_239 = arg924_1 = None
        convert_element_type_800: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg925_1, torch.bfloat16);  arg925_1 = None
        convert_element_type_801: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg926_1, torch.bfloat16);  arg926_1 = None
        convert_element_type_802: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_280, torch.bfloat16);  add_280 = None
        view_740: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_802, _shape_param_842);  convert_element_type_802 = _shape_param_842 = None
        permute_400: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_801, [1, 0]);  convert_element_type_801 = None
        addmm_300: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_800, view_740, permute_400);  convert_element_type_800 = None
        view_741: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_300, _shape_param_843);  addmm_300 = _shape_param_843 = None
        add_281: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_741, add_268);  view_741 = add_268 = None
        mul_240: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_281, arg927_1);  arg927_1 = None
        add_282: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_240, arg928_1);  mul_240 = arg928_1 = None
        convert_element_type_803: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg929_1, torch.bfloat16);  arg929_1 = None
        convert_element_type_804: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg930_1, torch.bfloat16);  arg930_1 = None
        convert_element_type_805: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_282, torch.bfloat16)
        view_742: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_805, _shape_param_844);  convert_element_type_805 = _shape_param_844 = None
        permute_401: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_804, [1, 0]);  convert_element_type_804 = None
        addmm_301: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_803, view_742, permute_401);  convert_element_type_803 = None
        view_743: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_301, _shape_param_845);  _shape_param_845 = None
        mul_241: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_743, arg931_1);  view_743 = arg931_1 = None
        add_283: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_241, arg932_1);  mul_241 = arg932_1 = None
        convert_element_type_806: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg933_1, torch.bfloat16);  arg933_1 = None
        convert_element_type_807: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg934_1, torch.bfloat16);  arg934_1 = None
        permute_402: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_807, [1, 0]);  convert_element_type_807 = None
        addmm_302: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_806, view_742, permute_402);  convert_element_type_806 = None
        view_744: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_302, _shape_param_846);  _shape_param_846 = None
        mul_242: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_744, arg935_1);  view_744 = arg935_1 = None
        add_284: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_242, arg936_1);  mul_242 = arg936_1 = None
        convert_element_type_808: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg937_1, torch.bfloat16);  arg937_1 = None
        convert_element_type_809: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg938_1, torch.bfloat16);  arg938_1 = None
        convert_element_type_810: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_284, torch.bfloat16);  add_284 = None
        view_745: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_810, _shape_param_847);  convert_element_type_810 = _shape_param_847 = None
        permute_403: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_809, [1, 0]);  convert_element_type_809 = None
        addmm_303: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_808, view_745, permute_403);  convert_element_type_808 = None
        view_746: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_303, _shape_param_848);  addmm_303 = _shape_param_848 = None
        view_747: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_746, _shape_param_849);  view_746 = _shape_param_849 = None
        permute_404: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_747, [0, 2, 1, 3]);  view_747 = None
        convert_element_type_811: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg939_1, torch.bfloat16);  arg939_1 = None
        convert_element_type_812: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg940_1, torch.bfloat16);  arg940_1 = None
        permute_405: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_812, [1, 0]);  convert_element_type_812 = None
        addmm_304: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_811, view_745, permute_405);  convert_element_type_811 = None
        view_748: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_304, _shape_param_850);  addmm_304 = _shape_param_850 = None
        view_749: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_748, _shape_param_851);  view_748 = _shape_param_851 = None
        permute_406: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_749, [0, 2, 1, 3]);  view_749 = None
        convert_element_type_813: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg941_1, torch.bfloat16);  arg941_1 = None
        convert_element_type_814: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg942_1, torch.bfloat16);  arg942_1 = None
        permute_407: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_814, [1, 0]);  convert_element_type_814 = None
        addmm_305: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_813, view_742, permute_407);  convert_element_type_813 = None
        view_750: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_305, _shape_param_852);  addmm_305 = _shape_param_852 = None
        view_751: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_750, _shape_param_853);  view_750 = _shape_param_853 = None
        permute_408: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_751, [0, 2, 1, 3]);  view_751 = None
        mul_243: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_404, 0.4204482076268573);  permute_404 = None
        permute_409: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_406, [0, 1, 3, 2]);  permute_406 = None
        mul_244: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_409, 0.4204482076268573);  permute_409 = None
        expand_80: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_243, _shape_param_854);  mul_243 = _shape_param_854 = None
        clone_80: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_752: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_80, _shape_param_855);  clone_80 = _shape_param_855 = None
        expand_81: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_244, _shape_param_856);  mul_244 = _shape_param_856 = None
        clone_81: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_753: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_857);  clone_81 = _shape_param_857 = None
        bmm_40: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_752, view_753)
        view_754: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_858);  bmm_40 = _shape_param_858 = None
        convert_element_type_815: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_754, torch.float32)
        amax_20: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_815, [-1], True)
        sub_20: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_815, amax_20);  convert_element_type_815 = amax_20 = None
        exp_20: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_21: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_816: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        eq_20: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_754, -inf);  view_754 = None
        logical_not_40: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        where_20: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_41, full_1, convert_element_type_816);  logical_not_41 = convert_element_type_816 = None
        inductor_lookup_seed_20: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 20)
        inductor_random_20: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_859, inductor_lookup_seed_20, 'rand');  _shape_param_859 = inductor_lookup_seed_20 = None
        convert_element_type_817: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_20, torch.bfloat16);  inductor_random_20 = None
        gt_20: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_817, 0.1);  convert_element_type_817 = None
        mul_245: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_20, where_20)
        mul_246: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_245, 1.1111111111111112);  mul_245 = None
        expand_82: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_246, _shape_param_860);  mul_246 = _shape_param_860 = None
        view_755: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_82, _shape_param_861);  expand_82 = _shape_param_861 = None
        expand_83: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_408, _shape_param_862);  permute_408 = _shape_param_862 = None
        clone_82: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_756: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_82, _shape_param_863);  clone_82 = _shape_param_863 = None
        bmm_41: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_755, view_756)
        view_757: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_41, _shape_param_864);  bmm_41 = _shape_param_864 = None
        permute_410: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_757, [0, 2, 1, 3]);  view_757 = None
        clone_83: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_410, memory_format = torch.contiguous_format);  permute_410 = None
        view_758: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_83, _shape_param_865);  clone_83 = _shape_param_865 = None
        convert_element_type_818: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg943_1, torch.bfloat16);  arg943_1 = None
        convert_element_type_819: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg944_1, torch.bfloat16);  arg944_1 = None
        view_759: "bf16[32768, 128]" = torch.ops.aten.view.default(view_758, _shape_param_866);  view_758 = _shape_param_866 = None
        permute_411: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_819, [1, 0]);  convert_element_type_819 = None
        addmm_306: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_818, view_759, permute_411);  convert_element_type_818 = None
        view_760: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_306, _shape_param_867);  _shape_param_867 = None
        add_285: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_760, add_283);  view_760 = add_283 = None
        mul_247: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_285, arg945_1);  add_285 = arg945_1 = None
        add_286: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_247, arg946_1);  mul_247 = arg946_1 = None
        convert_element_type_820: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg947_1, torch.bfloat16);  arg947_1 = None
        convert_element_type_821: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg948_1, torch.bfloat16);  arg948_1 = None
        convert_element_type_822: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_286, torch.bfloat16)
        view_761: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_822, _shape_param_868);  convert_element_type_822 = _shape_param_868 = None
        permute_412: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_821, [1, 0]);  convert_element_type_821 = None
        addmm_307: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_820, view_761, permute_412);  convert_element_type_820 = None
        view_762: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_307, _shape_param_869);  addmm_307 = _shape_param_869 = None
        relu_80: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_762);  view_762 = None
        convert_element_type_823: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg949_1, torch.bfloat16);  arg949_1 = None
        convert_element_type_824: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg950_1, torch.bfloat16);  arg950_1 = None
        view_763: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_80, _shape_param_870);  _shape_param_870 = None
        permute_413: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_824, [1, 0]);  convert_element_type_824 = None
        addmm_308: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_823, view_763, permute_413);  convert_element_type_823 = None
        view_764: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_308, _shape_param_871);  addmm_308 = _shape_param_871 = None
        add_287: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_764, add_286);  view_764 = add_286 = None
        mul_248: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_287, arg951_1);  arg951_1 = None
        add_288: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_248, arg952_1);  mul_248 = arg952_1 = None
        convert_element_type_825: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg953_1, torch.bfloat16);  arg953_1 = None
        convert_element_type_826: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg954_1, torch.bfloat16);  arg954_1 = None
        convert_element_type_827: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_288, torch.bfloat16)
        view_765: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_827, _shape_param_872);  convert_element_type_827 = _shape_param_872 = None
        permute_414: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_826, [1, 0]);  convert_element_type_826 = None
        addmm_309: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_825, view_765, permute_414);  convert_element_type_825 = None
        view_766: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_309, _shape_param_873);  addmm_309 = _shape_param_873 = None
        relu_81: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_766);  view_766 = None
        convert_element_type_828: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg955_1, torch.bfloat16);  arg955_1 = None
        convert_element_type_829: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg956_1, torch.bfloat16);  arg956_1 = None
        view_767: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_81, _shape_param_874);  _shape_param_874 = None
        permute_415: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_829, [1, 0]);  convert_element_type_829 = None
        addmm_310: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_828, view_767, permute_415);  convert_element_type_828 = None
        view_768: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_310, _shape_param_875);  addmm_310 = _shape_param_875 = None
        add_289: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_768, add_288);  view_768 = add_288 = None
        mul_249: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_289, arg957_1);  arg957_1 = None
        add_290: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_249, arg958_1);  mul_249 = arg958_1 = None
        convert_element_type_830: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg959_1, torch.bfloat16);  arg959_1 = None
        convert_element_type_831: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg960_1, torch.bfloat16);  arg960_1 = None
        convert_element_type_832: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_290, torch.bfloat16)
        view_769: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_832, _shape_param_876);  convert_element_type_832 = _shape_param_876 = None
        permute_416: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_831, [1, 0]);  convert_element_type_831 = None
        addmm_311: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_830, view_769, permute_416);  convert_element_type_830 = None
        view_770: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_311, _shape_param_877);  addmm_311 = _shape_param_877 = None
        relu_82: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_770);  view_770 = None
        convert_element_type_833: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg961_1, torch.bfloat16);  arg961_1 = None
        convert_element_type_834: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg962_1, torch.bfloat16);  arg962_1 = None
        view_771: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_82, _shape_param_878);  _shape_param_878 = None
        permute_417: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_834, [1, 0]);  convert_element_type_834 = None
        addmm_312: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_833, view_771, permute_417);  convert_element_type_833 = None
        view_772: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_312, _shape_param_879);  addmm_312 = _shape_param_879 = None
        add_291: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_772, add_290);  view_772 = add_290 = None
        mul_250: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_291, arg963_1);  arg963_1 = None
        add_292: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_250, arg964_1);  mul_250 = arg964_1 = None
        convert_element_type_835: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg965_1, torch.bfloat16);  arg965_1 = None
        convert_element_type_836: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg966_1, torch.bfloat16);  arg966_1 = None
        convert_element_type_837: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_292, torch.bfloat16)
        view_773: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_837, _shape_param_880);  convert_element_type_837 = _shape_param_880 = None
        permute_418: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_836, [1, 0]);  convert_element_type_836 = None
        addmm_313: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_835, view_773, permute_418);  convert_element_type_835 = None
        view_774: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_313, _shape_param_881);  addmm_313 = _shape_param_881 = None
        relu_83: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_774);  view_774 = None
        convert_element_type_838: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg967_1, torch.bfloat16);  arg967_1 = None
        convert_element_type_839: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg968_1, torch.bfloat16);  arg968_1 = None
        view_775: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_83, _shape_param_882);  _shape_param_882 = None
        permute_419: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_839, [1, 0]);  convert_element_type_839 = None
        addmm_314: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_838, view_775, permute_419);  convert_element_type_838 = None
        view_776: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_314, _shape_param_883);  addmm_314 = _shape_param_883 = None
        add_293: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_776, add_292);  view_776 = add_292 = None
        mul_251: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, arg969_1);  arg969_1 = None
        add_294: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_251, arg970_1);  mul_251 = arg970_1 = None
        convert_element_type_840: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg971_1, torch.bfloat16);  arg971_1 = None
        convert_element_type_841: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg972_1, torch.bfloat16);  arg972_1 = None
        convert_element_type_842: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_294, torch.bfloat16);  add_294 = None
        view_777: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_842, _shape_param_884);  convert_element_type_842 = _shape_param_884 = None
        permute_420: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_841, [1, 0]);  convert_element_type_841 = None
        addmm_315: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_840, view_777, permute_420);  convert_element_type_840 = None
        view_778: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_315, _shape_param_885);  addmm_315 = _shape_param_885 = None
        add_295: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_778, add_282);  view_778 = add_282 = None
        mul_252: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_295, arg973_1);  arg973_1 = None
        add_296: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_252, arg974_1);  mul_252 = arg974_1 = None
        convert_element_type_843: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg975_1, torch.bfloat16);  arg975_1 = None
        convert_element_type_844: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg976_1, torch.bfloat16);  arg976_1 = None
        convert_element_type_845: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_296, torch.bfloat16)
        view_779: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_845, _shape_param_886);  convert_element_type_845 = _shape_param_886 = None
        permute_421: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_844, [1, 0]);  convert_element_type_844 = None
        addmm_316: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_843, view_779, permute_421);  convert_element_type_843 = None
        view_780: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_316, _shape_param_887);  _shape_param_887 = None
        mul_253: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_780, arg977_1);  view_780 = arg977_1 = None
        add_297: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_253, arg978_1);  mul_253 = arg978_1 = None
        convert_element_type_846: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg979_1, torch.bfloat16);  arg979_1 = None
        convert_element_type_847: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg980_1, torch.bfloat16);  arg980_1 = None
        permute_422: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_847, [1, 0]);  convert_element_type_847 = None
        addmm_317: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_846, view_779, permute_422);  convert_element_type_846 = None
        view_781: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_317, _shape_param_888);  _shape_param_888 = None
        mul_254: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_781, arg981_1);  view_781 = arg981_1 = None
        add_298: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_254, arg982_1);  mul_254 = arg982_1 = None
        convert_element_type_848: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg983_1, torch.bfloat16);  arg983_1 = None
        convert_element_type_849: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg984_1, torch.bfloat16);  arg984_1 = None
        convert_element_type_850: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_298, torch.bfloat16);  add_298 = None
        view_782: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_850, _shape_param_889);  convert_element_type_850 = _shape_param_889 = None
        permute_423: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_849, [1, 0]);  convert_element_type_849 = None
        addmm_318: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_848, view_782, permute_423);  convert_element_type_848 = None
        view_783: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_318, _shape_param_890);  addmm_318 = _shape_param_890 = None
        view_784: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_783, _shape_param_891);  view_783 = _shape_param_891 = None
        permute_424: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
        convert_element_type_851: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg985_1, torch.bfloat16);  arg985_1 = None
        convert_element_type_852: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg986_1, torch.bfloat16);  arg986_1 = None
        permute_425: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_852, [1, 0]);  convert_element_type_852 = None
        addmm_319: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_851, view_782, permute_425);  convert_element_type_851 = None
        view_785: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_319, _shape_param_892);  addmm_319 = _shape_param_892 = None
        view_786: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_785, _shape_param_893);  view_785 = _shape_param_893 = None
        permute_426: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_786, [0, 2, 1, 3]);  view_786 = None
        convert_element_type_853: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg987_1, torch.bfloat16);  arg987_1 = None
        convert_element_type_854: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg988_1, torch.bfloat16);  arg988_1 = None
        permute_427: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_854, [1, 0]);  convert_element_type_854 = None
        addmm_320: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_853, view_779, permute_427);  convert_element_type_853 = None
        view_787: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_320, _shape_param_894);  addmm_320 = _shape_param_894 = None
        view_788: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_787, _shape_param_895);  view_787 = _shape_param_895 = None
        permute_428: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_788, [0, 2, 1, 3]);  view_788 = None
        mul_255: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_424, 0.4204482076268573);  permute_424 = None
        permute_429: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_426, [0, 1, 3, 2]);  permute_426 = None
        mul_256: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_429, 0.4204482076268573);  permute_429 = None
        expand_84: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_255, _shape_param_896);  mul_255 = _shape_param_896 = None
        clone_84: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_789: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_84, _shape_param_897);  clone_84 = _shape_param_897 = None
        expand_85: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_256, _shape_param_898);  mul_256 = _shape_param_898 = None
        clone_85: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_790: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_899);  clone_85 = _shape_param_899 = None
        bmm_42: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_789, view_790)
        view_791: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_900);  bmm_42 = _shape_param_900 = None
        convert_element_type_855: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_791, torch.float32)
        amax_21: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_855, [-1], True)
        sub_21: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_855, amax_21);  convert_element_type_855 = amax_21 = None
        exp_21: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_22: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_856: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        eq_21: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_791, -inf);  view_791 = None
        logical_not_42: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        where_21: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_43, full_1, convert_element_type_856);  logical_not_43 = convert_element_type_856 = None
        inductor_lookup_seed_21: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 21)
        inductor_random_21: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_901, inductor_lookup_seed_21, 'rand');  _shape_param_901 = inductor_lookup_seed_21 = None
        convert_element_type_857: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_21, torch.bfloat16);  inductor_random_21 = None
        gt_21: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_857, 0.1);  convert_element_type_857 = None
        mul_257: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_21, where_21)
        mul_258: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_257, 1.1111111111111112);  mul_257 = None
        expand_86: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_258, _shape_param_902);  mul_258 = _shape_param_902 = None
        view_792: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_86, _shape_param_903);  expand_86 = _shape_param_903 = None
        expand_87: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_428, _shape_param_904);  permute_428 = _shape_param_904 = None
        clone_86: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_793: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_86, _shape_param_905);  clone_86 = _shape_param_905 = None
        bmm_43: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_792, view_793)
        view_794: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_43, _shape_param_906);  bmm_43 = _shape_param_906 = None
        permute_430: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None
        clone_87: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_430, memory_format = torch.contiguous_format);  permute_430 = None
        view_795: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_87, _shape_param_907);  clone_87 = _shape_param_907 = None
        convert_element_type_858: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg989_1, torch.bfloat16);  arg989_1 = None
        convert_element_type_859: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg990_1, torch.bfloat16);  arg990_1 = None
        view_796: "bf16[32768, 128]" = torch.ops.aten.view.default(view_795, _shape_param_908);  view_795 = _shape_param_908 = None
        permute_431: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_859, [1, 0]);  convert_element_type_859 = None
        addmm_321: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_858, view_796, permute_431);  convert_element_type_858 = None
        view_797: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_321, _shape_param_909);  _shape_param_909 = None
        add_299: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_797, add_297);  view_797 = add_297 = None
        mul_259: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, arg991_1);  add_299 = arg991_1 = None
        add_300: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_259, arg992_1);  mul_259 = arg992_1 = None
        convert_element_type_860: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg993_1, torch.bfloat16);  arg993_1 = None
        convert_element_type_861: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg994_1, torch.bfloat16);  arg994_1 = None
        convert_element_type_862: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_300, torch.bfloat16)
        view_798: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_862, _shape_param_910);  convert_element_type_862 = _shape_param_910 = None
        permute_432: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_861, [1, 0]);  convert_element_type_861 = None
        addmm_322: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_860, view_798, permute_432);  convert_element_type_860 = None
        view_799: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_322, _shape_param_911);  addmm_322 = _shape_param_911 = None
        relu_84: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_799);  view_799 = None
        convert_element_type_863: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg995_1, torch.bfloat16);  arg995_1 = None
        convert_element_type_864: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg996_1, torch.bfloat16);  arg996_1 = None
        view_800: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_84, _shape_param_912);  _shape_param_912 = None
        permute_433: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_864, [1, 0]);  convert_element_type_864 = None
        addmm_323: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_863, view_800, permute_433);  convert_element_type_863 = None
        view_801: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_323, _shape_param_913);  addmm_323 = _shape_param_913 = None
        add_301: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_801, add_300);  view_801 = add_300 = None
        mul_260: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_301, arg997_1);  arg997_1 = None
        add_302: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_260, arg998_1);  mul_260 = arg998_1 = None
        convert_element_type_865: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg999_1, torch.bfloat16);  arg999_1 = None
        convert_element_type_866: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1000_1, torch.bfloat16);  arg1000_1 = None
        convert_element_type_867: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_302, torch.bfloat16)
        view_802: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_867, _shape_param_914);  convert_element_type_867 = _shape_param_914 = None
        permute_434: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_866, [1, 0]);  convert_element_type_866 = None
        addmm_324: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_865, view_802, permute_434);  convert_element_type_865 = None
        view_803: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_324, _shape_param_915);  addmm_324 = _shape_param_915 = None
        relu_85: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_803);  view_803 = None
        convert_element_type_868: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1001_1, torch.bfloat16);  arg1001_1 = None
        convert_element_type_869: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1002_1, torch.bfloat16);  arg1002_1 = None
        view_804: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_85, _shape_param_916);  _shape_param_916 = None
        permute_435: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_869, [1, 0]);  convert_element_type_869 = None
        addmm_325: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_868, view_804, permute_435);  convert_element_type_868 = None
        view_805: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_325, _shape_param_917);  addmm_325 = _shape_param_917 = None
        add_303: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_805, add_302);  view_805 = add_302 = None
        mul_261: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_303, arg1003_1);  arg1003_1 = None
        add_304: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_261, arg1004_1);  mul_261 = arg1004_1 = None
        convert_element_type_870: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1005_1, torch.bfloat16);  arg1005_1 = None
        convert_element_type_871: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1006_1, torch.bfloat16);  arg1006_1 = None
        convert_element_type_872: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_304, torch.bfloat16)
        view_806: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_872, _shape_param_918);  convert_element_type_872 = _shape_param_918 = None
        permute_436: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_871, [1, 0]);  convert_element_type_871 = None
        addmm_326: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_870, view_806, permute_436);  convert_element_type_870 = None
        view_807: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_326, _shape_param_919);  addmm_326 = _shape_param_919 = None
        relu_86: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_807);  view_807 = None
        convert_element_type_873: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1007_1, torch.bfloat16);  arg1007_1 = None
        convert_element_type_874: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1008_1, torch.bfloat16);  arg1008_1 = None
        view_808: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_86, _shape_param_920);  _shape_param_920 = None
        permute_437: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_874, [1, 0]);  convert_element_type_874 = None
        addmm_327: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_873, view_808, permute_437);  convert_element_type_873 = None
        view_809: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_327, _shape_param_921);  addmm_327 = _shape_param_921 = None
        add_305: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_809, add_304);  view_809 = add_304 = None
        mul_262: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_305, arg1009_1);  arg1009_1 = None
        add_306: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_262, arg1010_1);  mul_262 = arg1010_1 = None
        convert_element_type_875: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1011_1, torch.bfloat16);  arg1011_1 = None
        convert_element_type_876: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1012_1, torch.bfloat16);  arg1012_1 = None
        convert_element_type_877: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_306, torch.bfloat16)
        view_810: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_877, _shape_param_922);  convert_element_type_877 = _shape_param_922 = None
        permute_438: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_876, [1, 0]);  convert_element_type_876 = None
        addmm_328: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_875, view_810, permute_438);  convert_element_type_875 = None
        view_811: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_328, _shape_param_923);  addmm_328 = _shape_param_923 = None
        relu_87: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_811);  view_811 = None
        convert_element_type_878: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1013_1, torch.bfloat16);  arg1013_1 = None
        convert_element_type_879: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1014_1, torch.bfloat16);  arg1014_1 = None
        view_812: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_87, _shape_param_924);  _shape_param_924 = None
        permute_439: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_879, [1, 0]);  convert_element_type_879 = None
        addmm_329: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_878, view_812, permute_439);  convert_element_type_878 = None
        view_813: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_329, _shape_param_925);  addmm_329 = _shape_param_925 = None
        add_307: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_813, add_306);  view_813 = add_306 = None
        mul_263: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_307, arg1015_1);  arg1015_1 = None
        add_308: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_263, arg1016_1);  mul_263 = arg1016_1 = None
        convert_element_type_880: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1017_1, torch.bfloat16);  arg1017_1 = None
        convert_element_type_881: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1018_1, torch.bfloat16);  arg1018_1 = None
        convert_element_type_882: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_308, torch.bfloat16);  add_308 = None
        view_814: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_882, _shape_param_926);  convert_element_type_882 = _shape_param_926 = None
        permute_440: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_881, [1, 0]);  convert_element_type_881 = None
        addmm_330: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_880, view_814, permute_440);  convert_element_type_880 = None
        view_815: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_330, _shape_param_927);  addmm_330 = _shape_param_927 = None
        add_309: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_815, add_296);  view_815 = add_296 = None
        mul_264: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_309, arg1019_1);  arg1019_1 = None
        add_310: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_264, arg1020_1);  mul_264 = arg1020_1 = None
        convert_element_type_883: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1021_1, torch.bfloat16);  arg1021_1 = None
        convert_element_type_884: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1022_1, torch.bfloat16);  arg1022_1 = None
        convert_element_type_885: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_310, torch.bfloat16)
        view_816: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_885, _shape_param_928);  convert_element_type_885 = _shape_param_928 = None
        permute_441: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_884, [1, 0]);  convert_element_type_884 = None
        addmm_331: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_883, view_816, permute_441);  convert_element_type_883 = None
        view_817: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_331, _shape_param_929);  _shape_param_929 = None
        mul_265: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_817, arg1023_1);  view_817 = arg1023_1 = None
        add_311: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_265, arg1024_1);  mul_265 = arg1024_1 = None
        convert_element_type_886: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1025_1, torch.bfloat16);  arg1025_1 = None
        convert_element_type_887: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1026_1, torch.bfloat16);  arg1026_1 = None
        permute_442: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_887, [1, 0]);  convert_element_type_887 = None
        addmm_332: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_886, view_816, permute_442);  convert_element_type_886 = None
        view_818: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_332, _shape_param_930);  _shape_param_930 = None
        mul_266: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_818, arg1027_1);  view_818 = arg1027_1 = None
        add_312: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_266, arg1028_1);  mul_266 = arg1028_1 = None
        convert_element_type_888: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1029_1, torch.bfloat16);  arg1029_1 = None
        convert_element_type_889: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1030_1, torch.bfloat16);  arg1030_1 = None
        convert_element_type_890: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_312, torch.bfloat16);  add_312 = None
        view_819: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_890, _shape_param_931);  convert_element_type_890 = _shape_param_931 = None
        permute_443: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_889, [1, 0]);  convert_element_type_889 = None
        addmm_333: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_888, view_819, permute_443);  convert_element_type_888 = None
        view_820: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_333, _shape_param_932);  addmm_333 = _shape_param_932 = None
        view_821: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_820, _shape_param_933);  view_820 = _shape_param_933 = None
        permute_444: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_821, [0, 2, 1, 3]);  view_821 = None
        convert_element_type_891: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1031_1, torch.bfloat16);  arg1031_1 = None
        convert_element_type_892: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1032_1, torch.bfloat16);  arg1032_1 = None
        permute_445: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_892, [1, 0]);  convert_element_type_892 = None
        addmm_334: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_891, view_819, permute_445);  convert_element_type_891 = None
        view_822: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_334, _shape_param_934);  addmm_334 = _shape_param_934 = None
        view_823: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_822, _shape_param_935);  view_822 = _shape_param_935 = None
        permute_446: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_823, [0, 2, 1, 3]);  view_823 = None
        convert_element_type_893: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1033_1, torch.bfloat16);  arg1033_1 = None
        convert_element_type_894: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1034_1, torch.bfloat16);  arg1034_1 = None
        permute_447: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_894, [1, 0]);  convert_element_type_894 = None
        addmm_335: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_893, view_816, permute_447);  convert_element_type_893 = None
        view_824: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_335, _shape_param_936);  addmm_335 = _shape_param_936 = None
        view_825: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_824, _shape_param_937);  view_824 = _shape_param_937 = None
        permute_448: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_825, [0, 2, 1, 3]);  view_825 = None
        mul_267: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_444, 0.4204482076268573);  permute_444 = None
        permute_449: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_446, [0, 1, 3, 2]);  permute_446 = None
        mul_268: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_449, 0.4204482076268573);  permute_449 = None
        expand_88: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_267, _shape_param_938);  mul_267 = _shape_param_938 = None
        clone_88: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_826: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_88, _shape_param_939);  clone_88 = _shape_param_939 = None
        expand_89: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_268, _shape_param_940);  mul_268 = _shape_param_940 = None
        clone_89: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_827: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_941);  clone_89 = _shape_param_941 = None
        bmm_44: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_826, view_827)
        view_828: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_942);  bmm_44 = _shape_param_942 = None
        convert_element_type_895: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_828, torch.float32)
        amax_22: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_895, [-1], True)
        sub_22: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_895, amax_22);  convert_element_type_895 = amax_22 = None
        exp_22: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_23: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_896: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        eq_22: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_828, -inf);  view_828 = None
        logical_not_44: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        where_22: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_45, full_1, convert_element_type_896);  logical_not_45 = convert_element_type_896 = None
        inductor_lookup_seed_22: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 22)
        inductor_random_22: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_943, inductor_lookup_seed_22, 'rand');  _shape_param_943 = inductor_lookup_seed_22 = None
        convert_element_type_897: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_22, torch.bfloat16);  inductor_random_22 = None
        gt_22: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_897, 0.1);  convert_element_type_897 = None
        mul_269: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_22, where_22)
        mul_270: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_269, 1.1111111111111112);  mul_269 = None
        expand_90: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_270, _shape_param_944);  mul_270 = _shape_param_944 = None
        view_829: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_90, _shape_param_945);  expand_90 = _shape_param_945 = None
        expand_91: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_448, _shape_param_946);  permute_448 = _shape_param_946 = None
        clone_90: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_830: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_90, _shape_param_947);  clone_90 = _shape_param_947 = None
        bmm_45: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_829, view_830)
        view_831: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_45, _shape_param_948);  bmm_45 = _shape_param_948 = None
        permute_450: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_831, [0, 2, 1, 3]);  view_831 = None
        clone_91: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_450, memory_format = torch.contiguous_format);  permute_450 = None
        view_832: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_91, _shape_param_949);  clone_91 = _shape_param_949 = None
        convert_element_type_898: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1035_1, torch.bfloat16);  arg1035_1 = None
        convert_element_type_899: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1036_1, torch.bfloat16);  arg1036_1 = None
        view_833: "bf16[32768, 128]" = torch.ops.aten.view.default(view_832, _shape_param_950);  view_832 = _shape_param_950 = None
        permute_451: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_899, [1, 0]);  convert_element_type_899 = None
        addmm_336: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_898, view_833, permute_451);  convert_element_type_898 = None
        view_834: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_336, _shape_param_951);  _shape_param_951 = None
        add_313: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_834, add_311);  view_834 = add_311 = None
        mul_271: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_313, arg1037_1);  add_313 = arg1037_1 = None
        add_314: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_271, arg1038_1);  mul_271 = arg1038_1 = None
        convert_element_type_900: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1039_1, torch.bfloat16);  arg1039_1 = None
        convert_element_type_901: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1040_1, torch.bfloat16);  arg1040_1 = None
        convert_element_type_902: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_314, torch.bfloat16)
        view_835: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_902, _shape_param_952);  convert_element_type_902 = _shape_param_952 = None
        permute_452: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_901, [1, 0]);  convert_element_type_901 = None
        addmm_337: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_900, view_835, permute_452);  convert_element_type_900 = None
        view_836: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_337, _shape_param_953);  addmm_337 = _shape_param_953 = None
        relu_88: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_836);  view_836 = None
        convert_element_type_903: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1041_1, torch.bfloat16);  arg1041_1 = None
        convert_element_type_904: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1042_1, torch.bfloat16);  arg1042_1 = None
        view_837: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_88, _shape_param_954);  _shape_param_954 = None
        permute_453: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_904, [1, 0]);  convert_element_type_904 = None
        addmm_338: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_903, view_837, permute_453);  convert_element_type_903 = None
        view_838: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_338, _shape_param_955);  addmm_338 = _shape_param_955 = None
        add_315: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_838, add_314);  view_838 = add_314 = None
        mul_272: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_315, arg1043_1);  arg1043_1 = None
        add_316: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_272, arg1044_1);  mul_272 = arg1044_1 = None
        convert_element_type_905: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1045_1, torch.bfloat16);  arg1045_1 = None
        convert_element_type_906: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1046_1, torch.bfloat16);  arg1046_1 = None
        convert_element_type_907: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_316, torch.bfloat16)
        view_839: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_907, _shape_param_956);  convert_element_type_907 = _shape_param_956 = None
        permute_454: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_906, [1, 0]);  convert_element_type_906 = None
        addmm_339: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_905, view_839, permute_454);  convert_element_type_905 = None
        view_840: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_339, _shape_param_957);  addmm_339 = _shape_param_957 = None
        relu_89: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_840);  view_840 = None
        convert_element_type_908: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1047_1, torch.bfloat16);  arg1047_1 = None
        convert_element_type_909: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1048_1, torch.bfloat16);  arg1048_1 = None
        view_841: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_89, _shape_param_958);  _shape_param_958 = None
        permute_455: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_909, [1, 0]);  convert_element_type_909 = None
        addmm_340: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_908, view_841, permute_455);  convert_element_type_908 = None
        view_842: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_340, _shape_param_959);  addmm_340 = _shape_param_959 = None
        add_317: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_842, add_316);  view_842 = add_316 = None
        mul_273: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_317, arg1049_1);  arg1049_1 = None
        add_318: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_273, arg1050_1);  mul_273 = arg1050_1 = None
        convert_element_type_910: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1051_1, torch.bfloat16);  arg1051_1 = None
        convert_element_type_911: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1052_1, torch.bfloat16);  arg1052_1 = None
        convert_element_type_912: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_318, torch.bfloat16)
        view_843: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_912, _shape_param_960);  convert_element_type_912 = _shape_param_960 = None
        permute_456: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_911, [1, 0]);  convert_element_type_911 = None
        addmm_341: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_910, view_843, permute_456);  convert_element_type_910 = None
        view_844: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_341, _shape_param_961);  addmm_341 = _shape_param_961 = None
        relu_90: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_844);  view_844 = None
        convert_element_type_913: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1053_1, torch.bfloat16);  arg1053_1 = None
        convert_element_type_914: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1054_1, torch.bfloat16);  arg1054_1 = None
        view_845: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_90, _shape_param_962);  _shape_param_962 = None
        permute_457: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_914, [1, 0]);  convert_element_type_914 = None
        addmm_342: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_913, view_845, permute_457);  convert_element_type_913 = None
        view_846: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_342, _shape_param_963);  addmm_342 = _shape_param_963 = None
        add_319: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_846, add_318);  view_846 = add_318 = None
        mul_274: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_319, arg1055_1);  arg1055_1 = None
        add_320: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_274, arg1056_1);  mul_274 = arg1056_1 = None
        convert_element_type_915: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1057_1, torch.bfloat16);  arg1057_1 = None
        convert_element_type_916: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1058_1, torch.bfloat16);  arg1058_1 = None
        convert_element_type_917: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_320, torch.bfloat16)
        view_847: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_917, _shape_param_964);  convert_element_type_917 = _shape_param_964 = None
        permute_458: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_916, [1, 0]);  convert_element_type_916 = None
        addmm_343: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_915, view_847, permute_458);  convert_element_type_915 = None
        view_848: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_343, _shape_param_965);  addmm_343 = _shape_param_965 = None
        relu_91: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_848);  view_848 = None
        convert_element_type_918: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1059_1, torch.bfloat16);  arg1059_1 = None
        convert_element_type_919: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1060_1, torch.bfloat16);  arg1060_1 = None
        view_849: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_91, _shape_param_966);  _shape_param_966 = None
        permute_459: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_919, [1, 0]);  convert_element_type_919 = None
        addmm_344: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_918, view_849, permute_459);  convert_element_type_918 = None
        view_850: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_344, _shape_param_967);  addmm_344 = _shape_param_967 = None
        add_321: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_850, add_320);  view_850 = add_320 = None
        mul_275: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_321, arg1061_1);  arg1061_1 = None
        add_322: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_275, arg1062_1);  mul_275 = arg1062_1 = None
        convert_element_type_920: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1063_1, torch.bfloat16);  arg1063_1 = None
        convert_element_type_921: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1064_1, torch.bfloat16);  arg1064_1 = None
        convert_element_type_922: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_322, torch.bfloat16);  add_322 = None
        view_851: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_922, _shape_param_968);  convert_element_type_922 = _shape_param_968 = None
        permute_460: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_921, [1, 0]);  convert_element_type_921 = None
        addmm_345: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_920, view_851, permute_460);  convert_element_type_920 = None
        view_852: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_345, _shape_param_969);  addmm_345 = _shape_param_969 = None
        add_323: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_852, add_310);  view_852 = add_310 = None
        mul_276: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_323, arg1065_1);  arg1065_1 = None
        add_324: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_276, arg1066_1);  mul_276 = arg1066_1 = None
        convert_element_type_923: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1067_1, torch.bfloat16);  arg1067_1 = None
        convert_element_type_924: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1068_1, torch.bfloat16);  arg1068_1 = None
        convert_element_type_925: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_324, torch.bfloat16)
        view_853: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_925, _shape_param_970);  convert_element_type_925 = _shape_param_970 = None
        permute_461: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_924, [1, 0]);  convert_element_type_924 = None
        addmm_346: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_923, view_853, permute_461);  convert_element_type_923 = None
        view_854: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_346, _shape_param_971);  _shape_param_971 = None
        mul_277: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_854, arg1069_1);  view_854 = arg1069_1 = None
        add_325: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_277, arg1070_1);  mul_277 = arg1070_1 = None
        convert_element_type_926: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1071_1, torch.bfloat16);  arg1071_1 = None
        convert_element_type_927: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1072_1, torch.bfloat16);  arg1072_1 = None
        permute_462: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_927, [1, 0]);  convert_element_type_927 = None
        addmm_347: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_926, view_853, permute_462);  convert_element_type_926 = None
        view_855: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_347, _shape_param_972);  _shape_param_972 = None
        mul_278: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_855, arg1073_1);  view_855 = arg1073_1 = None
        add_326: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_278, arg1074_1);  mul_278 = arg1074_1 = None
        convert_element_type_928: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1075_1, torch.bfloat16);  arg1075_1 = None
        convert_element_type_929: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1076_1, torch.bfloat16);  arg1076_1 = None
        convert_element_type_930: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_326, torch.bfloat16);  add_326 = None
        view_856: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_930, _shape_param_973);  convert_element_type_930 = _shape_param_973 = None
        permute_463: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_929, [1, 0]);  convert_element_type_929 = None
        addmm_348: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_928, view_856, permute_463);  convert_element_type_928 = None
        view_857: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_348, _shape_param_974);  addmm_348 = _shape_param_974 = None
        view_858: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_857, _shape_param_975);  view_857 = _shape_param_975 = None
        permute_464: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None
        convert_element_type_931: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1077_1, torch.bfloat16);  arg1077_1 = None
        convert_element_type_932: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1078_1, torch.bfloat16);  arg1078_1 = None
        permute_465: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_932, [1, 0]);  convert_element_type_932 = None
        addmm_349: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_931, view_856, permute_465);  convert_element_type_931 = None
        view_859: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_349, _shape_param_976);  addmm_349 = _shape_param_976 = None
        view_860: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_859, _shape_param_977);  view_859 = _shape_param_977 = None
        permute_466: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_860, [0, 2, 1, 3]);  view_860 = None
        convert_element_type_933: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1079_1, torch.bfloat16);  arg1079_1 = None
        convert_element_type_934: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1080_1, torch.bfloat16);  arg1080_1 = None
        permute_467: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_934, [1, 0]);  convert_element_type_934 = None
        addmm_350: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_933, view_853, permute_467);  convert_element_type_933 = None
        view_861: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_350, _shape_param_978);  addmm_350 = _shape_param_978 = None
        view_862: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_861, _shape_param_979);  view_861 = _shape_param_979 = None
        permute_468: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_862, [0, 2, 1, 3]);  view_862 = None
        mul_279: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_464, 0.4204482076268573);  permute_464 = None
        permute_469: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_466, [0, 1, 3, 2]);  permute_466 = None
        mul_280: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_469, 0.4204482076268573);  permute_469 = None
        expand_92: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_279, _shape_param_980);  mul_279 = _shape_param_980 = None
        clone_92: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_863: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_92, _shape_param_981);  clone_92 = _shape_param_981 = None
        expand_93: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_280, _shape_param_982);  mul_280 = _shape_param_982 = None
        clone_93: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_864: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_983);  clone_93 = _shape_param_983 = None
        bmm_46: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_863, view_864)
        view_865: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_984);  bmm_46 = _shape_param_984 = None
        convert_element_type_935: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_865, torch.float32)
        amax_23: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_935, [-1], True)
        sub_23: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_935, amax_23);  convert_element_type_935 = amax_23 = None
        exp_23: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_24: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_936: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        eq_23: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_865, -inf);  view_865 = None
        logical_not_46: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        where_23: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_47, full_1, convert_element_type_936);  logical_not_47 = full_1 = convert_element_type_936 = None
        inductor_lookup_seed_23: "i64[]" = torch.ops.prims.inductor_lookup_seed.default(inductor_seeds, 23);  inductor_seeds = None
        inductor_random_23: "f32[256, 4, 128, 128]" = torch.ops.prims.inductor_random.default(_shape_param_985, inductor_lookup_seed_23, 'rand');  _shape_param_985 = inductor_lookup_seed_23 = None
        convert_element_type_937: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(inductor_random_23, torch.bfloat16);  inductor_random_23 = None
        gt_23: "b8[256, 4, 128, 128]" = torch.ops.aten.gt.Scalar(convert_element_type_937, 0.1);  convert_element_type_937 = None
        mul_281: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(gt_23, where_23)
        mul_282: "bf16[256, 4, 128, 128]" = torch.ops.aten.mul.Tensor(mul_281, 1.1111111111111112);  mul_281 = None
        expand_94: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(mul_282, _shape_param_986);  mul_282 = _shape_param_986 = None
        view_866: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_94, _shape_param_987);  expand_94 = _shape_param_987 = None
        expand_95: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_468, _shape_param_988);  permute_468 = _shape_param_988 = None
        clone_94: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_867: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_94, _shape_param_989);  clone_94 = _shape_param_989 = None
        bmm_47: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_866, view_867)
        view_868: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_47, _shape_param_990);  bmm_47 = _shape_param_990 = None
        permute_470: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_868, [0, 2, 1, 3]);  view_868 = None
        clone_95: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_470, memory_format = torch.contiguous_format);  permute_470 = None
        view_869: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_95, _shape_param_991);  clone_95 = _shape_param_991 = None
        convert_element_type_938: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1081_1, torch.bfloat16);  arg1081_1 = None
        convert_element_type_939: "bf16[128, 128]" = torch.ops.prims.convert_element_type.default(arg1082_1, torch.bfloat16);  arg1082_1 = None
        view_870: "bf16[32768, 128]" = torch.ops.aten.view.default(view_869, _shape_param_992);  view_869 = _shape_param_992 = None
        permute_471: "bf16[128, 128]" = torch.ops.aten.permute.default(convert_element_type_939, [1, 0]);  convert_element_type_939 = None
        addmm_351: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_938, view_870, permute_471);  convert_element_type_938 = None
        view_871: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_351, _shape_param_993);  _shape_param_993 = None
        add_327: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_325);  view_871 = add_325 = None
        mul_283: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, arg1083_1);  add_327 = arg1083_1 = None
        add_328: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_283, arg1084_1);  mul_283 = arg1084_1 = None
        convert_element_type_940: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1085_1, torch.bfloat16);  arg1085_1 = None
        convert_element_type_941: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1086_1, torch.bfloat16);  arg1086_1 = None
        convert_element_type_942: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_328, torch.bfloat16)
        view_872: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_942, _shape_param_994);  convert_element_type_942 = _shape_param_994 = None
        permute_472: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_941, [1, 0]);  convert_element_type_941 = None
        addmm_352: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_940, view_872, permute_472);  convert_element_type_940 = None
        view_873: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_352, _shape_param_995);  addmm_352 = _shape_param_995 = None
        relu_92: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None
        convert_element_type_943: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1087_1, torch.bfloat16);  arg1087_1 = None
        convert_element_type_944: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1088_1, torch.bfloat16);  arg1088_1 = None
        view_874: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_92, _shape_param_996);  _shape_param_996 = None
        permute_473: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_944, [1, 0]);  convert_element_type_944 = None
        addmm_353: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_943, view_874, permute_473);  convert_element_type_943 = None
        view_875: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_353, _shape_param_997);  addmm_353 = _shape_param_997 = None
        add_329: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_328);  view_875 = add_328 = None
        mul_284: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, arg1089_1);  arg1089_1 = None
        add_330: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_284, arg1090_1);  mul_284 = arg1090_1 = None
        convert_element_type_945: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1091_1, torch.bfloat16);  arg1091_1 = None
        convert_element_type_946: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1092_1, torch.bfloat16);  arg1092_1 = None
        convert_element_type_947: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_330, torch.bfloat16)
        view_876: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_947, _shape_param_998);  convert_element_type_947 = _shape_param_998 = None
        permute_474: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_946, [1, 0]);  convert_element_type_946 = None
        addmm_354: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_945, view_876, permute_474);  convert_element_type_945 = None
        view_877: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_354, _shape_param_999);  addmm_354 = _shape_param_999 = None
        relu_93: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None
        convert_element_type_948: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1093_1, torch.bfloat16);  arg1093_1 = None
        convert_element_type_949: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1094_1, torch.bfloat16);  arg1094_1 = None
        view_878: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_93, _shape_param_1000);  _shape_param_1000 = None
        permute_475: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_949, [1, 0]);  convert_element_type_949 = None
        addmm_355: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_948, view_878, permute_475);  convert_element_type_948 = None
        view_879: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_355, _shape_param_1001);  addmm_355 = _shape_param_1001 = None
        add_331: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_330);  view_879 = add_330 = None
        mul_285: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_331, arg1095_1);  arg1095_1 = None
        add_332: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_285, arg1096_1);  mul_285 = arg1096_1 = None
        convert_element_type_950: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1097_1, torch.bfloat16);  arg1097_1 = None
        convert_element_type_951: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1098_1, torch.bfloat16);  arg1098_1 = None
        convert_element_type_952: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_332, torch.bfloat16)
        view_880: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_952, _shape_param_1002);  convert_element_type_952 = _shape_param_1002 = None
        permute_476: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_951, [1, 0]);  convert_element_type_951 = None
        addmm_356: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_950, view_880, permute_476);  convert_element_type_950 = None
        view_881: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_356, _shape_param_1003);  addmm_356 = _shape_param_1003 = None
        relu_94: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_881);  view_881 = None
        convert_element_type_953: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1099_1, torch.bfloat16);  arg1099_1 = None
        convert_element_type_954: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1100_1, torch.bfloat16);  arg1100_1 = None
        view_882: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_94, _shape_param_1004);  _shape_param_1004 = None
        permute_477: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_954, [1, 0]);  convert_element_type_954 = None
        addmm_357: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_953, view_882, permute_477);  convert_element_type_953 = None
        view_883: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_357, _shape_param_1005);  addmm_357 = _shape_param_1005 = None
        add_333: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_883, add_332);  view_883 = add_332 = None
        mul_286: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_333, arg1101_1);  arg1101_1 = None
        add_334: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_286, arg1102_1);  mul_286 = arg1102_1 = None
        convert_element_type_955: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1103_1, torch.bfloat16);  arg1103_1 = None
        convert_element_type_956: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1104_1, torch.bfloat16);  arg1104_1 = None
        convert_element_type_957: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_334, torch.bfloat16)
        view_884: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_957, _shape_param_1006);  convert_element_type_957 = _shape_param_1006 = None
        permute_478: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_956, [1, 0]);  convert_element_type_956 = None
        addmm_358: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_955, view_884, permute_478);  convert_element_type_955 = None
        view_885: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_358, _shape_param_1007);  addmm_358 = _shape_param_1007 = None
        relu_95: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_885);  view_885 = None
        convert_element_type_958: "bf16[128]" = torch.ops.prims.convert_element_type.default(arg1105_1, torch.bfloat16);  arg1105_1 = None
        convert_element_type_959: "bf16[128, 512]" = torch.ops.prims.convert_element_type.default(arg1106_1, torch.bfloat16);  arg1106_1 = None
        view_886: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_95, _shape_param_1008);  _shape_param_1008 = None
        permute_479: "bf16[512, 128]" = torch.ops.aten.permute.default(convert_element_type_959, [1, 0]);  convert_element_type_959 = None
        addmm_359: "bf16[32768, 128]" = torch.ops.aten.addmm.default(convert_element_type_958, view_886, permute_479);  convert_element_type_958 = None
        view_887: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_359, _shape_param_1009);  addmm_359 = _shape_param_1009 = None
        add_335: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(view_887, add_334);  view_887 = add_334 = None
        mul_287: "f32[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_335, arg1107_1);  arg1107_1 = None
        add_336: "f32[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_287, arg1108_1);  mul_287 = arg1108_1 = None
        convert_element_type_960: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1109_1, torch.bfloat16);  arg1109_1 = None
        convert_element_type_961: "bf16[512, 128]" = torch.ops.prims.convert_element_type.default(arg1110_1, torch.bfloat16);  arg1110_1 = None
        convert_element_type_962: "bf16[256, 128, 128]" = torch.ops.prims.convert_element_type.default(add_336, torch.bfloat16);  add_336 = None
        view_888: "bf16[32768, 128]" = torch.ops.aten.view.default(convert_element_type_962, _shape_param_1010);  convert_element_type_962 = _shape_param_1010 = None
        permute_480: "bf16[128, 512]" = torch.ops.aten.permute.default(convert_element_type_961, [1, 0]);  convert_element_type_961 = None
        addmm_360: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_960, view_888, permute_480);  convert_element_type_960 = None
        view_889: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_360, _shape_param_1011);  addmm_360 = _shape_param_1011 = None
        add_337: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(view_889, add_324);  view_889 = add_324 = None
        mul_288: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_337, arg1111_1);  arg1111_1 = None
        add_338: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_288, arg1112_1);  mul_288 = arg1112_1 = None
        convert_element_type_963: "bf16[512]" = torch.ops.prims.convert_element_type.default(arg1113_1, torch.bfloat16);  arg1113_1 = None
        convert_element_type_964: "bf16[512, 512]" = torch.ops.prims.convert_element_type.default(arg1114_1, torch.bfloat16);  arg1114_1 = None
        convert_element_type_965: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_338, torch.bfloat16);  add_338 = None
        view_890: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_965, _shape_param_1012);  convert_element_type_965 = _shape_param_1012 = None
        permute_481: "bf16[512, 512]" = torch.ops.aten.permute.default(convert_element_type_964, [1, 0]);  convert_element_type_964 = None
        addmm_361: "bf16[32768, 512]" = torch.ops.aten.addmm.default(convert_element_type_963, view_890, permute_481);  convert_element_type_963 = None
        view_891: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_361, _shape_param_1013);  _shape_param_1013 = None
        relu_96: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_891);  view_891 = None
        convert_element_type_966: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(relu_96, torch.float32);  relu_96 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_966, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None
        add_339: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
        sub_24: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_966, getitem_1);  convert_element_type_966 = None
        mul_289: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt);  sub_24 = None
        mul_290: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_289, arg1115_1);  mul_289 = arg1115_1 = None
        add_340: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_290, arg1116_1);  mul_290 = arg1116_1 = None
        permute_482: "f32[128, 30522]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        cat_1: "f32[512, 30522]" = torch.ops.aten.cat.default([permute_482, arg1117_1]);  permute_482 = arg1117_1 = None
        convert_element_type_967: "bf16[512, 30522]" = torch.ops.prims.convert_element_type.default(cat_1, torch.bfloat16);  cat_1 = None
        convert_element_type_968: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_340, torch.bfloat16);  add_340 = None
        view_892: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_968, _shape_param_1014);  convert_element_type_968 = _shape_param_1014 = None
        constant_pad_nd_2: "bf16[512, 30528]" = torch.ops.aten.constant_pad_nd.default(convert_element_type_967, _shape_param_1015);  _shape_param_1015 = None
        mm: "bf16[32768, 30528]" = torch.ops.aten.mm.default(view_892, constant_pad_nd_2);  constant_pad_nd_2 = None
        slice_4: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        view_893: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_4, _shape_param_1016);  slice_4 = _shape_param_1016 = None
        add_341: "f32[256, 128, 30522]" = torch.ops.aten.add.Tensor(view_893, arg1118_1);  view_893 = arg1118_1 = None
        convert_element_type_969: "bf16[256, 128, 30522]" = torch.ops.prims.convert_element_type.default(add_341, torch.bfloat16);  add_341 = None
        view_894: "bf16[32768, 30522]" = torch.ops.aten.view.default(convert_element_type_969, _shape_param_1017);  convert_element_type_969 = _shape_param_1017 = None
        view_895: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(view_894, _shape_param_1018);  _shape_param_1018 = None
        view_896: "i64[32768]" = torch.ops.aten.view.default(arg1119_1, [-1]);  arg1119_1 = None
        convert_element_type_970: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_894, torch.float32);  view_894 = None
        amax_24: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type_970, [1], True)
        sub_25: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_970, amax_24);  convert_element_type_970 = amax_24 = None
        exp_24: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_25)
        sum_25: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_26: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_971: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        convert_element_type_972: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(convert_element_type_971, torch.float32)
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_896, -100)
        full_2: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[32768]" = torch.ops.aten.where.self(ne, view_896, full_2);  view_896 = full_2 = None
        unsqueeze: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "f32[32768, 1]" = torch.ops.aten.gather.default(convert_element_type_972, 1, unsqueeze);  convert_element_type_972 = unsqueeze = None
        squeeze: "f32[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_3: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[32768]" = torch.ops.aten.where.self(ne, neg, full_3);  neg = full_3 = None
        sum_26: "i64[]" = torch.ops.aten.sum.default(ne);  ne = None
        convert_element_type_973: "f32[]" = torch.ops.prims.convert_element_type.default(sum_26, torch.float32);  sum_26 = None
        sum_27: "f32[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        div_24: "f32[]" = torch.ops.aten.div.Tensor(sum_27, convert_element_type_973);  sum_27 = None
        permute_483: "bf16[512, 32768]" = torch.ops.aten.permute.default(view_892, [1, 0]);  view_892 = None
        permute_484: "bf16[30522, 512]" = torch.ops.aten.permute.default(convert_element_type_967, [1, 0]);  convert_element_type_967 = None
        permute_485: "bf16[512, 512]" = torch.ops.aten.permute.default(permute_481, [1, 0]);  permute_481 = None
        permute_486: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_480, [1, 0]);  permute_480 = None
        permute_487: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_479, [1, 0]);  permute_479 = None
        le: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_95, 0);  relu_95 = None
        permute_488: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_478, [1, 0]);  permute_478 = None
        permute_489: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_477, [1, 0]);  permute_477 = None
        le_1: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_94, 0);  relu_94 = None
        permute_490: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_476, [1, 0]);  permute_476 = None
        permute_491: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_475, [1, 0]);  permute_475 = None
        le_2: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_93, 0);  relu_93 = None
        permute_492: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_474, [1, 0]);  permute_474 = None
        permute_493: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_473, [1, 0]);  permute_473 = None
        le_3: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_92, 0);  relu_92 = None
        permute_494: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_472, [1, 0]);  permute_472 = None
        permute_495: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_471, [1, 0]);  permute_471 = None
        permute_496: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_866, [0, 2, 1]);  view_866 = None
        permute_497: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_867, [0, 2, 1]);  view_867 = None
        permute_498: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_863, [0, 2, 1]);  view_863 = None
        permute_499: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_864, [0, 2, 1]);  view_864 = None
        permute_500: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_467, [1, 0]);  permute_467 = None
        permute_501: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_465, [1, 0]);  permute_465 = None
        permute_502: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_463, [1, 0]);  permute_463 = None
        permute_503: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_462, [1, 0]);  permute_462 = None
        permute_504: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_461, [1, 0]);  permute_461 = None
        permute_505: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_460, [1, 0]);  permute_460 = None
        permute_506: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_459, [1, 0]);  permute_459 = None
        le_4: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_91, 0);  relu_91 = None
        permute_507: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_458, [1, 0]);  permute_458 = None
        permute_508: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_457, [1, 0]);  permute_457 = None
        le_5: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_90, 0);  relu_90 = None
        permute_509: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_456, [1, 0]);  permute_456 = None
        permute_510: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_455, [1, 0]);  permute_455 = None
        le_6: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_89, 0);  relu_89 = None
        permute_511: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_454, [1, 0]);  permute_454 = None
        permute_512: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_453, [1, 0]);  permute_453 = None
        le_7: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_88, 0);  relu_88 = None
        permute_513: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_452, [1, 0]);  permute_452 = None
        permute_514: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_451, [1, 0]);  permute_451 = None
        permute_515: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_829, [0, 2, 1]);  view_829 = None
        permute_516: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_830, [0, 2, 1]);  view_830 = None
        permute_517: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_826, [0, 2, 1]);  view_826 = None
        permute_518: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_827, [0, 2, 1]);  view_827 = None
        permute_519: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_447, [1, 0]);  permute_447 = None
        permute_520: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_445, [1, 0]);  permute_445 = None
        permute_521: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_443, [1, 0]);  permute_443 = None
        permute_522: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_442, [1, 0]);  permute_442 = None
        permute_523: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_441, [1, 0]);  permute_441 = None
        permute_524: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_440, [1, 0]);  permute_440 = None
        permute_525: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_439, [1, 0]);  permute_439 = None
        le_8: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_87, 0);  relu_87 = None
        permute_526: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_438, [1, 0]);  permute_438 = None
        permute_527: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_437, [1, 0]);  permute_437 = None
        le_9: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_86, 0);  relu_86 = None
        permute_528: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_436, [1, 0]);  permute_436 = None
        permute_529: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_435, [1, 0]);  permute_435 = None
        le_10: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_85, 0);  relu_85 = None
        permute_530: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_434, [1, 0]);  permute_434 = None
        permute_531: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_433, [1, 0]);  permute_433 = None
        le_11: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_84, 0);  relu_84 = None
        permute_532: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_432, [1, 0]);  permute_432 = None
        permute_533: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_431, [1, 0]);  permute_431 = None
        permute_534: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_792, [0, 2, 1]);  view_792 = None
        permute_535: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_793, [0, 2, 1]);  view_793 = None
        permute_536: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_789, [0, 2, 1]);  view_789 = None
        permute_537: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_790, [0, 2, 1]);  view_790 = None
        permute_538: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_427, [1, 0]);  permute_427 = None
        permute_539: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_425, [1, 0]);  permute_425 = None
        permute_540: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_423, [1, 0]);  permute_423 = None
        permute_541: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_422, [1, 0]);  permute_422 = None
        permute_542: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_421, [1, 0]);  permute_421 = None
        permute_543: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_420, [1, 0]);  permute_420 = None
        permute_544: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_419, [1, 0]);  permute_419 = None
        le_12: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_83, 0);  relu_83 = None
        permute_545: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_418, [1, 0]);  permute_418 = None
        permute_546: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_417, [1, 0]);  permute_417 = None
        le_13: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_82, 0);  relu_82 = None
        permute_547: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_416, [1, 0]);  permute_416 = None
        permute_548: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_415, [1, 0]);  permute_415 = None
        le_14: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_81, 0);  relu_81 = None
        permute_549: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_414, [1, 0]);  permute_414 = None
        permute_550: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_413, [1, 0]);  permute_413 = None
        le_15: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_80, 0);  relu_80 = None
        permute_551: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_412, [1, 0]);  permute_412 = None
        permute_552: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_411, [1, 0]);  permute_411 = None
        permute_553: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_755, [0, 2, 1]);  view_755 = None
        permute_554: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_756, [0, 2, 1]);  view_756 = None
        permute_555: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_752, [0, 2, 1]);  view_752 = None
        permute_556: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_753, [0, 2, 1]);  view_753 = None
        permute_557: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_407, [1, 0]);  permute_407 = None
        permute_558: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_405, [1, 0]);  permute_405 = None
        permute_559: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_403, [1, 0]);  permute_403 = None
        permute_560: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_402, [1, 0]);  permute_402 = None
        permute_561: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_401, [1, 0]);  permute_401 = None
        permute_562: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_400, [1, 0]);  permute_400 = None
        permute_563: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_399, [1, 0]);  permute_399 = None
        le_16: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_79, 0);  relu_79 = None
        permute_564: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_398, [1, 0]);  permute_398 = None
        permute_565: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_397, [1, 0]);  permute_397 = None
        le_17: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_78, 0);  relu_78 = None
        permute_566: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_396, [1, 0]);  permute_396 = None
        permute_567: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_395, [1, 0]);  permute_395 = None
        le_18: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_77, 0);  relu_77 = None
        permute_568: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_394, [1, 0]);  permute_394 = None
        permute_569: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_393, [1, 0]);  permute_393 = None
        le_19: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_76, 0);  relu_76 = None
        permute_570: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_392, [1, 0]);  permute_392 = None
        permute_571: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_391, [1, 0]);  permute_391 = None
        permute_572: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_718, [0, 2, 1]);  view_718 = None
        permute_573: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_719, [0, 2, 1]);  view_719 = None
        permute_574: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_715, [0, 2, 1]);  view_715 = None
        permute_575: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_716, [0, 2, 1]);  view_716 = None
        permute_576: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_387, [1, 0]);  permute_387 = None
        permute_577: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_385, [1, 0]);  permute_385 = None
        permute_578: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_383, [1, 0]);  permute_383 = None
        permute_579: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_382, [1, 0]);  permute_382 = None
        permute_580: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_381, [1, 0]);  permute_381 = None
        permute_581: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_380, [1, 0]);  permute_380 = None
        permute_582: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_379, [1, 0]);  permute_379 = None
        le_20: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_75, 0);  relu_75 = None
        permute_583: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_378, [1, 0]);  permute_378 = None
        permute_584: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_377, [1, 0]);  permute_377 = None
        le_21: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_74, 0);  relu_74 = None
        permute_585: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_376, [1, 0]);  permute_376 = None
        permute_586: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_375, [1, 0]);  permute_375 = None
        le_22: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_73, 0);  relu_73 = None
        permute_587: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_374, [1, 0]);  permute_374 = None
        permute_588: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_373, [1, 0]);  permute_373 = None
        le_23: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_72, 0);  relu_72 = None
        permute_589: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_372, [1, 0]);  permute_372 = None
        permute_590: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_371, [1, 0]);  permute_371 = None
        permute_591: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_681, [0, 2, 1]);  view_681 = None
        permute_592: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_682, [0, 2, 1]);  view_682 = None
        permute_593: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_678, [0, 2, 1]);  view_678 = None
        permute_594: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_679, [0, 2, 1]);  view_679 = None
        permute_595: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_367, [1, 0]);  permute_367 = None
        permute_596: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_365, [1, 0]);  permute_365 = None
        permute_597: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_363, [1, 0]);  permute_363 = None
        permute_598: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_362, [1, 0]);  permute_362 = None
        permute_599: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_361, [1, 0]);  permute_361 = None
        permute_600: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_360, [1, 0]);  permute_360 = None
        permute_601: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_359, [1, 0]);  permute_359 = None
        le_24: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_71, 0);  relu_71 = None
        permute_602: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_358, [1, 0]);  permute_358 = None
        permute_603: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_357, [1, 0]);  permute_357 = None
        le_25: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_70, 0);  relu_70 = None
        permute_604: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_356, [1, 0]);  permute_356 = None
        permute_605: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_355, [1, 0]);  permute_355 = None
        le_26: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_69, 0);  relu_69 = None
        permute_606: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_354, [1, 0]);  permute_354 = None
        permute_607: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_353, [1, 0]);  permute_353 = None
        le_27: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_68, 0);  relu_68 = None
        permute_608: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_352, [1, 0]);  permute_352 = None
        permute_609: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_351, [1, 0]);  permute_351 = None
        permute_610: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_644, [0, 2, 1]);  view_644 = None
        permute_611: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_645, [0, 2, 1]);  view_645 = None
        permute_612: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_641, [0, 2, 1]);  view_641 = None
        permute_613: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_642, [0, 2, 1]);  view_642 = None
        permute_614: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_347, [1, 0]);  permute_347 = None
        permute_615: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_345, [1, 0]);  permute_345 = None
        permute_616: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_343, [1, 0]);  permute_343 = None
        permute_617: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_342, [1, 0]);  permute_342 = None
        permute_618: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_341, [1, 0]);  permute_341 = None
        permute_619: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_340, [1, 0]);  permute_340 = None
        permute_620: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_339, [1, 0]);  permute_339 = None
        le_28: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_67, 0);  relu_67 = None
        permute_621: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_338, [1, 0]);  permute_338 = None
        permute_622: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_337, [1, 0]);  permute_337 = None
        le_29: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_66, 0);  relu_66 = None
        permute_623: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_336, [1, 0]);  permute_336 = None
        permute_624: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_335, [1, 0]);  permute_335 = None
        le_30: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_65, 0);  relu_65 = None
        permute_625: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_334, [1, 0]);  permute_334 = None
        permute_626: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_333, [1, 0]);  permute_333 = None
        le_31: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_64, 0);  relu_64 = None
        permute_627: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_332, [1, 0]);  permute_332 = None
        permute_628: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_331, [1, 0]);  permute_331 = None
        permute_629: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_607, [0, 2, 1]);  view_607 = None
        permute_630: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_608, [0, 2, 1]);  view_608 = None
        permute_631: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_604, [0, 2, 1]);  view_604 = None
        permute_632: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_605, [0, 2, 1]);  view_605 = None
        permute_633: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_327, [1, 0]);  permute_327 = None
        permute_634: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_325, [1, 0]);  permute_325 = None
        permute_635: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_323, [1, 0]);  permute_323 = None
        permute_636: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_322, [1, 0]);  permute_322 = None
        permute_637: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_321, [1, 0]);  permute_321 = None
        permute_638: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_320, [1, 0]);  permute_320 = None
        permute_639: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_319, [1, 0]);  permute_319 = None
        le_32: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_63, 0);  relu_63 = None
        permute_640: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_318, [1, 0]);  permute_318 = None
        permute_641: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_317, [1, 0]);  permute_317 = None
        le_33: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_62, 0);  relu_62 = None
        permute_642: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_316, [1, 0]);  permute_316 = None
        permute_643: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_315, [1, 0]);  permute_315 = None
        le_34: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_61, 0);  relu_61 = None
        permute_644: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_314, [1, 0]);  permute_314 = None
        permute_645: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_313, [1, 0]);  permute_313 = None
        le_35: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_60, 0);  relu_60 = None
        permute_646: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_312, [1, 0]);  permute_312 = None
        permute_647: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_311, [1, 0]);  permute_311 = None
        permute_648: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_570, [0, 2, 1]);  view_570 = None
        permute_649: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_571, [0, 2, 1]);  view_571 = None
        permute_650: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_567, [0, 2, 1]);  view_567 = None
        permute_651: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_568, [0, 2, 1]);  view_568 = None
        permute_652: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_307, [1, 0]);  permute_307 = None
        permute_653: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_305, [1, 0]);  permute_305 = None
        permute_654: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_303, [1, 0]);  permute_303 = None
        permute_655: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_302, [1, 0]);  permute_302 = None
        permute_656: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_301, [1, 0]);  permute_301 = None
        permute_657: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_300, [1, 0]);  permute_300 = None
        permute_658: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_299, [1, 0]);  permute_299 = None
        le_36: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_59, 0);  relu_59 = None
        permute_659: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_298, [1, 0]);  permute_298 = None
        permute_660: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_297, [1, 0]);  permute_297 = None
        le_37: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_58, 0);  relu_58 = None
        permute_661: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_296, [1, 0]);  permute_296 = None
        permute_662: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_295, [1, 0]);  permute_295 = None
        le_38: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_57, 0);  relu_57 = None
        permute_663: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_294, [1, 0]);  permute_294 = None
        permute_664: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_293, [1, 0]);  permute_293 = None
        le_39: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_56, 0);  relu_56 = None
        permute_665: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_292, [1, 0]);  permute_292 = None
        permute_666: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_291, [1, 0]);  permute_291 = None
        permute_667: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_533, [0, 2, 1]);  view_533 = None
        permute_668: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_534, [0, 2, 1]);  view_534 = None
        permute_669: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_530, [0, 2, 1]);  view_530 = None
        permute_670: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_531, [0, 2, 1]);  view_531 = None
        permute_671: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_287, [1, 0]);  permute_287 = None
        permute_672: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_285, [1, 0]);  permute_285 = None
        permute_673: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_283, [1, 0]);  permute_283 = None
        permute_674: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_282, [1, 0]);  permute_282 = None
        permute_675: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_281, [1, 0]);  permute_281 = None
        permute_676: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_280, [1, 0]);  permute_280 = None
        permute_677: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_279, [1, 0]);  permute_279 = None
        le_40: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_55, 0);  relu_55 = None
        permute_678: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_278, [1, 0]);  permute_278 = None
        permute_679: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_277, [1, 0]);  permute_277 = None
        le_41: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_54, 0);  relu_54 = None
        permute_680: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_276, [1, 0]);  permute_276 = None
        permute_681: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_275, [1, 0]);  permute_275 = None
        le_42: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_53, 0);  relu_53 = None
        permute_682: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_274, [1, 0]);  permute_274 = None
        permute_683: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_273, [1, 0]);  permute_273 = None
        le_43: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_52, 0);  relu_52 = None
        permute_684: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_272, [1, 0]);  permute_272 = None
        permute_685: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_271, [1, 0]);  permute_271 = None
        permute_686: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_496, [0, 2, 1]);  view_496 = None
        permute_687: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_497, [0, 2, 1]);  view_497 = None
        permute_688: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_493, [0, 2, 1]);  view_493 = None
        permute_689: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_494, [0, 2, 1]);  view_494 = None
        permute_690: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_267, [1, 0]);  permute_267 = None
        permute_691: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_265, [1, 0]);  permute_265 = None
        permute_692: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_263, [1, 0]);  permute_263 = None
        permute_693: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_262, [1, 0]);  permute_262 = None
        permute_694: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_261, [1, 0]);  permute_261 = None
        permute_695: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_260, [1, 0]);  permute_260 = None
        permute_696: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_259, [1, 0]);  permute_259 = None
        le_44: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_51, 0);  relu_51 = None
        permute_697: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_258, [1, 0]);  permute_258 = None
        permute_698: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_257, [1, 0]);  permute_257 = None
        le_45: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_50, 0);  relu_50 = None
        permute_699: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_256, [1, 0]);  permute_256 = None
        permute_700: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_255, [1, 0]);  permute_255 = None
        le_46: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_49, 0);  relu_49 = None
        permute_701: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_254, [1, 0]);  permute_254 = None
        permute_702: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_253, [1, 0]);  permute_253 = None
        le_47: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_48, 0);  relu_48 = None
        permute_703: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_252, [1, 0]);  permute_252 = None
        permute_704: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_251, [1, 0]);  permute_251 = None
        permute_705: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_459, [0, 2, 1]);  view_459 = None
        permute_706: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_460, [0, 2, 1]);  view_460 = None
        permute_707: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_456, [0, 2, 1]);  view_456 = None
        permute_708: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_457, [0, 2, 1]);  view_457 = None
        permute_709: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_247, [1, 0]);  permute_247 = None
        permute_710: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_245, [1, 0]);  permute_245 = None
        permute_711: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_243, [1, 0]);  permute_243 = None
        permute_712: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_242, [1, 0]);  permute_242 = None
        permute_713: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_241, [1, 0]);  permute_241 = None
        permute_714: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_240, [1, 0]);  permute_240 = None
        permute_715: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_239, [1, 0]);  permute_239 = None
        le_48: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_47, 0);  relu_47 = None
        permute_716: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_238, [1, 0]);  permute_238 = None
        permute_717: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_237, [1, 0]);  permute_237 = None
        le_49: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_46, 0);  relu_46 = None
        permute_718: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_236, [1, 0]);  permute_236 = None
        permute_719: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_235, [1, 0]);  permute_235 = None
        le_50: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_45, 0);  relu_45 = None
        permute_720: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_234, [1, 0]);  permute_234 = None
        permute_721: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_233, [1, 0]);  permute_233 = None
        le_51: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_44, 0);  relu_44 = None
        permute_722: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_232, [1, 0]);  permute_232 = None
        permute_723: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_231, [1, 0]);  permute_231 = None
        permute_724: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_422, [0, 2, 1]);  view_422 = None
        permute_725: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_423, [0, 2, 1]);  view_423 = None
        permute_726: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_419, [0, 2, 1]);  view_419 = None
        permute_727: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_420, [0, 2, 1]);  view_420 = None
        permute_728: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_227, [1, 0]);  permute_227 = None
        permute_729: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_225, [1, 0]);  permute_225 = None
        permute_730: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_223, [1, 0]);  permute_223 = None
        permute_731: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_222, [1, 0]);  permute_222 = None
        permute_732: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_221, [1, 0]);  permute_221 = None
        permute_733: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_220, [1, 0]);  permute_220 = None
        permute_734: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_219, [1, 0]);  permute_219 = None
        le_52: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_43, 0);  relu_43 = None
        permute_735: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_218, [1, 0]);  permute_218 = None
        permute_736: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_217, [1, 0]);  permute_217 = None
        le_53: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_42, 0);  relu_42 = None
        permute_737: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_216, [1, 0]);  permute_216 = None
        permute_738: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_215, [1, 0]);  permute_215 = None
        le_54: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_41, 0);  relu_41 = None
        permute_739: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_214, [1, 0]);  permute_214 = None
        permute_740: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_213, [1, 0]);  permute_213 = None
        le_55: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_40, 0);  relu_40 = None
        permute_741: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_212, [1, 0]);  permute_212 = None
        permute_742: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_211, [1, 0]);  permute_211 = None
        permute_743: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_385, [0, 2, 1]);  view_385 = None
        permute_744: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_386, [0, 2, 1]);  view_386 = None
        permute_745: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_382, [0, 2, 1]);  view_382 = None
        permute_746: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_383, [0, 2, 1]);  view_383 = None
        permute_747: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_207, [1, 0]);  permute_207 = None
        permute_748: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_205, [1, 0]);  permute_205 = None
        permute_749: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_203, [1, 0]);  permute_203 = None
        permute_750: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_202, [1, 0]);  permute_202 = None
        permute_751: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_201, [1, 0]);  permute_201 = None
        permute_752: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_200, [1, 0]);  permute_200 = None
        permute_753: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_199, [1, 0]);  permute_199 = None
        le_56: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_39, 0);  relu_39 = None
        permute_754: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_198, [1, 0]);  permute_198 = None
        permute_755: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_197, [1, 0]);  permute_197 = None
        le_57: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_38, 0);  relu_38 = None
        permute_756: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_196, [1, 0]);  permute_196 = None
        permute_757: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_195, [1, 0]);  permute_195 = None
        le_58: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_37, 0);  relu_37 = None
        permute_758: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_194, [1, 0]);  permute_194 = None
        permute_759: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_193, [1, 0]);  permute_193 = None
        le_59: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_36, 0);  relu_36 = None
        permute_760: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_192, [1, 0]);  permute_192 = None
        permute_761: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_191, [1, 0]);  permute_191 = None
        permute_762: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_348, [0, 2, 1]);  view_348 = None
        permute_763: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_349, [0, 2, 1]);  view_349 = None
        permute_764: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_345, [0, 2, 1]);  view_345 = None
        permute_765: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_346, [0, 2, 1]);  view_346 = None
        permute_766: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_187, [1, 0]);  permute_187 = None
        permute_767: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_185, [1, 0]);  permute_185 = None
        permute_768: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_183, [1, 0]);  permute_183 = None
        permute_769: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_182, [1, 0]);  permute_182 = None
        permute_770: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_181, [1, 0]);  permute_181 = None
        permute_771: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_180, [1, 0]);  permute_180 = None
        permute_772: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_179, [1, 0]);  permute_179 = None
        le_60: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_35, 0);  relu_35 = None
        permute_773: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_178, [1, 0]);  permute_178 = None
        permute_774: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_177, [1, 0]);  permute_177 = None
        le_61: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_34, 0);  relu_34 = None
        permute_775: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_176, [1, 0]);  permute_176 = None
        permute_776: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_175, [1, 0]);  permute_175 = None
        le_62: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_33, 0);  relu_33 = None
        permute_777: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_174, [1, 0]);  permute_174 = None
        permute_778: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_173, [1, 0]);  permute_173 = None
        le_63: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_32, 0);  relu_32 = None
        permute_779: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_172, [1, 0]);  permute_172 = None
        permute_780: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_171, [1, 0]);  permute_171 = None
        permute_781: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_311, [0, 2, 1]);  view_311 = None
        permute_782: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_312, [0, 2, 1]);  view_312 = None
        permute_783: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_308, [0, 2, 1]);  view_308 = None
        permute_784: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_309, [0, 2, 1]);  view_309 = None
        permute_785: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_167, [1, 0]);  permute_167 = None
        permute_786: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_165, [1, 0]);  permute_165 = None
        permute_787: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_163, [1, 0]);  permute_163 = None
        permute_788: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_162, [1, 0]);  permute_162 = None
        permute_789: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_161, [1, 0]);  permute_161 = None
        permute_790: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_160, [1, 0]);  permute_160 = None
        permute_791: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_159, [1, 0]);  permute_159 = None
        le_64: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_31, 0);  relu_31 = None
        permute_792: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_158, [1, 0]);  permute_158 = None
        permute_793: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_157, [1, 0]);  permute_157 = None
        le_65: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_30, 0);  relu_30 = None
        permute_794: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_156, [1, 0]);  permute_156 = None
        permute_795: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_155, [1, 0]);  permute_155 = None
        le_66: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_29, 0);  relu_29 = None
        permute_796: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_154, [1, 0]);  permute_154 = None
        permute_797: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_153, [1, 0]);  permute_153 = None
        le_67: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_28, 0);  relu_28 = None
        permute_798: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_152, [1, 0]);  permute_152 = None
        permute_799: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_151, [1, 0]);  permute_151 = None
        permute_800: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_274, [0, 2, 1]);  view_274 = None
        permute_801: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_275, [0, 2, 1]);  view_275 = None
        permute_802: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_271, [0, 2, 1]);  view_271 = None
        permute_803: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_272, [0, 2, 1]);  view_272 = None
        permute_804: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_147, [1, 0]);  permute_147 = None
        permute_805: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_145, [1, 0]);  permute_145 = None
        permute_806: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_143, [1, 0]);  permute_143 = None
        permute_807: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_142, [1, 0]);  permute_142 = None
        permute_808: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_141, [1, 0]);  permute_141 = None
        permute_809: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_140, [1, 0]);  permute_140 = None
        permute_810: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_139, [1, 0]);  permute_139 = None
        le_68: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_27, 0);  relu_27 = None
        permute_811: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_138, [1, 0]);  permute_138 = None
        permute_812: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_137, [1, 0]);  permute_137 = None
        le_69: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_26, 0);  relu_26 = None
        permute_813: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_136, [1, 0]);  permute_136 = None
        permute_814: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_135, [1, 0]);  permute_135 = None
        le_70: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_25, 0);  relu_25 = None
        permute_815: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_134, [1, 0]);  permute_134 = None
        permute_816: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_133, [1, 0]);  permute_133 = None
        le_71: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_24, 0);  relu_24 = None
        permute_817: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_132, [1, 0]);  permute_132 = None
        permute_818: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_131, [1, 0]);  permute_131 = None
        permute_819: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_237, [0, 2, 1]);  view_237 = None
        permute_820: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_238, [0, 2, 1]);  view_238 = None
        permute_821: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_234, [0, 2, 1]);  view_234 = None
        permute_822: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_235, [0, 2, 1]);  view_235 = None
        permute_823: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_127, [1, 0]);  permute_127 = None
        permute_824: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_125, [1, 0]);  permute_125 = None
        permute_825: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_123, [1, 0]);  permute_123 = None
        permute_826: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_122, [1, 0]);  permute_122 = None
        permute_827: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_121, [1, 0]);  permute_121 = None
        permute_828: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_120, [1, 0]);  permute_120 = None
        permute_829: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_119, [1, 0]);  permute_119 = None
        le_72: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_23, 0);  relu_23 = None
        permute_830: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_118, [1, 0]);  permute_118 = None
        permute_831: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_117, [1, 0]);  permute_117 = None
        le_73: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_22, 0);  relu_22 = None
        permute_832: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_116, [1, 0]);  permute_116 = None
        permute_833: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_115, [1, 0]);  permute_115 = None
        le_74: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_21, 0);  relu_21 = None
        permute_834: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_114, [1, 0]);  permute_114 = None
        permute_835: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_113, [1, 0]);  permute_113 = None
        le_75: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_20, 0);  relu_20 = None
        permute_836: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_112, [1, 0]);  permute_112 = None
        permute_837: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_111, [1, 0]);  permute_111 = None
        permute_838: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_200, [0, 2, 1]);  view_200 = None
        permute_839: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_201, [0, 2, 1]);  view_201 = None
        permute_840: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_197, [0, 2, 1]);  view_197 = None
        permute_841: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_198, [0, 2, 1]);  view_198 = None
        permute_842: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_107, [1, 0]);  permute_107 = None
        permute_843: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_105, [1, 0]);  permute_105 = None
        permute_844: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_103, [1, 0]);  permute_103 = None
        permute_845: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_102, [1, 0]);  permute_102 = None
        permute_846: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_101, [1, 0]);  permute_101 = None
        permute_847: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_100, [1, 0]);  permute_100 = None
        permute_848: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_99, [1, 0]);  permute_99 = None
        le_76: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_19, 0);  relu_19 = None
        permute_849: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_98, [1, 0]);  permute_98 = None
        permute_850: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_97, [1, 0]);  permute_97 = None
        le_77: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_18, 0);  relu_18 = None
        permute_851: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_96, [1, 0]);  permute_96 = None
        permute_852: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_95, [1, 0]);  permute_95 = None
        le_78: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_17, 0);  relu_17 = None
        permute_853: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_94, [1, 0]);  permute_94 = None
        permute_854: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_93, [1, 0]);  permute_93 = None
        le_79: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_16, 0);  relu_16 = None
        permute_855: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_92, [1, 0]);  permute_92 = None
        permute_856: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_91, [1, 0]);  permute_91 = None
        permute_857: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_163, [0, 2, 1]);  view_163 = None
        permute_858: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_164, [0, 2, 1]);  view_164 = None
        permute_859: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_160, [0, 2, 1]);  view_160 = None
        permute_860: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_161, [0, 2, 1]);  view_161 = None
        permute_861: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_87, [1, 0]);  permute_87 = None
        permute_862: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_85, [1, 0]);  permute_85 = None
        permute_863: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_83, [1, 0]);  permute_83 = None
        permute_864: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_82, [1, 0]);  permute_82 = None
        permute_865: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_81, [1, 0]);  permute_81 = None
        permute_866: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_80, [1, 0]);  permute_80 = None
        permute_867: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_79, [1, 0]);  permute_79 = None
        le_80: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_15, 0);  relu_15 = None
        permute_868: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_78, [1, 0]);  permute_78 = None
        permute_869: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_77, [1, 0]);  permute_77 = None
        le_81: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_14, 0);  relu_14 = None
        permute_870: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_76, [1, 0]);  permute_76 = None
        permute_871: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_75, [1, 0]);  permute_75 = None
        le_82: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_13, 0);  relu_13 = None
        permute_872: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_74, [1, 0]);  permute_74 = None
        permute_873: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_73, [1, 0]);  permute_73 = None
        le_83: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_12, 0);  relu_12 = None
        permute_874: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_72, [1, 0]);  permute_72 = None
        permute_875: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_71, [1, 0]);  permute_71 = None
        permute_876: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_126, [0, 2, 1]);  view_126 = None
        permute_877: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_127, [0, 2, 1]);  view_127 = None
        permute_878: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_123, [0, 2, 1]);  view_123 = None
        permute_879: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_124, [0, 2, 1]);  view_124 = None
        permute_880: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_67, [1, 0]);  permute_67 = None
        permute_881: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_65, [1, 0]);  permute_65 = None
        permute_882: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_63, [1, 0]);  permute_63 = None
        permute_883: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_62, [1, 0]);  permute_62 = None
        permute_884: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_61, [1, 0]);  permute_61 = None
        permute_885: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_60, [1, 0]);  permute_60 = None
        permute_886: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_59, [1, 0]);  permute_59 = None
        le_84: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_11, 0);  relu_11 = None
        permute_887: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_58, [1, 0]);  permute_58 = None
        permute_888: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_57, [1, 0]);  permute_57 = None
        le_85: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_10, 0);  relu_10 = None
        permute_889: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_56, [1, 0]);  permute_56 = None
        permute_890: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_55, [1, 0]);  permute_55 = None
        le_86: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_9, 0);  relu_9 = None
        permute_891: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_54, [1, 0]);  permute_54 = None
        permute_892: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_53, [1, 0]);  permute_53 = None
        le_87: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_8, 0);  relu_8 = None
        permute_893: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_52, [1, 0]);  permute_52 = None
        permute_894: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_51, [1, 0]);  permute_51 = None
        permute_895: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_89, [0, 2, 1]);  view_89 = None
        permute_896: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_90, [0, 2, 1]);  view_90 = None
        permute_897: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_86, [0, 2, 1]);  view_86 = None
        permute_898: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_87, [0, 2, 1]);  view_87 = None
        permute_899: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_47, [1, 0]);  permute_47 = None
        permute_900: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_45, [1, 0]);  permute_45 = None
        permute_901: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_43, [1, 0]);  permute_43 = None
        permute_902: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_42, [1, 0]);  permute_42 = None
        permute_903: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_41, [1, 0]);  permute_41 = None
        permute_904: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_40, [1, 0]);  permute_40 = None
        permute_905: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_39, [1, 0]);  permute_39 = None
        le_88: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_7, 0);  relu_7 = None
        permute_906: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_38, [1, 0]);  permute_38 = None
        permute_907: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_37, [1, 0]);  permute_37 = None
        le_89: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_6, 0);  relu_6 = None
        permute_908: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_36, [1, 0]);  permute_36 = None
        permute_909: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_35, [1, 0]);  permute_35 = None
        le_90: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_5, 0);  relu_5 = None
        permute_910: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_34, [1, 0]);  permute_34 = None
        permute_911: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_33, [1, 0]);  permute_33 = None
        le_91: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_4, 0);  relu_4 = None
        permute_912: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_32, [1, 0]);  permute_32 = None
        permute_913: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_31, [1, 0]);  permute_31 = None
        permute_914: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_52, [0, 2, 1]);  view_52 = None
        permute_915: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_53, [0, 2, 1]);  view_53 = None
        permute_916: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_49, [0, 2, 1]);  view_49 = None
        permute_917: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_50, [0, 2, 1]);  view_50 = None
        permute_918: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_27, [1, 0]);  permute_27 = None
        permute_919: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_25, [1, 0]);  permute_25 = None
        permute_920: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_23, [1, 0]);  permute_23 = None
        permute_921: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_22, [1, 0]);  permute_22 = None
        permute_922: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_21, [1, 0]);  permute_21 = None
        permute_923: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_20, [1, 0]);  permute_20 = None
        permute_924: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_19, [1, 0]);  permute_19 = None
        le_92: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_3, 0);  relu_3 = None
        permute_925: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_18, [1, 0]);  permute_18 = None
        permute_926: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_17, [1, 0]);  permute_17 = None
        le_93: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_2, 0);  relu_2 = None
        permute_927: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_16, [1, 0]);  permute_16 = None
        permute_928: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_15, [1, 0]);  permute_15 = None
        le_94: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu_1, 0);  relu_1 = None
        permute_929: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_14, [1, 0]);  permute_14 = None
        permute_930: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_13, [1, 0]);  permute_13 = None
        le_95: "b8[256, 128, 512]" = torch.ops.aten.le.Scalar(relu, 0);  relu = None
        permute_931: "bf16[512, 128]" = torch.ops.aten.permute.default(permute_12, [1, 0]);  permute_12 = None
        permute_932: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_11, [1, 0]);  permute_11 = None
        permute_933: "bf16[1024, 128, 128]" = torch.ops.aten.permute.default(view_15, [0, 2, 1]);  view_15 = None
        permute_934: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_16, [0, 2, 1]);  view_16 = None
        permute_935: "bf16[1024, 32, 128]" = torch.ops.aten.permute.default(view_12, [0, 2, 1]);  view_12 = None
        permute_936: "bf16[1024, 128, 32]" = torch.ops.aten.permute.default(view_13, [0, 2, 1]);  view_13 = None
        permute_937: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_7, [1, 0]);  permute_7 = None
        permute_938: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_5, [1, 0]);  permute_5 = None
        permute_939: "bf16[128, 128]" = torch.ops.aten.permute.default(permute_3, [1, 0]);  permute_3 = None
        permute_940: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_2, [1, 0]);  permute_2 = None
        permute_941: "bf16[128, 512]" = torch.ops.aten.permute.default(permute_1, [1, 0]);  permute_1 = None
        permute_942: "bf16[512, 384]" = torch.ops.aten.permute.default(permute, [1, 0]);  permute = None
        return (full, view, addmm, embedding_1, embedding_2, view_2, addmm_1, addmm_2, view_5, bmm, amax, sum_1, logical_not_1, gt, view_19, addmm_6, view_21, view_23, add_7, view_25, view_27, add_9, view_29, view_31, add_11, view_33, view_35, add_13, view_37, add_15, view_39, addmm_16, addmm_17, view_42, where_1, gt_1, view_56, addmm_21, view_58, view_60, add_21, view_62, view_64, add_23, view_66, view_68, add_25, view_70, view_72, add_27, view_74, add_29, view_76, addmm_31, addmm_32, view_79, where_2, gt_2, view_93, addmm_36, view_95, view_97, add_35, view_99, view_101, add_37, view_103, view_105, add_39, view_107, view_109, add_41, view_111, add_43, view_113, addmm_46, addmm_47, view_116, where_3, gt_3, view_130, addmm_51, view_132, view_134, add_49, view_136, view_138, add_51, view_140, view_142, add_53, view_144, view_146, add_55, view_148, add_57, view_150, addmm_61, addmm_62, view_153, where_4, gt_4, view_167, addmm_66, view_169, view_171, add_63, view_173, view_175, add_65, view_177, view_179, add_67, view_181, view_183, add_69, view_185, add_71, view_187, addmm_76, addmm_77, view_190, where_5, gt_5, view_204, addmm_81, view_206, view_208, add_77, view_210, view_212, add_79, view_214, view_216, add_81, view_218, view_220, add_83, view_222, add_85, view_224, addmm_91, addmm_92, view_227, where_6, gt_6, view_241, addmm_96, view_243, view_245, add_91, view_247, view_249, add_93, view_251, view_253, add_95, view_255, view_257, add_97, view_259, add_99, view_261, addmm_106, addmm_107, view_264, where_7, gt_7, view_278, addmm_111, view_280, view_282, add_105, view_284, view_286, add_107, view_288, view_290, add_109, view_292, view_294, add_111, view_296, add_113, view_298, addmm_121, addmm_122, view_301, where_8, gt_8, view_315, addmm_126, view_317, view_319, add_119, view_321, view_323, add_121, view_325, view_327, add_123, view_329, view_331, add_125, view_333, add_127, view_335, addmm_136, addmm_137, view_338, where_9, gt_9, view_352, addmm_141, view_354, view_356, add_133, view_358, view_360, add_135, view_362, view_364, add_137, view_366, view_368, add_139, view_370, add_141, view_372, addmm_151, addmm_152, view_375, where_10, gt_10, view_389, addmm_156, view_391, view_393, add_147, view_395, view_397, add_149, view_399, view_401, add_151, view_403, view_405, add_153, view_407, add_155, view_409, addmm_166, addmm_167, view_412, where_11, gt_11, view_426, addmm_171, view_428, view_430, add_161, view_432, view_434, add_163, view_436, view_438, add_165, view_440, view_442, add_167, view_444, add_169, view_446, addmm_181, addmm_182, view_449, where_12, gt_12, view_463, addmm_186, view_465, view_467, add_175, view_469, view_471, add_177, view_473, view_475, add_179, view_477, view_479, add_181, view_481, add_183, view_483, addmm_196, addmm_197, view_486, where_13, gt_13, view_500, addmm_201, view_502, view_504, add_189, view_506, view_508, add_191, view_510, view_512, add_193, view_514, view_516, add_195, view_518, add_197, view_520, addmm_211, addmm_212, view_523, where_14, gt_14, view_537, addmm_216, view_539, view_541, add_203, view_543, view_545, add_205, view_547, view_549, add_207, view_551, view_553, add_209, view_555, add_211, view_557, addmm_226, addmm_227, view_560, where_15, gt_15, view_574, addmm_231, view_576, view_578, add_217, view_580, view_582, add_219, view_584, view_586, add_221, view_588, view_590, add_223, view_592, add_225, view_594, addmm_241, addmm_242, view_597, where_16, gt_16, view_611, addmm_246, view_613, view_615, add_231, view_617, view_619, add_233, view_621, view_623, add_235, view_625, view_627, add_237, view_629, add_239, view_631, addmm_256, addmm_257, view_634, where_17, gt_17, view_648, addmm_261, view_650, view_652, add_245, view_654, view_656, add_247, view_658, view_660, add_249, view_662, view_664, add_251, view_666, add_253, view_668, addmm_271, addmm_272, view_671, where_18, gt_18, view_685, addmm_276, view_687, view_689, add_259, view_691, view_693, add_261, view_695, view_697, add_263, view_699, view_701, add_265, view_703, add_267, view_705, addmm_286, addmm_287, view_708, where_19, gt_19, view_722, addmm_291, view_724, view_726, add_273, view_728, view_730, add_275, view_732, view_734, add_277, view_736, view_738, add_279, view_740, add_281, view_742, addmm_301, addmm_302, view_745, where_20, gt_20, view_759, addmm_306, view_761, view_763, add_287, view_765, view_767, add_289, view_769, view_771, add_291, view_773, view_775, add_293, view_777, add_295, view_779, addmm_316, addmm_317, view_782, where_21, gt_21, view_796, addmm_321, view_798, view_800, add_301, view_802, view_804, add_303, view_806, view_808, add_305, view_810, view_812, add_307, view_814, add_309, view_816, addmm_331, addmm_332, view_819, where_22, gt_22, view_833, addmm_336, view_835, view_837, add_315, view_839, view_841, add_317, view_843, view_845, add_319, view_847, view_849, add_321, view_851, add_323, view_853, addmm_346, addmm_347, view_856, where_23, gt_23, view_870, addmm_351, view_872, view_874, add_329, view_876, view_878, add_331, view_880, view_882, add_333, view_884, view_886, add_335, view_888, add_337, view_890, addmm_361, getitem_1, rsqrt, view_895, convert_element_type_971, convert_element_type_973, div_24, permute_483, permute_484, permute_485, permute_486, permute_487, le, permute_488, permute_489, le_1, permute_490, permute_491, le_2, permute_492, permute_493, le_3, permute_494, permute_495, permute_496, permute_497, permute_498, permute_499, permute_500, permute_501, permute_502, permute_503, permute_504, permute_505, permute_506, le_4, permute_507, permute_508, le_5, permute_509, permute_510, le_6, permute_511, permute_512, le_7, permute_513, permute_514, permute_515, permute_516, permute_517, permute_518, permute_519, permute_520, permute_521, permute_522, permute_523, permute_524, permute_525, le_8, permute_526, permute_527, le_9, permute_528, permute_529, le_10, permute_530, permute_531, le_11, permute_532, permute_533, permute_534, permute_535, permute_536, permute_537, permute_538, permute_539, permute_540, permute_541, permute_542, permute_543, permute_544, le_12, permute_545, permute_546, le_13, permute_547, permute_548, le_14, permute_549, permute_550, le_15, permute_551, permute_552, permute_553, permute_554, permute_555, permute_556, permute_557, permute_558, permute_559, permute_560, permute_561, permute_562, permute_563, le_16, permute_564, permute_565, le_17, permute_566, permute_567, le_18, permute_568, permute_569, le_19, permute_570, permute_571, permute_572, permute_573, permute_574, permute_575, permute_576, permute_577, permute_578, permute_579, permute_580, permute_581, permute_582, le_20, permute_583, permute_584, le_21, permute_585, permute_586, le_22, permute_587, permute_588, le_23, permute_589, permute_590, permute_591, permute_592, permute_593, permute_594, permute_595, permute_596, permute_597, permute_598, permute_599, permute_600, permute_601, le_24, permute_602, permute_603, le_25, permute_604, permute_605, le_26, permute_606, permute_607, le_27, permute_608, permute_609, permute_610, permute_611, permute_612, permute_613, permute_614, permute_615, permute_616, permute_617, permute_618, permute_619, permute_620, le_28, permute_621, permute_622, le_29, permute_623, permute_624, le_30, permute_625, permute_626, le_31, permute_627, permute_628, permute_629, permute_630, permute_631, permute_632, permute_633, permute_634, permute_635, permute_636, permute_637, permute_638, permute_639, le_32, permute_640, permute_641, le_33, permute_642, permute_643, le_34, permute_644, permute_645, le_35, permute_646, permute_647, permute_648, permute_649, permute_650, permute_651, permute_652, permute_653, permute_654, permute_655, permute_656, permute_657, permute_658, le_36, permute_659, permute_660, le_37, permute_661, permute_662, le_38, permute_663, permute_664, le_39, permute_665, permute_666, permute_667, permute_668, permute_669, permute_670, permute_671, permute_672, permute_673, permute_674, permute_675, permute_676, permute_677, le_40, permute_678, permute_679, le_41, permute_680, permute_681, le_42, permute_682, permute_683, le_43, permute_684, permute_685, permute_686, permute_687, permute_688, permute_689, permute_690, permute_691, permute_692, permute_693, permute_694, permute_695, permute_696, le_44, permute_697, permute_698, le_45, permute_699, permute_700, le_46, permute_701, permute_702, le_47, permute_703, permute_704, permute_705, permute_706, permute_707, permute_708, permute_709, permute_710, permute_711, permute_712, permute_713, permute_714, permute_715, le_48, permute_716, permute_717, le_49, permute_718, permute_719, le_50, permute_720, permute_721, le_51, permute_722, permute_723, permute_724, permute_725, permute_726, permute_727, permute_728, permute_729, permute_730, permute_731, permute_732, permute_733, permute_734, le_52, permute_735, permute_736, le_53, permute_737, permute_738, le_54, permute_739, permute_740, le_55, permute_741, permute_742, permute_743, permute_744, permute_745, permute_746, permute_747, permute_748, permute_749, permute_750, permute_751, permute_752, permute_753, le_56, permute_754, permute_755, le_57, permute_756, permute_757, le_58, permute_758, permute_759, le_59, permute_760, permute_761, permute_762, permute_763, permute_764, permute_765, permute_766, permute_767, permute_768, permute_769, permute_770, permute_771, permute_772, le_60, permute_773, permute_774, le_61, permute_775, permute_776, le_62, permute_777, permute_778, le_63, permute_779, permute_780, permute_781, permute_782, permute_783, permute_784, permute_785, permute_786, permute_787, permute_788, permute_789, permute_790, permute_791, le_64, permute_792, permute_793, le_65, permute_794, permute_795, le_66, permute_796, permute_797, le_67, permute_798, permute_799, permute_800, permute_801, permute_802, permute_803, permute_804, permute_805, permute_806, permute_807, permute_808, permute_809, permute_810, le_68, permute_811, permute_812, le_69, permute_813, permute_814, le_70, permute_815, permute_816, le_71, permute_817, permute_818, permute_819, permute_820, permute_821, permute_822, permute_823, permute_824, permute_825, permute_826, permute_827, permute_828, permute_829, le_72, permute_830, permute_831, le_73, permute_832, permute_833, le_74, permute_834, permute_835, le_75, permute_836, permute_837, permute_838, permute_839, permute_840, permute_841, permute_842, permute_843, permute_844, permute_845, permute_846, permute_847, permute_848, le_76, permute_849, permute_850, le_77, permute_851, permute_852, le_78, permute_853, permute_854, le_79, permute_855, permute_856, permute_857, permute_858, permute_859, permute_860, permute_861, permute_862, permute_863, permute_864, permute_865, permute_866, permute_867, le_80, permute_868, permute_869, le_81, permute_870, permute_871, le_82, permute_872, permute_873, le_83, permute_874, permute_875, permute_876, permute_877, permute_878, permute_879, permute_880, permute_881, permute_882, permute_883, permute_884, permute_885, permute_886, le_84, permute_887, permute_888, le_85, permute_889, permute_890, le_86, permute_891, permute_892, le_87, permute_893, permute_894, permute_895, permute_896, permute_897, permute_898, permute_899, permute_900, permute_901, permute_902, permute_903, permute_904, permute_905, le_88, permute_906, permute_907, le_89, permute_908, permute_909, le_90, permute_910, permute_911, le_91, permute_912, permute_913, permute_914, permute_915, permute_916, permute_917, permute_918, permute_919, permute_920, permute_921, permute_922, permute_923, permute_924, le_92, permute_925, permute_926, le_93, permute_927, permute_928, le_94, permute_929, permute_930, le_95, permute_931, permute_932, permute_933, permute_934, permute_935, permute_936, permute_937, permute_938, permute_939, permute_940, permute_941, permute_942)



def _default_make_inputs():
    configs = load_shape_configs(__file__)
    if not configs:
        raise RuntimeError(
            "no shapes.json next to this repro — pass an explicit config "
            "via make_inputs(shape_config=...)")
    return make_inputs_from_config(next(iter(configs.values())))


def make_inputs(shape_config=None):
    """Generate inputs for a specific shape config, or default."""
    if shape_config is not None:
        return make_inputs_from_config(shape_config)
    return _default_make_inputs()


if __name__ == "__main__":
    benchmark_repro(__file__, Repro, make_inputs)

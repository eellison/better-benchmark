"""
Standalone repro captured via capture_hook.
Label: MobileBertForMaskedLM
Pattern hash: c377b4a966c3
Shape hash: 9c0f2037
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
    def forward(self, arg0_1: "bf16[30522, 128]", arg1_1: "i64[256, 128]", arg2_1: "bf16[512, 384]", arg3_1: "bf16[512]", arg4_1: "i64[1, 512]", arg5_1: "bf16[512, 512]", arg6_1: "bf16[2, 512]", arg7_1: "bf16[512]", arg8_1: "bf16[512]", arg9_1: "bf16[128, 512]", arg10_1: "bf16[128]", arg11_1: "bf16[128]", arg12_1: "bf16[128]", arg13_1: "bf16[128, 128]", arg14_1: "bf16[128]", arg15_1: "bf16[128, 128]", arg16_1: "bf16[128]", arg17_1: "bf16[128, 512]", arg18_1: "bf16[128]", arg19_1: "bf16[128, 128]", arg20_1: "bf16[128]", arg21_1: "bf16[128, 512]", arg22_1: "bf16[128]", arg23_1: "bf16[128]", arg24_1: "bf16[128]", arg25_1: "bf16[128]", arg26_1: "bf16[128]", arg27_1: "bf16[512, 128]", arg28_1: "bf16[512]", arg29_1: "bf16[128, 512]", arg30_1: "bf16[128]", arg31_1: "bf16[128]", arg32_1: "bf16[128]", arg33_1: "bf16[512, 128]", arg34_1: "bf16[512]", arg35_1: "bf16[128, 512]", arg36_1: "bf16[128]", arg37_1: "bf16[128]", arg38_1: "bf16[128]", arg39_1: "bf16[512, 128]", arg40_1: "bf16[512]", arg41_1: "bf16[128, 512]", arg42_1: "bf16[128]", arg43_1: "bf16[128]", arg44_1: "bf16[128]", arg45_1: "bf16[512, 128]", arg46_1: "bf16[512]", arg47_1: "bf16[128, 512]", arg48_1: "bf16[128]", arg49_1: "bf16[128]", arg50_1: "bf16[128]", arg51_1: "bf16[512, 128]", arg52_1: "bf16[512]", arg53_1: "bf16[512]", arg54_1: "bf16[512]", arg55_1: "bf16[128, 512]", arg56_1: "bf16[128]", arg57_1: "bf16[128]", arg58_1: "bf16[128]", arg59_1: "bf16[128, 128]", arg60_1: "bf16[128]", arg61_1: "bf16[128, 128]", arg62_1: "bf16[128]", arg63_1: "bf16[128, 512]", arg64_1: "bf16[128]", arg65_1: "bf16[128, 128]", arg66_1: "bf16[128]", arg67_1: "bf16[128, 512]", arg68_1: "bf16[128]", arg69_1: "bf16[128]", arg70_1: "bf16[128]", arg71_1: "bf16[128]", arg72_1: "bf16[128]", arg73_1: "bf16[512, 128]", arg74_1: "bf16[512]", arg75_1: "bf16[128, 512]", arg76_1: "bf16[128]", arg77_1: "bf16[128]", arg78_1: "bf16[128]", arg79_1: "bf16[512, 128]", arg80_1: "bf16[512]", arg81_1: "bf16[128, 512]", arg82_1: "bf16[128]", arg83_1: "bf16[128]", arg84_1: "bf16[128]", arg85_1: "bf16[512, 128]", arg86_1: "bf16[512]", arg87_1: "bf16[128, 512]", arg88_1: "bf16[128]", arg89_1: "bf16[128]", arg90_1: "bf16[128]", arg91_1: "bf16[512, 128]", arg92_1: "bf16[512]", arg93_1: "bf16[128, 512]", arg94_1: "bf16[128]", arg95_1: "bf16[128]", arg96_1: "bf16[128]", arg97_1: "bf16[512, 128]", arg98_1: "bf16[512]", arg99_1: "bf16[512]", arg100_1: "bf16[512]", arg101_1: "bf16[128, 512]", arg102_1: "bf16[128]", arg103_1: "bf16[128]", arg104_1: "bf16[128]", arg105_1: "bf16[128, 128]", arg106_1: "bf16[128]", arg107_1: "bf16[128, 128]", arg108_1: "bf16[128]", arg109_1: "bf16[128, 512]", arg110_1: "bf16[128]", arg111_1: "bf16[128, 128]", arg112_1: "bf16[128]", arg113_1: "bf16[128, 512]", arg114_1: "bf16[128]", arg115_1: "bf16[128]", arg116_1: "bf16[128]", arg117_1: "bf16[128]", arg118_1: "bf16[128]", arg119_1: "bf16[512, 128]", arg120_1: "bf16[512]", arg121_1: "bf16[128, 512]", arg122_1: "bf16[128]", arg123_1: "bf16[128]", arg124_1: "bf16[128]", arg125_1: "bf16[512, 128]", arg126_1: "bf16[512]", arg127_1: "bf16[128, 512]", arg128_1: "bf16[128]", arg129_1: "bf16[128]", arg130_1: "bf16[128]", arg131_1: "bf16[512, 128]", arg132_1: "bf16[512]", arg133_1: "bf16[128, 512]", arg134_1: "bf16[128]", arg135_1: "bf16[128]", arg136_1: "bf16[128]", arg137_1: "bf16[512, 128]", arg138_1: "bf16[512]", arg139_1: "bf16[128, 512]", arg140_1: "bf16[128]", arg141_1: "bf16[128]", arg142_1: "bf16[128]", arg143_1: "bf16[512, 128]", arg144_1: "bf16[512]", arg145_1: "bf16[512]", arg146_1: "bf16[512]", arg147_1: "bf16[128, 512]", arg148_1: "bf16[128]", arg149_1: "bf16[128]", arg150_1: "bf16[128]", arg151_1: "bf16[128, 128]", arg152_1: "bf16[128]", arg153_1: "bf16[128, 128]", arg154_1: "bf16[128]", arg155_1: "bf16[128, 512]", arg156_1: "bf16[128]", arg157_1: "bf16[128, 128]", arg158_1: "bf16[128]", arg159_1: "bf16[128, 512]", arg160_1: "bf16[128]", arg161_1: "bf16[128]", arg162_1: "bf16[128]", arg163_1: "bf16[128]", arg164_1: "bf16[128]", arg165_1: "bf16[512, 128]", arg166_1: "bf16[512]", arg167_1: "bf16[128, 512]", arg168_1: "bf16[128]", arg169_1: "bf16[128]", arg170_1: "bf16[128]", arg171_1: "bf16[512, 128]", arg172_1: "bf16[512]", arg173_1: "bf16[128, 512]", arg174_1: "bf16[128]", arg175_1: "bf16[128]", arg176_1: "bf16[128]", arg177_1: "bf16[512, 128]", arg178_1: "bf16[512]", arg179_1: "bf16[128, 512]", arg180_1: "bf16[128]", arg181_1: "bf16[128]", arg182_1: "bf16[128]", arg183_1: "bf16[512, 128]", arg184_1: "bf16[512]", arg185_1: "bf16[128, 512]", arg186_1: "bf16[128]", arg187_1: "bf16[128]", arg188_1: "bf16[128]", arg189_1: "bf16[512, 128]", arg190_1: "bf16[512]", arg191_1: "bf16[512]", arg192_1: "bf16[512]", arg193_1: "bf16[128, 512]", arg194_1: "bf16[128]", arg195_1: "bf16[128]", arg196_1: "bf16[128]", arg197_1: "bf16[128, 128]", arg198_1: "bf16[128]", arg199_1: "bf16[128, 128]", arg200_1: "bf16[128]", arg201_1: "bf16[128, 512]", arg202_1: "bf16[128]", arg203_1: "bf16[128, 128]", arg204_1: "bf16[128]", arg205_1: "bf16[128, 512]", arg206_1: "bf16[128]", arg207_1: "bf16[128]", arg208_1: "bf16[128]", arg209_1: "bf16[128]", arg210_1: "bf16[128]", arg211_1: "bf16[512, 128]", arg212_1: "bf16[512]", arg213_1: "bf16[128, 512]", arg214_1: "bf16[128]", arg215_1: "bf16[128]", arg216_1: "bf16[128]", arg217_1: "bf16[512, 128]", arg218_1: "bf16[512]", arg219_1: "bf16[128, 512]", arg220_1: "bf16[128]", arg221_1: "bf16[128]", arg222_1: "bf16[128]", arg223_1: "bf16[512, 128]", arg224_1: "bf16[512]", arg225_1: "bf16[128, 512]", arg226_1: "bf16[128]", arg227_1: "bf16[128]", arg228_1: "bf16[128]", arg229_1: "bf16[512, 128]", arg230_1: "bf16[512]", arg231_1: "bf16[128, 512]", arg232_1: "bf16[128]", arg233_1: "bf16[128]", arg234_1: "bf16[128]", arg235_1: "bf16[512, 128]", arg236_1: "bf16[512]", arg237_1: "bf16[512]", arg238_1: "bf16[512]", arg239_1: "bf16[128, 512]", arg240_1: "bf16[128]", arg241_1: "bf16[128]", arg242_1: "bf16[128]", arg243_1: "bf16[128, 128]", arg244_1: "bf16[128]", arg245_1: "bf16[128, 128]", arg246_1: "bf16[128]", arg247_1: "bf16[128, 512]", arg248_1: "bf16[128]", arg249_1: "bf16[128, 128]", arg250_1: "bf16[128]", arg251_1: "bf16[128, 512]", arg252_1: "bf16[128]", arg253_1: "bf16[128]", arg254_1: "bf16[128]", arg255_1: "bf16[128]", arg256_1: "bf16[128]", arg257_1: "bf16[512, 128]", arg258_1: "bf16[512]", arg259_1: "bf16[128, 512]", arg260_1: "bf16[128]", arg261_1: "bf16[128]", arg262_1: "bf16[128]", arg263_1: "bf16[512, 128]", arg264_1: "bf16[512]", arg265_1: "bf16[128, 512]", arg266_1: "bf16[128]", arg267_1: "bf16[128]", arg268_1: "bf16[128]", arg269_1: "bf16[512, 128]", arg270_1: "bf16[512]", arg271_1: "bf16[128, 512]", arg272_1: "bf16[128]", arg273_1: "bf16[128]", arg274_1: "bf16[128]", arg275_1: "bf16[512, 128]", arg276_1: "bf16[512]", arg277_1: "bf16[128, 512]", arg278_1: "bf16[128]", arg279_1: "bf16[128]", arg280_1: "bf16[128]", arg281_1: "bf16[512, 128]", arg282_1: "bf16[512]", arg283_1: "bf16[512]", arg284_1: "bf16[512]", arg285_1: "bf16[128, 512]", arg286_1: "bf16[128]", arg287_1: "bf16[128]", arg288_1: "bf16[128]", arg289_1: "bf16[128, 128]", arg290_1: "bf16[128]", arg291_1: "bf16[128, 128]", arg292_1: "bf16[128]", arg293_1: "bf16[128, 512]", arg294_1: "bf16[128]", arg295_1: "bf16[128, 128]", arg296_1: "bf16[128]", arg297_1: "bf16[128, 512]", arg298_1: "bf16[128]", arg299_1: "bf16[128]", arg300_1: "bf16[128]", arg301_1: "bf16[128]", arg302_1: "bf16[128]", arg303_1: "bf16[512, 128]", arg304_1: "bf16[512]", arg305_1: "bf16[128, 512]", arg306_1: "bf16[128]", arg307_1: "bf16[128]", arg308_1: "bf16[128]", arg309_1: "bf16[512, 128]", arg310_1: "bf16[512]", arg311_1: "bf16[128, 512]", arg312_1: "bf16[128]", arg313_1: "bf16[128]", arg314_1: "bf16[128]", arg315_1: "bf16[512, 128]", arg316_1: "bf16[512]", arg317_1: "bf16[128, 512]", arg318_1: "bf16[128]", arg319_1: "bf16[128]", arg320_1: "bf16[128]", arg321_1: "bf16[512, 128]", arg322_1: "bf16[512]", arg323_1: "bf16[128, 512]", arg324_1: "bf16[128]", arg325_1: "bf16[128]", arg326_1: "bf16[128]", arg327_1: "bf16[512, 128]", arg328_1: "bf16[512]", arg329_1: "bf16[512]", arg330_1: "bf16[512]", arg331_1: "bf16[128, 512]", arg332_1: "bf16[128]", arg333_1: "bf16[128]", arg334_1: "bf16[128]", arg335_1: "bf16[128, 128]", arg336_1: "bf16[128]", arg337_1: "bf16[128, 128]", arg338_1: "bf16[128]", arg339_1: "bf16[128, 512]", arg340_1: "bf16[128]", arg341_1: "bf16[128, 128]", arg342_1: "bf16[128]", arg343_1: "bf16[128, 512]", arg344_1: "bf16[128]", arg345_1: "bf16[128]", arg346_1: "bf16[128]", arg347_1: "bf16[128]", arg348_1: "bf16[128]", arg349_1: "bf16[512, 128]", arg350_1: "bf16[512]", arg351_1: "bf16[128, 512]", arg352_1: "bf16[128]", arg353_1: "bf16[128]", arg354_1: "bf16[128]", arg355_1: "bf16[512, 128]", arg356_1: "bf16[512]", arg357_1: "bf16[128, 512]", arg358_1: "bf16[128]", arg359_1: "bf16[128]", arg360_1: "bf16[128]", arg361_1: "bf16[512, 128]", arg362_1: "bf16[512]", arg363_1: "bf16[128, 512]", arg364_1: "bf16[128]", arg365_1: "bf16[128]", arg366_1: "bf16[128]", arg367_1: "bf16[512, 128]", arg368_1: "bf16[512]", arg369_1: "bf16[128, 512]", arg370_1: "bf16[128]", arg371_1: "bf16[128]", arg372_1: "bf16[128]", arg373_1: "bf16[512, 128]", arg374_1: "bf16[512]", arg375_1: "bf16[512]", arg376_1: "bf16[512]", arg377_1: "bf16[128, 512]", arg378_1: "bf16[128]", arg379_1: "bf16[128]", arg380_1: "bf16[128]", arg381_1: "bf16[128, 128]", arg382_1: "bf16[128]", arg383_1: "bf16[128, 128]", arg384_1: "bf16[128]", arg385_1: "bf16[128, 512]", arg386_1: "bf16[128]", arg387_1: "bf16[128, 128]", arg388_1: "bf16[128]", arg389_1: "bf16[128, 512]", arg390_1: "bf16[128]", arg391_1: "bf16[128]", arg392_1: "bf16[128]", arg393_1: "bf16[128]", arg394_1: "bf16[128]", arg395_1: "bf16[512, 128]", arg396_1: "bf16[512]", arg397_1: "bf16[128, 512]", arg398_1: "bf16[128]", arg399_1: "bf16[128]", arg400_1: "bf16[128]", arg401_1: "bf16[512, 128]", arg402_1: "bf16[512]", arg403_1: "bf16[128, 512]", arg404_1: "bf16[128]", arg405_1: "bf16[128]", arg406_1: "bf16[128]", arg407_1: "bf16[512, 128]", arg408_1: "bf16[512]", arg409_1: "bf16[128, 512]", arg410_1: "bf16[128]", arg411_1: "bf16[128]", arg412_1: "bf16[128]", arg413_1: "bf16[512, 128]", arg414_1: "bf16[512]", arg415_1: "bf16[128, 512]", arg416_1: "bf16[128]", arg417_1: "bf16[128]", arg418_1: "bf16[128]", arg419_1: "bf16[512, 128]", arg420_1: "bf16[512]", arg421_1: "bf16[512]", arg422_1: "bf16[512]", arg423_1: "bf16[128, 512]", arg424_1: "bf16[128]", arg425_1: "bf16[128]", arg426_1: "bf16[128]", arg427_1: "bf16[128, 128]", arg428_1: "bf16[128]", arg429_1: "bf16[128, 128]", arg430_1: "bf16[128]", arg431_1: "bf16[128, 512]", arg432_1: "bf16[128]", arg433_1: "bf16[128, 128]", arg434_1: "bf16[128]", arg435_1: "bf16[128, 512]", arg436_1: "bf16[128]", arg437_1: "bf16[128]", arg438_1: "bf16[128]", arg439_1: "bf16[128]", arg440_1: "bf16[128]", arg441_1: "bf16[512, 128]", arg442_1: "bf16[512]", arg443_1: "bf16[128, 512]", arg444_1: "bf16[128]", arg445_1: "bf16[128]", arg446_1: "bf16[128]", arg447_1: "bf16[512, 128]", arg448_1: "bf16[512]", arg449_1: "bf16[128, 512]", arg450_1: "bf16[128]", arg451_1: "bf16[128]", arg452_1: "bf16[128]", arg453_1: "bf16[512, 128]", arg454_1: "bf16[512]", arg455_1: "bf16[128, 512]", arg456_1: "bf16[128]", arg457_1: "bf16[128]", arg458_1: "bf16[128]", arg459_1: "bf16[512, 128]", arg460_1: "bf16[512]", arg461_1: "bf16[128, 512]", arg462_1: "bf16[128]", arg463_1: "bf16[128]", arg464_1: "bf16[128]", arg465_1: "bf16[512, 128]", arg466_1: "bf16[512]", arg467_1: "bf16[512]", arg468_1: "bf16[512]", arg469_1: "bf16[128, 512]", arg470_1: "bf16[128]", arg471_1: "bf16[128]", arg472_1: "bf16[128]", arg473_1: "bf16[128, 128]", arg474_1: "bf16[128]", arg475_1: "bf16[128, 128]", arg476_1: "bf16[128]", arg477_1: "bf16[128, 512]", arg478_1: "bf16[128]", arg479_1: "bf16[128, 128]", arg480_1: "bf16[128]", arg481_1: "bf16[128, 512]", arg482_1: "bf16[128]", arg483_1: "bf16[128]", arg484_1: "bf16[128]", arg485_1: "bf16[128]", arg486_1: "bf16[128]", arg487_1: "bf16[512, 128]", arg488_1: "bf16[512]", arg489_1: "bf16[128, 512]", arg490_1: "bf16[128]", arg491_1: "bf16[128]", arg492_1: "bf16[128]", arg493_1: "bf16[512, 128]", arg494_1: "bf16[512]", arg495_1: "bf16[128, 512]", arg496_1: "bf16[128]", arg497_1: "bf16[128]", arg498_1: "bf16[128]", arg499_1: "bf16[512, 128]", arg500_1: "bf16[512]", arg501_1: "bf16[128, 512]", arg502_1: "bf16[128]", arg503_1: "bf16[128]", arg504_1: "bf16[128]", arg505_1: "bf16[512, 128]", arg506_1: "bf16[512]", arg507_1: "bf16[128, 512]", arg508_1: "bf16[128]", arg509_1: "bf16[128]", arg510_1: "bf16[128]", arg511_1: "bf16[512, 128]", arg512_1: "bf16[512]", arg513_1: "bf16[512]", arg514_1: "bf16[512]", arg515_1: "bf16[128, 512]", arg516_1: "bf16[128]", arg517_1: "bf16[128]", arg518_1: "bf16[128]", arg519_1: "bf16[128, 128]", arg520_1: "bf16[128]", arg521_1: "bf16[128, 128]", arg522_1: "bf16[128]", arg523_1: "bf16[128, 512]", arg524_1: "bf16[128]", arg525_1: "bf16[128, 128]", arg526_1: "bf16[128]", arg527_1: "bf16[128, 512]", arg528_1: "bf16[128]", arg529_1: "bf16[128]", arg530_1: "bf16[128]", arg531_1: "bf16[128]", arg532_1: "bf16[128]", arg533_1: "bf16[512, 128]", arg534_1: "bf16[512]", arg535_1: "bf16[128, 512]", arg536_1: "bf16[128]", arg537_1: "bf16[128]", arg538_1: "bf16[128]", arg539_1: "bf16[512, 128]", arg540_1: "bf16[512]", arg541_1: "bf16[128, 512]", arg542_1: "bf16[128]", arg543_1: "bf16[128]", arg544_1: "bf16[128]", arg545_1: "bf16[512, 128]", arg546_1: "bf16[512]", arg547_1: "bf16[128, 512]", arg548_1: "bf16[128]", arg549_1: "bf16[128]", arg550_1: "bf16[128]", arg551_1: "bf16[512, 128]", arg552_1: "bf16[512]", arg553_1: "bf16[128, 512]", arg554_1: "bf16[128]", arg555_1: "bf16[128]", arg556_1: "bf16[128]", arg557_1: "bf16[512, 128]", arg558_1: "bf16[512]", arg559_1: "bf16[512]", arg560_1: "bf16[512]", arg561_1: "bf16[128, 512]", arg562_1: "bf16[128]", arg563_1: "bf16[128]", arg564_1: "bf16[128]", arg565_1: "bf16[128, 128]", arg566_1: "bf16[128]", arg567_1: "bf16[128, 128]", arg568_1: "bf16[128]", arg569_1: "bf16[128, 512]", arg570_1: "bf16[128]", arg571_1: "bf16[128, 128]", arg572_1: "bf16[128]", arg573_1: "bf16[128, 512]", arg574_1: "bf16[128]", arg575_1: "bf16[128]", arg576_1: "bf16[128]", arg577_1: "bf16[128]", arg578_1: "bf16[128]", arg579_1: "bf16[512, 128]", arg580_1: "bf16[512]", arg581_1: "bf16[128, 512]", arg582_1: "bf16[128]", arg583_1: "bf16[128]", arg584_1: "bf16[128]", arg585_1: "bf16[512, 128]", arg586_1: "bf16[512]", arg587_1: "bf16[128, 512]", arg588_1: "bf16[128]", arg589_1: "bf16[128]", arg590_1: "bf16[128]", arg591_1: "bf16[512, 128]", arg592_1: "bf16[512]", arg593_1: "bf16[128, 512]", arg594_1: "bf16[128]", arg595_1: "bf16[128]", arg596_1: "bf16[128]", arg597_1: "bf16[512, 128]", arg598_1: "bf16[512]", arg599_1: "bf16[128, 512]", arg600_1: "bf16[128]", arg601_1: "bf16[128]", arg602_1: "bf16[128]", arg603_1: "bf16[512, 128]", arg604_1: "bf16[512]", arg605_1: "bf16[512]", arg606_1: "bf16[512]", arg607_1: "bf16[128, 512]", arg608_1: "bf16[128]", arg609_1: "bf16[128]", arg610_1: "bf16[128]", arg611_1: "bf16[128, 128]", arg612_1: "bf16[128]", arg613_1: "bf16[128, 128]", arg614_1: "bf16[128]", arg615_1: "bf16[128, 512]", arg616_1: "bf16[128]", arg617_1: "bf16[128, 128]", arg618_1: "bf16[128]", arg619_1: "bf16[128, 512]", arg620_1: "bf16[128]", arg621_1: "bf16[128]", arg622_1: "bf16[128]", arg623_1: "bf16[128]", arg624_1: "bf16[128]", arg625_1: "bf16[512, 128]", arg626_1: "bf16[512]", arg627_1: "bf16[128, 512]", arg628_1: "bf16[128]", arg629_1: "bf16[128]", arg630_1: "bf16[128]", arg631_1: "bf16[512, 128]", arg632_1: "bf16[512]", arg633_1: "bf16[128, 512]", arg634_1: "bf16[128]", arg635_1: "bf16[128]", arg636_1: "bf16[128]", arg637_1: "bf16[512, 128]", arg638_1: "bf16[512]", arg639_1: "bf16[128, 512]", arg640_1: "bf16[128]", arg641_1: "bf16[128]", arg642_1: "bf16[128]", arg643_1: "bf16[512, 128]", arg644_1: "bf16[512]", arg645_1: "bf16[128, 512]", arg646_1: "bf16[128]", arg647_1: "bf16[128]", arg648_1: "bf16[128]", arg649_1: "bf16[512, 128]", arg650_1: "bf16[512]", arg651_1: "bf16[512]", arg652_1: "bf16[512]", arg653_1: "bf16[128, 512]", arg654_1: "bf16[128]", arg655_1: "bf16[128]", arg656_1: "bf16[128]", arg657_1: "bf16[128, 128]", arg658_1: "bf16[128]", arg659_1: "bf16[128, 128]", arg660_1: "bf16[128]", arg661_1: "bf16[128, 512]", arg662_1: "bf16[128]", arg663_1: "bf16[128, 128]", arg664_1: "bf16[128]", arg665_1: "bf16[128, 512]", arg666_1: "bf16[128]", arg667_1: "bf16[128]", arg668_1: "bf16[128]", arg669_1: "bf16[128]", arg670_1: "bf16[128]", arg671_1: "bf16[512, 128]", arg672_1: "bf16[512]", arg673_1: "bf16[128, 512]", arg674_1: "bf16[128]", arg675_1: "bf16[128]", arg676_1: "bf16[128]", arg677_1: "bf16[512, 128]", arg678_1: "bf16[512]", arg679_1: "bf16[128, 512]", arg680_1: "bf16[128]", arg681_1: "bf16[128]", arg682_1: "bf16[128]", arg683_1: "bf16[512, 128]", arg684_1: "bf16[512]", arg685_1: "bf16[128, 512]", arg686_1: "bf16[128]", arg687_1: "bf16[128]", arg688_1: "bf16[128]", arg689_1: "bf16[512, 128]", arg690_1: "bf16[512]", arg691_1: "bf16[128, 512]", arg692_1: "bf16[128]", arg693_1: "bf16[128]", arg694_1: "bf16[128]", arg695_1: "bf16[512, 128]", arg696_1: "bf16[512]", arg697_1: "bf16[512]", arg698_1: "bf16[512]", arg699_1: "bf16[128, 512]", arg700_1: "bf16[128]", arg701_1: "bf16[128]", arg702_1: "bf16[128]", arg703_1: "bf16[128, 128]", arg704_1: "bf16[128]", arg705_1: "bf16[128, 128]", arg706_1: "bf16[128]", arg707_1: "bf16[128, 512]", arg708_1: "bf16[128]", arg709_1: "bf16[128, 128]", arg710_1: "bf16[128]", arg711_1: "bf16[128, 512]", arg712_1: "bf16[128]", arg713_1: "bf16[128]", arg714_1: "bf16[128]", arg715_1: "bf16[128]", arg716_1: "bf16[128]", arg717_1: "bf16[512, 128]", arg718_1: "bf16[512]", arg719_1: "bf16[128, 512]", arg720_1: "bf16[128]", arg721_1: "bf16[128]", arg722_1: "bf16[128]", arg723_1: "bf16[512, 128]", arg724_1: "bf16[512]", arg725_1: "bf16[128, 512]", arg726_1: "bf16[128]", arg727_1: "bf16[128]", arg728_1: "bf16[128]", arg729_1: "bf16[512, 128]", arg730_1: "bf16[512]", arg731_1: "bf16[128, 512]", arg732_1: "bf16[128]", arg733_1: "bf16[128]", arg734_1: "bf16[128]", arg735_1: "bf16[512, 128]", arg736_1: "bf16[512]", arg737_1: "bf16[128, 512]", arg738_1: "bf16[128]", arg739_1: "bf16[128]", arg740_1: "bf16[128]", arg741_1: "bf16[512, 128]", arg742_1: "bf16[512]", arg743_1: "bf16[512]", arg744_1: "bf16[512]", arg745_1: "bf16[128, 512]", arg746_1: "bf16[128]", arg747_1: "bf16[128]", arg748_1: "bf16[128]", arg749_1: "bf16[128, 128]", arg750_1: "bf16[128]", arg751_1: "bf16[128, 128]", arg752_1: "bf16[128]", arg753_1: "bf16[128, 512]", arg754_1: "bf16[128]", arg755_1: "bf16[128, 128]", arg756_1: "bf16[128]", arg757_1: "bf16[128, 512]", arg758_1: "bf16[128]", arg759_1: "bf16[128]", arg760_1: "bf16[128]", arg761_1: "bf16[128]", arg762_1: "bf16[128]", arg763_1: "bf16[512, 128]", arg764_1: "bf16[512]", arg765_1: "bf16[128, 512]", arg766_1: "bf16[128]", arg767_1: "bf16[128]", arg768_1: "bf16[128]", arg769_1: "bf16[512, 128]", arg770_1: "bf16[512]", arg771_1: "bf16[128, 512]", arg772_1: "bf16[128]", arg773_1: "bf16[128]", arg774_1: "bf16[128]", arg775_1: "bf16[512, 128]", arg776_1: "bf16[512]", arg777_1: "bf16[128, 512]", arg778_1: "bf16[128]", arg779_1: "bf16[128]", arg780_1: "bf16[128]", arg781_1: "bf16[512, 128]", arg782_1: "bf16[512]", arg783_1: "bf16[128, 512]", arg784_1: "bf16[128]", arg785_1: "bf16[128]", arg786_1: "bf16[128]", arg787_1: "bf16[512, 128]", arg788_1: "bf16[512]", arg789_1: "bf16[512]", arg790_1: "bf16[512]", arg791_1: "bf16[128, 512]", arg792_1: "bf16[128]", arg793_1: "bf16[128]", arg794_1: "bf16[128]", arg795_1: "bf16[128, 128]", arg796_1: "bf16[128]", arg797_1: "bf16[128, 128]", arg798_1: "bf16[128]", arg799_1: "bf16[128, 512]", arg800_1: "bf16[128]", arg801_1: "bf16[128, 128]", arg802_1: "bf16[128]", arg803_1: "bf16[128, 512]", arg804_1: "bf16[128]", arg805_1: "bf16[128]", arg806_1: "bf16[128]", arg807_1: "bf16[128]", arg808_1: "bf16[128]", arg809_1: "bf16[512, 128]", arg810_1: "bf16[512]", arg811_1: "bf16[128, 512]", arg812_1: "bf16[128]", arg813_1: "bf16[128]", arg814_1: "bf16[128]", arg815_1: "bf16[512, 128]", arg816_1: "bf16[512]", arg817_1: "bf16[128, 512]", arg818_1: "bf16[128]", arg819_1: "bf16[128]", arg820_1: "bf16[128]", arg821_1: "bf16[512, 128]", arg822_1: "bf16[512]", arg823_1: "bf16[128, 512]", arg824_1: "bf16[128]", arg825_1: "bf16[128]", arg826_1: "bf16[128]", arg827_1: "bf16[512, 128]", arg828_1: "bf16[512]", arg829_1: "bf16[128, 512]", arg830_1: "bf16[128]", arg831_1: "bf16[128]", arg832_1: "bf16[128]", arg833_1: "bf16[512, 128]", arg834_1: "bf16[512]", arg835_1: "bf16[512]", arg836_1: "bf16[512]", arg837_1: "bf16[128, 512]", arg838_1: "bf16[128]", arg839_1: "bf16[128]", arg840_1: "bf16[128]", arg841_1: "bf16[128, 128]", arg842_1: "bf16[128]", arg843_1: "bf16[128, 128]", arg844_1: "bf16[128]", arg845_1: "bf16[128, 512]", arg846_1: "bf16[128]", arg847_1: "bf16[128, 128]", arg848_1: "bf16[128]", arg849_1: "bf16[128, 512]", arg850_1: "bf16[128]", arg851_1: "bf16[128]", arg852_1: "bf16[128]", arg853_1: "bf16[128]", arg854_1: "bf16[128]", arg855_1: "bf16[512, 128]", arg856_1: "bf16[512]", arg857_1: "bf16[128, 512]", arg858_1: "bf16[128]", arg859_1: "bf16[128]", arg860_1: "bf16[128]", arg861_1: "bf16[512, 128]", arg862_1: "bf16[512]", arg863_1: "bf16[128, 512]", arg864_1: "bf16[128]", arg865_1: "bf16[128]", arg866_1: "bf16[128]", arg867_1: "bf16[512, 128]", arg868_1: "bf16[512]", arg869_1: "bf16[128, 512]", arg870_1: "bf16[128]", arg871_1: "bf16[128]", arg872_1: "bf16[128]", arg873_1: "bf16[512, 128]", arg874_1: "bf16[512]", arg875_1: "bf16[128, 512]", arg876_1: "bf16[128]", arg877_1: "bf16[128]", arg878_1: "bf16[128]", arg879_1: "bf16[512, 128]", arg880_1: "bf16[512]", arg881_1: "bf16[512]", arg882_1: "bf16[512]", arg883_1: "bf16[128, 512]", arg884_1: "bf16[128]", arg885_1: "bf16[128]", arg886_1: "bf16[128]", arg887_1: "bf16[128, 128]", arg888_1: "bf16[128]", arg889_1: "bf16[128, 128]", arg890_1: "bf16[128]", arg891_1: "bf16[128, 512]", arg892_1: "bf16[128]", arg893_1: "bf16[128, 128]", arg894_1: "bf16[128]", arg895_1: "bf16[128, 512]", arg896_1: "bf16[128]", arg897_1: "bf16[128]", arg898_1: "bf16[128]", arg899_1: "bf16[128]", arg900_1: "bf16[128]", arg901_1: "bf16[512, 128]", arg902_1: "bf16[512]", arg903_1: "bf16[128, 512]", arg904_1: "bf16[128]", arg905_1: "bf16[128]", arg906_1: "bf16[128]", arg907_1: "bf16[512, 128]", arg908_1: "bf16[512]", arg909_1: "bf16[128, 512]", arg910_1: "bf16[128]", arg911_1: "bf16[128]", arg912_1: "bf16[128]", arg913_1: "bf16[512, 128]", arg914_1: "bf16[512]", arg915_1: "bf16[128, 512]", arg916_1: "bf16[128]", arg917_1: "bf16[128]", arg918_1: "bf16[128]", arg919_1: "bf16[512, 128]", arg920_1: "bf16[512]", arg921_1: "bf16[128, 512]", arg922_1: "bf16[128]", arg923_1: "bf16[128]", arg924_1: "bf16[128]", arg925_1: "bf16[512, 128]", arg926_1: "bf16[512]", arg927_1: "bf16[512]", arg928_1: "bf16[512]", arg929_1: "bf16[128, 512]", arg930_1: "bf16[128]", arg931_1: "bf16[128]", arg932_1: "bf16[128]", arg933_1: "bf16[128, 128]", arg934_1: "bf16[128]", arg935_1: "bf16[128, 128]", arg936_1: "bf16[128]", arg937_1: "bf16[128, 512]", arg938_1: "bf16[128]", arg939_1: "bf16[128, 128]", arg940_1: "bf16[128]", arg941_1: "bf16[128, 512]", arg942_1: "bf16[128]", arg943_1: "bf16[128]", arg944_1: "bf16[128]", arg945_1: "bf16[128]", arg946_1: "bf16[128]", arg947_1: "bf16[512, 128]", arg948_1: "bf16[512]", arg949_1: "bf16[128, 512]", arg950_1: "bf16[128]", arg951_1: "bf16[128]", arg952_1: "bf16[128]", arg953_1: "bf16[512, 128]", arg954_1: "bf16[512]", arg955_1: "bf16[128, 512]", arg956_1: "bf16[128]", arg957_1: "bf16[128]", arg958_1: "bf16[128]", arg959_1: "bf16[512, 128]", arg960_1: "bf16[512]", arg961_1: "bf16[128, 512]", arg962_1: "bf16[128]", arg963_1: "bf16[128]", arg964_1: "bf16[128]", arg965_1: "bf16[512, 128]", arg966_1: "bf16[512]", arg967_1: "bf16[128, 512]", arg968_1: "bf16[128]", arg969_1: "bf16[128]", arg970_1: "bf16[128]", arg971_1: "bf16[512, 128]", arg972_1: "bf16[512]", arg973_1: "bf16[512]", arg974_1: "bf16[512]", arg975_1: "bf16[128, 512]", arg976_1: "bf16[128]", arg977_1: "bf16[128]", arg978_1: "bf16[128]", arg979_1: "bf16[128, 128]", arg980_1: "bf16[128]", arg981_1: "bf16[128, 128]", arg982_1: "bf16[128]", arg983_1: "bf16[128, 512]", arg984_1: "bf16[128]", arg985_1: "bf16[128, 128]", arg986_1: "bf16[128]", arg987_1: "bf16[128, 512]", arg988_1: "bf16[128]", arg989_1: "bf16[128]", arg990_1: "bf16[128]", arg991_1: "bf16[128]", arg992_1: "bf16[128]", arg993_1: "bf16[512, 128]", arg994_1: "bf16[512]", arg995_1: "bf16[128, 512]", arg996_1: "bf16[128]", arg997_1: "bf16[128]", arg998_1: "bf16[128]", arg999_1: "bf16[512, 128]", arg1000_1: "bf16[512]", arg1001_1: "bf16[128, 512]", arg1002_1: "bf16[128]", arg1003_1: "bf16[128]", arg1004_1: "bf16[128]", arg1005_1: "bf16[512, 128]", arg1006_1: "bf16[512]", arg1007_1: "bf16[128, 512]", arg1008_1: "bf16[128]", arg1009_1: "bf16[128]", arg1010_1: "bf16[128]", arg1011_1: "bf16[512, 128]", arg1012_1: "bf16[512]", arg1013_1: "bf16[128, 512]", arg1014_1: "bf16[128]", arg1015_1: "bf16[128]", arg1016_1: "bf16[128]", arg1017_1: "bf16[512, 128]", arg1018_1: "bf16[512]", arg1019_1: "bf16[512]", arg1020_1: "bf16[512]", arg1021_1: "bf16[128, 512]", arg1022_1: "bf16[128]", arg1023_1: "bf16[128]", arg1024_1: "bf16[128]", arg1025_1: "bf16[128, 128]", arg1026_1: "bf16[128]", arg1027_1: "bf16[128, 128]", arg1028_1: "bf16[128]", arg1029_1: "bf16[128, 512]", arg1030_1: "bf16[128]", arg1031_1: "bf16[128, 128]", arg1032_1: "bf16[128]", arg1033_1: "bf16[128, 512]", arg1034_1: "bf16[128]", arg1035_1: "bf16[128]", arg1036_1: "bf16[128]", arg1037_1: "bf16[128]", arg1038_1: "bf16[128]", arg1039_1: "bf16[512, 128]", arg1040_1: "bf16[512]", arg1041_1: "bf16[128, 512]", arg1042_1: "bf16[128]", arg1043_1: "bf16[128]", arg1044_1: "bf16[128]", arg1045_1: "bf16[512, 128]", arg1046_1: "bf16[512]", arg1047_1: "bf16[128, 512]", arg1048_1: "bf16[128]", arg1049_1: "bf16[128]", arg1050_1: "bf16[128]", arg1051_1: "bf16[512, 128]", arg1052_1: "bf16[512]", arg1053_1: "bf16[128, 512]", arg1054_1: "bf16[128]", arg1055_1: "bf16[128]", arg1056_1: "bf16[128]", arg1057_1: "bf16[512, 128]", arg1058_1: "bf16[512]", arg1059_1: "bf16[128, 512]", arg1060_1: "bf16[128]", arg1061_1: "bf16[128]", arg1062_1: "bf16[128]", arg1063_1: "bf16[512, 128]", arg1064_1: "bf16[512]", arg1065_1: "bf16[512]", arg1066_1: "bf16[512]", arg1067_1: "bf16[128, 512]", arg1068_1: "bf16[128]", arg1069_1: "bf16[128]", arg1070_1: "bf16[128]", arg1071_1: "bf16[128, 128]", arg1072_1: "bf16[128]", arg1073_1: "bf16[128, 128]", arg1074_1: "bf16[128]", arg1075_1: "bf16[128, 512]", arg1076_1: "bf16[128]", arg1077_1: "bf16[128, 128]", arg1078_1: "bf16[128]", arg1079_1: "bf16[128, 512]", arg1080_1: "bf16[128]", arg1081_1: "bf16[128]", arg1082_1: "bf16[128]", arg1083_1: "bf16[128]", arg1084_1: "bf16[128]", arg1085_1: "bf16[512, 128]", arg1086_1: "bf16[512]", arg1087_1: "bf16[128, 512]", arg1088_1: "bf16[128]", arg1089_1: "bf16[128]", arg1090_1: "bf16[128]", arg1091_1: "bf16[512, 128]", arg1092_1: "bf16[512]", arg1093_1: "bf16[128, 512]", arg1094_1: "bf16[128]", arg1095_1: "bf16[128]", arg1096_1: "bf16[128]", arg1097_1: "bf16[512, 128]", arg1098_1: "bf16[512]", arg1099_1: "bf16[128, 512]", arg1100_1: "bf16[128]", arg1101_1: "bf16[128]", arg1102_1: "bf16[128]", arg1103_1: "bf16[512, 128]", arg1104_1: "bf16[512]", arg1105_1: "bf16[128, 512]", arg1106_1: "bf16[128]", arg1107_1: "bf16[128]", arg1108_1: "bf16[128]", arg1109_1: "bf16[512, 128]", arg1110_1: "bf16[512]", arg1111_1: "bf16[512]", arg1112_1: "bf16[512]", arg1113_1: "bf16[512, 512]", arg1114_1: "bf16[512]", arg1115_1: "i64[256, 128]", arg1116_1: "bf16[512]", arg1117_1: "bf16[512]", arg1118_1: "bf16[384, 30522]", arg1119_1: "bf16[30522]", _shape_param_0, _shape_param_1, _shape_param_2, _shape_param_3, _shape_param_4, _shape_param_5, _shape_param_6, _shape_param_7, _shape_param_8, _shape_param_9, _shape_param_10, _shape_param_11, _shape_param_12, _shape_param_13, _shape_param_14, _shape_param_15, _shape_param_16, _shape_param_17, _shape_param_18, _shape_param_19, _shape_param_20, _shape_param_21, _shape_param_22, _shape_param_23, _shape_param_24, _shape_param_25, _shape_param_26, _shape_param_27, _shape_param_28, _shape_param_29, _shape_param_30, _shape_param_31, _shape_param_32, _shape_param_33, _shape_param_34, _shape_param_35, _shape_param_36, _shape_param_37, _shape_param_38, _shape_param_39, _shape_param_40, _shape_param_41, _shape_param_42, _shape_param_43, _shape_param_44, _shape_param_45, _shape_param_46, _shape_param_47, _shape_param_48, _shape_param_49, _shape_param_50, _shape_param_51, _shape_param_52, _shape_param_53, _shape_param_54, _shape_param_55, _shape_param_56, _shape_param_57, _shape_param_58, _shape_param_59, _shape_param_60, _shape_param_61, _shape_param_62, _shape_param_63, _shape_param_64, _shape_param_65, _shape_param_66, _shape_param_67, _shape_param_68, _shape_param_69, _shape_param_70, _shape_param_71, _shape_param_72, _shape_param_73, _shape_param_74, _shape_param_75, _shape_param_76, _shape_param_77, _shape_param_78, _shape_param_79, _shape_param_80, _shape_param_81, _shape_param_82, _shape_param_83, _shape_param_84, _shape_param_85, _shape_param_86, _shape_param_87, _shape_param_88, _shape_param_89, _shape_param_90, _shape_param_91, _shape_param_92, _shape_param_93, _shape_param_94, _shape_param_95, _shape_param_96, _shape_param_97, _shape_param_98, _shape_param_99, _shape_param_100, _shape_param_101, _shape_param_102, _shape_param_103, _shape_param_104, _shape_param_105, _shape_param_106, _shape_param_107, _shape_param_108, _shape_param_109, _shape_param_110, _shape_param_111, _shape_param_112, _shape_param_113, _shape_param_114, _shape_param_115, _shape_param_116, _shape_param_117, _shape_param_118, _shape_param_119, _shape_param_120, _shape_param_121, _shape_param_122, _shape_param_123, _shape_param_124, _shape_param_125, _shape_param_126, _shape_param_127, _shape_param_128, _shape_param_129, _shape_param_130, _shape_param_131, _shape_param_132, _shape_param_133, _shape_param_134, _shape_param_135, _shape_param_136, _shape_param_137, _shape_param_138, _shape_param_139, _shape_param_140, _shape_param_141, _shape_param_142, _shape_param_143, _shape_param_144, _shape_param_145, _shape_param_146, _shape_param_147, _shape_param_148, _shape_param_149, _shape_param_150, _shape_param_151, _shape_param_152, _shape_param_153, _shape_param_154, _shape_param_155, _shape_param_156, _shape_param_157, _shape_param_158, _shape_param_159, _shape_param_160, _shape_param_161, _shape_param_162, _shape_param_163, _shape_param_164, _shape_param_165, _shape_param_166, _shape_param_167, _shape_param_168, _shape_param_169, _shape_param_170, _shape_param_171, _shape_param_172, _shape_param_173, _shape_param_174, _shape_param_175, _shape_param_176, _shape_param_177, _shape_param_178, _shape_param_179, _shape_param_180, _shape_param_181, _shape_param_182, _shape_param_183, _shape_param_184, _shape_param_185, _shape_param_186, _shape_param_187, _shape_param_188, _shape_param_189, _shape_param_190, _shape_param_191, _shape_param_192, _shape_param_193, _shape_param_194, _shape_param_195, _shape_param_196, _shape_param_197, _shape_param_198, _shape_param_199, _shape_param_200, _shape_param_201, _shape_param_202, _shape_param_203, _shape_param_204, _shape_param_205, _shape_param_206, _shape_param_207, _shape_param_208, _shape_param_209, _shape_param_210, _shape_param_211, _shape_param_212, _shape_param_213, _shape_param_214, _shape_param_215, _shape_param_216, _shape_param_217, _shape_param_218, _shape_param_219, _shape_param_220, _shape_param_221, _shape_param_222, _shape_param_223, _shape_param_224, _shape_param_225, _shape_param_226, _shape_param_227, _shape_param_228, _shape_param_229, _shape_param_230, _shape_param_231, _shape_param_232, _shape_param_233, _shape_param_234, _shape_param_235, _shape_param_236, _shape_param_237, _shape_param_238, _shape_param_239, _shape_param_240, _shape_param_241, _shape_param_242, _shape_param_243, _shape_param_244, _shape_param_245, _shape_param_246, _shape_param_247, _shape_param_248, _shape_param_249, _shape_param_250, _shape_param_251, _shape_param_252, _shape_param_253, _shape_param_254, _shape_param_255, _shape_param_256, _shape_param_257, _shape_param_258, _shape_param_259, _shape_param_260, _shape_param_261, _shape_param_262, _shape_param_263, _shape_param_264, _shape_param_265, _shape_param_266, _shape_param_267, _shape_param_268, _shape_param_269, _shape_param_270, _shape_param_271, _shape_param_272, _shape_param_273, _shape_param_274, _shape_param_275, _shape_param_276, _shape_param_277, _shape_param_278, _shape_param_279, _shape_param_280, _shape_param_281, _shape_param_282, _shape_param_283, _shape_param_284, _shape_param_285, _shape_param_286, _shape_param_287, _shape_param_288, _shape_param_289, _shape_param_290, _shape_param_291, _shape_param_292, _shape_param_293, _shape_param_294, _shape_param_295, _shape_param_296, _shape_param_297, _shape_param_298, _shape_param_299, _shape_param_300, _shape_param_301, _shape_param_302, _shape_param_303, _shape_param_304, _shape_param_305, _shape_param_306, _shape_param_307, _shape_param_308, _shape_param_309, _shape_param_310, _shape_param_311, _shape_param_312, _shape_param_313, _shape_param_314, _shape_param_315, _shape_param_316, _shape_param_317, _shape_param_318, _shape_param_319, _shape_param_320, _shape_param_321, _shape_param_322, _shape_param_323, _shape_param_324, _shape_param_325, _shape_param_326, _shape_param_327, _shape_param_328, _shape_param_329, _shape_param_330, _shape_param_331, _shape_param_332, _shape_param_333, _shape_param_334, _shape_param_335, _shape_param_336, _shape_param_337, _shape_param_338, _shape_param_339, _shape_param_340, _shape_param_341, _shape_param_342, _shape_param_343, _shape_param_344, _shape_param_345, _shape_param_346, _shape_param_347, _shape_param_348, _shape_param_349, _shape_param_350, _shape_param_351, _shape_param_352, _shape_param_353, _shape_param_354, _shape_param_355, _shape_param_356, _shape_param_357, _shape_param_358, _shape_param_359, _shape_param_360, _shape_param_361, _shape_param_362, _shape_param_363, _shape_param_364, _shape_param_365, _shape_param_366, _shape_param_367, _shape_param_368, _shape_param_369, _shape_param_370, _shape_param_371, _shape_param_372, _shape_param_373, _shape_param_374, _shape_param_375, _shape_param_376, _shape_param_377, _shape_param_378, _shape_param_379, _shape_param_380, _shape_param_381, _shape_param_382, _shape_param_383, _shape_param_384, _shape_param_385, _shape_param_386, _shape_param_387, _shape_param_388, _shape_param_389, _shape_param_390, _shape_param_391, _shape_param_392, _shape_param_393, _shape_param_394, _shape_param_395, _shape_param_396, _shape_param_397, _shape_param_398, _shape_param_399, _shape_param_400, _shape_param_401, _shape_param_402, _shape_param_403, _shape_param_404, _shape_param_405, _shape_param_406, _shape_param_407, _shape_param_408, _shape_param_409, _shape_param_410, _shape_param_411, _shape_param_412, _shape_param_413, _shape_param_414, _shape_param_415, _shape_param_416, _shape_param_417, _shape_param_418, _shape_param_419, _shape_param_420, _shape_param_421, _shape_param_422, _shape_param_423, _shape_param_424, _shape_param_425, _shape_param_426, _shape_param_427, _shape_param_428, _shape_param_429, _shape_param_430, _shape_param_431, _shape_param_432, _shape_param_433, _shape_param_434, _shape_param_435, _shape_param_436, _shape_param_437, _shape_param_438, _shape_param_439, _shape_param_440, _shape_param_441, _shape_param_442, _shape_param_443, _shape_param_444, _shape_param_445, _shape_param_446, _shape_param_447, _shape_param_448, _shape_param_449, _shape_param_450, _shape_param_451, _shape_param_452, _shape_param_453, _shape_param_454, _shape_param_455, _shape_param_456, _shape_param_457, _shape_param_458, _shape_param_459, _shape_param_460, _shape_param_461, _shape_param_462, _shape_param_463, _shape_param_464, _shape_param_465, _shape_param_466, _shape_param_467, _shape_param_468, _shape_param_469, _shape_param_470, _shape_param_471, _shape_param_472, _shape_param_473, _shape_param_474, _shape_param_475, _shape_param_476, _shape_param_477, _shape_param_478, _shape_param_479, _shape_param_480, _shape_param_481, _shape_param_482, _shape_param_483, _shape_param_484, _shape_param_485, _shape_param_486, _shape_param_487, _shape_param_488, _shape_param_489, _shape_param_490, _shape_param_491, _shape_param_492, _shape_param_493, _shape_param_494, _shape_param_495, _shape_param_496, _shape_param_497, _shape_param_498, _shape_param_499, _shape_param_500, _shape_param_501, _shape_param_502, _shape_param_503, _shape_param_504, _shape_param_505, _shape_param_506, _shape_param_507, _shape_param_508, _shape_param_509, _shape_param_510, _shape_param_511, _shape_param_512, _shape_param_513, _shape_param_514, _shape_param_515, _shape_param_516, _shape_param_517, _shape_param_518, _shape_param_519, _shape_param_520, _shape_param_521, _shape_param_522, _shape_param_523, _shape_param_524, _shape_param_525, _shape_param_526, _shape_param_527, _shape_param_528, _shape_param_529, _shape_param_530, _shape_param_531, _shape_param_532, _shape_param_533, _shape_param_534, _shape_param_535, _shape_param_536, _shape_param_537, _shape_param_538, _shape_param_539, _shape_param_540, _shape_param_541, _shape_param_542, _shape_param_543, _shape_param_544, _shape_param_545, _shape_param_546, _shape_param_547, _shape_param_548, _shape_param_549, _shape_param_550, _shape_param_551, _shape_param_552, _shape_param_553, _shape_param_554, _shape_param_555, _shape_param_556, _shape_param_557, _shape_param_558, _shape_param_559, _shape_param_560, _shape_param_561, _shape_param_562, _shape_param_563, _shape_param_564, _shape_param_565, _shape_param_566, _shape_param_567, _shape_param_568, _shape_param_569, _shape_param_570, _shape_param_571, _shape_param_572, _shape_param_573, _shape_param_574, _shape_param_575, _shape_param_576, _shape_param_577, _shape_param_578, _shape_param_579, _shape_param_580, _shape_param_581, _shape_param_582, _shape_param_583, _shape_param_584, _shape_param_585, _shape_param_586, _shape_param_587, _shape_param_588, _shape_param_589, _shape_param_590, _shape_param_591, _shape_param_592, _shape_param_593, _shape_param_594, _shape_param_595, _shape_param_596, _shape_param_597, _shape_param_598, _shape_param_599, _shape_param_600, _shape_param_601, _shape_param_602, _shape_param_603, _shape_param_604, _shape_param_605, _shape_param_606, _shape_param_607, _shape_param_608, _shape_param_609, _shape_param_610, _shape_param_611, _shape_param_612, _shape_param_613, _shape_param_614, _shape_param_615, _shape_param_616, _shape_param_617, _shape_param_618, _shape_param_619, _shape_param_620, _shape_param_621, _shape_param_622, _shape_param_623, _shape_param_624, _shape_param_625, _shape_param_626, _shape_param_627, _shape_param_628, _shape_param_629, _shape_param_630, _shape_param_631, _shape_param_632, _shape_param_633, _shape_param_634, _shape_param_635, _shape_param_636, _shape_param_637, _shape_param_638, _shape_param_639, _shape_param_640, _shape_param_641, _shape_param_642, _shape_param_643, _shape_param_644, _shape_param_645, _shape_param_646, _shape_param_647, _shape_param_648, _shape_param_649, _shape_param_650, _shape_param_651, _shape_param_652, _shape_param_653, _shape_param_654, _shape_param_655, _shape_param_656, _shape_param_657, _shape_param_658, _shape_param_659, _shape_param_660, _shape_param_661, _shape_param_662, _shape_param_663, _shape_param_664, _shape_param_665, _shape_param_666, _shape_param_667, _shape_param_668, _shape_param_669, _shape_param_670, _shape_param_671, _shape_param_672, _shape_param_673, _shape_param_674, _shape_param_675, _shape_param_676, _shape_param_677, _shape_param_678, _shape_param_679, _shape_param_680, _shape_param_681, _shape_param_682, _shape_param_683, _shape_param_684, _shape_param_685, _shape_param_686, _shape_param_687, _shape_param_688, _shape_param_689, _shape_param_690, _shape_param_691, _shape_param_692, _shape_param_693, _shape_param_694, _shape_param_695, _shape_param_696, _shape_param_697, _shape_param_698, _shape_param_699, _shape_param_700, _shape_param_701, _shape_param_702, _shape_param_703, _shape_param_704, _shape_param_705, _shape_param_706, _shape_param_707, _shape_param_708, _shape_param_709, _shape_param_710, _shape_param_711, _shape_param_712, _shape_param_713, _shape_param_714, _shape_param_715, _shape_param_716, _shape_param_717, _shape_param_718, _shape_param_719, _shape_param_720, _shape_param_721, _shape_param_722, _shape_param_723, _shape_param_724, _shape_param_725, _shape_param_726, _shape_param_727, _shape_param_728, _shape_param_729, _shape_param_730, _shape_param_731, _shape_param_732, _shape_param_733, _shape_param_734, _shape_param_735, _shape_param_736, _shape_param_737, _shape_param_738, _shape_param_739, _shape_param_740, _shape_param_741, _shape_param_742, _shape_param_743, _shape_param_744, _shape_param_745, _shape_param_746, _shape_param_747, _shape_param_748, _shape_param_749, _shape_param_750, _shape_param_751, _shape_param_752, _shape_param_753, _shape_param_754, _shape_param_755, _shape_param_756, _shape_param_757, _shape_param_758, _shape_param_759, _shape_param_760, _shape_param_761, _shape_param_762, _shape_param_763, _shape_param_764, _shape_param_765, _shape_param_766, _shape_param_767, _shape_param_768, _shape_param_769, _shape_param_770, _shape_param_771, _shape_param_772, _shape_param_773, _shape_param_774, _shape_param_775, _shape_param_776, _shape_param_777, _shape_param_778, _shape_param_779, _shape_param_780, _shape_param_781, _shape_param_782, _shape_param_783, _shape_param_784, _shape_param_785, _shape_param_786, _shape_param_787, _shape_param_788, _shape_param_789, _shape_param_790, _shape_param_791, _shape_param_792, _shape_param_793, _shape_param_794, _shape_param_795, _shape_param_796, _shape_param_797, _shape_param_798, _shape_param_799, _shape_param_800, _shape_param_801, _shape_param_802, _shape_param_803, _shape_param_804, _shape_param_805, _shape_param_806, _shape_param_807, _shape_param_808, _shape_param_809, _shape_param_810, _shape_param_811, _shape_param_812, _shape_param_813, _shape_param_814, _shape_param_815, _shape_param_816, _shape_param_817, _shape_param_818, _shape_param_819, _shape_param_820, _shape_param_821, _shape_param_822, _shape_param_823, _shape_param_824, _shape_param_825, _shape_param_826, _shape_param_827, _shape_param_828, _shape_param_829, _shape_param_830, _shape_param_831, _shape_param_832, _shape_param_833, _shape_param_834, _shape_param_835, _shape_param_836, _shape_param_837, _shape_param_838, _shape_param_839, _shape_param_840, _shape_param_841, _shape_param_842, _shape_param_843, _shape_param_844, _shape_param_845, _shape_param_846, _shape_param_847, _shape_param_848, _shape_param_849, _shape_param_850, _shape_param_851, _shape_param_852, _shape_param_853, _shape_param_854, _shape_param_855, _shape_param_856, _shape_param_857, _shape_param_858, _shape_param_859, _shape_param_860, _shape_param_861, _shape_param_862, _shape_param_863, _shape_param_864, _shape_param_865, _shape_param_866, _shape_param_867, _shape_param_868, _shape_param_869, _shape_param_870, _shape_param_871, _shape_param_872, _shape_param_873, _shape_param_874, _shape_param_875, _shape_param_876, _shape_param_877, _shape_param_878, _shape_param_879, _shape_param_880, _shape_param_881, _shape_param_882, _shape_param_883, _shape_param_884, _shape_param_885, _shape_param_886, _shape_param_887, _shape_param_888, _shape_param_889, _shape_param_890, _shape_param_891, _shape_param_892, _shape_param_893, _shape_param_894, _shape_param_895, _shape_param_896, _shape_param_897, _shape_param_898, _shape_param_899, _shape_param_900, _shape_param_901, _shape_param_902, _shape_param_903, _shape_param_904, _shape_param_905, _shape_param_906, _shape_param_907, _shape_param_908, _shape_param_909, _shape_param_910, _shape_param_911, _shape_param_912, _shape_param_913, _shape_param_914, _shape_param_915, _shape_param_916, _shape_param_917, _shape_param_918, _shape_param_919, _shape_param_920, _shape_param_921, _shape_param_922, _shape_param_923, _shape_param_924, _shape_param_925, _shape_param_926, _shape_param_927, _shape_param_928, _shape_param_929, _shape_param_930, _shape_param_931, _shape_param_932, _shape_param_933, _shape_param_934, _shape_param_935, _shape_param_936, _shape_param_937, _shape_param_938, _shape_param_939, _shape_param_940, _shape_param_941, _shape_param_942, _shape_param_943, _shape_param_944, _shape_param_945, _shape_param_946, _shape_param_947, _shape_param_948, _shape_param_949, _shape_param_950, _shape_param_951, _shape_param_952, _shape_param_953, _shape_param_954, _shape_param_955, _shape_param_956, _shape_param_957, _shape_param_958, _shape_param_959, _shape_param_960, _shape_param_961, _shape_param_962, _shape_param_963, _shape_param_964, _shape_param_965, _shape_param_966, _shape_param_967, _shape_param_968, _shape_param_969, _shape_param_970, _shape_param_971, _shape_param_972, _shape_param_973, _shape_param_974, _shape_param_975, _shape_param_976, _shape_param_977, _shape_param_978, _shape_param_979, _shape_param_980, _shape_param_981, _shape_param_982, _shape_param_983, _shape_param_984, _shape_param_985, _shape_param_986, _shape_param_987, _shape_param_988, _shape_param_989, _shape_param_990, _shape_param_991, _shape_param_992, _shape_param_993, _shape_param_994, _shape_param_995, _shape_param_996, _shape_param_997, _shape_param_998, _shape_param_999, _shape_param_1000, _shape_param_1001, _shape_param_1002, _shape_param_1003, _shape_param_1004, _shape_param_1005, _shape_param_1006, _shape_param_1007, _shape_param_1008, _shape_param_1009, _shape_param_1010, _shape_param_1011, _shape_param_1012, _shape_param_1013, _shape_param_1014, _shape_param_1015, _shape_param_1016, _shape_param_1017, _shape_param_1018, _shape_param_1019, _shape_param_1020, _shape_param_1021, _shape_param_1022, _shape_param_1023, _shape_param_1024, _shape_param_1025, _shape_param_1026, _shape_param_1027, _shape_param_1028, _shape_param_1029, _shape_param_1030, _shape_param_1031, _shape_param_1032, _shape_param_1033, _shape_param_1034, _shape_param_1035, _shape_param_1036, _shape_param_1037, _shape_param_1038, _shape_param_1039, _shape_param_1040, _shape_param_1041, _shape_param_1042, _shape_param_1043, _shape_param_1044, _shape_param_1045, _shape_param_1046, _shape_param_1047, _shape_param_1048, _shape_param_1049, _shape_param_1050, _shape_param_1051, _shape_param_1052, _shape_param_1053, _shape_param_1054, _shape_param_1055, _shape_param_1056, _shape_param_1057, _shape_param_1058, _shape_param_1059, _shape_param_1060, _shape_param_1061, _shape_param_1062, _shape_param_1063, _shape_param_1064, _shape_param_1065, _shape_param_1066, _shape_param_1067, _shape_param_1068, _shape_param_1069, _shape_param_1070, _shape_param_1071, _shape_param_1072, _shape_param_1073, _shape_param_1074, _shape_param_1075, _shape_param_1076, _shape_param_1077, _shape_param_1078, _shape_param_1079, _shape_param_1080, _shape_param_1081, _shape_param_1082, _shape_param_1083, _shape_param_1084, _shape_param_1085, _shape_param_1086, _shape_param_1087, _shape_param_1088, _shape_param_1089):
        # No stacktrace found for following nodes
        embedding: "bf16[256, 128, 128]" = torch.ops.aten.embedding.default(arg0_1, arg1_1, 0);  arg1_1 = None
        slice_1: "bf16[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 1, 9223372036854775807)
        constant_pad_nd: "bf16[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_1, [0, 0, 0, 1, 0, 0], 0.0);  slice_1 = None
        slice_2: "bf16[256, 127, 128]" = torch.ops.aten.slice.Tensor(embedding, 1, 0, -1)
        constant_pad_nd_1: "bf16[256, 128, 128]" = torch.ops.aten.constant_pad_nd.default(slice_2, [0, 0, 1, 0, 0, 0], 0.0);  slice_2 = None
        cat: "bf16[256, 128, 384]" = torch.ops.aten.cat.default([constant_pad_nd, embedding, constant_pad_nd_1], 2);  constant_pad_nd = embedding = constant_pad_nd_1 = None
        view: "bf16[32768, 384]" = torch.ops.aten.view.default(cat, _shape_param_0);  cat = _shape_param_0 = None
        permute: "bf16[384, 512]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        addmm: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg3_1, view, permute);  arg3_1 = view = permute = None
        view_1: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm, _shape_param_1);  addmm = _shape_param_1 = None
        slice_3: "i64[1, 128]" = torch.ops.aten.slice.Tensor(arg4_1, 1, 0, 128);  arg4_1 = None
        embedding_1: "bf16[1, 128, 512]" = torch.ops.aten.embedding.default(arg5_1, slice_3);  arg5_1 = slice_3 = None
        add: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_1, embedding_1);  view_1 = embedding_1 = None
        full: "i64[256, 128]" = torch.ops.aten.full.default(_shape_param_2, 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_2 = None
        embedding_2: "bf16[256, 128, 512]" = torch.ops.aten.embedding.default(arg6_1, full);  arg6_1 = full = None
        add_1: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(add, embedding_2);  add = embedding_2 = None
        mul: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_1, arg7_1);  add_1 = arg7_1 = None
        add_2: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul, arg8_1);  mul = arg8_1 = None
        view_2: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_3);  _shape_param_3 = None
        permute_1: "bf16[512, 128]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        addmm_1: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg10_1, view_2, permute_1);  arg10_1 = view_2 = permute_1 = None
        view_3: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_1, _shape_param_4);  addmm_1 = _shape_param_4 = None
        mul_1: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_3, arg11_1);  view_3 = arg11_1 = None
        add_3: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_1, arg12_1);  mul_1 = arg12_1 = None
        view_4: "bf16[32768, 128]" = torch.ops.aten.view.default(add_3, _shape_param_5);  _shape_param_5 = None
        permute_2: "bf16[128, 128]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        addmm_2: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg14_1, view_4, permute_2);  arg14_1 = view_4 = permute_2 = None
        view_5: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_2, _shape_param_6);  addmm_2 = _shape_param_6 = None
        view_6: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_5, _shape_param_7);  view_5 = _shape_param_7 = None
        permute_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_6, [0, 2, 1, 3]);  view_6 = None
        mul_2: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_3, 0.4204482076268573);  permute_3 = None
        expand: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_2, _shape_param_8);  mul_2 = _shape_param_8 = None
        clone: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_7: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone, _shape_param_9);  clone = _shape_param_9 = None
        view_8: "bf16[32768, 128]" = torch.ops.aten.view.default(add_3, _shape_param_10);  add_3 = _shape_param_10 = None
        permute_4: "bf16[128, 128]" = torch.ops.aten.permute.default(arg15_1, [1, 0]);  arg15_1 = None
        addmm_3: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg16_1, view_8, permute_4);  arg16_1 = view_8 = permute_4 = None
        view_9: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_3, _shape_param_11);  addmm_3 = _shape_param_11 = None
        view_10: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_9, _shape_param_12);  view_9 = _shape_param_12 = None
        permute_5: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        permute_6: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_5, [0, 1, 3, 2]);  permute_5 = None
        mul_3: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_6, 0.4204482076268573);  permute_6 = None
        expand_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_3, _shape_param_13);  mul_3 = _shape_param_13 = None
        clone_1: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_11: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_1, _shape_param_14);  clone_1 = _shape_param_14 = None
        bmm: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_7, view_11);  view_7 = view_11 = None
        view_12: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm, _shape_param_15);  bmm = _shape_param_15 = None
        eq: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_12, -inf)
        logical_not: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_16, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_16 = None
        convert_element_type: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_12, torch.float32);  view_12 = None
        amax: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type, [-1], True)
        sub: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type, amax);  convert_element_type = amax = None
        exp: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None
        sum_1: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        convert_element_type_1: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div, torch.bfloat16);  div = None
        where: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_1, convert_element_type_1);  logical_not_1 = full_1 = convert_element_type_1 = None
        expand_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where, _shape_param_17);  where = _shape_param_17 = None
        view_13: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_2, _shape_param_18);  expand_2 = _shape_param_18 = None
        view_14: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_19);  _shape_param_19 = None
        permute_7: "bf16[512, 128]" = torch.ops.aten.permute.default(arg17_1, [1, 0]);  arg17_1 = None
        addmm_4: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg18_1, view_14, permute_7);  arg18_1 = view_14 = permute_7 = None
        view_15: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_4, _shape_param_20);  addmm_4 = _shape_param_20 = None
        view_16: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_15, _shape_param_21);  view_15 = _shape_param_21 = None
        permute_8: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_16, [0, 2, 1, 3]);  view_16 = None
        expand_3: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_8, _shape_param_22);  permute_8 = _shape_param_22 = None
        clone_2: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_17: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_2, _shape_param_23);  clone_2 = _shape_param_23 = None
        bmm_1: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_13, view_17);  view_13 = view_17 = None
        view_18: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_1, _shape_param_24);  bmm_1 = _shape_param_24 = None
        permute_9: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_3: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_9, memory_format = torch.contiguous_format);  permute_9 = None
        view_19: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_3, _shape_param_25);  clone_3 = _shape_param_25 = None
        view_20: "bf16[32768, 128]" = torch.ops.aten.view.default(view_19, _shape_param_26);  view_19 = _shape_param_26 = None
        permute_10: "bf16[128, 128]" = torch.ops.aten.permute.default(arg19_1, [1, 0]);  arg19_1 = None
        addmm_5: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg20_1, view_20, permute_10);  arg20_1 = view_20 = permute_10 = None
        view_21: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_5, _shape_param_27);  addmm_5 = _shape_param_27 = None
        view_22: "bf16[32768, 512]" = torch.ops.aten.view.default(add_2, _shape_param_28);  _shape_param_28 = None
        permute_11: "bf16[512, 128]" = torch.ops.aten.permute.default(arg21_1, [1, 0]);  arg21_1 = None
        addmm_6: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg22_1, view_22, permute_11);  arg22_1 = view_22 = permute_11 = None
        view_23: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_6, _shape_param_29);  addmm_6 = _shape_param_29 = None
        mul_4: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_23, arg23_1);  view_23 = arg23_1 = None
        add_4: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_4, arg24_1);  mul_4 = arg24_1 = None
        add_5: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_21, add_4);  view_21 = add_4 = None
        mul_5: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_5, arg25_1);  add_5 = arg25_1 = None
        add_6: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_5, arg26_1);  mul_5 = arg26_1 = None
        view_24: "bf16[32768, 128]" = torch.ops.aten.view.default(add_6, _shape_param_30);  _shape_param_30 = None
        permute_12: "bf16[128, 512]" = torch.ops.aten.permute.default(arg27_1, [1, 0]);  arg27_1 = None
        addmm_7: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg28_1, view_24, permute_12);  arg28_1 = view_24 = permute_12 = None
        view_25: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_7, _shape_param_31);  addmm_7 = _shape_param_31 = None
        relu: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_25);  view_25 = None
        view_26: "bf16[32768, 512]" = torch.ops.aten.view.default(relu, _shape_param_32);  relu = _shape_param_32 = None
        permute_13: "bf16[512, 128]" = torch.ops.aten.permute.default(arg29_1, [1, 0]);  arg29_1 = None
        addmm_8: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg30_1, view_26, permute_13);  arg30_1 = view_26 = permute_13 = None
        view_27: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_8, _shape_param_33);  addmm_8 = _shape_param_33 = None
        add_7: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_27, add_6);  view_27 = add_6 = None
        mul_6: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_7, arg31_1);  add_7 = arg31_1 = None
        add_8: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_6, arg32_1);  mul_6 = arg32_1 = None
        view_28: "bf16[32768, 128]" = torch.ops.aten.view.default(add_8, _shape_param_34);  _shape_param_34 = None
        permute_14: "bf16[128, 512]" = torch.ops.aten.permute.default(arg33_1, [1, 0]);  arg33_1 = None
        addmm_9: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg34_1, view_28, permute_14);  arg34_1 = view_28 = permute_14 = None
        view_29: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_9, _shape_param_35);  addmm_9 = _shape_param_35 = None
        relu_1: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_29);  view_29 = None
        view_30: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_1, _shape_param_36);  relu_1 = _shape_param_36 = None
        permute_15: "bf16[512, 128]" = torch.ops.aten.permute.default(arg35_1, [1, 0]);  arg35_1 = None
        addmm_10: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg36_1, view_30, permute_15);  arg36_1 = view_30 = permute_15 = None
        view_31: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_10, _shape_param_37);  addmm_10 = _shape_param_37 = None
        add_9: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_31, add_8);  view_31 = add_8 = None
        mul_7: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_9, arg37_1);  add_9 = arg37_1 = None
        add_10: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_7, arg38_1);  mul_7 = arg38_1 = None
        view_32: "bf16[32768, 128]" = torch.ops.aten.view.default(add_10, _shape_param_38);  _shape_param_38 = None
        permute_16: "bf16[128, 512]" = torch.ops.aten.permute.default(arg39_1, [1, 0]);  arg39_1 = None
        addmm_11: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg40_1, view_32, permute_16);  arg40_1 = view_32 = permute_16 = None
        view_33: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_11, _shape_param_39);  addmm_11 = _shape_param_39 = None
        relu_2: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_33);  view_33 = None
        view_34: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_2, _shape_param_40);  relu_2 = _shape_param_40 = None
        permute_17: "bf16[512, 128]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg42_1, view_34, permute_17);  arg42_1 = view_34 = permute_17 = None
        view_35: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_12, _shape_param_41);  addmm_12 = _shape_param_41 = None
        add_11: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_35, add_10);  view_35 = add_10 = None
        mul_8: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_11, arg43_1);  add_11 = arg43_1 = None
        add_12: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_8, arg44_1);  mul_8 = arg44_1 = None
        view_36: "bf16[32768, 128]" = torch.ops.aten.view.default(add_12, _shape_param_42);  _shape_param_42 = None
        permute_18: "bf16[128, 512]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_13: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg46_1, view_36, permute_18);  arg46_1 = view_36 = permute_18 = None
        view_37: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_13, _shape_param_43);  addmm_13 = _shape_param_43 = None
        relu_3: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_37);  view_37 = None
        view_38: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_3, _shape_param_44);  relu_3 = _shape_param_44 = None
        permute_19: "bf16[512, 128]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_14: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg48_1, view_38, permute_19);  arg48_1 = view_38 = permute_19 = None
        view_39: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_14, _shape_param_45);  addmm_14 = _shape_param_45 = None
        add_13: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_39, add_12);  view_39 = add_12 = None
        mul_9: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_13, arg49_1);  add_13 = arg49_1 = None
        add_14: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_9, arg50_1);  mul_9 = arg50_1 = None
        view_40: "bf16[32768, 128]" = torch.ops.aten.view.default(add_14, _shape_param_46);  add_14 = _shape_param_46 = None
        permute_20: "bf16[128, 512]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_15: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg52_1, view_40, permute_20);  arg52_1 = view_40 = permute_20 = None
        view_41: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_15, _shape_param_47);  addmm_15 = _shape_param_47 = None
        add_15: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_41, add_2);  view_41 = add_2 = None
        mul_10: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_15, arg53_1);  add_15 = arg53_1 = None
        add_16: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_10, arg54_1);  mul_10 = arg54_1 = None
        view_42: "bf16[32768, 512]" = torch.ops.aten.view.default(add_16, _shape_param_48);  _shape_param_48 = None
        permute_21: "bf16[512, 128]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_16: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg56_1, view_42, permute_21);  arg56_1 = view_42 = permute_21 = None
        view_43: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_16, _shape_param_49);  addmm_16 = _shape_param_49 = None
        mul_11: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_43, arg57_1);  view_43 = arg57_1 = None
        add_17: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_11, arg58_1);  mul_11 = arg58_1 = None
        view_44: "bf16[32768, 128]" = torch.ops.aten.view.default(add_17, _shape_param_50);  _shape_param_50 = None
        permute_22: "bf16[128, 128]" = torch.ops.aten.permute.default(arg59_1, [1, 0]);  arg59_1 = None
        addmm_17: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg60_1, view_44, permute_22);  arg60_1 = view_44 = permute_22 = None
        view_45: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_17, _shape_param_51);  addmm_17 = _shape_param_51 = None
        view_46: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_45, _shape_param_52);  view_45 = _shape_param_52 = None
        permute_23: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None
        mul_12: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_23, 0.4204482076268573);  permute_23 = None
        expand_4: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_12, _shape_param_53);  mul_12 = _shape_param_53 = None
        clone_4: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_47: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_4, _shape_param_54);  clone_4 = _shape_param_54 = None
        view_48: "bf16[32768, 128]" = torch.ops.aten.view.default(add_17, _shape_param_55);  add_17 = _shape_param_55 = None
        permute_24: "bf16[128, 128]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_18: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg62_1, view_48, permute_24);  arg62_1 = view_48 = permute_24 = None
        view_49: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_18, _shape_param_56);  addmm_18 = _shape_param_56 = None
        view_50: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_49, _shape_param_57);  view_49 = _shape_param_57 = None
        permute_25: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_50, [0, 2, 1, 3]);  view_50 = None
        permute_26: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_25, [0, 1, 3, 2]);  permute_25 = None
        mul_13: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_26, 0.4204482076268573);  permute_26 = None
        expand_5: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_13, _shape_param_58);  mul_13 = _shape_param_58 = None
        clone_5: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_51: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_5, _shape_param_59);  clone_5 = _shape_param_59 = None
        bmm_2: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_47, view_51);  view_47 = view_51 = None
        view_52: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_2, _shape_param_60);  bmm_2 = _shape_param_60 = None
        eq_1: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_52, -inf)
        logical_not_2: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_61, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_61 = None
        convert_element_type_2: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_52, torch.float32);  view_52 = None
        amax_1: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_2, [-1], True)
        sub_1: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_2, amax_1);  convert_element_type_2 = amax_1 = None
        exp_1: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_2: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        convert_element_type_3: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16);  div_1 = None
        where_1: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_2, convert_element_type_3);  logical_not_3 = full_2 = convert_element_type_3 = None
        expand_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_1, _shape_param_62);  where_1 = _shape_param_62 = None
        view_53: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_6, _shape_param_63);  expand_6 = _shape_param_63 = None
        view_54: "bf16[32768, 512]" = torch.ops.aten.view.default(add_16, _shape_param_64);  _shape_param_64 = None
        permute_27: "bf16[512, 128]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_19: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg64_1, view_54, permute_27);  arg64_1 = view_54 = permute_27 = None
        view_55: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_19, _shape_param_65);  addmm_19 = _shape_param_65 = None
        view_56: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_55, _shape_param_66);  view_55 = _shape_param_66 = None
        permute_28: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_56, [0, 2, 1, 3]);  view_56 = None
        expand_7: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_28, _shape_param_67);  permute_28 = _shape_param_67 = None
        clone_6: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_7, memory_format = torch.contiguous_format);  expand_7 = None
        view_57: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_6, _shape_param_68);  clone_6 = _shape_param_68 = None
        bmm_3: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_53, view_57);  view_53 = view_57 = None
        view_58: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_3, _shape_param_69);  bmm_3 = _shape_param_69 = None
        permute_29: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None
        clone_7: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_29, memory_format = torch.contiguous_format);  permute_29 = None
        view_59: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_7, _shape_param_70);  clone_7 = _shape_param_70 = None
        view_60: "bf16[32768, 128]" = torch.ops.aten.view.default(view_59, _shape_param_71);  view_59 = _shape_param_71 = None
        permute_30: "bf16[128, 128]" = torch.ops.aten.permute.default(arg65_1, [1, 0]);  arg65_1 = None
        addmm_20: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg66_1, view_60, permute_30);  arg66_1 = view_60 = permute_30 = None
        view_61: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_20, _shape_param_72);  addmm_20 = _shape_param_72 = None
        view_62: "bf16[32768, 512]" = torch.ops.aten.view.default(add_16, _shape_param_73);  _shape_param_73 = None
        permute_31: "bf16[512, 128]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_21: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg68_1, view_62, permute_31);  arg68_1 = view_62 = permute_31 = None
        view_63: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_21, _shape_param_74);  addmm_21 = _shape_param_74 = None
        mul_14: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_63, arg69_1);  view_63 = arg69_1 = None
        add_18: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_14, arg70_1);  mul_14 = arg70_1 = None
        add_19: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_61, add_18);  view_61 = add_18 = None
        mul_15: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_19, arg71_1);  add_19 = arg71_1 = None
        add_20: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_15, arg72_1);  mul_15 = arg72_1 = None
        view_64: "bf16[32768, 128]" = torch.ops.aten.view.default(add_20, _shape_param_75);  _shape_param_75 = None
        permute_32: "bf16[128, 512]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_22: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg74_1, view_64, permute_32);  arg74_1 = view_64 = permute_32 = None
        view_65: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_22, _shape_param_76);  addmm_22 = _shape_param_76 = None
        relu_4: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_65);  view_65 = None
        view_66: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_4, _shape_param_77);  relu_4 = _shape_param_77 = None
        permute_33: "bf16[512, 128]" = torch.ops.aten.permute.default(arg75_1, [1, 0]);  arg75_1 = None
        addmm_23: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg76_1, view_66, permute_33);  arg76_1 = view_66 = permute_33 = None
        view_67: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_23, _shape_param_78);  addmm_23 = _shape_param_78 = None
        add_21: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_67, add_20);  view_67 = add_20 = None
        mul_16: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_21, arg77_1);  add_21 = arg77_1 = None
        add_22: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_16, arg78_1);  mul_16 = arg78_1 = None
        view_68: "bf16[32768, 128]" = torch.ops.aten.view.default(add_22, _shape_param_79);  _shape_param_79 = None
        permute_34: "bf16[128, 512]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_24: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg80_1, view_68, permute_34);  arg80_1 = view_68 = permute_34 = None
        view_69: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_24, _shape_param_80);  addmm_24 = _shape_param_80 = None
        relu_5: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_69);  view_69 = None
        view_70: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_5, _shape_param_81);  relu_5 = _shape_param_81 = None
        permute_35: "bf16[512, 128]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_25: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg82_1, view_70, permute_35);  arg82_1 = view_70 = permute_35 = None
        view_71: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_25, _shape_param_82);  addmm_25 = _shape_param_82 = None
        add_23: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_71, add_22);  view_71 = add_22 = None
        mul_17: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_23, arg83_1);  add_23 = arg83_1 = None
        add_24: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_17, arg84_1);  mul_17 = arg84_1 = None
        view_72: "bf16[32768, 128]" = torch.ops.aten.view.default(add_24, _shape_param_83);  _shape_param_83 = None
        permute_36: "bf16[128, 512]" = torch.ops.aten.permute.default(arg85_1, [1, 0]);  arg85_1 = None
        addmm_26: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg86_1, view_72, permute_36);  arg86_1 = view_72 = permute_36 = None
        view_73: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_26, _shape_param_84);  addmm_26 = _shape_param_84 = None
        relu_6: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_73);  view_73 = None
        view_74: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_6, _shape_param_85);  relu_6 = _shape_param_85 = None
        permute_37: "bf16[512, 128]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_27: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg88_1, view_74, permute_37);  arg88_1 = view_74 = permute_37 = None
        view_75: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_27, _shape_param_86);  addmm_27 = _shape_param_86 = None
        add_25: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_75, add_24);  view_75 = add_24 = None
        mul_18: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_25, arg89_1);  add_25 = arg89_1 = None
        add_26: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_18, arg90_1);  mul_18 = arg90_1 = None
        view_76: "bf16[32768, 128]" = torch.ops.aten.view.default(add_26, _shape_param_87);  _shape_param_87 = None
        permute_38: "bf16[128, 512]" = torch.ops.aten.permute.default(arg91_1, [1, 0]);  arg91_1 = None
        addmm_28: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg92_1, view_76, permute_38);  arg92_1 = view_76 = permute_38 = None
        view_77: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_28, _shape_param_88);  addmm_28 = _shape_param_88 = None
        relu_7: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_77);  view_77 = None
        view_78: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_7, _shape_param_89);  relu_7 = _shape_param_89 = None
        permute_39: "bf16[512, 128]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_29: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg94_1, view_78, permute_39);  arg94_1 = view_78 = permute_39 = None
        view_79: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_29, _shape_param_90);  addmm_29 = _shape_param_90 = None
        add_27: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_79, add_26);  view_79 = add_26 = None
        mul_19: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_27, arg95_1);  add_27 = arg95_1 = None
        add_28: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_19, arg96_1);  mul_19 = arg96_1 = None
        view_80: "bf16[32768, 128]" = torch.ops.aten.view.default(add_28, _shape_param_91);  add_28 = _shape_param_91 = None
        permute_40: "bf16[128, 512]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_30: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg98_1, view_80, permute_40);  arg98_1 = view_80 = permute_40 = None
        view_81: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_30, _shape_param_92);  addmm_30 = _shape_param_92 = None
        add_29: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_81, add_16);  view_81 = add_16 = None
        mul_20: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_29, arg99_1);  add_29 = arg99_1 = None
        add_30: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_20, arg100_1);  mul_20 = arg100_1 = None
        view_82: "bf16[32768, 512]" = torch.ops.aten.view.default(add_30, _shape_param_93);  _shape_param_93 = None
        permute_41: "bf16[512, 128]" = torch.ops.aten.permute.default(arg101_1, [1, 0]);  arg101_1 = None
        addmm_31: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg102_1, view_82, permute_41);  arg102_1 = view_82 = permute_41 = None
        view_83: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_31, _shape_param_94);  addmm_31 = _shape_param_94 = None
        mul_21: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_83, arg103_1);  view_83 = arg103_1 = None
        add_31: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_21, arg104_1);  mul_21 = arg104_1 = None
        view_84: "bf16[32768, 128]" = torch.ops.aten.view.default(add_31, _shape_param_95);  _shape_param_95 = None
        permute_42: "bf16[128, 128]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_32: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg106_1, view_84, permute_42);  arg106_1 = view_84 = permute_42 = None
        view_85: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_32, _shape_param_96);  addmm_32 = _shape_param_96 = None
        view_86: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_85, _shape_param_97);  view_85 = _shape_param_97 = None
        permute_43: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None
        mul_22: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_43, 0.4204482076268573);  permute_43 = None
        expand_8: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_22, _shape_param_98);  mul_22 = _shape_param_98 = None
        clone_8: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_87: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_8, _shape_param_99);  clone_8 = _shape_param_99 = None
        view_88: "bf16[32768, 128]" = torch.ops.aten.view.default(add_31, _shape_param_100);  add_31 = _shape_param_100 = None
        permute_44: "bf16[128, 128]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_33: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg108_1, view_88, permute_44);  arg108_1 = view_88 = permute_44 = None
        view_89: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_33, _shape_param_101);  addmm_33 = _shape_param_101 = None
        view_90: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_89, _shape_param_102);  view_89 = _shape_param_102 = None
        permute_45: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_90, [0, 2, 1, 3]);  view_90 = None
        permute_46: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_45, [0, 1, 3, 2]);  permute_45 = None
        mul_23: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_46, 0.4204482076268573);  permute_46 = None
        expand_9: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_23, _shape_param_103);  mul_23 = _shape_param_103 = None
        clone_9: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_9, memory_format = torch.contiguous_format);  expand_9 = None
        view_91: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_9, _shape_param_104);  clone_9 = _shape_param_104 = None
        bmm_4: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_87, view_91);  view_87 = view_91 = None
        view_92: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_4, _shape_param_105);  bmm_4 = _shape_param_105 = None
        eq_2: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_92, -inf)
        logical_not_4: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_106, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_106 = None
        convert_element_type_4: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_92, torch.float32);  view_92 = None
        amax_2: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_4, [-1], True)
        sub_2: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_4, amax_2);  convert_element_type_4 = amax_2 = None
        exp_2: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_2);  sub_2 = None
        sum_3: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        convert_element_type_5: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_2, torch.bfloat16);  div_2 = None
        where_2: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_3, convert_element_type_5);  logical_not_5 = full_3 = convert_element_type_5 = None
        expand_10: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_2, _shape_param_107);  where_2 = _shape_param_107 = None
        view_93: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_10, _shape_param_108);  expand_10 = _shape_param_108 = None
        view_94: "bf16[32768, 512]" = torch.ops.aten.view.default(add_30, _shape_param_109);  _shape_param_109 = None
        permute_47: "bf16[512, 128]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_34: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg110_1, view_94, permute_47);  arg110_1 = view_94 = permute_47 = None
        view_95: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_34, _shape_param_110);  addmm_34 = _shape_param_110 = None
        view_96: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_95, _shape_param_111);  view_95 = _shape_param_111 = None
        permute_48: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_96, [0, 2, 1, 3]);  view_96 = None
        expand_11: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_48, _shape_param_112);  permute_48 = _shape_param_112 = None
        clone_10: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_11, memory_format = torch.contiguous_format);  expand_11 = None
        view_97: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_10, _shape_param_113);  clone_10 = _shape_param_113 = None
        bmm_5: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_93, view_97);  view_93 = view_97 = None
        view_98: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_5, _shape_param_114);  bmm_5 = _shape_param_114 = None
        permute_49: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None
        clone_11: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_49, memory_format = torch.contiguous_format);  permute_49 = None
        view_99: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_11, _shape_param_115);  clone_11 = _shape_param_115 = None
        view_100: "bf16[32768, 128]" = torch.ops.aten.view.default(view_99, _shape_param_116);  view_99 = _shape_param_116 = None
        permute_50: "bf16[128, 128]" = torch.ops.aten.permute.default(arg111_1, [1, 0]);  arg111_1 = None
        addmm_35: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg112_1, view_100, permute_50);  arg112_1 = view_100 = permute_50 = None
        view_101: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_35, _shape_param_117);  addmm_35 = _shape_param_117 = None
        view_102: "bf16[32768, 512]" = torch.ops.aten.view.default(add_30, _shape_param_118);  _shape_param_118 = None
        permute_51: "bf16[512, 128]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_36: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg114_1, view_102, permute_51);  arg114_1 = view_102 = permute_51 = None
        view_103: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_36, _shape_param_119);  addmm_36 = _shape_param_119 = None
        mul_24: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_103, arg115_1);  view_103 = arg115_1 = None
        add_32: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_24, arg116_1);  mul_24 = arg116_1 = None
        add_33: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_101, add_32);  view_101 = add_32 = None
        mul_25: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_33, arg117_1);  add_33 = arg117_1 = None
        add_34: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_25, arg118_1);  mul_25 = arg118_1 = None
        view_104: "bf16[32768, 128]" = torch.ops.aten.view.default(add_34, _shape_param_120);  _shape_param_120 = None
        permute_52: "bf16[128, 512]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_37: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg120_1, view_104, permute_52);  arg120_1 = view_104 = permute_52 = None
        view_105: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_37, _shape_param_121);  addmm_37 = _shape_param_121 = None
        relu_8: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_105);  view_105 = None
        view_106: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_8, _shape_param_122);  relu_8 = _shape_param_122 = None
        permute_53: "bf16[512, 128]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_38: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg122_1, view_106, permute_53);  arg122_1 = view_106 = permute_53 = None
        view_107: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_38, _shape_param_123);  addmm_38 = _shape_param_123 = None
        add_35: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_107, add_34);  view_107 = add_34 = None
        mul_26: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_35, arg123_1);  add_35 = arg123_1 = None
        add_36: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_26, arg124_1);  mul_26 = arg124_1 = None
        view_108: "bf16[32768, 128]" = torch.ops.aten.view.default(add_36, _shape_param_124);  _shape_param_124 = None
        permute_54: "bf16[128, 512]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_39: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg126_1, view_108, permute_54);  arg126_1 = view_108 = permute_54 = None
        view_109: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_39, _shape_param_125);  addmm_39 = _shape_param_125 = None
        relu_9: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_109);  view_109 = None
        view_110: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_9, _shape_param_126);  relu_9 = _shape_param_126 = None
        permute_55: "bf16[512, 128]" = torch.ops.aten.permute.default(arg127_1, [1, 0]);  arg127_1 = None
        addmm_40: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg128_1, view_110, permute_55);  arg128_1 = view_110 = permute_55 = None
        view_111: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_40, _shape_param_127);  addmm_40 = _shape_param_127 = None
        add_37: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_111, add_36);  view_111 = add_36 = None
        mul_27: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_37, arg129_1);  add_37 = arg129_1 = None
        add_38: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_27, arg130_1);  mul_27 = arg130_1 = None
        view_112: "bf16[32768, 128]" = torch.ops.aten.view.default(add_38, _shape_param_128);  _shape_param_128 = None
        permute_56: "bf16[128, 512]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_41: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg132_1, view_112, permute_56);  arg132_1 = view_112 = permute_56 = None
        view_113: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_41, _shape_param_129);  addmm_41 = _shape_param_129 = None
        relu_10: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_113);  view_113 = None
        view_114: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_10, _shape_param_130);  relu_10 = _shape_param_130 = None
        permute_57: "bf16[512, 128]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_42: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg134_1, view_114, permute_57);  arg134_1 = view_114 = permute_57 = None
        view_115: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_42, _shape_param_131);  addmm_42 = _shape_param_131 = None
        add_39: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_115, add_38);  view_115 = add_38 = None
        mul_28: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_39, arg135_1);  add_39 = arg135_1 = None
        add_40: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_28, arg136_1);  mul_28 = arg136_1 = None
        view_116: "bf16[32768, 128]" = torch.ops.aten.view.default(add_40, _shape_param_132);  _shape_param_132 = None
        permute_58: "bf16[128, 512]" = torch.ops.aten.permute.default(arg137_1, [1, 0]);  arg137_1 = None
        addmm_43: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg138_1, view_116, permute_58);  arg138_1 = view_116 = permute_58 = None
        view_117: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_43, _shape_param_133);  addmm_43 = _shape_param_133 = None
        relu_11: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_117);  view_117 = None
        view_118: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_11, _shape_param_134);  relu_11 = _shape_param_134 = None
        permute_59: "bf16[512, 128]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_44: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg140_1, view_118, permute_59);  arg140_1 = view_118 = permute_59 = None
        view_119: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_44, _shape_param_135);  addmm_44 = _shape_param_135 = None
        add_41: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_119, add_40);  view_119 = add_40 = None
        mul_29: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_41, arg141_1);  add_41 = arg141_1 = None
        add_42: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_29, arg142_1);  mul_29 = arg142_1 = None
        view_120: "bf16[32768, 128]" = torch.ops.aten.view.default(add_42, _shape_param_136);  add_42 = _shape_param_136 = None
        permute_60: "bf16[128, 512]" = torch.ops.aten.permute.default(arg143_1, [1, 0]);  arg143_1 = None
        addmm_45: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg144_1, view_120, permute_60);  arg144_1 = view_120 = permute_60 = None
        view_121: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_45, _shape_param_137);  addmm_45 = _shape_param_137 = None
        add_43: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_121, add_30);  view_121 = add_30 = None
        mul_30: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_43, arg145_1);  add_43 = arg145_1 = None
        add_44: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_30, arg146_1);  mul_30 = arg146_1 = None
        view_122: "bf16[32768, 512]" = torch.ops.aten.view.default(add_44, _shape_param_138);  _shape_param_138 = None
        permute_61: "bf16[512, 128]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_46: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg148_1, view_122, permute_61);  arg148_1 = view_122 = permute_61 = None
        view_123: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_46, _shape_param_139);  addmm_46 = _shape_param_139 = None
        mul_31: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_123, arg149_1);  view_123 = arg149_1 = None
        add_45: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_31, arg150_1);  mul_31 = arg150_1 = None
        view_124: "bf16[32768, 128]" = torch.ops.aten.view.default(add_45, _shape_param_140);  _shape_param_140 = None
        permute_62: "bf16[128, 128]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_47: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg152_1, view_124, permute_62);  arg152_1 = view_124 = permute_62 = None
        view_125: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_47, _shape_param_141);  addmm_47 = _shape_param_141 = None
        view_126: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_125, _shape_param_142);  view_125 = _shape_param_142 = None
        permute_63: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None
        mul_32: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_63, 0.4204482076268573);  permute_63 = None
        expand_12: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_32, _shape_param_143);  mul_32 = _shape_param_143 = None
        clone_12: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_127: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_12, _shape_param_144);  clone_12 = _shape_param_144 = None
        view_128: "bf16[32768, 128]" = torch.ops.aten.view.default(add_45, _shape_param_145);  add_45 = _shape_param_145 = None
        permute_64: "bf16[128, 128]" = torch.ops.aten.permute.default(arg153_1, [1, 0]);  arg153_1 = None
        addmm_48: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg154_1, view_128, permute_64);  arg154_1 = view_128 = permute_64 = None
        view_129: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_48, _shape_param_146);  addmm_48 = _shape_param_146 = None
        view_130: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_129, _shape_param_147);  view_129 = _shape_param_147 = None
        permute_65: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_130, [0, 2, 1, 3]);  view_130 = None
        permute_66: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_65, [0, 1, 3, 2]);  permute_65 = None
        mul_33: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_66, 0.4204482076268573);  permute_66 = None
        expand_13: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_33, _shape_param_148);  mul_33 = _shape_param_148 = None
        clone_13: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_131: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_13, _shape_param_149);  clone_13 = _shape_param_149 = None
        bmm_6: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_127, view_131);  view_127 = view_131 = None
        view_132: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_6, _shape_param_150);  bmm_6 = _shape_param_150 = None
        eq_3: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_132, -inf)
        logical_not_6: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_4: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_151, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_151 = None
        convert_element_type_6: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_132, torch.float32);  view_132 = None
        amax_3: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_6, [-1], True)
        sub_3: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_6, amax_3);  convert_element_type_6 = amax_3 = None
        exp_3: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_3);  sub_3 = None
        sum_4: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        convert_element_type_7: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_3, torch.bfloat16);  div_3 = None
        where_3: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_7, full_4, convert_element_type_7);  logical_not_7 = full_4 = convert_element_type_7 = None
        expand_14: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_3, _shape_param_152);  where_3 = _shape_param_152 = None
        view_133: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_14, _shape_param_153);  expand_14 = _shape_param_153 = None
        view_134: "bf16[32768, 512]" = torch.ops.aten.view.default(add_44, _shape_param_154);  _shape_param_154 = None
        permute_67: "bf16[512, 128]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_49: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg156_1, view_134, permute_67);  arg156_1 = view_134 = permute_67 = None
        view_135: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_49, _shape_param_155);  addmm_49 = _shape_param_155 = None
        view_136: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_135, _shape_param_156);  view_135 = _shape_param_156 = None
        permute_68: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_136, [0, 2, 1, 3]);  view_136 = None
        expand_15: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_68, _shape_param_157);  permute_68 = _shape_param_157 = None
        clone_14: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_137: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_14, _shape_param_158);  clone_14 = _shape_param_158 = None
        bmm_7: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_133, view_137);  view_133 = view_137 = None
        view_138: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_7, _shape_param_159);  bmm_7 = _shape_param_159 = None
        permute_69: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_15: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_69, memory_format = torch.contiguous_format);  permute_69 = None
        view_139: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_15, _shape_param_160);  clone_15 = _shape_param_160 = None
        view_140: "bf16[32768, 128]" = torch.ops.aten.view.default(view_139, _shape_param_161);  view_139 = _shape_param_161 = None
        permute_70: "bf16[128, 128]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_50: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg158_1, view_140, permute_70);  arg158_1 = view_140 = permute_70 = None
        view_141: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_50, _shape_param_162);  addmm_50 = _shape_param_162 = None
        view_142: "bf16[32768, 512]" = torch.ops.aten.view.default(add_44, _shape_param_163);  _shape_param_163 = None
        permute_71: "bf16[512, 128]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_51: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg160_1, view_142, permute_71);  arg160_1 = view_142 = permute_71 = None
        view_143: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_51, _shape_param_164);  addmm_51 = _shape_param_164 = None
        mul_34: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_143, arg161_1);  view_143 = arg161_1 = None
        add_46: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_34, arg162_1);  mul_34 = arg162_1 = None
        add_47: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_141, add_46);  view_141 = add_46 = None
        mul_35: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_47, arg163_1);  add_47 = arg163_1 = None
        add_48: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_35, arg164_1);  mul_35 = arg164_1 = None
        view_144: "bf16[32768, 128]" = torch.ops.aten.view.default(add_48, _shape_param_165);  _shape_param_165 = None
        permute_72: "bf16[128, 512]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_52: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg166_1, view_144, permute_72);  arg166_1 = view_144 = permute_72 = None
        view_145: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_52, _shape_param_166);  addmm_52 = _shape_param_166 = None
        relu_12: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_145);  view_145 = None
        view_146: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_12, _shape_param_167);  relu_12 = _shape_param_167 = None
        permute_73: "bf16[512, 128]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_53: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg168_1, view_146, permute_73);  arg168_1 = view_146 = permute_73 = None
        view_147: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_53, _shape_param_168);  addmm_53 = _shape_param_168 = None
        add_49: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_147, add_48);  view_147 = add_48 = None
        mul_36: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_49, arg169_1);  add_49 = arg169_1 = None
        add_50: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_36, arg170_1);  mul_36 = arg170_1 = None
        view_148: "bf16[32768, 128]" = torch.ops.aten.view.default(add_50, _shape_param_169);  _shape_param_169 = None
        permute_74: "bf16[128, 512]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_54: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg172_1, view_148, permute_74);  arg172_1 = view_148 = permute_74 = None
        view_149: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_54, _shape_param_170);  addmm_54 = _shape_param_170 = None
        relu_13: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_149);  view_149 = None
        view_150: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_13, _shape_param_171);  relu_13 = _shape_param_171 = None
        permute_75: "bf16[512, 128]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_55: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg174_1, view_150, permute_75);  arg174_1 = view_150 = permute_75 = None
        view_151: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_55, _shape_param_172);  addmm_55 = _shape_param_172 = None
        add_51: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_151, add_50);  view_151 = add_50 = None
        mul_37: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_51, arg175_1);  add_51 = arg175_1 = None
        add_52: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_37, arg176_1);  mul_37 = arg176_1 = None
        view_152: "bf16[32768, 128]" = torch.ops.aten.view.default(add_52, _shape_param_173);  _shape_param_173 = None
        permute_76: "bf16[128, 512]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_56: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg178_1, view_152, permute_76);  arg178_1 = view_152 = permute_76 = None
        view_153: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_56, _shape_param_174);  addmm_56 = _shape_param_174 = None
        relu_14: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_153);  view_153 = None
        view_154: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_14, _shape_param_175);  relu_14 = _shape_param_175 = None
        permute_77: "bf16[512, 128]" = torch.ops.aten.permute.default(arg179_1, [1, 0]);  arg179_1 = None
        addmm_57: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg180_1, view_154, permute_77);  arg180_1 = view_154 = permute_77 = None
        view_155: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_57, _shape_param_176);  addmm_57 = _shape_param_176 = None
        add_53: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_155, add_52);  view_155 = add_52 = None
        mul_38: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_53, arg181_1);  add_53 = arg181_1 = None
        add_54: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_38, arg182_1);  mul_38 = arg182_1 = None
        view_156: "bf16[32768, 128]" = torch.ops.aten.view.default(add_54, _shape_param_177);  _shape_param_177 = None
        permute_78: "bf16[128, 512]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_58: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg184_1, view_156, permute_78);  arg184_1 = view_156 = permute_78 = None
        view_157: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_58, _shape_param_178);  addmm_58 = _shape_param_178 = None
        relu_15: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_157);  view_157 = None
        view_158: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_15, _shape_param_179);  relu_15 = _shape_param_179 = None
        permute_79: "bf16[512, 128]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_59: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg186_1, view_158, permute_79);  arg186_1 = view_158 = permute_79 = None
        view_159: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_59, _shape_param_180);  addmm_59 = _shape_param_180 = None
        add_55: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_159, add_54);  view_159 = add_54 = None
        mul_39: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_55, arg187_1);  add_55 = arg187_1 = None
        add_56: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_39, arg188_1);  mul_39 = arg188_1 = None
        view_160: "bf16[32768, 128]" = torch.ops.aten.view.default(add_56, _shape_param_181);  add_56 = _shape_param_181 = None
        permute_80: "bf16[128, 512]" = torch.ops.aten.permute.default(arg189_1, [1, 0]);  arg189_1 = None
        addmm_60: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg190_1, view_160, permute_80);  arg190_1 = view_160 = permute_80 = None
        view_161: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_60, _shape_param_182);  addmm_60 = _shape_param_182 = None
        add_57: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_161, add_44);  view_161 = add_44 = None
        mul_40: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_57, arg191_1);  add_57 = arg191_1 = None
        add_58: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_40, arg192_1);  mul_40 = arg192_1 = None
        view_162: "bf16[32768, 512]" = torch.ops.aten.view.default(add_58, _shape_param_183);  _shape_param_183 = None
        permute_81: "bf16[512, 128]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_61: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg194_1, view_162, permute_81);  arg194_1 = view_162 = permute_81 = None
        view_163: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_61, _shape_param_184);  addmm_61 = _shape_param_184 = None
        mul_41: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_163, arg195_1);  view_163 = arg195_1 = None
        add_59: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_41, arg196_1);  mul_41 = arg196_1 = None
        view_164: "bf16[32768, 128]" = torch.ops.aten.view.default(add_59, _shape_param_185);  _shape_param_185 = None
        permute_82: "bf16[128, 128]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_62: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg198_1, view_164, permute_82);  arg198_1 = view_164 = permute_82 = None
        view_165: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_62, _shape_param_186);  addmm_62 = _shape_param_186 = None
        view_166: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_165, _shape_param_187);  view_165 = _shape_param_187 = None
        permute_83: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None
        mul_42: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_83, 0.4204482076268573);  permute_83 = None
        expand_16: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_42, _shape_param_188);  mul_42 = _shape_param_188 = None
        clone_16: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_16, memory_format = torch.contiguous_format);  expand_16 = None
        view_167: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_16, _shape_param_189);  clone_16 = _shape_param_189 = None
        view_168: "bf16[32768, 128]" = torch.ops.aten.view.default(add_59, _shape_param_190);  add_59 = _shape_param_190 = None
        permute_84: "bf16[128, 128]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_63: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg200_1, view_168, permute_84);  arg200_1 = view_168 = permute_84 = None
        view_169: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_63, _shape_param_191);  addmm_63 = _shape_param_191 = None
        view_170: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_169, _shape_param_192);  view_169 = _shape_param_192 = None
        permute_85: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_170, [0, 2, 1, 3]);  view_170 = None
        permute_86: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_85, [0, 1, 3, 2]);  permute_85 = None
        mul_43: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_86, 0.4204482076268573);  permute_86 = None
        expand_17: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_43, _shape_param_193);  mul_43 = _shape_param_193 = None
        clone_17: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_171: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_17, _shape_param_194);  clone_17 = _shape_param_194 = None
        bmm_8: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_167, view_171);  view_167 = view_171 = None
        view_172: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_8, _shape_param_195);  bmm_8 = _shape_param_195 = None
        eq_4: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_172, -inf)
        logical_not_8: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_5: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_196, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_196 = None
        convert_element_type_8: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_172, torch.float32);  view_172 = None
        amax_4: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_8, [-1], True)
        sub_4: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_8, amax_4);  convert_element_type_8 = amax_4 = None
        exp_4: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_5: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        convert_element_type_9: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_4, torch.bfloat16);  div_4 = None
        where_4: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_9, full_5, convert_element_type_9);  logical_not_9 = full_5 = convert_element_type_9 = None
        expand_18: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_4, _shape_param_197);  where_4 = _shape_param_197 = None
        view_173: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_18, _shape_param_198);  expand_18 = _shape_param_198 = None
        view_174: "bf16[32768, 512]" = torch.ops.aten.view.default(add_58, _shape_param_199);  _shape_param_199 = None
        permute_87: "bf16[512, 128]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_64: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg202_1, view_174, permute_87);  arg202_1 = view_174 = permute_87 = None
        view_175: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_64, _shape_param_200);  addmm_64 = _shape_param_200 = None
        view_176: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_175, _shape_param_201);  view_175 = _shape_param_201 = None
        permute_88: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_176, [0, 2, 1, 3]);  view_176 = None
        expand_19: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_88, _shape_param_202);  permute_88 = _shape_param_202 = None
        clone_18: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_19, memory_format = torch.contiguous_format);  expand_19 = None
        view_177: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_18, _shape_param_203);  clone_18 = _shape_param_203 = None
        bmm_9: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_173, view_177);  view_173 = view_177 = None
        view_178: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_9, _shape_param_204);  bmm_9 = _shape_param_204 = None
        permute_89: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_178, [0, 2, 1, 3]);  view_178 = None
        clone_19: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_89, memory_format = torch.contiguous_format);  permute_89 = None
        view_179: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_19, _shape_param_205);  clone_19 = _shape_param_205 = None
        view_180: "bf16[32768, 128]" = torch.ops.aten.view.default(view_179, _shape_param_206);  view_179 = _shape_param_206 = None
        permute_90: "bf16[128, 128]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_65: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg204_1, view_180, permute_90);  arg204_1 = view_180 = permute_90 = None
        view_181: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_65, _shape_param_207);  addmm_65 = _shape_param_207 = None
        view_182: "bf16[32768, 512]" = torch.ops.aten.view.default(add_58, _shape_param_208);  _shape_param_208 = None
        permute_91: "bf16[512, 128]" = torch.ops.aten.permute.default(arg205_1, [1, 0]);  arg205_1 = None
        addmm_66: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg206_1, view_182, permute_91);  arg206_1 = view_182 = permute_91 = None
        view_183: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_66, _shape_param_209);  addmm_66 = _shape_param_209 = None
        mul_44: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_183, arg207_1);  view_183 = arg207_1 = None
        add_60: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_44, arg208_1);  mul_44 = arg208_1 = None
        add_61: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_181, add_60);  view_181 = add_60 = None
        mul_45: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_61, arg209_1);  add_61 = arg209_1 = None
        add_62: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_45, arg210_1);  mul_45 = arg210_1 = None
        view_184: "bf16[32768, 128]" = torch.ops.aten.view.default(add_62, _shape_param_210);  _shape_param_210 = None
        permute_92: "bf16[128, 512]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_67: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg212_1, view_184, permute_92);  arg212_1 = view_184 = permute_92 = None
        view_185: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_67, _shape_param_211);  addmm_67 = _shape_param_211 = None
        relu_16: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_185);  view_185 = None
        view_186: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_16, _shape_param_212);  relu_16 = _shape_param_212 = None
        permute_93: "bf16[512, 128]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_68: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg214_1, view_186, permute_93);  arg214_1 = view_186 = permute_93 = None
        view_187: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_68, _shape_param_213);  addmm_68 = _shape_param_213 = None
        add_63: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_187, add_62);  view_187 = add_62 = None
        mul_46: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_63, arg215_1);  add_63 = arg215_1 = None
        add_64: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_46, arg216_1);  mul_46 = arg216_1 = None
        view_188: "bf16[32768, 128]" = torch.ops.aten.view.default(add_64, _shape_param_214);  _shape_param_214 = None
        permute_94: "bf16[128, 512]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_69: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg218_1, view_188, permute_94);  arg218_1 = view_188 = permute_94 = None
        view_189: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_69, _shape_param_215);  addmm_69 = _shape_param_215 = None
        relu_17: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_189);  view_189 = None
        view_190: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_17, _shape_param_216);  relu_17 = _shape_param_216 = None
        permute_95: "bf16[512, 128]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_70: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg220_1, view_190, permute_95);  arg220_1 = view_190 = permute_95 = None
        view_191: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_70, _shape_param_217);  addmm_70 = _shape_param_217 = None
        add_65: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_191, add_64);  view_191 = add_64 = None
        mul_47: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_65, arg221_1);  add_65 = arg221_1 = None
        add_66: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_47, arg222_1);  mul_47 = arg222_1 = None
        view_192: "bf16[32768, 128]" = torch.ops.aten.view.default(add_66, _shape_param_218);  _shape_param_218 = None
        permute_96: "bf16[128, 512]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_71: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg224_1, view_192, permute_96);  arg224_1 = view_192 = permute_96 = None
        view_193: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_71, _shape_param_219);  addmm_71 = _shape_param_219 = None
        relu_18: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_193);  view_193 = None
        view_194: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_18, _shape_param_220);  relu_18 = _shape_param_220 = None
        permute_97: "bf16[512, 128]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_72: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg226_1, view_194, permute_97);  arg226_1 = view_194 = permute_97 = None
        view_195: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_72, _shape_param_221);  addmm_72 = _shape_param_221 = None
        add_67: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_195, add_66);  view_195 = add_66 = None
        mul_48: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_67, arg227_1);  add_67 = arg227_1 = None
        add_68: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_48, arg228_1);  mul_48 = arg228_1 = None
        view_196: "bf16[32768, 128]" = torch.ops.aten.view.default(add_68, _shape_param_222);  _shape_param_222 = None
        permute_98: "bf16[128, 512]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_73: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg230_1, view_196, permute_98);  arg230_1 = view_196 = permute_98 = None
        view_197: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_73, _shape_param_223);  addmm_73 = _shape_param_223 = None
        relu_19: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_197);  view_197 = None
        view_198: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_19, _shape_param_224);  relu_19 = _shape_param_224 = None
        permute_99: "bf16[512, 128]" = torch.ops.aten.permute.default(arg231_1, [1, 0]);  arg231_1 = None
        addmm_74: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg232_1, view_198, permute_99);  arg232_1 = view_198 = permute_99 = None
        view_199: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_74, _shape_param_225);  addmm_74 = _shape_param_225 = None
        add_69: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_199, add_68);  view_199 = add_68 = None
        mul_49: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_69, arg233_1);  add_69 = arg233_1 = None
        add_70: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_49, arg234_1);  mul_49 = arg234_1 = None
        view_200: "bf16[32768, 128]" = torch.ops.aten.view.default(add_70, _shape_param_226);  add_70 = _shape_param_226 = None
        permute_100: "bf16[128, 512]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_75: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg236_1, view_200, permute_100);  arg236_1 = view_200 = permute_100 = None
        view_201: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_75, _shape_param_227);  addmm_75 = _shape_param_227 = None
        add_71: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_201, add_58);  view_201 = add_58 = None
        mul_50: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_71, arg237_1);  add_71 = arg237_1 = None
        add_72: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_50, arg238_1);  mul_50 = arg238_1 = None
        view_202: "bf16[32768, 512]" = torch.ops.aten.view.default(add_72, _shape_param_228);  _shape_param_228 = None
        permute_101: "bf16[512, 128]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_76: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg240_1, view_202, permute_101);  arg240_1 = view_202 = permute_101 = None
        view_203: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_76, _shape_param_229);  addmm_76 = _shape_param_229 = None
        mul_51: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_203, arg241_1);  view_203 = arg241_1 = None
        add_73: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_51, arg242_1);  mul_51 = arg242_1 = None
        view_204: "bf16[32768, 128]" = torch.ops.aten.view.default(add_73, _shape_param_230);  _shape_param_230 = None
        permute_102: "bf16[128, 128]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_77: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg244_1, view_204, permute_102);  arg244_1 = view_204 = permute_102 = None
        view_205: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_77, _shape_param_231);  addmm_77 = _shape_param_231 = None
        view_206: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_205, _shape_param_232);  view_205 = _shape_param_232 = None
        permute_103: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        mul_52: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_103, 0.4204482076268573);  permute_103 = None
        expand_20: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_52, _shape_param_233);  mul_52 = _shape_param_233 = None
        clone_20: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_207: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_20, _shape_param_234);  clone_20 = _shape_param_234 = None
        view_208: "bf16[32768, 128]" = torch.ops.aten.view.default(add_73, _shape_param_235);  add_73 = _shape_param_235 = None
        permute_104: "bf16[128, 128]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_78: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg246_1, view_208, permute_104);  arg246_1 = view_208 = permute_104 = None
        view_209: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_78, _shape_param_236);  addmm_78 = _shape_param_236 = None
        view_210: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_209, _shape_param_237);  view_209 = _shape_param_237 = None
        permute_105: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_210, [0, 2, 1, 3]);  view_210 = None
        permute_106: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_105, [0, 1, 3, 2]);  permute_105 = None
        mul_53: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_106, 0.4204482076268573);  permute_106 = None
        expand_21: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_53, _shape_param_238);  mul_53 = _shape_param_238 = None
        clone_21: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_21, memory_format = torch.contiguous_format);  expand_21 = None
        view_211: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_21, _shape_param_239);  clone_21 = _shape_param_239 = None
        bmm_10: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_207, view_211);  view_207 = view_211 = None
        view_212: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_10, _shape_param_240);  bmm_10 = _shape_param_240 = None
        eq_5: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_212, -inf)
        logical_not_10: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_241, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_241 = None
        convert_element_type_10: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_212, torch.float32);  view_212 = None
        amax_5: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_10, [-1], True)
        sub_5: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_10, amax_5);  convert_element_type_10 = amax_5 = None
        exp_5: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_5);  sub_5 = None
        sum_6: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        convert_element_type_11: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_5, torch.bfloat16);  div_5 = None
        where_5: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_11, full_6, convert_element_type_11);  logical_not_11 = full_6 = convert_element_type_11 = None
        expand_22: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_5, _shape_param_242);  where_5 = _shape_param_242 = None
        view_213: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_22, _shape_param_243);  expand_22 = _shape_param_243 = None
        view_214: "bf16[32768, 512]" = torch.ops.aten.view.default(add_72, _shape_param_244);  _shape_param_244 = None
        permute_107: "bf16[512, 128]" = torch.ops.aten.permute.default(arg247_1, [1, 0]);  arg247_1 = None
        addmm_79: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg248_1, view_214, permute_107);  arg248_1 = view_214 = permute_107 = None
        view_215: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_79, _shape_param_245);  addmm_79 = _shape_param_245 = None
        view_216: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_215, _shape_param_246);  view_215 = _shape_param_246 = None
        permute_108: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None
        expand_23: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_108, _shape_param_247);  permute_108 = _shape_param_247 = None
        clone_22: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_217: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_22, _shape_param_248);  clone_22 = _shape_param_248 = None
        bmm_11: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_213, view_217);  view_213 = view_217 = None
        view_218: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_11, _shape_param_249);  bmm_11 = _shape_param_249 = None
        permute_109: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_218, [0, 2, 1, 3]);  view_218 = None
        clone_23: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_109, memory_format = torch.contiguous_format);  permute_109 = None
        view_219: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_23, _shape_param_250);  clone_23 = _shape_param_250 = None
        view_220: "bf16[32768, 128]" = torch.ops.aten.view.default(view_219, _shape_param_251);  view_219 = _shape_param_251 = None
        permute_110: "bf16[128, 128]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_80: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg250_1, view_220, permute_110);  arg250_1 = view_220 = permute_110 = None
        view_221: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_80, _shape_param_252);  addmm_80 = _shape_param_252 = None
        view_222: "bf16[32768, 512]" = torch.ops.aten.view.default(add_72, _shape_param_253);  _shape_param_253 = None
        permute_111: "bf16[512, 128]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_81: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg252_1, view_222, permute_111);  arg252_1 = view_222 = permute_111 = None
        view_223: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_81, _shape_param_254);  addmm_81 = _shape_param_254 = None
        mul_54: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_223, arg253_1);  view_223 = arg253_1 = None
        add_74: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_54, arg254_1);  mul_54 = arg254_1 = None
        add_75: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_221, add_74);  view_221 = add_74 = None
        mul_55: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_75, arg255_1);  add_75 = arg255_1 = None
        add_76: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_55, arg256_1);  mul_55 = arg256_1 = None
        view_224: "bf16[32768, 128]" = torch.ops.aten.view.default(add_76, _shape_param_255);  _shape_param_255 = None
        permute_112: "bf16[128, 512]" = torch.ops.aten.permute.default(arg257_1, [1, 0]);  arg257_1 = None
        addmm_82: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg258_1, view_224, permute_112);  arg258_1 = view_224 = permute_112 = None
        view_225: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_82, _shape_param_256);  addmm_82 = _shape_param_256 = None
        relu_20: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_225);  view_225 = None
        view_226: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_20, _shape_param_257);  relu_20 = _shape_param_257 = None
        permute_113: "bf16[512, 128]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_83: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg260_1, view_226, permute_113);  arg260_1 = view_226 = permute_113 = None
        view_227: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_83, _shape_param_258);  addmm_83 = _shape_param_258 = None
        add_77: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_227, add_76);  view_227 = add_76 = None
        mul_56: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_77, arg261_1);  add_77 = arg261_1 = None
        add_78: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_56, arg262_1);  mul_56 = arg262_1 = None
        view_228: "bf16[32768, 128]" = torch.ops.aten.view.default(add_78, _shape_param_259);  _shape_param_259 = None
        permute_114: "bf16[128, 512]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_84: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg264_1, view_228, permute_114);  arg264_1 = view_228 = permute_114 = None
        view_229: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_84, _shape_param_260);  addmm_84 = _shape_param_260 = None
        relu_21: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_229);  view_229 = None
        view_230: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_21, _shape_param_261);  relu_21 = _shape_param_261 = None
        permute_115: "bf16[512, 128]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_85: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg266_1, view_230, permute_115);  arg266_1 = view_230 = permute_115 = None
        view_231: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_85, _shape_param_262);  addmm_85 = _shape_param_262 = None
        add_79: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_231, add_78);  view_231 = add_78 = None
        mul_57: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_79, arg267_1);  add_79 = arg267_1 = None
        add_80: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_57, arg268_1);  mul_57 = arg268_1 = None
        view_232: "bf16[32768, 128]" = torch.ops.aten.view.default(add_80, _shape_param_263);  _shape_param_263 = None
        permute_116: "bf16[128, 512]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_86: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg270_1, view_232, permute_116);  arg270_1 = view_232 = permute_116 = None
        view_233: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_86, _shape_param_264);  addmm_86 = _shape_param_264 = None
        relu_22: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_233);  view_233 = None
        view_234: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_22, _shape_param_265);  relu_22 = _shape_param_265 = None
        permute_117: "bf16[512, 128]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_87: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg272_1, view_234, permute_117);  arg272_1 = view_234 = permute_117 = None
        view_235: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_87, _shape_param_266);  addmm_87 = _shape_param_266 = None
        add_81: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_235, add_80);  view_235 = add_80 = None
        mul_58: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_81, arg273_1);  add_81 = arg273_1 = None
        add_82: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_58, arg274_1);  mul_58 = arg274_1 = None
        view_236: "bf16[32768, 128]" = torch.ops.aten.view.default(add_82, _shape_param_267);  _shape_param_267 = None
        permute_118: "bf16[128, 512]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_88: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg276_1, view_236, permute_118);  arg276_1 = view_236 = permute_118 = None
        view_237: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_88, _shape_param_268);  addmm_88 = _shape_param_268 = None
        relu_23: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_237);  view_237 = None
        view_238: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_23, _shape_param_269);  relu_23 = _shape_param_269 = None
        permute_119: "bf16[512, 128]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_89: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg278_1, view_238, permute_119);  arg278_1 = view_238 = permute_119 = None
        view_239: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_89, _shape_param_270);  addmm_89 = _shape_param_270 = None
        add_83: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_239, add_82);  view_239 = add_82 = None
        mul_59: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_83, arg279_1);  add_83 = arg279_1 = None
        add_84: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_59, arg280_1);  mul_59 = arg280_1 = None
        view_240: "bf16[32768, 128]" = torch.ops.aten.view.default(add_84, _shape_param_271);  add_84 = _shape_param_271 = None
        permute_120: "bf16[128, 512]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_90: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg282_1, view_240, permute_120);  arg282_1 = view_240 = permute_120 = None
        view_241: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_90, _shape_param_272);  addmm_90 = _shape_param_272 = None
        add_85: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_241, add_72);  view_241 = add_72 = None
        mul_60: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_85, arg283_1);  add_85 = arg283_1 = None
        add_86: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_60, arg284_1);  mul_60 = arg284_1 = None
        view_242: "bf16[32768, 512]" = torch.ops.aten.view.default(add_86, _shape_param_273);  _shape_param_273 = None
        permute_121: "bf16[512, 128]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_91: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg286_1, view_242, permute_121);  arg286_1 = view_242 = permute_121 = None
        view_243: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_91, _shape_param_274);  addmm_91 = _shape_param_274 = None
        mul_61: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_243, arg287_1);  view_243 = arg287_1 = None
        add_87: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_61, arg288_1);  mul_61 = arg288_1 = None
        view_244: "bf16[32768, 128]" = torch.ops.aten.view.default(add_87, _shape_param_275);  _shape_param_275 = None
        permute_122: "bf16[128, 128]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_92: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg290_1, view_244, permute_122);  arg290_1 = view_244 = permute_122 = None
        view_245: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_92, _shape_param_276);  addmm_92 = _shape_param_276 = None
        view_246: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_245, _shape_param_277);  view_245 = _shape_param_277 = None
        permute_123: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_246, [0, 2, 1, 3]);  view_246 = None
        mul_62: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_123, 0.4204482076268573);  permute_123 = None
        expand_24: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_62, _shape_param_278);  mul_62 = _shape_param_278 = None
        clone_24: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_24, memory_format = torch.contiguous_format);  expand_24 = None
        view_247: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_24, _shape_param_279);  clone_24 = _shape_param_279 = None
        view_248: "bf16[32768, 128]" = torch.ops.aten.view.default(add_87, _shape_param_280);  add_87 = _shape_param_280 = None
        permute_124: "bf16[128, 128]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_93: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg292_1, view_248, permute_124);  arg292_1 = view_248 = permute_124 = None
        view_249: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_93, _shape_param_281);  addmm_93 = _shape_param_281 = None
        view_250: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_249, _shape_param_282);  view_249 = _shape_param_282 = None
        permute_125: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None
        permute_126: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_125, [0, 1, 3, 2]);  permute_125 = None
        mul_63: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_126, 0.4204482076268573);  permute_126 = None
        expand_25: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_63, _shape_param_283);  mul_63 = _shape_param_283 = None
        clone_25: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_251: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_25, _shape_param_284);  clone_25 = _shape_param_284 = None
        bmm_12: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_247, view_251);  view_247 = view_251 = None
        view_252: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_12, _shape_param_285);  bmm_12 = _shape_param_285 = None
        eq_6: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_252, -inf)
        logical_not_12: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_7: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_286, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_286 = None
        convert_element_type_12: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_252, torch.float32);  view_252 = None
        amax_6: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_12, [-1], True)
        sub_6: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_12, amax_6);  convert_element_type_12 = amax_6 = None
        exp_6: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_6);  sub_6 = None
        sum_7: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        convert_element_type_13: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_6, torch.bfloat16);  div_6 = None
        where_6: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_13, full_7, convert_element_type_13);  logical_not_13 = full_7 = convert_element_type_13 = None
        expand_26: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_6, _shape_param_287);  where_6 = _shape_param_287 = None
        view_253: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_26, _shape_param_288);  expand_26 = _shape_param_288 = None
        view_254: "bf16[32768, 512]" = torch.ops.aten.view.default(add_86, _shape_param_289);  _shape_param_289 = None
        permute_127: "bf16[512, 128]" = torch.ops.aten.permute.default(arg293_1, [1, 0]);  arg293_1 = None
        addmm_94: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg294_1, view_254, permute_127);  arg294_1 = view_254 = permute_127 = None
        view_255: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_94, _shape_param_290);  addmm_94 = _shape_param_290 = None
        view_256: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_255, _shape_param_291);  view_255 = _shape_param_291 = None
        permute_128: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None
        expand_27: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_128, _shape_param_292);  permute_128 = _shape_param_292 = None
        clone_26: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_257: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_26, _shape_param_293);  clone_26 = _shape_param_293 = None
        bmm_13: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_253, view_257);  view_253 = view_257 = None
        view_258: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_13, _shape_param_294);  bmm_13 = _shape_param_294 = None
        permute_129: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_258, [0, 2, 1, 3]);  view_258 = None
        clone_27: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_129, memory_format = torch.contiguous_format);  permute_129 = None
        view_259: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_27, _shape_param_295);  clone_27 = _shape_param_295 = None
        view_260: "bf16[32768, 128]" = torch.ops.aten.view.default(view_259, _shape_param_296);  view_259 = _shape_param_296 = None
        permute_130: "bf16[128, 128]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_95: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg296_1, view_260, permute_130);  arg296_1 = view_260 = permute_130 = None
        view_261: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_95, _shape_param_297);  addmm_95 = _shape_param_297 = None
        view_262: "bf16[32768, 512]" = torch.ops.aten.view.default(add_86, _shape_param_298);  _shape_param_298 = None
        permute_131: "bf16[512, 128]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_96: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg298_1, view_262, permute_131);  arg298_1 = view_262 = permute_131 = None
        view_263: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_96, _shape_param_299);  addmm_96 = _shape_param_299 = None
        mul_64: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_263, arg299_1);  view_263 = arg299_1 = None
        add_88: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_64, arg300_1);  mul_64 = arg300_1 = None
        add_89: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_261, add_88);  view_261 = add_88 = None
        mul_65: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_89, arg301_1);  add_89 = arg301_1 = None
        add_90: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_65, arg302_1);  mul_65 = arg302_1 = None
        view_264: "bf16[32768, 128]" = torch.ops.aten.view.default(add_90, _shape_param_300);  _shape_param_300 = None
        permute_132: "bf16[128, 512]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_97: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg304_1, view_264, permute_132);  arg304_1 = view_264 = permute_132 = None
        view_265: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_97, _shape_param_301);  addmm_97 = _shape_param_301 = None
        relu_24: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_265);  view_265 = None
        view_266: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_24, _shape_param_302);  relu_24 = _shape_param_302 = None
        permute_133: "bf16[512, 128]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_98: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg306_1, view_266, permute_133);  arg306_1 = view_266 = permute_133 = None
        view_267: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_98, _shape_param_303);  addmm_98 = _shape_param_303 = None
        add_91: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_267, add_90);  view_267 = add_90 = None
        mul_66: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_91, arg307_1);  add_91 = arg307_1 = None
        add_92: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_66, arg308_1);  mul_66 = arg308_1 = None
        view_268: "bf16[32768, 128]" = torch.ops.aten.view.default(add_92, _shape_param_304);  _shape_param_304 = None
        permute_134: "bf16[128, 512]" = torch.ops.aten.permute.default(arg309_1, [1, 0]);  arg309_1 = None
        addmm_99: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg310_1, view_268, permute_134);  arg310_1 = view_268 = permute_134 = None
        view_269: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_99, _shape_param_305);  addmm_99 = _shape_param_305 = None
        relu_25: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_269);  view_269 = None
        view_270: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_25, _shape_param_306);  relu_25 = _shape_param_306 = None
        permute_135: "bf16[512, 128]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_100: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg312_1, view_270, permute_135);  arg312_1 = view_270 = permute_135 = None
        view_271: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_100, _shape_param_307);  addmm_100 = _shape_param_307 = None
        add_93: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_271, add_92);  view_271 = add_92 = None
        mul_67: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_93, arg313_1);  add_93 = arg313_1 = None
        add_94: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_67, arg314_1);  mul_67 = arg314_1 = None
        view_272: "bf16[32768, 128]" = torch.ops.aten.view.default(add_94, _shape_param_308);  _shape_param_308 = None
        permute_136: "bf16[128, 512]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_101: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg316_1, view_272, permute_136);  arg316_1 = view_272 = permute_136 = None
        view_273: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_101, _shape_param_309);  addmm_101 = _shape_param_309 = None
        relu_26: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_273);  view_273 = None
        view_274: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_26, _shape_param_310);  relu_26 = _shape_param_310 = None
        permute_137: "bf16[512, 128]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_102: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg318_1, view_274, permute_137);  arg318_1 = view_274 = permute_137 = None
        view_275: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_102, _shape_param_311);  addmm_102 = _shape_param_311 = None
        add_95: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_275, add_94);  view_275 = add_94 = None
        mul_68: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_95, arg319_1);  add_95 = arg319_1 = None
        add_96: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_68, arg320_1);  mul_68 = arg320_1 = None
        view_276: "bf16[32768, 128]" = torch.ops.aten.view.default(add_96, _shape_param_312);  _shape_param_312 = None
        permute_138: "bf16[128, 512]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_103: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg322_1, view_276, permute_138);  arg322_1 = view_276 = permute_138 = None
        view_277: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_103, _shape_param_313);  addmm_103 = _shape_param_313 = None
        relu_27: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_277);  view_277 = None
        view_278: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_27, _shape_param_314);  relu_27 = _shape_param_314 = None
        permute_139: "bf16[512, 128]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_104: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg324_1, view_278, permute_139);  arg324_1 = view_278 = permute_139 = None
        view_279: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_104, _shape_param_315);  addmm_104 = _shape_param_315 = None
        add_97: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_279, add_96);  view_279 = add_96 = None
        mul_69: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_97, arg325_1);  add_97 = arg325_1 = None
        add_98: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_69, arg326_1);  mul_69 = arg326_1 = None
        view_280: "bf16[32768, 128]" = torch.ops.aten.view.default(add_98, _shape_param_316);  add_98 = _shape_param_316 = None
        permute_140: "bf16[128, 512]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_105: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg328_1, view_280, permute_140);  arg328_1 = view_280 = permute_140 = None
        view_281: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_105, _shape_param_317);  addmm_105 = _shape_param_317 = None
        add_99: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_281, add_86);  view_281 = add_86 = None
        mul_70: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_99, arg329_1);  add_99 = arg329_1 = None
        add_100: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_70, arg330_1);  mul_70 = arg330_1 = None
        view_282: "bf16[32768, 512]" = torch.ops.aten.view.default(add_100, _shape_param_318);  _shape_param_318 = None
        permute_141: "bf16[512, 128]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_106: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg332_1, view_282, permute_141);  arg332_1 = view_282 = permute_141 = None
        view_283: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_106, _shape_param_319);  addmm_106 = _shape_param_319 = None
        mul_71: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_283, arg333_1);  view_283 = arg333_1 = None
        add_101: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_71, arg334_1);  mul_71 = arg334_1 = None
        view_284: "bf16[32768, 128]" = torch.ops.aten.view.default(add_101, _shape_param_320);  _shape_param_320 = None
        permute_142: "bf16[128, 128]" = torch.ops.aten.permute.default(arg335_1, [1, 0]);  arg335_1 = None
        addmm_107: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg336_1, view_284, permute_142);  arg336_1 = view_284 = permute_142 = None
        view_285: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_107, _shape_param_321);  addmm_107 = _shape_param_321 = None
        view_286: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_285, _shape_param_322);  view_285 = _shape_param_322 = None
        permute_143: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_286, [0, 2, 1, 3]);  view_286 = None
        mul_72: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_143, 0.4204482076268573);  permute_143 = None
        expand_28: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_72, _shape_param_323);  mul_72 = _shape_param_323 = None
        clone_28: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_287: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_28, _shape_param_324);  clone_28 = _shape_param_324 = None
        view_288: "bf16[32768, 128]" = torch.ops.aten.view.default(add_101, _shape_param_325);  add_101 = _shape_param_325 = None
        permute_144: "bf16[128, 128]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_108: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg338_1, view_288, permute_144);  arg338_1 = view_288 = permute_144 = None
        view_289: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_108, _shape_param_326);  addmm_108 = _shape_param_326 = None
        view_290: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_289, _shape_param_327);  view_289 = _shape_param_327 = None
        permute_145: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None
        permute_146: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_145, [0, 1, 3, 2]);  permute_145 = None
        mul_73: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_146, 0.4204482076268573);  permute_146 = None
        expand_29: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_73, _shape_param_328);  mul_73 = _shape_param_328 = None
        clone_29: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_29, memory_format = torch.contiguous_format);  expand_29 = None
        view_291: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_29, _shape_param_329);  clone_29 = _shape_param_329 = None
        bmm_14: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_287, view_291);  view_287 = view_291 = None
        view_292: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_14, _shape_param_330);  bmm_14 = _shape_param_330 = None
        eq_7: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_292, -inf)
        logical_not_14: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_8: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_331, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_331 = None
        convert_element_type_14: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_292, torch.float32);  view_292 = None
        amax_7: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_14, [-1], True)
        sub_7: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_14, amax_7);  convert_element_type_14 = amax_7 = None
        exp_7: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_7);  sub_7 = None
        sum_8: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        convert_element_type_15: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_7, torch.bfloat16);  div_7 = None
        where_7: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_15, full_8, convert_element_type_15);  logical_not_15 = full_8 = convert_element_type_15 = None
        expand_30: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_7, _shape_param_332);  where_7 = _shape_param_332 = None
        view_293: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_30, _shape_param_333);  expand_30 = _shape_param_333 = None
        view_294: "bf16[32768, 512]" = torch.ops.aten.view.default(add_100, _shape_param_334);  _shape_param_334 = None
        permute_147: "bf16[512, 128]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_109: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg340_1, view_294, permute_147);  arg340_1 = view_294 = permute_147 = None
        view_295: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_109, _shape_param_335);  addmm_109 = _shape_param_335 = None
        view_296: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_295, _shape_param_336);  view_295 = _shape_param_336 = None
        permute_148: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None
        expand_31: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_148, _shape_param_337);  permute_148 = _shape_param_337 = None
        clone_30: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_31, memory_format = torch.contiguous_format);  expand_31 = None
        view_297: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_30, _shape_param_338);  clone_30 = _shape_param_338 = None
        bmm_15: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_293, view_297);  view_293 = view_297 = None
        view_298: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_15, _shape_param_339);  bmm_15 = _shape_param_339 = None
        permute_149: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_298, [0, 2, 1, 3]);  view_298 = None
        clone_31: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_149, memory_format = torch.contiguous_format);  permute_149 = None
        view_299: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_31, _shape_param_340);  clone_31 = _shape_param_340 = None
        view_300: "bf16[32768, 128]" = torch.ops.aten.view.default(view_299, _shape_param_341);  view_299 = _shape_param_341 = None
        permute_150: "bf16[128, 128]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_110: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg342_1, view_300, permute_150);  arg342_1 = view_300 = permute_150 = None
        view_301: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_110, _shape_param_342);  addmm_110 = _shape_param_342 = None
        view_302: "bf16[32768, 512]" = torch.ops.aten.view.default(add_100, _shape_param_343);  _shape_param_343 = None
        permute_151: "bf16[512, 128]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_111: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg344_1, view_302, permute_151);  arg344_1 = view_302 = permute_151 = None
        view_303: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_111, _shape_param_344);  addmm_111 = _shape_param_344 = None
        mul_74: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_303, arg345_1);  view_303 = arg345_1 = None
        add_102: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_74, arg346_1);  mul_74 = arg346_1 = None
        add_103: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_301, add_102);  view_301 = add_102 = None
        mul_75: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_103, arg347_1);  add_103 = arg347_1 = None
        add_104: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_75, arg348_1);  mul_75 = arg348_1 = None
        view_304: "bf16[32768, 128]" = torch.ops.aten.view.default(add_104, _shape_param_345);  _shape_param_345 = None
        permute_152: "bf16[128, 512]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_112: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg350_1, view_304, permute_152);  arg350_1 = view_304 = permute_152 = None
        view_305: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_112, _shape_param_346);  addmm_112 = _shape_param_346 = None
        relu_28: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_305);  view_305 = None
        view_306: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_28, _shape_param_347);  relu_28 = _shape_param_347 = None
        permute_153: "bf16[512, 128]" = torch.ops.aten.permute.default(arg351_1, [1, 0]);  arg351_1 = None
        addmm_113: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg352_1, view_306, permute_153);  arg352_1 = view_306 = permute_153 = None
        view_307: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_113, _shape_param_348);  addmm_113 = _shape_param_348 = None
        add_105: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_307, add_104);  view_307 = add_104 = None
        mul_76: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_105, arg353_1);  add_105 = arg353_1 = None
        add_106: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_76, arg354_1);  mul_76 = arg354_1 = None
        view_308: "bf16[32768, 128]" = torch.ops.aten.view.default(add_106, _shape_param_349);  _shape_param_349 = None
        permute_154: "bf16[128, 512]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_114: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg356_1, view_308, permute_154);  arg356_1 = view_308 = permute_154 = None
        view_309: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_114, _shape_param_350);  addmm_114 = _shape_param_350 = None
        relu_29: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_309);  view_309 = None
        view_310: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_29, _shape_param_351);  relu_29 = _shape_param_351 = None
        permute_155: "bf16[512, 128]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_115: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg358_1, view_310, permute_155);  arg358_1 = view_310 = permute_155 = None
        view_311: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_115, _shape_param_352);  addmm_115 = _shape_param_352 = None
        add_107: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_311, add_106);  view_311 = add_106 = None
        mul_77: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_107, arg359_1);  add_107 = arg359_1 = None
        add_108: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_77, arg360_1);  mul_77 = arg360_1 = None
        view_312: "bf16[32768, 128]" = torch.ops.aten.view.default(add_108, _shape_param_353);  _shape_param_353 = None
        permute_156: "bf16[128, 512]" = torch.ops.aten.permute.default(arg361_1, [1, 0]);  arg361_1 = None
        addmm_116: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg362_1, view_312, permute_156);  arg362_1 = view_312 = permute_156 = None
        view_313: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_116, _shape_param_354);  addmm_116 = _shape_param_354 = None
        relu_30: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_313);  view_313 = None
        view_314: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_30, _shape_param_355);  relu_30 = _shape_param_355 = None
        permute_157: "bf16[512, 128]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_117: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg364_1, view_314, permute_157);  arg364_1 = view_314 = permute_157 = None
        view_315: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_117, _shape_param_356);  addmm_117 = _shape_param_356 = None
        add_109: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_315, add_108);  view_315 = add_108 = None
        mul_78: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_109, arg365_1);  add_109 = arg365_1 = None
        add_110: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_78, arg366_1);  mul_78 = arg366_1 = None
        view_316: "bf16[32768, 128]" = torch.ops.aten.view.default(add_110, _shape_param_357);  _shape_param_357 = None
        permute_158: "bf16[128, 512]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_118: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg368_1, view_316, permute_158);  arg368_1 = view_316 = permute_158 = None
        view_317: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_118, _shape_param_358);  addmm_118 = _shape_param_358 = None
        relu_31: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_317);  view_317 = None
        view_318: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_31, _shape_param_359);  relu_31 = _shape_param_359 = None
        permute_159: "bf16[512, 128]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_119: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg370_1, view_318, permute_159);  arg370_1 = view_318 = permute_159 = None
        view_319: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_119, _shape_param_360);  addmm_119 = _shape_param_360 = None
        add_111: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_319, add_110);  view_319 = add_110 = None
        mul_79: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_111, arg371_1);  add_111 = arg371_1 = None
        add_112: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_79, arg372_1);  mul_79 = arg372_1 = None
        view_320: "bf16[32768, 128]" = torch.ops.aten.view.default(add_112, _shape_param_361);  add_112 = _shape_param_361 = None
        permute_160: "bf16[128, 512]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_120: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg374_1, view_320, permute_160);  arg374_1 = view_320 = permute_160 = None
        view_321: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_120, _shape_param_362);  addmm_120 = _shape_param_362 = None
        add_113: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_321, add_100);  view_321 = add_100 = None
        mul_80: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_113, arg375_1);  add_113 = arg375_1 = None
        add_114: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_80, arg376_1);  mul_80 = arg376_1 = None
        view_322: "bf16[32768, 512]" = torch.ops.aten.view.default(add_114, _shape_param_363);  _shape_param_363 = None
        permute_161: "bf16[512, 128]" = torch.ops.aten.permute.default(arg377_1, [1, 0]);  arg377_1 = None
        addmm_121: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg378_1, view_322, permute_161);  arg378_1 = view_322 = permute_161 = None
        view_323: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_121, _shape_param_364);  addmm_121 = _shape_param_364 = None
        mul_81: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_323, arg379_1);  view_323 = arg379_1 = None
        add_115: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_81, arg380_1);  mul_81 = arg380_1 = None
        view_324: "bf16[32768, 128]" = torch.ops.aten.view.default(add_115, _shape_param_365);  _shape_param_365 = None
        permute_162: "bf16[128, 128]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_122: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg382_1, view_324, permute_162);  arg382_1 = view_324 = permute_162 = None
        view_325: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_122, _shape_param_366);  addmm_122 = _shape_param_366 = None
        view_326: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_325, _shape_param_367);  view_325 = _shape_param_367 = None
        permute_163: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_326, [0, 2, 1, 3]);  view_326 = None
        mul_82: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_163, 0.4204482076268573);  permute_163 = None
        expand_32: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_82, _shape_param_368);  mul_82 = _shape_param_368 = None
        clone_32: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_327: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_32, _shape_param_369);  clone_32 = _shape_param_369 = None
        view_328: "bf16[32768, 128]" = torch.ops.aten.view.default(add_115, _shape_param_370);  add_115 = _shape_param_370 = None
        permute_164: "bf16[128, 128]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_123: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg384_1, view_328, permute_164);  arg384_1 = view_328 = permute_164 = None
        view_329: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_123, _shape_param_371);  addmm_123 = _shape_param_371 = None
        view_330: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_329, _shape_param_372);  view_329 = _shape_param_372 = None
        permute_165: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None
        permute_166: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_165, [0, 1, 3, 2]);  permute_165 = None
        mul_83: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_166, 0.4204482076268573);  permute_166 = None
        expand_33: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_83, _shape_param_373);  mul_83 = _shape_param_373 = None
        clone_33: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_331: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_33, _shape_param_374);  clone_33 = _shape_param_374 = None
        bmm_16: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_327, view_331);  view_327 = view_331 = None
        view_332: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_16, _shape_param_375);  bmm_16 = _shape_param_375 = None
        eq_8: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_332, -inf)
        logical_not_16: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_9: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_376, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_376 = None
        convert_element_type_16: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_332, torch.float32);  view_332 = None
        amax_8: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_16, [-1], True)
        sub_8: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_16, amax_8);  convert_element_type_16 = amax_8 = None
        exp_8: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_8);  sub_8 = None
        sum_9: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        convert_element_type_17: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_8, torch.bfloat16);  div_8 = None
        where_8: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_17, full_9, convert_element_type_17);  logical_not_17 = full_9 = convert_element_type_17 = None
        expand_34: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_8, _shape_param_377);  where_8 = _shape_param_377 = None
        view_333: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_34, _shape_param_378);  expand_34 = _shape_param_378 = None
        view_334: "bf16[32768, 512]" = torch.ops.aten.view.default(add_114, _shape_param_379);  _shape_param_379 = None
        permute_167: "bf16[512, 128]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_124: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg386_1, view_334, permute_167);  arg386_1 = view_334 = permute_167 = None
        view_335: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_124, _shape_param_380);  addmm_124 = _shape_param_380 = None
        view_336: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_335, _shape_param_381);  view_335 = _shape_param_381 = None
        permute_168: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None
        expand_35: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_168, _shape_param_382);  permute_168 = _shape_param_382 = None
        clone_34: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_337: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_34, _shape_param_383);  clone_34 = _shape_param_383 = None
        bmm_17: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_333, view_337);  view_333 = view_337 = None
        view_338: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_17, _shape_param_384);  bmm_17 = _shape_param_384 = None
        permute_169: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_338, [0, 2, 1, 3]);  view_338 = None
        clone_35: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_169, memory_format = torch.contiguous_format);  permute_169 = None
        view_339: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_35, _shape_param_385);  clone_35 = _shape_param_385 = None
        view_340: "bf16[32768, 128]" = torch.ops.aten.view.default(view_339, _shape_param_386);  view_339 = _shape_param_386 = None
        permute_170: "bf16[128, 128]" = torch.ops.aten.permute.default(arg387_1, [1, 0]);  arg387_1 = None
        addmm_125: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg388_1, view_340, permute_170);  arg388_1 = view_340 = permute_170 = None
        view_341: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_125, _shape_param_387);  addmm_125 = _shape_param_387 = None
        view_342: "bf16[32768, 512]" = torch.ops.aten.view.default(add_114, _shape_param_388);  _shape_param_388 = None
        permute_171: "bf16[512, 128]" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_126: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg390_1, view_342, permute_171);  arg390_1 = view_342 = permute_171 = None
        view_343: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_126, _shape_param_389);  addmm_126 = _shape_param_389 = None
        mul_84: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_343, arg391_1);  view_343 = arg391_1 = None
        add_116: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_84, arg392_1);  mul_84 = arg392_1 = None
        add_117: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_341, add_116);  view_341 = add_116 = None
        mul_85: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_117, arg393_1);  add_117 = arg393_1 = None
        add_118: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_85, arg394_1);  mul_85 = arg394_1 = None
        view_344: "bf16[32768, 128]" = torch.ops.aten.view.default(add_118, _shape_param_390);  _shape_param_390 = None
        permute_172: "bf16[128, 512]" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_127: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg396_1, view_344, permute_172);  arg396_1 = view_344 = permute_172 = None
        view_345: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_127, _shape_param_391);  addmm_127 = _shape_param_391 = None
        relu_32: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_345);  view_345 = None
        view_346: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_32, _shape_param_392);  relu_32 = _shape_param_392 = None
        permute_173: "bf16[512, 128]" = torch.ops.aten.permute.default(arg397_1, [1, 0]);  arg397_1 = None
        addmm_128: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg398_1, view_346, permute_173);  arg398_1 = view_346 = permute_173 = None
        view_347: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_128, _shape_param_393);  addmm_128 = _shape_param_393 = None
        add_119: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_347, add_118);  view_347 = add_118 = None
        mul_86: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_119, arg399_1);  add_119 = arg399_1 = None
        add_120: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_86, arg400_1);  mul_86 = arg400_1 = None
        view_348: "bf16[32768, 128]" = torch.ops.aten.view.default(add_120, _shape_param_394);  _shape_param_394 = None
        permute_174: "bf16[128, 512]" = torch.ops.aten.permute.default(arg401_1, [1, 0]);  arg401_1 = None
        addmm_129: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg402_1, view_348, permute_174);  arg402_1 = view_348 = permute_174 = None
        view_349: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_129, _shape_param_395);  addmm_129 = _shape_param_395 = None
        relu_33: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_349);  view_349 = None
        view_350: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_33, _shape_param_396);  relu_33 = _shape_param_396 = None
        permute_175: "bf16[512, 128]" = torch.ops.aten.permute.default(arg403_1, [1, 0]);  arg403_1 = None
        addmm_130: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg404_1, view_350, permute_175);  arg404_1 = view_350 = permute_175 = None
        view_351: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_130, _shape_param_397);  addmm_130 = _shape_param_397 = None
        add_121: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_351, add_120);  view_351 = add_120 = None
        mul_87: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_121, arg405_1);  add_121 = arg405_1 = None
        add_122: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_87, arg406_1);  mul_87 = arg406_1 = None
        view_352: "bf16[32768, 128]" = torch.ops.aten.view.default(add_122, _shape_param_398);  _shape_param_398 = None
        permute_176: "bf16[128, 512]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        addmm_131: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg408_1, view_352, permute_176);  arg408_1 = view_352 = permute_176 = None
        view_353: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_131, _shape_param_399);  addmm_131 = _shape_param_399 = None
        relu_34: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_353);  view_353 = None
        view_354: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_34, _shape_param_400);  relu_34 = _shape_param_400 = None
        permute_177: "bf16[512, 128]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_132: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg410_1, view_354, permute_177);  arg410_1 = view_354 = permute_177 = None
        view_355: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_132, _shape_param_401);  addmm_132 = _shape_param_401 = None
        add_123: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_355, add_122);  view_355 = add_122 = None
        mul_88: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_123, arg411_1);  add_123 = arg411_1 = None
        add_124: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_88, arg412_1);  mul_88 = arg412_1 = None
        view_356: "bf16[32768, 128]" = torch.ops.aten.view.default(add_124, _shape_param_402);  _shape_param_402 = None
        permute_178: "bf16[128, 512]" = torch.ops.aten.permute.default(arg413_1, [1, 0]);  arg413_1 = None
        addmm_133: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg414_1, view_356, permute_178);  arg414_1 = view_356 = permute_178 = None
        view_357: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_133, _shape_param_403);  addmm_133 = _shape_param_403 = None
        relu_35: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_357);  view_357 = None
        view_358: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_35, _shape_param_404);  relu_35 = _shape_param_404 = None
        permute_179: "bf16[512, 128]" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_134: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg416_1, view_358, permute_179);  arg416_1 = view_358 = permute_179 = None
        view_359: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_134, _shape_param_405);  addmm_134 = _shape_param_405 = None
        add_125: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_359, add_124);  view_359 = add_124 = None
        mul_89: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_125, arg417_1);  add_125 = arg417_1 = None
        add_126: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_89, arg418_1);  mul_89 = arg418_1 = None
        view_360: "bf16[32768, 128]" = torch.ops.aten.view.default(add_126, _shape_param_406);  add_126 = _shape_param_406 = None
        permute_180: "bf16[128, 512]" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_135: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg420_1, view_360, permute_180);  arg420_1 = view_360 = permute_180 = None
        view_361: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_135, _shape_param_407);  addmm_135 = _shape_param_407 = None
        add_127: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_361, add_114);  view_361 = add_114 = None
        mul_90: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_127, arg421_1);  add_127 = arg421_1 = None
        add_128: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_90, arg422_1);  mul_90 = arg422_1 = None
        view_362: "bf16[32768, 512]" = torch.ops.aten.view.default(add_128, _shape_param_408);  _shape_param_408 = None
        permute_181: "bf16[512, 128]" = torch.ops.aten.permute.default(arg423_1, [1, 0]);  arg423_1 = None
        addmm_136: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg424_1, view_362, permute_181);  arg424_1 = view_362 = permute_181 = None
        view_363: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_136, _shape_param_409);  addmm_136 = _shape_param_409 = None
        mul_91: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_363, arg425_1);  view_363 = arg425_1 = None
        add_129: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_91, arg426_1);  mul_91 = arg426_1 = None
        view_364: "bf16[32768, 128]" = torch.ops.aten.view.default(add_129, _shape_param_410);  _shape_param_410 = None
        permute_182: "bf16[128, 128]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        addmm_137: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg428_1, view_364, permute_182);  arg428_1 = view_364 = permute_182 = None
        view_365: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_137, _shape_param_411);  addmm_137 = _shape_param_411 = None
        view_366: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_365, _shape_param_412);  view_365 = _shape_param_412 = None
        permute_183: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_366, [0, 2, 1, 3]);  view_366 = None
        mul_92: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_183, 0.4204482076268573);  permute_183 = None
        expand_36: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_92, _shape_param_413);  mul_92 = _shape_param_413 = None
        clone_36: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_36, memory_format = torch.contiguous_format);  expand_36 = None
        view_367: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_36, _shape_param_414);  clone_36 = _shape_param_414 = None
        view_368: "bf16[32768, 128]" = torch.ops.aten.view.default(add_129, _shape_param_415);  add_129 = _shape_param_415 = None
        permute_184: "bf16[128, 128]" = torch.ops.aten.permute.default(arg429_1, [1, 0]);  arg429_1 = None
        addmm_138: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg430_1, view_368, permute_184);  arg430_1 = view_368 = permute_184 = None
        view_369: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_138, _shape_param_416);  addmm_138 = _shape_param_416 = None
        view_370: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_369, _shape_param_417);  view_369 = _shape_param_417 = None
        permute_185: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None
        permute_186: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_185, [0, 1, 3, 2]);  permute_185 = None
        mul_93: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_186, 0.4204482076268573);  permute_186 = None
        expand_37: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_93, _shape_param_418);  mul_93 = _shape_param_418 = None
        clone_37: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_371: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_37, _shape_param_419);  clone_37 = _shape_param_419 = None
        bmm_18: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_367, view_371);  view_367 = view_371 = None
        view_372: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_18, _shape_param_420);  bmm_18 = _shape_param_420 = None
        eq_9: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_372, -inf)
        logical_not_18: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_10: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_421, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_421 = None
        convert_element_type_18: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_372, torch.float32);  view_372 = None
        amax_9: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_18, [-1], True)
        sub_9: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_18, amax_9);  convert_element_type_18 = amax_9 = None
        exp_9: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_10: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        convert_element_type_19: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_9, torch.bfloat16);  div_9 = None
        where_9: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_19, full_10, convert_element_type_19);  logical_not_19 = full_10 = convert_element_type_19 = None
        expand_38: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_9, _shape_param_422);  where_9 = _shape_param_422 = None
        view_373: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_38, _shape_param_423);  expand_38 = _shape_param_423 = None
        view_374: "bf16[32768, 512]" = torch.ops.aten.view.default(add_128, _shape_param_424);  _shape_param_424 = None
        permute_187: "bf16[512, 128]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_139: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg432_1, view_374, permute_187);  arg432_1 = view_374 = permute_187 = None
        view_375: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_139, _shape_param_425);  addmm_139 = _shape_param_425 = None
        view_376: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_375, _shape_param_426);  view_375 = _shape_param_426 = None
        permute_188: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        expand_39: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_188, _shape_param_427);  permute_188 = _shape_param_427 = None
        clone_38: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_39, memory_format = torch.contiguous_format);  expand_39 = None
        view_377: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_38, _shape_param_428);  clone_38 = _shape_param_428 = None
        bmm_19: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_373, view_377);  view_373 = view_377 = None
        view_378: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_19, _shape_param_429);  bmm_19 = _shape_param_429 = None
        permute_189: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_378, [0, 2, 1, 3]);  view_378 = None
        clone_39: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None
        view_379: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_39, _shape_param_430);  clone_39 = _shape_param_430 = None
        view_380: "bf16[32768, 128]" = torch.ops.aten.view.default(view_379, _shape_param_431);  view_379 = _shape_param_431 = None
        permute_190: "bf16[128, 128]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
        addmm_140: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg434_1, view_380, permute_190);  arg434_1 = view_380 = permute_190 = None
        view_381: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_140, _shape_param_432);  addmm_140 = _shape_param_432 = None
        view_382: "bf16[32768, 512]" = torch.ops.aten.view.default(add_128, _shape_param_433);  _shape_param_433 = None
        permute_191: "bf16[512, 128]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_141: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg436_1, view_382, permute_191);  arg436_1 = view_382 = permute_191 = None
        view_383: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_141, _shape_param_434);  addmm_141 = _shape_param_434 = None
        mul_94: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_383, arg437_1);  view_383 = arg437_1 = None
        add_130: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_94, arg438_1);  mul_94 = arg438_1 = None
        add_131: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_381, add_130);  view_381 = add_130 = None
        mul_95: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_131, arg439_1);  add_131 = arg439_1 = None
        add_132: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_95, arg440_1);  mul_95 = arg440_1 = None
        view_384: "bf16[32768, 128]" = torch.ops.aten.view.default(add_132, _shape_param_435);  _shape_param_435 = None
        permute_192: "bf16[128, 512]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_142: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg442_1, view_384, permute_192);  arg442_1 = view_384 = permute_192 = None
        view_385: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_142, _shape_param_436);  addmm_142 = _shape_param_436 = None
        relu_36: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_385);  view_385 = None
        view_386: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_36, _shape_param_437);  relu_36 = _shape_param_437 = None
        permute_193: "bf16[512, 128]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        addmm_143: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg444_1, view_386, permute_193);  arg444_1 = view_386 = permute_193 = None
        view_387: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_143, _shape_param_438);  addmm_143 = _shape_param_438 = None
        add_133: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_387, add_132);  view_387 = add_132 = None
        mul_96: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_133, arg445_1);  add_133 = arg445_1 = None
        add_134: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_96, arg446_1);  mul_96 = arg446_1 = None
        view_388: "bf16[32768, 128]" = torch.ops.aten.view.default(add_134, _shape_param_439);  _shape_param_439 = None
        permute_194: "bf16[128, 512]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_144: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg448_1, view_388, permute_194);  arg448_1 = view_388 = permute_194 = None
        view_389: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_144, _shape_param_440);  addmm_144 = _shape_param_440 = None
        relu_37: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_389);  view_389 = None
        view_390: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_37, _shape_param_441);  relu_37 = _shape_param_441 = None
        permute_195: "bf16[512, 128]" = torch.ops.aten.permute.default(arg449_1, [1, 0]);  arg449_1 = None
        addmm_145: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg450_1, view_390, permute_195);  arg450_1 = view_390 = permute_195 = None
        view_391: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_145, _shape_param_442);  addmm_145 = _shape_param_442 = None
        add_135: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_391, add_134);  view_391 = add_134 = None
        mul_97: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_135, arg451_1);  add_135 = arg451_1 = None
        add_136: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_97, arg452_1);  mul_97 = arg452_1 = None
        view_392: "bf16[32768, 128]" = torch.ops.aten.view.default(add_136, _shape_param_443);  _shape_param_443 = None
        permute_196: "bf16[128, 512]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_146: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg454_1, view_392, permute_196);  arg454_1 = view_392 = permute_196 = None
        view_393: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_146, _shape_param_444);  addmm_146 = _shape_param_444 = None
        relu_38: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_393);  view_393 = None
        view_394: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_38, _shape_param_445);  relu_38 = _shape_param_445 = None
        permute_197: "bf16[512, 128]" = torch.ops.aten.permute.default(arg455_1, [1, 0]);  arg455_1 = None
        addmm_147: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg456_1, view_394, permute_197);  arg456_1 = view_394 = permute_197 = None
        view_395: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_147, _shape_param_446);  addmm_147 = _shape_param_446 = None
        add_137: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_395, add_136);  view_395 = add_136 = None
        mul_98: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_137, arg457_1);  add_137 = arg457_1 = None
        add_138: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_98, arg458_1);  mul_98 = arg458_1 = None
        view_396: "bf16[32768, 128]" = torch.ops.aten.view.default(add_138, _shape_param_447);  _shape_param_447 = None
        permute_198: "bf16[128, 512]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_148: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg460_1, view_396, permute_198);  arg460_1 = view_396 = permute_198 = None
        view_397: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_148, _shape_param_448);  addmm_148 = _shape_param_448 = None
        relu_39: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_397);  view_397 = None
        view_398: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_39, _shape_param_449);  relu_39 = _shape_param_449 = None
        permute_199: "bf16[512, 128]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_149: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg462_1, view_398, permute_199);  arg462_1 = view_398 = permute_199 = None
        view_399: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_149, _shape_param_450);  addmm_149 = _shape_param_450 = None
        add_139: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_399, add_138);  view_399 = add_138 = None
        mul_99: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_139, arg463_1);  add_139 = arg463_1 = None
        add_140: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_99, arg464_1);  mul_99 = arg464_1 = None
        view_400: "bf16[32768, 128]" = torch.ops.aten.view.default(add_140, _shape_param_451);  add_140 = _shape_param_451 = None
        permute_200: "bf16[128, 512]" = torch.ops.aten.permute.default(arg465_1, [1, 0]);  arg465_1 = None
        addmm_150: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg466_1, view_400, permute_200);  arg466_1 = view_400 = permute_200 = None
        view_401: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_150, _shape_param_452);  addmm_150 = _shape_param_452 = None
        add_141: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_401, add_128);  view_401 = add_128 = None
        mul_100: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_141, arg467_1);  add_141 = arg467_1 = None
        add_142: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_100, arg468_1);  mul_100 = arg468_1 = None
        view_402: "bf16[32768, 512]" = torch.ops.aten.view.default(add_142, _shape_param_453);  _shape_param_453 = None
        permute_201: "bf16[512, 128]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
        addmm_151: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg470_1, view_402, permute_201);  arg470_1 = view_402 = permute_201 = None
        view_403: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_151, _shape_param_454);  addmm_151 = _shape_param_454 = None
        mul_101: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_403, arg471_1);  view_403 = arg471_1 = None
        add_143: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_101, arg472_1);  mul_101 = arg472_1 = None
        view_404: "bf16[32768, 128]" = torch.ops.aten.view.default(add_143, _shape_param_455);  _shape_param_455 = None
        permute_202: "bf16[128, 128]" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_152: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg474_1, view_404, permute_202);  arg474_1 = view_404 = permute_202 = None
        view_405: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_152, _shape_param_456);  addmm_152 = _shape_param_456 = None
        view_406: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_405, _shape_param_457);  view_405 = _shape_param_457 = None
        permute_203: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_406, [0, 2, 1, 3]);  view_406 = None
        mul_102: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_203, 0.4204482076268573);  permute_203 = None
        expand_40: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_102, _shape_param_458);  mul_102 = _shape_param_458 = None
        clone_40: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_407: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_40, _shape_param_459);  clone_40 = _shape_param_459 = None
        view_408: "bf16[32768, 128]" = torch.ops.aten.view.default(add_143, _shape_param_460);  add_143 = _shape_param_460 = None
        permute_204: "bf16[128, 128]" = torch.ops.aten.permute.default(arg475_1, [1, 0]);  arg475_1 = None
        addmm_153: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg476_1, view_408, permute_204);  arg476_1 = view_408 = permute_204 = None
        view_409: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_153, _shape_param_461);  addmm_153 = _shape_param_461 = None
        view_410: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_409, _shape_param_462);  view_409 = _shape_param_462 = None
        permute_205: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        permute_206: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_205, [0, 1, 3, 2]);  permute_205 = None
        mul_103: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_206, 0.4204482076268573);  permute_206 = None
        expand_41: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_103, _shape_param_463);  mul_103 = _shape_param_463 = None
        clone_41: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_41, memory_format = torch.contiguous_format);  expand_41 = None
        view_411: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_41, _shape_param_464);  clone_41 = _shape_param_464 = None
        bmm_20: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_407, view_411);  view_407 = view_411 = None
        view_412: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_20, _shape_param_465);  bmm_20 = _shape_param_465 = None
        eq_10: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_412, -inf)
        logical_not_20: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_11: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_466, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_466 = None
        convert_element_type_20: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_412, torch.float32);  view_412 = None
        amax_10: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_20, [-1], True)
        sub_10: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_20, amax_10);  convert_element_type_20 = amax_10 = None
        exp_10: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_10);  sub_10 = None
        sum_11: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        convert_element_type_21: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_10, torch.bfloat16);  div_10 = None
        where_10: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_21, full_11, convert_element_type_21);  logical_not_21 = full_11 = convert_element_type_21 = None
        expand_42: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_10, _shape_param_467);  where_10 = _shape_param_467 = None
        view_413: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_42, _shape_param_468);  expand_42 = _shape_param_468 = None
        view_414: "bf16[32768, 512]" = torch.ops.aten.view.default(add_142, _shape_param_469);  _shape_param_469 = None
        permute_207: "bf16[512, 128]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_154: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg478_1, view_414, permute_207);  arg478_1 = view_414 = permute_207 = None
        view_415: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_154, _shape_param_470);  addmm_154 = _shape_param_470 = None
        view_416: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_415, _shape_param_471);  view_415 = _shape_param_471 = None
        permute_208: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_416, [0, 2, 1, 3]);  view_416 = None
        expand_43: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_208, _shape_param_472);  permute_208 = _shape_param_472 = None
        clone_42: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_417: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_42, _shape_param_473);  clone_42 = _shape_param_473 = None
        bmm_21: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_413, view_417);  view_413 = view_417 = None
        view_418: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_21, _shape_param_474);  bmm_21 = _shape_param_474 = None
        permute_209: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_418, [0, 2, 1, 3]);  view_418 = None
        clone_43: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_209, memory_format = torch.contiguous_format);  permute_209 = None
        view_419: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_43, _shape_param_475);  clone_43 = _shape_param_475 = None
        view_420: "bf16[32768, 128]" = torch.ops.aten.view.default(view_419, _shape_param_476);  view_419 = _shape_param_476 = None
        permute_210: "bf16[128, 128]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        addmm_155: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg480_1, view_420, permute_210);  arg480_1 = view_420 = permute_210 = None
        view_421: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_155, _shape_param_477);  addmm_155 = _shape_param_477 = None
        view_422: "bf16[32768, 512]" = torch.ops.aten.view.default(add_142, _shape_param_478);  _shape_param_478 = None
        permute_211: "bf16[512, 128]" = torch.ops.aten.permute.default(arg481_1, [1, 0]);  arg481_1 = None
        addmm_156: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg482_1, view_422, permute_211);  arg482_1 = view_422 = permute_211 = None
        view_423: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_156, _shape_param_479);  addmm_156 = _shape_param_479 = None
        mul_104: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_423, arg483_1);  view_423 = arg483_1 = None
        add_144: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_104, arg484_1);  mul_104 = arg484_1 = None
        add_145: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_421, add_144);  view_421 = add_144 = None
        mul_105: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_145, arg485_1);  add_145 = arg485_1 = None
        add_146: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_105, arg486_1);  mul_105 = arg486_1 = None
        view_424: "bf16[32768, 128]" = torch.ops.aten.view.default(add_146, _shape_param_480);  _shape_param_480 = None
        permute_212: "bf16[128, 512]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_157: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg488_1, view_424, permute_212);  arg488_1 = view_424 = permute_212 = None
        view_425: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_157, _shape_param_481);  addmm_157 = _shape_param_481 = None
        relu_40: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_425);  view_425 = None
        view_426: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_40, _shape_param_482);  relu_40 = _shape_param_482 = None
        permute_213: "bf16[512, 128]" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_158: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg490_1, view_426, permute_213);  arg490_1 = view_426 = permute_213 = None
        view_427: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_158, _shape_param_483);  addmm_158 = _shape_param_483 = None
        add_147: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_427, add_146);  view_427 = add_146 = None
        mul_106: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_147, arg491_1);  add_147 = arg491_1 = None
        add_148: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_106, arg492_1);  mul_106 = arg492_1 = None
        view_428: "bf16[32768, 128]" = torch.ops.aten.view.default(add_148, _shape_param_484);  _shape_param_484 = None
        permute_214: "bf16[128, 512]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_159: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg494_1, view_428, permute_214);  arg494_1 = view_428 = permute_214 = None
        view_429: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_159, _shape_param_485);  addmm_159 = _shape_param_485 = None
        relu_41: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_429);  view_429 = None
        view_430: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_41, _shape_param_486);  relu_41 = _shape_param_486 = None
        permute_215: "bf16[512, 128]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        addmm_160: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg496_1, view_430, permute_215);  arg496_1 = view_430 = permute_215 = None
        view_431: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_160, _shape_param_487);  addmm_160 = _shape_param_487 = None
        add_149: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_431, add_148);  view_431 = add_148 = None
        mul_107: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_149, arg497_1);  add_149 = arg497_1 = None
        add_150: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_107, arg498_1);  mul_107 = arg498_1 = None
        view_432: "bf16[32768, 128]" = torch.ops.aten.view.default(add_150, _shape_param_488);  _shape_param_488 = None
        permute_216: "bf16[128, 512]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_161: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg500_1, view_432, permute_216);  arg500_1 = view_432 = permute_216 = None
        view_433: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_161, _shape_param_489);  addmm_161 = _shape_param_489 = None
        relu_42: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_433);  view_433 = None
        view_434: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_42, _shape_param_490);  relu_42 = _shape_param_490 = None
        permute_217: "bf16[512, 128]" = torch.ops.aten.permute.default(arg501_1, [1, 0]);  arg501_1 = None
        addmm_162: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg502_1, view_434, permute_217);  arg502_1 = view_434 = permute_217 = None
        view_435: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_162, _shape_param_491);  addmm_162 = _shape_param_491 = None
        add_151: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_435, add_150);  view_435 = add_150 = None
        mul_108: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_151, arg503_1);  add_151 = arg503_1 = None
        add_152: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_108, arg504_1);  mul_108 = arg504_1 = None
        view_436: "bf16[32768, 128]" = torch.ops.aten.view.default(add_152, _shape_param_492);  _shape_param_492 = None
        permute_218: "bf16[128, 512]" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
        addmm_163: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg506_1, view_436, permute_218);  arg506_1 = view_436 = permute_218 = None
        view_437: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_163, _shape_param_493);  addmm_163 = _shape_param_493 = None
        relu_43: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_437);  view_437 = None
        view_438: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_43, _shape_param_494);  relu_43 = _shape_param_494 = None
        permute_219: "bf16[512, 128]" = torch.ops.aten.permute.default(arg507_1, [1, 0]);  arg507_1 = None
        addmm_164: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg508_1, view_438, permute_219);  arg508_1 = view_438 = permute_219 = None
        view_439: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_164, _shape_param_495);  addmm_164 = _shape_param_495 = None
        add_153: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_439, add_152);  view_439 = add_152 = None
        mul_109: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_153, arg509_1);  add_153 = arg509_1 = None
        add_154: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_109, arg510_1);  mul_109 = arg510_1 = None
        view_440: "bf16[32768, 128]" = torch.ops.aten.view.default(add_154, _shape_param_496);  add_154 = _shape_param_496 = None
        permute_220: "bf16[128, 512]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
        addmm_165: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg512_1, view_440, permute_220);  arg512_1 = view_440 = permute_220 = None
        view_441: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_165, _shape_param_497);  addmm_165 = _shape_param_497 = None
        add_155: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_441, add_142);  view_441 = add_142 = None
        mul_110: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_155, arg513_1);  add_155 = arg513_1 = None
        add_156: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_110, arg514_1);  mul_110 = arg514_1 = None
        view_442: "bf16[32768, 512]" = torch.ops.aten.view.default(add_156, _shape_param_498);  _shape_param_498 = None
        permute_221: "bf16[512, 128]" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        addmm_166: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg516_1, view_442, permute_221);  arg516_1 = view_442 = permute_221 = None
        view_443: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_166, _shape_param_499);  addmm_166 = _shape_param_499 = None
        mul_111: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_443, arg517_1);  view_443 = arg517_1 = None
        add_157: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_111, arg518_1);  mul_111 = arg518_1 = None
        view_444: "bf16[32768, 128]" = torch.ops.aten.view.default(add_157, _shape_param_500);  _shape_param_500 = None
        permute_222: "bf16[128, 128]" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
        addmm_167: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg520_1, view_444, permute_222);  arg520_1 = view_444 = permute_222 = None
        view_445: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_167, _shape_param_501);  addmm_167 = _shape_param_501 = None
        view_446: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_445, _shape_param_502);  view_445 = _shape_param_502 = None
        permute_223: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_446, [0, 2, 1, 3]);  view_446 = None
        mul_112: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_223, 0.4204482076268573);  permute_223 = None
        expand_44: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_112, _shape_param_503);  mul_112 = _shape_param_503 = None
        clone_44: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_44, memory_format = torch.contiguous_format);  expand_44 = None
        view_447: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_44, _shape_param_504);  clone_44 = _shape_param_504 = None
        view_448: "bf16[32768, 128]" = torch.ops.aten.view.default(add_157, _shape_param_505);  add_157 = _shape_param_505 = None
        permute_224: "bf16[128, 128]" = torch.ops.aten.permute.default(arg521_1, [1, 0]);  arg521_1 = None
        addmm_168: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg522_1, view_448, permute_224);  arg522_1 = view_448 = permute_224 = None
        view_449: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_168, _shape_param_506);  addmm_168 = _shape_param_506 = None
        view_450: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_449, _shape_param_507);  view_449 = _shape_param_507 = None
        permute_225: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_450, [0, 2, 1, 3]);  view_450 = None
        permute_226: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_225, [0, 1, 3, 2]);  permute_225 = None
        mul_113: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_226, 0.4204482076268573);  permute_226 = None
        expand_45: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_113, _shape_param_508);  mul_113 = _shape_param_508 = None
        clone_45: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_451: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_45, _shape_param_509);  clone_45 = _shape_param_509 = None
        bmm_22: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_447, view_451);  view_447 = view_451 = None
        view_452: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_22, _shape_param_510);  bmm_22 = _shape_param_510 = None
        eq_11: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_452, -inf)
        logical_not_22: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_12: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_511, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_511 = None
        convert_element_type_22: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_452, torch.float32);  view_452 = None
        amax_11: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_22, [-1], True)
        sub_11: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_22, amax_11);  convert_element_type_22 = amax_11 = None
        exp_11: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_11);  sub_11 = None
        sum_12: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        convert_element_type_23: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_11, torch.bfloat16);  div_11 = None
        where_11: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_23, full_12, convert_element_type_23);  logical_not_23 = full_12 = convert_element_type_23 = None
        expand_46: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_11, _shape_param_512);  where_11 = _shape_param_512 = None
        view_453: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_46, _shape_param_513);  expand_46 = _shape_param_513 = None
        view_454: "bf16[32768, 512]" = torch.ops.aten.view.default(add_156, _shape_param_514);  _shape_param_514 = None
        permute_227: "bf16[512, 128]" = torch.ops.aten.permute.default(arg523_1, [1, 0]);  arg523_1 = None
        addmm_169: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg524_1, view_454, permute_227);  arg524_1 = view_454 = permute_227 = None
        view_455: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_169, _shape_param_515);  addmm_169 = _shape_param_515 = None
        view_456: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_455, _shape_param_516);  view_455 = _shape_param_516 = None
        permute_228: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_456, [0, 2, 1, 3]);  view_456 = None
        expand_47: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_228, _shape_param_517);  permute_228 = _shape_param_517 = None
        clone_46: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_457: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_46, _shape_param_518);  clone_46 = _shape_param_518 = None
        bmm_23: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_453, view_457);  view_453 = view_457 = None
        view_458: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_23, _shape_param_519);  bmm_23 = _shape_param_519 = None
        permute_229: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_458, [0, 2, 1, 3]);  view_458 = None
        clone_47: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_229, memory_format = torch.contiguous_format);  permute_229 = None
        view_459: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_47, _shape_param_520);  clone_47 = _shape_param_520 = None
        view_460: "bf16[32768, 128]" = torch.ops.aten.view.default(view_459, _shape_param_521);  view_459 = _shape_param_521 = None
        permute_230: "bf16[128, 128]" = torch.ops.aten.permute.default(arg525_1, [1, 0]);  arg525_1 = None
        addmm_170: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg526_1, view_460, permute_230);  arg526_1 = view_460 = permute_230 = None
        view_461: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_170, _shape_param_522);  addmm_170 = _shape_param_522 = None
        view_462: "bf16[32768, 512]" = torch.ops.aten.view.default(add_156, _shape_param_523);  _shape_param_523 = None
        permute_231: "bf16[512, 128]" = torch.ops.aten.permute.default(arg527_1, [1, 0]);  arg527_1 = None
        addmm_171: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg528_1, view_462, permute_231);  arg528_1 = view_462 = permute_231 = None
        view_463: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_171, _shape_param_524);  addmm_171 = _shape_param_524 = None
        mul_114: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_463, arg529_1);  view_463 = arg529_1 = None
        add_158: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_114, arg530_1);  mul_114 = arg530_1 = None
        add_159: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_461, add_158);  view_461 = add_158 = None
        mul_115: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_159, arg531_1);  add_159 = arg531_1 = None
        add_160: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_115, arg532_1);  mul_115 = arg532_1 = None
        view_464: "bf16[32768, 128]" = torch.ops.aten.view.default(add_160, _shape_param_525);  _shape_param_525 = None
        permute_232: "bf16[128, 512]" = torch.ops.aten.permute.default(arg533_1, [1, 0]);  arg533_1 = None
        addmm_172: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg534_1, view_464, permute_232);  arg534_1 = view_464 = permute_232 = None
        view_465: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_172, _shape_param_526);  addmm_172 = _shape_param_526 = None
        relu_44: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_465);  view_465 = None
        view_466: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_44, _shape_param_527);  relu_44 = _shape_param_527 = None
        permute_233: "bf16[512, 128]" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
        addmm_173: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg536_1, view_466, permute_233);  arg536_1 = view_466 = permute_233 = None
        view_467: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_173, _shape_param_528);  addmm_173 = _shape_param_528 = None
        add_161: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_467, add_160);  view_467 = add_160 = None
        mul_116: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_161, arg537_1);  add_161 = arg537_1 = None
        add_162: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_116, arg538_1);  mul_116 = arg538_1 = None
        view_468: "bf16[32768, 128]" = torch.ops.aten.view.default(add_162, _shape_param_529);  _shape_param_529 = None
        permute_234: "bf16[128, 512]" = torch.ops.aten.permute.default(arg539_1, [1, 0]);  arg539_1 = None
        addmm_174: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg540_1, view_468, permute_234);  arg540_1 = view_468 = permute_234 = None
        view_469: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_174, _shape_param_530);  addmm_174 = _shape_param_530 = None
        relu_45: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_469);  view_469 = None
        view_470: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_45, _shape_param_531);  relu_45 = _shape_param_531 = None
        permute_235: "bf16[512, 128]" = torch.ops.aten.permute.default(arg541_1, [1, 0]);  arg541_1 = None
        addmm_175: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg542_1, view_470, permute_235);  arg542_1 = view_470 = permute_235 = None
        view_471: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_175, _shape_param_532);  addmm_175 = _shape_param_532 = None
        add_163: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_471, add_162);  view_471 = add_162 = None
        mul_117: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_163, arg543_1);  add_163 = arg543_1 = None
        add_164: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_117, arg544_1);  mul_117 = arg544_1 = None
        view_472: "bf16[32768, 128]" = torch.ops.aten.view.default(add_164, _shape_param_533);  _shape_param_533 = None
        permute_236: "bf16[128, 512]" = torch.ops.aten.permute.default(arg545_1, [1, 0]);  arg545_1 = None
        addmm_176: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg546_1, view_472, permute_236);  arg546_1 = view_472 = permute_236 = None
        view_473: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_176, _shape_param_534);  addmm_176 = _shape_param_534 = None
        relu_46: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_473);  view_473 = None
        view_474: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_46, _shape_param_535);  relu_46 = _shape_param_535 = None
        permute_237: "bf16[512, 128]" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
        addmm_177: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg548_1, view_474, permute_237);  arg548_1 = view_474 = permute_237 = None
        view_475: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_177, _shape_param_536);  addmm_177 = _shape_param_536 = None
        add_165: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_475, add_164);  view_475 = add_164 = None
        mul_118: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_165, arg549_1);  add_165 = arg549_1 = None
        add_166: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_118, arg550_1);  mul_118 = arg550_1 = None
        view_476: "bf16[32768, 128]" = torch.ops.aten.view.default(add_166, _shape_param_537);  _shape_param_537 = None
        permute_238: "bf16[128, 512]" = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
        addmm_178: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg552_1, view_476, permute_238);  arg552_1 = view_476 = permute_238 = None
        view_477: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_178, _shape_param_538);  addmm_178 = _shape_param_538 = None
        relu_47: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_477);  view_477 = None
        view_478: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_47, _shape_param_539);  relu_47 = _shape_param_539 = None
        permute_239: "bf16[512, 128]" = torch.ops.aten.permute.default(arg553_1, [1, 0]);  arg553_1 = None
        addmm_179: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg554_1, view_478, permute_239);  arg554_1 = view_478 = permute_239 = None
        view_479: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_179, _shape_param_540);  addmm_179 = _shape_param_540 = None
        add_167: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_479, add_166);  view_479 = add_166 = None
        mul_119: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_167, arg555_1);  add_167 = arg555_1 = None
        add_168: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_119, arg556_1);  mul_119 = arg556_1 = None
        view_480: "bf16[32768, 128]" = torch.ops.aten.view.default(add_168, _shape_param_541);  add_168 = _shape_param_541 = None
        permute_240: "bf16[128, 512]" = torch.ops.aten.permute.default(arg557_1, [1, 0]);  arg557_1 = None
        addmm_180: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg558_1, view_480, permute_240);  arg558_1 = view_480 = permute_240 = None
        view_481: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_180, _shape_param_542);  addmm_180 = _shape_param_542 = None
        add_169: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_481, add_156);  view_481 = add_156 = None
        mul_120: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_169, arg559_1);  add_169 = arg559_1 = None
        add_170: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_120, arg560_1);  mul_120 = arg560_1 = None
        view_482: "bf16[32768, 512]" = torch.ops.aten.view.default(add_170, _shape_param_543);  _shape_param_543 = None
        permute_241: "bf16[512, 128]" = torch.ops.aten.permute.default(arg561_1, [1, 0]);  arg561_1 = None
        addmm_181: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg562_1, view_482, permute_241);  arg562_1 = view_482 = permute_241 = None
        view_483: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_181, _shape_param_544);  addmm_181 = _shape_param_544 = None
        mul_121: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_483, arg563_1);  view_483 = arg563_1 = None
        add_171: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_121, arg564_1);  mul_121 = arg564_1 = None
        view_484: "bf16[32768, 128]" = torch.ops.aten.view.default(add_171, _shape_param_545);  _shape_param_545 = None
        permute_242: "bf16[128, 128]" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
        addmm_182: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg566_1, view_484, permute_242);  arg566_1 = view_484 = permute_242 = None
        view_485: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_182, _shape_param_546);  addmm_182 = _shape_param_546 = None
        view_486: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_485, _shape_param_547);  view_485 = _shape_param_547 = None
        permute_243: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_486, [0, 2, 1, 3]);  view_486 = None
        mul_122: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_243, 0.4204482076268573);  permute_243 = None
        expand_48: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_122, _shape_param_548);  mul_122 = _shape_param_548 = None
        clone_48: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_487: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_48, _shape_param_549);  clone_48 = _shape_param_549 = None
        view_488: "bf16[32768, 128]" = torch.ops.aten.view.default(add_171, _shape_param_550);  add_171 = _shape_param_550 = None
        permute_244: "bf16[128, 128]" = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
        addmm_183: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg568_1, view_488, permute_244);  arg568_1 = view_488 = permute_244 = None
        view_489: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_183, _shape_param_551);  addmm_183 = _shape_param_551 = None
        view_490: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_489, _shape_param_552);  view_489 = _shape_param_552 = None
        permute_245: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_490, [0, 2, 1, 3]);  view_490 = None
        permute_246: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_245, [0, 1, 3, 2]);  permute_245 = None
        mul_123: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_246, 0.4204482076268573);  permute_246 = None
        expand_49: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_123, _shape_param_553);  mul_123 = _shape_param_553 = None
        clone_49: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_49, memory_format = torch.contiguous_format);  expand_49 = None
        view_491: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_49, _shape_param_554);  clone_49 = _shape_param_554 = None
        bmm_24: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_487, view_491);  view_487 = view_491 = None
        view_492: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_24, _shape_param_555);  bmm_24 = _shape_param_555 = None
        eq_12: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_492, -inf)
        logical_not_24: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        full_13: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_556, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_556 = None
        convert_element_type_24: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_492, torch.float32);  view_492 = None
        amax_12: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_24, [-1], True)
        sub_12: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_24, amax_12);  convert_element_type_24 = amax_12 = None
        exp_12: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_12);  sub_12 = None
        sum_13: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        convert_element_type_25: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_12, torch.bfloat16);  div_12 = None
        where_12: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_25, full_13, convert_element_type_25);  logical_not_25 = full_13 = convert_element_type_25 = None
        expand_50: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_12, _shape_param_557);  where_12 = _shape_param_557 = None
        view_493: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_50, _shape_param_558);  expand_50 = _shape_param_558 = None
        view_494: "bf16[32768, 512]" = torch.ops.aten.view.default(add_170, _shape_param_559);  _shape_param_559 = None
        permute_247: "bf16[512, 128]" = torch.ops.aten.permute.default(arg569_1, [1, 0]);  arg569_1 = None
        addmm_184: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg570_1, view_494, permute_247);  arg570_1 = view_494 = permute_247 = None
        view_495: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_184, _shape_param_560);  addmm_184 = _shape_param_560 = None
        view_496: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_495, _shape_param_561);  view_495 = _shape_param_561 = None
        permute_248: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_496, [0, 2, 1, 3]);  view_496 = None
        expand_51: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_248, _shape_param_562);  permute_248 = _shape_param_562 = None
        clone_50: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_51, memory_format = torch.contiguous_format);  expand_51 = None
        view_497: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_50, _shape_param_563);  clone_50 = _shape_param_563 = None
        bmm_25: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_493, view_497);  view_493 = view_497 = None
        view_498: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_25, _shape_param_564);  bmm_25 = _shape_param_564 = None
        permute_249: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_498, [0, 2, 1, 3]);  view_498 = None
        clone_51: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_249, memory_format = torch.contiguous_format);  permute_249 = None
        view_499: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_51, _shape_param_565);  clone_51 = _shape_param_565 = None
        view_500: "bf16[32768, 128]" = torch.ops.aten.view.default(view_499, _shape_param_566);  view_499 = _shape_param_566 = None
        permute_250: "bf16[128, 128]" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
        addmm_185: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg572_1, view_500, permute_250);  arg572_1 = view_500 = permute_250 = None
        view_501: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_185, _shape_param_567);  addmm_185 = _shape_param_567 = None
        view_502: "bf16[32768, 512]" = torch.ops.aten.view.default(add_170, _shape_param_568);  _shape_param_568 = None
        permute_251: "bf16[512, 128]" = torch.ops.aten.permute.default(arg573_1, [1, 0]);  arg573_1 = None
        addmm_186: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg574_1, view_502, permute_251);  arg574_1 = view_502 = permute_251 = None
        view_503: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_186, _shape_param_569);  addmm_186 = _shape_param_569 = None
        mul_124: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_503, arg575_1);  view_503 = arg575_1 = None
        add_172: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_124, arg576_1);  mul_124 = arg576_1 = None
        add_173: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_501, add_172);  view_501 = add_172 = None
        mul_125: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_173, arg577_1);  add_173 = arg577_1 = None
        add_174: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_125, arg578_1);  mul_125 = arg578_1 = None
        view_504: "bf16[32768, 128]" = torch.ops.aten.view.default(add_174, _shape_param_570);  _shape_param_570 = None
        permute_252: "bf16[128, 512]" = torch.ops.aten.permute.default(arg579_1, [1, 0]);  arg579_1 = None
        addmm_187: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg580_1, view_504, permute_252);  arg580_1 = view_504 = permute_252 = None
        view_505: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_187, _shape_param_571);  addmm_187 = _shape_param_571 = None
        relu_48: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_505);  view_505 = None
        view_506: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_48, _shape_param_572);  relu_48 = _shape_param_572 = None
        permute_253: "bf16[512, 128]" = torch.ops.aten.permute.default(arg581_1, [1, 0]);  arg581_1 = None
        addmm_188: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg582_1, view_506, permute_253);  arg582_1 = view_506 = permute_253 = None
        view_507: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_188, _shape_param_573);  addmm_188 = _shape_param_573 = None
        add_175: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_507, add_174);  view_507 = add_174 = None
        mul_126: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_175, arg583_1);  add_175 = arg583_1 = None
        add_176: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_126, arg584_1);  mul_126 = arg584_1 = None
        view_508: "bf16[32768, 128]" = torch.ops.aten.view.default(add_176, _shape_param_574);  _shape_param_574 = None
        permute_254: "bf16[128, 512]" = torch.ops.aten.permute.default(arg585_1, [1, 0]);  arg585_1 = None
        addmm_189: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg586_1, view_508, permute_254);  arg586_1 = view_508 = permute_254 = None
        view_509: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_189, _shape_param_575);  addmm_189 = _shape_param_575 = None
        relu_49: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_509);  view_509 = None
        view_510: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_49, _shape_param_576);  relu_49 = _shape_param_576 = None
        permute_255: "bf16[512, 128]" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
        addmm_190: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg588_1, view_510, permute_255);  arg588_1 = view_510 = permute_255 = None
        view_511: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_190, _shape_param_577);  addmm_190 = _shape_param_577 = None
        add_177: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_511, add_176);  view_511 = add_176 = None
        mul_127: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_177, arg589_1);  add_177 = arg589_1 = None
        add_178: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_127, arg590_1);  mul_127 = arg590_1 = None
        view_512: "bf16[32768, 128]" = torch.ops.aten.view.default(add_178, _shape_param_578);  _shape_param_578 = None
        permute_256: "bf16[128, 512]" = torch.ops.aten.permute.default(arg591_1, [1, 0]);  arg591_1 = None
        addmm_191: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg592_1, view_512, permute_256);  arg592_1 = view_512 = permute_256 = None
        view_513: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_191, _shape_param_579);  addmm_191 = _shape_param_579 = None
        relu_50: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_513);  view_513 = None
        view_514: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_50, _shape_param_580);  relu_50 = _shape_param_580 = None
        permute_257: "bf16[512, 128]" = torch.ops.aten.permute.default(arg593_1, [1, 0]);  arg593_1 = None
        addmm_192: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg594_1, view_514, permute_257);  arg594_1 = view_514 = permute_257 = None
        view_515: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_192, _shape_param_581);  addmm_192 = _shape_param_581 = None
        add_179: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_515, add_178);  view_515 = add_178 = None
        mul_128: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_179, arg595_1);  add_179 = arg595_1 = None
        add_180: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_128, arg596_1);  mul_128 = arg596_1 = None
        view_516: "bf16[32768, 128]" = torch.ops.aten.view.default(add_180, _shape_param_582);  _shape_param_582 = None
        permute_258: "bf16[128, 512]" = torch.ops.aten.permute.default(arg597_1, [1, 0]);  arg597_1 = None
        addmm_193: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg598_1, view_516, permute_258);  arg598_1 = view_516 = permute_258 = None
        view_517: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_193, _shape_param_583);  addmm_193 = _shape_param_583 = None
        relu_51: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_517);  view_517 = None
        view_518: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_51, _shape_param_584);  relu_51 = _shape_param_584 = None
        permute_259: "bf16[512, 128]" = torch.ops.aten.permute.default(arg599_1, [1, 0]);  arg599_1 = None
        addmm_194: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg600_1, view_518, permute_259);  arg600_1 = view_518 = permute_259 = None
        view_519: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_194, _shape_param_585);  addmm_194 = _shape_param_585 = None
        add_181: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_519, add_180);  view_519 = add_180 = None
        mul_129: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_181, arg601_1);  add_181 = arg601_1 = None
        add_182: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_129, arg602_1);  mul_129 = arg602_1 = None
        view_520: "bf16[32768, 128]" = torch.ops.aten.view.default(add_182, _shape_param_586);  add_182 = _shape_param_586 = None
        permute_260: "bf16[128, 512]" = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
        addmm_195: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg604_1, view_520, permute_260);  arg604_1 = view_520 = permute_260 = None
        view_521: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_195, _shape_param_587);  addmm_195 = _shape_param_587 = None
        add_183: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_521, add_170);  view_521 = add_170 = None
        mul_130: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_183, arg605_1);  add_183 = arg605_1 = None
        add_184: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_130, arg606_1);  mul_130 = arg606_1 = None
        view_522: "bf16[32768, 512]" = torch.ops.aten.view.default(add_184, _shape_param_588);  _shape_param_588 = None
        permute_261: "bf16[512, 128]" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
        addmm_196: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg608_1, view_522, permute_261);  arg608_1 = view_522 = permute_261 = None
        view_523: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_196, _shape_param_589);  addmm_196 = _shape_param_589 = None
        mul_131: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_523, arg609_1);  view_523 = arg609_1 = None
        add_185: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_131, arg610_1);  mul_131 = arg610_1 = None
        view_524: "bf16[32768, 128]" = torch.ops.aten.view.default(add_185, _shape_param_590);  _shape_param_590 = None
        permute_262: "bf16[128, 128]" = torch.ops.aten.permute.default(arg611_1, [1, 0]);  arg611_1 = None
        addmm_197: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg612_1, view_524, permute_262);  arg612_1 = view_524 = permute_262 = None
        view_525: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_197, _shape_param_591);  addmm_197 = _shape_param_591 = None
        view_526: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_525, _shape_param_592);  view_525 = _shape_param_592 = None
        permute_263: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_526, [0, 2, 1, 3]);  view_526 = None
        mul_132: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_263, 0.4204482076268573);  permute_263 = None
        expand_52: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_132, _shape_param_593);  mul_132 = _shape_param_593 = None
        clone_52: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_527: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_52, _shape_param_594);  clone_52 = _shape_param_594 = None
        view_528: "bf16[32768, 128]" = torch.ops.aten.view.default(add_185, _shape_param_595);  add_185 = _shape_param_595 = None
        permute_264: "bf16[128, 128]" = torch.ops.aten.permute.default(arg613_1, [1, 0]);  arg613_1 = None
        addmm_198: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg614_1, view_528, permute_264);  arg614_1 = view_528 = permute_264 = None
        view_529: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_198, _shape_param_596);  addmm_198 = _shape_param_596 = None
        view_530: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_529, _shape_param_597);  view_529 = _shape_param_597 = None
        permute_265: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_530, [0, 2, 1, 3]);  view_530 = None
        permute_266: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_265, [0, 1, 3, 2]);  permute_265 = None
        mul_133: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_266, 0.4204482076268573);  permute_266 = None
        expand_53: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_133, _shape_param_598);  mul_133 = _shape_param_598 = None
        clone_53: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_531: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_53, _shape_param_599);  clone_53 = _shape_param_599 = None
        bmm_26: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_527, view_531);  view_527 = view_531 = None
        view_532: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_26, _shape_param_600);  bmm_26 = _shape_param_600 = None
        eq_13: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_532, -inf)
        logical_not_26: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        full_14: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_601, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_601 = None
        convert_element_type_26: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_532, torch.float32);  view_532 = None
        amax_13: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_26, [-1], True)
        sub_13: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_26, amax_13);  convert_element_type_26 = amax_13 = None
        exp_13: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_14: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        convert_element_type_27: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_13, torch.bfloat16);  div_13 = None
        where_13: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_27, full_14, convert_element_type_27);  logical_not_27 = full_14 = convert_element_type_27 = None
        expand_54: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_13, _shape_param_602);  where_13 = _shape_param_602 = None
        view_533: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_54, _shape_param_603);  expand_54 = _shape_param_603 = None
        view_534: "bf16[32768, 512]" = torch.ops.aten.view.default(add_184, _shape_param_604);  _shape_param_604 = None
        permute_267: "bf16[512, 128]" = torch.ops.aten.permute.default(arg615_1, [1, 0]);  arg615_1 = None
        addmm_199: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg616_1, view_534, permute_267);  arg616_1 = view_534 = permute_267 = None
        view_535: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_199, _shape_param_605);  addmm_199 = _shape_param_605 = None
        view_536: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_535, _shape_param_606);  view_535 = _shape_param_606 = None
        permute_268: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_536, [0, 2, 1, 3]);  view_536 = None
        expand_55: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_268, _shape_param_607);  permute_268 = _shape_param_607 = None
        clone_54: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_537: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_54, _shape_param_608);  clone_54 = _shape_param_608 = None
        bmm_27: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_533, view_537);  view_533 = view_537 = None
        view_538: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_27, _shape_param_609);  bmm_27 = _shape_param_609 = None
        permute_269: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_538, [0, 2, 1, 3]);  view_538 = None
        clone_55: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_269, memory_format = torch.contiguous_format);  permute_269 = None
        view_539: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_55, _shape_param_610);  clone_55 = _shape_param_610 = None
        view_540: "bf16[32768, 128]" = torch.ops.aten.view.default(view_539, _shape_param_611);  view_539 = _shape_param_611 = None
        permute_270: "bf16[128, 128]" = torch.ops.aten.permute.default(arg617_1, [1, 0]);  arg617_1 = None
        addmm_200: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg618_1, view_540, permute_270);  arg618_1 = view_540 = permute_270 = None
        view_541: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_200, _shape_param_612);  addmm_200 = _shape_param_612 = None
        view_542: "bf16[32768, 512]" = torch.ops.aten.view.default(add_184, _shape_param_613);  _shape_param_613 = None
        permute_271: "bf16[512, 128]" = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        addmm_201: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg620_1, view_542, permute_271);  arg620_1 = view_542 = permute_271 = None
        view_543: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_201, _shape_param_614);  addmm_201 = _shape_param_614 = None
        mul_134: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_543, arg621_1);  view_543 = arg621_1 = None
        add_186: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_134, arg622_1);  mul_134 = arg622_1 = None
        add_187: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_541, add_186);  view_541 = add_186 = None
        mul_135: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_187, arg623_1);  add_187 = arg623_1 = None
        add_188: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_135, arg624_1);  mul_135 = arg624_1 = None
        view_544: "bf16[32768, 128]" = torch.ops.aten.view.default(add_188, _shape_param_615);  _shape_param_615 = None
        permute_272: "bf16[128, 512]" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
        addmm_202: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg626_1, view_544, permute_272);  arg626_1 = view_544 = permute_272 = None
        view_545: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_202, _shape_param_616);  addmm_202 = _shape_param_616 = None
        relu_52: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_545);  view_545 = None
        view_546: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_52, _shape_param_617);  relu_52 = _shape_param_617 = None
        permute_273: "bf16[512, 128]" = torch.ops.aten.permute.default(arg627_1, [1, 0]);  arg627_1 = None
        addmm_203: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg628_1, view_546, permute_273);  arg628_1 = view_546 = permute_273 = None
        view_547: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_203, _shape_param_618);  addmm_203 = _shape_param_618 = None
        add_189: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_547, add_188);  view_547 = add_188 = None
        mul_136: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_189, arg629_1);  add_189 = arg629_1 = None
        add_190: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_136, arg630_1);  mul_136 = arg630_1 = None
        view_548: "bf16[32768, 128]" = torch.ops.aten.view.default(add_190, _shape_param_619);  _shape_param_619 = None
        permute_274: "bf16[128, 512]" = torch.ops.aten.permute.default(arg631_1, [1, 0]);  arg631_1 = None
        addmm_204: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg632_1, view_548, permute_274);  arg632_1 = view_548 = permute_274 = None
        view_549: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_204, _shape_param_620);  addmm_204 = _shape_param_620 = None
        relu_53: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_549);  view_549 = None
        view_550: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_53, _shape_param_621);  relu_53 = _shape_param_621 = None
        permute_275: "bf16[512, 128]" = torch.ops.aten.permute.default(arg633_1, [1, 0]);  arg633_1 = None
        addmm_205: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg634_1, view_550, permute_275);  arg634_1 = view_550 = permute_275 = None
        view_551: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_205, _shape_param_622);  addmm_205 = _shape_param_622 = None
        add_191: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_551, add_190);  view_551 = add_190 = None
        mul_137: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_191, arg635_1);  add_191 = arg635_1 = None
        add_192: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_137, arg636_1);  mul_137 = arg636_1 = None
        view_552: "bf16[32768, 128]" = torch.ops.aten.view.default(add_192, _shape_param_623);  _shape_param_623 = None
        permute_276: "bf16[128, 512]" = torch.ops.aten.permute.default(arg637_1, [1, 0]);  arg637_1 = None
        addmm_206: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg638_1, view_552, permute_276);  arg638_1 = view_552 = permute_276 = None
        view_553: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_206, _shape_param_624);  addmm_206 = _shape_param_624 = None
        relu_54: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_553);  view_553 = None
        view_554: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_54, _shape_param_625);  relu_54 = _shape_param_625 = None
        permute_277: "bf16[512, 128]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
        addmm_207: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg640_1, view_554, permute_277);  arg640_1 = view_554 = permute_277 = None
        view_555: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_207, _shape_param_626);  addmm_207 = _shape_param_626 = None
        add_193: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_555, add_192);  view_555 = add_192 = None
        mul_138: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_193, arg641_1);  add_193 = arg641_1 = None
        add_194: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_138, arg642_1);  mul_138 = arg642_1 = None
        view_556: "bf16[32768, 128]" = torch.ops.aten.view.default(add_194, _shape_param_627);  _shape_param_627 = None
        permute_278: "bf16[128, 512]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        addmm_208: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg644_1, view_556, permute_278);  arg644_1 = view_556 = permute_278 = None
        view_557: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_208, _shape_param_628);  addmm_208 = _shape_param_628 = None
        relu_55: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_557);  view_557 = None
        view_558: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_55, _shape_param_629);  relu_55 = _shape_param_629 = None
        permute_279: "bf16[512, 128]" = torch.ops.aten.permute.default(arg645_1, [1, 0]);  arg645_1 = None
        addmm_209: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg646_1, view_558, permute_279);  arg646_1 = view_558 = permute_279 = None
        view_559: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_209, _shape_param_630);  addmm_209 = _shape_param_630 = None
        add_195: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_559, add_194);  view_559 = add_194 = None
        mul_139: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_195, arg647_1);  add_195 = arg647_1 = None
        add_196: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_139, arg648_1);  mul_139 = arg648_1 = None
        view_560: "bf16[32768, 128]" = torch.ops.aten.view.default(add_196, _shape_param_631);  add_196 = _shape_param_631 = None
        permute_280: "bf16[128, 512]" = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
        addmm_210: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg650_1, view_560, permute_280);  arg650_1 = view_560 = permute_280 = None
        view_561: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_210, _shape_param_632);  addmm_210 = _shape_param_632 = None
        add_197: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_561, add_184);  view_561 = add_184 = None
        mul_140: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_197, arg651_1);  add_197 = arg651_1 = None
        add_198: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_140, arg652_1);  mul_140 = arg652_1 = None
        view_562: "bf16[32768, 512]" = torch.ops.aten.view.default(add_198, _shape_param_633);  _shape_param_633 = None
        permute_281: "bf16[512, 128]" = torch.ops.aten.permute.default(arg653_1, [1, 0]);  arg653_1 = None
        addmm_211: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg654_1, view_562, permute_281);  arg654_1 = view_562 = permute_281 = None
        view_563: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_211, _shape_param_634);  addmm_211 = _shape_param_634 = None
        mul_141: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_563, arg655_1);  view_563 = arg655_1 = None
        add_199: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_141, arg656_1);  mul_141 = arg656_1 = None
        view_564: "bf16[32768, 128]" = torch.ops.aten.view.default(add_199, _shape_param_635);  _shape_param_635 = None
        permute_282: "bf16[128, 128]" = torch.ops.aten.permute.default(arg657_1, [1, 0]);  arg657_1 = None
        addmm_212: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg658_1, view_564, permute_282);  arg658_1 = view_564 = permute_282 = None
        view_565: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_212, _shape_param_636);  addmm_212 = _shape_param_636 = None
        view_566: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_565, _shape_param_637);  view_565 = _shape_param_637 = None
        permute_283: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_566, [0, 2, 1, 3]);  view_566 = None
        mul_142: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_283, 0.4204482076268573);  permute_283 = None
        expand_56: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_142, _shape_param_638);  mul_142 = _shape_param_638 = None
        clone_56: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_56, memory_format = torch.contiguous_format);  expand_56 = None
        view_567: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_56, _shape_param_639);  clone_56 = _shape_param_639 = None
        view_568: "bf16[32768, 128]" = torch.ops.aten.view.default(add_199, _shape_param_640);  add_199 = _shape_param_640 = None
        permute_284: "bf16[128, 128]" = torch.ops.aten.permute.default(arg659_1, [1, 0]);  arg659_1 = None
        addmm_213: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg660_1, view_568, permute_284);  arg660_1 = view_568 = permute_284 = None
        view_569: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_213, _shape_param_641);  addmm_213 = _shape_param_641 = None
        view_570: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_569, _shape_param_642);  view_569 = _shape_param_642 = None
        permute_285: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_570, [0, 2, 1, 3]);  view_570 = None
        permute_286: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_285, [0, 1, 3, 2]);  permute_285 = None
        mul_143: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_286, 0.4204482076268573);  permute_286 = None
        expand_57: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_143, _shape_param_643);  mul_143 = _shape_param_643 = None
        clone_57: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_571: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_57, _shape_param_644);  clone_57 = _shape_param_644 = None
        bmm_28: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_567, view_571);  view_567 = view_571 = None
        view_572: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_28, _shape_param_645);  bmm_28 = _shape_param_645 = None
        eq_14: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_572, -inf)
        logical_not_28: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        full_15: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_646, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_646 = None
        convert_element_type_28: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_572, torch.float32);  view_572 = None
        amax_14: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_28, [-1], True)
        sub_14: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_28, amax_14);  convert_element_type_28 = amax_14 = None
        exp_14: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_14);  sub_14 = None
        sum_15: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        convert_element_type_29: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_14, torch.bfloat16);  div_14 = None
        where_14: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_29, full_15, convert_element_type_29);  logical_not_29 = full_15 = convert_element_type_29 = None
        expand_58: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_14, _shape_param_647);  where_14 = _shape_param_647 = None
        view_573: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_58, _shape_param_648);  expand_58 = _shape_param_648 = None
        view_574: "bf16[32768, 512]" = torch.ops.aten.view.default(add_198, _shape_param_649);  _shape_param_649 = None
        permute_287: "bf16[512, 128]" = torch.ops.aten.permute.default(arg661_1, [1, 0]);  arg661_1 = None
        addmm_214: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg662_1, view_574, permute_287);  arg662_1 = view_574 = permute_287 = None
        view_575: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_214, _shape_param_650);  addmm_214 = _shape_param_650 = None
        view_576: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_575, _shape_param_651);  view_575 = _shape_param_651 = None
        permute_288: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_576, [0, 2, 1, 3]);  view_576 = None
        expand_59: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_288, _shape_param_652);  permute_288 = _shape_param_652 = None
        clone_58: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_59, memory_format = torch.contiguous_format);  expand_59 = None
        view_577: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_58, _shape_param_653);  clone_58 = _shape_param_653 = None
        bmm_29: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_573, view_577);  view_573 = view_577 = None
        view_578: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_29, _shape_param_654);  bmm_29 = _shape_param_654 = None
        permute_289: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_578, [0, 2, 1, 3]);  view_578 = None
        clone_59: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_289, memory_format = torch.contiguous_format);  permute_289 = None
        view_579: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_59, _shape_param_655);  clone_59 = _shape_param_655 = None
        view_580: "bf16[32768, 128]" = torch.ops.aten.view.default(view_579, _shape_param_656);  view_579 = _shape_param_656 = None
        permute_290: "bf16[128, 128]" = torch.ops.aten.permute.default(arg663_1, [1, 0]);  arg663_1 = None
        addmm_215: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg664_1, view_580, permute_290);  arg664_1 = view_580 = permute_290 = None
        view_581: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_215, _shape_param_657);  addmm_215 = _shape_param_657 = None
        view_582: "bf16[32768, 512]" = torch.ops.aten.view.default(add_198, _shape_param_658);  _shape_param_658 = None
        permute_291: "bf16[512, 128]" = torch.ops.aten.permute.default(arg665_1, [1, 0]);  arg665_1 = None
        addmm_216: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg666_1, view_582, permute_291);  arg666_1 = view_582 = permute_291 = None
        view_583: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_216, _shape_param_659);  addmm_216 = _shape_param_659 = None
        mul_144: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_583, arg667_1);  view_583 = arg667_1 = None
        add_200: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_144, arg668_1);  mul_144 = arg668_1 = None
        add_201: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_581, add_200);  view_581 = add_200 = None
        mul_145: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_201, arg669_1);  add_201 = arg669_1 = None
        add_202: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_145, arg670_1);  mul_145 = arg670_1 = None
        view_584: "bf16[32768, 128]" = torch.ops.aten.view.default(add_202, _shape_param_660);  _shape_param_660 = None
        permute_292: "bf16[128, 512]" = torch.ops.aten.permute.default(arg671_1, [1, 0]);  arg671_1 = None
        addmm_217: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg672_1, view_584, permute_292);  arg672_1 = view_584 = permute_292 = None
        view_585: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_217, _shape_param_661);  addmm_217 = _shape_param_661 = None
        relu_56: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_585);  view_585 = None
        view_586: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_56, _shape_param_662);  relu_56 = _shape_param_662 = None
        permute_293: "bf16[512, 128]" = torch.ops.aten.permute.default(arg673_1, [1, 0]);  arg673_1 = None
        addmm_218: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg674_1, view_586, permute_293);  arg674_1 = view_586 = permute_293 = None
        view_587: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_218, _shape_param_663);  addmm_218 = _shape_param_663 = None
        add_203: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_587, add_202);  view_587 = add_202 = None
        mul_146: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_203, arg675_1);  add_203 = arg675_1 = None
        add_204: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_146, arg676_1);  mul_146 = arg676_1 = None
        view_588: "bf16[32768, 128]" = torch.ops.aten.view.default(add_204, _shape_param_664);  _shape_param_664 = None
        permute_294: "bf16[128, 512]" = torch.ops.aten.permute.default(arg677_1, [1, 0]);  arg677_1 = None
        addmm_219: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg678_1, view_588, permute_294);  arg678_1 = view_588 = permute_294 = None
        view_589: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_219, _shape_param_665);  addmm_219 = _shape_param_665 = None
        relu_57: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_589);  view_589 = None
        view_590: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_57, _shape_param_666);  relu_57 = _shape_param_666 = None
        permute_295: "bf16[512, 128]" = torch.ops.aten.permute.default(arg679_1, [1, 0]);  arg679_1 = None
        addmm_220: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg680_1, view_590, permute_295);  arg680_1 = view_590 = permute_295 = None
        view_591: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_220, _shape_param_667);  addmm_220 = _shape_param_667 = None
        add_205: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_591, add_204);  view_591 = add_204 = None
        mul_147: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_205, arg681_1);  add_205 = arg681_1 = None
        add_206: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_147, arg682_1);  mul_147 = arg682_1 = None
        view_592: "bf16[32768, 128]" = torch.ops.aten.view.default(add_206, _shape_param_668);  _shape_param_668 = None
        permute_296: "bf16[128, 512]" = torch.ops.aten.permute.default(arg683_1, [1, 0]);  arg683_1 = None
        addmm_221: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg684_1, view_592, permute_296);  arg684_1 = view_592 = permute_296 = None
        view_593: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_221, _shape_param_669);  addmm_221 = _shape_param_669 = None
        relu_58: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_593);  view_593 = None
        view_594: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_58, _shape_param_670);  relu_58 = _shape_param_670 = None
        permute_297: "bf16[512, 128]" = torch.ops.aten.permute.default(arg685_1, [1, 0]);  arg685_1 = None
        addmm_222: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg686_1, view_594, permute_297);  arg686_1 = view_594 = permute_297 = None
        view_595: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_222, _shape_param_671);  addmm_222 = _shape_param_671 = None
        add_207: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_595, add_206);  view_595 = add_206 = None
        mul_148: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_207, arg687_1);  add_207 = arg687_1 = None
        add_208: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_148, arg688_1);  mul_148 = arg688_1 = None
        view_596: "bf16[32768, 128]" = torch.ops.aten.view.default(add_208, _shape_param_672);  _shape_param_672 = None
        permute_298: "bf16[128, 512]" = torch.ops.aten.permute.default(arg689_1, [1, 0]);  arg689_1 = None
        addmm_223: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg690_1, view_596, permute_298);  arg690_1 = view_596 = permute_298 = None
        view_597: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_223, _shape_param_673);  addmm_223 = _shape_param_673 = None
        relu_59: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_597);  view_597 = None
        view_598: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_59, _shape_param_674);  relu_59 = _shape_param_674 = None
        permute_299: "bf16[512, 128]" = torch.ops.aten.permute.default(arg691_1, [1, 0]);  arg691_1 = None
        addmm_224: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg692_1, view_598, permute_299);  arg692_1 = view_598 = permute_299 = None
        view_599: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_224, _shape_param_675);  addmm_224 = _shape_param_675 = None
        add_209: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_599, add_208);  view_599 = add_208 = None
        mul_149: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_209, arg693_1);  add_209 = arg693_1 = None
        add_210: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_149, arg694_1);  mul_149 = arg694_1 = None
        view_600: "bf16[32768, 128]" = torch.ops.aten.view.default(add_210, _shape_param_676);  add_210 = _shape_param_676 = None
        permute_300: "bf16[128, 512]" = torch.ops.aten.permute.default(arg695_1, [1, 0]);  arg695_1 = None
        addmm_225: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg696_1, view_600, permute_300);  arg696_1 = view_600 = permute_300 = None
        view_601: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_225, _shape_param_677);  addmm_225 = _shape_param_677 = None
        add_211: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_601, add_198);  view_601 = add_198 = None
        mul_150: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_211, arg697_1);  add_211 = arg697_1 = None
        add_212: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_150, arg698_1);  mul_150 = arg698_1 = None
        view_602: "bf16[32768, 512]" = torch.ops.aten.view.default(add_212, _shape_param_678);  _shape_param_678 = None
        permute_301: "bf16[512, 128]" = torch.ops.aten.permute.default(arg699_1, [1, 0]);  arg699_1 = None
        addmm_226: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg700_1, view_602, permute_301);  arg700_1 = view_602 = permute_301 = None
        view_603: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_226, _shape_param_679);  addmm_226 = _shape_param_679 = None
        mul_151: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_603, arg701_1);  view_603 = arg701_1 = None
        add_213: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_151, arg702_1);  mul_151 = arg702_1 = None
        view_604: "bf16[32768, 128]" = torch.ops.aten.view.default(add_213, _shape_param_680);  _shape_param_680 = None
        permute_302: "bf16[128, 128]" = torch.ops.aten.permute.default(arg703_1, [1, 0]);  arg703_1 = None
        addmm_227: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg704_1, view_604, permute_302);  arg704_1 = view_604 = permute_302 = None
        view_605: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_227, _shape_param_681);  addmm_227 = _shape_param_681 = None
        view_606: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_605, _shape_param_682);  view_605 = _shape_param_682 = None
        permute_303: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_606, [0, 2, 1, 3]);  view_606 = None
        mul_152: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_303, 0.4204482076268573);  permute_303 = None
        expand_60: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_152, _shape_param_683);  mul_152 = _shape_param_683 = None
        clone_60: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_607: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_60, _shape_param_684);  clone_60 = _shape_param_684 = None
        view_608: "bf16[32768, 128]" = torch.ops.aten.view.default(add_213, _shape_param_685);  add_213 = _shape_param_685 = None
        permute_304: "bf16[128, 128]" = torch.ops.aten.permute.default(arg705_1, [1, 0]);  arg705_1 = None
        addmm_228: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg706_1, view_608, permute_304);  arg706_1 = view_608 = permute_304 = None
        view_609: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_228, _shape_param_686);  addmm_228 = _shape_param_686 = None
        view_610: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_609, _shape_param_687);  view_609 = _shape_param_687 = None
        permute_305: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_610, [0, 2, 1, 3]);  view_610 = None
        permute_306: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_305, [0, 1, 3, 2]);  permute_305 = None
        mul_153: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_306, 0.4204482076268573);  permute_306 = None
        expand_61: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_153, _shape_param_688);  mul_153 = _shape_param_688 = None
        clone_61: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_61, memory_format = torch.contiguous_format);  expand_61 = None
        view_611: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_61, _shape_param_689);  clone_61 = _shape_param_689 = None
        bmm_30: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_607, view_611);  view_607 = view_611 = None
        view_612: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_30, _shape_param_690);  bmm_30 = _shape_param_690 = None
        eq_15: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_612, -inf)
        logical_not_30: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        full_16: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_691, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_691 = None
        convert_element_type_30: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_612, torch.float32);  view_612 = None
        amax_15: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_30, [-1], True)
        sub_15: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_30, amax_15);  convert_element_type_30 = amax_15 = None
        exp_15: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_15);  sub_15 = None
        sum_16: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        convert_element_type_31: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_15, torch.bfloat16);  div_15 = None
        where_15: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_31, full_16, convert_element_type_31);  logical_not_31 = full_16 = convert_element_type_31 = None
        expand_62: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_15, _shape_param_692);  where_15 = _shape_param_692 = None
        view_613: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_62, _shape_param_693);  expand_62 = _shape_param_693 = None
        view_614: "bf16[32768, 512]" = torch.ops.aten.view.default(add_212, _shape_param_694);  _shape_param_694 = None
        permute_307: "bf16[512, 128]" = torch.ops.aten.permute.default(arg707_1, [1, 0]);  arg707_1 = None
        addmm_229: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg708_1, view_614, permute_307);  arg708_1 = view_614 = permute_307 = None
        view_615: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_229, _shape_param_695);  addmm_229 = _shape_param_695 = None
        view_616: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_615, _shape_param_696);  view_615 = _shape_param_696 = None
        permute_308: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_616, [0, 2, 1, 3]);  view_616 = None
        expand_63: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_308, _shape_param_697);  permute_308 = _shape_param_697 = None
        clone_62: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_617: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_62, _shape_param_698);  clone_62 = _shape_param_698 = None
        bmm_31: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_613, view_617);  view_613 = view_617 = None
        view_618: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_31, _shape_param_699);  bmm_31 = _shape_param_699 = None
        permute_309: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_618, [0, 2, 1, 3]);  view_618 = None
        clone_63: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_309, memory_format = torch.contiguous_format);  permute_309 = None
        view_619: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_63, _shape_param_700);  clone_63 = _shape_param_700 = None
        view_620: "bf16[32768, 128]" = torch.ops.aten.view.default(view_619, _shape_param_701);  view_619 = _shape_param_701 = None
        permute_310: "bf16[128, 128]" = torch.ops.aten.permute.default(arg709_1, [1, 0]);  arg709_1 = None
        addmm_230: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg710_1, view_620, permute_310);  arg710_1 = view_620 = permute_310 = None
        view_621: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_230, _shape_param_702);  addmm_230 = _shape_param_702 = None
        view_622: "bf16[32768, 512]" = torch.ops.aten.view.default(add_212, _shape_param_703);  _shape_param_703 = None
        permute_311: "bf16[512, 128]" = torch.ops.aten.permute.default(arg711_1, [1, 0]);  arg711_1 = None
        addmm_231: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg712_1, view_622, permute_311);  arg712_1 = view_622 = permute_311 = None
        view_623: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_231, _shape_param_704);  addmm_231 = _shape_param_704 = None
        mul_154: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_623, arg713_1);  view_623 = arg713_1 = None
        add_214: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_154, arg714_1);  mul_154 = arg714_1 = None
        add_215: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_621, add_214);  view_621 = add_214 = None
        mul_155: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_215, arg715_1);  add_215 = arg715_1 = None
        add_216: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_155, arg716_1);  mul_155 = arg716_1 = None
        view_624: "bf16[32768, 128]" = torch.ops.aten.view.default(add_216, _shape_param_705);  _shape_param_705 = None
        permute_312: "bf16[128, 512]" = torch.ops.aten.permute.default(arg717_1, [1, 0]);  arg717_1 = None
        addmm_232: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg718_1, view_624, permute_312);  arg718_1 = view_624 = permute_312 = None
        view_625: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_232, _shape_param_706);  addmm_232 = _shape_param_706 = None
        relu_60: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_625);  view_625 = None
        view_626: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_60, _shape_param_707);  relu_60 = _shape_param_707 = None
        permute_313: "bf16[512, 128]" = torch.ops.aten.permute.default(arg719_1, [1, 0]);  arg719_1 = None
        addmm_233: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg720_1, view_626, permute_313);  arg720_1 = view_626 = permute_313 = None
        view_627: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_233, _shape_param_708);  addmm_233 = _shape_param_708 = None
        add_217: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_627, add_216);  view_627 = add_216 = None
        mul_156: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_217, arg721_1);  add_217 = arg721_1 = None
        add_218: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_156, arg722_1);  mul_156 = arg722_1 = None
        view_628: "bf16[32768, 128]" = torch.ops.aten.view.default(add_218, _shape_param_709);  _shape_param_709 = None
        permute_314: "bf16[128, 512]" = torch.ops.aten.permute.default(arg723_1, [1, 0]);  arg723_1 = None
        addmm_234: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg724_1, view_628, permute_314);  arg724_1 = view_628 = permute_314 = None
        view_629: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_234, _shape_param_710);  addmm_234 = _shape_param_710 = None
        relu_61: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_629);  view_629 = None
        view_630: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_61, _shape_param_711);  relu_61 = _shape_param_711 = None
        permute_315: "bf16[512, 128]" = torch.ops.aten.permute.default(arg725_1, [1, 0]);  arg725_1 = None
        addmm_235: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg726_1, view_630, permute_315);  arg726_1 = view_630 = permute_315 = None
        view_631: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_235, _shape_param_712);  addmm_235 = _shape_param_712 = None
        add_219: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_631, add_218);  view_631 = add_218 = None
        mul_157: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_219, arg727_1);  add_219 = arg727_1 = None
        add_220: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_157, arg728_1);  mul_157 = arg728_1 = None
        view_632: "bf16[32768, 128]" = torch.ops.aten.view.default(add_220, _shape_param_713);  _shape_param_713 = None
        permute_316: "bf16[128, 512]" = torch.ops.aten.permute.default(arg729_1, [1, 0]);  arg729_1 = None
        addmm_236: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg730_1, view_632, permute_316);  arg730_1 = view_632 = permute_316 = None
        view_633: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_236, _shape_param_714);  addmm_236 = _shape_param_714 = None
        relu_62: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_633);  view_633 = None
        view_634: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_62, _shape_param_715);  relu_62 = _shape_param_715 = None
        permute_317: "bf16[512, 128]" = torch.ops.aten.permute.default(arg731_1, [1, 0]);  arg731_1 = None
        addmm_237: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg732_1, view_634, permute_317);  arg732_1 = view_634 = permute_317 = None
        view_635: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_237, _shape_param_716);  addmm_237 = _shape_param_716 = None
        add_221: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_635, add_220);  view_635 = add_220 = None
        mul_158: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_221, arg733_1);  add_221 = arg733_1 = None
        add_222: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_158, arg734_1);  mul_158 = arg734_1 = None
        view_636: "bf16[32768, 128]" = torch.ops.aten.view.default(add_222, _shape_param_717);  _shape_param_717 = None
        permute_318: "bf16[128, 512]" = torch.ops.aten.permute.default(arg735_1, [1, 0]);  arg735_1 = None
        addmm_238: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg736_1, view_636, permute_318);  arg736_1 = view_636 = permute_318 = None
        view_637: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_238, _shape_param_718);  addmm_238 = _shape_param_718 = None
        relu_63: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_637);  view_637 = None
        view_638: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_63, _shape_param_719);  relu_63 = _shape_param_719 = None
        permute_319: "bf16[512, 128]" = torch.ops.aten.permute.default(arg737_1, [1, 0]);  arg737_1 = None
        addmm_239: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg738_1, view_638, permute_319);  arg738_1 = view_638 = permute_319 = None
        view_639: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_239, _shape_param_720);  addmm_239 = _shape_param_720 = None
        add_223: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_639, add_222);  view_639 = add_222 = None
        mul_159: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_223, arg739_1);  add_223 = arg739_1 = None
        add_224: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_159, arg740_1);  mul_159 = arg740_1 = None
        view_640: "bf16[32768, 128]" = torch.ops.aten.view.default(add_224, _shape_param_721);  add_224 = _shape_param_721 = None
        permute_320: "bf16[128, 512]" = torch.ops.aten.permute.default(arg741_1, [1, 0]);  arg741_1 = None
        addmm_240: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg742_1, view_640, permute_320);  arg742_1 = view_640 = permute_320 = None
        view_641: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_240, _shape_param_722);  addmm_240 = _shape_param_722 = None
        add_225: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_641, add_212);  view_641 = add_212 = None
        mul_160: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_225, arg743_1);  add_225 = arg743_1 = None
        add_226: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_160, arg744_1);  mul_160 = arg744_1 = None
        view_642: "bf16[32768, 512]" = torch.ops.aten.view.default(add_226, _shape_param_723);  _shape_param_723 = None
        permute_321: "bf16[512, 128]" = torch.ops.aten.permute.default(arg745_1, [1, 0]);  arg745_1 = None
        addmm_241: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg746_1, view_642, permute_321);  arg746_1 = view_642 = permute_321 = None
        view_643: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_241, _shape_param_724);  addmm_241 = _shape_param_724 = None
        mul_161: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_643, arg747_1);  view_643 = arg747_1 = None
        add_227: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_161, arg748_1);  mul_161 = arg748_1 = None
        view_644: "bf16[32768, 128]" = torch.ops.aten.view.default(add_227, _shape_param_725);  _shape_param_725 = None
        permute_322: "bf16[128, 128]" = torch.ops.aten.permute.default(arg749_1, [1, 0]);  arg749_1 = None
        addmm_242: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg750_1, view_644, permute_322);  arg750_1 = view_644 = permute_322 = None
        view_645: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_242, _shape_param_726);  addmm_242 = _shape_param_726 = None
        view_646: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_645, _shape_param_727);  view_645 = _shape_param_727 = None
        permute_323: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_646, [0, 2, 1, 3]);  view_646 = None
        mul_162: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_323, 0.4204482076268573);  permute_323 = None
        expand_64: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_162, _shape_param_728);  mul_162 = _shape_param_728 = None
        clone_64: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_64, memory_format = torch.contiguous_format);  expand_64 = None
        view_647: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_64, _shape_param_729);  clone_64 = _shape_param_729 = None
        view_648: "bf16[32768, 128]" = torch.ops.aten.view.default(add_227, _shape_param_730);  add_227 = _shape_param_730 = None
        permute_324: "bf16[128, 128]" = torch.ops.aten.permute.default(arg751_1, [1, 0]);  arg751_1 = None
        addmm_243: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg752_1, view_648, permute_324);  arg752_1 = view_648 = permute_324 = None
        view_649: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_243, _shape_param_731);  addmm_243 = _shape_param_731 = None
        view_650: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_649, _shape_param_732);  view_649 = _shape_param_732 = None
        permute_325: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_650, [0, 2, 1, 3]);  view_650 = None
        permute_326: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_325, [0, 1, 3, 2]);  permute_325 = None
        mul_163: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_326, 0.4204482076268573);  permute_326 = None
        expand_65: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_163, _shape_param_733);  mul_163 = _shape_param_733 = None
        clone_65: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_651: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_65, _shape_param_734);  clone_65 = _shape_param_734 = None
        bmm_32: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_647, view_651);  view_647 = view_651 = None
        view_652: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_32, _shape_param_735);  bmm_32 = _shape_param_735 = None
        eq_16: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_652, -inf)
        logical_not_32: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        full_17: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_736, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_736 = None
        convert_element_type_32: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_652, torch.float32);  view_652 = None
        amax_16: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_32, [-1], True)
        sub_16: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_32, amax_16);  convert_element_type_32 = amax_16 = None
        exp_16: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_16);  sub_16 = None
        sum_17: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        convert_element_type_33: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_16, torch.bfloat16);  div_16 = None
        where_16: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_33, full_17, convert_element_type_33);  logical_not_33 = full_17 = convert_element_type_33 = None
        expand_66: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_16, _shape_param_737);  where_16 = _shape_param_737 = None
        view_653: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_66, _shape_param_738);  expand_66 = _shape_param_738 = None
        view_654: "bf16[32768, 512]" = torch.ops.aten.view.default(add_226, _shape_param_739);  _shape_param_739 = None
        permute_327: "bf16[512, 128]" = torch.ops.aten.permute.default(arg753_1, [1, 0]);  arg753_1 = None
        addmm_244: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg754_1, view_654, permute_327);  arg754_1 = view_654 = permute_327 = None
        view_655: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_244, _shape_param_740);  addmm_244 = _shape_param_740 = None
        view_656: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_655, _shape_param_741);  view_655 = _shape_param_741 = None
        permute_328: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_656, [0, 2, 1, 3]);  view_656 = None
        expand_67: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_328, _shape_param_742);  permute_328 = _shape_param_742 = None
        clone_66: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_657: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_66, _shape_param_743);  clone_66 = _shape_param_743 = None
        bmm_33: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_653, view_657);  view_653 = view_657 = None
        view_658: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_33, _shape_param_744);  bmm_33 = _shape_param_744 = None
        permute_329: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None
        clone_67: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_329, memory_format = torch.contiguous_format);  permute_329 = None
        view_659: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_67, _shape_param_745);  clone_67 = _shape_param_745 = None
        view_660: "bf16[32768, 128]" = torch.ops.aten.view.default(view_659, _shape_param_746);  view_659 = _shape_param_746 = None
        permute_330: "bf16[128, 128]" = torch.ops.aten.permute.default(arg755_1, [1, 0]);  arg755_1 = None
        addmm_245: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg756_1, view_660, permute_330);  arg756_1 = view_660 = permute_330 = None
        view_661: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_245, _shape_param_747);  addmm_245 = _shape_param_747 = None
        view_662: "bf16[32768, 512]" = torch.ops.aten.view.default(add_226, _shape_param_748);  _shape_param_748 = None
        permute_331: "bf16[512, 128]" = torch.ops.aten.permute.default(arg757_1, [1, 0]);  arg757_1 = None
        addmm_246: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg758_1, view_662, permute_331);  arg758_1 = view_662 = permute_331 = None
        view_663: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_246, _shape_param_749);  addmm_246 = _shape_param_749 = None
        mul_164: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_663, arg759_1);  view_663 = arg759_1 = None
        add_228: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_164, arg760_1);  mul_164 = arg760_1 = None
        add_229: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_661, add_228);  view_661 = add_228 = None
        mul_165: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_229, arg761_1);  add_229 = arg761_1 = None
        add_230: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_165, arg762_1);  mul_165 = arg762_1 = None
        view_664: "bf16[32768, 128]" = torch.ops.aten.view.default(add_230, _shape_param_750);  _shape_param_750 = None
        permute_332: "bf16[128, 512]" = torch.ops.aten.permute.default(arg763_1, [1, 0]);  arg763_1 = None
        addmm_247: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg764_1, view_664, permute_332);  arg764_1 = view_664 = permute_332 = None
        view_665: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_247, _shape_param_751);  addmm_247 = _shape_param_751 = None
        relu_64: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_665);  view_665 = None
        view_666: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_64, _shape_param_752);  relu_64 = _shape_param_752 = None
        permute_333: "bf16[512, 128]" = torch.ops.aten.permute.default(arg765_1, [1, 0]);  arg765_1 = None
        addmm_248: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg766_1, view_666, permute_333);  arg766_1 = view_666 = permute_333 = None
        view_667: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_248, _shape_param_753);  addmm_248 = _shape_param_753 = None
        add_231: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_667, add_230);  view_667 = add_230 = None
        mul_166: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_231, arg767_1);  add_231 = arg767_1 = None
        add_232: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_166, arg768_1);  mul_166 = arg768_1 = None
        view_668: "bf16[32768, 128]" = torch.ops.aten.view.default(add_232, _shape_param_754);  _shape_param_754 = None
        permute_334: "bf16[128, 512]" = torch.ops.aten.permute.default(arg769_1, [1, 0]);  arg769_1 = None
        addmm_249: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg770_1, view_668, permute_334);  arg770_1 = view_668 = permute_334 = None
        view_669: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_249, _shape_param_755);  addmm_249 = _shape_param_755 = None
        relu_65: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_669);  view_669 = None
        view_670: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_65, _shape_param_756);  relu_65 = _shape_param_756 = None
        permute_335: "bf16[512, 128]" = torch.ops.aten.permute.default(arg771_1, [1, 0]);  arg771_1 = None
        addmm_250: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg772_1, view_670, permute_335);  arg772_1 = view_670 = permute_335 = None
        view_671: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_250, _shape_param_757);  addmm_250 = _shape_param_757 = None
        add_233: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_671, add_232);  view_671 = add_232 = None
        mul_167: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_233, arg773_1);  add_233 = arg773_1 = None
        add_234: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_167, arg774_1);  mul_167 = arg774_1 = None
        view_672: "bf16[32768, 128]" = torch.ops.aten.view.default(add_234, _shape_param_758);  _shape_param_758 = None
        permute_336: "bf16[128, 512]" = torch.ops.aten.permute.default(arg775_1, [1, 0]);  arg775_1 = None
        addmm_251: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg776_1, view_672, permute_336);  arg776_1 = view_672 = permute_336 = None
        view_673: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_251, _shape_param_759);  addmm_251 = _shape_param_759 = None
        relu_66: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_673);  view_673 = None
        view_674: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_66, _shape_param_760);  relu_66 = _shape_param_760 = None
        permute_337: "bf16[512, 128]" = torch.ops.aten.permute.default(arg777_1, [1, 0]);  arg777_1 = None
        addmm_252: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg778_1, view_674, permute_337);  arg778_1 = view_674 = permute_337 = None
        view_675: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_252, _shape_param_761);  addmm_252 = _shape_param_761 = None
        add_235: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_675, add_234);  view_675 = add_234 = None
        mul_168: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_235, arg779_1);  add_235 = arg779_1 = None
        add_236: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_168, arg780_1);  mul_168 = arg780_1 = None
        view_676: "bf16[32768, 128]" = torch.ops.aten.view.default(add_236, _shape_param_762);  _shape_param_762 = None
        permute_338: "bf16[128, 512]" = torch.ops.aten.permute.default(arg781_1, [1, 0]);  arg781_1 = None
        addmm_253: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg782_1, view_676, permute_338);  arg782_1 = view_676 = permute_338 = None
        view_677: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_253, _shape_param_763);  addmm_253 = _shape_param_763 = None
        relu_67: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_677);  view_677 = None
        view_678: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_67, _shape_param_764);  relu_67 = _shape_param_764 = None
        permute_339: "bf16[512, 128]" = torch.ops.aten.permute.default(arg783_1, [1, 0]);  arg783_1 = None
        addmm_254: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg784_1, view_678, permute_339);  arg784_1 = view_678 = permute_339 = None
        view_679: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_254, _shape_param_765);  addmm_254 = _shape_param_765 = None
        add_237: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_679, add_236);  view_679 = add_236 = None
        mul_169: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_237, arg785_1);  add_237 = arg785_1 = None
        add_238: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_169, arg786_1);  mul_169 = arg786_1 = None
        view_680: "bf16[32768, 128]" = torch.ops.aten.view.default(add_238, _shape_param_766);  add_238 = _shape_param_766 = None
        permute_340: "bf16[128, 512]" = torch.ops.aten.permute.default(arg787_1, [1, 0]);  arg787_1 = None
        addmm_255: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg788_1, view_680, permute_340);  arg788_1 = view_680 = permute_340 = None
        view_681: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_255, _shape_param_767);  addmm_255 = _shape_param_767 = None
        add_239: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_681, add_226);  view_681 = add_226 = None
        mul_170: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_239, arg789_1);  add_239 = arg789_1 = None
        add_240: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_170, arg790_1);  mul_170 = arg790_1 = None
        view_682: "bf16[32768, 512]" = torch.ops.aten.view.default(add_240, _shape_param_768);  _shape_param_768 = None
        permute_341: "bf16[512, 128]" = torch.ops.aten.permute.default(arg791_1, [1, 0]);  arg791_1 = None
        addmm_256: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg792_1, view_682, permute_341);  arg792_1 = view_682 = permute_341 = None
        view_683: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_256, _shape_param_769);  addmm_256 = _shape_param_769 = None
        mul_171: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_683, arg793_1);  view_683 = arg793_1 = None
        add_241: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_171, arg794_1);  mul_171 = arg794_1 = None
        view_684: "bf16[32768, 128]" = torch.ops.aten.view.default(add_241, _shape_param_770);  _shape_param_770 = None
        permute_342: "bf16[128, 128]" = torch.ops.aten.permute.default(arg795_1, [1, 0]);  arg795_1 = None
        addmm_257: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg796_1, view_684, permute_342);  arg796_1 = view_684 = permute_342 = None
        view_685: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_257, _shape_param_771);  addmm_257 = _shape_param_771 = None
        view_686: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_685, _shape_param_772);  view_685 = _shape_param_772 = None
        permute_343: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_686, [0, 2, 1, 3]);  view_686 = None
        mul_172: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_343, 0.4204482076268573);  permute_343 = None
        expand_68: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_172, _shape_param_773);  mul_172 = _shape_param_773 = None
        clone_68: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_687: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_68, _shape_param_774);  clone_68 = _shape_param_774 = None
        view_688: "bf16[32768, 128]" = torch.ops.aten.view.default(add_241, _shape_param_775);  add_241 = _shape_param_775 = None
        permute_344: "bf16[128, 128]" = torch.ops.aten.permute.default(arg797_1, [1, 0]);  arg797_1 = None
        addmm_258: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg798_1, view_688, permute_344);  arg798_1 = view_688 = permute_344 = None
        view_689: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_258, _shape_param_776);  addmm_258 = _shape_param_776 = None
        view_690: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_689, _shape_param_777);  view_689 = _shape_param_777 = None
        permute_345: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_690, [0, 2, 1, 3]);  view_690 = None
        permute_346: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_345, [0, 1, 3, 2]);  permute_345 = None
        mul_173: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_346, 0.4204482076268573);  permute_346 = None
        expand_69: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_173, _shape_param_778);  mul_173 = _shape_param_778 = None
        clone_69: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_69, memory_format = torch.contiguous_format);  expand_69 = None
        view_691: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_69, _shape_param_779);  clone_69 = _shape_param_779 = None
        bmm_34: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_687, view_691);  view_687 = view_691 = None
        view_692: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_34, _shape_param_780);  bmm_34 = _shape_param_780 = None
        eq_17: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_692, -inf)
        logical_not_34: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        full_18: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_781, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_781 = None
        convert_element_type_34: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_692, torch.float32);  view_692 = None
        amax_17: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_34, [-1], True)
        sub_17: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_34, amax_17);  convert_element_type_34 = amax_17 = None
        exp_17: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_18: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        convert_element_type_35: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_17, torch.bfloat16);  div_17 = None
        where_17: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_35, full_18, convert_element_type_35);  logical_not_35 = full_18 = convert_element_type_35 = None
        expand_70: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_17, _shape_param_782);  where_17 = _shape_param_782 = None
        view_693: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_70, _shape_param_783);  expand_70 = _shape_param_783 = None
        view_694: "bf16[32768, 512]" = torch.ops.aten.view.default(add_240, _shape_param_784);  _shape_param_784 = None
        permute_347: "bf16[512, 128]" = torch.ops.aten.permute.default(arg799_1, [1, 0]);  arg799_1 = None
        addmm_259: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg800_1, view_694, permute_347);  arg800_1 = view_694 = permute_347 = None
        view_695: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_259, _shape_param_785);  addmm_259 = _shape_param_785 = None
        view_696: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_695, _shape_param_786);  view_695 = _shape_param_786 = None
        permute_348: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_696, [0, 2, 1, 3]);  view_696 = None
        expand_71: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_348, _shape_param_787);  permute_348 = _shape_param_787 = None
        clone_70: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_71, memory_format = torch.contiguous_format);  expand_71 = None
        view_697: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_70, _shape_param_788);  clone_70 = _shape_param_788 = None
        bmm_35: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_693, view_697);  view_693 = view_697 = None
        view_698: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_35, _shape_param_789);  bmm_35 = _shape_param_789 = None
        permute_349: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None
        clone_71: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_349, memory_format = torch.contiguous_format);  permute_349 = None
        view_699: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_71, _shape_param_790);  clone_71 = _shape_param_790 = None
        view_700: "bf16[32768, 128]" = torch.ops.aten.view.default(view_699, _shape_param_791);  view_699 = _shape_param_791 = None
        permute_350: "bf16[128, 128]" = torch.ops.aten.permute.default(arg801_1, [1, 0]);  arg801_1 = None
        addmm_260: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg802_1, view_700, permute_350);  arg802_1 = view_700 = permute_350 = None
        view_701: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_260, _shape_param_792);  addmm_260 = _shape_param_792 = None
        view_702: "bf16[32768, 512]" = torch.ops.aten.view.default(add_240, _shape_param_793);  _shape_param_793 = None
        permute_351: "bf16[512, 128]" = torch.ops.aten.permute.default(arg803_1, [1, 0]);  arg803_1 = None
        addmm_261: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg804_1, view_702, permute_351);  arg804_1 = view_702 = permute_351 = None
        view_703: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_261, _shape_param_794);  addmm_261 = _shape_param_794 = None
        mul_174: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_703, arg805_1);  view_703 = arg805_1 = None
        add_242: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_174, arg806_1);  mul_174 = arg806_1 = None
        add_243: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_701, add_242);  view_701 = add_242 = None
        mul_175: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_243, arg807_1);  add_243 = arg807_1 = None
        add_244: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_175, arg808_1);  mul_175 = arg808_1 = None
        view_704: "bf16[32768, 128]" = torch.ops.aten.view.default(add_244, _shape_param_795);  _shape_param_795 = None
        permute_352: "bf16[128, 512]" = torch.ops.aten.permute.default(arg809_1, [1, 0]);  arg809_1 = None
        addmm_262: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg810_1, view_704, permute_352);  arg810_1 = view_704 = permute_352 = None
        view_705: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_262, _shape_param_796);  addmm_262 = _shape_param_796 = None
        relu_68: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_705);  view_705 = None
        view_706: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_68, _shape_param_797);  relu_68 = _shape_param_797 = None
        permute_353: "bf16[512, 128]" = torch.ops.aten.permute.default(arg811_1, [1, 0]);  arg811_1 = None
        addmm_263: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg812_1, view_706, permute_353);  arg812_1 = view_706 = permute_353 = None
        view_707: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_263, _shape_param_798);  addmm_263 = _shape_param_798 = None
        add_245: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_707, add_244);  view_707 = add_244 = None
        mul_176: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_245, arg813_1);  add_245 = arg813_1 = None
        add_246: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_176, arg814_1);  mul_176 = arg814_1 = None
        view_708: "bf16[32768, 128]" = torch.ops.aten.view.default(add_246, _shape_param_799);  _shape_param_799 = None
        permute_354: "bf16[128, 512]" = torch.ops.aten.permute.default(arg815_1, [1, 0]);  arg815_1 = None
        addmm_264: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg816_1, view_708, permute_354);  arg816_1 = view_708 = permute_354 = None
        view_709: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_264, _shape_param_800);  addmm_264 = _shape_param_800 = None
        relu_69: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_709);  view_709 = None
        view_710: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_69, _shape_param_801);  relu_69 = _shape_param_801 = None
        permute_355: "bf16[512, 128]" = torch.ops.aten.permute.default(arg817_1, [1, 0]);  arg817_1 = None
        addmm_265: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg818_1, view_710, permute_355);  arg818_1 = view_710 = permute_355 = None
        view_711: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_265, _shape_param_802);  addmm_265 = _shape_param_802 = None
        add_247: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_711, add_246);  view_711 = add_246 = None
        mul_177: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_247, arg819_1);  add_247 = arg819_1 = None
        add_248: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_177, arg820_1);  mul_177 = arg820_1 = None
        view_712: "bf16[32768, 128]" = torch.ops.aten.view.default(add_248, _shape_param_803);  _shape_param_803 = None
        permute_356: "bf16[128, 512]" = torch.ops.aten.permute.default(arg821_1, [1, 0]);  arg821_1 = None
        addmm_266: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg822_1, view_712, permute_356);  arg822_1 = view_712 = permute_356 = None
        view_713: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_266, _shape_param_804);  addmm_266 = _shape_param_804 = None
        relu_70: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_713);  view_713 = None
        view_714: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_70, _shape_param_805);  relu_70 = _shape_param_805 = None
        permute_357: "bf16[512, 128]" = torch.ops.aten.permute.default(arg823_1, [1, 0]);  arg823_1 = None
        addmm_267: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg824_1, view_714, permute_357);  arg824_1 = view_714 = permute_357 = None
        view_715: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_267, _shape_param_806);  addmm_267 = _shape_param_806 = None
        add_249: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_715, add_248);  view_715 = add_248 = None
        mul_178: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_249, arg825_1);  add_249 = arg825_1 = None
        add_250: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_178, arg826_1);  mul_178 = arg826_1 = None
        view_716: "bf16[32768, 128]" = torch.ops.aten.view.default(add_250, _shape_param_807);  _shape_param_807 = None
        permute_358: "bf16[128, 512]" = torch.ops.aten.permute.default(arg827_1, [1, 0]);  arg827_1 = None
        addmm_268: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg828_1, view_716, permute_358);  arg828_1 = view_716 = permute_358 = None
        view_717: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_268, _shape_param_808);  addmm_268 = _shape_param_808 = None
        relu_71: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_717);  view_717 = None
        view_718: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_71, _shape_param_809);  relu_71 = _shape_param_809 = None
        permute_359: "bf16[512, 128]" = torch.ops.aten.permute.default(arg829_1, [1, 0]);  arg829_1 = None
        addmm_269: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg830_1, view_718, permute_359);  arg830_1 = view_718 = permute_359 = None
        view_719: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_269, _shape_param_810);  addmm_269 = _shape_param_810 = None
        add_251: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_719, add_250);  view_719 = add_250 = None
        mul_179: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_251, arg831_1);  add_251 = arg831_1 = None
        add_252: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_179, arg832_1);  mul_179 = arg832_1 = None
        view_720: "bf16[32768, 128]" = torch.ops.aten.view.default(add_252, _shape_param_811);  add_252 = _shape_param_811 = None
        permute_360: "bf16[128, 512]" = torch.ops.aten.permute.default(arg833_1, [1, 0]);  arg833_1 = None
        addmm_270: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg834_1, view_720, permute_360);  arg834_1 = view_720 = permute_360 = None
        view_721: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_270, _shape_param_812);  addmm_270 = _shape_param_812 = None
        add_253: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_721, add_240);  view_721 = add_240 = None
        mul_180: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_253, arg835_1);  add_253 = arg835_1 = None
        add_254: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_180, arg836_1);  mul_180 = arg836_1 = None
        view_722: "bf16[32768, 512]" = torch.ops.aten.view.default(add_254, _shape_param_813);  _shape_param_813 = None
        permute_361: "bf16[512, 128]" = torch.ops.aten.permute.default(arg837_1, [1, 0]);  arg837_1 = None
        addmm_271: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg838_1, view_722, permute_361);  arg838_1 = view_722 = permute_361 = None
        view_723: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_271, _shape_param_814);  addmm_271 = _shape_param_814 = None
        mul_181: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_723, arg839_1);  view_723 = arg839_1 = None
        add_255: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_181, arg840_1);  mul_181 = arg840_1 = None
        view_724: "bf16[32768, 128]" = torch.ops.aten.view.default(add_255, _shape_param_815);  _shape_param_815 = None
        permute_362: "bf16[128, 128]" = torch.ops.aten.permute.default(arg841_1, [1, 0]);  arg841_1 = None
        addmm_272: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg842_1, view_724, permute_362);  arg842_1 = view_724 = permute_362 = None
        view_725: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_272, _shape_param_816);  addmm_272 = _shape_param_816 = None
        view_726: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_725, _shape_param_817);  view_725 = _shape_param_817 = None
        permute_363: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_726, [0, 2, 1, 3]);  view_726 = None
        mul_182: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_363, 0.4204482076268573);  permute_363 = None
        expand_72: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_182, _shape_param_818);  mul_182 = _shape_param_818 = None
        clone_72: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_727: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_72, _shape_param_819);  clone_72 = _shape_param_819 = None
        view_728: "bf16[32768, 128]" = torch.ops.aten.view.default(add_255, _shape_param_820);  add_255 = _shape_param_820 = None
        permute_364: "bf16[128, 128]" = torch.ops.aten.permute.default(arg843_1, [1, 0]);  arg843_1 = None
        addmm_273: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg844_1, view_728, permute_364);  arg844_1 = view_728 = permute_364 = None
        view_729: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_273, _shape_param_821);  addmm_273 = _shape_param_821 = None
        view_730: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_729, _shape_param_822);  view_729 = _shape_param_822 = None
        permute_365: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_730, [0, 2, 1, 3]);  view_730 = None
        permute_366: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_365, [0, 1, 3, 2]);  permute_365 = None
        mul_183: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_366, 0.4204482076268573);  permute_366 = None
        expand_73: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_183, _shape_param_823);  mul_183 = _shape_param_823 = None
        clone_73: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_731: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_73, _shape_param_824);  clone_73 = _shape_param_824 = None
        bmm_36: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_727, view_731);  view_727 = view_731 = None
        view_732: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_36, _shape_param_825);  bmm_36 = _shape_param_825 = None
        eq_18: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_732, -inf)
        logical_not_36: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        full_19: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_826, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_826 = None
        convert_element_type_36: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_732, torch.float32);  view_732 = None
        amax_18: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_36, [-1], True)
        sub_18: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_36, amax_18);  convert_element_type_36 = amax_18 = None
        exp_18: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_18);  sub_18 = None
        sum_19: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        convert_element_type_37: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_18, torch.bfloat16);  div_18 = None
        where_18: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_37, full_19, convert_element_type_37);  logical_not_37 = full_19 = convert_element_type_37 = None
        expand_74: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_18, _shape_param_827);  where_18 = _shape_param_827 = None
        view_733: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_74, _shape_param_828);  expand_74 = _shape_param_828 = None
        view_734: "bf16[32768, 512]" = torch.ops.aten.view.default(add_254, _shape_param_829);  _shape_param_829 = None
        permute_367: "bf16[512, 128]" = torch.ops.aten.permute.default(arg845_1, [1, 0]);  arg845_1 = None
        addmm_274: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg846_1, view_734, permute_367);  arg846_1 = view_734 = permute_367 = None
        view_735: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_274, _shape_param_830);  addmm_274 = _shape_param_830 = None
        view_736: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_735, _shape_param_831);  view_735 = _shape_param_831 = None
        permute_368: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_736, [0, 2, 1, 3]);  view_736 = None
        expand_75: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_368, _shape_param_832);  permute_368 = _shape_param_832 = None
        clone_74: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_737: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_74, _shape_param_833);  clone_74 = _shape_param_833 = None
        bmm_37: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_733, view_737);  view_733 = view_737 = None
        view_738: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_37, _shape_param_834);  bmm_37 = _shape_param_834 = None
        permute_369: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None
        clone_75: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_369, memory_format = torch.contiguous_format);  permute_369 = None
        view_739: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_75, _shape_param_835);  clone_75 = _shape_param_835 = None
        view_740: "bf16[32768, 128]" = torch.ops.aten.view.default(view_739, _shape_param_836);  view_739 = _shape_param_836 = None
        permute_370: "bf16[128, 128]" = torch.ops.aten.permute.default(arg847_1, [1, 0]);  arg847_1 = None
        addmm_275: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg848_1, view_740, permute_370);  arg848_1 = view_740 = permute_370 = None
        view_741: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_275, _shape_param_837);  addmm_275 = _shape_param_837 = None
        view_742: "bf16[32768, 512]" = torch.ops.aten.view.default(add_254, _shape_param_838);  _shape_param_838 = None
        permute_371: "bf16[512, 128]" = torch.ops.aten.permute.default(arg849_1, [1, 0]);  arg849_1 = None
        addmm_276: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg850_1, view_742, permute_371);  arg850_1 = view_742 = permute_371 = None
        view_743: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_276, _shape_param_839);  addmm_276 = _shape_param_839 = None
        mul_184: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_743, arg851_1);  view_743 = arg851_1 = None
        add_256: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_184, arg852_1);  mul_184 = arg852_1 = None
        add_257: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_741, add_256);  view_741 = add_256 = None
        mul_185: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_257, arg853_1);  add_257 = arg853_1 = None
        add_258: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_185, arg854_1);  mul_185 = arg854_1 = None
        view_744: "bf16[32768, 128]" = torch.ops.aten.view.default(add_258, _shape_param_840);  _shape_param_840 = None
        permute_372: "bf16[128, 512]" = torch.ops.aten.permute.default(arg855_1, [1, 0]);  arg855_1 = None
        addmm_277: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg856_1, view_744, permute_372);  arg856_1 = view_744 = permute_372 = None
        view_745: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_277, _shape_param_841);  addmm_277 = _shape_param_841 = None
        relu_72: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_745);  view_745 = None
        view_746: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_72, _shape_param_842);  relu_72 = _shape_param_842 = None
        permute_373: "bf16[512, 128]" = torch.ops.aten.permute.default(arg857_1, [1, 0]);  arg857_1 = None
        addmm_278: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg858_1, view_746, permute_373);  arg858_1 = view_746 = permute_373 = None
        view_747: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_278, _shape_param_843);  addmm_278 = _shape_param_843 = None
        add_259: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_747, add_258);  view_747 = add_258 = None
        mul_186: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_259, arg859_1);  add_259 = arg859_1 = None
        add_260: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_186, arg860_1);  mul_186 = arg860_1 = None
        view_748: "bf16[32768, 128]" = torch.ops.aten.view.default(add_260, _shape_param_844);  _shape_param_844 = None
        permute_374: "bf16[128, 512]" = torch.ops.aten.permute.default(arg861_1, [1, 0]);  arg861_1 = None
        addmm_279: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg862_1, view_748, permute_374);  arg862_1 = view_748 = permute_374 = None
        view_749: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_279, _shape_param_845);  addmm_279 = _shape_param_845 = None
        relu_73: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_749);  view_749 = None
        view_750: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_73, _shape_param_846);  relu_73 = _shape_param_846 = None
        permute_375: "bf16[512, 128]" = torch.ops.aten.permute.default(arg863_1, [1, 0]);  arg863_1 = None
        addmm_280: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg864_1, view_750, permute_375);  arg864_1 = view_750 = permute_375 = None
        view_751: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_280, _shape_param_847);  addmm_280 = _shape_param_847 = None
        add_261: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_751, add_260);  view_751 = add_260 = None
        mul_187: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_261, arg865_1);  add_261 = arg865_1 = None
        add_262: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_187, arg866_1);  mul_187 = arg866_1 = None
        view_752: "bf16[32768, 128]" = torch.ops.aten.view.default(add_262, _shape_param_848);  _shape_param_848 = None
        permute_376: "bf16[128, 512]" = torch.ops.aten.permute.default(arg867_1, [1, 0]);  arg867_1 = None
        addmm_281: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg868_1, view_752, permute_376);  arg868_1 = view_752 = permute_376 = None
        view_753: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_281, _shape_param_849);  addmm_281 = _shape_param_849 = None
        relu_74: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_753);  view_753 = None
        view_754: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_74, _shape_param_850);  relu_74 = _shape_param_850 = None
        permute_377: "bf16[512, 128]" = torch.ops.aten.permute.default(arg869_1, [1, 0]);  arg869_1 = None
        addmm_282: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg870_1, view_754, permute_377);  arg870_1 = view_754 = permute_377 = None
        view_755: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_282, _shape_param_851);  addmm_282 = _shape_param_851 = None
        add_263: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_755, add_262);  view_755 = add_262 = None
        mul_188: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_263, arg871_1);  add_263 = arg871_1 = None
        add_264: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_188, arg872_1);  mul_188 = arg872_1 = None
        view_756: "bf16[32768, 128]" = torch.ops.aten.view.default(add_264, _shape_param_852);  _shape_param_852 = None
        permute_378: "bf16[128, 512]" = torch.ops.aten.permute.default(arg873_1, [1, 0]);  arg873_1 = None
        addmm_283: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg874_1, view_756, permute_378);  arg874_1 = view_756 = permute_378 = None
        view_757: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_283, _shape_param_853);  addmm_283 = _shape_param_853 = None
        relu_75: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_757);  view_757 = None
        view_758: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_75, _shape_param_854);  relu_75 = _shape_param_854 = None
        permute_379: "bf16[512, 128]" = torch.ops.aten.permute.default(arg875_1, [1, 0]);  arg875_1 = None
        addmm_284: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg876_1, view_758, permute_379);  arg876_1 = view_758 = permute_379 = None
        view_759: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_284, _shape_param_855);  addmm_284 = _shape_param_855 = None
        add_265: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_759, add_264);  view_759 = add_264 = None
        mul_189: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_265, arg877_1);  add_265 = arg877_1 = None
        add_266: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_189, arg878_1);  mul_189 = arg878_1 = None
        view_760: "bf16[32768, 128]" = torch.ops.aten.view.default(add_266, _shape_param_856);  add_266 = _shape_param_856 = None
        permute_380: "bf16[128, 512]" = torch.ops.aten.permute.default(arg879_1, [1, 0]);  arg879_1 = None
        addmm_285: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg880_1, view_760, permute_380);  arg880_1 = view_760 = permute_380 = None
        view_761: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_285, _shape_param_857);  addmm_285 = _shape_param_857 = None
        add_267: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_761, add_254);  view_761 = add_254 = None
        mul_190: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_267, arg881_1);  add_267 = arg881_1 = None
        add_268: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_190, arg882_1);  mul_190 = arg882_1 = None
        view_762: "bf16[32768, 512]" = torch.ops.aten.view.default(add_268, _shape_param_858);  _shape_param_858 = None
        permute_381: "bf16[512, 128]" = torch.ops.aten.permute.default(arg883_1, [1, 0]);  arg883_1 = None
        addmm_286: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg884_1, view_762, permute_381);  arg884_1 = view_762 = permute_381 = None
        view_763: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_286, _shape_param_859);  addmm_286 = _shape_param_859 = None
        mul_191: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_763, arg885_1);  view_763 = arg885_1 = None
        add_269: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_191, arg886_1);  mul_191 = arg886_1 = None
        view_764: "bf16[32768, 128]" = torch.ops.aten.view.default(add_269, _shape_param_860);  _shape_param_860 = None
        permute_382: "bf16[128, 128]" = torch.ops.aten.permute.default(arg887_1, [1, 0]);  arg887_1 = None
        addmm_287: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg888_1, view_764, permute_382);  arg888_1 = view_764 = permute_382 = None
        view_765: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_287, _shape_param_861);  addmm_287 = _shape_param_861 = None
        view_766: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_765, _shape_param_862);  view_765 = _shape_param_862 = None
        permute_383: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None
        mul_192: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_383, 0.4204482076268573);  permute_383 = None
        expand_76: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_192, _shape_param_863);  mul_192 = _shape_param_863 = None
        clone_76: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_76, memory_format = torch.contiguous_format);  expand_76 = None
        view_767: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_76, _shape_param_864);  clone_76 = _shape_param_864 = None
        view_768: "bf16[32768, 128]" = torch.ops.aten.view.default(add_269, _shape_param_865);  add_269 = _shape_param_865 = None
        permute_384: "bf16[128, 128]" = torch.ops.aten.permute.default(arg889_1, [1, 0]);  arg889_1 = None
        addmm_288: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg890_1, view_768, permute_384);  arg890_1 = view_768 = permute_384 = None
        view_769: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_288, _shape_param_866);  addmm_288 = _shape_param_866 = None
        view_770: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_769, _shape_param_867);  view_769 = _shape_param_867 = None
        permute_385: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_770, [0, 2, 1, 3]);  view_770 = None
        permute_386: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_385, [0, 1, 3, 2]);  permute_385 = None
        mul_193: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_386, 0.4204482076268573);  permute_386 = None
        expand_77: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_193, _shape_param_868);  mul_193 = _shape_param_868 = None
        clone_77: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_771: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_77, _shape_param_869);  clone_77 = _shape_param_869 = None
        bmm_38: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_767, view_771);  view_767 = view_771 = None
        view_772: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_38, _shape_param_870);  bmm_38 = _shape_param_870 = None
        eq_19: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_772, -inf)
        logical_not_38: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        full_20: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_871, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_871 = None
        convert_element_type_38: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_772, torch.float32);  view_772 = None
        amax_19: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_38, [-1], True)
        sub_19: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_38, amax_19);  convert_element_type_38 = amax_19 = None
        exp_19: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_19);  sub_19 = None
        sum_20: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        convert_element_type_39: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_19, torch.bfloat16);  div_19 = None
        where_19: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_39, full_20, convert_element_type_39);  logical_not_39 = full_20 = convert_element_type_39 = None
        expand_78: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_19, _shape_param_872);  where_19 = _shape_param_872 = None
        view_773: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_78, _shape_param_873);  expand_78 = _shape_param_873 = None
        view_774: "bf16[32768, 512]" = torch.ops.aten.view.default(add_268, _shape_param_874);  _shape_param_874 = None
        permute_387: "bf16[512, 128]" = torch.ops.aten.permute.default(arg891_1, [1, 0]);  arg891_1 = None
        addmm_289: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg892_1, view_774, permute_387);  arg892_1 = view_774 = permute_387 = None
        view_775: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_289, _shape_param_875);  addmm_289 = _shape_param_875 = None
        view_776: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_775, _shape_param_876);  view_775 = _shape_param_876 = None
        permute_388: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_776, [0, 2, 1, 3]);  view_776 = None
        expand_79: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_388, _shape_param_877);  permute_388 = _shape_param_877 = None
        clone_78: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_79, memory_format = torch.contiguous_format);  expand_79 = None
        view_777: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_78, _shape_param_878);  clone_78 = _shape_param_878 = None
        bmm_39: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_773, view_777);  view_773 = view_777 = None
        view_778: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_39, _shape_param_879);  bmm_39 = _shape_param_879 = None
        permute_389: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None
        clone_79: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_389, memory_format = torch.contiguous_format);  permute_389 = None
        view_779: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_79, _shape_param_880);  clone_79 = _shape_param_880 = None
        view_780: "bf16[32768, 128]" = torch.ops.aten.view.default(view_779, _shape_param_881);  view_779 = _shape_param_881 = None
        permute_390: "bf16[128, 128]" = torch.ops.aten.permute.default(arg893_1, [1, 0]);  arg893_1 = None
        addmm_290: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg894_1, view_780, permute_390);  arg894_1 = view_780 = permute_390 = None
        view_781: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_290, _shape_param_882);  addmm_290 = _shape_param_882 = None
        view_782: "bf16[32768, 512]" = torch.ops.aten.view.default(add_268, _shape_param_883);  _shape_param_883 = None
        permute_391: "bf16[512, 128]" = torch.ops.aten.permute.default(arg895_1, [1, 0]);  arg895_1 = None
        addmm_291: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg896_1, view_782, permute_391);  arg896_1 = view_782 = permute_391 = None
        view_783: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_291, _shape_param_884);  addmm_291 = _shape_param_884 = None
        mul_194: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_783, arg897_1);  view_783 = arg897_1 = None
        add_270: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_194, arg898_1);  mul_194 = arg898_1 = None
        add_271: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_781, add_270);  view_781 = add_270 = None
        mul_195: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_271, arg899_1);  add_271 = arg899_1 = None
        add_272: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_195, arg900_1);  mul_195 = arg900_1 = None
        view_784: "bf16[32768, 128]" = torch.ops.aten.view.default(add_272, _shape_param_885);  _shape_param_885 = None
        permute_392: "bf16[128, 512]" = torch.ops.aten.permute.default(arg901_1, [1, 0]);  arg901_1 = None
        addmm_292: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg902_1, view_784, permute_392);  arg902_1 = view_784 = permute_392 = None
        view_785: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_292, _shape_param_886);  addmm_292 = _shape_param_886 = None
        relu_76: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_785);  view_785 = None
        view_786: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_76, _shape_param_887);  relu_76 = _shape_param_887 = None
        permute_393: "bf16[512, 128]" = torch.ops.aten.permute.default(arg903_1, [1, 0]);  arg903_1 = None
        addmm_293: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg904_1, view_786, permute_393);  arg904_1 = view_786 = permute_393 = None
        view_787: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_293, _shape_param_888);  addmm_293 = _shape_param_888 = None
        add_273: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_787, add_272);  view_787 = add_272 = None
        mul_196: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_273, arg905_1);  add_273 = arg905_1 = None
        add_274: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_196, arg906_1);  mul_196 = arg906_1 = None
        view_788: "bf16[32768, 128]" = torch.ops.aten.view.default(add_274, _shape_param_889);  _shape_param_889 = None
        permute_394: "bf16[128, 512]" = torch.ops.aten.permute.default(arg907_1, [1, 0]);  arg907_1 = None
        addmm_294: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg908_1, view_788, permute_394);  arg908_1 = view_788 = permute_394 = None
        view_789: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_294, _shape_param_890);  addmm_294 = _shape_param_890 = None
        relu_77: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_789);  view_789 = None
        view_790: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_77, _shape_param_891);  relu_77 = _shape_param_891 = None
        permute_395: "bf16[512, 128]" = torch.ops.aten.permute.default(arg909_1, [1, 0]);  arg909_1 = None
        addmm_295: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg910_1, view_790, permute_395);  arg910_1 = view_790 = permute_395 = None
        view_791: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_295, _shape_param_892);  addmm_295 = _shape_param_892 = None
        add_275: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_791, add_274);  view_791 = add_274 = None
        mul_197: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_275, arg911_1);  add_275 = arg911_1 = None
        add_276: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_197, arg912_1);  mul_197 = arg912_1 = None
        view_792: "bf16[32768, 128]" = torch.ops.aten.view.default(add_276, _shape_param_893);  _shape_param_893 = None
        permute_396: "bf16[128, 512]" = torch.ops.aten.permute.default(arg913_1, [1, 0]);  arg913_1 = None
        addmm_296: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg914_1, view_792, permute_396);  arg914_1 = view_792 = permute_396 = None
        view_793: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_296, _shape_param_894);  addmm_296 = _shape_param_894 = None
        relu_78: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_793);  view_793 = None
        view_794: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_78, _shape_param_895);  relu_78 = _shape_param_895 = None
        permute_397: "bf16[512, 128]" = torch.ops.aten.permute.default(arg915_1, [1, 0]);  arg915_1 = None
        addmm_297: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg916_1, view_794, permute_397);  arg916_1 = view_794 = permute_397 = None
        view_795: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_297, _shape_param_896);  addmm_297 = _shape_param_896 = None
        add_277: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_795, add_276);  view_795 = add_276 = None
        mul_198: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_277, arg917_1);  add_277 = arg917_1 = None
        add_278: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_198, arg918_1);  mul_198 = arg918_1 = None
        view_796: "bf16[32768, 128]" = torch.ops.aten.view.default(add_278, _shape_param_897);  _shape_param_897 = None
        permute_398: "bf16[128, 512]" = torch.ops.aten.permute.default(arg919_1, [1, 0]);  arg919_1 = None
        addmm_298: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg920_1, view_796, permute_398);  arg920_1 = view_796 = permute_398 = None
        view_797: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_298, _shape_param_898);  addmm_298 = _shape_param_898 = None
        relu_79: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_797);  view_797 = None
        view_798: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_79, _shape_param_899);  relu_79 = _shape_param_899 = None
        permute_399: "bf16[512, 128]" = torch.ops.aten.permute.default(arg921_1, [1, 0]);  arg921_1 = None
        addmm_299: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg922_1, view_798, permute_399);  arg922_1 = view_798 = permute_399 = None
        view_799: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_299, _shape_param_900);  addmm_299 = _shape_param_900 = None
        add_279: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_799, add_278);  view_799 = add_278 = None
        mul_199: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_279, arg923_1);  add_279 = arg923_1 = None
        add_280: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_199, arg924_1);  mul_199 = arg924_1 = None
        view_800: "bf16[32768, 128]" = torch.ops.aten.view.default(add_280, _shape_param_901);  add_280 = _shape_param_901 = None
        permute_400: "bf16[128, 512]" = torch.ops.aten.permute.default(arg925_1, [1, 0]);  arg925_1 = None
        addmm_300: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg926_1, view_800, permute_400);  arg926_1 = view_800 = permute_400 = None
        view_801: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_300, _shape_param_902);  addmm_300 = _shape_param_902 = None
        add_281: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_801, add_268);  view_801 = add_268 = None
        mul_200: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_281, arg927_1);  add_281 = arg927_1 = None
        add_282: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_200, arg928_1);  mul_200 = arg928_1 = None
        view_802: "bf16[32768, 512]" = torch.ops.aten.view.default(add_282, _shape_param_903);  _shape_param_903 = None
        permute_401: "bf16[512, 128]" = torch.ops.aten.permute.default(arg929_1, [1, 0]);  arg929_1 = None
        addmm_301: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg930_1, view_802, permute_401);  arg930_1 = view_802 = permute_401 = None
        view_803: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_301, _shape_param_904);  addmm_301 = _shape_param_904 = None
        mul_201: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_803, arg931_1);  view_803 = arg931_1 = None
        add_283: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_201, arg932_1);  mul_201 = arg932_1 = None
        view_804: "bf16[32768, 128]" = torch.ops.aten.view.default(add_283, _shape_param_905);  _shape_param_905 = None
        permute_402: "bf16[128, 128]" = torch.ops.aten.permute.default(arg933_1, [1, 0]);  arg933_1 = None
        addmm_302: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg934_1, view_804, permute_402);  arg934_1 = view_804 = permute_402 = None
        view_805: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_302, _shape_param_906);  addmm_302 = _shape_param_906 = None
        view_806: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_805, _shape_param_907);  view_805 = _shape_param_907 = None
        permute_403: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_806, [0, 2, 1, 3]);  view_806 = None
        mul_202: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_403, 0.4204482076268573);  permute_403 = None
        expand_80: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_202, _shape_param_908);  mul_202 = _shape_param_908 = None
        clone_80: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_807: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_80, _shape_param_909);  clone_80 = _shape_param_909 = None
        view_808: "bf16[32768, 128]" = torch.ops.aten.view.default(add_283, _shape_param_910);  add_283 = _shape_param_910 = None
        permute_404: "bf16[128, 128]" = torch.ops.aten.permute.default(arg935_1, [1, 0]);  arg935_1 = None
        addmm_303: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg936_1, view_808, permute_404);  arg936_1 = view_808 = permute_404 = None
        view_809: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_303, _shape_param_911);  addmm_303 = _shape_param_911 = None
        view_810: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_809, _shape_param_912);  view_809 = _shape_param_912 = None
        permute_405: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_810, [0, 2, 1, 3]);  view_810 = None
        permute_406: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_405, [0, 1, 3, 2]);  permute_405 = None
        mul_203: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_406, 0.4204482076268573);  permute_406 = None
        expand_81: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_203, _shape_param_913);  mul_203 = _shape_param_913 = None
        clone_81: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_81, memory_format = torch.contiguous_format);  expand_81 = None
        view_811: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_81, _shape_param_914);  clone_81 = _shape_param_914 = None
        bmm_40: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_807, view_811);  view_807 = view_811 = None
        view_812: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_40, _shape_param_915);  bmm_40 = _shape_param_915 = None
        eq_20: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_812, -inf)
        logical_not_40: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        full_21: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_916, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_916 = None
        convert_element_type_40: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_812, torch.float32);  view_812 = None
        amax_20: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_40, [-1], True)
        sub_20: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_40, amax_20);  convert_element_type_40 = amax_20 = None
        exp_20: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_20);  sub_20 = None
        sum_21: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        convert_element_type_41: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_20, torch.bfloat16);  div_20 = None
        where_20: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_41, full_21, convert_element_type_41);  logical_not_41 = full_21 = convert_element_type_41 = None
        expand_82: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_20, _shape_param_917);  where_20 = _shape_param_917 = None
        view_813: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_82, _shape_param_918);  expand_82 = _shape_param_918 = None
        view_814: "bf16[32768, 512]" = torch.ops.aten.view.default(add_282, _shape_param_919);  _shape_param_919 = None
        permute_407: "bf16[512, 128]" = torch.ops.aten.permute.default(arg937_1, [1, 0]);  arg937_1 = None
        addmm_304: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg938_1, view_814, permute_407);  arg938_1 = view_814 = permute_407 = None
        view_815: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_304, _shape_param_920);  addmm_304 = _shape_param_920 = None
        view_816: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_815, _shape_param_921);  view_815 = _shape_param_921 = None
        permute_408: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_816, [0, 2, 1, 3]);  view_816 = None
        expand_83: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_408, _shape_param_922);  permute_408 = _shape_param_922 = None
        clone_82: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_817: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_82, _shape_param_923);  clone_82 = _shape_param_923 = None
        bmm_41: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_813, view_817);  view_813 = view_817 = None
        view_818: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_41, _shape_param_924);  bmm_41 = _shape_param_924 = None
        permute_409: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_818, [0, 2, 1, 3]);  view_818 = None
        clone_83: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_409, memory_format = torch.contiguous_format);  permute_409 = None
        view_819: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_83, _shape_param_925);  clone_83 = _shape_param_925 = None
        view_820: "bf16[32768, 128]" = torch.ops.aten.view.default(view_819, _shape_param_926);  view_819 = _shape_param_926 = None
        permute_410: "bf16[128, 128]" = torch.ops.aten.permute.default(arg939_1, [1, 0]);  arg939_1 = None
        addmm_305: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg940_1, view_820, permute_410);  arg940_1 = view_820 = permute_410 = None
        view_821: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_305, _shape_param_927);  addmm_305 = _shape_param_927 = None
        view_822: "bf16[32768, 512]" = torch.ops.aten.view.default(add_282, _shape_param_928);  _shape_param_928 = None
        permute_411: "bf16[512, 128]" = torch.ops.aten.permute.default(arg941_1, [1, 0]);  arg941_1 = None
        addmm_306: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg942_1, view_822, permute_411);  arg942_1 = view_822 = permute_411 = None
        view_823: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_306, _shape_param_929);  addmm_306 = _shape_param_929 = None
        mul_204: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_823, arg943_1);  view_823 = arg943_1 = None
        add_284: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_204, arg944_1);  mul_204 = arg944_1 = None
        add_285: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_821, add_284);  view_821 = add_284 = None
        mul_205: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_285, arg945_1);  add_285 = arg945_1 = None
        add_286: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_205, arg946_1);  mul_205 = arg946_1 = None
        view_824: "bf16[32768, 128]" = torch.ops.aten.view.default(add_286, _shape_param_930);  _shape_param_930 = None
        permute_412: "bf16[128, 512]" = torch.ops.aten.permute.default(arg947_1, [1, 0]);  arg947_1 = None
        addmm_307: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg948_1, view_824, permute_412);  arg948_1 = view_824 = permute_412 = None
        view_825: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_307, _shape_param_931);  addmm_307 = _shape_param_931 = None
        relu_80: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_825);  view_825 = None
        view_826: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_80, _shape_param_932);  relu_80 = _shape_param_932 = None
        permute_413: "bf16[512, 128]" = torch.ops.aten.permute.default(arg949_1, [1, 0]);  arg949_1 = None
        addmm_308: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg950_1, view_826, permute_413);  arg950_1 = view_826 = permute_413 = None
        view_827: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_308, _shape_param_933);  addmm_308 = _shape_param_933 = None
        add_287: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_827, add_286);  view_827 = add_286 = None
        mul_206: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_287, arg951_1);  add_287 = arg951_1 = None
        add_288: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_206, arg952_1);  mul_206 = arg952_1 = None
        view_828: "bf16[32768, 128]" = torch.ops.aten.view.default(add_288, _shape_param_934);  _shape_param_934 = None
        permute_414: "bf16[128, 512]" = torch.ops.aten.permute.default(arg953_1, [1, 0]);  arg953_1 = None
        addmm_309: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg954_1, view_828, permute_414);  arg954_1 = view_828 = permute_414 = None
        view_829: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_309, _shape_param_935);  addmm_309 = _shape_param_935 = None
        relu_81: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_829);  view_829 = None
        view_830: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_81, _shape_param_936);  relu_81 = _shape_param_936 = None
        permute_415: "bf16[512, 128]" = torch.ops.aten.permute.default(arg955_1, [1, 0]);  arg955_1 = None
        addmm_310: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg956_1, view_830, permute_415);  arg956_1 = view_830 = permute_415 = None
        view_831: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_310, _shape_param_937);  addmm_310 = _shape_param_937 = None
        add_289: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_831, add_288);  view_831 = add_288 = None
        mul_207: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_289, arg957_1);  add_289 = arg957_1 = None
        add_290: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_207, arg958_1);  mul_207 = arg958_1 = None
        view_832: "bf16[32768, 128]" = torch.ops.aten.view.default(add_290, _shape_param_938);  _shape_param_938 = None
        permute_416: "bf16[128, 512]" = torch.ops.aten.permute.default(arg959_1, [1, 0]);  arg959_1 = None
        addmm_311: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg960_1, view_832, permute_416);  arg960_1 = view_832 = permute_416 = None
        view_833: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_311, _shape_param_939);  addmm_311 = _shape_param_939 = None
        relu_82: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_833);  view_833 = None
        view_834: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_82, _shape_param_940);  relu_82 = _shape_param_940 = None
        permute_417: "bf16[512, 128]" = torch.ops.aten.permute.default(arg961_1, [1, 0]);  arg961_1 = None
        addmm_312: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg962_1, view_834, permute_417);  arg962_1 = view_834 = permute_417 = None
        view_835: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_312, _shape_param_941);  addmm_312 = _shape_param_941 = None
        add_291: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_835, add_290);  view_835 = add_290 = None
        mul_208: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_291, arg963_1);  add_291 = arg963_1 = None
        add_292: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_208, arg964_1);  mul_208 = arg964_1 = None
        view_836: "bf16[32768, 128]" = torch.ops.aten.view.default(add_292, _shape_param_942);  _shape_param_942 = None
        permute_418: "bf16[128, 512]" = torch.ops.aten.permute.default(arg965_1, [1, 0]);  arg965_1 = None
        addmm_313: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg966_1, view_836, permute_418);  arg966_1 = view_836 = permute_418 = None
        view_837: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_313, _shape_param_943);  addmm_313 = _shape_param_943 = None
        relu_83: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_837);  view_837 = None
        view_838: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_83, _shape_param_944);  relu_83 = _shape_param_944 = None
        permute_419: "bf16[512, 128]" = torch.ops.aten.permute.default(arg967_1, [1, 0]);  arg967_1 = None
        addmm_314: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg968_1, view_838, permute_419);  arg968_1 = view_838 = permute_419 = None
        view_839: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_314, _shape_param_945);  addmm_314 = _shape_param_945 = None
        add_293: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_839, add_292);  view_839 = add_292 = None
        mul_209: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_293, arg969_1);  add_293 = arg969_1 = None
        add_294: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_209, arg970_1);  mul_209 = arg970_1 = None
        view_840: "bf16[32768, 128]" = torch.ops.aten.view.default(add_294, _shape_param_946);  add_294 = _shape_param_946 = None
        permute_420: "bf16[128, 512]" = torch.ops.aten.permute.default(arg971_1, [1, 0]);  arg971_1 = None
        addmm_315: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg972_1, view_840, permute_420);  arg972_1 = view_840 = permute_420 = None
        view_841: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_315, _shape_param_947);  addmm_315 = _shape_param_947 = None
        add_295: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_841, add_282);  view_841 = add_282 = None
        mul_210: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_295, arg973_1);  add_295 = arg973_1 = None
        add_296: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_210, arg974_1);  mul_210 = arg974_1 = None
        view_842: "bf16[32768, 512]" = torch.ops.aten.view.default(add_296, _shape_param_948);  _shape_param_948 = None
        permute_421: "bf16[512, 128]" = torch.ops.aten.permute.default(arg975_1, [1, 0]);  arg975_1 = None
        addmm_316: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg976_1, view_842, permute_421);  arg976_1 = view_842 = permute_421 = None
        view_843: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_316, _shape_param_949);  addmm_316 = _shape_param_949 = None
        mul_211: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_843, arg977_1);  view_843 = arg977_1 = None
        add_297: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_211, arg978_1);  mul_211 = arg978_1 = None
        view_844: "bf16[32768, 128]" = torch.ops.aten.view.default(add_297, _shape_param_950);  _shape_param_950 = None
        permute_422: "bf16[128, 128]" = torch.ops.aten.permute.default(arg979_1, [1, 0]);  arg979_1 = None
        addmm_317: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg980_1, view_844, permute_422);  arg980_1 = view_844 = permute_422 = None
        view_845: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_317, _shape_param_951);  addmm_317 = _shape_param_951 = None
        view_846: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_845, _shape_param_952);  view_845 = _shape_param_952 = None
        permute_423: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None
        mul_212: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_423, 0.4204482076268573);  permute_423 = None
        expand_84: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_212, _shape_param_953);  mul_212 = _shape_param_953 = None
        clone_84: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_84, memory_format = torch.contiguous_format);  expand_84 = None
        view_847: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_84, _shape_param_954);  clone_84 = _shape_param_954 = None
        view_848: "bf16[32768, 128]" = torch.ops.aten.view.default(add_297, _shape_param_955);  add_297 = _shape_param_955 = None
        permute_424: "bf16[128, 128]" = torch.ops.aten.permute.default(arg981_1, [1, 0]);  arg981_1 = None
        addmm_318: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg982_1, view_848, permute_424);  arg982_1 = view_848 = permute_424 = None
        view_849: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_318, _shape_param_956);  addmm_318 = _shape_param_956 = None
        view_850: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_849, _shape_param_957);  view_849 = _shape_param_957 = None
        permute_425: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_850, [0, 2, 1, 3]);  view_850 = None
        permute_426: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_425, [0, 1, 3, 2]);  permute_425 = None
        mul_213: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_426, 0.4204482076268573);  permute_426 = None
        expand_85: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_213, _shape_param_958);  mul_213 = _shape_param_958 = None
        clone_85: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_851: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_85, _shape_param_959);  clone_85 = _shape_param_959 = None
        bmm_42: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_847, view_851);  view_847 = view_851 = None
        view_852: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_42, _shape_param_960);  bmm_42 = _shape_param_960 = None
        eq_21: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_852, -inf)
        logical_not_42: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        full_22: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_961, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_961 = None
        convert_element_type_42: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_852, torch.float32);  view_852 = None
        amax_21: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_42, [-1], True)
        sub_21: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_42, amax_21);  convert_element_type_42 = amax_21 = None
        exp_21: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_22: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        convert_element_type_43: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_21, torch.bfloat16);  div_21 = None
        where_21: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_43, full_22, convert_element_type_43);  logical_not_43 = full_22 = convert_element_type_43 = None
        expand_86: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_21, _shape_param_962);  where_21 = _shape_param_962 = None
        view_853: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_86, _shape_param_963);  expand_86 = _shape_param_963 = None
        view_854: "bf16[32768, 512]" = torch.ops.aten.view.default(add_296, _shape_param_964);  _shape_param_964 = None
        permute_427: "bf16[512, 128]" = torch.ops.aten.permute.default(arg983_1, [1, 0]);  arg983_1 = None
        addmm_319: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg984_1, view_854, permute_427);  arg984_1 = view_854 = permute_427 = None
        view_855: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_319, _shape_param_965);  addmm_319 = _shape_param_965 = None
        view_856: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_855, _shape_param_966);  view_855 = _shape_param_966 = None
        permute_428: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_856, [0, 2, 1, 3]);  view_856 = None
        expand_87: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_428, _shape_param_967);  permute_428 = _shape_param_967 = None
        clone_86: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_857: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_86, _shape_param_968);  clone_86 = _shape_param_968 = None
        bmm_43: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_853, view_857);  view_853 = view_857 = None
        view_858: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_43, _shape_param_969);  bmm_43 = _shape_param_969 = None
        permute_429: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_858, [0, 2, 1, 3]);  view_858 = None
        clone_87: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_429, memory_format = torch.contiguous_format);  permute_429 = None
        view_859: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_87, _shape_param_970);  clone_87 = _shape_param_970 = None
        view_860: "bf16[32768, 128]" = torch.ops.aten.view.default(view_859, _shape_param_971);  view_859 = _shape_param_971 = None
        permute_430: "bf16[128, 128]" = torch.ops.aten.permute.default(arg985_1, [1, 0]);  arg985_1 = None
        addmm_320: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg986_1, view_860, permute_430);  arg986_1 = view_860 = permute_430 = None
        view_861: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_320, _shape_param_972);  addmm_320 = _shape_param_972 = None
        view_862: "bf16[32768, 512]" = torch.ops.aten.view.default(add_296, _shape_param_973);  _shape_param_973 = None
        permute_431: "bf16[512, 128]" = torch.ops.aten.permute.default(arg987_1, [1, 0]);  arg987_1 = None
        addmm_321: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg988_1, view_862, permute_431);  arg988_1 = view_862 = permute_431 = None
        view_863: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_321, _shape_param_974);  addmm_321 = _shape_param_974 = None
        mul_214: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_863, arg989_1);  view_863 = arg989_1 = None
        add_298: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_214, arg990_1);  mul_214 = arg990_1 = None
        add_299: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_861, add_298);  view_861 = add_298 = None
        mul_215: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_299, arg991_1);  add_299 = arg991_1 = None
        add_300: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_215, arg992_1);  mul_215 = arg992_1 = None
        view_864: "bf16[32768, 128]" = torch.ops.aten.view.default(add_300, _shape_param_975);  _shape_param_975 = None
        permute_432: "bf16[128, 512]" = torch.ops.aten.permute.default(arg993_1, [1, 0]);  arg993_1 = None
        addmm_322: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg994_1, view_864, permute_432);  arg994_1 = view_864 = permute_432 = None
        view_865: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_322, _shape_param_976);  addmm_322 = _shape_param_976 = None
        relu_84: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_865);  view_865 = None
        view_866: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_84, _shape_param_977);  relu_84 = _shape_param_977 = None
        permute_433: "bf16[512, 128]" = torch.ops.aten.permute.default(arg995_1, [1, 0]);  arg995_1 = None
        addmm_323: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg996_1, view_866, permute_433);  arg996_1 = view_866 = permute_433 = None
        view_867: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_323, _shape_param_978);  addmm_323 = _shape_param_978 = None
        add_301: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_867, add_300);  view_867 = add_300 = None
        mul_216: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_301, arg997_1);  add_301 = arg997_1 = None
        add_302: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_216, arg998_1);  mul_216 = arg998_1 = None
        view_868: "bf16[32768, 128]" = torch.ops.aten.view.default(add_302, _shape_param_979);  _shape_param_979 = None
        permute_434: "bf16[128, 512]" = torch.ops.aten.permute.default(arg999_1, [1, 0]);  arg999_1 = None
        addmm_324: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1000_1, view_868, permute_434);  arg1000_1 = view_868 = permute_434 = None
        view_869: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_324, _shape_param_980);  addmm_324 = _shape_param_980 = None
        relu_85: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_869);  view_869 = None
        view_870: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_85, _shape_param_981);  relu_85 = _shape_param_981 = None
        permute_435: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1001_1, [1, 0]);  arg1001_1 = None
        addmm_325: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1002_1, view_870, permute_435);  arg1002_1 = view_870 = permute_435 = None
        view_871: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_325, _shape_param_982);  addmm_325 = _shape_param_982 = None
        add_303: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_871, add_302);  view_871 = add_302 = None
        mul_217: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_303, arg1003_1);  add_303 = arg1003_1 = None
        add_304: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_217, arg1004_1);  mul_217 = arg1004_1 = None
        view_872: "bf16[32768, 128]" = torch.ops.aten.view.default(add_304, _shape_param_983);  _shape_param_983 = None
        permute_436: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1005_1, [1, 0]);  arg1005_1 = None
        addmm_326: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1006_1, view_872, permute_436);  arg1006_1 = view_872 = permute_436 = None
        view_873: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_326, _shape_param_984);  addmm_326 = _shape_param_984 = None
        relu_86: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_873);  view_873 = None
        view_874: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_86, _shape_param_985);  relu_86 = _shape_param_985 = None
        permute_437: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1007_1, [1, 0]);  arg1007_1 = None
        addmm_327: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1008_1, view_874, permute_437);  arg1008_1 = view_874 = permute_437 = None
        view_875: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_327, _shape_param_986);  addmm_327 = _shape_param_986 = None
        add_305: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_875, add_304);  view_875 = add_304 = None
        mul_218: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_305, arg1009_1);  add_305 = arg1009_1 = None
        add_306: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_218, arg1010_1);  mul_218 = arg1010_1 = None
        view_876: "bf16[32768, 128]" = torch.ops.aten.view.default(add_306, _shape_param_987);  _shape_param_987 = None
        permute_438: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1011_1, [1, 0]);  arg1011_1 = None
        addmm_328: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1012_1, view_876, permute_438);  arg1012_1 = view_876 = permute_438 = None
        view_877: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_328, _shape_param_988);  addmm_328 = _shape_param_988 = None
        relu_87: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_877);  view_877 = None
        view_878: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_87, _shape_param_989);  relu_87 = _shape_param_989 = None
        permute_439: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1013_1, [1, 0]);  arg1013_1 = None
        addmm_329: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1014_1, view_878, permute_439);  arg1014_1 = view_878 = permute_439 = None
        view_879: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_329, _shape_param_990);  addmm_329 = _shape_param_990 = None
        add_307: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_879, add_306);  view_879 = add_306 = None
        mul_219: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_307, arg1015_1);  add_307 = arg1015_1 = None
        add_308: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_219, arg1016_1);  mul_219 = arg1016_1 = None
        view_880: "bf16[32768, 128]" = torch.ops.aten.view.default(add_308, _shape_param_991);  add_308 = _shape_param_991 = None
        permute_440: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1017_1, [1, 0]);  arg1017_1 = None
        addmm_330: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1018_1, view_880, permute_440);  arg1018_1 = view_880 = permute_440 = None
        view_881: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_330, _shape_param_992);  addmm_330 = _shape_param_992 = None
        add_309: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_881, add_296);  view_881 = add_296 = None
        mul_220: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_309, arg1019_1);  add_309 = arg1019_1 = None
        add_310: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_220, arg1020_1);  mul_220 = arg1020_1 = None
        view_882: "bf16[32768, 512]" = torch.ops.aten.view.default(add_310, _shape_param_993);  _shape_param_993 = None
        permute_441: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1021_1, [1, 0]);  arg1021_1 = None
        addmm_331: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1022_1, view_882, permute_441);  arg1022_1 = view_882 = permute_441 = None
        view_883: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_331, _shape_param_994);  addmm_331 = _shape_param_994 = None
        mul_221: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_883, arg1023_1);  view_883 = arg1023_1 = None
        add_311: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_221, arg1024_1);  mul_221 = arg1024_1 = None
        view_884: "bf16[32768, 128]" = torch.ops.aten.view.default(add_311, _shape_param_995);  _shape_param_995 = None
        permute_442: "bf16[128, 128]" = torch.ops.aten.permute.default(arg1025_1, [1, 0]);  arg1025_1 = None
        addmm_332: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1026_1, view_884, permute_442);  arg1026_1 = view_884 = permute_442 = None
        view_885: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_332, _shape_param_996);  addmm_332 = _shape_param_996 = None
        view_886: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_885, _shape_param_997);  view_885 = _shape_param_997 = None
        permute_443: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_886, [0, 2, 1, 3]);  view_886 = None
        mul_222: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_443, 0.4204482076268573);  permute_443 = None
        expand_88: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_222, _shape_param_998);  mul_222 = _shape_param_998 = None
        clone_88: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_887: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_88, _shape_param_999);  clone_88 = _shape_param_999 = None
        view_888: "bf16[32768, 128]" = torch.ops.aten.view.default(add_311, _shape_param_1000);  add_311 = _shape_param_1000 = None
        permute_444: "bf16[128, 128]" = torch.ops.aten.permute.default(arg1027_1, [1, 0]);  arg1027_1 = None
        addmm_333: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1028_1, view_888, permute_444);  arg1028_1 = view_888 = permute_444 = None
        view_889: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_333, _shape_param_1001);  addmm_333 = _shape_param_1001 = None
        view_890: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_889, _shape_param_1002);  view_889 = _shape_param_1002 = None
        permute_445: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_890, [0, 2, 1, 3]);  view_890 = None
        permute_446: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_445, [0, 1, 3, 2]);  permute_445 = None
        mul_223: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_446, 0.4204482076268573);  permute_446 = None
        expand_89: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_223, _shape_param_1003);  mul_223 = _shape_param_1003 = None
        clone_89: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_89, memory_format = torch.contiguous_format);  expand_89 = None
        view_891: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_89, _shape_param_1004);  clone_89 = _shape_param_1004 = None
        bmm_44: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_887, view_891);  view_887 = view_891 = None
        view_892: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_44, _shape_param_1005);  bmm_44 = _shape_param_1005 = None
        eq_22: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_892, -inf)
        logical_not_44: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        full_23: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_1006, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1006 = None
        convert_element_type_44: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_892, torch.float32);  view_892 = None
        amax_22: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_44, [-1], True)
        sub_22: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_44, amax_22);  convert_element_type_44 = amax_22 = None
        exp_22: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_22);  sub_22 = None
        sum_23: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        convert_element_type_45: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_22, torch.bfloat16);  div_22 = None
        where_22: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_45, full_23, convert_element_type_45);  logical_not_45 = full_23 = convert_element_type_45 = None
        expand_90: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_22, _shape_param_1007);  where_22 = _shape_param_1007 = None
        view_893: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_90, _shape_param_1008);  expand_90 = _shape_param_1008 = None
        view_894: "bf16[32768, 512]" = torch.ops.aten.view.default(add_310, _shape_param_1009);  _shape_param_1009 = None
        permute_447: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1029_1, [1, 0]);  arg1029_1 = None
        addmm_334: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1030_1, view_894, permute_447);  arg1030_1 = view_894 = permute_447 = None
        view_895: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_334, _shape_param_1010);  addmm_334 = _shape_param_1010 = None
        view_896: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_895, _shape_param_1011);  view_895 = _shape_param_1011 = None
        permute_448: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_896, [0, 2, 1, 3]);  view_896 = None
        expand_91: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_448, _shape_param_1012);  permute_448 = _shape_param_1012 = None
        clone_90: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_91, memory_format = torch.contiguous_format);  expand_91 = None
        view_897: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_90, _shape_param_1013);  clone_90 = _shape_param_1013 = None
        bmm_45: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_893, view_897);  view_893 = view_897 = None
        view_898: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_45, _shape_param_1014);  bmm_45 = _shape_param_1014 = None
        permute_449: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_898, [0, 2, 1, 3]);  view_898 = None
        clone_91: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_449, memory_format = torch.contiguous_format);  permute_449 = None
        view_899: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_91, _shape_param_1015);  clone_91 = _shape_param_1015 = None
        view_900: "bf16[32768, 128]" = torch.ops.aten.view.default(view_899, _shape_param_1016);  view_899 = _shape_param_1016 = None
        permute_450: "bf16[128, 128]" = torch.ops.aten.permute.default(arg1031_1, [1, 0]);  arg1031_1 = None
        addmm_335: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1032_1, view_900, permute_450);  arg1032_1 = view_900 = permute_450 = None
        view_901: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_335, _shape_param_1017);  addmm_335 = _shape_param_1017 = None
        view_902: "bf16[32768, 512]" = torch.ops.aten.view.default(add_310, _shape_param_1018);  _shape_param_1018 = None
        permute_451: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1033_1, [1, 0]);  arg1033_1 = None
        addmm_336: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1034_1, view_902, permute_451);  arg1034_1 = view_902 = permute_451 = None
        view_903: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_336, _shape_param_1019);  addmm_336 = _shape_param_1019 = None
        mul_224: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_903, arg1035_1);  view_903 = arg1035_1 = None
        add_312: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_224, arg1036_1);  mul_224 = arg1036_1 = None
        add_313: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_901, add_312);  view_901 = add_312 = None
        mul_225: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_313, arg1037_1);  add_313 = arg1037_1 = None
        add_314: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_225, arg1038_1);  mul_225 = arg1038_1 = None
        view_904: "bf16[32768, 128]" = torch.ops.aten.view.default(add_314, _shape_param_1020);  _shape_param_1020 = None
        permute_452: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1039_1, [1, 0]);  arg1039_1 = None
        addmm_337: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1040_1, view_904, permute_452);  arg1040_1 = view_904 = permute_452 = None
        view_905: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_337, _shape_param_1021);  addmm_337 = _shape_param_1021 = None
        relu_88: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_905);  view_905 = None
        view_906: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_88, _shape_param_1022);  relu_88 = _shape_param_1022 = None
        permute_453: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1041_1, [1, 0]);  arg1041_1 = None
        addmm_338: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1042_1, view_906, permute_453);  arg1042_1 = view_906 = permute_453 = None
        view_907: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_338, _shape_param_1023);  addmm_338 = _shape_param_1023 = None
        add_315: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_907, add_314);  view_907 = add_314 = None
        mul_226: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_315, arg1043_1);  add_315 = arg1043_1 = None
        add_316: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_226, arg1044_1);  mul_226 = arg1044_1 = None
        view_908: "bf16[32768, 128]" = torch.ops.aten.view.default(add_316, _shape_param_1024);  _shape_param_1024 = None
        permute_454: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1045_1, [1, 0]);  arg1045_1 = None
        addmm_339: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1046_1, view_908, permute_454);  arg1046_1 = view_908 = permute_454 = None
        view_909: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_339, _shape_param_1025);  addmm_339 = _shape_param_1025 = None
        relu_89: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_909);  view_909 = None
        view_910: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_89, _shape_param_1026);  relu_89 = _shape_param_1026 = None
        permute_455: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1047_1, [1, 0]);  arg1047_1 = None
        addmm_340: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1048_1, view_910, permute_455);  arg1048_1 = view_910 = permute_455 = None
        view_911: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_340, _shape_param_1027);  addmm_340 = _shape_param_1027 = None
        add_317: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_911, add_316);  view_911 = add_316 = None
        mul_227: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_317, arg1049_1);  add_317 = arg1049_1 = None
        add_318: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_227, arg1050_1);  mul_227 = arg1050_1 = None
        view_912: "bf16[32768, 128]" = torch.ops.aten.view.default(add_318, _shape_param_1028);  _shape_param_1028 = None
        permute_456: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1051_1, [1, 0]);  arg1051_1 = None
        addmm_341: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1052_1, view_912, permute_456);  arg1052_1 = view_912 = permute_456 = None
        view_913: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_341, _shape_param_1029);  addmm_341 = _shape_param_1029 = None
        relu_90: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_913);  view_913 = None
        view_914: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_90, _shape_param_1030);  relu_90 = _shape_param_1030 = None
        permute_457: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1053_1, [1, 0]);  arg1053_1 = None
        addmm_342: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1054_1, view_914, permute_457);  arg1054_1 = view_914 = permute_457 = None
        view_915: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_342, _shape_param_1031);  addmm_342 = _shape_param_1031 = None
        add_319: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_915, add_318);  view_915 = add_318 = None
        mul_228: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_319, arg1055_1);  add_319 = arg1055_1 = None
        add_320: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_228, arg1056_1);  mul_228 = arg1056_1 = None
        view_916: "bf16[32768, 128]" = torch.ops.aten.view.default(add_320, _shape_param_1032);  _shape_param_1032 = None
        permute_458: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1057_1, [1, 0]);  arg1057_1 = None
        addmm_343: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1058_1, view_916, permute_458);  arg1058_1 = view_916 = permute_458 = None
        view_917: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_343, _shape_param_1033);  addmm_343 = _shape_param_1033 = None
        relu_91: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_917);  view_917 = None
        view_918: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_91, _shape_param_1034);  relu_91 = _shape_param_1034 = None
        permute_459: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1059_1, [1, 0]);  arg1059_1 = None
        addmm_344: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1060_1, view_918, permute_459);  arg1060_1 = view_918 = permute_459 = None
        view_919: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_344, _shape_param_1035);  addmm_344 = _shape_param_1035 = None
        add_321: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_919, add_320);  view_919 = add_320 = None
        mul_229: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_321, arg1061_1);  add_321 = arg1061_1 = None
        add_322: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_229, arg1062_1);  mul_229 = arg1062_1 = None
        view_920: "bf16[32768, 128]" = torch.ops.aten.view.default(add_322, _shape_param_1036);  add_322 = _shape_param_1036 = None
        permute_460: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1063_1, [1, 0]);  arg1063_1 = None
        addmm_345: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1064_1, view_920, permute_460);  arg1064_1 = view_920 = permute_460 = None
        view_921: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_345, _shape_param_1037);  addmm_345 = _shape_param_1037 = None
        add_323: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_921, add_310);  view_921 = add_310 = None
        mul_230: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_323, arg1065_1);  add_323 = arg1065_1 = None
        add_324: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_230, arg1066_1);  mul_230 = arg1066_1 = None
        view_922: "bf16[32768, 512]" = torch.ops.aten.view.default(add_324, _shape_param_1038);  _shape_param_1038 = None
        permute_461: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1067_1, [1, 0]);  arg1067_1 = None
        addmm_346: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1068_1, view_922, permute_461);  arg1068_1 = view_922 = permute_461 = None
        view_923: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_346, _shape_param_1039);  addmm_346 = _shape_param_1039 = None
        mul_231: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_923, arg1069_1);  view_923 = arg1069_1 = None
        add_325: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_231, arg1070_1);  mul_231 = arg1070_1 = None
        view_924: "bf16[32768, 128]" = torch.ops.aten.view.default(add_325, _shape_param_1040);  _shape_param_1040 = None
        permute_462: "bf16[128, 128]" = torch.ops.aten.permute.default(arg1071_1, [1, 0]);  arg1071_1 = None
        addmm_347: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1072_1, view_924, permute_462);  arg1072_1 = view_924 = permute_462 = None
        view_925: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_347, _shape_param_1041);  addmm_347 = _shape_param_1041 = None
        view_926: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_925, _shape_param_1042);  view_925 = _shape_param_1042 = None
        permute_463: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_926, [0, 2, 1, 3]);  view_926 = None
        mul_232: "bf16[256, 4, 128, 32]" = torch.ops.aten.mul.Scalar(permute_463, 0.4204482076268573);  permute_463 = None
        expand_92: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(mul_232, _shape_param_1043);  mul_232 = _shape_param_1043 = None
        clone_92: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_927: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_92, _shape_param_1044);  clone_92 = _shape_param_1044 = None
        view_928: "bf16[32768, 128]" = torch.ops.aten.view.default(add_325, _shape_param_1045);  add_325 = _shape_param_1045 = None
        permute_464: "bf16[128, 128]" = torch.ops.aten.permute.default(arg1073_1, [1, 0]);  arg1073_1 = None
        addmm_348: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1074_1, view_928, permute_464);  arg1074_1 = view_928 = permute_464 = None
        view_929: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_348, _shape_param_1046);  addmm_348 = _shape_param_1046 = None
        view_930: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_929, _shape_param_1047);  view_929 = _shape_param_1047 = None
        permute_465: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_930, [0, 2, 1, 3]);  view_930 = None
        permute_466: "bf16[256, 4, 32, 128]" = torch.ops.aten.permute.default(permute_465, [0, 1, 3, 2]);  permute_465 = None
        mul_233: "bf16[256, 4, 32, 128]" = torch.ops.aten.mul.Scalar(permute_466, 0.4204482076268573);  permute_466 = None
        expand_93: "bf16[256, 4, 32, 128]" = torch.ops.aten.expand.default(mul_233, _shape_param_1048);  mul_233 = _shape_param_1048 = None
        clone_93: "bf16[256, 4, 32, 128]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_931: "bf16[1024, 32, 128]" = torch.ops.aten.view.default(clone_93, _shape_param_1049);  clone_93 = _shape_param_1049 = None
        bmm_46: "bf16[1024, 128, 128]" = torch.ops.aten.bmm.default(view_927, view_931);  view_927 = view_931 = None
        view_932: "bf16[256, 4, 128, 128]" = torch.ops.aten.view.default(bmm_46, _shape_param_1050);  bmm_46 = _shape_param_1050 = None
        eq_23: "b8[256, 4, 128, 128]" = torch.ops.aten.eq.Scalar(view_932, -inf)
        logical_not_46: "b8[256, 4, 128, 128]" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[256, 4, 128, 1]" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[256, 4, 128, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        full_24: "bf16[256, 4, 128, 128]" = torch.ops.aten.full.default(_shape_param_1051, 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False);  _shape_param_1051 = None
        convert_element_type_46: "f32[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(view_932, torch.float32);  view_932 = None
        amax_23: "f32[256, 4, 128, 1]" = torch.ops.aten.amax.default(convert_element_type_46, [-1], True)
        sub_23: "f32[256, 4, 128, 128]" = torch.ops.aten.sub.Tensor(convert_element_type_46, amax_23);  convert_element_type_46 = amax_23 = None
        exp_23: "f32[256, 4, 128, 128]" = torch.ops.aten.exp.default(sub_23);  sub_23 = None
        sum_24: "f32[256, 4, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[256, 4, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        convert_element_type_47: "bf16[256, 4, 128, 128]" = torch.ops.prims.convert_element_type.default(div_23, torch.bfloat16);  div_23 = None
        where_23: "bf16[256, 4, 128, 128]" = torch.ops.aten.where.self(logical_not_47, full_24, convert_element_type_47);  logical_not_47 = full_24 = convert_element_type_47 = None
        expand_94: "bf16[256, 4, 128, 128]" = torch.ops.aten.expand.default(where_23, _shape_param_1052);  where_23 = _shape_param_1052 = None
        view_933: "bf16[1024, 128, 128]" = torch.ops.aten.view.default(expand_94, _shape_param_1053);  expand_94 = _shape_param_1053 = None
        view_934: "bf16[32768, 512]" = torch.ops.aten.view.default(add_324, _shape_param_1054);  _shape_param_1054 = None
        permute_467: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1075_1, [1, 0]);  arg1075_1 = None
        addmm_349: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1076_1, view_934, permute_467);  arg1076_1 = view_934 = permute_467 = None
        view_935: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_349, _shape_param_1055);  addmm_349 = _shape_param_1055 = None
        view_936: "bf16[256, 128, 4, 32]" = torch.ops.aten.view.default(view_935, _shape_param_1056);  view_935 = _shape_param_1056 = None
        permute_468: "bf16[256, 4, 128, 32]" = torch.ops.aten.permute.default(view_936, [0, 2, 1, 3]);  view_936 = None
        expand_95: "bf16[256, 4, 128, 32]" = torch.ops.aten.expand.default(permute_468, _shape_param_1057);  permute_468 = _shape_param_1057 = None
        clone_94: "bf16[256, 4, 128, 32]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_937: "bf16[1024, 128, 32]" = torch.ops.aten.view.default(clone_94, _shape_param_1058);  clone_94 = _shape_param_1058 = None
        bmm_47: "bf16[1024, 128, 32]" = torch.ops.aten.bmm.default(view_933, view_937);  view_933 = view_937 = None
        view_938: "bf16[256, 4, 128, 32]" = torch.ops.aten.view.default(bmm_47, _shape_param_1059);  bmm_47 = _shape_param_1059 = None
        permute_469: "bf16[256, 128, 4, 32]" = torch.ops.aten.permute.default(view_938, [0, 2, 1, 3]);  view_938 = None
        clone_95: "bf16[256, 128, 4, 32]" = torch.ops.aten.clone.default(permute_469, memory_format = torch.contiguous_format);  permute_469 = None
        view_939: "bf16[256, 128, 128]" = torch.ops.aten.view.default(clone_95, _shape_param_1060);  clone_95 = _shape_param_1060 = None
        view_940: "bf16[32768, 128]" = torch.ops.aten.view.default(view_939, _shape_param_1061);  view_939 = _shape_param_1061 = None
        permute_470: "bf16[128, 128]" = torch.ops.aten.permute.default(arg1077_1, [1, 0]);  arg1077_1 = None
        addmm_350: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1078_1, view_940, permute_470);  arg1078_1 = view_940 = permute_470 = None
        view_941: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_350, _shape_param_1062);  addmm_350 = _shape_param_1062 = None
        view_942: "bf16[32768, 512]" = torch.ops.aten.view.default(add_324, _shape_param_1063);  _shape_param_1063 = None
        permute_471: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1079_1, [1, 0]);  arg1079_1 = None
        addmm_351: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1080_1, view_942, permute_471);  arg1080_1 = view_942 = permute_471 = None
        view_943: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_351, _shape_param_1064);  addmm_351 = _shape_param_1064 = None
        mul_234: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(view_943, arg1081_1);  view_943 = arg1081_1 = None
        add_326: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_234, arg1082_1);  mul_234 = arg1082_1 = None
        add_327: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_941, add_326);  view_941 = add_326 = None
        mul_235: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_327, arg1083_1);  add_327 = arg1083_1 = None
        add_328: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_235, arg1084_1);  mul_235 = arg1084_1 = None
        view_944: "bf16[32768, 128]" = torch.ops.aten.view.default(add_328, _shape_param_1065);  _shape_param_1065 = None
        permute_472: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1085_1, [1, 0]);  arg1085_1 = None
        addmm_352: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1086_1, view_944, permute_472);  arg1086_1 = view_944 = permute_472 = None
        view_945: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_352, _shape_param_1066);  addmm_352 = _shape_param_1066 = None
        relu_92: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_945);  view_945 = None
        view_946: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_92, _shape_param_1067);  relu_92 = _shape_param_1067 = None
        permute_473: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1087_1, [1, 0]);  arg1087_1 = None
        addmm_353: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1088_1, view_946, permute_473);  arg1088_1 = view_946 = permute_473 = None
        view_947: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_353, _shape_param_1068);  addmm_353 = _shape_param_1068 = None
        add_329: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_947, add_328);  view_947 = add_328 = None
        mul_236: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_329, arg1089_1);  add_329 = arg1089_1 = None
        add_330: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_236, arg1090_1);  mul_236 = arg1090_1 = None
        view_948: "bf16[32768, 128]" = torch.ops.aten.view.default(add_330, _shape_param_1069);  _shape_param_1069 = None
        permute_474: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1091_1, [1, 0]);  arg1091_1 = None
        addmm_354: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1092_1, view_948, permute_474);  arg1092_1 = view_948 = permute_474 = None
        view_949: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_354, _shape_param_1070);  addmm_354 = _shape_param_1070 = None
        relu_93: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_949);  view_949 = None
        view_950: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_93, _shape_param_1071);  relu_93 = _shape_param_1071 = None
        permute_475: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1093_1, [1, 0]);  arg1093_1 = None
        addmm_355: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1094_1, view_950, permute_475);  arg1094_1 = view_950 = permute_475 = None
        view_951: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_355, _shape_param_1072);  addmm_355 = _shape_param_1072 = None
        add_331: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_951, add_330);  view_951 = add_330 = None
        mul_237: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_331, arg1095_1);  add_331 = arg1095_1 = None
        add_332: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_237, arg1096_1);  mul_237 = arg1096_1 = None
        view_952: "bf16[32768, 128]" = torch.ops.aten.view.default(add_332, _shape_param_1073);  _shape_param_1073 = None
        permute_476: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1097_1, [1, 0]);  arg1097_1 = None
        addmm_356: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1098_1, view_952, permute_476);  arg1098_1 = view_952 = permute_476 = None
        view_953: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_356, _shape_param_1074);  addmm_356 = _shape_param_1074 = None
        relu_94: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_953);  view_953 = None
        view_954: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_94, _shape_param_1075);  relu_94 = _shape_param_1075 = None
        permute_477: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1099_1, [1, 0]);  arg1099_1 = None
        addmm_357: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1100_1, view_954, permute_477);  arg1100_1 = view_954 = permute_477 = None
        view_955: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_357, _shape_param_1076);  addmm_357 = _shape_param_1076 = None
        add_333: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_955, add_332);  view_955 = add_332 = None
        mul_238: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_333, arg1101_1);  add_333 = arg1101_1 = None
        add_334: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_238, arg1102_1);  mul_238 = arg1102_1 = None
        view_956: "bf16[32768, 128]" = torch.ops.aten.view.default(add_334, _shape_param_1077);  _shape_param_1077 = None
        permute_478: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1103_1, [1, 0]);  arg1103_1 = None
        addmm_358: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1104_1, view_956, permute_478);  arg1104_1 = view_956 = permute_478 = None
        view_957: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_358, _shape_param_1078);  addmm_358 = _shape_param_1078 = None
        relu_95: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_957);  view_957 = None
        view_958: "bf16[32768, 512]" = torch.ops.aten.view.default(relu_95, _shape_param_1079);  relu_95 = _shape_param_1079 = None
        permute_479: "bf16[512, 128]" = torch.ops.aten.permute.default(arg1105_1, [1, 0]);  arg1105_1 = None
        addmm_359: "bf16[32768, 128]" = torch.ops.aten.addmm.default(arg1106_1, view_958, permute_479);  arg1106_1 = view_958 = permute_479 = None
        view_959: "bf16[256, 128, 128]" = torch.ops.aten.view.default(addmm_359, _shape_param_1080);  addmm_359 = _shape_param_1080 = None
        add_335: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(view_959, add_334);  view_959 = add_334 = None
        mul_239: "bf16[256, 128, 128]" = torch.ops.aten.mul.Tensor(add_335, arg1107_1);  add_335 = arg1107_1 = None
        add_336: "bf16[256, 128, 128]" = torch.ops.aten.add.Tensor(mul_239, arg1108_1);  mul_239 = arg1108_1 = None
        view_960: "bf16[32768, 128]" = torch.ops.aten.view.default(add_336, _shape_param_1081);  add_336 = _shape_param_1081 = None
        permute_480: "bf16[128, 512]" = torch.ops.aten.permute.default(arg1109_1, [1, 0]);  arg1109_1 = None
        addmm_360: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1110_1, view_960, permute_480);  arg1110_1 = view_960 = permute_480 = None
        view_961: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_360, _shape_param_1082);  addmm_360 = _shape_param_1082 = None
        add_337: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(view_961, add_324);  view_961 = add_324 = None
        mul_240: "bf16[256, 128, 512]" = torch.ops.aten.mul.Tensor(add_337, arg1111_1);  add_337 = arg1111_1 = None
        add_338: "bf16[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_240, arg1112_1);  mul_240 = arg1112_1 = None
        view_962: "bf16[32768, 512]" = torch.ops.aten.view.default(add_338, _shape_param_1083);  add_338 = _shape_param_1083 = None
        permute_481: "bf16[512, 512]" = torch.ops.aten.permute.default(arg1113_1, [1, 0]);  arg1113_1 = None
        addmm_361: "bf16[32768, 512]" = torch.ops.aten.addmm.default(arg1114_1, view_962, permute_481);  arg1114_1 = view_962 = permute_481 = None
        view_963: "bf16[256, 128, 512]" = torch.ops.aten.view.default(addmm_361, _shape_param_1084);  addmm_361 = _shape_param_1084 = None
        relu_96: "bf16[256, 128, 512]" = torch.ops.aten.relu.default(view_963);  view_963 = None
        convert_element_type_48: "f32[256, 128, 512]" = torch.ops.prims.convert_element_type.default(relu_96, torch.float32);  relu_96 = None
        var_mean = torch.ops.aten.var_mean.correction(convert_element_type_48, [2], correction = 0, keepdim = True)
        getitem: "f32[256, 128, 1]" = var_mean[0]
        getitem_1: "f32[256, 128, 1]" = var_mean[1];  var_mean = None
        view_964: "i64[32768]" = torch.ops.aten.view.default(arg1115_1, [-1]);  arg1115_1 = None
        ne: "b8[32768]" = torch.ops.aten.ne.Scalar(view_964, -100)
        sub_24: "f32[256, 128, 512]" = torch.ops.aten.sub.Tensor(convert_element_type_48, getitem_1);  convert_element_type_48 = getitem_1 = None
        add_339: "f32[256, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-12);  getitem = None
        rsqrt: "f32[256, 128, 1]" = torch.ops.aten.rsqrt.default(add_339);  add_339 = None
        mul_241: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt);  sub_24 = rsqrt = None
        mul_242: "f32[256, 128, 512]" = torch.ops.aten.mul.Tensor(mul_241, arg1116_1);  mul_241 = arg1116_1 = None
        add_340: "f32[256, 128, 512]" = torch.ops.aten.add.Tensor(mul_242, arg1117_1);  mul_242 = arg1117_1 = None
        convert_element_type_49: "bf16[256, 128, 512]" = torch.ops.prims.convert_element_type.default(add_340, torch.bfloat16);  add_340 = None
        view_965: "bf16[32768, 512]" = torch.ops.aten.view.default(convert_element_type_49, _shape_param_1085);  convert_element_type_49 = _shape_param_1085 = None
        permute_482: "bf16[128, 30522]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        cat_1: "bf16[512, 30522]" = torch.ops.aten.cat.default([permute_482, arg1118_1]);  permute_482 = arg1118_1 = None
        constant_pad_nd_2: "bf16[512, 30528]" = torch.ops.aten.constant_pad_nd.default(cat_1, _shape_param_1086);  cat_1 = _shape_param_1086 = None
        mm: "bf16[32768, 30528]" = torch.ops.aten.mm.default(view_965, constant_pad_nd_2);  view_965 = constant_pad_nd_2 = None
        slice_4: "bf16[32768, 30522]" = torch.ops.aten.slice.Tensor(mm, 1, 0, -6);  mm = None
        view_966: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(slice_4, _shape_param_1087);  slice_4 = _shape_param_1087 = None
        add_341: "bf16[256, 128, 30522]" = torch.ops.aten.add.Tensor(view_966, arg1119_1);  view_966 = arg1119_1 = None
        view_967: "bf16[32768, 30522]" = torch.ops.aten.view.default(add_341, _shape_param_1088);  add_341 = _shape_param_1088 = None
        convert_element_type_50: "f32[32768, 30522]" = torch.ops.prims.convert_element_type.default(view_967, torch.float32)
        amax_24: "f32[32768, 1]" = torch.ops.aten.amax.default(convert_element_type_50, [1], True)
        sub_25: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(convert_element_type_50, amax_24);  convert_element_type_50 = amax_24 = None
        exp_24: "f32[32768, 30522]" = torch.ops.aten.exp.default(sub_25)
        sum_25: "f32[32768, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [1], True);  exp_24 = None
        log: "f32[32768, 1]" = torch.ops.aten.log.default(sum_25);  sum_25 = None
        sub_26: "f32[32768, 30522]" = torch.ops.aten.sub.Tensor(sub_25, log);  sub_25 = log = None
        convert_element_type_51: "bf16[32768, 30522]" = torch.ops.prims.convert_element_type.default(sub_26, torch.bfloat16);  sub_26 = None
        ne_1: "b8[32768]" = torch.ops.aten.ne.Scalar(view_964, -100)
        full_25: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_24: "i64[32768]" = torch.ops.aten.where.self(ne_1, view_964, full_25);  ne_1 = full_25 = None
        unsqueeze: "i64[32768, 1]" = torch.ops.aten.unsqueeze.default(where_24, 1);  where_24 = None
        gather: "bf16[32768, 1]" = torch.ops.aten.gather.default(convert_element_type_51, 1, unsqueeze);  convert_element_type_51 = unsqueeze = None
        squeeze: "bf16[32768]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "bf16[32768]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_26: "bf16[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "bf16[32768]" = torch.ops.aten.where.self(ne, neg, full_26);  ne = neg = full_26 = None
        sum_26: "bf16[]" = torch.ops.aten.sum.default(where_25);  where_25 = None
        ne_2: "b8[32768]" = torch.ops.aten.ne.Scalar(view_964, -100);  view_964 = None
        sum_27: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type_52: "bf16[]" = torch.ops.prims.convert_element_type.default(sum_27, torch.bfloat16);  sum_27 = None
        div_24: "bf16[]" = torch.ops.aten.div.Tensor(sum_26, convert_element_type_52);  sum_26 = convert_element_type_52 = None
        view_968: "bf16[256, 128, 30522]" = torch.ops.aten.view.default(view_967, _shape_param_1089);  view_967 = _shape_param_1089 = None
        return (div_24, view_968)



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

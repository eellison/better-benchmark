import torch
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[16, 128]", arg1_1: "i64[16, 128]", arg2_1: "f32[8008, 2560]", arg3_1: "f32[128, 2560]", arg4_1: "f32[2560]", arg5_1: "f32[2560]", arg6_1: "f32[2560, 2560]", arg7_1: "f32[2560]", arg8_1: "f32[2560, 2560]", arg9_1: "f32[2560]", arg10_1: "f32[2560, 2560]", arg11_1: "f32[2560]", arg12_1: "f32[2560, 2560]", arg13_1: "f32[2560]", arg14_1: "f32[2560]", arg15_1: "f32[2560]", arg16_1: "f32[10240, 2560]", arg17_1: "f32[10240]", arg18_1: "f32[2560, 10240]", arg19_1: "f32[2560]", arg20_1: "f32[2560]", arg21_1: "f32[2560]", arg22_1: "f32[2560, 2560]", arg23_1: "f32[2560]", arg24_1: "f32[2560, 2560]", arg25_1: "f32[2560]", arg26_1: "f32[2560, 2560]", arg27_1: "f32[2560]", arg28_1: "f32[2560, 2560]", arg29_1: "f32[2560]", arg30_1: "f32[2560]", arg31_1: "f32[2560]", arg32_1: "f32[10240, 2560]", arg33_1: "f32[10240]", arg34_1: "f32[2560, 10240]", arg35_1: "f32[2560]", arg36_1: "f32[2560]", arg37_1: "f32[2560]", arg38_1: "f32[128, 2560]", arg39_1: "f32[2560]", arg40_1: "f32[2560]", arg41_1: "f32[2560, 2560]", arg42_1: "f32[2560]", arg43_1: "f32[2560, 2560]", arg44_1: "f32[2560]", arg45_1: "f32[2560, 2560]", arg46_1: "f32[2560]", arg47_1: "f32[2560, 2560]", arg48_1: "f32[2560]", arg49_1: "f32[2560]", arg50_1: "f32[2560]", arg51_1: "f32[2560, 2560]", arg52_1: "f32[2560]", arg53_1: "f32[2560, 2560]", arg54_1: "f32[2560]", arg55_1: "f32[2560, 2560]", arg56_1: "f32[2560]", arg57_1: "f32[2560, 2560]", arg58_1: "f32[2560]", arg59_1: "f32[2560]", arg60_1: "f32[2560]", arg61_1: "f32[10240, 2560]", arg62_1: "f32[10240]", arg63_1: "f32[2560, 10240]", arg64_1: "f32[2560]", arg65_1: "f32[2560]", arg66_1: "f32[2560]", arg67_1: "f32[2560, 2560]", arg68_1: "f32[2560]", arg69_1: "f32[2560, 2560]", arg70_1: "f32[2560]", arg71_1: "f32[2560, 2560]", arg72_1: "f32[2560]", arg73_1: "f32[2560, 2560]", arg74_1: "f32[2560]", arg75_1: "f32[2560]", arg76_1: "f32[2560]", arg77_1: "f32[2560, 2560]", arg78_1: "f32[2560]", arg79_1: "f32[2560, 2560]", arg80_1: "f32[2560]", arg81_1: "f32[2560, 2560]", arg82_1: "f32[2560]", arg83_1: "f32[2560, 2560]", arg84_1: "f32[2560]", arg85_1: "f32[2560]", arg86_1: "f32[2560]", arg87_1: "f32[10240, 2560]", arg88_1: "f32[10240]", arg89_1: "f32[2560, 10240]", arg90_1: "f32[2560]", arg91_1: "f32[2560]", arg92_1: "f32[2560]", arg93_1: "f32[2560, 2560]", arg94_1: "f32[2560]", arg95_1: "f32[2560, 2560]", arg96_1: "f32[2560]", arg97_1: "f32[2560, 2560]", arg98_1: "f32[2560]", arg99_1: "f32[2560, 2560]", arg100_1: "f32[2560]", arg101_1: "f32[2560]", arg102_1: "f32[2560]", arg103_1: "f32[2560, 2560]", arg104_1: "f32[2560]", arg105_1: "f32[2560, 2560]", arg106_1: "f32[2560]", arg107_1: "f32[2560, 2560]", arg108_1: "f32[2560]", arg109_1: "f32[2560, 2560]", arg110_1: "f32[2560]", arg111_1: "f32[2560]", arg112_1: "f32[2560]", arg113_1: "f32[10240, 2560]", arg114_1: "f32[10240]", arg115_1: "f32[2560, 10240]", arg116_1: "f32[2560]", arg117_1: "f32[2560]", arg118_1: "f32[2560]", arg119_1: "f32[2560, 2560]", arg120_1: "f32[2560]", arg121_1: "f32[2560, 2560]", arg122_1: "f32[2560]", arg123_1: "f32[2560, 2560]", arg124_1: "f32[2560]", arg125_1: "f32[2560, 2560]", arg126_1: "f32[2560]", arg127_1: "f32[2560]", arg128_1: "f32[2560]", arg129_1: "f32[2560, 2560]", arg130_1: "f32[2560]", arg131_1: "f32[2560, 2560]", arg132_1: "f32[2560]", arg133_1: "f32[2560, 2560]", arg134_1: "f32[2560]", arg135_1: "f32[2560, 2560]", arg136_1: "f32[2560]", arg137_1: "f32[2560]", arg138_1: "f32[2560]", arg139_1: "f32[10240, 2560]", arg140_1: "f32[10240]", arg141_1: "f32[2560, 10240]", arg142_1: "f32[2560]", arg143_1: "f32[2560]", arg144_1: "f32[2560]", arg145_1: "f32[2560, 2560]", arg146_1: "f32[2560]", arg147_1: "f32[2560, 2560]", arg148_1: "f32[2560]", arg149_1: "f32[2560, 2560]", arg150_1: "f32[2560]", arg151_1: "f32[2560, 2560]", arg152_1: "f32[2560]", arg153_1: "f32[2560]", arg154_1: "f32[2560]", arg155_1: "f32[2560, 2560]", arg156_1: "f32[2560]", arg157_1: "f32[2560, 2560]", arg158_1: "f32[2560]", arg159_1: "f32[2560, 2560]", arg160_1: "f32[2560]", arg161_1: "f32[2560, 2560]", arg162_1: "f32[2560]", arg163_1: "f32[2560]", arg164_1: "f32[2560]", arg165_1: "f32[10240, 2560]", arg166_1: "f32[10240]", arg167_1: "f32[2560, 10240]", arg168_1: "f32[2560]", arg169_1: "f32[2560]", arg170_1: "f32[2560]", arg171_1: "f32[2560, 2560]", arg172_1: "f32[2560]", arg173_1: "f32[2560, 2560]", arg174_1: "f32[2560]", arg175_1: "f32[2560, 2560]", arg176_1: "f32[2560]", arg177_1: "f32[2560, 2560]", arg178_1: "f32[2560]", arg179_1: "f32[2560]", arg180_1: "f32[2560]", arg181_1: "f32[2560, 2560]", arg182_1: "f32[2560]", arg183_1: "f32[2560, 2560]", arg184_1: "f32[2560]", arg185_1: "f32[2560, 2560]", arg186_1: "f32[2560]", arg187_1: "f32[2560, 2560]", arg188_1: "f32[2560]", arg189_1: "f32[2560]", arg190_1: "f32[2560]", arg191_1: "f32[10240, 2560]", arg192_1: "f32[10240]", arg193_1: "f32[2560, 10240]", arg194_1: "f32[2560]", arg195_1: "f32[2560]", arg196_1: "f32[2560]", arg197_1: "f32[2560, 2560]", arg198_1: "f32[2560]", arg199_1: "f32[2560, 2560]", arg200_1: "f32[2560]", arg201_1: "f32[2560, 2560]", arg202_1: "f32[2560]", arg203_1: "f32[2560, 2560]", arg204_1: "f32[2560]", arg205_1: "f32[2560]", arg206_1: "f32[2560]", arg207_1: "f32[2560, 2560]", arg208_1: "f32[2560]", arg209_1: "f32[2560, 2560]", arg210_1: "f32[2560]", arg211_1: "f32[2560, 2560]", arg212_1: "f32[2560]", arg213_1: "f32[2560, 2560]", arg214_1: "f32[2560]", arg215_1: "f32[2560]", arg216_1: "f32[2560]", arg217_1: "f32[10240, 2560]", arg218_1: "f32[10240]", arg219_1: "f32[2560, 10240]", arg220_1: "f32[2560]", arg221_1: "f32[2560]", arg222_1: "f32[2560]", arg223_1: "f32[2560, 2560]", arg224_1: "f32[2560]", arg225_1: "f32[2560, 2560]", arg226_1: "f32[2560]", arg227_1: "f32[2560, 2560]", arg228_1: "f32[2560]", arg229_1: "f32[2560, 2560]", arg230_1: "f32[2560]", arg231_1: "f32[2560]", arg232_1: "f32[2560]", arg233_1: "f32[2560, 2560]", arg234_1: "f32[2560]", arg235_1: "f32[2560, 2560]", arg236_1: "f32[2560]", arg237_1: "f32[2560, 2560]", arg238_1: "f32[2560]", arg239_1: "f32[2560, 2560]", arg240_1: "f32[2560]", arg241_1: "f32[2560]", arg242_1: "f32[2560]", arg243_1: "f32[10240, 2560]", arg244_1: "f32[10240]", arg245_1: "f32[2560, 10240]", arg246_1: "f32[2560]", arg247_1: "f32[2560]", arg248_1: "f32[2560]", arg249_1: "f32[2560, 2560]", arg250_1: "f32[2560]", arg251_1: "f32[2560, 2560]", arg252_1: "f32[2560]", arg253_1: "f32[2560, 2560]", arg254_1: "f32[2560]", arg255_1: "f32[2560, 2560]", arg256_1: "f32[2560]", arg257_1: "f32[2560]", arg258_1: "f32[2560]", arg259_1: "f32[2560, 2560]", arg260_1: "f32[2560]", arg261_1: "f32[2560, 2560]", arg262_1: "f32[2560]", arg263_1: "f32[2560, 2560]", arg264_1: "f32[2560]", arg265_1: "f32[2560, 2560]", arg266_1: "f32[2560]", arg267_1: "f32[2560]", arg268_1: "f32[2560]", arg269_1: "f32[10240, 2560]", arg270_1: "f32[10240]", arg271_1: "f32[2560, 10240]", arg272_1: "f32[2560]", arg273_1: "f32[2560]", arg274_1: "f32[2560]", arg275_1: "f32[2560, 2560]", arg276_1: "f32[2560]", arg277_1: "f32[2560, 2560]", arg278_1: "f32[2560]", arg279_1: "f32[2560, 2560]", arg280_1: "f32[2560]", arg281_1: "f32[2560, 2560]", arg282_1: "f32[2560]", arg283_1: "f32[2560]", arg284_1: "f32[2560]", arg285_1: "f32[2560, 2560]", arg286_1: "f32[2560]", arg287_1: "f32[2560, 2560]", arg288_1: "f32[2560]", arg289_1: "f32[2560, 2560]", arg290_1: "f32[2560]", arg291_1: "f32[2560, 2560]", arg292_1: "f32[2560]", arg293_1: "f32[2560]", arg294_1: "f32[2560]", arg295_1: "f32[10240, 2560]", arg296_1: "f32[10240]", arg297_1: "f32[2560, 10240]", arg298_1: "f32[2560]", arg299_1: "f32[2560]", arg300_1: "f32[2560]", arg301_1: "f32[2560, 2560]", arg302_1: "f32[2560]", arg303_1: "f32[2560, 2560]", arg304_1: "f32[2560]", arg305_1: "f32[2560, 2560]", arg306_1: "f32[2560]", arg307_1: "f32[2560, 2560]", arg308_1: "f32[2560]", arg309_1: "f32[2560]", arg310_1: "f32[2560]", arg311_1: "f32[2560, 2560]", arg312_1: "f32[2560]", arg313_1: "f32[2560, 2560]", arg314_1: "f32[2560]", arg315_1: "f32[2560, 2560]", arg316_1: "f32[2560]", arg317_1: "f32[2560, 2560]", arg318_1: "f32[2560]", arg319_1: "f32[2560]", arg320_1: "f32[2560]", arg321_1: "f32[10240, 2560]", arg322_1: "f32[10240]", arg323_1: "f32[2560, 10240]", arg324_1: "f32[2560]", arg325_1: "f32[2560]", arg326_1: "f32[2560]", arg327_1: "f32[2560, 2560]", arg328_1: "f32[2560]", arg329_1: "f32[2560, 2560]", arg330_1: "f32[2560]", arg331_1: "f32[2560, 2560]", arg332_1: "f32[2560]", arg333_1: "f32[2560, 2560]", arg334_1: "f32[2560]", arg335_1: "f32[2560]", arg336_1: "f32[2560]", arg337_1: "f32[2560, 2560]", arg338_1: "f32[2560]", arg339_1: "f32[2560, 2560]", arg340_1: "f32[2560]", arg341_1: "f32[2560, 2560]", arg342_1: "f32[2560]", arg343_1: "f32[2560, 2560]", arg344_1: "f32[2560]", arg345_1: "f32[2560]", arg346_1: "f32[2560]", arg347_1: "f32[10240, 2560]", arg348_1: "f32[10240]", arg349_1: "f32[2560, 10240]", arg350_1: "f32[2560]", arg351_1: "f32[2560]", arg352_1: "f32[2560]", arg353_1: "f32[2560, 2560]", arg354_1: "f32[2560]", arg355_1: "f32[2560, 2560]", arg356_1: "f32[2560]", arg357_1: "f32[2560, 2560]", arg358_1: "f32[2560]", arg359_1: "f32[2560, 2560]", arg360_1: "f32[2560]", arg361_1: "f32[2560]", arg362_1: "f32[2560]", arg363_1: "f32[2560, 2560]", arg364_1: "f32[2560]", arg365_1: "f32[2560, 2560]", arg366_1: "f32[2560]", arg367_1: "f32[2560, 2560]", arg368_1: "f32[2560]", arg369_1: "f32[2560, 2560]", arg370_1: "f32[2560]", arg371_1: "f32[2560]", arg372_1: "f32[2560]", arg373_1: "f32[10240, 2560]", arg374_1: "f32[10240]", arg375_1: "f32[2560, 10240]", arg376_1: "f32[2560]", arg377_1: "f32[2560]", arg378_1: "f32[2560]", arg379_1: "f32[2560, 2560]", arg380_1: "f32[2560]", arg381_1: "f32[2560, 2560]", arg382_1: "f32[2560]", arg383_1: "f32[2560, 2560]", arg384_1: "f32[2560]", arg385_1: "f32[2560, 2560]", arg386_1: "f32[2560]", arg387_1: "f32[2560]", arg388_1: "f32[2560]", arg389_1: "f32[2560, 2560]", arg390_1: "f32[2560]", arg391_1: "f32[2560, 2560]", arg392_1: "f32[2560]", arg393_1: "f32[2560, 2560]", arg394_1: "f32[2560]", arg395_1: "f32[2560, 2560]", arg396_1: "f32[2560]", arg397_1: "f32[2560]", arg398_1: "f32[2560]", arg399_1: "f32[10240, 2560]", arg400_1: "f32[10240]", arg401_1: "f32[2560, 10240]", arg402_1: "f32[2560]", arg403_1: "f32[2560]", arg404_1: "f32[2560]", arg405_1: "f32[2560, 2560]", arg406_1: "f32[2560]", arg407_1: "f32[2560, 2560]", arg408_1: "f32[2560]", arg409_1: "f32[2560, 2560]", arg410_1: "f32[2560]", arg411_1: "f32[2560, 2560]", arg412_1: "f32[2560]", arg413_1: "f32[2560]", arg414_1: "f32[2560]", arg415_1: "f32[2560, 2560]", arg416_1: "f32[2560]", arg417_1: "f32[2560, 2560]", arg418_1: "f32[2560]", arg419_1: "f32[2560, 2560]", arg420_1: "f32[2560]", arg421_1: "f32[2560, 2560]", arg422_1: "f32[2560]", arg423_1: "f32[2560]", arg424_1: "f32[2560]", arg425_1: "f32[10240, 2560]", arg426_1: "f32[10240]", arg427_1: "f32[2560, 10240]", arg428_1: "f32[2560]", arg429_1: "f32[2560]", arg430_1: "f32[2560]", arg431_1: "f32[2560, 2560]", arg432_1: "f32[2560]", arg433_1: "f32[2560, 2560]", arg434_1: "f32[2560]", arg435_1: "f32[2560, 2560]", arg436_1: "f32[2560]", arg437_1: "f32[2560, 2560]", arg438_1: "f32[2560]", arg439_1: "f32[2560]", arg440_1: "f32[2560]", arg441_1: "f32[2560, 2560]", arg442_1: "f32[2560]", arg443_1: "f32[2560, 2560]", arg444_1: "f32[2560]", arg445_1: "f32[2560, 2560]", arg446_1: "f32[2560]", arg447_1: "f32[2560, 2560]", arg448_1: "f32[2560]", arg449_1: "f32[2560]", arg450_1: "f32[2560]", arg451_1: "f32[10240, 2560]", arg452_1: "f32[10240]", arg453_1: "f32[2560, 10240]", arg454_1: "f32[2560]", arg455_1: "f32[2560]", arg456_1: "f32[2560]", arg457_1: "f32[2560, 2560]", arg458_1: "f32[2560]", arg459_1: "f32[2560, 2560]", arg460_1: "f32[2560]", arg461_1: "f32[2560, 2560]", arg462_1: "f32[2560]", arg463_1: "f32[2560, 2560]", arg464_1: "f32[2560]", arg465_1: "f32[2560]", arg466_1: "f32[2560]", arg467_1: "f32[2560, 2560]", arg468_1: "f32[2560]", arg469_1: "f32[2560, 2560]", arg470_1: "f32[2560]", arg471_1: "f32[2560, 2560]", arg472_1: "f32[2560]", arg473_1: "f32[2560, 2560]", arg474_1: "f32[2560]", arg475_1: "f32[2560]", arg476_1: "f32[2560]", arg477_1: "f32[10240, 2560]", arg478_1: "f32[10240]", arg479_1: "f32[2560, 10240]", arg480_1: "f32[2560]", arg481_1: "f32[2560]", arg482_1: "f32[2560]", arg483_1: "f32[2560, 2560]", arg484_1: "f32[2560]", arg485_1: "f32[2560, 2560]", arg486_1: "f32[2560]", arg487_1: "f32[2560, 2560]", arg488_1: "f32[2560]", arg489_1: "f32[2560, 2560]", arg490_1: "f32[2560]", arg491_1: "f32[2560]", arg492_1: "f32[2560]", arg493_1: "f32[2560, 2560]", arg494_1: "f32[2560]", arg495_1: "f32[2560, 2560]", arg496_1: "f32[2560]", arg497_1: "f32[2560, 2560]", arg498_1: "f32[2560]", arg499_1: "f32[2560, 2560]", arg500_1: "f32[2560]", arg501_1: "f32[2560]", arg502_1: "f32[2560]", arg503_1: "f32[10240, 2560]", arg504_1: "f32[10240]", arg505_1: "f32[2560, 10240]", arg506_1: "f32[2560]", arg507_1: "f32[2560]", arg508_1: "f32[2560]", arg509_1: "f32[2560, 2560]", arg510_1: "f32[2560]", arg511_1: "f32[2560, 2560]", arg512_1: "f32[2560]", arg513_1: "f32[2560, 2560]", arg514_1: "f32[2560]", arg515_1: "f32[2560, 2560]", arg516_1: "f32[2560]", arg517_1: "f32[2560]", arg518_1: "f32[2560]", arg519_1: "f32[2560, 2560]", arg520_1: "f32[2560]", arg521_1: "f32[2560, 2560]", arg522_1: "f32[2560]", arg523_1: "f32[2560, 2560]", arg524_1: "f32[2560]", arg525_1: "f32[2560, 2560]", arg526_1: "f32[2560]", arg527_1: "f32[2560]", arg528_1: "f32[2560]", arg529_1: "f32[10240, 2560]", arg530_1: "f32[10240]", arg531_1: "f32[2560, 10240]", arg532_1: "f32[2560]", arg533_1: "f32[2560]", arg534_1: "f32[2560]", arg535_1: "f32[2560, 2560]", arg536_1: "f32[2560]", arg537_1: "f32[2560, 2560]", arg538_1: "f32[2560]", arg539_1: "f32[2560, 2560]", arg540_1: "f32[2560]", arg541_1: "f32[2560, 2560]", arg542_1: "f32[2560]", arg543_1: "f32[2560]", arg544_1: "f32[2560]", arg545_1: "f32[2560, 2560]", arg546_1: "f32[2560]", arg547_1: "f32[2560, 2560]", arg548_1: "f32[2560]", arg549_1: "f32[2560, 2560]", arg550_1: "f32[2560]", arg551_1: "f32[2560, 2560]", arg552_1: "f32[2560]", arg553_1: "f32[2560]", arg554_1: "f32[2560]", arg555_1: "f32[10240, 2560]", arg556_1: "f32[10240]", arg557_1: "f32[2560, 10240]", arg558_1: "f32[2560]", arg559_1: "f32[2560]", arg560_1: "f32[2560]", arg561_1: "f32[2560, 2560]", arg562_1: "f32[2560]", arg563_1: "f32[2560, 2560]", arg564_1: "f32[2560]", arg565_1: "f32[2560, 2560]", arg566_1: "f32[2560]", arg567_1: "f32[2560, 2560]", arg568_1: "f32[2560]", arg569_1: "f32[2560]", arg570_1: "f32[2560]", arg571_1: "f32[2560, 2560]", arg572_1: "f32[2560]", arg573_1: "f32[2560, 2560]", arg574_1: "f32[2560]", arg575_1: "f32[2560, 2560]", arg576_1: "f32[2560]", arg577_1: "f32[2560, 2560]", arg578_1: "f32[2560]", arg579_1: "f32[2560]", arg580_1: "f32[2560]", arg581_1: "f32[10240, 2560]", arg582_1: "f32[10240]", arg583_1: "f32[2560, 10240]", arg584_1: "f32[2560]", arg585_1: "f32[2560]", arg586_1: "f32[2560]", arg587_1: "f32[2560, 2560]", arg588_1: "f32[2560]", arg589_1: "f32[2560, 2560]", arg590_1: "f32[2560]", arg591_1: "f32[2560, 2560]", arg592_1: "f32[2560]", arg593_1: "f32[2560, 2560]", arg594_1: "f32[2560]", arg595_1: "f32[2560]", arg596_1: "f32[2560]", arg597_1: "f32[2560, 2560]", arg598_1: "f32[2560]", arg599_1: "f32[2560, 2560]", arg600_1: "f32[2560]", arg601_1: "f32[2560, 2560]", arg602_1: "f32[2560]", arg603_1: "f32[2560, 2560]", arg604_1: "f32[2560]", arg605_1: "f32[2560]", arg606_1: "f32[2560]", arg607_1: "f32[10240, 2560]", arg608_1: "f32[10240]", arg609_1: "f32[2560, 10240]", arg610_1: "f32[2560]", arg611_1: "f32[2560]", arg612_1: "f32[2560]", arg613_1: "f32[2560, 2560]", arg614_1: "f32[2560]", arg615_1: "f32[2560, 2560]", arg616_1: "f32[2560]", arg617_1: "f32[2560, 2560]", arg618_1: "f32[2560]", arg619_1: "f32[2560, 2560]", arg620_1: "f32[2560]", arg621_1: "f32[2560]", arg622_1: "f32[2560]", arg623_1: "f32[2560, 2560]", arg624_1: "f32[2560]", arg625_1: "f32[2560, 2560]", arg626_1: "f32[2560]", arg627_1: "f32[2560, 2560]", arg628_1: "f32[2560]", arg629_1: "f32[2560, 2560]", arg630_1: "f32[2560]", arg631_1: "f32[2560]", arg632_1: "f32[2560]", arg633_1: "f32[10240, 2560]", arg634_1: "f32[10240]", arg635_1: "f32[2560, 10240]", arg636_1: "f32[2560]", arg637_1: "f32[2560]", arg638_1: "f32[2560]", arg639_1: "f32[2560, 2560]", arg640_1: "f32[2560]", arg641_1: "f32[2560, 2560]", arg642_1: "f32[2560]", arg643_1: "f32[2560, 2560]", arg644_1: "f32[2560]", arg645_1: "f32[2560, 2560]", arg646_1: "f32[2560]", arg647_1: "f32[2560]", arg648_1: "f32[2560]", arg649_1: "f32[2560, 2560]", arg650_1: "f32[2560]", arg651_1: "f32[2560, 2560]", arg652_1: "f32[2560]", arg653_1: "f32[2560, 2560]", arg654_1: "f32[2560]", arg655_1: "f32[2560, 2560]", arg656_1: "f32[2560]", arg657_1: "f32[2560]", arg658_1: "f32[2560]", arg659_1: "f32[10240, 2560]", arg660_1: "f32[10240]", arg661_1: "f32[2560, 10240]", arg662_1: "f32[2560]", arg663_1: "f32[2560]", arg664_1: "f32[2560]", arg665_1: "f32[1, 8008]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding: "f32[16, 128, 2560]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0)
        mul: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding, 1.0);  embedding = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:79 in forward, code: position_ids = torch.arange(
        iota: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_1: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg3_1, iota);  arg3_1 = iota = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:484 in forward, code: hidden_states = inputs_embeds + embed_pos
        add: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul, embedding_1);  mul = embedding_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean = torch.ops.aten.var_mean.correction(add, [2], correction = 0, keepdim = True)
        getitem: "f32[16, 128, 1]" = var_mean[0]
        getitem_1: "f32[16, 128, 1]" = var_mean[1];  var_mean = None
        sub: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add, getitem_1);  getitem_1 = None
        add_3: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem, 1e-05);  getitem = None
        rsqrt: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_1: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub, rsqrt);  sub = rsqrt = None
        mul_2: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_1, arg4_1);  mul_1 = arg4_1 = None
        add_4: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_2, arg5_1);  mul_2 = arg5_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_4, [2048, 2560])
        permute: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        addmm: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg7_1, view, permute);  arg7_1 = view = permute = None
        view_1: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm, [16, 128, 2560]);  addmm = None
        view_2: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_1, [16, 128, -1, 80]);  view_1 = None
        permute_1: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_3: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_1, 0.334370152488211);  permute_1 = None
        expand_1: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_3, [16, 32, 128, 80]);  mul_3 = None
        clone_1: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_9: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_1, [512, 128, 80]);  clone_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_3: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_4, [2048, 2560])
        permute_2: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg8_1, [1, 0]);  arg8_1 = None
        addmm_1: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg9_1, view_3, permute_2);  arg9_1 = view_3 = permute_2 = None
        view_4: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_1, [16, 128, 2560]);  addmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_7: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_4, [16, 128, -1, 80]);  view_4 = None
        permute_4: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_7, [0, 2, 1, 3]);  view_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_6: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_4, [0, 1, 3, 2]);  permute_4 = None
        mul_4: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_6, 0.334370152488211);  permute_6 = None
        expand_2: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_4, [16, 32, 80, 128]);  mul_4 = None
        clone_2: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_2, memory_format = torch.contiguous_format);  expand_2 = None
        view_10: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_2, [512, 80, 128]);  clone_2 = None
        bmm: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_9, view_10);  view_9 = view_10 = None
        view_11: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm, [16, 32, 128, 128]);  bmm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_3: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_1: "i64[128]" = torch.ops.aten.add.Tensor(iota_3, 0);  iota_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_2, 0);  unsqueeze_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[16, 1, 128, 128]" = torch.ops.aten.expand.default(ge, [16, -1, 128, 128]);  ge = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_1: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_1, full_default);  full_default_1 = full_default = None
        add_5: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_11, where);  view_11 = where = None
        eq: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_5, -inf)
        logical_not: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq);  eq = None
        any_1: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not, -1, True);  logical_not = None
        logical_not_1: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_1);  any_1 = None
        full_default_2: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_5, [-1], True)
        sub_1: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_5, amax);  add_5 = amax = None
        exp: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_1);  sub_1 = None
        sum_1: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        where_1: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_1, full_default_2, div);  logical_not_1 = full_default_2 = div = None
        expand_3: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_1, [16, 32, 128, 128]);  where_1 = None
        view_12: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_3, [512, 128, 128]);  expand_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_5: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_4, [2048, 2560]);  add_4 = None
        permute_3: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        addmm_2: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg11_1, view_5, permute_3);  arg11_1 = view_5 = permute_3 = None
        view_6: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_2, [16, 128, 2560]);  addmm_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_8: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_6, [16, 128, -1, 80]);  view_6 = None
        permute_5: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_8, [0, 2, 1, 3]);  view_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_4: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_5, [16, 32, 128, 80]);  permute_5 = None
        clone_3: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_4, memory_format = torch.contiguous_format);  expand_4 = None
        view_13: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_3, [512, 128, 80]);  clone_3 = None
        bmm_1: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_12, view_13);  view_12 = view_13 = None
        view_14: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_1, [16, 32, 128, 80]);  bmm_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_14, [0, 2, 1, 3]);  view_14 = None
        clone_4: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_15: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_4, [16, 128, -1]);  clone_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_16: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_15, [2048, 2560]);  view_15 = None
        permute_8: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        addmm_3: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg13_1, view_16, permute_8);  arg13_1 = view_16 = permute_8 = None
        view_17: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_3, [16, 128, 2560]);  addmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_6: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add, view_17);  add = view_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_1 = torch.ops.aten.var_mean.correction(add_6, [2], correction = 0, keepdim = True)
        getitem_2: "f32[16, 128, 1]" = var_mean_1[0]
        getitem_3: "f32[16, 128, 1]" = var_mean_1[1];  var_mean_1 = None
        sub_2: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_6, getitem_3);  getitem_3 = None
        add_7: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_2, 1e-05);  getitem_2 = None
        rsqrt_1: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_7);  add_7 = None
        mul_5: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_2, rsqrt_1);  sub_2 = rsqrt_1 = None
        mul_6: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_5, arg14_1);  mul_5 = arg14_1 = None
        add_8: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_6, arg15_1);  mul_6 = arg15_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_18: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_8, [2048, 2560]);  add_8 = None
        permute_9: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg16_1, [1, 0]);  arg16_1 = None
        addmm_4: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg17_1, view_18, permute_9);  arg17_1 = view_18 = permute_9 = None
        view_19: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_4, [16, 128, 10240]);  addmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_7: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_19, 0.5)
        mul_8: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_19, 0.7071067811865476);  view_19 = None
        erf: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_8);  mul_8 = None
        add_9: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_9: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_7, add_9);  mul_7 = add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        view_20: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_9, [2048, 10240]);  mul_9 = None
        permute_10: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg18_1, [1, 0]);  arg18_1 = None
        addmm_5: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg19_1, view_20, permute_10);  arg19_1 = view_20 = permute_10 = None
        view_21: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_5, [16, 128, 2560]);  addmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_10: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_6, view_21);  add_6 = view_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:279 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_2 = torch.ops.aten.var_mean.correction(add_10, [2], correction = 0, keepdim = True)
        getitem_4: "f32[16, 128, 1]" = var_mean_2[0]
        getitem_5: "f32[16, 128, 1]" = var_mean_2[1];  var_mean_2 = None
        sub_3: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_10, getitem_5);  getitem_5 = None
        add_11: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_4, 1e-05);  getitem_4 = None
        rsqrt_2: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_11);  add_11 = None
        mul_10: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_3, rsqrt_2);  sub_3 = rsqrt_2 = None
        mul_11: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_10, arg20_1);  mul_10 = arg20_1 = None
        add_12: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_11, arg21_1);  mul_11 = arg21_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_22: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_12, [2048, 2560])
        permute_11: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg22_1, [1, 0]);  arg22_1 = None
        addmm_6: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg23_1, view_22, permute_11);  arg23_1 = view_22 = permute_11 = None
        view_23: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_6, [16, 128, 2560]);  addmm_6 = None
        view_24: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_23, [16, 128, -1, 80]);  view_23 = None
        permute_12: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_24, [0, 2, 1, 3]);  view_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_12: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_12, 0.334370152488211);  permute_12 = None
        expand_5: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_12, [16, 32, 128, 80]);  mul_12 = None
        clone_8: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_5, memory_format = torch.contiguous_format);  expand_5 = None
        view_31: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_8, [512, 128, 80]);  clone_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_25: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_12, [2048, 2560])
        permute_13: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg24_1, [1, 0]);  arg24_1 = None
        addmm_7: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg25_1, view_25, permute_13);  arg25_1 = view_25 = permute_13 = None
        view_26: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_7, [16, 128, 2560]);  addmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_29: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_26, [16, 128, -1, 80]);  view_26 = None
        permute_15: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_29, [0, 2, 1, 3]);  view_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_17: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_15, [0, 1, 3, 2]);  permute_15 = None
        mul_13: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_17, 0.334370152488211);  permute_17 = None
        expand_6: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_13, [16, 32, 80, 128]);  mul_13 = None
        clone_9: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_6, memory_format = torch.contiguous_format);  expand_6 = None
        view_32: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_9, [512, 80, 128]);  clone_9 = None
        bmm_2: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_31, view_32);  view_31 = view_32 = None
        view_33: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_2, [16, 32, 128, 128]);  bmm_2 = None
        full_default_4: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_3: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_2: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand, full_default_4, full_default_3);  expand = full_default_4 = full_default_3 = None
        add_13: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_33, where_2);  view_33 = where_2 = None
        eq_1: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_13, -inf)
        logical_not_2: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_1);  eq_1 = None
        any_2: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_2, -1, True);  logical_not_2 = None
        logical_not_3: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_2);  any_2 = None
        full_default_5: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_1: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_13, [-1], True)
        sub_4: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_13, amax_1);  add_13 = amax_1 = None
        exp_1: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_4);  sub_4 = None
        sum_2: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_1, [-1], True)
        div_1: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_1, sum_2);  exp_1 = sum_2 = None
        where_3: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_3, full_default_5, div_1);  logical_not_3 = full_default_5 = div_1 = None
        expand_7: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_3, [16, 32, 128, 128]);  where_3 = None
        view_34: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_7, [512, 128, 128]);  expand_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_27: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_12, [2048, 2560]);  add_12 = None
        permute_14: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg26_1, [1, 0]);  arg26_1 = None
        addmm_8: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg27_1, view_27, permute_14);  arg27_1 = view_27 = permute_14 = None
        view_28: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_8, [16, 128, 2560]);  addmm_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_30: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_28, [16, 128, -1, 80]);  view_28 = None
        permute_16: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_30, [0, 2, 1, 3]);  view_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_8: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_16, [16, 32, 128, 80]);  permute_16 = None
        clone_10: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_8, memory_format = torch.contiguous_format);  expand_8 = None
        view_35: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_10, [512, 128, 80]);  clone_10 = None
        bmm_3: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_34, view_35);  view_34 = view_35 = None
        view_36: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_3, [16, 32, 128, 80]);  bmm_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_18: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_36, [0, 2, 1, 3]);  view_36 = None
        clone_11: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_18, memory_format = torch.contiguous_format);  permute_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_37: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_11, [16, 128, -1]);  clone_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_38: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_37, [2048, 2560]);  view_37 = None
        permute_19: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg28_1, [1, 0]);  arg28_1 = None
        addmm_9: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg29_1, view_38, permute_19);  arg29_1 = view_38 = permute_19 = None
        view_39: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_9, [16, 128, 2560]);  addmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:286 in forward, code: hidden_states = residual + hidden_states
        add_14: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_10, view_39);  add_10 = view_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:289 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_3 = torch.ops.aten.var_mean.correction(add_14, [2], correction = 0, keepdim = True)
        getitem_6: "f32[16, 128, 1]" = var_mean_3[0]
        getitem_7: "f32[16, 128, 1]" = var_mean_3[1];  var_mean_3 = None
        sub_5: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_14, getitem_7);  getitem_7 = None
        add_15: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_6, 1e-05);  getitem_6 = None
        rsqrt_3: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_15);  add_15 = None
        mul_14: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_5, rsqrt_3);  sub_5 = rsqrt_3 = None
        mul_15: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_14, arg30_1);  mul_14 = arg30_1 = None
        add_16: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_15, arg31_1);  mul_15 = arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:290 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_40: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_16, [2048, 2560]);  add_16 = None
        permute_20: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg32_1, [1, 0]);  arg32_1 = None
        addmm_10: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg33_1, view_40, permute_20);  arg33_1 = view_40 = permute_20 = None
        view_41: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_10, [16, 128, 10240]);  addmm_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_16: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_41, 0.5)
        mul_17: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_41, 0.7071067811865476);  view_41 = None
        erf_1: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_17);  mul_17 = None
        add_17: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_1, 1);  erf_1 = None
        mul_18: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_16, add_17);  mul_16 = add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:292 in forward, code: hidden_states = self.fc2(hidden_states)
        view_42: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_18, [2048, 10240]);  mul_18 = None
        permute_21: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg34_1, [1, 0]);  arg34_1 = None
        addmm_11: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg35_1, view_42, permute_21);  arg35_1 = view_42 = permute_21 = None
        view_43: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_11, [16, 128, 2560]);  addmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:294 in forward, code: hidden_states = residual + hidden_states
        add_18: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_14, view_43);  add_14 = view_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean_4 = torch.ops.aten.var_mean.correction(add_18, [2], correction = 0, keepdim = True)
        getitem_8: "f32[16, 128, 1]" = var_mean_4[0]
        getitem_9: "f32[16, 128, 1]" = var_mean_4[1];  var_mean_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:96 in forward, code: return super().forward(input_ids) * self.embed_scale
        embedding_2: "f32[16, 128, 2560]" = torch.ops.aten.embedding.default(arg2_1, arg1_1, 0);  arg1_1 = None
        mul_21: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(embedding_2, 1.0);  embedding_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:585 in forward, code: position_ids = torch.arange(seq_length, device=inputs_embeds.device) + past_key_values_length
        iota_5: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_21: "i64[128]" = torch.ops.aten.add.Tensor(iota_5, 0);  iota_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:82 in forward, code: return super().forward(position_ids)
        embedding_3: "f32[128, 2560]" = torch.ops.aten.embedding.default(arg38_1, add_21);  arg38_1 = add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:616 in forward, code: hidden_states = inputs_embeds + position_ids
        add_26: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_21, embedding_3);  mul_21 = embedding_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_5 = torch.ops.aten.var_mean.correction(add_26, [2], correction = 0, keepdim = True)
        getitem_10: "f32[16, 128, 1]" = var_mean_5[0]
        getitem_11: "f32[16, 128, 1]" = var_mean_5[1];  var_mean_5 = None
        sub_7: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_26, getitem_11);  getitem_11 = None
        add_27: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_10, 1e-05);  getitem_10 = None
        rsqrt_5: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_27);  add_27 = None
        mul_22: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_7, rsqrt_5);  sub_7 = rsqrt_5 = None
        mul_23: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_22, arg39_1);  mul_22 = arg39_1 = None
        add_28: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_23, arg40_1);  mul_23 = arg40_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_44: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_28, [2048, 2560])
        permute_22: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg41_1, [1, 0]);  arg41_1 = None
        addmm_12: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg42_1, view_44, permute_22);  arg42_1 = view_44 = permute_22 = None
        view_45: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_12, [16, 128, 2560]);  addmm_12 = None
        view_46: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_45, [16, 128, -1, 80]);  view_45 = None
        permute_23: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_46, [0, 2, 1, 3]);  view_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_47: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_28, [2048, 2560])
        permute_24: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg43_1, [1, 0]);  arg43_1 = None
        addmm_13: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg44_1, view_47, permute_24);  arg44_1 = view_47 = permute_24 = None
        view_48: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_13, [16, 128, 2560]);  addmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_51: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_48, [16, 128, -1, 80]);  view_48 = None
        permute_26: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_51, [0, 2, 1, 3]);  view_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_49: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_28, [2048, 2560]);  add_28 = None
        permute_25: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg45_1, [1, 0]);  arg45_1 = None
        addmm_14: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg46_1, view_49, permute_25);  arg46_1 = view_49 = permute_25 = None
        view_50: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_14, [16, 128, 2560]);  addmm_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_52: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_50, [16, 128, -1, 80]);  view_50 = None
        permute_27: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_52, [0, 2, 1, 3]);  view_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:512 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        iota_9: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_23: "i64[128]" = torch.ops.aten.add.Tensor(iota_9, 0);  iota_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:363 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_6: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_23, 0);  add_23 = None
        unsqueeze_7: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, 1);  unsqueeze_6 = None
        unsqueeze_8: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_7, 2);  unsqueeze_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_8: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_22: "i64[128]" = torch.ops.aten.add.Tensor(iota_8, 0);  iota_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_22, 0);  add_22 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 3);  unsqueeze_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:78 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_8, unsqueeze_5);  unsqueeze_8 = unsqueeze_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_9: "b8[16, 1, 128, 128]" = torch.ops.aten.expand.default(le, [16, -1, 128, 128]);  le = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_7: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_6: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_4: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_7, full_default_6);  full_default_7 = full_default_6 = None
        expand_11: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_4, [16, 32, 128, 128]);  where_4 = None
        _scaled_dot_product_efficient_attention = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_23, permute_26, permute_27, expand_11, False, scale = 0.11180339887498948);  permute_23 = permute_26 = permute_27 = expand_11 = None
        getitem_12: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention[0];  _scaled_dot_product_efficient_attention = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_28: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_12, [0, 2, 1, 3]);  getitem_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_53: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_28, [16, 128, -1]);  permute_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_54: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_53, [2048, 2560]);  view_53 = None
        permute_29: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg47_1, [1, 0]);  arg47_1 = None
        addmm_15: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg48_1, view_54, permute_29);  arg48_1 = view_54 = permute_29 = None
        view_55: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_15, [16, 128, 2560]);  addmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_29: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_26, view_55);  add_26 = view_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_6 = torch.ops.aten.var_mean.correction(add_29, [2], correction = 0, keepdim = True)
        getitem_16: "f32[16, 128, 1]" = var_mean_6[0]
        getitem_17: "f32[16, 128, 1]" = var_mean_6[1];  var_mean_6 = None
        sub_8: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_29, getitem_17);  getitem_17 = None
        add_30: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_16, 1e-05);  getitem_16 = None
        rsqrt_6: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_30);  add_30 = None
        mul_24: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_8, rsqrt_6);  sub_8 = rsqrt_6 = None
        mul_25: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_24, arg49_1);  mul_24 = arg49_1 = None
        add_31: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_25, arg50_1);  mul_25 = arg50_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_56: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_31, [2048, 2560]);  add_31 = None
        permute_30: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg51_1, [1, 0]);  arg51_1 = None
        addmm_16: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg52_1, view_56, permute_30);  arg52_1 = view_56 = permute_30 = None
        view_57: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_16, [16, 128, 2560]);  addmm_16 = None
        view_58: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_57, [16, 128, -1, 80]);  view_57 = None
        permute_31: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_58, [0, 2, 1, 3]);  view_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_26: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_31, 0.334370152488211);  permute_31 = None
        expand_12: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_26, [16, 32, 128, 80]);  mul_26 = None
        clone_17: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_12, memory_format = torch.contiguous_format);  expand_12 = None
        view_65: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_17, [512, 128, 80]);  clone_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:508 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_6: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_18, getitem_9);  add_18 = getitem_9 = None
        add_19: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_8, 1e-05);  getitem_8 = None
        rsqrt_4: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_19);  add_19 = None
        mul_19: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_6, rsqrt_4);  sub_6 = rsqrt_4 = None
        mul_20: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_19, arg36_1);  mul_19 = arg36_1 = None
        add_20: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_20, arg37_1);  mul_20 = arg37_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_59: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_32: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg53_1, [1, 0]);  arg53_1 = None
        addmm_17: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg54_1, view_59, permute_32);  arg54_1 = view_59 = permute_32 = None
        view_60: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_17, [16, 128, 2560]);  addmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_63: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_60, [16, 128, -1, 80]);  view_60 = None
        permute_34: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_63, [0, 2, 1, 3]);  view_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_36: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_34, [0, 1, 3, 2]);  permute_34 = None
        mul_27: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_36, 0.334370152488211);  permute_36 = None
        expand_13: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_27, [16, 32, 80, 128]);  mul_27 = None
        clone_18: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_13, memory_format = torch.contiguous_format);  expand_13 = None
        view_66: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_18, [512, 80, 128]);  clone_18 = None
        bmm_4: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_65, view_66);  view_65 = view_66 = None
        view_67: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_4, [16, 32, 128, 128]);  bmm_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:511 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        iota_12: "i64[128]" = torch.ops.prims.iota.default(128, start = 0, step = 1, dtype = torch.int64, device = device(type='cuda', index=0), requires_grad = False)
        add_24: "i64[128]" = torch.ops.aten.add.Tensor(iota_12, 0);  iota_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:362 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze_9: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_24, 0);  add_24 = None
        unsqueeze_10: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_9, 1);  unsqueeze_9 = None
        unsqueeze_11: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, 3);  unsqueeze_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:87 in bidirectional_mask_function, code: return q_idx >= 0
        ge_1: "b8[1, 1, 128, 1]" = torch.ops.aten.ge.Scalar(unsqueeze_11, 0);  unsqueeze_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/masking_utils.py:520 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand_10: "b8[16, 1, 128, 128]" = torch.ops.aten.expand.default(ge_1, [16, -1, 128, 128]);  ge_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_9: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_8: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_5: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_9, full_default_8);  full_default_9 = full_default_8 = None
        add_32: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_67, where_5);  view_67 = where_5 = None
        eq_2: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_32, -inf)
        logical_not_4: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_2);  eq_2 = None
        any_3: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_4, -1, True);  logical_not_4 = None
        logical_not_5: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_3);  any_3 = None
        full_default_10: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_2: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_32, [-1], True)
        sub_9: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_32, amax_2);  add_32 = amax_2 = None
        exp_2: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_9);  sub_9 = None
        sum_3: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_2, [-1], True)
        div_2: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_2, sum_3);  exp_2 = sum_3 = None
        where_6: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_5, full_default_10, div_2);  logical_not_5 = full_default_10 = div_2 = None
        expand_14: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_6, [16, 32, 128, 128]);  where_6 = None
        view_68: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_14, [512, 128, 128]);  expand_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_61: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_33: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg55_1, [1, 0]);  arg55_1 = None
        addmm_18: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg56_1, view_61, permute_33);  arg56_1 = view_61 = permute_33 = None
        view_62: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_18, [16, 128, 2560]);  addmm_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_64: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_62, [16, 128, -1, 80]);  view_62 = None
        permute_35: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_64, [0, 2, 1, 3]);  view_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_15: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_35, [16, 32, 128, 80]);  permute_35 = None
        clone_19: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_15, memory_format = torch.contiguous_format);  expand_15 = None
        view_69: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_19, [512, 128, 80]);  clone_19 = None
        bmm_5: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_68, view_69);  view_68 = view_69 = None
        view_70: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_5, [16, 32, 128, 80]);  bmm_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_37: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_70, [0, 2, 1, 3]);  view_70 = None
        clone_20: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_37, memory_format = torch.contiguous_format);  permute_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_71: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_20, [16, 128, -1]);  clone_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_72: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_71, [2048, 2560]);  view_71 = None
        permute_38: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg57_1, [1, 0]);  arg57_1 = None
        addmm_19: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg58_1, view_72, permute_38);  arg58_1 = view_72 = permute_38 = None
        view_73: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_19, [16, 128, 2560]);  addmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_33: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_29, view_73);  add_29 = view_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_7 = torch.ops.aten.var_mean.correction(add_33, [2], correction = 0, keepdim = True)
        getitem_18: "f32[16, 128, 1]" = var_mean_7[0]
        getitem_19: "f32[16, 128, 1]" = var_mean_7[1];  var_mean_7 = None
        sub_10: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_33, getitem_19);  getitem_19 = None
        add_34: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_18, 1e-05);  getitem_18 = None
        rsqrt_7: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_34);  add_34 = None
        mul_28: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_10, rsqrt_7);  sub_10 = rsqrt_7 = None
        mul_29: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_28, arg59_1);  mul_28 = arg59_1 = None
        add_35: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_29, arg60_1);  mul_29 = arg60_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_74: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_35, [2048, 2560]);  add_35 = None
        permute_39: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg61_1, [1, 0]);  arg61_1 = None
        addmm_20: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg62_1, view_74, permute_39);  arg62_1 = view_74 = permute_39 = None
        view_75: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_20, [16, 128, 10240]);  addmm_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_30: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_75, 0.5)
        mul_31: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_75, 0.7071067811865476);  view_75 = None
        erf_2: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_31);  mul_31 = None
        add_36: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_2, 1);  erf_2 = None
        mul_32: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_30, add_36);  mul_30 = add_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_76: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_32, [2048, 10240]);  mul_32 = None
        permute_40: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg63_1, [1, 0]);  arg63_1 = None
        addmm_21: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg64_1, view_76, permute_40);  arg64_1 = view_76 = permute_40 = None
        view_77: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_21, [16, 128, 2560]);  addmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_37: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_33, view_77);  add_33 = view_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_8 = torch.ops.aten.var_mean.correction(add_37, [2], correction = 0, keepdim = True)
        getitem_20: "f32[16, 128, 1]" = var_mean_8[0]
        getitem_21: "f32[16, 128, 1]" = var_mean_8[1];  var_mean_8 = None
        sub_11: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_37, getitem_21);  getitem_21 = None
        add_38: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_20, 1e-05);  getitem_20 = None
        rsqrt_8: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_38);  add_38 = None
        mul_33: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_11, rsqrt_8);  sub_11 = rsqrt_8 = None
        mul_34: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_33, arg65_1);  mul_33 = arg65_1 = None
        add_39: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_34, arg66_1);  mul_34 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_78: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_39, [2048, 2560])
        permute_41: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg67_1, [1, 0]);  arg67_1 = None
        addmm_22: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg68_1, view_78, permute_41);  arg68_1 = view_78 = permute_41 = None
        view_79: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_22, [16, 128, 2560]);  addmm_22 = None
        view_80: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_79, [16, 128, -1, 80]);  view_79 = None
        permute_42: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_80, [0, 2, 1, 3]);  view_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_81: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_39, [2048, 2560])
        permute_43: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg69_1, [1, 0]);  arg69_1 = None
        addmm_23: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg70_1, view_81, permute_43);  arg70_1 = view_81 = permute_43 = None
        view_82: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_23, [16, 128, 2560]);  addmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_85: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_82, [16, 128, -1, 80]);  view_82 = None
        permute_45: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_85, [0, 2, 1, 3]);  view_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_83: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_39, [2048, 2560]);  add_39 = None
        permute_44: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg71_1, [1, 0]);  arg71_1 = None
        addmm_24: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg72_1, view_83, permute_44);  arg72_1 = view_83 = permute_44 = None
        view_84: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_24, [16, 128, 2560]);  addmm_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_86: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_84, [16, 128, -1, 80]);  view_84 = None
        permute_46: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_86, [0, 2, 1, 3]);  view_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_12: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_11: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_7: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_12, full_default_11);  full_default_12 = full_default_11 = None
        expand_16: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_7, [16, 32, 128, 128]);  where_7 = None
        _scaled_dot_product_efficient_attention_1 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_42, permute_45, permute_46, expand_16, False, scale = 0.11180339887498948);  permute_42 = permute_45 = permute_46 = expand_16 = None
        getitem_22: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_1[0];  _scaled_dot_product_efficient_attention_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_47: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_22, [0, 2, 1, 3]);  getitem_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_87: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_47, [16, 128, -1]);  permute_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_88: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_87, [2048, 2560]);  view_87 = None
        permute_48: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg73_1, [1, 0]);  arg73_1 = None
        addmm_25: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg74_1, view_88, permute_48);  arg74_1 = view_88 = permute_48 = None
        view_89: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_25, [16, 128, 2560]);  addmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_40: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_37, view_89);  add_37 = view_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_9 = torch.ops.aten.var_mean.correction(add_40, [2], correction = 0, keepdim = True)
        getitem_26: "f32[16, 128, 1]" = var_mean_9[0]
        getitem_27: "f32[16, 128, 1]" = var_mean_9[1];  var_mean_9 = None
        sub_12: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_40, getitem_27);  getitem_27 = None
        add_41: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_26, 1e-05);  getitem_26 = None
        rsqrt_9: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_41);  add_41 = None
        mul_35: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_12, rsqrt_9);  sub_12 = rsqrt_9 = None
        mul_36: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_35, arg75_1);  mul_35 = arg75_1 = None
        add_42: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_36, arg76_1);  mul_36 = arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_90: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_42, [2048, 2560]);  add_42 = None
        permute_49: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg77_1, [1, 0]);  arg77_1 = None
        addmm_26: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg78_1, view_90, permute_49);  arg78_1 = view_90 = permute_49 = None
        view_91: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_26, [16, 128, 2560]);  addmm_26 = None
        view_92: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_91, [16, 128, -1, 80]);  view_91 = None
        permute_50: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_92, [0, 2, 1, 3]);  view_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_37: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_50, 0.334370152488211);  permute_50 = None
        expand_17: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_37, [16, 32, 128, 80]);  mul_37 = None
        clone_25: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_17, memory_format = torch.contiguous_format);  expand_17 = None
        view_99: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_25, [512, 128, 80]);  clone_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_93: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_51: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg79_1, [1, 0]);  arg79_1 = None
        addmm_27: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg80_1, view_93, permute_51);  arg80_1 = view_93 = permute_51 = None
        view_94: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_27, [16, 128, 2560]);  addmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_97: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_94, [16, 128, -1, 80]);  view_94 = None
        permute_53: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_97, [0, 2, 1, 3]);  view_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_55: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_53, [0, 1, 3, 2]);  permute_53 = None
        mul_38: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_55, 0.334370152488211);  permute_55 = None
        expand_18: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_38, [16, 32, 80, 128]);  mul_38 = None
        clone_26: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_18, memory_format = torch.contiguous_format);  expand_18 = None
        view_100: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_26, [512, 80, 128]);  clone_26 = None
        bmm_6: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_99, view_100);  view_99 = view_100 = None
        view_101: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_6, [16, 32, 128, 128]);  bmm_6 = None
        full_default_14: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_13: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_8: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_14, full_default_13);  full_default_14 = full_default_13 = None
        add_43: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_101, where_8);  view_101 = where_8 = None
        eq_3: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_43, -inf)
        logical_not_6: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_3);  eq_3 = None
        any_4: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_6, -1, True);  logical_not_6 = None
        logical_not_7: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_4);  any_4 = None
        full_default_15: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_3: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_43, [-1], True)
        sub_13: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_43, amax_3);  add_43 = amax_3 = None
        exp_3: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_13);  sub_13 = None
        sum_4: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_3, [-1], True)
        div_3: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_3, sum_4);  exp_3 = sum_4 = None
        where_9: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_7, full_default_15, div_3);  logical_not_7 = full_default_15 = div_3 = None
        expand_19: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_9, [16, 32, 128, 128]);  where_9 = None
        view_102: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_19, [512, 128, 128]);  expand_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_95: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_52: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg81_1, [1, 0]);  arg81_1 = None
        addmm_28: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg82_1, view_95, permute_52);  arg82_1 = view_95 = permute_52 = None
        view_96: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_28, [16, 128, 2560]);  addmm_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_98: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_96, [16, 128, -1, 80]);  view_96 = None
        permute_54: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_98, [0, 2, 1, 3]);  view_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_20: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_54, [16, 32, 128, 80]);  permute_54 = None
        clone_27: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_20, memory_format = torch.contiguous_format);  expand_20 = None
        view_103: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_27, [512, 128, 80]);  clone_27 = None
        bmm_7: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_102, view_103);  view_102 = view_103 = None
        view_104: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_7, [16, 32, 128, 80]);  bmm_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_56: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_104, [0, 2, 1, 3]);  view_104 = None
        clone_28: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_56, memory_format = torch.contiguous_format);  permute_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_105: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_28, [16, 128, -1]);  clone_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_106: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_105, [2048, 2560]);  view_105 = None
        permute_57: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg83_1, [1, 0]);  arg83_1 = None
        addmm_29: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg84_1, view_106, permute_57);  arg84_1 = view_106 = permute_57 = None
        view_107: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_29, [16, 128, 2560]);  addmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_44: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_40, view_107);  add_40 = view_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_10 = torch.ops.aten.var_mean.correction(add_44, [2], correction = 0, keepdim = True)
        getitem_28: "f32[16, 128, 1]" = var_mean_10[0]
        getitem_29: "f32[16, 128, 1]" = var_mean_10[1];  var_mean_10 = None
        sub_14: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_44, getitem_29);  getitem_29 = None
        add_45: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_28, 1e-05);  getitem_28 = None
        rsqrt_10: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_45);  add_45 = None
        mul_39: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_14, rsqrt_10);  sub_14 = rsqrt_10 = None
        mul_40: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_39, arg85_1);  mul_39 = arg85_1 = None
        add_46: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_40, arg86_1);  mul_40 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_108: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_46, [2048, 2560]);  add_46 = None
        permute_58: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg87_1, [1, 0]);  arg87_1 = None
        addmm_30: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg88_1, view_108, permute_58);  arg88_1 = view_108 = permute_58 = None
        view_109: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_30, [16, 128, 10240]);  addmm_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_41: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_109, 0.5)
        mul_42: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_109, 0.7071067811865476);  view_109 = None
        erf_3: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_42);  mul_42 = None
        add_47: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_3, 1);  erf_3 = None
        mul_43: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_41, add_47);  mul_41 = add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_110: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_43, [2048, 10240]);  mul_43 = None
        permute_59: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg89_1, [1, 0]);  arg89_1 = None
        addmm_31: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg90_1, view_110, permute_59);  arg90_1 = view_110 = permute_59 = None
        view_111: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_31, [16, 128, 2560]);  addmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_48: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_44, view_111);  add_44 = view_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_11 = torch.ops.aten.var_mean.correction(add_48, [2], correction = 0, keepdim = True)
        getitem_30: "f32[16, 128, 1]" = var_mean_11[0]
        getitem_31: "f32[16, 128, 1]" = var_mean_11[1];  var_mean_11 = None
        sub_15: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_48, getitem_31);  getitem_31 = None
        add_49: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_30, 1e-05);  getitem_30 = None
        rsqrt_11: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_49);  add_49 = None
        mul_44: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_15, rsqrt_11);  sub_15 = rsqrt_11 = None
        mul_45: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_44, arg91_1);  mul_44 = arg91_1 = None
        add_50: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_45, arg92_1);  mul_45 = arg92_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_112: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_50, [2048, 2560])
        permute_60: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg93_1, [1, 0]);  arg93_1 = None
        addmm_32: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg94_1, view_112, permute_60);  arg94_1 = view_112 = permute_60 = None
        view_113: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_32, [16, 128, 2560]);  addmm_32 = None
        view_114: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_113, [16, 128, -1, 80]);  view_113 = None
        permute_61: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_114, [0, 2, 1, 3]);  view_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_115: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_50, [2048, 2560])
        permute_62: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg95_1, [1, 0]);  arg95_1 = None
        addmm_33: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg96_1, view_115, permute_62);  arg96_1 = view_115 = permute_62 = None
        view_116: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_33, [16, 128, 2560]);  addmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_119: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_116, [16, 128, -1, 80]);  view_116 = None
        permute_64: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_119, [0, 2, 1, 3]);  view_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_117: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_50, [2048, 2560]);  add_50 = None
        permute_63: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg97_1, [1, 0]);  arg97_1 = None
        addmm_34: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg98_1, view_117, permute_63);  arg98_1 = view_117 = permute_63 = None
        view_118: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_34, [16, 128, 2560]);  addmm_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_120: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_118, [16, 128, -1, 80]);  view_118 = None
        permute_65: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_120, [0, 2, 1, 3]);  view_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_17: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_16: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_10: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_17, full_default_16);  full_default_17 = full_default_16 = None
        expand_21: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_10, [16, 32, 128, 128]);  where_10 = None
        _scaled_dot_product_efficient_attention_2 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_61, permute_64, permute_65, expand_21, False, scale = 0.11180339887498948);  permute_61 = permute_64 = permute_65 = expand_21 = None
        getitem_32: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_2[0];  _scaled_dot_product_efficient_attention_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_66: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_32, [0, 2, 1, 3]);  getitem_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_121: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_66, [16, 128, -1]);  permute_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_122: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_121, [2048, 2560]);  view_121 = None
        permute_67: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg99_1, [1, 0]);  arg99_1 = None
        addmm_35: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg100_1, view_122, permute_67);  arg100_1 = view_122 = permute_67 = None
        view_123: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_35, [16, 128, 2560]);  addmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_51: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_48, view_123);  add_48 = view_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_12 = torch.ops.aten.var_mean.correction(add_51, [2], correction = 0, keepdim = True)
        getitem_36: "f32[16, 128, 1]" = var_mean_12[0]
        getitem_37: "f32[16, 128, 1]" = var_mean_12[1];  var_mean_12 = None
        sub_16: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_51, getitem_37);  getitem_37 = None
        add_52: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_36, 1e-05);  getitem_36 = None
        rsqrt_12: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_52);  add_52 = None
        mul_46: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_16, rsqrt_12);  sub_16 = rsqrt_12 = None
        mul_47: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_46, arg101_1);  mul_46 = arg101_1 = None
        add_53: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_47, arg102_1);  mul_47 = arg102_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_124: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_53, [2048, 2560]);  add_53 = None
        permute_68: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg103_1, [1, 0]);  arg103_1 = None
        addmm_36: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg104_1, view_124, permute_68);  arg104_1 = view_124 = permute_68 = None
        view_125: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_36, [16, 128, 2560]);  addmm_36 = None
        view_126: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_125, [16, 128, -1, 80]);  view_125 = None
        permute_69: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_126, [0, 2, 1, 3]);  view_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_48: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_69, 0.334370152488211);  permute_69 = None
        expand_22: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_48, [16, 32, 128, 80]);  mul_48 = None
        clone_33: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_22, memory_format = torch.contiguous_format);  expand_22 = None
        view_133: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_33, [512, 128, 80]);  clone_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_127: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_70: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg105_1, [1, 0]);  arg105_1 = None
        addmm_37: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg106_1, view_127, permute_70);  arg106_1 = view_127 = permute_70 = None
        view_128: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_37, [16, 128, 2560]);  addmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_131: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_128, [16, 128, -1, 80]);  view_128 = None
        permute_72: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_131, [0, 2, 1, 3]);  view_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_74: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_72, [0, 1, 3, 2]);  permute_72 = None
        mul_49: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_74, 0.334370152488211);  permute_74 = None
        expand_23: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_49, [16, 32, 80, 128]);  mul_49 = None
        clone_34: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_23, memory_format = torch.contiguous_format);  expand_23 = None
        view_134: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_34, [512, 80, 128]);  clone_34 = None
        bmm_8: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_133, view_134);  view_133 = view_134 = None
        view_135: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_8, [16, 32, 128, 128]);  bmm_8 = None
        full_default_19: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_18: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_11: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_19, full_default_18);  full_default_19 = full_default_18 = None
        add_54: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_135, where_11);  view_135 = where_11 = None
        eq_4: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_54, -inf)
        logical_not_8: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_4);  eq_4 = None
        any_5: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_8, -1, True);  logical_not_8 = None
        logical_not_9: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_5);  any_5 = None
        full_default_20: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_4: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_54, [-1], True)
        sub_17: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_54, amax_4);  add_54 = amax_4 = None
        exp_4: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_17);  sub_17 = None
        sum_5: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_4, [-1], True)
        div_4: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_4, sum_5);  exp_4 = sum_5 = None
        where_12: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_9, full_default_20, div_4);  logical_not_9 = full_default_20 = div_4 = None
        expand_24: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_12, [16, 32, 128, 128]);  where_12 = None
        view_136: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_24, [512, 128, 128]);  expand_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_129: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_71: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg107_1, [1, 0]);  arg107_1 = None
        addmm_38: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg108_1, view_129, permute_71);  arg108_1 = view_129 = permute_71 = None
        view_130: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_38, [16, 128, 2560]);  addmm_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_132: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_130, [16, 128, -1, 80]);  view_130 = None
        permute_73: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_132, [0, 2, 1, 3]);  view_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_25: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_73, [16, 32, 128, 80]);  permute_73 = None
        clone_35: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_25, memory_format = torch.contiguous_format);  expand_25 = None
        view_137: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_35, [512, 128, 80]);  clone_35 = None
        bmm_9: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_136, view_137);  view_136 = view_137 = None
        view_138: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_9, [16, 32, 128, 80]);  bmm_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_75: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_138, [0, 2, 1, 3]);  view_138 = None
        clone_36: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_75, memory_format = torch.contiguous_format);  permute_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_139: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_36, [16, 128, -1]);  clone_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_140: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_139, [2048, 2560]);  view_139 = None
        permute_76: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg109_1, [1, 0]);  arg109_1 = None
        addmm_39: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg110_1, view_140, permute_76);  arg110_1 = view_140 = permute_76 = None
        view_141: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_39, [16, 128, 2560]);  addmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_55: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_51, view_141);  add_51 = view_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_13 = torch.ops.aten.var_mean.correction(add_55, [2], correction = 0, keepdim = True)
        getitem_38: "f32[16, 128, 1]" = var_mean_13[0]
        getitem_39: "f32[16, 128, 1]" = var_mean_13[1];  var_mean_13 = None
        sub_18: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_55, getitem_39);  getitem_39 = None
        add_56: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_38, 1e-05);  getitem_38 = None
        rsqrt_13: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_56);  add_56 = None
        mul_50: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_18, rsqrt_13);  sub_18 = rsqrt_13 = None
        mul_51: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_50, arg111_1);  mul_50 = arg111_1 = None
        add_57: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_51, arg112_1);  mul_51 = arg112_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_142: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_57, [2048, 2560]);  add_57 = None
        permute_77: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg113_1, [1, 0]);  arg113_1 = None
        addmm_40: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg114_1, view_142, permute_77);  arg114_1 = view_142 = permute_77 = None
        view_143: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_40, [16, 128, 10240]);  addmm_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_52: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_143, 0.5)
        mul_53: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_143, 0.7071067811865476);  view_143 = None
        erf_4: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_53);  mul_53 = None
        add_58: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_4, 1);  erf_4 = None
        mul_54: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_52, add_58);  mul_52 = add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_144: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_54, [2048, 10240]);  mul_54 = None
        permute_78: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg115_1, [1, 0]);  arg115_1 = None
        addmm_41: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg116_1, view_144, permute_78);  arg116_1 = view_144 = permute_78 = None
        view_145: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_41, [16, 128, 2560]);  addmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_59: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_55, view_145);  add_55 = view_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_14 = torch.ops.aten.var_mean.correction(add_59, [2], correction = 0, keepdim = True)
        getitem_40: "f32[16, 128, 1]" = var_mean_14[0]
        getitem_41: "f32[16, 128, 1]" = var_mean_14[1];  var_mean_14 = None
        sub_19: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_59, getitem_41);  getitem_41 = None
        add_60: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_40, 1e-05);  getitem_40 = None
        rsqrt_14: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_60);  add_60 = None
        mul_55: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_19, rsqrt_14);  sub_19 = rsqrt_14 = None
        mul_56: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_55, arg117_1);  mul_55 = arg117_1 = None
        add_61: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_56, arg118_1);  mul_56 = arg118_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_146: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_61, [2048, 2560])
        permute_79: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg119_1, [1, 0]);  arg119_1 = None
        addmm_42: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg120_1, view_146, permute_79);  arg120_1 = view_146 = permute_79 = None
        view_147: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_42, [16, 128, 2560]);  addmm_42 = None
        view_148: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_147, [16, 128, -1, 80]);  view_147 = None
        permute_80: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_148, [0, 2, 1, 3]);  view_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_149: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_61, [2048, 2560])
        permute_81: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg121_1, [1, 0]);  arg121_1 = None
        addmm_43: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg122_1, view_149, permute_81);  arg122_1 = view_149 = permute_81 = None
        view_150: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_43, [16, 128, 2560]);  addmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_153: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_150, [16, 128, -1, 80]);  view_150 = None
        permute_83: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_153, [0, 2, 1, 3]);  view_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_151: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_61, [2048, 2560]);  add_61 = None
        permute_82: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg123_1, [1, 0]);  arg123_1 = None
        addmm_44: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg124_1, view_151, permute_82);  arg124_1 = view_151 = permute_82 = None
        view_152: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_44, [16, 128, 2560]);  addmm_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_154: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_152, [16, 128, -1, 80]);  view_152 = None
        permute_84: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_154, [0, 2, 1, 3]);  view_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_22: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_21: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_13: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_22, full_default_21);  full_default_22 = full_default_21 = None
        expand_26: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_13, [16, 32, 128, 128]);  where_13 = None
        _scaled_dot_product_efficient_attention_3 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_80, permute_83, permute_84, expand_26, False, scale = 0.11180339887498948);  permute_80 = permute_83 = permute_84 = expand_26 = None
        getitem_42: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_3[0];  _scaled_dot_product_efficient_attention_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_85: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_42, [0, 2, 1, 3]);  getitem_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_155: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_85, [16, 128, -1]);  permute_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_156: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_155, [2048, 2560]);  view_155 = None
        permute_86: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg125_1, [1, 0]);  arg125_1 = None
        addmm_45: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg126_1, view_156, permute_86);  arg126_1 = view_156 = permute_86 = None
        view_157: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_45, [16, 128, 2560]);  addmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_62: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_59, view_157);  add_59 = view_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_15 = torch.ops.aten.var_mean.correction(add_62, [2], correction = 0, keepdim = True)
        getitem_46: "f32[16, 128, 1]" = var_mean_15[0]
        getitem_47: "f32[16, 128, 1]" = var_mean_15[1];  var_mean_15 = None
        sub_20: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_62, getitem_47);  getitem_47 = None
        add_63: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_46, 1e-05);  getitem_46 = None
        rsqrt_15: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_63);  add_63 = None
        mul_57: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_20, rsqrt_15);  sub_20 = rsqrt_15 = None
        mul_58: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_57, arg127_1);  mul_57 = arg127_1 = None
        add_64: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_58, arg128_1);  mul_58 = arg128_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_158: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_64, [2048, 2560]);  add_64 = None
        permute_87: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg129_1, [1, 0]);  arg129_1 = None
        addmm_46: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg130_1, view_158, permute_87);  arg130_1 = view_158 = permute_87 = None
        view_159: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_46, [16, 128, 2560]);  addmm_46 = None
        view_160: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_159, [16, 128, -1, 80]);  view_159 = None
        permute_88: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_160, [0, 2, 1, 3]);  view_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_59: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_88, 0.334370152488211);  permute_88 = None
        expand_27: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_59, [16, 32, 128, 80]);  mul_59 = None
        clone_41: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_27, memory_format = torch.contiguous_format);  expand_27 = None
        view_167: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_41, [512, 128, 80]);  clone_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_161: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_89: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg131_1, [1, 0]);  arg131_1 = None
        addmm_47: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg132_1, view_161, permute_89);  arg132_1 = view_161 = permute_89 = None
        view_162: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_47, [16, 128, 2560]);  addmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_165: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_162, [16, 128, -1, 80]);  view_162 = None
        permute_91: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_165, [0, 2, 1, 3]);  view_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_93: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_91, [0, 1, 3, 2]);  permute_91 = None
        mul_60: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_93, 0.334370152488211);  permute_93 = None
        expand_28: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_60, [16, 32, 80, 128]);  mul_60 = None
        clone_42: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_28, memory_format = torch.contiguous_format);  expand_28 = None
        view_168: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_42, [512, 80, 128]);  clone_42 = None
        bmm_10: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_167, view_168);  view_167 = view_168 = None
        view_169: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_10, [16, 32, 128, 128]);  bmm_10 = None
        full_default_24: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_23: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_14: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_24, full_default_23);  full_default_24 = full_default_23 = None
        add_65: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_169, where_14);  view_169 = where_14 = None
        eq_5: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_65, -inf)
        logical_not_10: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_5);  eq_5 = None
        any_6: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_10, -1, True);  logical_not_10 = None
        logical_not_11: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_6);  any_6 = None
        full_default_25: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_5: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_65, [-1], True)
        sub_21: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_65, amax_5);  add_65 = amax_5 = None
        exp_5: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_21);  sub_21 = None
        sum_6: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_5, [-1], True)
        div_5: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_5, sum_6);  exp_5 = sum_6 = None
        where_15: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_11, full_default_25, div_5);  logical_not_11 = full_default_25 = div_5 = None
        expand_29: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_15, [16, 32, 128, 128]);  where_15 = None
        view_170: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_29, [512, 128, 128]);  expand_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_163: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_90: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg133_1, [1, 0]);  arg133_1 = None
        addmm_48: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg134_1, view_163, permute_90);  arg134_1 = view_163 = permute_90 = None
        view_164: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_48, [16, 128, 2560]);  addmm_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_166: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_164, [16, 128, -1, 80]);  view_164 = None
        permute_92: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_166, [0, 2, 1, 3]);  view_166 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_30: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_92, [16, 32, 128, 80]);  permute_92 = None
        clone_43: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_30, memory_format = torch.contiguous_format);  expand_30 = None
        view_171: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_43, [512, 128, 80]);  clone_43 = None
        bmm_11: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_170, view_171);  view_170 = view_171 = None
        view_172: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_11, [16, 32, 128, 80]);  bmm_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_94: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_172, [0, 2, 1, 3]);  view_172 = None
        clone_44: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_94, memory_format = torch.contiguous_format);  permute_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_173: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_44, [16, 128, -1]);  clone_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_174: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_173, [2048, 2560]);  view_173 = None
        permute_95: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg135_1, [1, 0]);  arg135_1 = None
        addmm_49: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg136_1, view_174, permute_95);  arg136_1 = view_174 = permute_95 = None
        view_175: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_49, [16, 128, 2560]);  addmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_66: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_62, view_175);  add_62 = view_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_16 = torch.ops.aten.var_mean.correction(add_66, [2], correction = 0, keepdim = True)
        getitem_48: "f32[16, 128, 1]" = var_mean_16[0]
        getitem_49: "f32[16, 128, 1]" = var_mean_16[1];  var_mean_16 = None
        sub_22: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_66, getitem_49);  getitem_49 = None
        add_67: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_48, 1e-05);  getitem_48 = None
        rsqrt_16: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_67);  add_67 = None
        mul_61: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_22, rsqrt_16);  sub_22 = rsqrt_16 = None
        mul_62: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_61, arg137_1);  mul_61 = arg137_1 = None
        add_68: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_62, arg138_1);  mul_62 = arg138_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_176: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_68, [2048, 2560]);  add_68 = None
        permute_96: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg139_1, [1, 0]);  arg139_1 = None
        addmm_50: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg140_1, view_176, permute_96);  arg140_1 = view_176 = permute_96 = None
        view_177: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_50, [16, 128, 10240]);  addmm_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_63: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_177, 0.5)
        mul_64: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_177, 0.7071067811865476);  view_177 = None
        erf_5: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_64);  mul_64 = None
        add_69: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_5, 1);  erf_5 = None
        mul_65: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_63, add_69);  mul_63 = add_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_178: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_65, [2048, 10240]);  mul_65 = None
        permute_97: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg141_1, [1, 0]);  arg141_1 = None
        addmm_51: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg142_1, view_178, permute_97);  arg142_1 = view_178 = permute_97 = None
        view_179: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_51, [16, 128, 2560]);  addmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_70: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_66, view_179);  add_66 = view_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_17 = torch.ops.aten.var_mean.correction(add_70, [2], correction = 0, keepdim = True)
        getitem_50: "f32[16, 128, 1]" = var_mean_17[0]
        getitem_51: "f32[16, 128, 1]" = var_mean_17[1];  var_mean_17 = None
        sub_23: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_70, getitem_51);  getitem_51 = None
        add_71: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_50, 1e-05);  getitem_50 = None
        rsqrt_17: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_71);  add_71 = None
        mul_66: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_23, rsqrt_17);  sub_23 = rsqrt_17 = None
        mul_67: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_66, arg143_1);  mul_66 = arg143_1 = None
        add_72: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_67, arg144_1);  mul_67 = arg144_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_180: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_72, [2048, 2560])
        permute_98: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg145_1, [1, 0]);  arg145_1 = None
        addmm_52: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg146_1, view_180, permute_98);  arg146_1 = view_180 = permute_98 = None
        view_181: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_52, [16, 128, 2560]);  addmm_52 = None
        view_182: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_181, [16, 128, -1, 80]);  view_181 = None
        permute_99: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_182, [0, 2, 1, 3]);  view_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_183: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_72, [2048, 2560])
        permute_100: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg147_1, [1, 0]);  arg147_1 = None
        addmm_53: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg148_1, view_183, permute_100);  arg148_1 = view_183 = permute_100 = None
        view_184: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_53, [16, 128, 2560]);  addmm_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_187: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_184, [16, 128, -1, 80]);  view_184 = None
        permute_102: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_187, [0, 2, 1, 3]);  view_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_185: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_72, [2048, 2560]);  add_72 = None
        permute_101: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg149_1, [1, 0]);  arg149_1 = None
        addmm_54: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg150_1, view_185, permute_101);  arg150_1 = view_185 = permute_101 = None
        view_186: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_54, [16, 128, 2560]);  addmm_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_188: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_186, [16, 128, -1, 80]);  view_186 = None
        permute_103: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_188, [0, 2, 1, 3]);  view_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_27: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_26: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_16: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_27, full_default_26);  full_default_27 = full_default_26 = None
        expand_31: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_16, [16, 32, 128, 128]);  where_16 = None
        _scaled_dot_product_efficient_attention_4 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_99, permute_102, permute_103, expand_31, False, scale = 0.11180339887498948);  permute_99 = permute_102 = permute_103 = expand_31 = None
        getitem_52: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_4[0];  _scaled_dot_product_efficient_attention_4 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_104: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_52, [0, 2, 1, 3]);  getitem_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_189: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_104, [16, 128, -1]);  permute_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_190: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_189, [2048, 2560]);  view_189 = None
        permute_105: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg151_1, [1, 0]);  arg151_1 = None
        addmm_55: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg152_1, view_190, permute_105);  arg152_1 = view_190 = permute_105 = None
        view_191: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_55, [16, 128, 2560]);  addmm_55 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_73: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_70, view_191);  add_70 = view_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_18 = torch.ops.aten.var_mean.correction(add_73, [2], correction = 0, keepdim = True)
        getitem_56: "f32[16, 128, 1]" = var_mean_18[0]
        getitem_57: "f32[16, 128, 1]" = var_mean_18[1];  var_mean_18 = None
        sub_24: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_73, getitem_57);  getitem_57 = None
        add_74: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_56, 1e-05);  getitem_56 = None
        rsqrt_18: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_74);  add_74 = None
        mul_68: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_24, rsqrt_18);  sub_24 = rsqrt_18 = None
        mul_69: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_68, arg153_1);  mul_68 = arg153_1 = None
        add_75: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_69, arg154_1);  mul_69 = arg154_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_192: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_75, [2048, 2560]);  add_75 = None
        permute_106: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg155_1, [1, 0]);  arg155_1 = None
        addmm_56: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg156_1, view_192, permute_106);  arg156_1 = view_192 = permute_106 = None
        view_193: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_56, [16, 128, 2560]);  addmm_56 = None
        view_194: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_193, [16, 128, -1, 80]);  view_193 = None
        permute_107: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_194, [0, 2, 1, 3]);  view_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_70: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_107, 0.334370152488211);  permute_107 = None
        expand_32: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_70, [16, 32, 128, 80]);  mul_70 = None
        clone_49: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_32, memory_format = torch.contiguous_format);  expand_32 = None
        view_201: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_49, [512, 128, 80]);  clone_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_195: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_108: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg157_1, [1, 0]);  arg157_1 = None
        addmm_57: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg158_1, view_195, permute_108);  arg158_1 = view_195 = permute_108 = None
        view_196: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_57, [16, 128, 2560]);  addmm_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_199: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_196, [16, 128, -1, 80]);  view_196 = None
        permute_110: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_199, [0, 2, 1, 3]);  view_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_112: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_110, [0, 1, 3, 2]);  permute_110 = None
        mul_71: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_112, 0.334370152488211);  permute_112 = None
        expand_33: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_71, [16, 32, 80, 128]);  mul_71 = None
        clone_50: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_33, memory_format = torch.contiguous_format);  expand_33 = None
        view_202: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_50, [512, 80, 128]);  clone_50 = None
        bmm_12: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_201, view_202);  view_201 = view_202 = None
        view_203: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_12, [16, 32, 128, 128]);  bmm_12 = None
        full_default_29: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_28: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_17: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_29, full_default_28);  full_default_29 = full_default_28 = None
        add_76: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_203, where_17);  view_203 = where_17 = None
        eq_6: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_76, -inf)
        logical_not_12: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_6);  eq_6 = None
        any_7: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_12, -1, True);  logical_not_12 = None
        logical_not_13: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_7);  any_7 = None
        full_default_30: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_6: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_76, [-1], True)
        sub_25: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_76, amax_6);  add_76 = amax_6 = None
        exp_6: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_25);  sub_25 = None
        sum_7: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_6, [-1], True)
        div_6: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_6, sum_7);  exp_6 = sum_7 = None
        where_18: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_13, full_default_30, div_6);  logical_not_13 = full_default_30 = div_6 = None
        expand_34: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_18, [16, 32, 128, 128]);  where_18 = None
        view_204: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_34, [512, 128, 128]);  expand_34 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_197: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_109: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg159_1, [1, 0]);  arg159_1 = None
        addmm_58: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg160_1, view_197, permute_109);  arg160_1 = view_197 = permute_109 = None
        view_198: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_58, [16, 128, 2560]);  addmm_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_200: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_198, [16, 128, -1, 80]);  view_198 = None
        permute_111: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_200, [0, 2, 1, 3]);  view_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_35: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_111, [16, 32, 128, 80]);  permute_111 = None
        clone_51: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_35, memory_format = torch.contiguous_format);  expand_35 = None
        view_205: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_51, [512, 128, 80]);  clone_51 = None
        bmm_13: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_204, view_205);  view_204 = view_205 = None
        view_206: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_13, [16, 32, 128, 80]);  bmm_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_113: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_206, [0, 2, 1, 3]);  view_206 = None
        clone_52: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_113, memory_format = torch.contiguous_format);  permute_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_207: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_52, [16, 128, -1]);  clone_52 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_208: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_207, [2048, 2560]);  view_207 = None
        permute_114: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg161_1, [1, 0]);  arg161_1 = None
        addmm_59: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg162_1, view_208, permute_114);  arg162_1 = view_208 = permute_114 = None
        view_209: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_59, [16, 128, 2560]);  addmm_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_77: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_73, view_209);  add_73 = view_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_19 = torch.ops.aten.var_mean.correction(add_77, [2], correction = 0, keepdim = True)
        getitem_58: "f32[16, 128, 1]" = var_mean_19[0]
        getitem_59: "f32[16, 128, 1]" = var_mean_19[1];  var_mean_19 = None
        sub_26: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_77, getitem_59);  getitem_59 = None
        add_78: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_58, 1e-05);  getitem_58 = None
        rsqrt_19: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_78);  add_78 = None
        mul_72: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_26, rsqrt_19);  sub_26 = rsqrt_19 = None
        mul_73: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_72, arg163_1);  mul_72 = arg163_1 = None
        add_79: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_73, arg164_1);  mul_73 = arg164_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_210: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_79, [2048, 2560]);  add_79 = None
        permute_115: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg165_1, [1, 0]);  arg165_1 = None
        addmm_60: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg166_1, view_210, permute_115);  arg166_1 = view_210 = permute_115 = None
        view_211: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_60, [16, 128, 10240]);  addmm_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_74: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_211, 0.5)
        mul_75: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_211, 0.7071067811865476);  view_211 = None
        erf_6: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_75);  mul_75 = None
        add_80: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_6, 1);  erf_6 = None
        mul_76: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_74, add_80);  mul_74 = add_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_212: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_76, [2048, 10240]);  mul_76 = None
        permute_116: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg167_1, [1, 0]);  arg167_1 = None
        addmm_61: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg168_1, view_212, permute_116);  arg168_1 = view_212 = permute_116 = None
        view_213: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_61, [16, 128, 2560]);  addmm_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_81: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_77, view_213);  add_77 = view_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_20 = torch.ops.aten.var_mean.correction(add_81, [2], correction = 0, keepdim = True)
        getitem_60: "f32[16, 128, 1]" = var_mean_20[0]
        getitem_61: "f32[16, 128, 1]" = var_mean_20[1];  var_mean_20 = None
        sub_27: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_81, getitem_61);  getitem_61 = None
        add_82: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_60, 1e-05);  getitem_60 = None
        rsqrt_20: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_82);  add_82 = None
        mul_77: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_27, rsqrt_20);  sub_27 = rsqrt_20 = None
        mul_78: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_77, arg169_1);  mul_77 = arg169_1 = None
        add_83: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_78, arg170_1);  mul_78 = arg170_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_214: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_83, [2048, 2560])
        permute_117: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg171_1, [1, 0]);  arg171_1 = None
        addmm_62: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg172_1, view_214, permute_117);  arg172_1 = view_214 = permute_117 = None
        view_215: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_62, [16, 128, 2560]);  addmm_62 = None
        view_216: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_215, [16, 128, -1, 80]);  view_215 = None
        permute_118: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_216, [0, 2, 1, 3]);  view_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_217: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_83, [2048, 2560])
        permute_119: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg173_1, [1, 0]);  arg173_1 = None
        addmm_63: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg174_1, view_217, permute_119);  arg174_1 = view_217 = permute_119 = None
        view_218: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_63, [16, 128, 2560]);  addmm_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_221: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_218, [16, 128, -1, 80]);  view_218 = None
        permute_121: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_221, [0, 2, 1, 3]);  view_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_219: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_83, [2048, 2560]);  add_83 = None
        permute_120: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg175_1, [1, 0]);  arg175_1 = None
        addmm_64: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg176_1, view_219, permute_120);  arg176_1 = view_219 = permute_120 = None
        view_220: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_64, [16, 128, 2560]);  addmm_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_222: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_220, [16, 128, -1, 80]);  view_220 = None
        permute_122: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_222, [0, 2, 1, 3]);  view_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_32: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_31: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_19: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_32, full_default_31);  full_default_32 = full_default_31 = None
        expand_36: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_19, [16, 32, 128, 128]);  where_19 = None
        _scaled_dot_product_efficient_attention_5 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_118, permute_121, permute_122, expand_36, False, scale = 0.11180339887498948);  permute_118 = permute_121 = permute_122 = expand_36 = None
        getitem_62: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_5[0];  _scaled_dot_product_efficient_attention_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_123: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_62, [0, 2, 1, 3]);  getitem_62 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_223: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_123, [16, 128, -1]);  permute_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_224: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_223, [2048, 2560]);  view_223 = None
        permute_124: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg177_1, [1, 0]);  arg177_1 = None
        addmm_65: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg178_1, view_224, permute_124);  arg178_1 = view_224 = permute_124 = None
        view_225: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_65, [16, 128, 2560]);  addmm_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_84: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_81, view_225);  add_81 = view_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_21 = torch.ops.aten.var_mean.correction(add_84, [2], correction = 0, keepdim = True)
        getitem_66: "f32[16, 128, 1]" = var_mean_21[0]
        getitem_67: "f32[16, 128, 1]" = var_mean_21[1];  var_mean_21 = None
        sub_28: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_84, getitem_67);  getitem_67 = None
        add_85: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_66, 1e-05);  getitem_66 = None
        rsqrt_21: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_85);  add_85 = None
        mul_79: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_28, rsqrt_21);  sub_28 = rsqrt_21 = None
        mul_80: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_79, arg179_1);  mul_79 = arg179_1 = None
        add_86: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_80, arg180_1);  mul_80 = arg180_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_226: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_86, [2048, 2560]);  add_86 = None
        permute_125: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg181_1, [1, 0]);  arg181_1 = None
        addmm_66: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg182_1, view_226, permute_125);  arg182_1 = view_226 = permute_125 = None
        view_227: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_66, [16, 128, 2560]);  addmm_66 = None
        view_228: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_227, [16, 128, -1, 80]);  view_227 = None
        permute_126: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_228, [0, 2, 1, 3]);  view_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_81: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_126, 0.334370152488211);  permute_126 = None
        expand_37: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_81, [16, 32, 128, 80]);  mul_81 = None
        clone_57: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_37, memory_format = torch.contiguous_format);  expand_37 = None
        view_235: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_57, [512, 128, 80]);  clone_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_229: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_127: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg183_1, [1, 0]);  arg183_1 = None
        addmm_67: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg184_1, view_229, permute_127);  arg184_1 = view_229 = permute_127 = None
        view_230: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_67, [16, 128, 2560]);  addmm_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_233: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_230, [16, 128, -1, 80]);  view_230 = None
        permute_129: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_233, [0, 2, 1, 3]);  view_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_131: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_129, [0, 1, 3, 2]);  permute_129 = None
        mul_82: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_131, 0.334370152488211);  permute_131 = None
        expand_38: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_82, [16, 32, 80, 128]);  mul_82 = None
        clone_58: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_38, memory_format = torch.contiguous_format);  expand_38 = None
        view_236: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_58, [512, 80, 128]);  clone_58 = None
        bmm_14: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_235, view_236);  view_235 = view_236 = None
        view_237: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_14, [16, 32, 128, 128]);  bmm_14 = None
        full_default_34: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_33: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_20: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_34, full_default_33);  full_default_34 = full_default_33 = None
        add_87: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_237, where_20);  view_237 = where_20 = None
        eq_7: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_87, -inf)
        logical_not_14: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_7);  eq_7 = None
        any_8: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_14, -1, True);  logical_not_14 = None
        logical_not_15: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_8);  any_8 = None
        full_default_35: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_7: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_87, [-1], True)
        sub_29: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_87, amax_7);  add_87 = amax_7 = None
        exp_7: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_29);  sub_29 = None
        sum_8: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_7, [-1], True)
        div_7: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_7, sum_8);  exp_7 = sum_8 = None
        where_21: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_15, full_default_35, div_7);  logical_not_15 = full_default_35 = div_7 = None
        expand_39: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_21, [16, 32, 128, 128]);  where_21 = None
        view_238: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_39, [512, 128, 128]);  expand_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_231: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_128: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg185_1, [1, 0]);  arg185_1 = None
        addmm_68: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg186_1, view_231, permute_128);  arg186_1 = view_231 = permute_128 = None
        view_232: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_68, [16, 128, 2560]);  addmm_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_234: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_232, [16, 128, -1, 80]);  view_232 = None
        permute_130: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_234, [0, 2, 1, 3]);  view_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_40: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_130, [16, 32, 128, 80]);  permute_130 = None
        clone_59: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_40, memory_format = torch.contiguous_format);  expand_40 = None
        view_239: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_59, [512, 128, 80]);  clone_59 = None
        bmm_15: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_238, view_239);  view_238 = view_239 = None
        view_240: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_15, [16, 32, 128, 80]);  bmm_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_132: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_240, [0, 2, 1, 3]);  view_240 = None
        clone_60: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_132, memory_format = torch.contiguous_format);  permute_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_241: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_60, [16, 128, -1]);  clone_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_242: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_241, [2048, 2560]);  view_241 = None
        permute_133: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg187_1, [1, 0]);  arg187_1 = None
        addmm_69: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg188_1, view_242, permute_133);  arg188_1 = view_242 = permute_133 = None
        view_243: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_69, [16, 128, 2560]);  addmm_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_88: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_84, view_243);  add_84 = view_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_22 = torch.ops.aten.var_mean.correction(add_88, [2], correction = 0, keepdim = True)
        getitem_68: "f32[16, 128, 1]" = var_mean_22[0]
        getitem_69: "f32[16, 128, 1]" = var_mean_22[1];  var_mean_22 = None
        sub_30: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_88, getitem_69);  getitem_69 = None
        add_89: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_68, 1e-05);  getitem_68 = None
        rsqrt_22: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_89);  add_89 = None
        mul_83: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_30, rsqrt_22);  sub_30 = rsqrt_22 = None
        mul_84: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_83, arg189_1);  mul_83 = arg189_1 = None
        add_90: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_84, arg190_1);  mul_84 = arg190_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_244: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_90, [2048, 2560]);  add_90 = None
        permute_134: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg191_1, [1, 0]);  arg191_1 = None
        addmm_70: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg192_1, view_244, permute_134);  arg192_1 = view_244 = permute_134 = None
        view_245: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_70, [16, 128, 10240]);  addmm_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_85: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_245, 0.5)
        mul_86: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_245, 0.7071067811865476);  view_245 = None
        erf_7: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_86);  mul_86 = None
        add_91: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_7, 1);  erf_7 = None
        mul_87: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_85, add_91);  mul_85 = add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_246: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_87, [2048, 10240]);  mul_87 = None
        permute_135: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg193_1, [1, 0]);  arg193_1 = None
        addmm_71: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg194_1, view_246, permute_135);  arg194_1 = view_246 = permute_135 = None
        view_247: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_71, [16, 128, 2560]);  addmm_71 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_92: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_88, view_247);  add_88 = view_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_23 = torch.ops.aten.var_mean.correction(add_92, [2], correction = 0, keepdim = True)
        getitem_70: "f32[16, 128, 1]" = var_mean_23[0]
        getitem_71: "f32[16, 128, 1]" = var_mean_23[1];  var_mean_23 = None
        sub_31: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_92, getitem_71);  getitem_71 = None
        add_93: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_70, 1e-05);  getitem_70 = None
        rsqrt_23: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_93);  add_93 = None
        mul_88: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_31, rsqrt_23);  sub_31 = rsqrt_23 = None
        mul_89: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_88, arg195_1);  mul_88 = arg195_1 = None
        add_94: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_89, arg196_1);  mul_89 = arg196_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_248: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_94, [2048, 2560])
        permute_136: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg197_1, [1, 0]);  arg197_1 = None
        addmm_72: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg198_1, view_248, permute_136);  arg198_1 = view_248 = permute_136 = None
        view_249: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_72, [16, 128, 2560]);  addmm_72 = None
        view_250: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_249, [16, 128, -1, 80]);  view_249 = None
        permute_137: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_250, [0, 2, 1, 3]);  view_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_251: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_94, [2048, 2560])
        permute_138: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg199_1, [1, 0]);  arg199_1 = None
        addmm_73: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg200_1, view_251, permute_138);  arg200_1 = view_251 = permute_138 = None
        view_252: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_73, [16, 128, 2560]);  addmm_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_255: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_252, [16, 128, -1, 80]);  view_252 = None
        permute_140: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_255, [0, 2, 1, 3]);  view_255 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_253: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_94, [2048, 2560]);  add_94 = None
        permute_139: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg201_1, [1, 0]);  arg201_1 = None
        addmm_74: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg202_1, view_253, permute_139);  arg202_1 = view_253 = permute_139 = None
        view_254: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_74, [16, 128, 2560]);  addmm_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_256: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_254, [16, 128, -1, 80]);  view_254 = None
        permute_141: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_256, [0, 2, 1, 3]);  view_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_37: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_36: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_22: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_37, full_default_36);  full_default_37 = full_default_36 = None
        expand_41: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_22, [16, 32, 128, 128]);  where_22 = None
        _scaled_dot_product_efficient_attention_6 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_137, permute_140, permute_141, expand_41, False, scale = 0.11180339887498948);  permute_137 = permute_140 = permute_141 = expand_41 = None
        getitem_72: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_6[0];  _scaled_dot_product_efficient_attention_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_142: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_72, [0, 2, 1, 3]);  getitem_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_257: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_142, [16, 128, -1]);  permute_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_258: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_257, [2048, 2560]);  view_257 = None
        permute_143: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg203_1, [1, 0]);  arg203_1 = None
        addmm_75: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg204_1, view_258, permute_143);  arg204_1 = view_258 = permute_143 = None
        view_259: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_75, [16, 128, 2560]);  addmm_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_95: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_92, view_259);  add_92 = view_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_24 = torch.ops.aten.var_mean.correction(add_95, [2], correction = 0, keepdim = True)
        getitem_76: "f32[16, 128, 1]" = var_mean_24[0]
        getitem_77: "f32[16, 128, 1]" = var_mean_24[1];  var_mean_24 = None
        sub_32: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_95, getitem_77);  getitem_77 = None
        add_96: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_76, 1e-05);  getitem_76 = None
        rsqrt_24: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_96);  add_96 = None
        mul_90: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_32, rsqrt_24);  sub_32 = rsqrt_24 = None
        mul_91: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_90, arg205_1);  mul_90 = arg205_1 = None
        add_97: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_91, arg206_1);  mul_91 = arg206_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_260: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_97, [2048, 2560]);  add_97 = None
        permute_144: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg207_1, [1, 0]);  arg207_1 = None
        addmm_76: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg208_1, view_260, permute_144);  arg208_1 = view_260 = permute_144 = None
        view_261: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_76, [16, 128, 2560]);  addmm_76 = None
        view_262: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_261, [16, 128, -1, 80]);  view_261 = None
        permute_145: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_262, [0, 2, 1, 3]);  view_262 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_92: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_145, 0.334370152488211);  permute_145 = None
        expand_42: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_92, [16, 32, 128, 80]);  mul_92 = None
        clone_65: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_42, memory_format = torch.contiguous_format);  expand_42 = None
        view_269: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_65, [512, 128, 80]);  clone_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_263: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_146: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg209_1, [1, 0]);  arg209_1 = None
        addmm_77: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg210_1, view_263, permute_146);  arg210_1 = view_263 = permute_146 = None
        view_264: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_77, [16, 128, 2560]);  addmm_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_267: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_264, [16, 128, -1, 80]);  view_264 = None
        permute_148: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_267, [0, 2, 1, 3]);  view_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_150: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_148, [0, 1, 3, 2]);  permute_148 = None
        mul_93: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_150, 0.334370152488211);  permute_150 = None
        expand_43: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_93, [16, 32, 80, 128]);  mul_93 = None
        clone_66: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_43, memory_format = torch.contiguous_format);  expand_43 = None
        view_270: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_66, [512, 80, 128]);  clone_66 = None
        bmm_16: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_269, view_270);  view_269 = view_270 = None
        view_271: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_16, [16, 32, 128, 128]);  bmm_16 = None
        full_default_39: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_38: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_23: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_39, full_default_38);  full_default_39 = full_default_38 = None
        add_98: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_271, where_23);  view_271 = where_23 = None
        eq_8: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_98, -inf)
        logical_not_16: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_8);  eq_8 = None
        any_9: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_16, -1, True);  logical_not_16 = None
        logical_not_17: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_9);  any_9 = None
        full_default_40: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_8: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_98, [-1], True)
        sub_33: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_98, amax_8);  add_98 = amax_8 = None
        exp_8: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_33);  sub_33 = None
        sum_9: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_8, [-1], True)
        div_8: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_8, sum_9);  exp_8 = sum_9 = None
        where_24: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_17, full_default_40, div_8);  logical_not_17 = full_default_40 = div_8 = None
        expand_44: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_24, [16, 32, 128, 128]);  where_24 = None
        view_272: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_44, [512, 128, 128]);  expand_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_265: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_147: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg211_1, [1, 0]);  arg211_1 = None
        addmm_78: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg212_1, view_265, permute_147);  arg212_1 = view_265 = permute_147 = None
        view_266: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_78, [16, 128, 2560]);  addmm_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_268: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_266, [16, 128, -1, 80]);  view_266 = None
        permute_149: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_268, [0, 2, 1, 3]);  view_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_45: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_149, [16, 32, 128, 80]);  permute_149 = None
        clone_67: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_45, memory_format = torch.contiguous_format);  expand_45 = None
        view_273: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_67, [512, 128, 80]);  clone_67 = None
        bmm_17: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_272, view_273);  view_272 = view_273 = None
        view_274: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_17, [16, 32, 128, 80]);  bmm_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_151: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_274, [0, 2, 1, 3]);  view_274 = None
        clone_68: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_151, memory_format = torch.contiguous_format);  permute_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_275: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_68, [16, 128, -1]);  clone_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_276: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_275, [2048, 2560]);  view_275 = None
        permute_152: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg213_1, [1, 0]);  arg213_1 = None
        addmm_79: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg214_1, view_276, permute_152);  arg214_1 = view_276 = permute_152 = None
        view_277: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_79, [16, 128, 2560]);  addmm_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_99: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_95, view_277);  add_95 = view_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_25 = torch.ops.aten.var_mean.correction(add_99, [2], correction = 0, keepdim = True)
        getitem_78: "f32[16, 128, 1]" = var_mean_25[0]
        getitem_79: "f32[16, 128, 1]" = var_mean_25[1];  var_mean_25 = None
        sub_34: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_99, getitem_79);  getitem_79 = None
        add_100: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_78, 1e-05);  getitem_78 = None
        rsqrt_25: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_100);  add_100 = None
        mul_94: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_34, rsqrt_25);  sub_34 = rsqrt_25 = None
        mul_95: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_94, arg215_1);  mul_94 = arg215_1 = None
        add_101: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_95, arg216_1);  mul_95 = arg216_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_278: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_101, [2048, 2560]);  add_101 = None
        permute_153: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg217_1, [1, 0]);  arg217_1 = None
        addmm_80: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg218_1, view_278, permute_153);  arg218_1 = view_278 = permute_153 = None
        view_279: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_80, [16, 128, 10240]);  addmm_80 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_96: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_279, 0.5)
        mul_97: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_279, 0.7071067811865476);  view_279 = None
        erf_8: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_97);  mul_97 = None
        add_102: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_8, 1);  erf_8 = None
        mul_98: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_96, add_102);  mul_96 = add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_280: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_98, [2048, 10240]);  mul_98 = None
        permute_154: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg219_1, [1, 0]);  arg219_1 = None
        addmm_81: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg220_1, view_280, permute_154);  arg220_1 = view_280 = permute_154 = None
        view_281: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_81, [16, 128, 2560]);  addmm_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_103: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_99, view_281);  add_99 = view_281 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_26 = torch.ops.aten.var_mean.correction(add_103, [2], correction = 0, keepdim = True)
        getitem_80: "f32[16, 128, 1]" = var_mean_26[0]
        getitem_81: "f32[16, 128, 1]" = var_mean_26[1];  var_mean_26 = None
        sub_35: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_103, getitem_81);  getitem_81 = None
        add_104: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_80, 1e-05);  getitem_80 = None
        rsqrt_26: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_104);  add_104 = None
        mul_99: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_35, rsqrt_26);  sub_35 = rsqrt_26 = None
        mul_100: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_99, arg221_1);  mul_99 = arg221_1 = None
        add_105: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_100, arg222_1);  mul_100 = arg222_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_282: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_105, [2048, 2560])
        permute_155: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg223_1, [1, 0]);  arg223_1 = None
        addmm_82: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg224_1, view_282, permute_155);  arg224_1 = view_282 = permute_155 = None
        view_283: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_82, [16, 128, 2560]);  addmm_82 = None
        view_284: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_283, [16, 128, -1, 80]);  view_283 = None
        permute_156: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_284, [0, 2, 1, 3]);  view_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_285: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_105, [2048, 2560])
        permute_157: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg225_1, [1, 0]);  arg225_1 = None
        addmm_83: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg226_1, view_285, permute_157);  arg226_1 = view_285 = permute_157 = None
        view_286: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_83, [16, 128, 2560]);  addmm_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_289: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_286, [16, 128, -1, 80]);  view_286 = None
        permute_159: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_289, [0, 2, 1, 3]);  view_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_287: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_105, [2048, 2560]);  add_105 = None
        permute_158: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg227_1, [1, 0]);  arg227_1 = None
        addmm_84: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg228_1, view_287, permute_158);  arg228_1 = view_287 = permute_158 = None
        view_288: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_84, [16, 128, 2560]);  addmm_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_290: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_288, [16, 128, -1, 80]);  view_288 = None
        permute_160: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_290, [0, 2, 1, 3]);  view_290 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_42: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_41: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_25: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_42, full_default_41);  full_default_42 = full_default_41 = None
        expand_46: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_25, [16, 32, 128, 128]);  where_25 = None
        _scaled_dot_product_efficient_attention_7 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_156, permute_159, permute_160, expand_46, False, scale = 0.11180339887498948);  permute_156 = permute_159 = permute_160 = expand_46 = None
        getitem_82: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_7[0];  _scaled_dot_product_efficient_attention_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_161: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_82, [0, 2, 1, 3]);  getitem_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_291: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_161, [16, 128, -1]);  permute_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_292: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_291, [2048, 2560]);  view_291 = None
        permute_162: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg229_1, [1, 0]);  arg229_1 = None
        addmm_85: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg230_1, view_292, permute_162);  arg230_1 = view_292 = permute_162 = None
        view_293: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_85, [16, 128, 2560]);  addmm_85 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_106: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_103, view_293);  add_103 = view_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_27 = torch.ops.aten.var_mean.correction(add_106, [2], correction = 0, keepdim = True)
        getitem_86: "f32[16, 128, 1]" = var_mean_27[0]
        getitem_87: "f32[16, 128, 1]" = var_mean_27[1];  var_mean_27 = None
        sub_36: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_106, getitem_87);  getitem_87 = None
        add_107: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_86, 1e-05);  getitem_86 = None
        rsqrt_27: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_107);  add_107 = None
        mul_101: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_36, rsqrt_27);  sub_36 = rsqrt_27 = None
        mul_102: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_101, arg231_1);  mul_101 = arg231_1 = None
        add_108: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_102, arg232_1);  mul_102 = arg232_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_294: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_108, [2048, 2560]);  add_108 = None
        permute_163: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg233_1, [1, 0]);  arg233_1 = None
        addmm_86: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg234_1, view_294, permute_163);  arg234_1 = view_294 = permute_163 = None
        view_295: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_86, [16, 128, 2560]);  addmm_86 = None
        view_296: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_295, [16, 128, -1, 80]);  view_295 = None
        permute_164: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_296, [0, 2, 1, 3]);  view_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_103: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_164, 0.334370152488211);  permute_164 = None
        expand_47: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_103, [16, 32, 128, 80]);  mul_103 = None
        clone_73: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_47, memory_format = torch.contiguous_format);  expand_47 = None
        view_303: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_73, [512, 128, 80]);  clone_73 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_297: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_165: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg235_1, [1, 0]);  arg235_1 = None
        addmm_87: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg236_1, view_297, permute_165);  arg236_1 = view_297 = permute_165 = None
        view_298: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_87, [16, 128, 2560]);  addmm_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_301: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_298, [16, 128, -1, 80]);  view_298 = None
        permute_167: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_301, [0, 2, 1, 3]);  view_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_169: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_167, [0, 1, 3, 2]);  permute_167 = None
        mul_104: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_169, 0.334370152488211);  permute_169 = None
        expand_48: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_104, [16, 32, 80, 128]);  mul_104 = None
        clone_74: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_48, memory_format = torch.contiguous_format);  expand_48 = None
        view_304: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_74, [512, 80, 128]);  clone_74 = None
        bmm_18: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_303, view_304);  view_303 = view_304 = None
        view_305: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_18, [16, 32, 128, 128]);  bmm_18 = None
        full_default_44: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_43: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_26: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_44, full_default_43);  full_default_44 = full_default_43 = None
        add_109: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_305, where_26);  view_305 = where_26 = None
        eq_9: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_109, -inf)
        logical_not_18: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_9);  eq_9 = None
        any_10: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_18, -1, True);  logical_not_18 = None
        logical_not_19: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_10);  any_10 = None
        full_default_45: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_9: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_109, [-1], True)
        sub_37: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_109, amax_9);  add_109 = amax_9 = None
        exp_9: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_37);  sub_37 = None
        sum_10: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_9, [-1], True)
        div_9: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_9, sum_10);  exp_9 = sum_10 = None
        where_27: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_19, full_default_45, div_9);  logical_not_19 = full_default_45 = div_9 = None
        expand_49: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_27, [16, 32, 128, 128]);  where_27 = None
        view_306: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_49, [512, 128, 128]);  expand_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_299: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_166: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg237_1, [1, 0]);  arg237_1 = None
        addmm_88: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg238_1, view_299, permute_166);  arg238_1 = view_299 = permute_166 = None
        view_300: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_88, [16, 128, 2560]);  addmm_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_302: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_300, [16, 128, -1, 80]);  view_300 = None
        permute_168: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_302, [0, 2, 1, 3]);  view_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_50: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_168, [16, 32, 128, 80]);  permute_168 = None
        clone_75: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_50, memory_format = torch.contiguous_format);  expand_50 = None
        view_307: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_75, [512, 128, 80]);  clone_75 = None
        bmm_19: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_306, view_307);  view_306 = view_307 = None
        view_308: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_19, [16, 32, 128, 80]);  bmm_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_170: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_308, [0, 2, 1, 3]);  view_308 = None
        clone_76: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_170, memory_format = torch.contiguous_format);  permute_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_309: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_76, [16, 128, -1]);  clone_76 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_310: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_309, [2048, 2560]);  view_309 = None
        permute_171: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg239_1, [1, 0]);  arg239_1 = None
        addmm_89: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg240_1, view_310, permute_171);  arg240_1 = view_310 = permute_171 = None
        view_311: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_89, [16, 128, 2560]);  addmm_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_110: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_106, view_311);  add_106 = view_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_28 = torch.ops.aten.var_mean.correction(add_110, [2], correction = 0, keepdim = True)
        getitem_88: "f32[16, 128, 1]" = var_mean_28[0]
        getitem_89: "f32[16, 128, 1]" = var_mean_28[1];  var_mean_28 = None
        sub_38: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_110, getitem_89);  getitem_89 = None
        add_111: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_88, 1e-05);  getitem_88 = None
        rsqrt_28: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_111);  add_111 = None
        mul_105: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_38, rsqrt_28);  sub_38 = rsqrt_28 = None
        mul_106: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_105, arg241_1);  mul_105 = arg241_1 = None
        add_112: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_106, arg242_1);  mul_106 = arg242_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_312: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_112, [2048, 2560]);  add_112 = None
        permute_172: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg243_1, [1, 0]);  arg243_1 = None
        addmm_90: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg244_1, view_312, permute_172);  arg244_1 = view_312 = permute_172 = None
        view_313: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_90, [16, 128, 10240]);  addmm_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_107: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_313, 0.5)
        mul_108: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_313, 0.7071067811865476);  view_313 = None
        erf_9: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_108);  mul_108 = None
        add_113: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_9, 1);  erf_9 = None
        mul_109: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_107, add_113);  mul_107 = add_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_314: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_109, [2048, 10240]);  mul_109 = None
        permute_173: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg245_1, [1, 0]);  arg245_1 = None
        addmm_91: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg246_1, view_314, permute_173);  arg246_1 = view_314 = permute_173 = None
        view_315: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_91, [16, 128, 2560]);  addmm_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_114: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_110, view_315);  add_110 = view_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_29 = torch.ops.aten.var_mean.correction(add_114, [2], correction = 0, keepdim = True)
        getitem_90: "f32[16, 128, 1]" = var_mean_29[0]
        getitem_91: "f32[16, 128, 1]" = var_mean_29[1];  var_mean_29 = None
        sub_39: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_114, getitem_91);  getitem_91 = None
        add_115: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_90, 1e-05);  getitem_90 = None
        rsqrt_29: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_115);  add_115 = None
        mul_110: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_39, rsqrt_29);  sub_39 = rsqrt_29 = None
        mul_111: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_110, arg247_1);  mul_110 = arg247_1 = None
        add_116: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_111, arg248_1);  mul_111 = arg248_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_316: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_116, [2048, 2560])
        permute_174: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg249_1, [1, 0]);  arg249_1 = None
        addmm_92: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg250_1, view_316, permute_174);  arg250_1 = view_316 = permute_174 = None
        view_317: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_92, [16, 128, 2560]);  addmm_92 = None
        view_318: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_317, [16, 128, -1, 80]);  view_317 = None
        permute_175: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_318, [0, 2, 1, 3]);  view_318 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_319: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_116, [2048, 2560])
        permute_176: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg251_1, [1, 0]);  arg251_1 = None
        addmm_93: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg252_1, view_319, permute_176);  arg252_1 = view_319 = permute_176 = None
        view_320: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_93, [16, 128, 2560]);  addmm_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_323: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_320, [16, 128, -1, 80]);  view_320 = None
        permute_178: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_323, [0, 2, 1, 3]);  view_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_321: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_116, [2048, 2560]);  add_116 = None
        permute_177: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg253_1, [1, 0]);  arg253_1 = None
        addmm_94: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg254_1, view_321, permute_177);  arg254_1 = view_321 = permute_177 = None
        view_322: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_94, [16, 128, 2560]);  addmm_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_324: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_322, [16, 128, -1, 80]);  view_322 = None
        permute_179: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_324, [0, 2, 1, 3]);  view_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_47: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_46: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_28: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_47, full_default_46);  full_default_47 = full_default_46 = None
        expand_51: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_28, [16, 32, 128, 128]);  where_28 = None
        _scaled_dot_product_efficient_attention_8 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_175, permute_178, permute_179, expand_51, False, scale = 0.11180339887498948);  permute_175 = permute_178 = permute_179 = expand_51 = None
        getitem_92: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_8[0];  _scaled_dot_product_efficient_attention_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_180: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_92, [0, 2, 1, 3]);  getitem_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_325: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_180, [16, 128, -1]);  permute_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_326: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_325, [2048, 2560]);  view_325 = None
        permute_181: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg255_1, [1, 0]);  arg255_1 = None
        addmm_95: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg256_1, view_326, permute_181);  arg256_1 = view_326 = permute_181 = None
        view_327: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_95, [16, 128, 2560]);  addmm_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_117: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_114, view_327);  add_114 = view_327 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_30 = torch.ops.aten.var_mean.correction(add_117, [2], correction = 0, keepdim = True)
        getitem_96: "f32[16, 128, 1]" = var_mean_30[0]
        getitem_97: "f32[16, 128, 1]" = var_mean_30[1];  var_mean_30 = None
        sub_40: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_117, getitem_97);  getitem_97 = None
        add_118: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_96, 1e-05);  getitem_96 = None
        rsqrt_30: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_118);  add_118 = None
        mul_112: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_40, rsqrt_30);  sub_40 = rsqrt_30 = None
        mul_113: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_112, arg257_1);  mul_112 = arg257_1 = None
        add_119: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_113, arg258_1);  mul_113 = arg258_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_328: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_119, [2048, 2560]);  add_119 = None
        permute_182: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg259_1, [1, 0]);  arg259_1 = None
        addmm_96: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg260_1, view_328, permute_182);  arg260_1 = view_328 = permute_182 = None
        view_329: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_96, [16, 128, 2560]);  addmm_96 = None
        view_330: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_329, [16, 128, -1, 80]);  view_329 = None
        permute_183: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_330, [0, 2, 1, 3]);  view_330 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_114: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_183, 0.334370152488211);  permute_183 = None
        expand_52: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_114, [16, 32, 128, 80]);  mul_114 = None
        clone_81: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_52, memory_format = torch.contiguous_format);  expand_52 = None
        view_337: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_81, [512, 128, 80]);  clone_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_331: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_184: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg261_1, [1, 0]);  arg261_1 = None
        addmm_97: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg262_1, view_331, permute_184);  arg262_1 = view_331 = permute_184 = None
        view_332: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_97, [16, 128, 2560]);  addmm_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_335: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_332, [16, 128, -1, 80]);  view_332 = None
        permute_186: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_335, [0, 2, 1, 3]);  view_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_188: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_186, [0, 1, 3, 2]);  permute_186 = None
        mul_115: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_188, 0.334370152488211);  permute_188 = None
        expand_53: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_115, [16, 32, 80, 128]);  mul_115 = None
        clone_82: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_53, memory_format = torch.contiguous_format);  expand_53 = None
        view_338: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_82, [512, 80, 128]);  clone_82 = None
        bmm_20: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_337, view_338);  view_337 = view_338 = None
        view_339: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_20, [16, 32, 128, 128]);  bmm_20 = None
        full_default_49: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_48: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_29: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_49, full_default_48);  full_default_49 = full_default_48 = None
        add_120: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_339, where_29);  view_339 = where_29 = None
        eq_10: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_120, -inf)
        logical_not_20: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_10);  eq_10 = None
        any_11: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_20, -1, True);  logical_not_20 = None
        logical_not_21: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_11);  any_11 = None
        full_default_50: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_10: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_120, [-1], True)
        sub_41: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_120, amax_10);  add_120 = amax_10 = None
        exp_10: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_41);  sub_41 = None
        sum_11: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_10, [-1], True)
        div_10: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_10, sum_11);  exp_10 = sum_11 = None
        where_30: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_21, full_default_50, div_10);  logical_not_21 = full_default_50 = div_10 = None
        expand_54: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_30, [16, 32, 128, 128]);  where_30 = None
        view_340: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_54, [512, 128, 128]);  expand_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_333: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_185: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg263_1, [1, 0]);  arg263_1 = None
        addmm_98: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg264_1, view_333, permute_185);  arg264_1 = view_333 = permute_185 = None
        view_334: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_98, [16, 128, 2560]);  addmm_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_336: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_334, [16, 128, -1, 80]);  view_334 = None
        permute_187: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_336, [0, 2, 1, 3]);  view_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_55: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_187, [16, 32, 128, 80]);  permute_187 = None
        clone_83: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_55, memory_format = torch.contiguous_format);  expand_55 = None
        view_341: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_83, [512, 128, 80]);  clone_83 = None
        bmm_21: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_340, view_341);  view_340 = view_341 = None
        view_342: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_21, [16, 32, 128, 80]);  bmm_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_189: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_342, [0, 2, 1, 3]);  view_342 = None
        clone_84: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_189, memory_format = torch.contiguous_format);  permute_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_343: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_84, [16, 128, -1]);  clone_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_344: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_343, [2048, 2560]);  view_343 = None
        permute_190: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg265_1, [1, 0]);  arg265_1 = None
        addmm_99: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg266_1, view_344, permute_190);  arg266_1 = view_344 = permute_190 = None
        view_345: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_99, [16, 128, 2560]);  addmm_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_121: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_117, view_345);  add_117 = view_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_31 = torch.ops.aten.var_mean.correction(add_121, [2], correction = 0, keepdim = True)
        getitem_98: "f32[16, 128, 1]" = var_mean_31[0]
        getitem_99: "f32[16, 128, 1]" = var_mean_31[1];  var_mean_31 = None
        sub_42: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_121, getitem_99);  getitem_99 = None
        add_122: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_98, 1e-05);  getitem_98 = None
        rsqrt_31: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_122);  add_122 = None
        mul_116: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_42, rsqrt_31);  sub_42 = rsqrt_31 = None
        mul_117: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_116, arg267_1);  mul_116 = arg267_1 = None
        add_123: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_117, arg268_1);  mul_117 = arg268_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_346: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_123, [2048, 2560]);  add_123 = None
        permute_191: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg269_1, [1, 0]);  arg269_1 = None
        addmm_100: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg270_1, view_346, permute_191);  arg270_1 = view_346 = permute_191 = None
        view_347: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_100, [16, 128, 10240]);  addmm_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_118: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_347, 0.5)
        mul_119: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_347, 0.7071067811865476);  view_347 = None
        erf_10: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_119);  mul_119 = None
        add_124: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_10, 1);  erf_10 = None
        mul_120: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_118, add_124);  mul_118 = add_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_348: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_120, [2048, 10240]);  mul_120 = None
        permute_192: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg271_1, [1, 0]);  arg271_1 = None
        addmm_101: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg272_1, view_348, permute_192);  arg272_1 = view_348 = permute_192 = None
        view_349: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_101, [16, 128, 2560]);  addmm_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_125: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_121, view_349);  add_121 = view_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_32 = torch.ops.aten.var_mean.correction(add_125, [2], correction = 0, keepdim = True)
        getitem_100: "f32[16, 128, 1]" = var_mean_32[0]
        getitem_101: "f32[16, 128, 1]" = var_mean_32[1];  var_mean_32 = None
        sub_43: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_125, getitem_101);  getitem_101 = None
        add_126: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_100, 1e-05);  getitem_100 = None
        rsqrt_32: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_126);  add_126 = None
        mul_121: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_43, rsqrt_32);  sub_43 = rsqrt_32 = None
        mul_122: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_121, arg273_1);  mul_121 = arg273_1 = None
        add_127: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_122, arg274_1);  mul_122 = arg274_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_350: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_127, [2048, 2560])
        permute_193: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg275_1, [1, 0]);  arg275_1 = None
        addmm_102: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg276_1, view_350, permute_193);  arg276_1 = view_350 = permute_193 = None
        view_351: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_102, [16, 128, 2560]);  addmm_102 = None
        view_352: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_351, [16, 128, -1, 80]);  view_351 = None
        permute_194: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_352, [0, 2, 1, 3]);  view_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_353: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_127, [2048, 2560])
        permute_195: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg277_1, [1, 0]);  arg277_1 = None
        addmm_103: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg278_1, view_353, permute_195);  arg278_1 = view_353 = permute_195 = None
        view_354: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_103, [16, 128, 2560]);  addmm_103 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_357: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_354, [16, 128, -1, 80]);  view_354 = None
        permute_197: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_357, [0, 2, 1, 3]);  view_357 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_355: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_127, [2048, 2560]);  add_127 = None
        permute_196: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg279_1, [1, 0]);  arg279_1 = None
        addmm_104: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg280_1, view_355, permute_196);  arg280_1 = view_355 = permute_196 = None
        view_356: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_104, [16, 128, 2560]);  addmm_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_358: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_356, [16, 128, -1, 80]);  view_356 = None
        permute_198: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_358, [0, 2, 1, 3]);  view_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_52: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_51: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_31: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_52, full_default_51);  full_default_52 = full_default_51 = None
        expand_56: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_31, [16, 32, 128, 128]);  where_31 = None
        _scaled_dot_product_efficient_attention_9 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_194, permute_197, permute_198, expand_56, False, scale = 0.11180339887498948);  permute_194 = permute_197 = permute_198 = expand_56 = None
        getitem_102: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_9[0];  _scaled_dot_product_efficient_attention_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_199: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_102, [0, 2, 1, 3]);  getitem_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_359: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_199, [16, 128, -1]);  permute_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_360: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_359, [2048, 2560]);  view_359 = None
        permute_200: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg281_1, [1, 0]);  arg281_1 = None
        addmm_105: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg282_1, view_360, permute_200);  arg282_1 = view_360 = permute_200 = None
        view_361: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_105, [16, 128, 2560]);  addmm_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_128: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_125, view_361);  add_125 = view_361 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_33 = torch.ops.aten.var_mean.correction(add_128, [2], correction = 0, keepdim = True)
        getitem_106: "f32[16, 128, 1]" = var_mean_33[0]
        getitem_107: "f32[16, 128, 1]" = var_mean_33[1];  var_mean_33 = None
        sub_44: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_128, getitem_107);  getitem_107 = None
        add_129: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_106, 1e-05);  getitem_106 = None
        rsqrt_33: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_129);  add_129 = None
        mul_123: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_44, rsqrt_33);  sub_44 = rsqrt_33 = None
        mul_124: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_123, arg283_1);  mul_123 = arg283_1 = None
        add_130: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_124, arg284_1);  mul_124 = arg284_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_362: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_130, [2048, 2560]);  add_130 = None
        permute_201: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg285_1, [1, 0]);  arg285_1 = None
        addmm_106: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg286_1, view_362, permute_201);  arg286_1 = view_362 = permute_201 = None
        view_363: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_106, [16, 128, 2560]);  addmm_106 = None
        view_364: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_363, [16, 128, -1, 80]);  view_363 = None
        permute_202: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_364, [0, 2, 1, 3]);  view_364 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_125: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_202, 0.334370152488211);  permute_202 = None
        expand_57: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_125, [16, 32, 128, 80]);  mul_125 = None
        clone_89: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_57, memory_format = torch.contiguous_format);  expand_57 = None
        view_371: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_89, [512, 128, 80]);  clone_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_365: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_203: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg287_1, [1, 0]);  arg287_1 = None
        addmm_107: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg288_1, view_365, permute_203);  arg288_1 = view_365 = permute_203 = None
        view_366: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_107, [16, 128, 2560]);  addmm_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_369: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_366, [16, 128, -1, 80]);  view_366 = None
        permute_205: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_369, [0, 2, 1, 3]);  view_369 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_207: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_205, [0, 1, 3, 2]);  permute_205 = None
        mul_126: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_207, 0.334370152488211);  permute_207 = None
        expand_58: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_126, [16, 32, 80, 128]);  mul_126 = None
        clone_90: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_58, memory_format = torch.contiguous_format);  expand_58 = None
        view_372: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_90, [512, 80, 128]);  clone_90 = None
        bmm_22: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_371, view_372);  view_371 = view_372 = None
        view_373: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_22, [16, 32, 128, 128]);  bmm_22 = None
        full_default_54: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_53: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_32: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_54, full_default_53);  full_default_54 = full_default_53 = None
        add_131: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_373, where_32);  view_373 = where_32 = None
        eq_11: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_131, -inf)
        logical_not_22: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_11);  eq_11 = None
        any_12: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_22, -1, True);  logical_not_22 = None
        logical_not_23: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_12);  any_12 = None
        full_default_55: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_11: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_131, [-1], True)
        sub_45: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_131, amax_11);  add_131 = amax_11 = None
        exp_11: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_45);  sub_45 = None
        sum_12: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_11, [-1], True)
        div_11: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_11, sum_12);  exp_11 = sum_12 = None
        where_33: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_23, full_default_55, div_11);  logical_not_23 = full_default_55 = div_11 = None
        expand_59: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_33, [16, 32, 128, 128]);  where_33 = None
        view_374: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_59, [512, 128, 128]);  expand_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_367: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_204: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg289_1, [1, 0]);  arg289_1 = None
        addmm_108: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg290_1, view_367, permute_204);  arg290_1 = view_367 = permute_204 = None
        view_368: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_108, [16, 128, 2560]);  addmm_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_370: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_368, [16, 128, -1, 80]);  view_368 = None
        permute_206: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_370, [0, 2, 1, 3]);  view_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_60: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_206, [16, 32, 128, 80]);  permute_206 = None
        clone_91: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_60, memory_format = torch.contiguous_format);  expand_60 = None
        view_375: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_91, [512, 128, 80]);  clone_91 = None
        bmm_23: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_374, view_375);  view_374 = view_375 = None
        view_376: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_23, [16, 32, 128, 80]);  bmm_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_208: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_376, [0, 2, 1, 3]);  view_376 = None
        clone_92: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_208, memory_format = torch.contiguous_format);  permute_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_377: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_92, [16, 128, -1]);  clone_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_378: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_377, [2048, 2560]);  view_377 = None
        permute_209: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg291_1, [1, 0]);  arg291_1 = None
        addmm_109: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg292_1, view_378, permute_209);  arg292_1 = view_378 = permute_209 = None
        view_379: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_109, [16, 128, 2560]);  addmm_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_132: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_128, view_379);  add_128 = view_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_34 = torch.ops.aten.var_mean.correction(add_132, [2], correction = 0, keepdim = True)
        getitem_108: "f32[16, 128, 1]" = var_mean_34[0]
        getitem_109: "f32[16, 128, 1]" = var_mean_34[1];  var_mean_34 = None
        sub_46: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_132, getitem_109);  getitem_109 = None
        add_133: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_108, 1e-05);  getitem_108 = None
        rsqrt_34: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_133);  add_133 = None
        mul_127: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_46, rsqrt_34);  sub_46 = rsqrt_34 = None
        mul_128: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_127, arg293_1);  mul_127 = arg293_1 = None
        add_134: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_128, arg294_1);  mul_128 = arg294_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_380: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_134, [2048, 2560]);  add_134 = None
        permute_210: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg295_1, [1, 0]);  arg295_1 = None
        addmm_110: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg296_1, view_380, permute_210);  arg296_1 = view_380 = permute_210 = None
        view_381: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_110, [16, 128, 10240]);  addmm_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_129: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_381, 0.5)
        mul_130: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_381, 0.7071067811865476);  view_381 = None
        erf_11: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_130);  mul_130 = None
        add_135: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_11, 1);  erf_11 = None
        mul_131: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_129, add_135);  mul_129 = add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_382: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_131, [2048, 10240]);  mul_131 = None
        permute_211: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg297_1, [1, 0]);  arg297_1 = None
        addmm_111: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg298_1, view_382, permute_211);  arg298_1 = view_382 = permute_211 = None
        view_383: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_111, [16, 128, 2560]);  addmm_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_136: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_132, view_383);  add_132 = view_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_35 = torch.ops.aten.var_mean.correction(add_136, [2], correction = 0, keepdim = True)
        getitem_110: "f32[16, 128, 1]" = var_mean_35[0]
        getitem_111: "f32[16, 128, 1]" = var_mean_35[1];  var_mean_35 = None
        sub_47: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_136, getitem_111);  getitem_111 = None
        add_137: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_110, 1e-05);  getitem_110 = None
        rsqrt_35: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_137);  add_137 = None
        mul_132: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_47, rsqrt_35);  sub_47 = rsqrt_35 = None
        mul_133: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_132, arg299_1);  mul_132 = arg299_1 = None
        add_138: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_133, arg300_1);  mul_133 = arg300_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_384: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_138, [2048, 2560])
        permute_212: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg301_1, [1, 0]);  arg301_1 = None
        addmm_112: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg302_1, view_384, permute_212);  arg302_1 = view_384 = permute_212 = None
        view_385: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_112, [16, 128, 2560]);  addmm_112 = None
        view_386: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_385, [16, 128, -1, 80]);  view_385 = None
        permute_213: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_386, [0, 2, 1, 3]);  view_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_387: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_138, [2048, 2560])
        permute_214: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg303_1, [1, 0]);  arg303_1 = None
        addmm_113: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg304_1, view_387, permute_214);  arg304_1 = view_387 = permute_214 = None
        view_388: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_113, [16, 128, 2560]);  addmm_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_391: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_388, [16, 128, -1, 80]);  view_388 = None
        permute_216: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_391, [0, 2, 1, 3]);  view_391 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_389: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_138, [2048, 2560]);  add_138 = None
        permute_215: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg305_1, [1, 0]);  arg305_1 = None
        addmm_114: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg306_1, view_389, permute_215);  arg306_1 = view_389 = permute_215 = None
        view_390: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_114, [16, 128, 2560]);  addmm_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_392: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_390, [16, 128, -1, 80]);  view_390 = None
        permute_217: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_392, [0, 2, 1, 3]);  view_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_57: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_56: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_34: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_57, full_default_56);  full_default_57 = full_default_56 = None
        expand_61: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_34, [16, 32, 128, 128]);  where_34 = None
        _scaled_dot_product_efficient_attention_10 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_213, permute_216, permute_217, expand_61, False, scale = 0.11180339887498948);  permute_213 = permute_216 = permute_217 = expand_61 = None
        getitem_112: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_10[0];  _scaled_dot_product_efficient_attention_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_218: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_112, [0, 2, 1, 3]);  getitem_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_393: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_218, [16, 128, -1]);  permute_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_394: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_393, [2048, 2560]);  view_393 = None
        permute_219: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg307_1, [1, 0]);  arg307_1 = None
        addmm_115: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg308_1, view_394, permute_219);  arg308_1 = view_394 = permute_219 = None
        view_395: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_115, [16, 128, 2560]);  addmm_115 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_139: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_136, view_395);  add_136 = view_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_36 = torch.ops.aten.var_mean.correction(add_139, [2], correction = 0, keepdim = True)
        getitem_116: "f32[16, 128, 1]" = var_mean_36[0]
        getitem_117: "f32[16, 128, 1]" = var_mean_36[1];  var_mean_36 = None
        sub_48: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_139, getitem_117);  getitem_117 = None
        add_140: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_116, 1e-05);  getitem_116 = None
        rsqrt_36: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_140);  add_140 = None
        mul_134: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_48, rsqrt_36);  sub_48 = rsqrt_36 = None
        mul_135: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_134, arg309_1);  mul_134 = arg309_1 = None
        add_141: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_135, arg310_1);  mul_135 = arg310_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_396: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_141, [2048, 2560]);  add_141 = None
        permute_220: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg311_1, [1, 0]);  arg311_1 = None
        addmm_116: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg312_1, view_396, permute_220);  arg312_1 = view_396 = permute_220 = None
        view_397: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_116, [16, 128, 2560]);  addmm_116 = None
        view_398: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_397, [16, 128, -1, 80]);  view_397 = None
        permute_221: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_398, [0, 2, 1, 3]);  view_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_136: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_221, 0.334370152488211);  permute_221 = None
        expand_62: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_136, [16, 32, 128, 80]);  mul_136 = None
        clone_97: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_62, memory_format = torch.contiguous_format);  expand_62 = None
        view_405: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_97, [512, 128, 80]);  clone_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_399: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_222: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg313_1, [1, 0]);  arg313_1 = None
        addmm_117: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg314_1, view_399, permute_222);  arg314_1 = view_399 = permute_222 = None
        view_400: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_117, [16, 128, 2560]);  addmm_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_403: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_400, [16, 128, -1, 80]);  view_400 = None
        permute_224: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_403, [0, 2, 1, 3]);  view_403 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_226: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_224, [0, 1, 3, 2]);  permute_224 = None
        mul_137: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_226, 0.334370152488211);  permute_226 = None
        expand_63: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_137, [16, 32, 80, 128]);  mul_137 = None
        clone_98: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_63, memory_format = torch.contiguous_format);  expand_63 = None
        view_406: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_98, [512, 80, 128]);  clone_98 = None
        bmm_24: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_405, view_406);  view_405 = view_406 = None
        view_407: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_24, [16, 32, 128, 128]);  bmm_24 = None
        full_default_59: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_58: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_35: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_59, full_default_58);  full_default_59 = full_default_58 = None
        add_142: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_407, where_35);  view_407 = where_35 = None
        eq_12: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_142, -inf)
        logical_not_24: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_12);  eq_12 = None
        any_13: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_24, -1, True);  logical_not_24 = None
        logical_not_25: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_13);  any_13 = None
        full_default_60: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_12: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_142, [-1], True)
        sub_49: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_142, amax_12);  add_142 = amax_12 = None
        exp_12: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_49);  sub_49 = None
        sum_13: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_12, [-1], True)
        div_12: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_12, sum_13);  exp_12 = sum_13 = None
        where_36: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_25, full_default_60, div_12);  logical_not_25 = full_default_60 = div_12 = None
        expand_64: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_36, [16, 32, 128, 128]);  where_36 = None
        view_408: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_64, [512, 128, 128]);  expand_64 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_401: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_223: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg315_1, [1, 0]);  arg315_1 = None
        addmm_118: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg316_1, view_401, permute_223);  arg316_1 = view_401 = permute_223 = None
        view_402: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_118, [16, 128, 2560]);  addmm_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_404: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_402, [16, 128, -1, 80]);  view_402 = None
        permute_225: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_404, [0, 2, 1, 3]);  view_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_65: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_225, [16, 32, 128, 80]);  permute_225 = None
        clone_99: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_65, memory_format = torch.contiguous_format);  expand_65 = None
        view_409: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_99, [512, 128, 80]);  clone_99 = None
        bmm_25: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_408, view_409);  view_408 = view_409 = None
        view_410: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_25, [16, 32, 128, 80]);  bmm_25 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_227: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_410, [0, 2, 1, 3]);  view_410 = None
        clone_100: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_227, memory_format = torch.contiguous_format);  permute_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_411: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_100, [16, 128, -1]);  clone_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_412: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_411, [2048, 2560]);  view_411 = None
        permute_228: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg317_1, [1, 0]);  arg317_1 = None
        addmm_119: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg318_1, view_412, permute_228);  arg318_1 = view_412 = permute_228 = None
        view_413: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_119, [16, 128, 2560]);  addmm_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_143: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_139, view_413);  add_139 = view_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_37 = torch.ops.aten.var_mean.correction(add_143, [2], correction = 0, keepdim = True)
        getitem_118: "f32[16, 128, 1]" = var_mean_37[0]
        getitem_119: "f32[16, 128, 1]" = var_mean_37[1];  var_mean_37 = None
        sub_50: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_143, getitem_119);  getitem_119 = None
        add_144: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_118, 1e-05);  getitem_118 = None
        rsqrt_37: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_144);  add_144 = None
        mul_138: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_50, rsqrt_37);  sub_50 = rsqrt_37 = None
        mul_139: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_138, arg319_1);  mul_138 = arg319_1 = None
        add_145: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_139, arg320_1);  mul_139 = arg320_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_414: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_145, [2048, 2560]);  add_145 = None
        permute_229: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg321_1, [1, 0]);  arg321_1 = None
        addmm_120: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg322_1, view_414, permute_229);  arg322_1 = view_414 = permute_229 = None
        view_415: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_120, [16, 128, 10240]);  addmm_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_140: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_415, 0.5)
        mul_141: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_415, 0.7071067811865476);  view_415 = None
        erf_12: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_141);  mul_141 = None
        add_146: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_12, 1);  erf_12 = None
        mul_142: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_140, add_146);  mul_140 = add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_416: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_142, [2048, 10240]);  mul_142 = None
        permute_230: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg323_1, [1, 0]);  arg323_1 = None
        addmm_121: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg324_1, view_416, permute_230);  arg324_1 = view_416 = permute_230 = None
        view_417: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_121, [16, 128, 2560]);  addmm_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_147: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_143, view_417);  add_143 = view_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_38 = torch.ops.aten.var_mean.correction(add_147, [2], correction = 0, keepdim = True)
        getitem_120: "f32[16, 128, 1]" = var_mean_38[0]
        getitem_121: "f32[16, 128, 1]" = var_mean_38[1];  var_mean_38 = None
        sub_51: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_147, getitem_121);  getitem_121 = None
        add_148: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_120, 1e-05);  getitem_120 = None
        rsqrt_38: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_148);  add_148 = None
        mul_143: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_51, rsqrt_38);  sub_51 = rsqrt_38 = None
        mul_144: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_143, arg325_1);  mul_143 = arg325_1 = None
        add_149: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_144, arg326_1);  mul_144 = arg326_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_418: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_149, [2048, 2560])
        permute_231: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg327_1, [1, 0]);  arg327_1 = None
        addmm_122: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg328_1, view_418, permute_231);  arg328_1 = view_418 = permute_231 = None
        view_419: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_122, [16, 128, 2560]);  addmm_122 = None
        view_420: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_419, [16, 128, -1, 80]);  view_419 = None
        permute_232: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_420, [0, 2, 1, 3]);  view_420 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_421: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_149, [2048, 2560])
        permute_233: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg329_1, [1, 0]);  arg329_1 = None
        addmm_123: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg330_1, view_421, permute_233);  arg330_1 = view_421 = permute_233 = None
        view_422: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_123, [16, 128, 2560]);  addmm_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_425: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_422, [16, 128, -1, 80]);  view_422 = None
        permute_235: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_425, [0, 2, 1, 3]);  view_425 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_423: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_149, [2048, 2560]);  add_149 = None
        permute_234: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg331_1, [1, 0]);  arg331_1 = None
        addmm_124: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg332_1, view_423, permute_234);  arg332_1 = view_423 = permute_234 = None
        view_424: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_124, [16, 128, 2560]);  addmm_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_426: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_424, [16, 128, -1, 80]);  view_424 = None
        permute_236: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_426, [0, 2, 1, 3]);  view_426 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_62: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_61: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_37: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_62, full_default_61);  full_default_62 = full_default_61 = None
        expand_66: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_37, [16, 32, 128, 128]);  where_37 = None
        _scaled_dot_product_efficient_attention_11 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_232, permute_235, permute_236, expand_66, False, scale = 0.11180339887498948);  permute_232 = permute_235 = permute_236 = expand_66 = None
        getitem_122: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_11[0];  _scaled_dot_product_efficient_attention_11 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_237: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_122, [0, 2, 1, 3]);  getitem_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_427: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_237, [16, 128, -1]);  permute_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_428: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_427, [2048, 2560]);  view_427 = None
        permute_238: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg333_1, [1, 0]);  arg333_1 = None
        addmm_125: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg334_1, view_428, permute_238);  arg334_1 = view_428 = permute_238 = None
        view_429: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_125, [16, 128, 2560]);  addmm_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_150: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_147, view_429);  add_147 = view_429 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_39 = torch.ops.aten.var_mean.correction(add_150, [2], correction = 0, keepdim = True)
        getitem_126: "f32[16, 128, 1]" = var_mean_39[0]
        getitem_127: "f32[16, 128, 1]" = var_mean_39[1];  var_mean_39 = None
        sub_52: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_150, getitem_127);  getitem_127 = None
        add_151: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_126, 1e-05);  getitem_126 = None
        rsqrt_39: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_151);  add_151 = None
        mul_145: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_52, rsqrt_39);  sub_52 = rsqrt_39 = None
        mul_146: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_145, arg335_1);  mul_145 = arg335_1 = None
        add_152: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_146, arg336_1);  mul_146 = arg336_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_430: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_152, [2048, 2560]);  add_152 = None
        permute_239: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg337_1, [1, 0]);  arg337_1 = None
        addmm_126: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg338_1, view_430, permute_239);  arg338_1 = view_430 = permute_239 = None
        view_431: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_126, [16, 128, 2560]);  addmm_126 = None
        view_432: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_431, [16, 128, -1, 80]);  view_431 = None
        permute_240: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_432, [0, 2, 1, 3]);  view_432 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_147: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_240, 0.334370152488211);  permute_240 = None
        expand_67: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_147, [16, 32, 128, 80]);  mul_147 = None
        clone_105: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_67, memory_format = torch.contiguous_format);  expand_67 = None
        view_439: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_105, [512, 128, 80]);  clone_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_433: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_241: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg339_1, [1, 0]);  arg339_1 = None
        addmm_127: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg340_1, view_433, permute_241);  arg340_1 = view_433 = permute_241 = None
        view_434: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_127, [16, 128, 2560]);  addmm_127 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_437: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_434, [16, 128, -1, 80]);  view_434 = None
        permute_243: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_437, [0, 2, 1, 3]);  view_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_245: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_243, [0, 1, 3, 2]);  permute_243 = None
        mul_148: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_245, 0.334370152488211);  permute_245 = None
        expand_68: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_148, [16, 32, 80, 128]);  mul_148 = None
        clone_106: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_68, memory_format = torch.contiguous_format);  expand_68 = None
        view_440: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_106, [512, 80, 128]);  clone_106 = None
        bmm_26: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_439, view_440);  view_439 = view_440 = None
        view_441: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_26, [16, 32, 128, 128]);  bmm_26 = None
        full_default_64: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_63: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_38: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_64, full_default_63);  full_default_64 = full_default_63 = None
        add_153: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_441, where_38);  view_441 = where_38 = None
        eq_13: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_153, -inf)
        logical_not_26: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_13);  eq_13 = None
        any_14: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_26, -1, True);  logical_not_26 = None
        logical_not_27: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_14);  any_14 = None
        full_default_65: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_13: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_153, [-1], True)
        sub_53: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_153, amax_13);  add_153 = amax_13 = None
        exp_13: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_53);  sub_53 = None
        sum_14: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_13, [-1], True)
        div_13: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_13, sum_14);  exp_13 = sum_14 = None
        where_39: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_27, full_default_65, div_13);  logical_not_27 = full_default_65 = div_13 = None
        expand_69: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_39, [16, 32, 128, 128]);  where_39 = None
        view_442: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_69, [512, 128, 128]);  expand_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_435: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_242: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg341_1, [1, 0]);  arg341_1 = None
        addmm_128: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg342_1, view_435, permute_242);  arg342_1 = view_435 = permute_242 = None
        view_436: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_128, [16, 128, 2560]);  addmm_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_438: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_436, [16, 128, -1, 80]);  view_436 = None
        permute_244: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_438, [0, 2, 1, 3]);  view_438 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_70: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_244, [16, 32, 128, 80]);  permute_244 = None
        clone_107: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_70, memory_format = torch.contiguous_format);  expand_70 = None
        view_443: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_107, [512, 128, 80]);  clone_107 = None
        bmm_27: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_442, view_443);  view_442 = view_443 = None
        view_444: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_27, [16, 32, 128, 80]);  bmm_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_246: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_444, [0, 2, 1, 3]);  view_444 = None
        clone_108: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_246, memory_format = torch.contiguous_format);  permute_246 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_445: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_108, [16, 128, -1]);  clone_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_446: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_445, [2048, 2560]);  view_445 = None
        permute_247: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg343_1, [1, 0]);  arg343_1 = None
        addmm_129: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg344_1, view_446, permute_247);  arg344_1 = view_446 = permute_247 = None
        view_447: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_129, [16, 128, 2560]);  addmm_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_154: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_150, view_447);  add_150 = view_447 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_40 = torch.ops.aten.var_mean.correction(add_154, [2], correction = 0, keepdim = True)
        getitem_128: "f32[16, 128, 1]" = var_mean_40[0]
        getitem_129: "f32[16, 128, 1]" = var_mean_40[1];  var_mean_40 = None
        sub_54: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_154, getitem_129);  getitem_129 = None
        add_155: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_128, 1e-05);  getitem_128 = None
        rsqrt_40: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_155);  add_155 = None
        mul_149: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_54, rsqrt_40);  sub_54 = rsqrt_40 = None
        mul_150: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_149, arg345_1);  mul_149 = arg345_1 = None
        add_156: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_150, arg346_1);  mul_150 = arg346_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_448: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_156, [2048, 2560]);  add_156 = None
        permute_248: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg347_1, [1, 0]);  arg347_1 = None
        addmm_130: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg348_1, view_448, permute_248);  arg348_1 = view_448 = permute_248 = None
        view_449: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_130, [16, 128, 10240]);  addmm_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_151: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_449, 0.5)
        mul_152: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_449, 0.7071067811865476);  view_449 = None
        erf_13: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_152);  mul_152 = None
        add_157: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_13, 1);  erf_13 = None
        mul_153: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_151, add_157);  mul_151 = add_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_450: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_153, [2048, 10240]);  mul_153 = None
        permute_249: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg349_1, [1, 0]);  arg349_1 = None
        addmm_131: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg350_1, view_450, permute_249);  arg350_1 = view_450 = permute_249 = None
        view_451: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_131, [16, 128, 2560]);  addmm_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_158: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_154, view_451);  add_154 = view_451 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_41 = torch.ops.aten.var_mean.correction(add_158, [2], correction = 0, keepdim = True)
        getitem_130: "f32[16, 128, 1]" = var_mean_41[0]
        getitem_131: "f32[16, 128, 1]" = var_mean_41[1];  var_mean_41 = None
        sub_55: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_158, getitem_131);  getitem_131 = None
        add_159: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_130, 1e-05);  getitem_130 = None
        rsqrt_41: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_159);  add_159 = None
        mul_154: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_55, rsqrt_41);  sub_55 = rsqrt_41 = None
        mul_155: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_154, arg351_1);  mul_154 = arg351_1 = None
        add_160: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_155, arg352_1);  mul_155 = arg352_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_452: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_160, [2048, 2560])
        permute_250: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg353_1, [1, 0]);  arg353_1 = None
        addmm_132: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg354_1, view_452, permute_250);  arg354_1 = view_452 = permute_250 = None
        view_453: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_132, [16, 128, 2560]);  addmm_132 = None
        view_454: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_453, [16, 128, -1, 80]);  view_453 = None
        permute_251: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_454, [0, 2, 1, 3]);  view_454 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_455: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_160, [2048, 2560])
        permute_252: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg355_1, [1, 0]);  arg355_1 = None
        addmm_133: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg356_1, view_455, permute_252);  arg356_1 = view_455 = permute_252 = None
        view_456: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_133, [16, 128, 2560]);  addmm_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_459: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_456, [16, 128, -1, 80]);  view_456 = None
        permute_254: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_459, [0, 2, 1, 3]);  view_459 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_457: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_160, [2048, 2560]);  add_160 = None
        permute_253: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg357_1, [1, 0]);  arg357_1 = None
        addmm_134: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg358_1, view_457, permute_253);  arg358_1 = view_457 = permute_253 = None
        view_458: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_134, [16, 128, 2560]);  addmm_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_460: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_458, [16, 128, -1, 80]);  view_458 = None
        permute_255: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_460, [0, 2, 1, 3]);  view_460 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_67: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_66: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_40: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_67, full_default_66);  full_default_67 = full_default_66 = None
        expand_71: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_40, [16, 32, 128, 128]);  where_40 = None
        _scaled_dot_product_efficient_attention_12 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_251, permute_254, permute_255, expand_71, False, scale = 0.11180339887498948);  permute_251 = permute_254 = permute_255 = expand_71 = None
        getitem_132: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_12[0];  _scaled_dot_product_efficient_attention_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_256: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_132, [0, 2, 1, 3]);  getitem_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_461: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_256, [16, 128, -1]);  permute_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_462: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_461, [2048, 2560]);  view_461 = None
        permute_257: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg359_1, [1, 0]);  arg359_1 = None
        addmm_135: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg360_1, view_462, permute_257);  arg360_1 = view_462 = permute_257 = None
        view_463: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_135, [16, 128, 2560]);  addmm_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_161: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_158, view_463);  add_158 = view_463 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_42 = torch.ops.aten.var_mean.correction(add_161, [2], correction = 0, keepdim = True)
        getitem_136: "f32[16, 128, 1]" = var_mean_42[0]
        getitem_137: "f32[16, 128, 1]" = var_mean_42[1];  var_mean_42 = None
        sub_56: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_161, getitem_137);  getitem_137 = None
        add_162: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_136, 1e-05);  getitem_136 = None
        rsqrt_42: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_162);  add_162 = None
        mul_156: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_56, rsqrt_42);  sub_56 = rsqrt_42 = None
        mul_157: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_156, arg361_1);  mul_156 = arg361_1 = None
        add_163: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_157, arg362_1);  mul_157 = arg362_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_464: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_163, [2048, 2560]);  add_163 = None
        permute_258: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg363_1, [1, 0]);  arg363_1 = None
        addmm_136: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg364_1, view_464, permute_258);  arg364_1 = view_464 = permute_258 = None
        view_465: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_136, [16, 128, 2560]);  addmm_136 = None
        view_466: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_465, [16, 128, -1, 80]);  view_465 = None
        permute_259: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_466, [0, 2, 1, 3]);  view_466 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_158: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_259, 0.334370152488211);  permute_259 = None
        expand_72: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_158, [16, 32, 128, 80]);  mul_158 = None
        clone_113: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_72, memory_format = torch.contiguous_format);  expand_72 = None
        view_473: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_113, [512, 128, 80]);  clone_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_467: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_260: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg365_1, [1, 0]);  arg365_1 = None
        addmm_137: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg366_1, view_467, permute_260);  arg366_1 = view_467 = permute_260 = None
        view_468: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_137, [16, 128, 2560]);  addmm_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_471: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_468, [16, 128, -1, 80]);  view_468 = None
        permute_262: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_471, [0, 2, 1, 3]);  view_471 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_264: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_262, [0, 1, 3, 2]);  permute_262 = None
        mul_159: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_264, 0.334370152488211);  permute_264 = None
        expand_73: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_159, [16, 32, 80, 128]);  mul_159 = None
        clone_114: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_73, memory_format = torch.contiguous_format);  expand_73 = None
        view_474: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_114, [512, 80, 128]);  clone_114 = None
        bmm_28: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_473, view_474);  view_473 = view_474 = None
        view_475: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_28, [16, 32, 128, 128]);  bmm_28 = None
        full_default_69: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_68: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_41: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_69, full_default_68);  full_default_69 = full_default_68 = None
        add_164: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_475, where_41);  view_475 = where_41 = None
        eq_14: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_164, -inf)
        logical_not_28: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_14);  eq_14 = None
        any_15: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_28, -1, True);  logical_not_28 = None
        logical_not_29: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_15);  any_15 = None
        full_default_70: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_14: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_164, [-1], True)
        sub_57: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_164, amax_14);  add_164 = amax_14 = None
        exp_14: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_57);  sub_57 = None
        sum_15: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_14, [-1], True)
        div_14: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_14, sum_15);  exp_14 = sum_15 = None
        where_42: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_29, full_default_70, div_14);  logical_not_29 = full_default_70 = div_14 = None
        expand_74: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_42, [16, 32, 128, 128]);  where_42 = None
        view_476: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_74, [512, 128, 128]);  expand_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_469: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_261: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg367_1, [1, 0]);  arg367_1 = None
        addmm_138: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg368_1, view_469, permute_261);  arg368_1 = view_469 = permute_261 = None
        view_470: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_138, [16, 128, 2560]);  addmm_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_472: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_470, [16, 128, -1, 80]);  view_470 = None
        permute_263: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_472, [0, 2, 1, 3]);  view_472 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_75: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_263, [16, 32, 128, 80]);  permute_263 = None
        clone_115: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_75, memory_format = torch.contiguous_format);  expand_75 = None
        view_477: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_115, [512, 128, 80]);  clone_115 = None
        bmm_29: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_476, view_477);  view_476 = view_477 = None
        view_478: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_29, [16, 32, 128, 80]);  bmm_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_265: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_478, [0, 2, 1, 3]);  view_478 = None
        clone_116: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_265, memory_format = torch.contiguous_format);  permute_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_479: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_116, [16, 128, -1]);  clone_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_480: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_479, [2048, 2560]);  view_479 = None
        permute_266: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg369_1, [1, 0]);  arg369_1 = None
        addmm_139: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg370_1, view_480, permute_266);  arg370_1 = view_480 = permute_266 = None
        view_481: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_139, [16, 128, 2560]);  addmm_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_165: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_161, view_481);  add_161 = view_481 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_43 = torch.ops.aten.var_mean.correction(add_165, [2], correction = 0, keepdim = True)
        getitem_138: "f32[16, 128, 1]" = var_mean_43[0]
        getitem_139: "f32[16, 128, 1]" = var_mean_43[1];  var_mean_43 = None
        sub_58: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_165, getitem_139);  getitem_139 = None
        add_166: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_138, 1e-05);  getitem_138 = None
        rsqrt_43: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_166);  add_166 = None
        mul_160: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_58, rsqrt_43);  sub_58 = rsqrt_43 = None
        mul_161: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_160, arg371_1);  mul_160 = arg371_1 = None
        add_167: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_161, arg372_1);  mul_161 = arg372_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_482: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_167, [2048, 2560]);  add_167 = None
        permute_267: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg373_1, [1, 0]);  arg373_1 = None
        addmm_140: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg374_1, view_482, permute_267);  arg374_1 = view_482 = permute_267 = None
        view_483: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_140, [16, 128, 10240]);  addmm_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_162: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_483, 0.5)
        mul_163: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_483, 0.7071067811865476);  view_483 = None
        erf_14: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_163);  mul_163 = None
        add_168: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_14, 1);  erf_14 = None
        mul_164: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_162, add_168);  mul_162 = add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_484: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_164, [2048, 10240]);  mul_164 = None
        permute_268: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg375_1, [1, 0]);  arg375_1 = None
        addmm_141: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg376_1, view_484, permute_268);  arg376_1 = view_484 = permute_268 = None
        view_485: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_141, [16, 128, 2560]);  addmm_141 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_169: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_165, view_485);  add_165 = view_485 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_44 = torch.ops.aten.var_mean.correction(add_169, [2], correction = 0, keepdim = True)
        getitem_140: "f32[16, 128, 1]" = var_mean_44[0]
        getitem_141: "f32[16, 128, 1]" = var_mean_44[1];  var_mean_44 = None
        sub_59: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_169, getitem_141);  getitem_141 = None
        add_170: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_140, 1e-05);  getitem_140 = None
        rsqrt_44: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_170);  add_170 = None
        mul_165: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_59, rsqrt_44);  sub_59 = rsqrt_44 = None
        mul_166: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_165, arg377_1);  mul_165 = arg377_1 = None
        add_171: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_166, arg378_1);  mul_166 = arg378_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_486: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_171, [2048, 2560])
        permute_269: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg379_1, [1, 0]);  arg379_1 = None
        addmm_142: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg380_1, view_486, permute_269);  arg380_1 = view_486 = permute_269 = None
        view_487: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_142, [16, 128, 2560]);  addmm_142 = None
        view_488: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_487, [16, 128, -1, 80]);  view_487 = None
        permute_270: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_488, [0, 2, 1, 3]);  view_488 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_489: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_171, [2048, 2560])
        permute_271: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg381_1, [1, 0]);  arg381_1 = None
        addmm_143: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg382_1, view_489, permute_271);  arg382_1 = view_489 = permute_271 = None
        view_490: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_143, [16, 128, 2560]);  addmm_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_493: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_490, [16, 128, -1, 80]);  view_490 = None
        permute_273: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_493, [0, 2, 1, 3]);  view_493 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_491: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_171, [2048, 2560]);  add_171 = None
        permute_272: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg383_1, [1, 0]);  arg383_1 = None
        addmm_144: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg384_1, view_491, permute_272);  arg384_1 = view_491 = permute_272 = None
        view_492: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_144, [16, 128, 2560]);  addmm_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_494: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_492, [16, 128, -1, 80]);  view_492 = None
        permute_274: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_494, [0, 2, 1, 3]);  view_494 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_72: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_71: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_43: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_72, full_default_71);  full_default_72 = full_default_71 = None
        expand_76: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_43, [16, 32, 128, 128]);  where_43 = None
        _scaled_dot_product_efficient_attention_13 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_270, permute_273, permute_274, expand_76, False, scale = 0.11180339887498948);  permute_270 = permute_273 = permute_274 = expand_76 = None
        getitem_142: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_13[0];  _scaled_dot_product_efficient_attention_13 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_275: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_142, [0, 2, 1, 3]);  getitem_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_495: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_275, [16, 128, -1]);  permute_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_496: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_495, [2048, 2560]);  view_495 = None
        permute_276: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg385_1, [1, 0]);  arg385_1 = None
        addmm_145: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg386_1, view_496, permute_276);  arg386_1 = view_496 = permute_276 = None
        view_497: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_145, [16, 128, 2560]);  addmm_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_172: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_169, view_497);  add_169 = view_497 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_45 = torch.ops.aten.var_mean.correction(add_172, [2], correction = 0, keepdim = True)
        getitem_146: "f32[16, 128, 1]" = var_mean_45[0]
        getitem_147: "f32[16, 128, 1]" = var_mean_45[1];  var_mean_45 = None
        sub_60: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_172, getitem_147);  getitem_147 = None
        add_173: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_146, 1e-05);  getitem_146 = None
        rsqrt_45: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_173);  add_173 = None
        mul_167: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_60, rsqrt_45);  sub_60 = rsqrt_45 = None
        mul_168: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_167, arg387_1);  mul_167 = arg387_1 = None
        add_174: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_168, arg388_1);  mul_168 = arg388_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_498: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_174, [2048, 2560]);  add_174 = None
        permute_277: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg389_1, [1, 0]);  arg389_1 = None
        addmm_146: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg390_1, view_498, permute_277);  arg390_1 = view_498 = permute_277 = None
        view_499: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_146, [16, 128, 2560]);  addmm_146 = None
        view_500: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_499, [16, 128, -1, 80]);  view_499 = None
        permute_278: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_500, [0, 2, 1, 3]);  view_500 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_169: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_278, 0.334370152488211);  permute_278 = None
        expand_77: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_169, [16, 32, 128, 80]);  mul_169 = None
        clone_121: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_77, memory_format = torch.contiguous_format);  expand_77 = None
        view_507: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_121, [512, 128, 80]);  clone_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_501: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_279: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg391_1, [1, 0]);  arg391_1 = None
        addmm_147: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg392_1, view_501, permute_279);  arg392_1 = view_501 = permute_279 = None
        view_502: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_147, [16, 128, 2560]);  addmm_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_505: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_502, [16, 128, -1, 80]);  view_502 = None
        permute_281: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_505, [0, 2, 1, 3]);  view_505 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_283: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_281, [0, 1, 3, 2]);  permute_281 = None
        mul_170: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_283, 0.334370152488211);  permute_283 = None
        expand_78: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_170, [16, 32, 80, 128]);  mul_170 = None
        clone_122: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_78, memory_format = torch.contiguous_format);  expand_78 = None
        view_508: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_122, [512, 80, 128]);  clone_122 = None
        bmm_30: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_507, view_508);  view_507 = view_508 = None
        view_509: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_30, [16, 32, 128, 128]);  bmm_30 = None
        full_default_74: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_73: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_44: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_74, full_default_73);  full_default_74 = full_default_73 = None
        add_175: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_509, where_44);  view_509 = where_44 = None
        eq_15: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_175, -inf)
        logical_not_30: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_15);  eq_15 = None
        any_16: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_30, -1, True);  logical_not_30 = None
        logical_not_31: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_16);  any_16 = None
        full_default_75: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_15: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_175, [-1], True)
        sub_61: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_175, amax_15);  add_175 = amax_15 = None
        exp_15: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_61);  sub_61 = None
        sum_16: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_15, [-1], True)
        div_15: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_15, sum_16);  exp_15 = sum_16 = None
        where_45: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_31, full_default_75, div_15);  logical_not_31 = full_default_75 = div_15 = None
        expand_79: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_45, [16, 32, 128, 128]);  where_45 = None
        view_510: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_79, [512, 128, 128]);  expand_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_503: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_280: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg393_1, [1, 0]);  arg393_1 = None
        addmm_148: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg394_1, view_503, permute_280);  arg394_1 = view_503 = permute_280 = None
        view_504: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_148, [16, 128, 2560]);  addmm_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_506: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_504, [16, 128, -1, 80]);  view_504 = None
        permute_282: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_506, [0, 2, 1, 3]);  view_506 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_80: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_282, [16, 32, 128, 80]);  permute_282 = None
        clone_123: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_80, memory_format = torch.contiguous_format);  expand_80 = None
        view_511: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_123, [512, 128, 80]);  clone_123 = None
        bmm_31: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_510, view_511);  view_510 = view_511 = None
        view_512: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_31, [16, 32, 128, 80]);  bmm_31 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_284: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_512, [0, 2, 1, 3]);  view_512 = None
        clone_124: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_284, memory_format = torch.contiguous_format);  permute_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_513: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_124, [16, 128, -1]);  clone_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_514: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_513, [2048, 2560]);  view_513 = None
        permute_285: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg395_1, [1, 0]);  arg395_1 = None
        addmm_149: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg396_1, view_514, permute_285);  arg396_1 = view_514 = permute_285 = None
        view_515: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_149, [16, 128, 2560]);  addmm_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_176: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_172, view_515);  add_172 = view_515 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_46 = torch.ops.aten.var_mean.correction(add_176, [2], correction = 0, keepdim = True)
        getitem_148: "f32[16, 128, 1]" = var_mean_46[0]
        getitem_149: "f32[16, 128, 1]" = var_mean_46[1];  var_mean_46 = None
        sub_62: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_176, getitem_149);  getitem_149 = None
        add_177: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_148, 1e-05);  getitem_148 = None
        rsqrt_46: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_177);  add_177 = None
        mul_171: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_62, rsqrt_46);  sub_62 = rsqrt_46 = None
        mul_172: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_171, arg397_1);  mul_171 = arg397_1 = None
        add_178: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_172, arg398_1);  mul_172 = arg398_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_516: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_178, [2048, 2560]);  add_178 = None
        permute_286: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg399_1, [1, 0]);  arg399_1 = None
        addmm_150: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg400_1, view_516, permute_286);  arg400_1 = view_516 = permute_286 = None
        view_517: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_150, [16, 128, 10240]);  addmm_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_173: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_517, 0.5)
        mul_174: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_517, 0.7071067811865476);  view_517 = None
        erf_15: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_174);  mul_174 = None
        add_179: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_15, 1);  erf_15 = None
        mul_175: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_173, add_179);  mul_173 = add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_518: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_175, [2048, 10240]);  mul_175 = None
        permute_287: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg401_1, [1, 0]);  arg401_1 = None
        addmm_151: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg402_1, view_518, permute_287);  arg402_1 = view_518 = permute_287 = None
        view_519: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_151, [16, 128, 2560]);  addmm_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_180: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_176, view_519);  add_176 = view_519 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_47 = torch.ops.aten.var_mean.correction(add_180, [2], correction = 0, keepdim = True)
        getitem_150: "f32[16, 128, 1]" = var_mean_47[0]
        getitem_151: "f32[16, 128, 1]" = var_mean_47[1];  var_mean_47 = None
        sub_63: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_180, getitem_151);  getitem_151 = None
        add_181: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_150, 1e-05);  getitem_150 = None
        rsqrt_47: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_181);  add_181 = None
        mul_176: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_63, rsqrt_47);  sub_63 = rsqrt_47 = None
        mul_177: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_176, arg403_1);  mul_176 = arg403_1 = None
        add_182: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_177, arg404_1);  mul_177 = arg404_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_520: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_182, [2048, 2560])
        permute_288: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg405_1, [1, 0]);  arg405_1 = None
        addmm_152: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg406_1, view_520, permute_288);  arg406_1 = view_520 = permute_288 = None
        view_521: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_152, [16, 128, 2560]);  addmm_152 = None
        view_522: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_521, [16, 128, -1, 80]);  view_521 = None
        permute_289: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_522, [0, 2, 1, 3]);  view_522 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_523: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_182, [2048, 2560])
        permute_290: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg407_1, [1, 0]);  arg407_1 = None
        addmm_153: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg408_1, view_523, permute_290);  arg408_1 = view_523 = permute_290 = None
        view_524: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_153, [16, 128, 2560]);  addmm_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_527: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_524, [16, 128, -1, 80]);  view_524 = None
        permute_292: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_527, [0, 2, 1, 3]);  view_527 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_525: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_182, [2048, 2560]);  add_182 = None
        permute_291: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg409_1, [1, 0]);  arg409_1 = None
        addmm_154: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg410_1, view_525, permute_291);  arg410_1 = view_525 = permute_291 = None
        view_526: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_154, [16, 128, 2560]);  addmm_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_528: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_526, [16, 128, -1, 80]);  view_526 = None
        permute_293: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_528, [0, 2, 1, 3]);  view_528 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_77: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_76: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_46: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_77, full_default_76);  full_default_77 = full_default_76 = None
        expand_81: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_46, [16, 32, 128, 128]);  where_46 = None
        _scaled_dot_product_efficient_attention_14 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_289, permute_292, permute_293, expand_81, False, scale = 0.11180339887498948);  permute_289 = permute_292 = permute_293 = expand_81 = None
        getitem_152: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_14[0];  _scaled_dot_product_efficient_attention_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_294: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_152, [0, 2, 1, 3]);  getitem_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_529: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_294, [16, 128, -1]);  permute_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_530: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_529, [2048, 2560]);  view_529 = None
        permute_295: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg411_1, [1, 0]);  arg411_1 = None
        addmm_155: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg412_1, view_530, permute_295);  arg412_1 = view_530 = permute_295 = None
        view_531: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_155, [16, 128, 2560]);  addmm_155 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_183: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_180, view_531);  add_180 = view_531 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_48 = torch.ops.aten.var_mean.correction(add_183, [2], correction = 0, keepdim = True)
        getitem_156: "f32[16, 128, 1]" = var_mean_48[0]
        getitem_157: "f32[16, 128, 1]" = var_mean_48[1];  var_mean_48 = None
        sub_64: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_183, getitem_157);  getitem_157 = None
        add_184: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_156, 1e-05);  getitem_156 = None
        rsqrt_48: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_184);  add_184 = None
        mul_178: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_64, rsqrt_48);  sub_64 = rsqrt_48 = None
        mul_179: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_178, arg413_1);  mul_178 = arg413_1 = None
        add_185: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_179, arg414_1);  mul_179 = arg414_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_532: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_185, [2048, 2560]);  add_185 = None
        permute_296: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg415_1, [1, 0]);  arg415_1 = None
        addmm_156: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg416_1, view_532, permute_296);  arg416_1 = view_532 = permute_296 = None
        view_533: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_156, [16, 128, 2560]);  addmm_156 = None
        view_534: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_533, [16, 128, -1, 80]);  view_533 = None
        permute_297: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_534, [0, 2, 1, 3]);  view_534 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_180: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_297, 0.334370152488211);  permute_297 = None
        expand_82: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_180, [16, 32, 128, 80]);  mul_180 = None
        clone_129: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_82, memory_format = torch.contiguous_format);  expand_82 = None
        view_541: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_129, [512, 128, 80]);  clone_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_535: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_298: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg417_1, [1, 0]);  arg417_1 = None
        addmm_157: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg418_1, view_535, permute_298);  arg418_1 = view_535 = permute_298 = None
        view_536: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_157, [16, 128, 2560]);  addmm_157 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_539: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_536, [16, 128, -1, 80]);  view_536 = None
        permute_300: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_539, [0, 2, 1, 3]);  view_539 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_302: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_300, [0, 1, 3, 2]);  permute_300 = None
        mul_181: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_302, 0.334370152488211);  permute_302 = None
        expand_83: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_181, [16, 32, 80, 128]);  mul_181 = None
        clone_130: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_83, memory_format = torch.contiguous_format);  expand_83 = None
        view_542: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_130, [512, 80, 128]);  clone_130 = None
        bmm_32: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_541, view_542);  view_541 = view_542 = None
        view_543: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_32, [16, 32, 128, 128]);  bmm_32 = None
        full_default_79: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_78: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_47: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_79, full_default_78);  full_default_79 = full_default_78 = None
        add_186: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_543, where_47);  view_543 = where_47 = None
        eq_16: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_186, -inf)
        logical_not_32: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_16);  eq_16 = None
        any_17: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_32, -1, True);  logical_not_32 = None
        logical_not_33: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_17);  any_17 = None
        full_default_80: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_16: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_186, [-1], True)
        sub_65: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_186, amax_16);  add_186 = amax_16 = None
        exp_16: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_65);  sub_65 = None
        sum_17: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_16, [-1], True)
        div_16: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_16, sum_17);  exp_16 = sum_17 = None
        where_48: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_33, full_default_80, div_16);  logical_not_33 = full_default_80 = div_16 = None
        expand_84: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_48, [16, 32, 128, 128]);  where_48 = None
        view_544: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_84, [512, 128, 128]);  expand_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_537: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_299: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg419_1, [1, 0]);  arg419_1 = None
        addmm_158: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg420_1, view_537, permute_299);  arg420_1 = view_537 = permute_299 = None
        view_538: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_158, [16, 128, 2560]);  addmm_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_540: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_538, [16, 128, -1, 80]);  view_538 = None
        permute_301: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_540, [0, 2, 1, 3]);  view_540 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_85: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_301, [16, 32, 128, 80]);  permute_301 = None
        clone_131: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_85, memory_format = torch.contiguous_format);  expand_85 = None
        view_545: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_131, [512, 128, 80]);  clone_131 = None
        bmm_33: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_544, view_545);  view_544 = view_545 = None
        view_546: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_33, [16, 32, 128, 80]);  bmm_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_303: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_546, [0, 2, 1, 3]);  view_546 = None
        clone_132: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_303, memory_format = torch.contiguous_format);  permute_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_547: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_132, [16, 128, -1]);  clone_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_548: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_547, [2048, 2560]);  view_547 = None
        permute_304: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg421_1, [1, 0]);  arg421_1 = None
        addmm_159: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg422_1, view_548, permute_304);  arg422_1 = view_548 = permute_304 = None
        view_549: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_159, [16, 128, 2560]);  addmm_159 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_187: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_183, view_549);  add_183 = view_549 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_49 = torch.ops.aten.var_mean.correction(add_187, [2], correction = 0, keepdim = True)
        getitem_158: "f32[16, 128, 1]" = var_mean_49[0]
        getitem_159: "f32[16, 128, 1]" = var_mean_49[1];  var_mean_49 = None
        sub_66: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_187, getitem_159);  getitem_159 = None
        add_188: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_158, 1e-05);  getitem_158 = None
        rsqrt_49: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_188);  add_188 = None
        mul_182: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_66, rsqrt_49);  sub_66 = rsqrt_49 = None
        mul_183: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_182, arg423_1);  mul_182 = arg423_1 = None
        add_189: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_183, arg424_1);  mul_183 = arg424_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_550: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_189, [2048, 2560]);  add_189 = None
        permute_305: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg425_1, [1, 0]);  arg425_1 = None
        addmm_160: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg426_1, view_550, permute_305);  arg426_1 = view_550 = permute_305 = None
        view_551: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_160, [16, 128, 10240]);  addmm_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_184: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_551, 0.5)
        mul_185: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_551, 0.7071067811865476);  view_551 = None
        erf_16: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_185);  mul_185 = None
        add_190: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_16, 1);  erf_16 = None
        mul_186: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_184, add_190);  mul_184 = add_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_552: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_186, [2048, 10240]);  mul_186 = None
        permute_306: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg427_1, [1, 0]);  arg427_1 = None
        addmm_161: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg428_1, view_552, permute_306);  arg428_1 = view_552 = permute_306 = None
        view_553: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_161, [16, 128, 2560]);  addmm_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_191: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_187, view_553);  add_187 = view_553 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_50 = torch.ops.aten.var_mean.correction(add_191, [2], correction = 0, keepdim = True)
        getitem_160: "f32[16, 128, 1]" = var_mean_50[0]
        getitem_161: "f32[16, 128, 1]" = var_mean_50[1];  var_mean_50 = None
        sub_67: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_191, getitem_161);  getitem_161 = None
        add_192: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_160, 1e-05);  getitem_160 = None
        rsqrt_50: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_192);  add_192 = None
        mul_187: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_67, rsqrt_50);  sub_67 = rsqrt_50 = None
        mul_188: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_187, arg429_1);  mul_187 = arg429_1 = None
        add_193: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_188, arg430_1);  mul_188 = arg430_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_554: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_193, [2048, 2560])
        permute_307: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg431_1, [1, 0]);  arg431_1 = None
        addmm_162: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg432_1, view_554, permute_307);  arg432_1 = view_554 = permute_307 = None
        view_555: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_162, [16, 128, 2560]);  addmm_162 = None
        view_556: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_555, [16, 128, -1, 80]);  view_555 = None
        permute_308: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_556, [0, 2, 1, 3]);  view_556 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_557: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_193, [2048, 2560])
        permute_309: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg433_1, [1, 0]);  arg433_1 = None
        addmm_163: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg434_1, view_557, permute_309);  arg434_1 = view_557 = permute_309 = None
        view_558: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_163, [16, 128, 2560]);  addmm_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_561: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_558, [16, 128, -1, 80]);  view_558 = None
        permute_311: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_561, [0, 2, 1, 3]);  view_561 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_559: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_193, [2048, 2560]);  add_193 = None
        permute_310: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg435_1, [1, 0]);  arg435_1 = None
        addmm_164: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg436_1, view_559, permute_310);  arg436_1 = view_559 = permute_310 = None
        view_560: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_164, [16, 128, 2560]);  addmm_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_562: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_560, [16, 128, -1, 80]);  view_560 = None
        permute_312: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_562, [0, 2, 1, 3]);  view_562 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_82: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_81: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_49: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_82, full_default_81);  full_default_82 = full_default_81 = None
        expand_86: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_49, [16, 32, 128, 128]);  where_49 = None
        _scaled_dot_product_efficient_attention_15 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_308, permute_311, permute_312, expand_86, False, scale = 0.11180339887498948);  permute_308 = permute_311 = permute_312 = expand_86 = None
        getitem_162: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_15[0];  _scaled_dot_product_efficient_attention_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_313: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_162, [0, 2, 1, 3]);  getitem_162 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_563: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_313, [16, 128, -1]);  permute_313 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_564: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_563, [2048, 2560]);  view_563 = None
        permute_314: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg437_1, [1, 0]);  arg437_1 = None
        addmm_165: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg438_1, view_564, permute_314);  arg438_1 = view_564 = permute_314 = None
        view_565: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_165, [16, 128, 2560]);  addmm_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_194: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_191, view_565);  add_191 = view_565 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_51 = torch.ops.aten.var_mean.correction(add_194, [2], correction = 0, keepdim = True)
        getitem_166: "f32[16, 128, 1]" = var_mean_51[0]
        getitem_167: "f32[16, 128, 1]" = var_mean_51[1];  var_mean_51 = None
        sub_68: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_194, getitem_167);  getitem_167 = None
        add_195: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_166, 1e-05);  getitem_166 = None
        rsqrt_51: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_195);  add_195 = None
        mul_189: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_68, rsqrt_51);  sub_68 = rsqrt_51 = None
        mul_190: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_189, arg439_1);  mul_189 = arg439_1 = None
        add_196: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_190, arg440_1);  mul_190 = arg440_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_566: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_196, [2048, 2560]);  add_196 = None
        permute_315: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg441_1, [1, 0]);  arg441_1 = None
        addmm_166: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg442_1, view_566, permute_315);  arg442_1 = view_566 = permute_315 = None
        view_567: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_166, [16, 128, 2560]);  addmm_166 = None
        view_568: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_567, [16, 128, -1, 80]);  view_567 = None
        permute_316: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_568, [0, 2, 1, 3]);  view_568 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_191: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_316, 0.334370152488211);  permute_316 = None
        expand_87: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_191, [16, 32, 128, 80]);  mul_191 = None
        clone_137: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_87, memory_format = torch.contiguous_format);  expand_87 = None
        view_575: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_137, [512, 128, 80]);  clone_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_569: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_317: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg443_1, [1, 0]);  arg443_1 = None
        addmm_167: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg444_1, view_569, permute_317);  arg444_1 = view_569 = permute_317 = None
        view_570: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_167, [16, 128, 2560]);  addmm_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_573: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_570, [16, 128, -1, 80]);  view_570 = None
        permute_319: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_573, [0, 2, 1, 3]);  view_573 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_321: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_319, [0, 1, 3, 2]);  permute_319 = None
        mul_192: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_321, 0.334370152488211);  permute_321 = None
        expand_88: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_192, [16, 32, 80, 128]);  mul_192 = None
        clone_138: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_88, memory_format = torch.contiguous_format);  expand_88 = None
        view_576: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_138, [512, 80, 128]);  clone_138 = None
        bmm_34: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_575, view_576);  view_575 = view_576 = None
        view_577: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_34, [16, 32, 128, 128]);  bmm_34 = None
        full_default_84: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_83: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_50: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_84, full_default_83);  full_default_84 = full_default_83 = None
        add_197: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_577, where_50);  view_577 = where_50 = None
        eq_17: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_197, -inf)
        logical_not_34: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_17);  eq_17 = None
        any_18: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_34, -1, True);  logical_not_34 = None
        logical_not_35: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_18);  any_18 = None
        full_default_85: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_17: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_197, [-1], True)
        sub_69: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_197, amax_17);  add_197 = amax_17 = None
        exp_17: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_69);  sub_69 = None
        sum_18: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_17, [-1], True)
        div_17: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_17, sum_18);  exp_17 = sum_18 = None
        where_51: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_35, full_default_85, div_17);  logical_not_35 = full_default_85 = div_17 = None
        expand_89: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_51, [16, 32, 128, 128]);  where_51 = None
        view_578: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_89, [512, 128, 128]);  expand_89 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_571: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_318: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg445_1, [1, 0]);  arg445_1 = None
        addmm_168: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg446_1, view_571, permute_318);  arg446_1 = view_571 = permute_318 = None
        view_572: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_168, [16, 128, 2560]);  addmm_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_574: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_572, [16, 128, -1, 80]);  view_572 = None
        permute_320: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_574, [0, 2, 1, 3]);  view_574 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_90: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_320, [16, 32, 128, 80]);  permute_320 = None
        clone_139: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_90, memory_format = torch.contiguous_format);  expand_90 = None
        view_579: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_139, [512, 128, 80]);  clone_139 = None
        bmm_35: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_578, view_579);  view_578 = view_579 = None
        view_580: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_35, [16, 32, 128, 80]);  bmm_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_322: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_580, [0, 2, 1, 3]);  view_580 = None
        clone_140: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_322, memory_format = torch.contiguous_format);  permute_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_581: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_140, [16, 128, -1]);  clone_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_582: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_581, [2048, 2560]);  view_581 = None
        permute_323: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg447_1, [1, 0]);  arg447_1 = None
        addmm_169: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg448_1, view_582, permute_323);  arg448_1 = view_582 = permute_323 = None
        view_583: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_169, [16, 128, 2560]);  addmm_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_198: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_194, view_583);  add_194 = view_583 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_52 = torch.ops.aten.var_mean.correction(add_198, [2], correction = 0, keepdim = True)
        getitem_168: "f32[16, 128, 1]" = var_mean_52[0]
        getitem_169: "f32[16, 128, 1]" = var_mean_52[1];  var_mean_52 = None
        sub_70: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_198, getitem_169);  getitem_169 = None
        add_199: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_168, 1e-05);  getitem_168 = None
        rsqrt_52: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_199);  add_199 = None
        mul_193: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_70, rsqrt_52);  sub_70 = rsqrt_52 = None
        mul_194: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_193, arg449_1);  mul_193 = arg449_1 = None
        add_200: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_194, arg450_1);  mul_194 = arg450_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_584: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_200, [2048, 2560]);  add_200 = None
        permute_324: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg451_1, [1, 0]);  arg451_1 = None
        addmm_170: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg452_1, view_584, permute_324);  arg452_1 = view_584 = permute_324 = None
        view_585: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_170, [16, 128, 10240]);  addmm_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_195: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_585, 0.5)
        mul_196: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_585, 0.7071067811865476);  view_585 = None
        erf_17: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_196);  mul_196 = None
        add_201: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_17, 1);  erf_17 = None
        mul_197: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_195, add_201);  mul_195 = add_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_586: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_197, [2048, 10240]);  mul_197 = None
        permute_325: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg453_1, [1, 0]);  arg453_1 = None
        addmm_171: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg454_1, view_586, permute_325);  arg454_1 = view_586 = permute_325 = None
        view_587: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_171, [16, 128, 2560]);  addmm_171 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_202: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_198, view_587);  add_198 = view_587 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_53 = torch.ops.aten.var_mean.correction(add_202, [2], correction = 0, keepdim = True)
        getitem_170: "f32[16, 128, 1]" = var_mean_53[0]
        getitem_171: "f32[16, 128, 1]" = var_mean_53[1];  var_mean_53 = None
        sub_71: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_202, getitem_171);  getitem_171 = None
        add_203: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_170, 1e-05);  getitem_170 = None
        rsqrt_53: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_203);  add_203 = None
        mul_198: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_71, rsqrt_53);  sub_71 = rsqrt_53 = None
        mul_199: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_198, arg455_1);  mul_198 = arg455_1 = None
        add_204: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_199, arg456_1);  mul_199 = arg456_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_588: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_204, [2048, 2560])
        permute_326: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg457_1, [1, 0]);  arg457_1 = None
        addmm_172: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg458_1, view_588, permute_326);  arg458_1 = view_588 = permute_326 = None
        view_589: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_172, [16, 128, 2560]);  addmm_172 = None
        view_590: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_589, [16, 128, -1, 80]);  view_589 = None
        permute_327: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_590, [0, 2, 1, 3]);  view_590 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_591: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_204, [2048, 2560])
        permute_328: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg459_1, [1, 0]);  arg459_1 = None
        addmm_173: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg460_1, view_591, permute_328);  arg460_1 = view_591 = permute_328 = None
        view_592: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_173, [16, 128, 2560]);  addmm_173 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_595: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_592, [16, 128, -1, 80]);  view_592 = None
        permute_330: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_595, [0, 2, 1, 3]);  view_595 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_593: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_204, [2048, 2560]);  add_204 = None
        permute_329: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg461_1, [1, 0]);  arg461_1 = None
        addmm_174: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg462_1, view_593, permute_329);  arg462_1 = view_593 = permute_329 = None
        view_594: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_174, [16, 128, 2560]);  addmm_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_596: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_594, [16, 128, -1, 80]);  view_594 = None
        permute_331: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_596, [0, 2, 1, 3]);  view_596 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_87: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_86: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_52: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_87, full_default_86);  full_default_87 = full_default_86 = None
        expand_91: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_52, [16, 32, 128, 128]);  where_52 = None
        _scaled_dot_product_efficient_attention_16 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_327, permute_330, permute_331, expand_91, False, scale = 0.11180339887498948);  permute_327 = permute_330 = permute_331 = expand_91 = None
        getitem_172: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_16[0];  _scaled_dot_product_efficient_attention_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_332: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_172, [0, 2, 1, 3]);  getitem_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_597: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_332, [16, 128, -1]);  permute_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_598: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_597, [2048, 2560]);  view_597 = None
        permute_333: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg463_1, [1, 0]);  arg463_1 = None
        addmm_175: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg464_1, view_598, permute_333);  arg464_1 = view_598 = permute_333 = None
        view_599: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_175, [16, 128, 2560]);  addmm_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_205: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_202, view_599);  add_202 = view_599 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_54 = torch.ops.aten.var_mean.correction(add_205, [2], correction = 0, keepdim = True)
        getitem_176: "f32[16, 128, 1]" = var_mean_54[0]
        getitem_177: "f32[16, 128, 1]" = var_mean_54[1];  var_mean_54 = None
        sub_72: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_205, getitem_177);  getitem_177 = None
        add_206: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_176, 1e-05);  getitem_176 = None
        rsqrt_54: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_206);  add_206 = None
        mul_200: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_72, rsqrt_54);  sub_72 = rsqrt_54 = None
        mul_201: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_200, arg465_1);  mul_200 = arg465_1 = None
        add_207: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_201, arg466_1);  mul_201 = arg466_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_600: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_207, [2048, 2560]);  add_207 = None
        permute_334: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg467_1, [1, 0]);  arg467_1 = None
        addmm_176: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg468_1, view_600, permute_334);  arg468_1 = view_600 = permute_334 = None
        view_601: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_176, [16, 128, 2560]);  addmm_176 = None
        view_602: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_601, [16, 128, -1, 80]);  view_601 = None
        permute_335: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_602, [0, 2, 1, 3]);  view_602 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_202: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_335, 0.334370152488211);  permute_335 = None
        expand_92: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_202, [16, 32, 128, 80]);  mul_202 = None
        clone_145: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_92, memory_format = torch.contiguous_format);  expand_92 = None
        view_609: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_145, [512, 128, 80]);  clone_145 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_603: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_336: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg469_1, [1, 0]);  arg469_1 = None
        addmm_177: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg470_1, view_603, permute_336);  arg470_1 = view_603 = permute_336 = None
        view_604: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_177, [16, 128, 2560]);  addmm_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_607: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_604, [16, 128, -1, 80]);  view_604 = None
        permute_338: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_607, [0, 2, 1, 3]);  view_607 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_340: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_338, [0, 1, 3, 2]);  permute_338 = None
        mul_203: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_340, 0.334370152488211);  permute_340 = None
        expand_93: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_203, [16, 32, 80, 128]);  mul_203 = None
        clone_146: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_93, memory_format = torch.contiguous_format);  expand_93 = None
        view_610: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_146, [512, 80, 128]);  clone_146 = None
        bmm_36: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_609, view_610);  view_609 = view_610 = None
        view_611: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_36, [16, 32, 128, 128]);  bmm_36 = None
        full_default_89: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_88: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_53: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_89, full_default_88);  full_default_89 = full_default_88 = None
        add_208: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_611, where_53);  view_611 = where_53 = None
        eq_18: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_208, -inf)
        logical_not_36: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_18);  eq_18 = None
        any_19: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_36, -1, True);  logical_not_36 = None
        logical_not_37: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_19);  any_19 = None
        full_default_90: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_18: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_208, [-1], True)
        sub_73: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_208, amax_18);  add_208 = amax_18 = None
        exp_18: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_73);  sub_73 = None
        sum_19: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_18, [-1], True)
        div_18: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_18, sum_19);  exp_18 = sum_19 = None
        where_54: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_37, full_default_90, div_18);  logical_not_37 = full_default_90 = div_18 = None
        expand_94: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_54, [16, 32, 128, 128]);  where_54 = None
        view_612: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_94, [512, 128, 128]);  expand_94 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_605: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_337: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg471_1, [1, 0]);  arg471_1 = None
        addmm_178: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg472_1, view_605, permute_337);  arg472_1 = view_605 = permute_337 = None
        view_606: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_178, [16, 128, 2560]);  addmm_178 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_608: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_606, [16, 128, -1, 80]);  view_606 = None
        permute_339: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_608, [0, 2, 1, 3]);  view_608 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_95: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_339, [16, 32, 128, 80]);  permute_339 = None
        clone_147: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_95, memory_format = torch.contiguous_format);  expand_95 = None
        view_613: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_147, [512, 128, 80]);  clone_147 = None
        bmm_37: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_612, view_613);  view_612 = view_613 = None
        view_614: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_37, [16, 32, 128, 80]);  bmm_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_341: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_614, [0, 2, 1, 3]);  view_614 = None
        clone_148: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_341, memory_format = torch.contiguous_format);  permute_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_615: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_148, [16, 128, -1]);  clone_148 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_616: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_615, [2048, 2560]);  view_615 = None
        permute_342: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg473_1, [1, 0]);  arg473_1 = None
        addmm_179: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg474_1, view_616, permute_342);  arg474_1 = view_616 = permute_342 = None
        view_617: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_179, [16, 128, 2560]);  addmm_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_209: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_205, view_617);  add_205 = view_617 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_55 = torch.ops.aten.var_mean.correction(add_209, [2], correction = 0, keepdim = True)
        getitem_178: "f32[16, 128, 1]" = var_mean_55[0]
        getitem_179: "f32[16, 128, 1]" = var_mean_55[1];  var_mean_55 = None
        sub_74: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_209, getitem_179);  getitem_179 = None
        add_210: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_178, 1e-05);  getitem_178 = None
        rsqrt_55: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_210);  add_210 = None
        mul_204: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_74, rsqrt_55);  sub_74 = rsqrt_55 = None
        mul_205: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_204, arg475_1);  mul_204 = arg475_1 = None
        add_211: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_205, arg476_1);  mul_205 = arg476_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_618: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_211, [2048, 2560]);  add_211 = None
        permute_343: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg477_1, [1, 0]);  arg477_1 = None
        addmm_180: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg478_1, view_618, permute_343);  arg478_1 = view_618 = permute_343 = None
        view_619: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_180, [16, 128, 10240]);  addmm_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_206: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_619, 0.5)
        mul_207: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_619, 0.7071067811865476);  view_619 = None
        erf_18: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_207);  mul_207 = None
        add_212: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_18, 1);  erf_18 = None
        mul_208: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_206, add_212);  mul_206 = add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_620: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_208, [2048, 10240]);  mul_208 = None
        permute_344: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg479_1, [1, 0]);  arg479_1 = None
        addmm_181: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg480_1, view_620, permute_344);  arg480_1 = view_620 = permute_344 = None
        view_621: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_181, [16, 128, 2560]);  addmm_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_213: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_209, view_621);  add_209 = view_621 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_56 = torch.ops.aten.var_mean.correction(add_213, [2], correction = 0, keepdim = True)
        getitem_180: "f32[16, 128, 1]" = var_mean_56[0]
        getitem_181: "f32[16, 128, 1]" = var_mean_56[1];  var_mean_56 = None
        sub_75: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_213, getitem_181);  getitem_181 = None
        add_214: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_180, 1e-05);  getitem_180 = None
        rsqrt_56: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_214);  add_214 = None
        mul_209: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_75, rsqrt_56);  sub_75 = rsqrt_56 = None
        mul_210: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_209, arg481_1);  mul_209 = arg481_1 = None
        add_215: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_210, arg482_1);  mul_210 = arg482_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_622: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_215, [2048, 2560])
        permute_345: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg483_1, [1, 0]);  arg483_1 = None
        addmm_182: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg484_1, view_622, permute_345);  arg484_1 = view_622 = permute_345 = None
        view_623: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_182, [16, 128, 2560]);  addmm_182 = None
        view_624: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_623, [16, 128, -1, 80]);  view_623 = None
        permute_346: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_624, [0, 2, 1, 3]);  view_624 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_625: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_215, [2048, 2560])
        permute_347: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg485_1, [1, 0]);  arg485_1 = None
        addmm_183: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg486_1, view_625, permute_347);  arg486_1 = view_625 = permute_347 = None
        view_626: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_183, [16, 128, 2560]);  addmm_183 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_629: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_626, [16, 128, -1, 80]);  view_626 = None
        permute_349: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_629, [0, 2, 1, 3]);  view_629 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_627: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_215, [2048, 2560]);  add_215 = None
        permute_348: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg487_1, [1, 0]);  arg487_1 = None
        addmm_184: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg488_1, view_627, permute_348);  arg488_1 = view_627 = permute_348 = None
        view_628: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_184, [16, 128, 2560]);  addmm_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_630: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_628, [16, 128, -1, 80]);  view_628 = None
        permute_350: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_630, [0, 2, 1, 3]);  view_630 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_92: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_91: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_55: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_92, full_default_91);  full_default_92 = full_default_91 = None
        expand_96: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_55, [16, 32, 128, 128]);  where_55 = None
        _scaled_dot_product_efficient_attention_17 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_346, permute_349, permute_350, expand_96, False, scale = 0.11180339887498948);  permute_346 = permute_349 = permute_350 = expand_96 = None
        getitem_182: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_17[0];  _scaled_dot_product_efficient_attention_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_351: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_182, [0, 2, 1, 3]);  getitem_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_631: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_351, [16, 128, -1]);  permute_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_632: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_631, [2048, 2560]);  view_631 = None
        permute_352: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg489_1, [1, 0]);  arg489_1 = None
        addmm_185: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg490_1, view_632, permute_352);  arg490_1 = view_632 = permute_352 = None
        view_633: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_185, [16, 128, 2560]);  addmm_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_216: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_213, view_633);  add_213 = view_633 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_57 = torch.ops.aten.var_mean.correction(add_216, [2], correction = 0, keepdim = True)
        getitem_186: "f32[16, 128, 1]" = var_mean_57[0]
        getitem_187: "f32[16, 128, 1]" = var_mean_57[1];  var_mean_57 = None
        sub_76: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_216, getitem_187);  getitem_187 = None
        add_217: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_186, 1e-05);  getitem_186 = None
        rsqrt_57: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_217);  add_217 = None
        mul_211: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_76, rsqrt_57);  sub_76 = rsqrt_57 = None
        mul_212: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_211, arg491_1);  mul_211 = arg491_1 = None
        add_218: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_212, arg492_1);  mul_212 = arg492_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_634: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_218, [2048, 2560]);  add_218 = None
        permute_353: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg493_1, [1, 0]);  arg493_1 = None
        addmm_186: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg494_1, view_634, permute_353);  arg494_1 = view_634 = permute_353 = None
        view_635: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_186, [16, 128, 2560]);  addmm_186 = None
        view_636: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_635, [16, 128, -1, 80]);  view_635 = None
        permute_354: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_636, [0, 2, 1, 3]);  view_636 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_213: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_354, 0.334370152488211);  permute_354 = None
        expand_97: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_213, [16, 32, 128, 80]);  mul_213 = None
        clone_153: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_97, memory_format = torch.contiguous_format);  expand_97 = None
        view_643: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_153, [512, 128, 80]);  clone_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_637: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_355: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg495_1, [1, 0]);  arg495_1 = None
        addmm_187: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg496_1, view_637, permute_355);  arg496_1 = view_637 = permute_355 = None
        view_638: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_187, [16, 128, 2560]);  addmm_187 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_641: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_638, [16, 128, -1, 80]);  view_638 = None
        permute_357: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_641, [0, 2, 1, 3]);  view_641 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_359: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_357, [0, 1, 3, 2]);  permute_357 = None
        mul_214: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_359, 0.334370152488211);  permute_359 = None
        expand_98: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_214, [16, 32, 80, 128]);  mul_214 = None
        clone_154: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_98, memory_format = torch.contiguous_format);  expand_98 = None
        view_644: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_154, [512, 80, 128]);  clone_154 = None
        bmm_38: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_643, view_644);  view_643 = view_644 = None
        view_645: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_38, [16, 32, 128, 128]);  bmm_38 = None
        full_default_94: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_93: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_56: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_94, full_default_93);  full_default_94 = full_default_93 = None
        add_219: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_645, where_56);  view_645 = where_56 = None
        eq_19: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_219, -inf)
        logical_not_38: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_19);  eq_19 = None
        any_20: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_38, -1, True);  logical_not_38 = None
        logical_not_39: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_20);  any_20 = None
        full_default_95: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_19: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_219, [-1], True)
        sub_77: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_219, amax_19);  add_219 = amax_19 = None
        exp_19: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_77);  sub_77 = None
        sum_20: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_19, [-1], True)
        div_19: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_19, sum_20);  exp_19 = sum_20 = None
        where_57: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_39, full_default_95, div_19);  logical_not_39 = full_default_95 = div_19 = None
        expand_99: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_57, [16, 32, 128, 128]);  where_57 = None
        view_646: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_99, [512, 128, 128]);  expand_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_639: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_356: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg497_1, [1, 0]);  arg497_1 = None
        addmm_188: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg498_1, view_639, permute_356);  arg498_1 = view_639 = permute_356 = None
        view_640: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_188, [16, 128, 2560]);  addmm_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_642: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_640, [16, 128, -1, 80]);  view_640 = None
        permute_358: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_642, [0, 2, 1, 3]);  view_642 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_100: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_358, [16, 32, 128, 80]);  permute_358 = None
        clone_155: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_100, memory_format = torch.contiguous_format);  expand_100 = None
        view_647: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_155, [512, 128, 80]);  clone_155 = None
        bmm_39: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_646, view_647);  view_646 = view_647 = None
        view_648: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_39, [16, 32, 128, 80]);  bmm_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_360: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_648, [0, 2, 1, 3]);  view_648 = None
        clone_156: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_360, memory_format = torch.contiguous_format);  permute_360 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_649: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_156, [16, 128, -1]);  clone_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_650: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_649, [2048, 2560]);  view_649 = None
        permute_361: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg499_1, [1, 0]);  arg499_1 = None
        addmm_189: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg500_1, view_650, permute_361);  arg500_1 = view_650 = permute_361 = None
        view_651: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_189, [16, 128, 2560]);  addmm_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_220: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_216, view_651);  add_216 = view_651 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_58 = torch.ops.aten.var_mean.correction(add_220, [2], correction = 0, keepdim = True)
        getitem_188: "f32[16, 128, 1]" = var_mean_58[0]
        getitem_189: "f32[16, 128, 1]" = var_mean_58[1];  var_mean_58 = None
        sub_78: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_220, getitem_189);  getitem_189 = None
        add_221: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_188, 1e-05);  getitem_188 = None
        rsqrt_58: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_221);  add_221 = None
        mul_215: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_78, rsqrt_58);  sub_78 = rsqrt_58 = None
        mul_216: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_215, arg501_1);  mul_215 = arg501_1 = None
        add_222: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_216, arg502_1);  mul_216 = arg502_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_652: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_222, [2048, 2560]);  add_222 = None
        permute_362: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg503_1, [1, 0]);  arg503_1 = None
        addmm_190: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg504_1, view_652, permute_362);  arg504_1 = view_652 = permute_362 = None
        view_653: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_190, [16, 128, 10240]);  addmm_190 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_217: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_653, 0.5)
        mul_218: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_653, 0.7071067811865476);  view_653 = None
        erf_19: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_218);  mul_218 = None
        add_223: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_19, 1);  erf_19 = None
        mul_219: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_217, add_223);  mul_217 = add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_654: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_219, [2048, 10240]);  mul_219 = None
        permute_363: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg505_1, [1, 0]);  arg505_1 = None
        addmm_191: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg506_1, view_654, permute_363);  arg506_1 = view_654 = permute_363 = None
        view_655: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_191, [16, 128, 2560]);  addmm_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_224: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_220, view_655);  add_220 = view_655 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_59 = torch.ops.aten.var_mean.correction(add_224, [2], correction = 0, keepdim = True)
        getitem_190: "f32[16, 128, 1]" = var_mean_59[0]
        getitem_191: "f32[16, 128, 1]" = var_mean_59[1];  var_mean_59 = None
        sub_79: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_224, getitem_191);  getitem_191 = None
        add_225: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_190, 1e-05);  getitem_190 = None
        rsqrt_59: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_225);  add_225 = None
        mul_220: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_79, rsqrt_59);  sub_79 = rsqrt_59 = None
        mul_221: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_220, arg507_1);  mul_220 = arg507_1 = None
        add_226: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_221, arg508_1);  mul_221 = arg508_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_656: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_226, [2048, 2560])
        permute_364: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg509_1, [1, 0]);  arg509_1 = None
        addmm_192: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg510_1, view_656, permute_364);  arg510_1 = view_656 = permute_364 = None
        view_657: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_192, [16, 128, 2560]);  addmm_192 = None
        view_658: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_657, [16, 128, -1, 80]);  view_657 = None
        permute_365: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_658, [0, 2, 1, 3]);  view_658 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_659: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_226, [2048, 2560])
        permute_366: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg511_1, [1, 0]);  arg511_1 = None
        addmm_193: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg512_1, view_659, permute_366);  arg512_1 = view_659 = permute_366 = None
        view_660: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_193, [16, 128, 2560]);  addmm_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_663: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_660, [16, 128, -1, 80]);  view_660 = None
        permute_368: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_663, [0, 2, 1, 3]);  view_663 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_661: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_226, [2048, 2560]);  add_226 = None
        permute_367: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg513_1, [1, 0]);  arg513_1 = None
        addmm_194: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg514_1, view_661, permute_367);  arg514_1 = view_661 = permute_367 = None
        view_662: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_194, [16, 128, 2560]);  addmm_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_664: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_662, [16, 128, -1, 80]);  view_662 = None
        permute_369: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_664, [0, 2, 1, 3]);  view_664 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_97: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_96: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_58: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_97, full_default_96);  full_default_97 = full_default_96 = None
        expand_101: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_58, [16, 32, 128, 128]);  where_58 = None
        _scaled_dot_product_efficient_attention_18 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_365, permute_368, permute_369, expand_101, False, scale = 0.11180339887498948);  permute_365 = permute_368 = permute_369 = expand_101 = None
        getitem_192: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_18[0];  _scaled_dot_product_efficient_attention_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_370: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_192, [0, 2, 1, 3]);  getitem_192 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_665: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_370, [16, 128, -1]);  permute_370 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_666: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_665, [2048, 2560]);  view_665 = None
        permute_371: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg515_1, [1, 0]);  arg515_1 = None
        addmm_195: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg516_1, view_666, permute_371);  arg516_1 = view_666 = permute_371 = None
        view_667: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_195, [16, 128, 2560]);  addmm_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_227: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_224, view_667);  add_224 = view_667 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_60 = torch.ops.aten.var_mean.correction(add_227, [2], correction = 0, keepdim = True)
        getitem_196: "f32[16, 128, 1]" = var_mean_60[0]
        getitem_197: "f32[16, 128, 1]" = var_mean_60[1];  var_mean_60 = None
        sub_80: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_227, getitem_197);  getitem_197 = None
        add_228: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_196, 1e-05);  getitem_196 = None
        rsqrt_60: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_228);  add_228 = None
        mul_222: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_80, rsqrt_60);  sub_80 = rsqrt_60 = None
        mul_223: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_222, arg517_1);  mul_222 = arg517_1 = None
        add_229: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_223, arg518_1);  mul_223 = arg518_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_668: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_229, [2048, 2560]);  add_229 = None
        permute_372: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg519_1, [1, 0]);  arg519_1 = None
        addmm_196: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg520_1, view_668, permute_372);  arg520_1 = view_668 = permute_372 = None
        view_669: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_196, [16, 128, 2560]);  addmm_196 = None
        view_670: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_669, [16, 128, -1, 80]);  view_669 = None
        permute_373: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_670, [0, 2, 1, 3]);  view_670 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_224: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_373, 0.334370152488211);  permute_373 = None
        expand_102: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_224, [16, 32, 128, 80]);  mul_224 = None
        clone_161: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_102, memory_format = torch.contiguous_format);  expand_102 = None
        view_677: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_161, [512, 128, 80]);  clone_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_671: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_374: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg521_1, [1, 0]);  arg521_1 = None
        addmm_197: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg522_1, view_671, permute_374);  arg522_1 = view_671 = permute_374 = None
        view_672: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_197, [16, 128, 2560]);  addmm_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_675: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_672, [16, 128, -1, 80]);  view_672 = None
        permute_376: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_675, [0, 2, 1, 3]);  view_675 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_378: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_376, [0, 1, 3, 2]);  permute_376 = None
        mul_225: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_378, 0.334370152488211);  permute_378 = None
        expand_103: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_225, [16, 32, 80, 128]);  mul_225 = None
        clone_162: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_103, memory_format = torch.contiguous_format);  expand_103 = None
        view_678: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_162, [512, 80, 128]);  clone_162 = None
        bmm_40: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_677, view_678);  view_677 = view_678 = None
        view_679: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_40, [16, 32, 128, 128]);  bmm_40 = None
        full_default_99: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_98: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_59: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_99, full_default_98);  full_default_99 = full_default_98 = None
        add_230: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_679, where_59);  view_679 = where_59 = None
        eq_20: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_230, -inf)
        logical_not_40: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_20);  eq_20 = None
        any_21: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_40, -1, True);  logical_not_40 = None
        logical_not_41: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_21);  any_21 = None
        full_default_100: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_20: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_230, [-1], True)
        sub_81: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_230, amax_20);  add_230 = amax_20 = None
        exp_20: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_81);  sub_81 = None
        sum_21: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_20, [-1], True)
        div_20: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_20, sum_21);  exp_20 = sum_21 = None
        where_60: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_41, full_default_100, div_20);  logical_not_41 = full_default_100 = div_20 = None
        expand_104: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_60, [16, 32, 128, 128]);  where_60 = None
        view_680: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_104, [512, 128, 128]);  expand_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_673: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_375: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg523_1, [1, 0]);  arg523_1 = None
        addmm_198: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg524_1, view_673, permute_375);  arg524_1 = view_673 = permute_375 = None
        view_674: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_198, [16, 128, 2560]);  addmm_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_676: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_674, [16, 128, -1, 80]);  view_674 = None
        permute_377: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_676, [0, 2, 1, 3]);  view_676 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_105: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_377, [16, 32, 128, 80]);  permute_377 = None
        clone_163: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_105, memory_format = torch.contiguous_format);  expand_105 = None
        view_681: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_163, [512, 128, 80]);  clone_163 = None
        bmm_41: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_680, view_681);  view_680 = view_681 = None
        view_682: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_41, [16, 32, 128, 80]);  bmm_41 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_379: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_682, [0, 2, 1, 3]);  view_682 = None
        clone_164: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_379, memory_format = torch.contiguous_format);  permute_379 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_683: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_164, [16, 128, -1]);  clone_164 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_684: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_683, [2048, 2560]);  view_683 = None
        permute_380: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg525_1, [1, 0]);  arg525_1 = None
        addmm_199: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg526_1, view_684, permute_380);  arg526_1 = view_684 = permute_380 = None
        view_685: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_199, [16, 128, 2560]);  addmm_199 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_231: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_227, view_685);  add_227 = view_685 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_61 = torch.ops.aten.var_mean.correction(add_231, [2], correction = 0, keepdim = True)
        getitem_198: "f32[16, 128, 1]" = var_mean_61[0]
        getitem_199: "f32[16, 128, 1]" = var_mean_61[1];  var_mean_61 = None
        sub_82: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_231, getitem_199);  getitem_199 = None
        add_232: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_198, 1e-05);  getitem_198 = None
        rsqrt_61: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_232);  add_232 = None
        mul_226: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_82, rsqrt_61);  sub_82 = rsqrt_61 = None
        mul_227: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_226, arg527_1);  mul_226 = arg527_1 = None
        add_233: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_227, arg528_1);  mul_227 = arg528_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_686: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_233, [2048, 2560]);  add_233 = None
        permute_381: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg529_1, [1, 0]);  arg529_1 = None
        addmm_200: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg530_1, view_686, permute_381);  arg530_1 = view_686 = permute_381 = None
        view_687: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_200, [16, 128, 10240]);  addmm_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_228: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_687, 0.5)
        mul_229: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_687, 0.7071067811865476);  view_687 = None
        erf_20: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_229);  mul_229 = None
        add_234: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_20, 1);  erf_20 = None
        mul_230: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_228, add_234);  mul_228 = add_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_688: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_230, [2048, 10240]);  mul_230 = None
        permute_382: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg531_1, [1, 0]);  arg531_1 = None
        addmm_201: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg532_1, view_688, permute_382);  arg532_1 = view_688 = permute_382 = None
        view_689: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_201, [16, 128, 2560]);  addmm_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_235: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_231, view_689);  add_231 = view_689 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_62 = torch.ops.aten.var_mean.correction(add_235, [2], correction = 0, keepdim = True)
        getitem_200: "f32[16, 128, 1]" = var_mean_62[0]
        getitem_201: "f32[16, 128, 1]" = var_mean_62[1];  var_mean_62 = None
        sub_83: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_235, getitem_201);  getitem_201 = None
        add_236: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_200, 1e-05);  getitem_200 = None
        rsqrt_62: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_236);  add_236 = None
        mul_231: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_83, rsqrt_62);  sub_83 = rsqrt_62 = None
        mul_232: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_231, arg533_1);  mul_231 = arg533_1 = None
        add_237: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_232, arg534_1);  mul_232 = arg534_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_690: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_237, [2048, 2560])
        permute_383: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg535_1, [1, 0]);  arg535_1 = None
        addmm_202: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg536_1, view_690, permute_383);  arg536_1 = view_690 = permute_383 = None
        view_691: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_202, [16, 128, 2560]);  addmm_202 = None
        view_692: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_691, [16, 128, -1, 80]);  view_691 = None
        permute_384: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_692, [0, 2, 1, 3]);  view_692 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_693: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_237, [2048, 2560])
        permute_385: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg537_1, [1, 0]);  arg537_1 = None
        addmm_203: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg538_1, view_693, permute_385);  arg538_1 = view_693 = permute_385 = None
        view_694: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_203, [16, 128, 2560]);  addmm_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_697: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_694, [16, 128, -1, 80]);  view_694 = None
        permute_387: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_697, [0, 2, 1, 3]);  view_697 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_695: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_237, [2048, 2560]);  add_237 = None
        permute_386: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg539_1, [1, 0]);  arg539_1 = None
        addmm_204: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg540_1, view_695, permute_386);  arg540_1 = view_695 = permute_386 = None
        view_696: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_204, [16, 128, 2560]);  addmm_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_698: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_696, [16, 128, -1, 80]);  view_696 = None
        permute_388: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_698, [0, 2, 1, 3]);  view_698 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_102: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_101: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_61: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_102, full_default_101);  full_default_102 = full_default_101 = None
        expand_106: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_61, [16, 32, 128, 128]);  where_61 = None
        _scaled_dot_product_efficient_attention_19 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_384, permute_387, permute_388, expand_106, False, scale = 0.11180339887498948);  permute_384 = permute_387 = permute_388 = expand_106 = None
        getitem_202: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_19[0];  _scaled_dot_product_efficient_attention_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_389: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_202, [0, 2, 1, 3]);  getitem_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_699: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_389, [16, 128, -1]);  permute_389 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_700: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_699, [2048, 2560]);  view_699 = None
        permute_390: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg541_1, [1, 0]);  arg541_1 = None
        addmm_205: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg542_1, view_700, permute_390);  arg542_1 = view_700 = permute_390 = None
        view_701: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_205, [16, 128, 2560]);  addmm_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_238: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_235, view_701);  add_235 = view_701 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_63 = torch.ops.aten.var_mean.correction(add_238, [2], correction = 0, keepdim = True)
        getitem_206: "f32[16, 128, 1]" = var_mean_63[0]
        getitem_207: "f32[16, 128, 1]" = var_mean_63[1];  var_mean_63 = None
        sub_84: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_238, getitem_207);  getitem_207 = None
        add_239: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_206, 1e-05);  getitem_206 = None
        rsqrt_63: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_239);  add_239 = None
        mul_233: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_84, rsqrt_63);  sub_84 = rsqrt_63 = None
        mul_234: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_233, arg543_1);  mul_233 = arg543_1 = None
        add_240: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_234, arg544_1);  mul_234 = arg544_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_702: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_240, [2048, 2560]);  add_240 = None
        permute_391: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg545_1, [1, 0]);  arg545_1 = None
        addmm_206: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg546_1, view_702, permute_391);  arg546_1 = view_702 = permute_391 = None
        view_703: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_206, [16, 128, 2560]);  addmm_206 = None
        view_704: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_703, [16, 128, -1, 80]);  view_703 = None
        permute_392: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_704, [0, 2, 1, 3]);  view_704 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_235: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_392, 0.334370152488211);  permute_392 = None
        expand_107: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_235, [16, 32, 128, 80]);  mul_235 = None
        clone_169: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_107, memory_format = torch.contiguous_format);  expand_107 = None
        view_711: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_169, [512, 128, 80]);  clone_169 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_705: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_393: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg547_1, [1, 0]);  arg547_1 = None
        addmm_207: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg548_1, view_705, permute_393);  arg548_1 = view_705 = permute_393 = None
        view_706: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_207, [16, 128, 2560]);  addmm_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_709: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_706, [16, 128, -1, 80]);  view_706 = None
        permute_395: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_709, [0, 2, 1, 3]);  view_709 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_397: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_395, [0, 1, 3, 2]);  permute_395 = None
        mul_236: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_397, 0.334370152488211);  permute_397 = None
        expand_108: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_236, [16, 32, 80, 128]);  mul_236 = None
        clone_170: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_108, memory_format = torch.contiguous_format);  expand_108 = None
        view_712: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_170, [512, 80, 128]);  clone_170 = None
        bmm_42: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_711, view_712);  view_711 = view_712 = None
        view_713: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_42, [16, 32, 128, 128]);  bmm_42 = None
        full_default_104: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_103: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_62: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_104, full_default_103);  full_default_104 = full_default_103 = None
        add_241: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_713, where_62);  view_713 = where_62 = None
        eq_21: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_241, -inf)
        logical_not_42: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_21);  eq_21 = None
        any_22: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_42, -1, True);  logical_not_42 = None
        logical_not_43: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_22);  any_22 = None
        full_default_105: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_21: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_241, [-1], True)
        sub_85: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_241, amax_21);  add_241 = amax_21 = None
        exp_21: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_85);  sub_85 = None
        sum_22: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_21, [-1], True)
        div_21: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_21, sum_22);  exp_21 = sum_22 = None
        where_63: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_43, full_default_105, div_21);  logical_not_43 = full_default_105 = div_21 = None
        expand_109: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_63, [16, 32, 128, 128]);  where_63 = None
        view_714: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_109, [512, 128, 128]);  expand_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_707: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_394: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg549_1, [1, 0]);  arg549_1 = None
        addmm_208: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg550_1, view_707, permute_394);  arg550_1 = view_707 = permute_394 = None
        view_708: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_208, [16, 128, 2560]);  addmm_208 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_710: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_708, [16, 128, -1, 80]);  view_708 = None
        permute_396: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_710, [0, 2, 1, 3]);  view_710 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_110: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_396, [16, 32, 128, 80]);  permute_396 = None
        clone_171: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_110, memory_format = torch.contiguous_format);  expand_110 = None
        view_715: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_171, [512, 128, 80]);  clone_171 = None
        bmm_43: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_714, view_715);  view_714 = view_715 = None
        view_716: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_43, [16, 32, 128, 80]);  bmm_43 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_398: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_716, [0, 2, 1, 3]);  view_716 = None
        clone_172: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_398, memory_format = torch.contiguous_format);  permute_398 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_717: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_172, [16, 128, -1]);  clone_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_718: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_717, [2048, 2560]);  view_717 = None
        permute_399: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg551_1, [1, 0]);  arg551_1 = None
        addmm_209: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg552_1, view_718, permute_399);  arg552_1 = view_718 = permute_399 = None
        view_719: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_209, [16, 128, 2560]);  addmm_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_242: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_238, view_719);  add_238 = view_719 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_64 = torch.ops.aten.var_mean.correction(add_242, [2], correction = 0, keepdim = True)
        getitem_208: "f32[16, 128, 1]" = var_mean_64[0]
        getitem_209: "f32[16, 128, 1]" = var_mean_64[1];  var_mean_64 = None
        sub_86: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_242, getitem_209);  getitem_209 = None
        add_243: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_208, 1e-05);  getitem_208 = None
        rsqrt_64: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_243);  add_243 = None
        mul_237: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_86, rsqrt_64);  sub_86 = rsqrt_64 = None
        mul_238: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_237, arg553_1);  mul_237 = arg553_1 = None
        add_244: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_238, arg554_1);  mul_238 = arg554_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_720: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_244, [2048, 2560]);  add_244 = None
        permute_400: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg555_1, [1, 0]);  arg555_1 = None
        addmm_210: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg556_1, view_720, permute_400);  arg556_1 = view_720 = permute_400 = None
        view_721: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_210, [16, 128, 10240]);  addmm_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_239: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_721, 0.5)
        mul_240: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_721, 0.7071067811865476);  view_721 = None
        erf_21: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_240);  mul_240 = None
        add_245: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_21, 1);  erf_21 = None
        mul_241: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_239, add_245);  mul_239 = add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_722: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_241, [2048, 10240]);  mul_241 = None
        permute_401: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg557_1, [1, 0]);  arg557_1 = None
        addmm_211: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg558_1, view_722, permute_401);  arg558_1 = view_722 = permute_401 = None
        view_723: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_211, [16, 128, 2560]);  addmm_211 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_246: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_242, view_723);  add_242 = view_723 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_65 = torch.ops.aten.var_mean.correction(add_246, [2], correction = 0, keepdim = True)
        getitem_210: "f32[16, 128, 1]" = var_mean_65[0]
        getitem_211: "f32[16, 128, 1]" = var_mean_65[1];  var_mean_65 = None
        sub_87: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_246, getitem_211);  getitem_211 = None
        add_247: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_210, 1e-05);  getitem_210 = None
        rsqrt_65: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_247);  add_247 = None
        mul_242: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_87, rsqrt_65);  sub_87 = rsqrt_65 = None
        mul_243: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_242, arg559_1);  mul_242 = arg559_1 = None
        add_248: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_243, arg560_1);  mul_243 = arg560_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_724: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_248, [2048, 2560])
        permute_402: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg561_1, [1, 0]);  arg561_1 = None
        addmm_212: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg562_1, view_724, permute_402);  arg562_1 = view_724 = permute_402 = None
        view_725: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_212, [16, 128, 2560]);  addmm_212 = None
        view_726: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_725, [16, 128, -1, 80]);  view_725 = None
        permute_403: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_726, [0, 2, 1, 3]);  view_726 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_727: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_248, [2048, 2560])
        permute_404: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg563_1, [1, 0]);  arg563_1 = None
        addmm_213: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg564_1, view_727, permute_404);  arg564_1 = view_727 = permute_404 = None
        view_728: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_213, [16, 128, 2560]);  addmm_213 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_731: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_728, [16, 128, -1, 80]);  view_728 = None
        permute_406: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_731, [0, 2, 1, 3]);  view_731 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_729: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_248, [2048, 2560]);  add_248 = None
        permute_405: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg565_1, [1, 0]);  arg565_1 = None
        addmm_214: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg566_1, view_729, permute_405);  arg566_1 = view_729 = permute_405 = None
        view_730: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_214, [16, 128, 2560]);  addmm_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_732: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_730, [16, 128, -1, 80]);  view_730 = None
        permute_407: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_732, [0, 2, 1, 3]);  view_732 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_107: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_106: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_64: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_107, full_default_106);  full_default_107 = full_default_106 = None
        expand_111: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_64, [16, 32, 128, 128]);  where_64 = None
        _scaled_dot_product_efficient_attention_20 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_403, permute_406, permute_407, expand_111, False, scale = 0.11180339887498948);  permute_403 = permute_406 = permute_407 = expand_111 = None
        getitem_212: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_20[0];  _scaled_dot_product_efficient_attention_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_408: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_212, [0, 2, 1, 3]);  getitem_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_733: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_408, [16, 128, -1]);  permute_408 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_734: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_733, [2048, 2560]);  view_733 = None
        permute_409: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg567_1, [1, 0]);  arg567_1 = None
        addmm_215: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg568_1, view_734, permute_409);  arg568_1 = view_734 = permute_409 = None
        view_735: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_215, [16, 128, 2560]);  addmm_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_249: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_246, view_735);  add_246 = view_735 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_66 = torch.ops.aten.var_mean.correction(add_249, [2], correction = 0, keepdim = True)
        getitem_216: "f32[16, 128, 1]" = var_mean_66[0]
        getitem_217: "f32[16, 128, 1]" = var_mean_66[1];  var_mean_66 = None
        sub_88: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_249, getitem_217);  getitem_217 = None
        add_250: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_216, 1e-05);  getitem_216 = None
        rsqrt_66: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_250);  add_250 = None
        mul_244: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_88, rsqrt_66);  sub_88 = rsqrt_66 = None
        mul_245: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_244, arg569_1);  mul_244 = arg569_1 = None
        add_251: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_245, arg570_1);  mul_245 = arg570_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_736: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_251, [2048, 2560]);  add_251 = None
        permute_410: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg571_1, [1, 0]);  arg571_1 = None
        addmm_216: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg572_1, view_736, permute_410);  arg572_1 = view_736 = permute_410 = None
        view_737: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_216, [16, 128, 2560]);  addmm_216 = None
        view_738: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_737, [16, 128, -1, 80]);  view_737 = None
        permute_411: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_738, [0, 2, 1, 3]);  view_738 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_246: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_411, 0.334370152488211);  permute_411 = None
        expand_112: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_246, [16, 32, 128, 80]);  mul_246 = None
        clone_177: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_112, memory_format = torch.contiguous_format);  expand_112 = None
        view_745: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_177, [512, 128, 80]);  clone_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_739: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_412: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg573_1, [1, 0]);  arg573_1 = None
        addmm_217: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg574_1, view_739, permute_412);  arg574_1 = view_739 = permute_412 = None
        view_740: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_217, [16, 128, 2560]);  addmm_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_743: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_740, [16, 128, -1, 80]);  view_740 = None
        permute_414: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_743, [0, 2, 1, 3]);  view_743 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_416: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_414, [0, 1, 3, 2]);  permute_414 = None
        mul_247: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_416, 0.334370152488211);  permute_416 = None
        expand_113: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_247, [16, 32, 80, 128]);  mul_247 = None
        clone_178: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_113, memory_format = torch.contiguous_format);  expand_113 = None
        view_746: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_178, [512, 80, 128]);  clone_178 = None
        bmm_44: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_745, view_746);  view_745 = view_746 = None
        view_747: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_44, [16, 32, 128, 128]);  bmm_44 = None
        full_default_109: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_108: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_65: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_109, full_default_108);  full_default_109 = full_default_108 = None
        add_252: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_747, where_65);  view_747 = where_65 = None
        eq_22: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_252, -inf)
        logical_not_44: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_22);  eq_22 = None
        any_23: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_44, -1, True);  logical_not_44 = None
        logical_not_45: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_23);  any_23 = None
        full_default_110: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_22: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_252, [-1], True)
        sub_89: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_252, amax_22);  add_252 = amax_22 = None
        exp_22: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_89);  sub_89 = None
        sum_23: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_22, [-1], True)
        div_22: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_22, sum_23);  exp_22 = sum_23 = None
        where_66: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_45, full_default_110, div_22);  logical_not_45 = full_default_110 = div_22 = None
        expand_114: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_66, [16, 32, 128, 128]);  where_66 = None
        view_748: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_114, [512, 128, 128]);  expand_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_741: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_413: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg575_1, [1, 0]);  arg575_1 = None
        addmm_218: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg576_1, view_741, permute_413);  arg576_1 = view_741 = permute_413 = None
        view_742: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_218, [16, 128, 2560]);  addmm_218 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_744: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_742, [16, 128, -1, 80]);  view_742 = None
        permute_415: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_744, [0, 2, 1, 3]);  view_744 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_115: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_415, [16, 32, 128, 80]);  permute_415 = None
        clone_179: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_115, memory_format = torch.contiguous_format);  expand_115 = None
        view_749: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_179, [512, 128, 80]);  clone_179 = None
        bmm_45: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_748, view_749);  view_748 = view_749 = None
        view_750: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_45, [16, 32, 128, 80]);  bmm_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_417: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_750, [0, 2, 1, 3]);  view_750 = None
        clone_180: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_417, memory_format = torch.contiguous_format);  permute_417 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_751: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_180, [16, 128, -1]);  clone_180 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_752: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_751, [2048, 2560]);  view_751 = None
        permute_418: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg577_1, [1, 0]);  arg577_1 = None
        addmm_219: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg578_1, view_752, permute_418);  arg578_1 = view_752 = permute_418 = None
        view_753: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_219, [16, 128, 2560]);  addmm_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_253: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_249, view_753);  add_249 = view_753 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_67 = torch.ops.aten.var_mean.correction(add_253, [2], correction = 0, keepdim = True)
        getitem_218: "f32[16, 128, 1]" = var_mean_67[0]
        getitem_219: "f32[16, 128, 1]" = var_mean_67[1];  var_mean_67 = None
        sub_90: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_253, getitem_219);  getitem_219 = None
        add_254: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_218, 1e-05);  getitem_218 = None
        rsqrt_67: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_254);  add_254 = None
        mul_248: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_90, rsqrt_67);  sub_90 = rsqrt_67 = None
        mul_249: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_248, arg579_1);  mul_248 = arg579_1 = None
        add_255: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_249, arg580_1);  mul_249 = arg580_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_754: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_255, [2048, 2560]);  add_255 = None
        permute_419: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg581_1, [1, 0]);  arg581_1 = None
        addmm_220: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg582_1, view_754, permute_419);  arg582_1 = view_754 = permute_419 = None
        view_755: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_220, [16, 128, 10240]);  addmm_220 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_250: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_755, 0.5)
        mul_251: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_755, 0.7071067811865476);  view_755 = None
        erf_22: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_251);  mul_251 = None
        add_256: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_22, 1);  erf_22 = None
        mul_252: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_250, add_256);  mul_250 = add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_756: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_252, [2048, 10240]);  mul_252 = None
        permute_420: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg583_1, [1, 0]);  arg583_1 = None
        addmm_221: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg584_1, view_756, permute_420);  arg584_1 = view_756 = permute_420 = None
        view_757: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_221, [16, 128, 2560]);  addmm_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_257: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_253, view_757);  add_253 = view_757 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_68 = torch.ops.aten.var_mean.correction(add_257, [2], correction = 0, keepdim = True)
        getitem_220: "f32[16, 128, 1]" = var_mean_68[0]
        getitem_221: "f32[16, 128, 1]" = var_mean_68[1];  var_mean_68 = None
        sub_91: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_257, getitem_221);  getitem_221 = None
        add_258: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_220, 1e-05);  getitem_220 = None
        rsqrt_68: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_258);  add_258 = None
        mul_253: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_91, rsqrt_68);  sub_91 = rsqrt_68 = None
        mul_254: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_253, arg585_1);  mul_253 = arg585_1 = None
        add_259: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_254, arg586_1);  mul_254 = arg586_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_758: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_259, [2048, 2560])
        permute_421: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg587_1, [1, 0]);  arg587_1 = None
        addmm_222: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg588_1, view_758, permute_421);  arg588_1 = view_758 = permute_421 = None
        view_759: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_222, [16, 128, 2560]);  addmm_222 = None
        view_760: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_759, [16, 128, -1, 80]);  view_759 = None
        permute_422: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_760, [0, 2, 1, 3]);  view_760 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_761: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_259, [2048, 2560])
        permute_423: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg589_1, [1, 0]);  arg589_1 = None
        addmm_223: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg590_1, view_761, permute_423);  arg590_1 = view_761 = permute_423 = None
        view_762: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_223, [16, 128, 2560]);  addmm_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_765: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_762, [16, 128, -1, 80]);  view_762 = None
        permute_425: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_765, [0, 2, 1, 3]);  view_765 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_763: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_259, [2048, 2560]);  add_259 = None
        permute_424: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg591_1, [1, 0]);  arg591_1 = None
        addmm_224: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg592_1, view_763, permute_424);  arg592_1 = view_763 = permute_424 = None
        view_764: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_224, [16, 128, 2560]);  addmm_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_766: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_764, [16, 128, -1, 80]);  view_764 = None
        permute_426: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_766, [0, 2, 1, 3]);  view_766 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_112: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_111: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_67: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_112, full_default_111);  full_default_112 = full_default_111 = None
        expand_116: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_67, [16, 32, 128, 128]);  where_67 = None
        _scaled_dot_product_efficient_attention_21 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_422, permute_425, permute_426, expand_116, False, scale = 0.11180339887498948);  permute_422 = permute_425 = permute_426 = expand_116 = None
        getitem_222: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_21[0];  _scaled_dot_product_efficient_attention_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_427: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_222, [0, 2, 1, 3]);  getitem_222 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_767: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_427, [16, 128, -1]);  permute_427 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_768: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_767, [2048, 2560]);  view_767 = None
        permute_428: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg593_1, [1, 0]);  arg593_1 = None
        addmm_225: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg594_1, view_768, permute_428);  arg594_1 = view_768 = permute_428 = None
        view_769: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_225, [16, 128, 2560]);  addmm_225 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_260: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_257, view_769);  add_257 = view_769 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_69 = torch.ops.aten.var_mean.correction(add_260, [2], correction = 0, keepdim = True)
        getitem_226: "f32[16, 128, 1]" = var_mean_69[0]
        getitem_227: "f32[16, 128, 1]" = var_mean_69[1];  var_mean_69 = None
        sub_92: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_260, getitem_227);  getitem_227 = None
        add_261: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_226, 1e-05);  getitem_226 = None
        rsqrt_69: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_261);  add_261 = None
        mul_255: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_92, rsqrt_69);  sub_92 = rsqrt_69 = None
        mul_256: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_255, arg595_1);  mul_255 = arg595_1 = None
        add_262: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_256, arg596_1);  mul_256 = arg596_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_770: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_262, [2048, 2560]);  add_262 = None
        permute_429: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg597_1, [1, 0]);  arg597_1 = None
        addmm_226: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg598_1, view_770, permute_429);  arg598_1 = view_770 = permute_429 = None
        view_771: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_226, [16, 128, 2560]);  addmm_226 = None
        view_772: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_771, [16, 128, -1, 80]);  view_771 = None
        permute_430: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_772, [0, 2, 1, 3]);  view_772 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_257: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_430, 0.334370152488211);  permute_430 = None
        expand_117: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_257, [16, 32, 128, 80]);  mul_257 = None
        clone_185: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_117, memory_format = torch.contiguous_format);  expand_117 = None
        view_779: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_185, [512, 128, 80]);  clone_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_773: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_431: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg599_1, [1, 0]);  arg599_1 = None
        addmm_227: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg600_1, view_773, permute_431);  arg600_1 = view_773 = permute_431 = None
        view_774: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_227, [16, 128, 2560]);  addmm_227 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_777: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_774, [16, 128, -1, 80]);  view_774 = None
        permute_433: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_777, [0, 2, 1, 3]);  view_777 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_435: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_433, [0, 1, 3, 2]);  permute_433 = None
        mul_258: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_435, 0.334370152488211);  permute_435 = None
        expand_118: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_258, [16, 32, 80, 128]);  mul_258 = None
        clone_186: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_118, memory_format = torch.contiguous_format);  expand_118 = None
        view_780: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_186, [512, 80, 128]);  clone_186 = None
        bmm_46: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_779, view_780);  view_779 = view_780 = None
        view_781: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_46, [16, 32, 128, 128]);  bmm_46 = None
        full_default_114: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_113: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_68: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_114, full_default_113);  full_default_114 = full_default_113 = None
        add_263: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_781, where_68);  view_781 = where_68 = None
        eq_23: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_263, -inf)
        logical_not_46: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_23);  eq_23 = None
        any_24: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_46, -1, True);  logical_not_46 = None
        logical_not_47: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_24);  any_24 = None
        full_default_115: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_23: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_263, [-1], True)
        sub_93: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_263, amax_23);  add_263 = amax_23 = None
        exp_23: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_93);  sub_93 = None
        sum_24: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_23, [-1], True)
        div_23: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_23, sum_24);  exp_23 = sum_24 = None
        where_69: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_47, full_default_115, div_23);  logical_not_47 = full_default_115 = div_23 = None
        expand_119: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_69, [16, 32, 128, 128]);  where_69 = None
        view_782: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_119, [512, 128, 128]);  expand_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_775: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_432: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg601_1, [1, 0]);  arg601_1 = None
        addmm_228: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg602_1, view_775, permute_432);  arg602_1 = view_775 = permute_432 = None
        view_776: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_228, [16, 128, 2560]);  addmm_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_778: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_776, [16, 128, -1, 80]);  view_776 = None
        permute_434: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_778, [0, 2, 1, 3]);  view_778 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_120: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_434, [16, 32, 128, 80]);  permute_434 = None
        clone_187: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_120, memory_format = torch.contiguous_format);  expand_120 = None
        view_783: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_187, [512, 128, 80]);  clone_187 = None
        bmm_47: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_782, view_783);  view_782 = view_783 = None
        view_784: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_47, [16, 32, 128, 80]);  bmm_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_436: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_784, [0, 2, 1, 3]);  view_784 = None
        clone_188: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_436, memory_format = torch.contiguous_format);  permute_436 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_785: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_188, [16, 128, -1]);  clone_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_786: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_785, [2048, 2560]);  view_785 = None
        permute_437: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg603_1, [1, 0]);  arg603_1 = None
        addmm_229: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg604_1, view_786, permute_437);  arg604_1 = view_786 = permute_437 = None
        view_787: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_229, [16, 128, 2560]);  addmm_229 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_264: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_260, view_787);  add_260 = view_787 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_70 = torch.ops.aten.var_mean.correction(add_264, [2], correction = 0, keepdim = True)
        getitem_228: "f32[16, 128, 1]" = var_mean_70[0]
        getitem_229: "f32[16, 128, 1]" = var_mean_70[1];  var_mean_70 = None
        sub_94: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_264, getitem_229);  getitem_229 = None
        add_265: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_228, 1e-05);  getitem_228 = None
        rsqrt_70: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_265);  add_265 = None
        mul_259: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_94, rsqrt_70);  sub_94 = rsqrt_70 = None
        mul_260: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_259, arg605_1);  mul_259 = arg605_1 = None
        add_266: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_260, arg606_1);  mul_260 = arg606_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_788: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_266, [2048, 2560]);  add_266 = None
        permute_438: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg607_1, [1, 0]);  arg607_1 = None
        addmm_230: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg608_1, view_788, permute_438);  arg608_1 = view_788 = permute_438 = None
        view_789: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_230, [16, 128, 10240]);  addmm_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_261: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_789, 0.5)
        mul_262: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_789, 0.7071067811865476);  view_789 = None
        erf_23: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_262);  mul_262 = None
        add_267: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_23, 1);  erf_23 = None
        mul_263: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_261, add_267);  mul_261 = add_267 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_790: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_263, [2048, 10240]);  mul_263 = None
        permute_439: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg609_1, [1, 0]);  arg609_1 = None
        addmm_231: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg610_1, view_790, permute_439);  arg610_1 = view_790 = permute_439 = None
        view_791: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_231, [16, 128, 2560]);  addmm_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_268: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_264, view_791);  add_264 = view_791 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_71 = torch.ops.aten.var_mean.correction(add_268, [2], correction = 0, keepdim = True)
        getitem_230: "f32[16, 128, 1]" = var_mean_71[0]
        getitem_231: "f32[16, 128, 1]" = var_mean_71[1];  var_mean_71 = None
        sub_95: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_268, getitem_231);  getitem_231 = None
        add_269: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_230, 1e-05);  getitem_230 = None
        rsqrt_71: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_269);  add_269 = None
        mul_264: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_95, rsqrt_71);  sub_95 = rsqrt_71 = None
        mul_265: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_264, arg611_1);  mul_264 = arg611_1 = None
        add_270: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_265, arg612_1);  mul_265 = arg612_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_792: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_270, [2048, 2560])
        permute_440: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg613_1, [1, 0]);  arg613_1 = None
        addmm_232: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg614_1, view_792, permute_440);  arg614_1 = view_792 = permute_440 = None
        view_793: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_232, [16, 128, 2560]);  addmm_232 = None
        view_794: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_793, [16, 128, -1, 80]);  view_793 = None
        permute_441: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_794, [0, 2, 1, 3]);  view_794 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_795: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_270, [2048, 2560])
        permute_442: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg615_1, [1, 0]);  arg615_1 = None
        addmm_233: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg616_1, view_795, permute_442);  arg616_1 = view_795 = permute_442 = None
        view_796: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_233, [16, 128, 2560]);  addmm_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_799: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_796, [16, 128, -1, 80]);  view_796 = None
        permute_444: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_799, [0, 2, 1, 3]);  view_799 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_797: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_270, [2048, 2560]);  add_270 = None
        permute_443: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg617_1, [1, 0]);  arg617_1 = None
        addmm_234: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg618_1, view_797, permute_443);  arg618_1 = view_797 = permute_443 = None
        view_798: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_234, [16, 128, 2560]);  addmm_234 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_800: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_798, [16, 128, -1, 80]);  view_798 = None
        permute_445: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_800, [0, 2, 1, 3]);  view_800 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_117: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_116: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_70: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_117, full_default_116);  full_default_117 = full_default_116 = None
        expand_121: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_70, [16, 32, 128, 128]);  where_70 = None
        _scaled_dot_product_efficient_attention_22 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_441, permute_444, permute_445, expand_121, False, scale = 0.11180339887498948);  permute_441 = permute_444 = permute_445 = expand_121 = None
        getitem_232: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_22[0];  _scaled_dot_product_efficient_attention_22 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_446: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_232, [0, 2, 1, 3]);  getitem_232 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_801: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_446, [16, 128, -1]);  permute_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_802: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_801, [2048, 2560]);  view_801 = None
        permute_447: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg619_1, [1, 0]);  arg619_1 = None
        addmm_235: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg620_1, view_802, permute_447);  arg620_1 = view_802 = permute_447 = None
        view_803: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_235, [16, 128, 2560]);  addmm_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_271: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_268, view_803);  add_268 = view_803 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_72 = torch.ops.aten.var_mean.correction(add_271, [2], correction = 0, keepdim = True)
        getitem_236: "f32[16, 128, 1]" = var_mean_72[0]
        getitem_237: "f32[16, 128, 1]" = var_mean_72[1];  var_mean_72 = None
        sub_96: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_271, getitem_237);  getitem_237 = None
        add_272: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_236, 1e-05);  getitem_236 = None
        rsqrt_72: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_272);  add_272 = None
        mul_266: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_96, rsqrt_72);  sub_96 = rsqrt_72 = None
        mul_267: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_266, arg621_1);  mul_266 = arg621_1 = None
        add_273: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_267, arg622_1);  mul_267 = arg622_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_804: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_273, [2048, 2560]);  add_273 = None
        permute_448: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg623_1, [1, 0]);  arg623_1 = None
        addmm_236: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg624_1, view_804, permute_448);  arg624_1 = view_804 = permute_448 = None
        view_805: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_236, [16, 128, 2560]);  addmm_236 = None
        view_806: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_805, [16, 128, -1, 80]);  view_805 = None
        permute_449: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_806, [0, 2, 1, 3]);  view_806 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_268: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_449, 0.334370152488211);  permute_449 = None
        expand_122: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_268, [16, 32, 128, 80]);  mul_268 = None
        clone_193: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_122, memory_format = torch.contiguous_format);  expand_122 = None
        view_813: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_193, [512, 128, 80]);  clone_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_807: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_450: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg625_1, [1, 0]);  arg625_1 = None
        addmm_237: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg626_1, view_807, permute_450);  arg626_1 = view_807 = permute_450 = None
        view_808: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_237, [16, 128, 2560]);  addmm_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_811: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_808, [16, 128, -1, 80]);  view_808 = None
        permute_452: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_811, [0, 2, 1, 3]);  view_811 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_454: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_452, [0, 1, 3, 2]);  permute_452 = None
        mul_269: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_454, 0.334370152488211);  permute_454 = None
        expand_123: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_269, [16, 32, 80, 128]);  mul_269 = None
        clone_194: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_123, memory_format = torch.contiguous_format);  expand_123 = None
        view_814: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_194, [512, 80, 128]);  clone_194 = None
        bmm_48: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_813, view_814);  view_813 = view_814 = None
        view_815: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_48, [16, 32, 128, 128]);  bmm_48 = None
        full_default_119: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_118: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_71: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_119, full_default_118);  full_default_119 = full_default_118 = None
        add_274: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_815, where_71);  view_815 = where_71 = None
        eq_24: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_274, -inf)
        logical_not_48: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_24);  eq_24 = None
        any_25: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_48, -1, True);  logical_not_48 = None
        logical_not_49: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_25);  any_25 = None
        full_default_120: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_24: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_274, [-1], True)
        sub_97: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_274, amax_24);  add_274 = amax_24 = None
        exp_24: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_97);  sub_97 = None
        sum_25: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_24, [-1], True)
        div_24: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_24, sum_25);  exp_24 = sum_25 = None
        where_72: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_49, full_default_120, div_24);  logical_not_49 = full_default_120 = div_24 = None
        expand_124: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_72, [16, 32, 128, 128]);  where_72 = None
        view_816: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_124, [512, 128, 128]);  expand_124 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_809: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_451: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg627_1, [1, 0]);  arg627_1 = None
        addmm_238: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg628_1, view_809, permute_451);  arg628_1 = view_809 = permute_451 = None
        view_810: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_238, [16, 128, 2560]);  addmm_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_812: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_810, [16, 128, -1, 80]);  view_810 = None
        permute_453: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_812, [0, 2, 1, 3]);  view_812 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_125: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_453, [16, 32, 128, 80]);  permute_453 = None
        clone_195: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_125, memory_format = torch.contiguous_format);  expand_125 = None
        view_817: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_195, [512, 128, 80]);  clone_195 = None
        bmm_49: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_816, view_817);  view_816 = view_817 = None
        view_818: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_49, [16, 32, 128, 80]);  bmm_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_455: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_818, [0, 2, 1, 3]);  view_818 = None
        clone_196: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_455, memory_format = torch.contiguous_format);  permute_455 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_819: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_196, [16, 128, -1]);  clone_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_820: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_819, [2048, 2560]);  view_819 = None
        permute_456: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg629_1, [1, 0]);  arg629_1 = None
        addmm_239: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg630_1, view_820, permute_456);  arg630_1 = view_820 = permute_456 = None
        view_821: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_239, [16, 128, 2560]);  addmm_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_275: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_271, view_821);  add_271 = view_821 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_73 = torch.ops.aten.var_mean.correction(add_275, [2], correction = 0, keepdim = True)
        getitem_238: "f32[16, 128, 1]" = var_mean_73[0]
        getitem_239: "f32[16, 128, 1]" = var_mean_73[1];  var_mean_73 = None
        sub_98: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_275, getitem_239);  getitem_239 = None
        add_276: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_238, 1e-05);  getitem_238 = None
        rsqrt_73: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_276);  add_276 = None
        mul_270: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_98, rsqrt_73);  sub_98 = rsqrt_73 = None
        mul_271: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_270, arg631_1);  mul_270 = arg631_1 = None
        add_277: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_271, arg632_1);  mul_271 = arg632_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_822: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_277, [2048, 2560]);  add_277 = None
        permute_457: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg633_1, [1, 0]);  arg633_1 = None
        addmm_240: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg634_1, view_822, permute_457);  arg634_1 = view_822 = permute_457 = None
        view_823: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_240, [16, 128, 10240]);  addmm_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_272: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_823, 0.5)
        mul_273: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_823, 0.7071067811865476);  view_823 = None
        erf_24: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_273);  mul_273 = None
        add_278: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_24, 1);  erf_24 = None
        mul_274: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_272, add_278);  mul_272 = add_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_824: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_274, [2048, 10240]);  mul_274 = None
        permute_458: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg635_1, [1, 0]);  arg635_1 = None
        addmm_241: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg636_1, view_824, permute_458);  arg636_1 = view_824 = permute_458 = None
        view_825: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_241, [16, 128, 2560]);  addmm_241 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_279: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_275, view_825);  add_275 = view_825 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:358 in forward, code: hidden_states = self.self_attn_layer_norm(hidden_states)
        var_mean_74 = torch.ops.aten.var_mean.correction(add_279, [2], correction = 0, keepdim = True)
        getitem_240: "f32[16, 128, 1]" = var_mean_74[0]
        getitem_241: "f32[16, 128, 1]" = var_mean_74[1];  var_mean_74 = None
        sub_99: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_279, getitem_241);  getitem_241 = None
        add_280: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_240, 1e-05);  getitem_240 = None
        rsqrt_74: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_280);  add_280 = None
        mul_275: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_99, rsqrt_74);  sub_99 = rsqrt_74 = None
        mul_276: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_275, arg637_1);  mul_275 = arg637_1 = None
        add_281: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_276, arg638_1);  mul_276 = arg638_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_826: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_281, [2048, 2560])
        permute_459: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg639_1, [1, 0]);  arg639_1 = None
        addmm_242: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg640_1, view_826, permute_459);  arg640_1 = view_826 = permute_459 = None
        view_827: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_242, [16, 128, 2560]);  addmm_242 = None
        view_828: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_827, [16, 128, -1, 80]);  view_827 = None
        permute_460: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_828, [0, 2, 1, 3]);  view_828 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_829: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_281, [2048, 2560])
        permute_461: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg641_1, [1, 0]);  arg641_1 = None
        addmm_243: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg642_1, view_829, permute_461);  arg642_1 = view_829 = permute_461 = None
        view_830: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_243, [16, 128, 2560]);  addmm_243 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_833: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_830, [16, 128, -1, 80]);  view_830 = None
        permute_463: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_833, [0, 2, 1, 3]);  view_833 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_831: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_281, [2048, 2560]);  add_281 = None
        permute_462: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg643_1, [1, 0]);  arg643_1 = None
        addmm_244: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg644_1, view_831, permute_462);  arg644_1 = view_831 = permute_462 = None
        view_832: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_244, [16, 128, 2560]);  addmm_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_834: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_832, [16, 128, -1, 80]);  view_832 = None
        permute_464: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_834, [0, 2, 1, 3]);  view_834 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        full_default_122: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_121: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_73: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_9, full_default_122, full_default_121);  expand_9 = full_default_122 = full_default_121 = None
        expand_126: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_73, [16, 32, 128, 128]);  where_73 = None
        _scaled_dot_product_efficient_attention_23 = torch.ops.aten._scaled_dot_product_efficient_attention.default(permute_460, permute_463, permute_464, expand_126, False, scale = 0.11180339887498948);  permute_460 = permute_463 = permute_464 = expand_126 = None
        getitem_242: "f32[16, 32, 128, 80]" = _scaled_dot_product_efficient_attention_23[0];  _scaled_dot_product_efficient_attention_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_465: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(getitem_242, [0, 2, 1, 3]);  getitem_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_835: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(permute_465, [16, 128, -1]);  permute_465 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_836: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_835, [2048, 2560]);  view_835 = None
        permute_466: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg645_1, [1, 0]);  arg645_1 = None
        addmm_245: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg646_1, view_836, permute_466);  arg646_1 = view_836 = permute_466 = None
        view_837: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_245, [16, 128, 2560]);  addmm_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:368 in forward, code: hidden_states = residual + hidden_states
        add_282: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_279, view_837);  add_279 = view_837 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:373 in forward, code: hidden_states = self.encoder_attn_layer_norm(hidden_states)
        var_mean_75 = torch.ops.aten.var_mean.correction(add_282, [2], correction = 0, keepdim = True)
        getitem_246: "f32[16, 128, 1]" = var_mean_75[0]
        getitem_247: "f32[16, 128, 1]" = var_mean_75[1];  var_mean_75 = None
        sub_100: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_282, getitem_247);  getitem_247 = None
        add_283: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_246, 1e-05);  getitem_246 = None
        rsqrt_75: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_283);  add_283 = None
        mul_277: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_100, rsqrt_75);  sub_100 = rsqrt_75 = None
        mul_278: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_277, arg647_1);  mul_277 = arg647_1 = None
        add_284: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_278, arg648_1);  mul_278 = arg648_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:193 in forward, code: query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        view_838: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_284, [2048, 2560]);  add_284 = None
        permute_467: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg649_1, [1, 0]);  arg649_1 = None
        addmm_246: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg650_1, view_838, permute_467);  arg650_1 = view_838 = permute_467 = None
        view_839: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_246, [16, 128, 2560]);  addmm_246 = None
        view_840: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_839, [16, 128, -1, 80]);  view_839 = None
        permute_468: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_840, [0, 2, 1, 3]);  view_840 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        mul_279: "f32[16, 32, 128, 80]" = torch.ops.aten.mul.Scalar(permute_468, 0.334370152488211);  permute_468 = None
        expand_127: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(mul_279, [16, 32, 128, 80]);  mul_279 = None
        clone_201: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_127, memory_format = torch.contiguous_format);  expand_127 = None
        view_847: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_201, [512, 128, 80]);  clone_201 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:213 in forward, code: key_states = self.k_proj(current_states)
        view_841: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_469: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg651_1, [1, 0]);  arg651_1 = None
        addmm_247: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg652_1, view_841, permute_469);  arg652_1 = view_841 = permute_469 = None
        view_842: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_247, [16, 128, 2560]);  addmm_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:216 in forward, code: key_states = key_states.view(kv_shape).transpose(1, 2)
        view_845: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_842, [16, 128, -1, 80]);  view_842 = None
        permute_471: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_845, [0, 2, 1, 3]);  view_845 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        permute_473: "f32[16, 32, 80, 128]" = torch.ops.aten.permute.default(permute_471, [0, 1, 3, 2]);  permute_471 = None
        mul_280: "f32[16, 32, 80, 128]" = torch.ops.aten.mul.Scalar(permute_473, 0.334370152488211);  permute_473 = None
        expand_128: "f32[16, 32, 80, 128]" = torch.ops.aten.expand.default(mul_280, [16, 32, 80, 128]);  mul_280 = None
        clone_202: "f32[16, 32, 80, 128]" = torch.ops.aten.clone.default(expand_128, memory_format = torch.contiguous_format);  expand_128 = None
        view_848: "f32[512, 80, 128]" = torch.ops.aten.reshape.default(clone_202, [512, 80, 128]);  clone_202 = None
        bmm_50: "f32[512, 128, 128]" = torch.ops.aten.bmm.default(view_847, view_848);  view_847 = view_848 = None
        view_849: "f32[16, 32, 128, 128]" = torch.ops.aten.reshape.default(bmm_50, [16, 32, 128, 128]);  bmm_50 = None
        full_default_124: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        full_default_123: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_74: "f32[16, 1, 128, 128]" = torch.ops.aten.where.self(expand_10, full_default_124, full_default_123);  expand_10 = full_default_124 = full_default_123 = None
        add_285: "f32[16, 32, 128, 128]" = torch.ops.aten.add.Tensor(view_849, where_74);  view_849 = where_74 = None
        eq_25: "b8[16, 32, 128, 128]" = torch.ops.aten.eq.Scalar(add_285, -inf)
        logical_not_50: "b8[16, 32, 128, 128]" = torch.ops.aten.logical_not.default(eq_25);  eq_25 = None
        any_26: "b8[16, 32, 128, 1]" = torch.ops.aten.any.dim(logical_not_50, -1, True);  logical_not_50 = None
        logical_not_51: "b8[16, 32, 128, 1]" = torch.ops.aten.logical_not.default(any_26);  any_26 = None
        full_default_125: "f32[16, 32, 128, 128]" = torch.ops.aten.full.default([16, 32, 128, 128], 0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        amax_25: "f32[16, 32, 128, 1]" = torch.ops.aten.amax.default(add_285, [-1], True)
        sub_101: "f32[16, 32, 128, 128]" = torch.ops.aten.sub.Tensor(add_285, amax_25);  add_285 = amax_25 = None
        exp_25: "f32[16, 32, 128, 128]" = torch.ops.aten.exp.default(sub_101);  sub_101 = None
        sum_26: "f32[16, 32, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp_25, [-1], True)
        div_25: "f32[16, 32, 128, 128]" = torch.ops.aten.div.Tensor(exp_25, sum_26);  exp_25 = sum_26 = None
        where_75: "f32[16, 32, 128, 128]" = torch.ops.aten.where.self(logical_not_51, full_default_125, div_25);  logical_not_51 = full_default_125 = div_25 = None
        expand_129: "f32[16, 32, 128, 128]" = torch.ops.aten.expand.default(where_75, [16, 32, 128, 128]);  where_75 = None
        view_850: "f32[512, 128, 128]" = torch.ops.aten.reshape.default(expand_129, [512, 128, 128]);  expand_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:214 in forward, code: value_states = self.v_proj(current_states)
        view_843: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_20, [2048, 2560])
        permute_470: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg653_1, [1, 0]);  arg653_1 = None
        addmm_248: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg654_1, view_843, permute_470);  arg654_1 = view_843 = permute_470 = None
        view_844: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_248, [16, 128, 2560]);  addmm_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:217 in forward, code: value_states = value_states.view(kv_shape).transpose(1, 2)
        view_846: "f32[16, 128, 32, 80]" = torch.ops.aten.reshape.default(view_844, [16, 128, -1, 80]);  view_844 = None
        permute_472: "f32[16, 32, 128, 80]" = torch.ops.aten.permute.default(view_846, [0, 2, 1, 3]);  view_846 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:92 in sdpa_attention_forward, code: attn_output = torch.nn.functional.scaled_dot_product_attention(
        expand_130: "f32[16, 32, 128, 80]" = torch.ops.aten.expand.default(permute_472, [16, 32, 128, 80]);  permute_472 = None
        clone_203: "f32[16, 32, 128, 80]" = torch.ops.aten.clone.default(expand_130, memory_format = torch.contiguous_format);  expand_130 = None
        view_851: "f32[512, 128, 80]" = torch.ops.aten.reshape.default(clone_203, [512, 128, 80]);  clone_203 = None
        bmm_51: "f32[512, 128, 80]" = torch.ops.aten.bmm.default(view_850, view_851);  view_850 = view_851 = None
        view_852: "f32[16, 32, 128, 80]" = torch.ops.aten.reshape.default(bmm_51, [16, 32, 128, 80]);  bmm_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/integrations/sdpa_attention.py:102 in sdpa_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_474: "f32[16, 128, 32, 80]" = torch.ops.aten.permute.default(view_852, [0, 2, 1, 3]);  view_852 = None
        clone_204: "f32[16, 128, 32, 80]" = torch.ops.aten.clone.default(permute_474, memory_format = torch.contiguous_format);  permute_474 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:240 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_853: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(clone_204, [16, 128, -1]);  clone_204 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:241 in forward, code: attn_output = self.out_proj(attn_output)
        view_854: "f32[2048, 2560]" = torch.ops.aten.reshape.default(view_853, [2048, 2560]);  view_853 = None
        permute_475: "f32[2560, 2560]" = torch.ops.aten.permute.default(arg655_1, [1, 0]);  arg655_1 = None
        addmm_249: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg656_1, view_854, permute_475);  arg656_1 = view_854 = permute_475 = None
        view_855: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_249, [16, 128, 2560]);  addmm_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:383 in forward, code: hidden_states = residual + hidden_states
        add_286: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_282, view_855);  add_282 = view_855 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:387 in forward, code: hidden_states = self.final_layer_norm(hidden_states)
        var_mean_76 = torch.ops.aten.var_mean.correction(add_286, [2], correction = 0, keepdim = True)
        getitem_248: "f32[16, 128, 1]" = var_mean_76[0]
        getitem_249: "f32[16, 128, 1]" = var_mean_76[1];  var_mean_76 = None
        sub_102: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_286, getitem_249);  getitem_249 = None
        add_287: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_248, 1e-05);  getitem_248 = None
        rsqrt_76: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_287);  add_287 = None
        mul_281: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_102, rsqrt_76);  sub_102 = rsqrt_76 = None
        mul_282: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_281, arg657_1);  mul_281 = arg657_1 = None
        add_288: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_282, arg658_1);  mul_282 = arg658_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:388 in forward, code: hidden_states = self.activation_fn(self.fc1(hidden_states))
        view_856: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_288, [2048, 2560]);  add_288 = None
        permute_476: "f32[2560, 10240]" = torch.ops.aten.permute.default(arg659_1, [1, 0]);  arg659_1 = None
        addmm_250: "f32[2048, 10240]" = torch.ops.aten.addmm.default(arg660_1, view_856, permute_476);  arg660_1 = view_856 = permute_476 = None
        view_857: "f32[16, 128, 10240]" = torch.ops.aten.reshape.default(addmm_250, [16, 128, 10240]);  addmm_250 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/activations.py:89 in forward, code: return self.act(input)
        mul_283: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_857, 0.5)
        mul_284: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(view_857, 0.7071067811865476);  view_857 = None
        erf_25: "f32[16, 128, 10240]" = torch.ops.aten.erf.default(mul_284);  mul_284 = None
        add_289: "f32[16, 128, 10240]" = torch.ops.aten.add.Tensor(erf_25, 1);  erf_25 = None
        mul_285: "f32[16, 128, 10240]" = torch.ops.aten.mul.Tensor(mul_283, add_289);  mul_283 = add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:390 in forward, code: hidden_states = self.fc2(hidden_states)
        view_858: "f32[2048, 10240]" = torch.ops.aten.reshape.default(mul_285, [2048, 10240]);  mul_285 = None
        permute_477: "f32[10240, 2560]" = torch.ops.aten.permute.default(arg661_1, [1, 0]);  arg661_1 = None
        addmm_251: "f32[2048, 2560]" = torch.ops.aten.addmm.default(arg662_1, view_858, permute_477);  arg662_1 = view_858 = permute_477 = None
        view_859: "f32[16, 128, 2560]" = torch.ops.aten.reshape.default(addmm_251, [16, 128, 2560]);  addmm_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:392 in forward, code: hidden_states = residual + hidden_states
        add_290: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(add_286, view_859);  add_286 = view_859 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:638 in forward, code: hidden_states = self.layer_norm(hidden_states)
        var_mean_77 = torch.ops.aten.var_mean.correction(add_290, [2], correction = 0, keepdim = True)
        getitem_250: "f32[16, 128, 1]" = var_mean_77[0]
        getitem_251: "f32[16, 128, 1]" = var_mean_77[1];  var_mean_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in forward, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_863: "i64[2048]" = torch.ops.aten.reshape.default(arg0_1, [-1]);  arg0_1 = None
        ne_1: "b8[2048]" = torch.ops.aten.ne.Scalar(view_863, -100)

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:638 in forward, code: hidden_states = self.layer_norm(hidden_states)
        sub_103: "f32[16, 128, 2560]" = torch.ops.aten.sub.Tensor(add_290, getitem_251);  add_290 = getitem_251 = None
        add_291: "f32[16, 128, 1]" = torch.ops.aten.add.Tensor(getitem_250, 1e-05);  getitem_250 = None
        rsqrt_77: "f32[16, 128, 1]" = torch.ops.aten.rsqrt.default(add_291);  add_291 = None
        mul_286: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(sub_103, rsqrt_77);  sub_103 = rsqrt_77 = None
        mul_287: "f32[16, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_286, arg663_1);  mul_286 = arg663_1 = None
        add_292: "f32[16, 128, 2560]" = torch.ops.aten.add.Tensor(mul_287, arg664_1);  mul_287 = arg664_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:885 in forward, code: lm_logits = self.lm_head(outputs[0])
        view_860: "f32[2048, 2560]" = torch.ops.aten.reshape.default(add_292, [2048, 2560]);  add_292 = None
        permute_478: "f32[2560, 8008]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        mm: "f32[2048, 8008]" = torch.ops.aten.mm.default(view_860, permute_478);  view_860 = permute_478 = None
        view_861: "f32[16, 128, 8008]" = torch.ops.aten.reshape.default(mm, [16, 128, 8008]);  mm = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:886 in forward, code: lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        add_293: "f32[16, 128, 8008]" = torch.ops.aten.add.Tensor(view_861, arg665_1);  view_861 = arg665_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/transformers/models/blenderbot/modeling_blenderbot.py:892 in forward, code: masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))
        view_862: "f32[2048, 8008]" = torch.ops.aten.reshape.default(add_293, [-1, 8008])
        amax_26: "f32[2048, 1]" = torch.ops.aten.amax.default(view_862, [1], True)
        sub_104: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(view_862, amax_26);  view_862 = amax_26 = None
        exp_26: "f32[2048, 8008]" = torch.ops.aten.exp.default(sub_104)
        sum_27: "f32[2048, 1]" = torch.ops.aten.sum.dim_IntList(exp_26, [1], True);  exp_26 = None
        log: "f32[2048, 1]" = torch.ops.aten.log.default(sum_27);  sum_27 = None
        sub_105: "f32[2048, 8008]" = torch.ops.aten.sub.Tensor(sub_104, log);  sub_104 = log = None
        ne: "b8[2048]" = torch.ops.aten.ne.Scalar(view_863, -100)
        full_default_126: "i64[]" = torch.ops.aten.full.default([], 0, dtype = torch.int64, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_76: "i64[2048]" = torch.ops.aten.where.self(ne, view_863, full_default_126);  ne = full_default_126 = None
        unsqueeze_12: "i64[2048, 1]" = torch.ops.aten.unsqueeze.default(where_76, 1);  where_76 = None
        gather: "f32[2048, 1]" = torch.ops.aten.gather.default(sub_105, 1, unsqueeze_12);  sub_105 = unsqueeze_12 = None
        squeeze: "f32[2048]" = torch.ops.aten.squeeze.dim(gather, 1);  gather = None
        neg: "f32[2048]" = torch.ops.aten.neg.default(squeeze);  squeeze = None
        full_default_127: "f32[]" = torch.ops.aten.full.default([], 0.0, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where_77: "f32[2048]" = torch.ops.aten.where.self(ne_1, neg, full_default_127);  ne_1 = neg = full_default_127 = None
        sum_29: "f32[]" = torch.ops.aten.sum.default(where_77);  where_77 = None
        ne_2: "b8[2048]" = torch.ops.aten.ne.Scalar(view_863, -100);  view_863 = None
        sum_28: "i64[]" = torch.ops.aten.sum.default(ne_2);  ne_2 = None
        convert_element_type: "f32[]" = torch.ops.prims.convert_element_type.default(sum_28, torch.float32);  sum_28 = None
        div_26: "f32[]" = torch.ops.aten.div.Tensor(sum_29, convert_element_type);  sum_29 = convert_element_type = None
        return (div_26, add_293, add_20)

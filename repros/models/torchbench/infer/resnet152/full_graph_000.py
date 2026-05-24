import torch
import torch._inductor.inductor_prims  # registers inductor prims ops
from torch import device
from math import inf, nan

class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f16[64, 3, 7, 7]", arg1_1: "f16[32, 3, 224, 224]", arg2_1: "f16[64]", arg3_1: "f16[64]", arg4_1: "f16[64]", arg5_1: "f16[64]", arg6_1: "f16[64, 64, 1, 1]", arg7_1: "f16[64]", arg8_1: "f16[64]", arg9_1: "f16[64]", arg10_1: "f16[64]", arg11_1: "f16[64, 64, 3, 3]", arg12_1: "f16[64]", arg13_1: "f16[64]", arg14_1: "f16[64]", arg15_1: "f16[64]", arg16_1: "f16[256, 64, 1, 1]", arg17_1: "f16[256]", arg18_1: "f16[256]", arg19_1: "f16[256]", arg20_1: "f16[256]", arg21_1: "f16[256, 64, 1, 1]", arg22_1: "f16[256]", arg23_1: "f16[256]", arg24_1: "f16[256]", arg25_1: "f16[256]", arg26_1: "f16[64, 256, 1, 1]", arg27_1: "f16[64]", arg28_1: "f16[64]", arg29_1: "f16[64]", arg30_1: "f16[64]", arg31_1: "f16[64, 64, 3, 3]", arg32_1: "f16[64]", arg33_1: "f16[64]", arg34_1: "f16[64]", arg35_1: "f16[64]", arg36_1: "f16[256, 64, 1, 1]", arg37_1: "f16[256]", arg38_1: "f16[256]", arg39_1: "f16[256]", arg40_1: "f16[256]", arg41_1: "f16[64, 256, 1, 1]", arg42_1: "f16[64]", arg43_1: "f16[64]", arg44_1: "f16[64]", arg45_1: "f16[64]", arg46_1: "f16[64, 64, 3, 3]", arg47_1: "f16[64]", arg48_1: "f16[64]", arg49_1: "f16[64]", arg50_1: "f16[64]", arg51_1: "f16[256, 64, 1, 1]", arg52_1: "f16[256]", arg53_1: "f16[256]", arg54_1: "f16[256]", arg55_1: "f16[256]", arg56_1: "f16[128, 256, 1, 1]", arg57_1: "f16[128]", arg58_1: "f16[128]", arg59_1: "f16[128]", arg60_1: "f16[128]", arg61_1: "f16[128, 128, 3, 3]", arg62_1: "f16[128]", arg63_1: "f16[128]", arg64_1: "f16[128]", arg65_1: "f16[128]", arg66_1: "f16[512, 128, 1, 1]", arg67_1: "f16[512]", arg68_1: "f16[512]", arg69_1: "f16[512]", arg70_1: "f16[512]", arg71_1: "f16[512, 256, 1, 1]", arg72_1: "f16[512]", arg73_1: "f16[512]", arg74_1: "f16[512]", arg75_1: "f16[512]", arg76_1: "f16[128, 512, 1, 1]", arg77_1: "f16[128]", arg78_1: "f16[128]", arg79_1: "f16[128]", arg80_1: "f16[128]", arg81_1: "f16[128, 128, 3, 3]", arg82_1: "f16[128]", arg83_1: "f16[128]", arg84_1: "f16[128]", arg85_1: "f16[128]", arg86_1: "f16[512, 128, 1, 1]", arg87_1: "f16[512]", arg88_1: "f16[512]", arg89_1: "f16[512]", arg90_1: "f16[512]", arg91_1: "f16[128, 512, 1, 1]", arg92_1: "f16[128]", arg93_1: "f16[128]", arg94_1: "f16[128]", arg95_1: "f16[128]", arg96_1: "f16[128, 128, 3, 3]", arg97_1: "f16[128]", arg98_1: "f16[128]", arg99_1: "f16[128]", arg100_1: "f16[128]", arg101_1: "f16[512, 128, 1, 1]", arg102_1: "f16[512]", arg103_1: "f16[512]", arg104_1: "f16[512]", arg105_1: "f16[512]", arg106_1: "f16[128, 512, 1, 1]", arg107_1: "f16[128]", arg108_1: "f16[128]", arg109_1: "f16[128]", arg110_1: "f16[128]", arg111_1: "f16[128, 128, 3, 3]", arg112_1: "f16[128]", arg113_1: "f16[128]", arg114_1: "f16[128]", arg115_1: "f16[128]", arg116_1: "f16[512, 128, 1, 1]", arg117_1: "f16[512]", arg118_1: "f16[512]", arg119_1: "f16[512]", arg120_1: "f16[512]", arg121_1: "f16[128, 512, 1, 1]", arg122_1: "f16[128]", arg123_1: "f16[128]", arg124_1: "f16[128]", arg125_1: "f16[128]", arg126_1: "f16[128, 128, 3, 3]", arg127_1: "f16[128]", arg128_1: "f16[128]", arg129_1: "f16[128]", arg130_1: "f16[128]", arg131_1: "f16[512, 128, 1, 1]", arg132_1: "f16[512]", arg133_1: "f16[512]", arg134_1: "f16[512]", arg135_1: "f16[512]", arg136_1: "f16[128, 512, 1, 1]", arg137_1: "f16[128]", arg138_1: "f16[128]", arg139_1: "f16[128]", arg140_1: "f16[128]", arg141_1: "f16[128, 128, 3, 3]", arg142_1: "f16[128]", arg143_1: "f16[128]", arg144_1: "f16[128]", arg145_1: "f16[128]", arg146_1: "f16[512, 128, 1, 1]", arg147_1: "f16[512]", arg148_1: "f16[512]", arg149_1: "f16[512]", arg150_1: "f16[512]", arg151_1: "f16[128, 512, 1, 1]", arg152_1: "f16[128]", arg153_1: "f16[128]", arg154_1: "f16[128]", arg155_1: "f16[128]", arg156_1: "f16[128, 128, 3, 3]", arg157_1: "f16[128]", arg158_1: "f16[128]", arg159_1: "f16[128]", arg160_1: "f16[128]", arg161_1: "f16[512, 128, 1, 1]", arg162_1: "f16[512]", arg163_1: "f16[512]", arg164_1: "f16[512]", arg165_1: "f16[512]", arg166_1: "f16[128, 512, 1, 1]", arg167_1: "f16[128]", arg168_1: "f16[128]", arg169_1: "f16[128]", arg170_1: "f16[128]", arg171_1: "f16[128, 128, 3, 3]", arg172_1: "f16[128]", arg173_1: "f16[128]", arg174_1: "f16[128]", arg175_1: "f16[128]", arg176_1: "f16[512, 128, 1, 1]", arg177_1: "f16[512]", arg178_1: "f16[512]", arg179_1: "f16[512]", arg180_1: "f16[512]", arg181_1: "f16[256, 512, 1, 1]", arg182_1: "f16[256]", arg183_1: "f16[256]", arg184_1: "f16[256]", arg185_1: "f16[256]", arg186_1: "f16[256, 256, 3, 3]", arg187_1: "f16[256]", arg188_1: "f16[256]", arg189_1: "f16[256]", arg190_1: "f16[256]", arg191_1: "f16[1024, 256, 1, 1]", arg192_1: "f16[1024]", arg193_1: "f16[1024]", arg194_1: "f16[1024]", arg195_1: "f16[1024]", arg196_1: "f16[1024, 512, 1, 1]", arg197_1: "f16[1024]", arg198_1: "f16[1024]", arg199_1: "f16[1024]", arg200_1: "f16[1024]", arg201_1: "f16[256, 1024, 1, 1]", arg202_1: "f16[256]", arg203_1: "f16[256]", arg204_1: "f16[256]", arg205_1: "f16[256]", arg206_1: "f16[256, 256, 3, 3]", arg207_1: "f16[256]", arg208_1: "f16[256]", arg209_1: "f16[256]", arg210_1: "f16[256]", arg211_1: "f16[1024, 256, 1, 1]", arg212_1: "f16[1024]", arg213_1: "f16[1024]", arg214_1: "f16[1024]", arg215_1: "f16[1024]", arg216_1: "f16[256, 1024, 1, 1]", arg217_1: "f16[256]", arg218_1: "f16[256]", arg219_1: "f16[256]", arg220_1: "f16[256]", arg221_1: "f16[256, 256, 3, 3]", arg222_1: "f16[256]", arg223_1: "f16[256]", arg224_1: "f16[256]", arg225_1: "f16[256]", arg226_1: "f16[1024, 256, 1, 1]", arg227_1: "f16[1024]", arg228_1: "f16[1024]", arg229_1: "f16[1024]", arg230_1: "f16[1024]", arg231_1: "f16[256, 1024, 1, 1]", arg232_1: "f16[256]", arg233_1: "f16[256]", arg234_1: "f16[256]", arg235_1: "f16[256]", arg236_1: "f16[256, 256, 3, 3]", arg237_1: "f16[256]", arg238_1: "f16[256]", arg239_1: "f16[256]", arg240_1: "f16[256]", arg241_1: "f16[1024, 256, 1, 1]", arg242_1: "f16[1024]", arg243_1: "f16[1024]", arg244_1: "f16[1024]", arg245_1: "f16[1024]", arg246_1: "f16[256, 1024, 1, 1]", arg247_1: "f16[256]", arg248_1: "f16[256]", arg249_1: "f16[256]", arg250_1: "f16[256]", arg251_1: "f16[256, 256, 3, 3]", arg252_1: "f16[256]", arg253_1: "f16[256]", arg254_1: "f16[256]", arg255_1: "f16[256]", arg256_1: "f16[1024, 256, 1, 1]", arg257_1: "f16[1024]", arg258_1: "f16[1024]", arg259_1: "f16[1024]", arg260_1: "f16[1024]", arg261_1: "f16[256, 1024, 1, 1]", arg262_1: "f16[256]", arg263_1: "f16[256]", arg264_1: "f16[256]", arg265_1: "f16[256]", arg266_1: "f16[256, 256, 3, 3]", arg267_1: "f16[256]", arg268_1: "f16[256]", arg269_1: "f16[256]", arg270_1: "f16[256]", arg271_1: "f16[1024, 256, 1, 1]", arg272_1: "f16[1024]", arg273_1: "f16[1024]", arg274_1: "f16[1024]", arg275_1: "f16[1024]", arg276_1: "f16[256, 1024, 1, 1]", arg277_1: "f16[256]", arg278_1: "f16[256]", arg279_1: "f16[256]", arg280_1: "f16[256]", arg281_1: "f16[256, 256, 3, 3]", arg282_1: "f16[256]", arg283_1: "f16[256]", arg284_1: "f16[256]", arg285_1: "f16[256]", arg286_1: "f16[1024, 256, 1, 1]", arg287_1: "f16[1024]", arg288_1: "f16[1024]", arg289_1: "f16[1024]", arg290_1: "f16[1024]", arg291_1: "f16[256, 1024, 1, 1]", arg292_1: "f16[256]", arg293_1: "f16[256]", arg294_1: "f16[256]", arg295_1: "f16[256]", arg296_1: "f16[256, 256, 3, 3]", arg297_1: "f16[256]", arg298_1: "f16[256]", arg299_1: "f16[256]", arg300_1: "f16[256]", arg301_1: "f16[1024, 256, 1, 1]", arg302_1: "f16[1024]", arg303_1: "f16[1024]", arg304_1: "f16[1024]", arg305_1: "f16[1024]", arg306_1: "f16[256, 1024, 1, 1]", arg307_1: "f16[256]", arg308_1: "f16[256]", arg309_1: "f16[256]", arg310_1: "f16[256]", arg311_1: "f16[256, 256, 3, 3]", arg312_1: "f16[256]", arg313_1: "f16[256]", arg314_1: "f16[256]", arg315_1: "f16[256]", arg316_1: "f16[1024, 256, 1, 1]", arg317_1: "f16[1024]", arg318_1: "f16[1024]", arg319_1: "f16[1024]", arg320_1: "f16[1024]", arg321_1: "f16[256, 1024, 1, 1]", arg322_1: "f16[256]", arg323_1: "f16[256]", arg324_1: "f16[256]", arg325_1: "f16[256]", arg326_1: "f16[256, 256, 3, 3]", arg327_1: "f16[256]", arg328_1: "f16[256]", arg329_1: "f16[256]", arg330_1: "f16[256]", arg331_1: "f16[1024, 256, 1, 1]", arg332_1: "f16[1024]", arg333_1: "f16[1024]", arg334_1: "f16[1024]", arg335_1: "f16[1024]", arg336_1: "f16[256, 1024, 1, 1]", arg337_1: "f16[256]", arg338_1: "f16[256]", arg339_1: "f16[256]", arg340_1: "f16[256]", arg341_1: "f16[256, 256, 3, 3]", arg342_1: "f16[256]", arg343_1: "f16[256]", arg344_1: "f16[256]", arg345_1: "f16[256]", arg346_1: "f16[1024, 256, 1, 1]", arg347_1: "f16[1024]", arg348_1: "f16[1024]", arg349_1: "f16[1024]", arg350_1: "f16[1024]", arg351_1: "f16[256, 1024, 1, 1]", arg352_1: "f16[256]", arg353_1: "f16[256]", arg354_1: "f16[256]", arg355_1: "f16[256]", arg356_1: "f16[256, 256, 3, 3]", arg357_1: "f16[256]", arg358_1: "f16[256]", arg359_1: "f16[256]", arg360_1: "f16[256]", arg361_1: "f16[1024, 256, 1, 1]", arg362_1: "f16[1024]", arg363_1: "f16[1024]", arg364_1: "f16[1024]", arg365_1: "f16[1024]", arg366_1: "f16[256, 1024, 1, 1]", arg367_1: "f16[256]", arg368_1: "f16[256]", arg369_1: "f16[256]", arg370_1: "f16[256]", arg371_1: "f16[256, 256, 3, 3]", arg372_1: "f16[256]", arg373_1: "f16[256]", arg374_1: "f16[256]", arg375_1: "f16[256]", arg376_1: "f16[1024, 256, 1, 1]", arg377_1: "f16[1024]", arg378_1: "f16[1024]", arg379_1: "f16[1024]", arg380_1: "f16[1024]", arg381_1: "f16[256, 1024, 1, 1]", arg382_1: "f16[256]", arg383_1: "f16[256]", arg384_1: "f16[256]", arg385_1: "f16[256]", arg386_1: "f16[256, 256, 3, 3]", arg387_1: "f16[256]", arg388_1: "f16[256]", arg389_1: "f16[256]", arg390_1: "f16[256]", arg391_1: "f16[1024, 256, 1, 1]", arg392_1: "f16[1024]", arg393_1: "f16[1024]", arg394_1: "f16[1024]", arg395_1: "f16[1024]", arg396_1: "f16[256, 1024, 1, 1]", arg397_1: "f16[256]", arg398_1: "f16[256]", arg399_1: "f16[256]", arg400_1: "f16[256]", arg401_1: "f16[256, 256, 3, 3]", arg402_1: "f16[256]", arg403_1: "f16[256]", arg404_1: "f16[256]", arg405_1: "f16[256]", arg406_1: "f16[1024, 256, 1, 1]", arg407_1: "f16[1024]", arg408_1: "f16[1024]", arg409_1: "f16[1024]", arg410_1: "f16[1024]", arg411_1: "f16[256, 1024, 1, 1]", arg412_1: "f16[256]", arg413_1: "f16[256]", arg414_1: "f16[256]", arg415_1: "f16[256]", arg416_1: "f16[256, 256, 3, 3]", arg417_1: "f16[256]", arg418_1: "f16[256]", arg419_1: "f16[256]", arg420_1: "f16[256]", arg421_1: "f16[1024, 256, 1, 1]", arg422_1: "f16[1024]", arg423_1: "f16[1024]", arg424_1: "f16[1024]", arg425_1: "f16[1024]", arg426_1: "f16[256, 1024, 1, 1]", arg427_1: "f16[256]", arg428_1: "f16[256]", arg429_1: "f16[256]", arg430_1: "f16[256]", arg431_1: "f16[256, 256, 3, 3]", arg432_1: "f16[256]", arg433_1: "f16[256]", arg434_1: "f16[256]", arg435_1: "f16[256]", arg436_1: "f16[1024, 256, 1, 1]", arg437_1: "f16[1024]", arg438_1: "f16[1024]", arg439_1: "f16[1024]", arg440_1: "f16[1024]", arg441_1: "f16[256, 1024, 1, 1]", arg442_1: "f16[256]", arg443_1: "f16[256]", arg444_1: "f16[256]", arg445_1: "f16[256]", arg446_1: "f16[256, 256, 3, 3]", arg447_1: "f16[256]", arg448_1: "f16[256]", arg449_1: "f16[256]", arg450_1: "f16[256]", arg451_1: "f16[1024, 256, 1, 1]", arg452_1: "f16[1024]", arg453_1: "f16[1024]", arg454_1: "f16[1024]", arg455_1: "f16[1024]", arg456_1: "f16[256, 1024, 1, 1]", arg457_1: "f16[256]", arg458_1: "f16[256]", arg459_1: "f16[256]", arg460_1: "f16[256]", arg461_1: "f16[256, 256, 3, 3]", arg462_1: "f16[256]", arg463_1: "f16[256]", arg464_1: "f16[256]", arg465_1: "f16[256]", arg466_1: "f16[1024, 256, 1, 1]", arg467_1: "f16[1024]", arg468_1: "f16[1024]", arg469_1: "f16[1024]", arg470_1: "f16[1024]", arg471_1: "f16[256, 1024, 1, 1]", arg472_1: "f16[256]", arg473_1: "f16[256]", arg474_1: "f16[256]", arg475_1: "f16[256]", arg476_1: "f16[256, 256, 3, 3]", arg477_1: "f16[256]", arg478_1: "f16[256]", arg479_1: "f16[256]", arg480_1: "f16[256]", arg481_1: "f16[1024, 256, 1, 1]", arg482_1: "f16[1024]", arg483_1: "f16[1024]", arg484_1: "f16[1024]", arg485_1: "f16[1024]", arg486_1: "f16[256, 1024, 1, 1]", arg487_1: "f16[256]", arg488_1: "f16[256]", arg489_1: "f16[256]", arg490_1: "f16[256]", arg491_1: "f16[256, 256, 3, 3]", arg492_1: "f16[256]", arg493_1: "f16[256]", arg494_1: "f16[256]", arg495_1: "f16[256]", arg496_1: "f16[1024, 256, 1, 1]", arg497_1: "f16[1024]", arg498_1: "f16[1024]", arg499_1: "f16[1024]", arg500_1: "f16[1024]", arg501_1: "f16[256, 1024, 1, 1]", arg502_1: "f16[256]", arg503_1: "f16[256]", arg504_1: "f16[256]", arg505_1: "f16[256]", arg506_1: "f16[256, 256, 3, 3]", arg507_1: "f16[256]", arg508_1: "f16[256]", arg509_1: "f16[256]", arg510_1: "f16[256]", arg511_1: "f16[1024, 256, 1, 1]", arg512_1: "f16[1024]", arg513_1: "f16[1024]", arg514_1: "f16[1024]", arg515_1: "f16[1024]", arg516_1: "f16[256, 1024, 1, 1]", arg517_1: "f16[256]", arg518_1: "f16[256]", arg519_1: "f16[256]", arg520_1: "f16[256]", arg521_1: "f16[256, 256, 3, 3]", arg522_1: "f16[256]", arg523_1: "f16[256]", arg524_1: "f16[256]", arg525_1: "f16[256]", arg526_1: "f16[1024, 256, 1, 1]", arg527_1: "f16[1024]", arg528_1: "f16[1024]", arg529_1: "f16[1024]", arg530_1: "f16[1024]", arg531_1: "f16[256, 1024, 1, 1]", arg532_1: "f16[256]", arg533_1: "f16[256]", arg534_1: "f16[256]", arg535_1: "f16[256]", arg536_1: "f16[256, 256, 3, 3]", arg537_1: "f16[256]", arg538_1: "f16[256]", arg539_1: "f16[256]", arg540_1: "f16[256]", arg541_1: "f16[1024, 256, 1, 1]", arg542_1: "f16[1024]", arg543_1: "f16[1024]", arg544_1: "f16[1024]", arg545_1: "f16[1024]", arg546_1: "f16[256, 1024, 1, 1]", arg547_1: "f16[256]", arg548_1: "f16[256]", arg549_1: "f16[256]", arg550_1: "f16[256]", arg551_1: "f16[256, 256, 3, 3]", arg552_1: "f16[256]", arg553_1: "f16[256]", arg554_1: "f16[256]", arg555_1: "f16[256]", arg556_1: "f16[1024, 256, 1, 1]", arg557_1: "f16[1024]", arg558_1: "f16[1024]", arg559_1: "f16[1024]", arg560_1: "f16[1024]", arg561_1: "f16[256, 1024, 1, 1]", arg562_1: "f16[256]", arg563_1: "f16[256]", arg564_1: "f16[256]", arg565_1: "f16[256]", arg566_1: "f16[256, 256, 3, 3]", arg567_1: "f16[256]", arg568_1: "f16[256]", arg569_1: "f16[256]", arg570_1: "f16[256]", arg571_1: "f16[1024, 256, 1, 1]", arg572_1: "f16[1024]", arg573_1: "f16[1024]", arg574_1: "f16[1024]", arg575_1: "f16[1024]", arg576_1: "f16[256, 1024, 1, 1]", arg577_1: "f16[256]", arg578_1: "f16[256]", arg579_1: "f16[256]", arg580_1: "f16[256]", arg581_1: "f16[256, 256, 3, 3]", arg582_1: "f16[256]", arg583_1: "f16[256]", arg584_1: "f16[256]", arg585_1: "f16[256]", arg586_1: "f16[1024, 256, 1, 1]", arg587_1: "f16[1024]", arg588_1: "f16[1024]", arg589_1: "f16[1024]", arg590_1: "f16[1024]", arg591_1: "f16[256, 1024, 1, 1]", arg592_1: "f16[256]", arg593_1: "f16[256]", arg594_1: "f16[256]", arg595_1: "f16[256]", arg596_1: "f16[256, 256, 3, 3]", arg597_1: "f16[256]", arg598_1: "f16[256]", arg599_1: "f16[256]", arg600_1: "f16[256]", arg601_1: "f16[1024, 256, 1, 1]", arg602_1: "f16[1024]", arg603_1: "f16[1024]", arg604_1: "f16[1024]", arg605_1: "f16[1024]", arg606_1: "f16[256, 1024, 1, 1]", arg607_1: "f16[256]", arg608_1: "f16[256]", arg609_1: "f16[256]", arg610_1: "f16[256]", arg611_1: "f16[256, 256, 3, 3]", arg612_1: "f16[256]", arg613_1: "f16[256]", arg614_1: "f16[256]", arg615_1: "f16[256]", arg616_1: "f16[1024, 256, 1, 1]", arg617_1: "f16[1024]", arg618_1: "f16[1024]", arg619_1: "f16[1024]", arg620_1: "f16[1024]", arg621_1: "f16[256, 1024, 1, 1]", arg622_1: "f16[256]", arg623_1: "f16[256]", arg624_1: "f16[256]", arg625_1: "f16[256]", arg626_1: "f16[256, 256, 3, 3]", arg627_1: "f16[256]", arg628_1: "f16[256]", arg629_1: "f16[256]", arg630_1: "f16[256]", arg631_1: "f16[1024, 256, 1, 1]", arg632_1: "f16[1024]", arg633_1: "f16[1024]", arg634_1: "f16[1024]", arg635_1: "f16[1024]", arg636_1: "f16[256, 1024, 1, 1]", arg637_1: "f16[256]", arg638_1: "f16[256]", arg639_1: "f16[256]", arg640_1: "f16[256]", arg641_1: "f16[256, 256, 3, 3]", arg642_1: "f16[256]", arg643_1: "f16[256]", arg644_1: "f16[256]", arg645_1: "f16[256]", arg646_1: "f16[1024, 256, 1, 1]", arg647_1: "f16[1024]", arg648_1: "f16[1024]", arg649_1: "f16[1024]", arg650_1: "f16[1024]", arg651_1: "f16[256, 1024, 1, 1]", arg652_1: "f16[256]", arg653_1: "f16[256]", arg654_1: "f16[256]", arg655_1: "f16[256]", arg656_1: "f16[256, 256, 3, 3]", arg657_1: "f16[256]", arg658_1: "f16[256]", arg659_1: "f16[256]", arg660_1: "f16[256]", arg661_1: "f16[1024, 256, 1, 1]", arg662_1: "f16[1024]", arg663_1: "f16[1024]", arg664_1: "f16[1024]", arg665_1: "f16[1024]", arg666_1: "f16[256, 1024, 1, 1]", arg667_1: "f16[256]", arg668_1: "f16[256]", arg669_1: "f16[256]", arg670_1: "f16[256]", arg671_1: "f16[256, 256, 3, 3]", arg672_1: "f16[256]", arg673_1: "f16[256]", arg674_1: "f16[256]", arg675_1: "f16[256]", arg676_1: "f16[1024, 256, 1, 1]", arg677_1: "f16[1024]", arg678_1: "f16[1024]", arg679_1: "f16[1024]", arg680_1: "f16[1024]", arg681_1: "f16[256, 1024, 1, 1]", arg682_1: "f16[256]", arg683_1: "f16[256]", arg684_1: "f16[256]", arg685_1: "f16[256]", arg686_1: "f16[256, 256, 3, 3]", arg687_1: "f16[256]", arg688_1: "f16[256]", arg689_1: "f16[256]", arg690_1: "f16[256]", arg691_1: "f16[1024, 256, 1, 1]", arg692_1: "f16[1024]", arg693_1: "f16[1024]", arg694_1: "f16[1024]", arg695_1: "f16[1024]", arg696_1: "f16[256, 1024, 1, 1]", arg697_1: "f16[256]", arg698_1: "f16[256]", arg699_1: "f16[256]", arg700_1: "f16[256]", arg701_1: "f16[256, 256, 3, 3]", arg702_1: "f16[256]", arg703_1: "f16[256]", arg704_1: "f16[256]", arg705_1: "f16[256]", arg706_1: "f16[1024, 256, 1, 1]", arg707_1: "f16[1024]", arg708_1: "f16[1024]", arg709_1: "f16[1024]", arg710_1: "f16[1024]", arg711_1: "f16[256, 1024, 1, 1]", arg712_1: "f16[256]", arg713_1: "f16[256]", arg714_1: "f16[256]", arg715_1: "f16[256]", arg716_1: "f16[256, 256, 3, 3]", arg717_1: "f16[256]", arg718_1: "f16[256]", arg719_1: "f16[256]", arg720_1: "f16[256]", arg721_1: "f16[1024, 256, 1, 1]", arg722_1: "f16[1024]", arg723_1: "f16[1024]", arg724_1: "f16[1024]", arg725_1: "f16[1024]", arg726_1: "f16[512, 1024, 1, 1]", arg727_1: "f16[512]", arg728_1: "f16[512]", arg729_1: "f16[512]", arg730_1: "f16[512]", arg731_1: "f16[512, 512, 3, 3]", arg732_1: "f16[512]", arg733_1: "f16[512]", arg734_1: "f16[512]", arg735_1: "f16[512]", arg736_1: "f16[2048, 512, 1, 1]", arg737_1: "f16[2048]", arg738_1: "f16[2048]", arg739_1: "f16[2048]", arg740_1: "f16[2048]", arg741_1: "f16[2048, 1024, 1, 1]", arg742_1: "f16[2048]", arg743_1: "f16[2048]", arg744_1: "f16[2048]", arg745_1: "f16[2048]", arg746_1: "f16[512, 2048, 1, 1]", arg747_1: "f16[512]", arg748_1: "f16[512]", arg749_1: "f16[512]", arg750_1: "f16[512]", arg751_1: "f16[512, 512, 3, 3]", arg752_1: "f16[512]", arg753_1: "f16[512]", arg754_1: "f16[512]", arg755_1: "f16[512]", arg756_1: "f16[2048, 512, 1, 1]", arg757_1: "f16[2048]", arg758_1: "f16[2048]", arg759_1: "f16[2048]", arg760_1: "f16[2048]", arg761_1: "f16[512, 2048, 1, 1]", arg762_1: "f16[512]", arg763_1: "f16[512]", arg764_1: "f16[512]", arg765_1: "f16[512]", arg766_1: "f16[512, 512, 3, 3]", arg767_1: "f16[512]", arg768_1: "f16[512]", arg769_1: "f16[512]", arg770_1: "f16[512]", arg771_1: "f16[2048, 512, 1, 1]", arg772_1: "f16[2048]", arg773_1: "f16[2048]", arg774_1: "f16[2048]", arg775_1: "f16[2048]", arg776_1: "f16[1000, 2048]", arg777_1: "f16[1000]"):
        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:268 in _forward_impl, code: x = self.conv1(x)
        convolution: "f16[32, 64, 112, 112]" = torch.ops.aten.convolution.default(arg1_1, arg0_1, None, [2, 2], [3, 3], [1, 1], False, [0, 0], 1);  arg1_1 = arg0_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:269 in _forward_impl, code: x = self.bn1(x)
        convert_element_type: "f32[64]" = torch.ops.prims.convert_element_type.default(arg2_1, torch.float32);  arg2_1 = None
        unsqueeze: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type, -1);  convert_element_type = None
        unsqueeze_1: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze, -1);  unsqueeze = None
        sub: "f32[32, 64, 112, 112]" = torch.ops.aten.sub.Tensor(convolution, unsqueeze_1);  convolution = unsqueeze_1 = None
        convert_element_type_1: "f32[64]" = torch.ops.prims.convert_element_type.default(arg3_1, torch.float32);  arg3_1 = None
        add: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_1, 1e-05);  convert_element_type_1 = None
        sqrt: "f32[64]" = torch.ops.aten.sqrt.default(add);  add = None
        reciprocal: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt);  sqrt = None
        mul: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal, 1);  reciprocal = None
        unsqueeze_2: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul, -1);  mul = None
        unsqueeze_3: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_2, -1);  unsqueeze_2 = None
        mul_1: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(sub, unsqueeze_3);  sub = unsqueeze_3 = None
        unsqueeze_4: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg4_1, -1);  arg4_1 = None
        unsqueeze_5: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, -1);  unsqueeze_4 = None
        mul_2: "f32[32, 64, 112, 112]" = torch.ops.aten.mul.Tensor(mul_1, unsqueeze_5);  mul_1 = unsqueeze_5 = None
        unsqueeze_6: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg5_1, -1);  arg5_1 = None
        unsqueeze_7: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_6, -1);  unsqueeze_6 = None
        add_1: "f32[32, 64, 112, 112]" = torch.ops.aten.add.Tensor(mul_2, unsqueeze_7);  mul_2 = unsqueeze_7 = None
        convert_element_type_2: "f16[32, 64, 112, 112]" = torch.ops.prims.convert_element_type.default(add_1, torch.float16);  add_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:270 in _forward_impl, code: x = self.relu(x)
        relu: "f16[32, 64, 112, 112]" = torch.ops.aten.relu.default(convert_element_type_2);  convert_element_type_2 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:271 in _forward_impl, code: x = self.maxpool(x)
        _low_memory_max_pool_with_offsets = torch.ops.prims._low_memory_max_pool_with_offsets.default(relu, [3, 3], [2, 2], [1, 1], [1, 1], False);  relu = None
        getitem: "f16[32, 64, 56, 56]" = _low_memory_max_pool_with_offsets[0];  _low_memory_max_pool_with_offsets = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_1: "f16[32, 64, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg6_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg6_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_3: "f32[64]" = torch.ops.prims.convert_element_type.default(arg7_1, torch.float32);  arg7_1 = None
        unsqueeze_8: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_3, -1);  convert_element_type_3 = None
        unsqueeze_9: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_8, -1);  unsqueeze_8 = None
        sub_1: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_1, unsqueeze_9);  convolution_1 = unsqueeze_9 = None
        convert_element_type_4: "f32[64]" = torch.ops.prims.convert_element_type.default(arg8_1, torch.float32);  arg8_1 = None
        add_2: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_4, 1e-05);  convert_element_type_4 = None
        sqrt_1: "f32[64]" = torch.ops.aten.sqrt.default(add_2);  add_2 = None
        reciprocal_1: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_1);  sqrt_1 = None
        mul_3: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_1, 1);  reciprocal_1 = None
        unsqueeze_10: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_3, -1);  mul_3 = None
        unsqueeze_11: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_10, -1);  unsqueeze_10 = None
        mul_4: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_1, unsqueeze_11);  sub_1 = unsqueeze_11 = None
        unsqueeze_12: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg9_1, -1);  arg9_1 = None
        unsqueeze_13: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_12, -1);  unsqueeze_12 = None
        mul_5: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_4, unsqueeze_13);  mul_4 = unsqueeze_13 = None
        unsqueeze_14: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg10_1, -1);  arg10_1 = None
        unsqueeze_15: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_14, -1);  unsqueeze_14 = None
        add_3: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_5, unsqueeze_15);  mul_5 = unsqueeze_15 = None
        convert_element_type_5: "f16[32, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_3, torch.float16);  add_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_1: "f16[32, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_5);  convert_element_type_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_2: "f16[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_1, arg11_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_1 = arg11_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_6: "f32[64]" = torch.ops.prims.convert_element_type.default(arg12_1, torch.float32);  arg12_1 = None
        unsqueeze_16: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_6, -1);  convert_element_type_6 = None
        unsqueeze_17: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_16, -1);  unsqueeze_16 = None
        sub_2: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_2, unsqueeze_17);  convolution_2 = unsqueeze_17 = None
        convert_element_type_7: "f32[64]" = torch.ops.prims.convert_element_type.default(arg13_1, torch.float32);  arg13_1 = None
        add_4: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_7, 1e-05);  convert_element_type_7 = None
        sqrt_2: "f32[64]" = torch.ops.aten.sqrt.default(add_4);  add_4 = None
        reciprocal_2: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_2);  sqrt_2 = None
        mul_6: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_2, 1);  reciprocal_2 = None
        unsqueeze_18: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_6, -1);  mul_6 = None
        unsqueeze_19: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_18, -1);  unsqueeze_18 = None
        mul_7: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_2, unsqueeze_19);  sub_2 = unsqueeze_19 = None
        unsqueeze_20: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg14_1, -1);  arg14_1 = None
        unsqueeze_21: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_20, -1);  unsqueeze_20 = None
        mul_8: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_7, unsqueeze_21);  mul_7 = unsqueeze_21 = None
        unsqueeze_22: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg15_1, -1);  arg15_1 = None
        unsqueeze_23: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_22, -1);  unsqueeze_22 = None
        add_5: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_8, unsqueeze_23);  mul_8 = unsqueeze_23 = None
        convert_element_type_8: "f16[32, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_5, torch.float16);  add_5 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_2: "f16[32, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_8);  convert_element_type_8 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_3: "f16[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_2, arg16_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_2 = arg16_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_9: "f32[256]" = torch.ops.prims.convert_element_type.default(arg17_1, torch.float32);  arg17_1 = None
        unsqueeze_24: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_9, -1);  convert_element_type_9 = None
        unsqueeze_25: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_24, -1);  unsqueeze_24 = None
        sub_3: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_3, unsqueeze_25);  convolution_3 = unsqueeze_25 = None
        convert_element_type_10: "f32[256]" = torch.ops.prims.convert_element_type.default(arg18_1, torch.float32);  arg18_1 = None
        add_6: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_10, 1e-05);  convert_element_type_10 = None
        sqrt_3: "f32[256]" = torch.ops.aten.sqrt.default(add_6);  add_6 = None
        reciprocal_3: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_3);  sqrt_3 = None
        mul_9: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_3, 1);  reciprocal_3 = None
        unsqueeze_26: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_9, -1);  mul_9 = None
        unsqueeze_27: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_26, -1);  unsqueeze_26 = None
        mul_10: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_3, unsqueeze_27);  sub_3 = unsqueeze_27 = None
        unsqueeze_28: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg19_1, -1);  arg19_1 = None
        unsqueeze_29: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_28, -1);  unsqueeze_28 = None
        mul_11: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_10, unsqueeze_29);  mul_10 = unsqueeze_29 = None
        unsqueeze_30: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg20_1, -1);  arg20_1 = None
        unsqueeze_31: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_30, -1);  unsqueeze_30 = None
        add_7: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_11, unsqueeze_31);  mul_11 = unsqueeze_31 = None
        convert_element_type_11: "f16[32, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_7, torch.float16);  add_7 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_4: "f16[32, 256, 56, 56]" = torch.ops.aten.convolution.default(getitem, arg21_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  getitem = arg21_1 = None
        convert_element_type_12: "f32[256]" = torch.ops.prims.convert_element_type.default(arg22_1, torch.float32);  arg22_1 = None
        unsqueeze_32: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_12, -1);  convert_element_type_12 = None
        unsqueeze_33: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_32, -1);  unsqueeze_32 = None
        sub_4: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_4, unsqueeze_33);  convolution_4 = unsqueeze_33 = None
        convert_element_type_13: "f32[256]" = torch.ops.prims.convert_element_type.default(arg23_1, torch.float32);  arg23_1 = None
        add_8: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_13, 1e-05);  convert_element_type_13 = None
        sqrt_4: "f32[256]" = torch.ops.aten.sqrt.default(add_8);  add_8 = None
        reciprocal_4: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_4);  sqrt_4 = None
        mul_12: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_4, 1);  reciprocal_4 = None
        unsqueeze_34: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_12, -1);  mul_12 = None
        unsqueeze_35: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_34, -1);  unsqueeze_34 = None
        mul_13: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_4, unsqueeze_35);  sub_4 = unsqueeze_35 = None
        unsqueeze_36: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg24_1, -1);  arg24_1 = None
        unsqueeze_37: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_36, -1);  unsqueeze_36 = None
        mul_14: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_13, unsqueeze_37);  mul_13 = unsqueeze_37 = None
        unsqueeze_38: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg25_1, -1);  arg25_1 = None
        unsqueeze_39: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_38, -1);  unsqueeze_38 = None
        add_9: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_14, unsqueeze_39);  mul_14 = unsqueeze_39 = None
        convert_element_type_14: "f16[32, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_9, torch.float16);  add_9 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_10: "f16[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_11, convert_element_type_14);  convert_element_type_11 = convert_element_type_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_3: "f16[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_10);  add_10 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_5: "f16[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_3, arg26_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg26_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_15: "f32[64]" = torch.ops.prims.convert_element_type.default(arg27_1, torch.float32);  arg27_1 = None
        unsqueeze_40: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_15, -1);  convert_element_type_15 = None
        unsqueeze_41: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_40, -1);  unsqueeze_40 = None
        sub_5: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_5, unsqueeze_41);  convolution_5 = unsqueeze_41 = None
        convert_element_type_16: "f32[64]" = torch.ops.prims.convert_element_type.default(arg28_1, torch.float32);  arg28_1 = None
        add_11: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_16, 1e-05);  convert_element_type_16 = None
        sqrt_5: "f32[64]" = torch.ops.aten.sqrt.default(add_11);  add_11 = None
        reciprocal_5: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_5);  sqrt_5 = None
        mul_15: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_5, 1);  reciprocal_5 = None
        unsqueeze_42: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_15, -1);  mul_15 = None
        unsqueeze_43: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_42, -1);  unsqueeze_42 = None
        mul_16: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_5, unsqueeze_43);  sub_5 = unsqueeze_43 = None
        unsqueeze_44: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg29_1, -1);  arg29_1 = None
        unsqueeze_45: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_44, -1);  unsqueeze_44 = None
        mul_17: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_16, unsqueeze_45);  mul_16 = unsqueeze_45 = None
        unsqueeze_46: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg30_1, -1);  arg30_1 = None
        unsqueeze_47: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_46, -1);  unsqueeze_46 = None
        add_12: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_17, unsqueeze_47);  mul_17 = unsqueeze_47 = None
        convert_element_type_17: "f16[32, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_12, torch.float16);  add_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_4: "f16[32, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_17);  convert_element_type_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_6: "f16[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_4, arg31_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_4 = arg31_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_18: "f32[64]" = torch.ops.prims.convert_element_type.default(arg32_1, torch.float32);  arg32_1 = None
        unsqueeze_48: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_18, -1);  convert_element_type_18 = None
        unsqueeze_49: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_48, -1);  unsqueeze_48 = None
        sub_6: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_6, unsqueeze_49);  convolution_6 = unsqueeze_49 = None
        convert_element_type_19: "f32[64]" = torch.ops.prims.convert_element_type.default(arg33_1, torch.float32);  arg33_1 = None
        add_13: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_19, 1e-05);  convert_element_type_19 = None
        sqrt_6: "f32[64]" = torch.ops.aten.sqrt.default(add_13);  add_13 = None
        reciprocal_6: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_6);  sqrt_6 = None
        mul_18: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_6, 1);  reciprocal_6 = None
        unsqueeze_50: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_18, -1);  mul_18 = None
        unsqueeze_51: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_50, -1);  unsqueeze_50 = None
        mul_19: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_6, unsqueeze_51);  sub_6 = unsqueeze_51 = None
        unsqueeze_52: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg34_1, -1);  arg34_1 = None
        unsqueeze_53: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_52, -1);  unsqueeze_52 = None
        mul_20: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_19, unsqueeze_53);  mul_19 = unsqueeze_53 = None
        unsqueeze_54: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg35_1, -1);  arg35_1 = None
        unsqueeze_55: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_54, -1);  unsqueeze_54 = None
        add_14: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_20, unsqueeze_55);  mul_20 = unsqueeze_55 = None
        convert_element_type_20: "f16[32, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_14, torch.float16);  add_14 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_5: "f16[32, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_20);  convert_element_type_20 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_7: "f16[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_5, arg36_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_5 = arg36_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_21: "f32[256]" = torch.ops.prims.convert_element_type.default(arg37_1, torch.float32);  arg37_1 = None
        unsqueeze_56: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_21, -1);  convert_element_type_21 = None
        unsqueeze_57: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_56, -1);  unsqueeze_56 = None
        sub_7: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_7, unsqueeze_57);  convolution_7 = unsqueeze_57 = None
        convert_element_type_22: "f32[256]" = torch.ops.prims.convert_element_type.default(arg38_1, torch.float32);  arg38_1 = None
        add_15: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_22, 1e-05);  convert_element_type_22 = None
        sqrt_7: "f32[256]" = torch.ops.aten.sqrt.default(add_15);  add_15 = None
        reciprocal_7: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_7);  sqrt_7 = None
        mul_21: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_7, 1);  reciprocal_7 = None
        unsqueeze_58: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_21, -1);  mul_21 = None
        unsqueeze_59: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_58, -1);  unsqueeze_58 = None
        mul_22: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_7, unsqueeze_59);  sub_7 = unsqueeze_59 = None
        unsqueeze_60: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg39_1, -1);  arg39_1 = None
        unsqueeze_61: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_60, -1);  unsqueeze_60 = None
        mul_23: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_22, unsqueeze_61);  mul_22 = unsqueeze_61 = None
        unsqueeze_62: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg40_1, -1);  arg40_1 = None
        unsqueeze_63: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_62, -1);  unsqueeze_62 = None
        add_16: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_23, unsqueeze_63);  mul_23 = unsqueeze_63 = None
        convert_element_type_23: "f16[32, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_16, torch.float16);  add_16 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_17: "f16[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_23, relu_3);  convert_element_type_23 = relu_3 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_6: "f16[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_17);  add_17 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_8: "f16[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_6, arg41_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg41_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_24: "f32[64]" = torch.ops.prims.convert_element_type.default(arg42_1, torch.float32);  arg42_1 = None
        unsqueeze_64: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_24, -1);  convert_element_type_24 = None
        unsqueeze_65: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_64, -1);  unsqueeze_64 = None
        sub_8: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_8, unsqueeze_65);  convolution_8 = unsqueeze_65 = None
        convert_element_type_25: "f32[64]" = torch.ops.prims.convert_element_type.default(arg43_1, torch.float32);  arg43_1 = None
        add_18: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_25, 1e-05);  convert_element_type_25 = None
        sqrt_8: "f32[64]" = torch.ops.aten.sqrt.default(add_18);  add_18 = None
        reciprocal_8: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_8);  sqrt_8 = None
        mul_24: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_8, 1);  reciprocal_8 = None
        unsqueeze_66: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_24, -1);  mul_24 = None
        unsqueeze_67: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_66, -1);  unsqueeze_66 = None
        mul_25: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_8, unsqueeze_67);  sub_8 = unsqueeze_67 = None
        unsqueeze_68: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg44_1, -1);  arg44_1 = None
        unsqueeze_69: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_68, -1);  unsqueeze_68 = None
        mul_26: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_25, unsqueeze_69);  mul_25 = unsqueeze_69 = None
        unsqueeze_70: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg45_1, -1);  arg45_1 = None
        unsqueeze_71: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_70, -1);  unsqueeze_70 = None
        add_19: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_26, unsqueeze_71);  mul_26 = unsqueeze_71 = None
        convert_element_type_26: "f16[32, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_19, torch.float16);  add_19 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_7: "f16[32, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_26);  convert_element_type_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_9: "f16[32, 64, 56, 56]" = torch.ops.aten.convolution.default(relu_7, arg46_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_7 = arg46_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_27: "f32[64]" = torch.ops.prims.convert_element_type.default(arg47_1, torch.float32);  arg47_1 = None
        unsqueeze_72: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_27, -1);  convert_element_type_27 = None
        unsqueeze_73: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_72, -1);  unsqueeze_72 = None
        sub_9: "f32[32, 64, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_9, unsqueeze_73);  convolution_9 = unsqueeze_73 = None
        convert_element_type_28: "f32[64]" = torch.ops.prims.convert_element_type.default(arg48_1, torch.float32);  arg48_1 = None
        add_20: "f32[64]" = torch.ops.aten.add.Tensor(convert_element_type_28, 1e-05);  convert_element_type_28 = None
        sqrt_9: "f32[64]" = torch.ops.aten.sqrt.default(add_20);  add_20 = None
        reciprocal_9: "f32[64]" = torch.ops.aten.reciprocal.default(sqrt_9);  sqrt_9 = None
        mul_27: "f32[64]" = torch.ops.aten.mul.Tensor(reciprocal_9, 1);  reciprocal_9 = None
        unsqueeze_74: "f32[64, 1]" = torch.ops.aten.unsqueeze.default(mul_27, -1);  mul_27 = None
        unsqueeze_75: "f32[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_74, -1);  unsqueeze_74 = None
        mul_28: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(sub_9, unsqueeze_75);  sub_9 = unsqueeze_75 = None
        unsqueeze_76: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg49_1, -1);  arg49_1 = None
        unsqueeze_77: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_76, -1);  unsqueeze_76 = None
        mul_29: "f32[32, 64, 56, 56]" = torch.ops.aten.mul.Tensor(mul_28, unsqueeze_77);  mul_28 = unsqueeze_77 = None
        unsqueeze_78: "f16[64, 1]" = torch.ops.aten.unsqueeze.default(arg50_1, -1);  arg50_1 = None
        unsqueeze_79: "f16[64, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_78, -1);  unsqueeze_78 = None
        add_21: "f32[32, 64, 56, 56]" = torch.ops.aten.add.Tensor(mul_29, unsqueeze_79);  mul_29 = unsqueeze_79 = None
        convert_element_type_29: "f16[32, 64, 56, 56]" = torch.ops.prims.convert_element_type.default(add_21, torch.float16);  add_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_8: "f16[32, 64, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_29);  convert_element_type_29 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_10: "f16[32, 256, 56, 56]" = torch.ops.aten.convolution.default(relu_8, arg51_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_8 = arg51_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_30: "f32[256]" = torch.ops.prims.convert_element_type.default(arg52_1, torch.float32);  arg52_1 = None
        unsqueeze_80: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_30, -1);  convert_element_type_30 = None
        unsqueeze_81: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_80, -1);  unsqueeze_80 = None
        sub_10: "f32[32, 256, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_10, unsqueeze_81);  convolution_10 = unsqueeze_81 = None
        convert_element_type_31: "f32[256]" = torch.ops.prims.convert_element_type.default(arg53_1, torch.float32);  arg53_1 = None
        add_22: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_31, 1e-05);  convert_element_type_31 = None
        sqrt_10: "f32[256]" = torch.ops.aten.sqrt.default(add_22);  add_22 = None
        reciprocal_10: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_10);  sqrt_10 = None
        mul_30: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_10, 1);  reciprocal_10 = None
        unsqueeze_82: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_30, -1);  mul_30 = None
        unsqueeze_83: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_82, -1);  unsqueeze_82 = None
        mul_31: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(sub_10, unsqueeze_83);  sub_10 = unsqueeze_83 = None
        unsqueeze_84: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg54_1, -1);  arg54_1 = None
        unsqueeze_85: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_84, -1);  unsqueeze_84 = None
        mul_32: "f32[32, 256, 56, 56]" = torch.ops.aten.mul.Tensor(mul_31, unsqueeze_85);  mul_31 = unsqueeze_85 = None
        unsqueeze_86: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg55_1, -1);  arg55_1 = None
        unsqueeze_87: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_86, -1);  unsqueeze_86 = None
        add_23: "f32[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(mul_32, unsqueeze_87);  mul_32 = unsqueeze_87 = None
        convert_element_type_32: "f16[32, 256, 56, 56]" = torch.ops.prims.convert_element_type.default(add_23, torch.float16);  add_23 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_24: "f16[32, 256, 56, 56]" = torch.ops.aten.add.Tensor(convert_element_type_32, relu_6);  convert_element_type_32 = relu_6 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_9: "f16[32, 256, 56, 56]" = torch.ops.aten.relu.default(add_24);  add_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_11: "f16[32, 128, 56, 56]" = torch.ops.aten.convolution.default(relu_9, arg56_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg56_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_33: "f32[128]" = torch.ops.prims.convert_element_type.default(arg57_1, torch.float32);  arg57_1 = None
        unsqueeze_88: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_33, -1);  convert_element_type_33 = None
        unsqueeze_89: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_88, -1);  unsqueeze_88 = None
        sub_11: "f32[32, 128, 56, 56]" = torch.ops.aten.sub.Tensor(convolution_11, unsqueeze_89);  convolution_11 = unsqueeze_89 = None
        convert_element_type_34: "f32[128]" = torch.ops.prims.convert_element_type.default(arg58_1, torch.float32);  arg58_1 = None
        add_25: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_34, 1e-05);  convert_element_type_34 = None
        sqrt_11: "f32[128]" = torch.ops.aten.sqrt.default(add_25);  add_25 = None
        reciprocal_11: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_11);  sqrt_11 = None
        mul_33: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_11, 1);  reciprocal_11 = None
        unsqueeze_90: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_33, -1);  mul_33 = None
        unsqueeze_91: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_90, -1);  unsqueeze_90 = None
        mul_34: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(sub_11, unsqueeze_91);  sub_11 = unsqueeze_91 = None
        unsqueeze_92: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg59_1, -1);  arg59_1 = None
        unsqueeze_93: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_92, -1);  unsqueeze_92 = None
        mul_35: "f32[32, 128, 56, 56]" = torch.ops.aten.mul.Tensor(mul_34, unsqueeze_93);  mul_34 = unsqueeze_93 = None
        unsqueeze_94: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg60_1, -1);  arg60_1 = None
        unsqueeze_95: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_94, -1);  unsqueeze_94 = None
        add_26: "f32[32, 128, 56, 56]" = torch.ops.aten.add.Tensor(mul_35, unsqueeze_95);  mul_35 = unsqueeze_95 = None
        convert_element_type_35: "f16[32, 128, 56, 56]" = torch.ops.prims.convert_element_type.default(add_26, torch.float16);  add_26 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_10: "f16[32, 128, 56, 56]" = torch.ops.aten.relu.default(convert_element_type_35);  convert_element_type_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_12: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_10, arg61_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_10 = arg61_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_36: "f32[128]" = torch.ops.prims.convert_element_type.default(arg62_1, torch.float32);  arg62_1 = None
        unsqueeze_96: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_36, -1);  convert_element_type_36 = None
        unsqueeze_97: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_96, -1);  unsqueeze_96 = None
        sub_12: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_12, unsqueeze_97);  convolution_12 = unsqueeze_97 = None
        convert_element_type_37: "f32[128]" = torch.ops.prims.convert_element_type.default(arg63_1, torch.float32);  arg63_1 = None
        add_27: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_37, 1e-05);  convert_element_type_37 = None
        sqrt_12: "f32[128]" = torch.ops.aten.sqrt.default(add_27);  add_27 = None
        reciprocal_12: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_12);  sqrt_12 = None
        mul_36: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_12, 1);  reciprocal_12 = None
        unsqueeze_98: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_36, -1);  mul_36 = None
        unsqueeze_99: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_98, -1);  unsqueeze_98 = None
        mul_37: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_12, unsqueeze_99);  sub_12 = unsqueeze_99 = None
        unsqueeze_100: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg64_1, -1);  arg64_1 = None
        unsqueeze_101: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_100, -1);  unsqueeze_100 = None
        mul_38: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_37, unsqueeze_101);  mul_37 = unsqueeze_101 = None
        unsqueeze_102: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg65_1, -1);  arg65_1 = None
        unsqueeze_103: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_102, -1);  unsqueeze_102 = None
        add_28: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_38, unsqueeze_103);  mul_38 = unsqueeze_103 = None
        convert_element_type_38: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_28, torch.float16);  add_28 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_11: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_38);  convert_element_type_38 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_13: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_11, arg66_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_11 = arg66_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_39: "f32[512]" = torch.ops.prims.convert_element_type.default(arg67_1, torch.float32);  arg67_1 = None
        unsqueeze_104: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_39, -1);  convert_element_type_39 = None
        unsqueeze_105: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_104, -1);  unsqueeze_104 = None
        sub_13: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_13, unsqueeze_105);  convolution_13 = unsqueeze_105 = None
        convert_element_type_40: "f32[512]" = torch.ops.prims.convert_element_type.default(arg68_1, torch.float32);  arg68_1 = None
        add_29: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_40, 1e-05);  convert_element_type_40 = None
        sqrt_13: "f32[512]" = torch.ops.aten.sqrt.default(add_29);  add_29 = None
        reciprocal_13: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_13);  sqrt_13 = None
        mul_39: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_13, 1);  reciprocal_13 = None
        unsqueeze_106: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_39, -1);  mul_39 = None
        unsqueeze_107: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_106, -1);  unsqueeze_106 = None
        mul_40: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_13, unsqueeze_107);  sub_13 = unsqueeze_107 = None
        unsqueeze_108: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg69_1, -1);  arg69_1 = None
        unsqueeze_109: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_108, -1);  unsqueeze_108 = None
        mul_41: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_40, unsqueeze_109);  mul_40 = unsqueeze_109 = None
        unsqueeze_110: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg70_1, -1);  arg70_1 = None
        unsqueeze_111: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_110, -1);  unsqueeze_110 = None
        add_30: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_41, unsqueeze_111);  mul_41 = unsqueeze_111 = None
        convert_element_type_41: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_30, torch.float16);  add_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_14: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_9, arg71_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_9 = arg71_1 = None
        convert_element_type_42: "f32[512]" = torch.ops.prims.convert_element_type.default(arg72_1, torch.float32);  arg72_1 = None
        unsqueeze_112: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_42, -1);  convert_element_type_42 = None
        unsqueeze_113: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_112, -1);  unsqueeze_112 = None
        sub_14: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_14, unsqueeze_113);  convolution_14 = unsqueeze_113 = None
        convert_element_type_43: "f32[512]" = torch.ops.prims.convert_element_type.default(arg73_1, torch.float32);  arg73_1 = None
        add_31: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_43, 1e-05);  convert_element_type_43 = None
        sqrt_14: "f32[512]" = torch.ops.aten.sqrt.default(add_31);  add_31 = None
        reciprocal_14: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_14);  sqrt_14 = None
        mul_42: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_14, 1);  reciprocal_14 = None
        unsqueeze_114: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_42, -1);  mul_42 = None
        unsqueeze_115: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_114, -1);  unsqueeze_114 = None
        mul_43: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_14, unsqueeze_115);  sub_14 = unsqueeze_115 = None
        unsqueeze_116: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg74_1, -1);  arg74_1 = None
        unsqueeze_117: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_116, -1);  unsqueeze_116 = None
        mul_44: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_43, unsqueeze_117);  mul_43 = unsqueeze_117 = None
        unsqueeze_118: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg75_1, -1);  arg75_1 = None
        unsqueeze_119: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_118, -1);  unsqueeze_118 = None
        add_32: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_44, unsqueeze_119);  mul_44 = unsqueeze_119 = None
        convert_element_type_44: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_32, torch.float16);  add_32 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_33: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_41, convert_element_type_44);  convert_element_type_41 = convert_element_type_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_12: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_33);  add_33 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_15: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_12, arg76_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg76_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_45: "f32[128]" = torch.ops.prims.convert_element_type.default(arg77_1, torch.float32);  arg77_1 = None
        unsqueeze_120: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_45, -1);  convert_element_type_45 = None
        unsqueeze_121: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_120, -1);  unsqueeze_120 = None
        sub_15: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_15, unsqueeze_121);  convolution_15 = unsqueeze_121 = None
        convert_element_type_46: "f32[128]" = torch.ops.prims.convert_element_type.default(arg78_1, torch.float32);  arg78_1 = None
        add_34: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_46, 1e-05);  convert_element_type_46 = None
        sqrt_15: "f32[128]" = torch.ops.aten.sqrt.default(add_34);  add_34 = None
        reciprocal_15: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_15);  sqrt_15 = None
        mul_45: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_15, 1);  reciprocal_15 = None
        unsqueeze_122: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_45, -1);  mul_45 = None
        unsqueeze_123: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_122, -1);  unsqueeze_122 = None
        mul_46: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_15, unsqueeze_123);  sub_15 = unsqueeze_123 = None
        unsqueeze_124: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg79_1, -1);  arg79_1 = None
        unsqueeze_125: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_124, -1);  unsqueeze_124 = None
        mul_47: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_46, unsqueeze_125);  mul_46 = unsqueeze_125 = None
        unsqueeze_126: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg80_1, -1);  arg80_1 = None
        unsqueeze_127: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_126, -1);  unsqueeze_126 = None
        add_35: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_47, unsqueeze_127);  mul_47 = unsqueeze_127 = None
        convert_element_type_47: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_35, torch.float16);  add_35 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_13: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_47);  convert_element_type_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_16: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_13, arg81_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_13 = arg81_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_48: "f32[128]" = torch.ops.prims.convert_element_type.default(arg82_1, torch.float32);  arg82_1 = None
        unsqueeze_128: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_48, -1);  convert_element_type_48 = None
        unsqueeze_129: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_128, -1);  unsqueeze_128 = None
        sub_16: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_16, unsqueeze_129);  convolution_16 = unsqueeze_129 = None
        convert_element_type_49: "f32[128]" = torch.ops.prims.convert_element_type.default(arg83_1, torch.float32);  arg83_1 = None
        add_36: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_49, 1e-05);  convert_element_type_49 = None
        sqrt_16: "f32[128]" = torch.ops.aten.sqrt.default(add_36);  add_36 = None
        reciprocal_16: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_16);  sqrt_16 = None
        mul_48: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_16, 1);  reciprocal_16 = None
        unsqueeze_130: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_48, -1);  mul_48 = None
        unsqueeze_131: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_130, -1);  unsqueeze_130 = None
        mul_49: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_16, unsqueeze_131);  sub_16 = unsqueeze_131 = None
        unsqueeze_132: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg84_1, -1);  arg84_1 = None
        unsqueeze_133: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_132, -1);  unsqueeze_132 = None
        mul_50: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_49, unsqueeze_133);  mul_49 = unsqueeze_133 = None
        unsqueeze_134: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg85_1, -1);  arg85_1 = None
        unsqueeze_135: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_134, -1);  unsqueeze_134 = None
        add_37: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_50, unsqueeze_135);  mul_50 = unsqueeze_135 = None
        convert_element_type_50: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_37, torch.float16);  add_37 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_14: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_50);  convert_element_type_50 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_17: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_14, arg86_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_14 = arg86_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_51: "f32[512]" = torch.ops.prims.convert_element_type.default(arg87_1, torch.float32);  arg87_1 = None
        unsqueeze_136: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_51, -1);  convert_element_type_51 = None
        unsqueeze_137: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_136, -1);  unsqueeze_136 = None
        sub_17: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_17, unsqueeze_137);  convolution_17 = unsqueeze_137 = None
        convert_element_type_52: "f32[512]" = torch.ops.prims.convert_element_type.default(arg88_1, torch.float32);  arg88_1 = None
        add_38: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_52, 1e-05);  convert_element_type_52 = None
        sqrt_17: "f32[512]" = torch.ops.aten.sqrt.default(add_38);  add_38 = None
        reciprocal_17: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_17);  sqrt_17 = None
        mul_51: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_17, 1);  reciprocal_17 = None
        unsqueeze_138: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_51, -1);  mul_51 = None
        unsqueeze_139: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_138, -1);  unsqueeze_138 = None
        mul_52: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_17, unsqueeze_139);  sub_17 = unsqueeze_139 = None
        unsqueeze_140: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg89_1, -1);  arg89_1 = None
        unsqueeze_141: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_140, -1);  unsqueeze_140 = None
        mul_53: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_52, unsqueeze_141);  mul_52 = unsqueeze_141 = None
        unsqueeze_142: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg90_1, -1);  arg90_1 = None
        unsqueeze_143: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_142, -1);  unsqueeze_142 = None
        add_39: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_53, unsqueeze_143);  mul_53 = unsqueeze_143 = None
        convert_element_type_53: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_39, torch.float16);  add_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_40: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_53, relu_12);  convert_element_type_53 = relu_12 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_15: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_40);  add_40 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_18: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_15, arg91_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg91_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_54: "f32[128]" = torch.ops.prims.convert_element_type.default(arg92_1, torch.float32);  arg92_1 = None
        unsqueeze_144: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_54, -1);  convert_element_type_54 = None
        unsqueeze_145: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_144, -1);  unsqueeze_144 = None
        sub_18: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_18, unsqueeze_145);  convolution_18 = unsqueeze_145 = None
        convert_element_type_55: "f32[128]" = torch.ops.prims.convert_element_type.default(arg93_1, torch.float32);  arg93_1 = None
        add_41: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_55, 1e-05);  convert_element_type_55 = None
        sqrt_18: "f32[128]" = torch.ops.aten.sqrt.default(add_41);  add_41 = None
        reciprocal_18: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_18);  sqrt_18 = None
        mul_54: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_18, 1);  reciprocal_18 = None
        unsqueeze_146: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_54, -1);  mul_54 = None
        unsqueeze_147: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_146, -1);  unsqueeze_146 = None
        mul_55: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_18, unsqueeze_147);  sub_18 = unsqueeze_147 = None
        unsqueeze_148: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg94_1, -1);  arg94_1 = None
        unsqueeze_149: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_148, -1);  unsqueeze_148 = None
        mul_56: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_55, unsqueeze_149);  mul_55 = unsqueeze_149 = None
        unsqueeze_150: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg95_1, -1);  arg95_1 = None
        unsqueeze_151: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_150, -1);  unsqueeze_150 = None
        add_42: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_56, unsqueeze_151);  mul_56 = unsqueeze_151 = None
        convert_element_type_56: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_42, torch.float16);  add_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_16: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_56);  convert_element_type_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_19: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_16, arg96_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_16 = arg96_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_57: "f32[128]" = torch.ops.prims.convert_element_type.default(arg97_1, torch.float32);  arg97_1 = None
        unsqueeze_152: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_57, -1);  convert_element_type_57 = None
        unsqueeze_153: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_152, -1);  unsqueeze_152 = None
        sub_19: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_19, unsqueeze_153);  convolution_19 = unsqueeze_153 = None
        convert_element_type_58: "f32[128]" = torch.ops.prims.convert_element_type.default(arg98_1, torch.float32);  arg98_1 = None
        add_43: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_58, 1e-05);  convert_element_type_58 = None
        sqrt_19: "f32[128]" = torch.ops.aten.sqrt.default(add_43);  add_43 = None
        reciprocal_19: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_19);  sqrt_19 = None
        mul_57: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_19, 1);  reciprocal_19 = None
        unsqueeze_154: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_57, -1);  mul_57 = None
        unsqueeze_155: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_154, -1);  unsqueeze_154 = None
        mul_58: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_19, unsqueeze_155);  sub_19 = unsqueeze_155 = None
        unsqueeze_156: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg99_1, -1);  arg99_1 = None
        unsqueeze_157: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_156, -1);  unsqueeze_156 = None
        mul_59: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_58, unsqueeze_157);  mul_58 = unsqueeze_157 = None
        unsqueeze_158: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg100_1, -1);  arg100_1 = None
        unsqueeze_159: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_158, -1);  unsqueeze_158 = None
        add_44: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_59, unsqueeze_159);  mul_59 = unsqueeze_159 = None
        convert_element_type_59: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_44, torch.float16);  add_44 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_17: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_59);  convert_element_type_59 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_20: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_17, arg101_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_17 = arg101_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_60: "f32[512]" = torch.ops.prims.convert_element_type.default(arg102_1, torch.float32);  arg102_1 = None
        unsqueeze_160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_60, -1);  convert_element_type_60 = None
        unsqueeze_161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_160, -1);  unsqueeze_160 = None
        sub_20: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_20, unsqueeze_161);  convolution_20 = unsqueeze_161 = None
        convert_element_type_61: "f32[512]" = torch.ops.prims.convert_element_type.default(arg103_1, torch.float32);  arg103_1 = None
        add_45: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_61, 1e-05);  convert_element_type_61 = None
        sqrt_20: "f32[512]" = torch.ops.aten.sqrt.default(add_45);  add_45 = None
        reciprocal_20: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_20);  sqrt_20 = None
        mul_60: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_20, 1);  reciprocal_20 = None
        unsqueeze_162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_60, -1);  mul_60 = None
        unsqueeze_163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_162, -1);  unsqueeze_162 = None
        mul_61: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_20, unsqueeze_163);  sub_20 = unsqueeze_163 = None
        unsqueeze_164: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg104_1, -1);  arg104_1 = None
        unsqueeze_165: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_164, -1);  unsqueeze_164 = None
        mul_62: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_61, unsqueeze_165);  mul_61 = unsqueeze_165 = None
        unsqueeze_166: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg105_1, -1);  arg105_1 = None
        unsqueeze_167: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_166, -1);  unsqueeze_166 = None
        add_46: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_62, unsqueeze_167);  mul_62 = unsqueeze_167 = None
        convert_element_type_62: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_46, torch.float16);  add_46 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_47: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_62, relu_15);  convert_element_type_62 = relu_15 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_18: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_47);  add_47 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_21: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_18, arg106_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg106_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_63: "f32[128]" = torch.ops.prims.convert_element_type.default(arg107_1, torch.float32);  arg107_1 = None
        unsqueeze_168: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_63, -1);  convert_element_type_63 = None
        unsqueeze_169: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_168, -1);  unsqueeze_168 = None
        sub_21: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_21, unsqueeze_169);  convolution_21 = unsqueeze_169 = None
        convert_element_type_64: "f32[128]" = torch.ops.prims.convert_element_type.default(arg108_1, torch.float32);  arg108_1 = None
        add_48: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_64, 1e-05);  convert_element_type_64 = None
        sqrt_21: "f32[128]" = torch.ops.aten.sqrt.default(add_48);  add_48 = None
        reciprocal_21: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_21);  sqrt_21 = None
        mul_63: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_21, 1);  reciprocal_21 = None
        unsqueeze_170: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_63, -1);  mul_63 = None
        unsqueeze_171: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_170, -1);  unsqueeze_170 = None
        mul_64: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_21, unsqueeze_171);  sub_21 = unsqueeze_171 = None
        unsqueeze_172: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg109_1, -1);  arg109_1 = None
        unsqueeze_173: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_172, -1);  unsqueeze_172 = None
        mul_65: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_64, unsqueeze_173);  mul_64 = unsqueeze_173 = None
        unsqueeze_174: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg110_1, -1);  arg110_1 = None
        unsqueeze_175: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_174, -1);  unsqueeze_174 = None
        add_49: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_65, unsqueeze_175);  mul_65 = unsqueeze_175 = None
        convert_element_type_65: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_49, torch.float16);  add_49 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_19: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_65);  convert_element_type_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_22: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_19, arg111_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_19 = arg111_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_66: "f32[128]" = torch.ops.prims.convert_element_type.default(arg112_1, torch.float32);  arg112_1 = None
        unsqueeze_176: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_66, -1);  convert_element_type_66 = None
        unsqueeze_177: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_176, -1);  unsqueeze_176 = None
        sub_22: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_22, unsqueeze_177);  convolution_22 = unsqueeze_177 = None
        convert_element_type_67: "f32[128]" = torch.ops.prims.convert_element_type.default(arg113_1, torch.float32);  arg113_1 = None
        add_50: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_67, 1e-05);  convert_element_type_67 = None
        sqrt_22: "f32[128]" = torch.ops.aten.sqrt.default(add_50);  add_50 = None
        reciprocal_22: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_22);  sqrt_22 = None
        mul_66: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_22, 1);  reciprocal_22 = None
        unsqueeze_178: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_66, -1);  mul_66 = None
        unsqueeze_179: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_178, -1);  unsqueeze_178 = None
        mul_67: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_22, unsqueeze_179);  sub_22 = unsqueeze_179 = None
        unsqueeze_180: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg114_1, -1);  arg114_1 = None
        unsqueeze_181: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_180, -1);  unsqueeze_180 = None
        mul_68: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_67, unsqueeze_181);  mul_67 = unsqueeze_181 = None
        unsqueeze_182: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg115_1, -1);  arg115_1 = None
        unsqueeze_183: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_182, -1);  unsqueeze_182 = None
        add_51: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_68, unsqueeze_183);  mul_68 = unsqueeze_183 = None
        convert_element_type_68: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_51, torch.float16);  add_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_20: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_68);  convert_element_type_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_23: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_20, arg116_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_20 = arg116_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_69: "f32[512]" = torch.ops.prims.convert_element_type.default(arg117_1, torch.float32);  arg117_1 = None
        unsqueeze_184: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_69, -1);  convert_element_type_69 = None
        unsqueeze_185: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_184, -1);  unsqueeze_184 = None
        sub_23: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_23, unsqueeze_185);  convolution_23 = unsqueeze_185 = None
        convert_element_type_70: "f32[512]" = torch.ops.prims.convert_element_type.default(arg118_1, torch.float32);  arg118_1 = None
        add_52: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_70, 1e-05);  convert_element_type_70 = None
        sqrt_23: "f32[512]" = torch.ops.aten.sqrt.default(add_52);  add_52 = None
        reciprocal_23: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_23);  sqrt_23 = None
        mul_69: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_23, 1);  reciprocal_23 = None
        unsqueeze_186: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_69, -1);  mul_69 = None
        unsqueeze_187: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_186, -1);  unsqueeze_186 = None
        mul_70: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_23, unsqueeze_187);  sub_23 = unsqueeze_187 = None
        unsqueeze_188: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg119_1, -1);  arg119_1 = None
        unsqueeze_189: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_188, -1);  unsqueeze_188 = None
        mul_71: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_70, unsqueeze_189);  mul_70 = unsqueeze_189 = None
        unsqueeze_190: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg120_1, -1);  arg120_1 = None
        unsqueeze_191: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_190, -1);  unsqueeze_190 = None
        add_53: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_71, unsqueeze_191);  mul_71 = unsqueeze_191 = None
        convert_element_type_71: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_53, torch.float16);  add_53 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_54: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_71, relu_18);  convert_element_type_71 = relu_18 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_21: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_54);  add_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_24: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_21, arg121_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg121_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_72: "f32[128]" = torch.ops.prims.convert_element_type.default(arg122_1, torch.float32);  arg122_1 = None
        unsqueeze_192: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_72, -1);  convert_element_type_72 = None
        unsqueeze_193: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_192, -1);  unsqueeze_192 = None
        sub_24: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_24, unsqueeze_193);  convolution_24 = unsqueeze_193 = None
        convert_element_type_73: "f32[128]" = torch.ops.prims.convert_element_type.default(arg123_1, torch.float32);  arg123_1 = None
        add_55: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_73, 1e-05);  convert_element_type_73 = None
        sqrt_24: "f32[128]" = torch.ops.aten.sqrt.default(add_55);  add_55 = None
        reciprocal_24: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_24);  sqrt_24 = None
        mul_72: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_24, 1);  reciprocal_24 = None
        unsqueeze_194: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_72, -1);  mul_72 = None
        unsqueeze_195: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_194, -1);  unsqueeze_194 = None
        mul_73: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_24, unsqueeze_195);  sub_24 = unsqueeze_195 = None
        unsqueeze_196: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg124_1, -1);  arg124_1 = None
        unsqueeze_197: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_196, -1);  unsqueeze_196 = None
        mul_74: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_73, unsqueeze_197);  mul_73 = unsqueeze_197 = None
        unsqueeze_198: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg125_1, -1);  arg125_1 = None
        unsqueeze_199: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_198, -1);  unsqueeze_198 = None
        add_56: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_74, unsqueeze_199);  mul_74 = unsqueeze_199 = None
        convert_element_type_74: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_56, torch.float16);  add_56 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_22: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_74);  convert_element_type_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_25: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_22, arg126_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_22 = arg126_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_75: "f32[128]" = torch.ops.prims.convert_element_type.default(arg127_1, torch.float32);  arg127_1 = None
        unsqueeze_200: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_75, -1);  convert_element_type_75 = None
        unsqueeze_201: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_200, -1);  unsqueeze_200 = None
        sub_25: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_25, unsqueeze_201);  convolution_25 = unsqueeze_201 = None
        convert_element_type_76: "f32[128]" = torch.ops.prims.convert_element_type.default(arg128_1, torch.float32);  arg128_1 = None
        add_57: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_76, 1e-05);  convert_element_type_76 = None
        sqrt_25: "f32[128]" = torch.ops.aten.sqrt.default(add_57);  add_57 = None
        reciprocal_25: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_25);  sqrt_25 = None
        mul_75: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_25, 1);  reciprocal_25 = None
        unsqueeze_202: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_75, -1);  mul_75 = None
        unsqueeze_203: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_202, -1);  unsqueeze_202 = None
        mul_76: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_25, unsqueeze_203);  sub_25 = unsqueeze_203 = None
        unsqueeze_204: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg129_1, -1);  arg129_1 = None
        unsqueeze_205: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_204, -1);  unsqueeze_204 = None
        mul_77: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_76, unsqueeze_205);  mul_76 = unsqueeze_205 = None
        unsqueeze_206: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg130_1, -1);  arg130_1 = None
        unsqueeze_207: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_206, -1);  unsqueeze_206 = None
        add_58: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_77, unsqueeze_207);  mul_77 = unsqueeze_207 = None
        convert_element_type_77: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_58, torch.float16);  add_58 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_23: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_77);  convert_element_type_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_26: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_23, arg131_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_23 = arg131_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_78: "f32[512]" = torch.ops.prims.convert_element_type.default(arg132_1, torch.float32);  arg132_1 = None
        unsqueeze_208: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_78, -1);  convert_element_type_78 = None
        unsqueeze_209: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_208, -1);  unsqueeze_208 = None
        sub_26: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_26, unsqueeze_209);  convolution_26 = unsqueeze_209 = None
        convert_element_type_79: "f32[512]" = torch.ops.prims.convert_element_type.default(arg133_1, torch.float32);  arg133_1 = None
        add_59: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_79, 1e-05);  convert_element_type_79 = None
        sqrt_26: "f32[512]" = torch.ops.aten.sqrt.default(add_59);  add_59 = None
        reciprocal_26: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_26);  sqrt_26 = None
        mul_78: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_26, 1);  reciprocal_26 = None
        unsqueeze_210: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_78, -1);  mul_78 = None
        unsqueeze_211: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_210, -1);  unsqueeze_210 = None
        mul_79: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_26, unsqueeze_211);  sub_26 = unsqueeze_211 = None
        unsqueeze_212: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg134_1, -1);  arg134_1 = None
        unsqueeze_213: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_212, -1);  unsqueeze_212 = None
        mul_80: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_79, unsqueeze_213);  mul_79 = unsqueeze_213 = None
        unsqueeze_214: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg135_1, -1);  arg135_1 = None
        unsqueeze_215: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_214, -1);  unsqueeze_214 = None
        add_60: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_80, unsqueeze_215);  mul_80 = unsqueeze_215 = None
        convert_element_type_80: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_60, torch.float16);  add_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_61: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_80, relu_21);  convert_element_type_80 = relu_21 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_24: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_61);  add_61 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_27: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_24, arg136_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg136_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_81: "f32[128]" = torch.ops.prims.convert_element_type.default(arg137_1, torch.float32);  arg137_1 = None
        unsqueeze_216: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_81, -1);  convert_element_type_81 = None
        unsqueeze_217: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_216, -1);  unsqueeze_216 = None
        sub_27: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_27, unsqueeze_217);  convolution_27 = unsqueeze_217 = None
        convert_element_type_82: "f32[128]" = torch.ops.prims.convert_element_type.default(arg138_1, torch.float32);  arg138_1 = None
        add_62: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_82, 1e-05);  convert_element_type_82 = None
        sqrt_27: "f32[128]" = torch.ops.aten.sqrt.default(add_62);  add_62 = None
        reciprocal_27: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_27);  sqrt_27 = None
        mul_81: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_27, 1);  reciprocal_27 = None
        unsqueeze_218: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_81, -1);  mul_81 = None
        unsqueeze_219: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_218, -1);  unsqueeze_218 = None
        mul_82: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_27, unsqueeze_219);  sub_27 = unsqueeze_219 = None
        unsqueeze_220: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg139_1, -1);  arg139_1 = None
        unsqueeze_221: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_220, -1);  unsqueeze_220 = None
        mul_83: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_82, unsqueeze_221);  mul_82 = unsqueeze_221 = None
        unsqueeze_222: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg140_1, -1);  arg140_1 = None
        unsqueeze_223: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_222, -1);  unsqueeze_222 = None
        add_63: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_83, unsqueeze_223);  mul_83 = unsqueeze_223 = None
        convert_element_type_83: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_63, torch.float16);  add_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_25: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_83);  convert_element_type_83 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_28: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_25, arg141_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_25 = arg141_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_84: "f32[128]" = torch.ops.prims.convert_element_type.default(arg142_1, torch.float32);  arg142_1 = None
        unsqueeze_224: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_84, -1);  convert_element_type_84 = None
        unsqueeze_225: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_224, -1);  unsqueeze_224 = None
        sub_28: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_28, unsqueeze_225);  convolution_28 = unsqueeze_225 = None
        convert_element_type_85: "f32[128]" = torch.ops.prims.convert_element_type.default(arg143_1, torch.float32);  arg143_1 = None
        add_64: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_85, 1e-05);  convert_element_type_85 = None
        sqrt_28: "f32[128]" = torch.ops.aten.sqrt.default(add_64);  add_64 = None
        reciprocal_28: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_28);  sqrt_28 = None
        mul_84: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_28, 1);  reciprocal_28 = None
        unsqueeze_226: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_84, -1);  mul_84 = None
        unsqueeze_227: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_226, -1);  unsqueeze_226 = None
        mul_85: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_28, unsqueeze_227);  sub_28 = unsqueeze_227 = None
        unsqueeze_228: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg144_1, -1);  arg144_1 = None
        unsqueeze_229: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_228, -1);  unsqueeze_228 = None
        mul_86: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_85, unsqueeze_229);  mul_85 = unsqueeze_229 = None
        unsqueeze_230: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg145_1, -1);  arg145_1 = None
        unsqueeze_231: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_230, -1);  unsqueeze_230 = None
        add_65: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_86, unsqueeze_231);  mul_86 = unsqueeze_231 = None
        convert_element_type_86: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_65, torch.float16);  add_65 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_26: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_86);  convert_element_type_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_29: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_26, arg146_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_26 = arg146_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_87: "f32[512]" = torch.ops.prims.convert_element_type.default(arg147_1, torch.float32);  arg147_1 = None
        unsqueeze_232: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_87, -1);  convert_element_type_87 = None
        unsqueeze_233: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_232, -1);  unsqueeze_232 = None
        sub_29: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_29, unsqueeze_233);  convolution_29 = unsqueeze_233 = None
        convert_element_type_88: "f32[512]" = torch.ops.prims.convert_element_type.default(arg148_1, torch.float32);  arg148_1 = None
        add_66: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_88, 1e-05);  convert_element_type_88 = None
        sqrt_29: "f32[512]" = torch.ops.aten.sqrt.default(add_66);  add_66 = None
        reciprocal_29: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_29);  sqrt_29 = None
        mul_87: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_29, 1);  reciprocal_29 = None
        unsqueeze_234: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_87, -1);  mul_87 = None
        unsqueeze_235: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_234, -1);  unsqueeze_234 = None
        mul_88: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_29, unsqueeze_235);  sub_29 = unsqueeze_235 = None
        unsqueeze_236: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg149_1, -1);  arg149_1 = None
        unsqueeze_237: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_236, -1);  unsqueeze_236 = None
        mul_89: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_88, unsqueeze_237);  mul_88 = unsqueeze_237 = None
        unsqueeze_238: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg150_1, -1);  arg150_1 = None
        unsqueeze_239: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_238, -1);  unsqueeze_238 = None
        add_67: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_89, unsqueeze_239);  mul_89 = unsqueeze_239 = None
        convert_element_type_89: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_67, torch.float16);  add_67 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_68: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_89, relu_24);  convert_element_type_89 = relu_24 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_27: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_68);  add_68 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_30: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_27, arg151_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg151_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_90: "f32[128]" = torch.ops.prims.convert_element_type.default(arg152_1, torch.float32);  arg152_1 = None
        unsqueeze_240: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_90, -1);  convert_element_type_90 = None
        unsqueeze_241: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_240, -1);  unsqueeze_240 = None
        sub_30: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_30, unsqueeze_241);  convolution_30 = unsqueeze_241 = None
        convert_element_type_91: "f32[128]" = torch.ops.prims.convert_element_type.default(arg153_1, torch.float32);  arg153_1 = None
        add_69: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_91, 1e-05);  convert_element_type_91 = None
        sqrt_30: "f32[128]" = torch.ops.aten.sqrt.default(add_69);  add_69 = None
        reciprocal_30: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_30);  sqrt_30 = None
        mul_90: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_30, 1);  reciprocal_30 = None
        unsqueeze_242: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_90, -1);  mul_90 = None
        unsqueeze_243: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_242, -1);  unsqueeze_242 = None
        mul_91: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_30, unsqueeze_243);  sub_30 = unsqueeze_243 = None
        unsqueeze_244: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg154_1, -1);  arg154_1 = None
        unsqueeze_245: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_244, -1);  unsqueeze_244 = None
        mul_92: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_91, unsqueeze_245);  mul_91 = unsqueeze_245 = None
        unsqueeze_246: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg155_1, -1);  arg155_1 = None
        unsqueeze_247: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_246, -1);  unsqueeze_246 = None
        add_70: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_92, unsqueeze_247);  mul_92 = unsqueeze_247 = None
        convert_element_type_92: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_70, torch.float16);  add_70 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_28: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_92);  convert_element_type_92 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_31: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_28, arg156_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_28 = arg156_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_93: "f32[128]" = torch.ops.prims.convert_element_type.default(arg157_1, torch.float32);  arg157_1 = None
        unsqueeze_248: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_93, -1);  convert_element_type_93 = None
        unsqueeze_249: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_248, -1);  unsqueeze_248 = None
        sub_31: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_31, unsqueeze_249);  convolution_31 = unsqueeze_249 = None
        convert_element_type_94: "f32[128]" = torch.ops.prims.convert_element_type.default(arg158_1, torch.float32);  arg158_1 = None
        add_71: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_94, 1e-05);  convert_element_type_94 = None
        sqrt_31: "f32[128]" = torch.ops.aten.sqrt.default(add_71);  add_71 = None
        reciprocal_31: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_31);  sqrt_31 = None
        mul_93: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_31, 1);  reciprocal_31 = None
        unsqueeze_250: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_93, -1);  mul_93 = None
        unsqueeze_251: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_250, -1);  unsqueeze_250 = None
        mul_94: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_31, unsqueeze_251);  sub_31 = unsqueeze_251 = None
        unsqueeze_252: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg159_1, -1);  arg159_1 = None
        unsqueeze_253: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_252, -1);  unsqueeze_252 = None
        mul_95: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_94, unsqueeze_253);  mul_94 = unsqueeze_253 = None
        unsqueeze_254: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg160_1, -1);  arg160_1 = None
        unsqueeze_255: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_254, -1);  unsqueeze_254 = None
        add_72: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_95, unsqueeze_255);  mul_95 = unsqueeze_255 = None
        convert_element_type_95: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_72, torch.float16);  add_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_29: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_95);  convert_element_type_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_32: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_29, arg161_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_29 = arg161_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_96: "f32[512]" = torch.ops.prims.convert_element_type.default(arg162_1, torch.float32);  arg162_1 = None
        unsqueeze_256: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_96, -1);  convert_element_type_96 = None
        unsqueeze_257: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_256, -1);  unsqueeze_256 = None
        sub_32: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_32, unsqueeze_257);  convolution_32 = unsqueeze_257 = None
        convert_element_type_97: "f32[512]" = torch.ops.prims.convert_element_type.default(arg163_1, torch.float32);  arg163_1 = None
        add_73: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_97, 1e-05);  convert_element_type_97 = None
        sqrt_32: "f32[512]" = torch.ops.aten.sqrt.default(add_73);  add_73 = None
        reciprocal_32: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_32);  sqrt_32 = None
        mul_96: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_32, 1);  reciprocal_32 = None
        unsqueeze_258: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_96, -1);  mul_96 = None
        unsqueeze_259: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_258, -1);  unsqueeze_258 = None
        mul_97: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_32, unsqueeze_259);  sub_32 = unsqueeze_259 = None
        unsqueeze_260: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg164_1, -1);  arg164_1 = None
        unsqueeze_261: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_260, -1);  unsqueeze_260 = None
        mul_98: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_97, unsqueeze_261);  mul_97 = unsqueeze_261 = None
        unsqueeze_262: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg165_1, -1);  arg165_1 = None
        unsqueeze_263: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_262, -1);  unsqueeze_262 = None
        add_74: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_98, unsqueeze_263);  mul_98 = unsqueeze_263 = None
        convert_element_type_98: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_74, torch.float16);  add_74 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_75: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_98, relu_27);  convert_element_type_98 = relu_27 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_30: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_75);  add_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_33: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_30, arg166_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg166_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_99: "f32[128]" = torch.ops.prims.convert_element_type.default(arg167_1, torch.float32);  arg167_1 = None
        unsqueeze_264: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_99, -1);  convert_element_type_99 = None
        unsqueeze_265: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_264, -1);  unsqueeze_264 = None
        sub_33: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_33, unsqueeze_265);  convolution_33 = unsqueeze_265 = None
        convert_element_type_100: "f32[128]" = torch.ops.prims.convert_element_type.default(arg168_1, torch.float32);  arg168_1 = None
        add_76: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_100, 1e-05);  convert_element_type_100 = None
        sqrt_33: "f32[128]" = torch.ops.aten.sqrt.default(add_76);  add_76 = None
        reciprocal_33: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_33);  sqrt_33 = None
        mul_99: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_33, 1);  reciprocal_33 = None
        unsqueeze_266: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_99, -1);  mul_99 = None
        unsqueeze_267: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_266, -1);  unsqueeze_266 = None
        mul_100: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_33, unsqueeze_267);  sub_33 = unsqueeze_267 = None
        unsqueeze_268: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg169_1, -1);  arg169_1 = None
        unsqueeze_269: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_268, -1);  unsqueeze_268 = None
        mul_101: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_100, unsqueeze_269);  mul_100 = unsqueeze_269 = None
        unsqueeze_270: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg170_1, -1);  arg170_1 = None
        unsqueeze_271: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_270, -1);  unsqueeze_270 = None
        add_77: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_101, unsqueeze_271);  mul_101 = unsqueeze_271 = None
        convert_element_type_101: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_77, torch.float16);  add_77 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_31: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_101);  convert_element_type_101 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_34: "f16[32, 128, 28, 28]" = torch.ops.aten.convolution.default(relu_31, arg171_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_31 = arg171_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_102: "f32[128]" = torch.ops.prims.convert_element_type.default(arg172_1, torch.float32);  arg172_1 = None
        unsqueeze_272: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_102, -1);  convert_element_type_102 = None
        unsqueeze_273: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_272, -1);  unsqueeze_272 = None
        sub_34: "f32[32, 128, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_34, unsqueeze_273);  convolution_34 = unsqueeze_273 = None
        convert_element_type_103: "f32[128]" = torch.ops.prims.convert_element_type.default(arg173_1, torch.float32);  arg173_1 = None
        add_78: "f32[128]" = torch.ops.aten.add.Tensor(convert_element_type_103, 1e-05);  convert_element_type_103 = None
        sqrt_34: "f32[128]" = torch.ops.aten.sqrt.default(add_78);  add_78 = None
        reciprocal_34: "f32[128]" = torch.ops.aten.reciprocal.default(sqrt_34);  sqrt_34 = None
        mul_102: "f32[128]" = torch.ops.aten.mul.Tensor(reciprocal_34, 1);  reciprocal_34 = None
        unsqueeze_274: "f32[128, 1]" = torch.ops.aten.unsqueeze.default(mul_102, -1);  mul_102 = None
        unsqueeze_275: "f32[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_274, -1);  unsqueeze_274 = None
        mul_103: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(sub_34, unsqueeze_275);  sub_34 = unsqueeze_275 = None
        unsqueeze_276: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg174_1, -1);  arg174_1 = None
        unsqueeze_277: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_276, -1);  unsqueeze_276 = None
        mul_104: "f32[32, 128, 28, 28]" = torch.ops.aten.mul.Tensor(mul_103, unsqueeze_277);  mul_103 = unsqueeze_277 = None
        unsqueeze_278: "f16[128, 1]" = torch.ops.aten.unsqueeze.default(arg175_1, -1);  arg175_1 = None
        unsqueeze_279: "f16[128, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_278, -1);  unsqueeze_278 = None
        add_79: "f32[32, 128, 28, 28]" = torch.ops.aten.add.Tensor(mul_104, unsqueeze_279);  mul_104 = unsqueeze_279 = None
        convert_element_type_104: "f16[32, 128, 28, 28]" = torch.ops.prims.convert_element_type.default(add_79, torch.float16);  add_79 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_32: "f16[32, 128, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_104);  convert_element_type_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_35: "f16[32, 512, 28, 28]" = torch.ops.aten.convolution.default(relu_32, arg176_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_32 = arg176_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_105: "f32[512]" = torch.ops.prims.convert_element_type.default(arg177_1, torch.float32);  arg177_1 = None
        unsqueeze_280: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_105, -1);  convert_element_type_105 = None
        unsqueeze_281: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_280, -1);  unsqueeze_280 = None
        sub_35: "f32[32, 512, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_35, unsqueeze_281);  convolution_35 = unsqueeze_281 = None
        convert_element_type_106: "f32[512]" = torch.ops.prims.convert_element_type.default(arg178_1, torch.float32);  arg178_1 = None
        add_80: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_106, 1e-05);  convert_element_type_106 = None
        sqrt_35: "f32[512]" = torch.ops.aten.sqrt.default(add_80);  add_80 = None
        reciprocal_35: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_35);  sqrt_35 = None
        mul_105: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_35, 1);  reciprocal_35 = None
        unsqueeze_282: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_105, -1);  mul_105 = None
        unsqueeze_283: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_282, -1);  unsqueeze_282 = None
        mul_106: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(sub_35, unsqueeze_283);  sub_35 = unsqueeze_283 = None
        unsqueeze_284: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg179_1, -1);  arg179_1 = None
        unsqueeze_285: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_284, -1);  unsqueeze_284 = None
        mul_107: "f32[32, 512, 28, 28]" = torch.ops.aten.mul.Tensor(mul_106, unsqueeze_285);  mul_106 = unsqueeze_285 = None
        unsqueeze_286: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg180_1, -1);  arg180_1 = None
        unsqueeze_287: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_286, -1);  unsqueeze_286 = None
        add_81: "f32[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(mul_107, unsqueeze_287);  mul_107 = unsqueeze_287 = None
        convert_element_type_107: "f16[32, 512, 28, 28]" = torch.ops.prims.convert_element_type.default(add_81, torch.float16);  add_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_82: "f16[32, 512, 28, 28]" = torch.ops.aten.add.Tensor(convert_element_type_107, relu_30);  convert_element_type_107 = relu_30 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_33: "f16[32, 512, 28, 28]" = torch.ops.aten.relu.default(add_82);  add_82 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_36: "f16[32, 256, 28, 28]" = torch.ops.aten.convolution.default(relu_33, arg181_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg181_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_108: "f32[256]" = torch.ops.prims.convert_element_type.default(arg182_1, torch.float32);  arg182_1 = None
        unsqueeze_288: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_108, -1);  convert_element_type_108 = None
        unsqueeze_289: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_288, -1);  unsqueeze_288 = None
        sub_36: "f32[32, 256, 28, 28]" = torch.ops.aten.sub.Tensor(convolution_36, unsqueeze_289);  convolution_36 = unsqueeze_289 = None
        convert_element_type_109: "f32[256]" = torch.ops.prims.convert_element_type.default(arg183_1, torch.float32);  arg183_1 = None
        add_83: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_109, 1e-05);  convert_element_type_109 = None
        sqrt_36: "f32[256]" = torch.ops.aten.sqrt.default(add_83);  add_83 = None
        reciprocal_36: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_36);  sqrt_36 = None
        mul_108: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_36, 1);  reciprocal_36 = None
        unsqueeze_290: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_108, -1);  mul_108 = None
        unsqueeze_291: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_290, -1);  unsqueeze_290 = None
        mul_109: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(sub_36, unsqueeze_291);  sub_36 = unsqueeze_291 = None
        unsqueeze_292: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg184_1, -1);  arg184_1 = None
        unsqueeze_293: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_292, -1);  unsqueeze_292 = None
        mul_110: "f32[32, 256, 28, 28]" = torch.ops.aten.mul.Tensor(mul_109, unsqueeze_293);  mul_109 = unsqueeze_293 = None
        unsqueeze_294: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg185_1, -1);  arg185_1 = None
        unsqueeze_295: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_294, -1);  unsqueeze_294 = None
        add_84: "f32[32, 256, 28, 28]" = torch.ops.aten.add.Tensor(mul_110, unsqueeze_295);  mul_110 = unsqueeze_295 = None
        convert_element_type_110: "f16[32, 256, 28, 28]" = torch.ops.prims.convert_element_type.default(add_84, torch.float16);  add_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_34: "f16[32, 256, 28, 28]" = torch.ops.aten.relu.default(convert_element_type_110);  convert_element_type_110 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_37: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_34, arg186_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_34 = arg186_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_111: "f32[256]" = torch.ops.prims.convert_element_type.default(arg187_1, torch.float32);  arg187_1 = None
        unsqueeze_296: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_111, -1);  convert_element_type_111 = None
        unsqueeze_297: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_296, -1);  unsqueeze_296 = None
        sub_37: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_37, unsqueeze_297);  convolution_37 = unsqueeze_297 = None
        convert_element_type_112: "f32[256]" = torch.ops.prims.convert_element_type.default(arg188_1, torch.float32);  arg188_1 = None
        add_85: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_112, 1e-05);  convert_element_type_112 = None
        sqrt_37: "f32[256]" = torch.ops.aten.sqrt.default(add_85);  add_85 = None
        reciprocal_37: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_37);  sqrt_37 = None
        mul_111: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_37, 1);  reciprocal_37 = None
        unsqueeze_298: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_111, -1);  mul_111 = None
        unsqueeze_299: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_298, -1);  unsqueeze_298 = None
        mul_112: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_37, unsqueeze_299);  sub_37 = unsqueeze_299 = None
        unsqueeze_300: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg189_1, -1);  arg189_1 = None
        unsqueeze_301: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_300, -1);  unsqueeze_300 = None
        mul_113: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_112, unsqueeze_301);  mul_112 = unsqueeze_301 = None
        unsqueeze_302: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg190_1, -1);  arg190_1 = None
        unsqueeze_303: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_302, -1);  unsqueeze_302 = None
        add_86: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_113, unsqueeze_303);  mul_113 = unsqueeze_303 = None
        convert_element_type_113: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_86, torch.float16);  add_86 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_35: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_113);  convert_element_type_113 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_38: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_35, arg191_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_35 = arg191_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_114: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg192_1, torch.float32);  arg192_1 = None
        unsqueeze_304: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_114, -1);  convert_element_type_114 = None
        unsqueeze_305: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_304, -1);  unsqueeze_304 = None
        sub_38: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_38, unsqueeze_305);  convolution_38 = unsqueeze_305 = None
        convert_element_type_115: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg193_1, torch.float32);  arg193_1 = None
        add_87: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_115, 1e-05);  convert_element_type_115 = None
        sqrt_38: "f32[1024]" = torch.ops.aten.sqrt.default(add_87);  add_87 = None
        reciprocal_38: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_38);  sqrt_38 = None
        mul_114: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_38, 1);  reciprocal_38 = None
        unsqueeze_306: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_114, -1);  mul_114 = None
        unsqueeze_307: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_306, -1);  unsqueeze_306 = None
        mul_115: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_38, unsqueeze_307);  sub_38 = unsqueeze_307 = None
        unsqueeze_308: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg194_1, -1);  arg194_1 = None
        unsqueeze_309: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_308, -1);  unsqueeze_308 = None
        mul_116: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_115, unsqueeze_309);  mul_115 = unsqueeze_309 = None
        unsqueeze_310: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg195_1, -1);  arg195_1 = None
        unsqueeze_311: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_310, -1);  unsqueeze_310 = None
        add_88: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_116, unsqueeze_311);  mul_116 = unsqueeze_311 = None
        convert_element_type_116: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_88, torch.float16);  add_88 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_39: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_33, arg196_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_33 = arg196_1 = None
        convert_element_type_117: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg197_1, torch.float32);  arg197_1 = None
        unsqueeze_312: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_117, -1);  convert_element_type_117 = None
        unsqueeze_313: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_312, -1);  unsqueeze_312 = None
        sub_39: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_39, unsqueeze_313);  convolution_39 = unsqueeze_313 = None
        convert_element_type_118: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg198_1, torch.float32);  arg198_1 = None
        add_89: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_118, 1e-05);  convert_element_type_118 = None
        sqrt_39: "f32[1024]" = torch.ops.aten.sqrt.default(add_89);  add_89 = None
        reciprocal_39: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_39);  sqrt_39 = None
        mul_117: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_39, 1);  reciprocal_39 = None
        unsqueeze_314: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_117, -1);  mul_117 = None
        unsqueeze_315: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_314, -1);  unsqueeze_314 = None
        mul_118: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_39, unsqueeze_315);  sub_39 = unsqueeze_315 = None
        unsqueeze_316: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg199_1, -1);  arg199_1 = None
        unsqueeze_317: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_316, -1);  unsqueeze_316 = None
        mul_119: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_118, unsqueeze_317);  mul_118 = unsqueeze_317 = None
        unsqueeze_318: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg200_1, -1);  arg200_1 = None
        unsqueeze_319: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_318, -1);  unsqueeze_318 = None
        add_90: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_119, unsqueeze_319);  mul_119 = unsqueeze_319 = None
        convert_element_type_119: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_90, torch.float16);  add_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_91: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_116, convert_element_type_119);  convert_element_type_116 = convert_element_type_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_36: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_91);  add_91 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_40: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_36, arg201_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg201_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_120: "f32[256]" = torch.ops.prims.convert_element_type.default(arg202_1, torch.float32);  arg202_1 = None
        unsqueeze_320: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_120, -1);  convert_element_type_120 = None
        unsqueeze_321: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_320, -1);  unsqueeze_320 = None
        sub_40: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_40, unsqueeze_321);  convolution_40 = unsqueeze_321 = None
        convert_element_type_121: "f32[256]" = torch.ops.prims.convert_element_type.default(arg203_1, torch.float32);  arg203_1 = None
        add_92: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_121, 1e-05);  convert_element_type_121 = None
        sqrt_40: "f32[256]" = torch.ops.aten.sqrt.default(add_92);  add_92 = None
        reciprocal_40: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_40);  sqrt_40 = None
        mul_120: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_40, 1);  reciprocal_40 = None
        unsqueeze_322: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_120, -1);  mul_120 = None
        unsqueeze_323: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_322, -1);  unsqueeze_322 = None
        mul_121: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_40, unsqueeze_323);  sub_40 = unsqueeze_323 = None
        unsqueeze_324: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg204_1, -1);  arg204_1 = None
        unsqueeze_325: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_324, -1);  unsqueeze_324 = None
        mul_122: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_121, unsqueeze_325);  mul_121 = unsqueeze_325 = None
        unsqueeze_326: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg205_1, -1);  arg205_1 = None
        unsqueeze_327: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_326, -1);  unsqueeze_326 = None
        add_93: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_122, unsqueeze_327);  mul_122 = unsqueeze_327 = None
        convert_element_type_122: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_93, torch.float16);  add_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_37: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_122);  convert_element_type_122 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_41: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_37, arg206_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_37 = arg206_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_123: "f32[256]" = torch.ops.prims.convert_element_type.default(arg207_1, torch.float32);  arg207_1 = None
        unsqueeze_328: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_123, -1);  convert_element_type_123 = None
        unsqueeze_329: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_328, -1);  unsqueeze_328 = None
        sub_41: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_41, unsqueeze_329);  convolution_41 = unsqueeze_329 = None
        convert_element_type_124: "f32[256]" = torch.ops.prims.convert_element_type.default(arg208_1, torch.float32);  arg208_1 = None
        add_94: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_124, 1e-05);  convert_element_type_124 = None
        sqrt_41: "f32[256]" = torch.ops.aten.sqrt.default(add_94);  add_94 = None
        reciprocal_41: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_41);  sqrt_41 = None
        mul_123: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_41, 1);  reciprocal_41 = None
        unsqueeze_330: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_123, -1);  mul_123 = None
        unsqueeze_331: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_330, -1);  unsqueeze_330 = None
        mul_124: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_41, unsqueeze_331);  sub_41 = unsqueeze_331 = None
        unsqueeze_332: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg209_1, -1);  arg209_1 = None
        unsqueeze_333: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_332, -1);  unsqueeze_332 = None
        mul_125: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_124, unsqueeze_333);  mul_124 = unsqueeze_333 = None
        unsqueeze_334: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg210_1, -1);  arg210_1 = None
        unsqueeze_335: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_334, -1);  unsqueeze_334 = None
        add_95: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_125, unsqueeze_335);  mul_125 = unsqueeze_335 = None
        convert_element_type_125: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_95, torch.float16);  add_95 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_38: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_125);  convert_element_type_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_42: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_38, arg211_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_38 = arg211_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_126: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg212_1, torch.float32);  arg212_1 = None
        unsqueeze_336: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_126, -1);  convert_element_type_126 = None
        unsqueeze_337: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_336, -1);  unsqueeze_336 = None
        sub_42: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_42, unsqueeze_337);  convolution_42 = unsqueeze_337 = None
        convert_element_type_127: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg213_1, torch.float32);  arg213_1 = None
        add_96: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_127, 1e-05);  convert_element_type_127 = None
        sqrt_42: "f32[1024]" = torch.ops.aten.sqrt.default(add_96);  add_96 = None
        reciprocal_42: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_42);  sqrt_42 = None
        mul_126: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_42, 1);  reciprocal_42 = None
        unsqueeze_338: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_126, -1);  mul_126 = None
        unsqueeze_339: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_338, -1);  unsqueeze_338 = None
        mul_127: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_42, unsqueeze_339);  sub_42 = unsqueeze_339 = None
        unsqueeze_340: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg214_1, -1);  arg214_1 = None
        unsqueeze_341: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_340, -1);  unsqueeze_340 = None
        mul_128: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_127, unsqueeze_341);  mul_127 = unsqueeze_341 = None
        unsqueeze_342: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg215_1, -1);  arg215_1 = None
        unsqueeze_343: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_342, -1);  unsqueeze_342 = None
        add_97: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_128, unsqueeze_343);  mul_128 = unsqueeze_343 = None
        convert_element_type_128: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_97, torch.float16);  add_97 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_98: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_128, relu_36);  convert_element_type_128 = relu_36 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_39: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_98);  add_98 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_43: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_39, arg216_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg216_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_129: "f32[256]" = torch.ops.prims.convert_element_type.default(arg217_1, torch.float32);  arg217_1 = None
        unsqueeze_344: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_129, -1);  convert_element_type_129 = None
        unsqueeze_345: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_344, -1);  unsqueeze_344 = None
        sub_43: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_43, unsqueeze_345);  convolution_43 = unsqueeze_345 = None
        convert_element_type_130: "f32[256]" = torch.ops.prims.convert_element_type.default(arg218_1, torch.float32);  arg218_1 = None
        add_99: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_130, 1e-05);  convert_element_type_130 = None
        sqrt_43: "f32[256]" = torch.ops.aten.sqrt.default(add_99);  add_99 = None
        reciprocal_43: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_43);  sqrt_43 = None
        mul_129: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_43, 1);  reciprocal_43 = None
        unsqueeze_346: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_129, -1);  mul_129 = None
        unsqueeze_347: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_346, -1);  unsqueeze_346 = None
        mul_130: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_43, unsqueeze_347);  sub_43 = unsqueeze_347 = None
        unsqueeze_348: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg219_1, -1);  arg219_1 = None
        unsqueeze_349: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_348, -1);  unsqueeze_348 = None
        mul_131: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_130, unsqueeze_349);  mul_130 = unsqueeze_349 = None
        unsqueeze_350: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg220_1, -1);  arg220_1 = None
        unsqueeze_351: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_350, -1);  unsqueeze_350 = None
        add_100: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_131, unsqueeze_351);  mul_131 = unsqueeze_351 = None
        convert_element_type_131: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_100, torch.float16);  add_100 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_40: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_131);  convert_element_type_131 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_44: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_40, arg221_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_40 = arg221_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_132: "f32[256]" = torch.ops.prims.convert_element_type.default(arg222_1, torch.float32);  arg222_1 = None
        unsqueeze_352: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_132, -1);  convert_element_type_132 = None
        unsqueeze_353: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_352, -1);  unsqueeze_352 = None
        sub_44: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_44, unsqueeze_353);  convolution_44 = unsqueeze_353 = None
        convert_element_type_133: "f32[256]" = torch.ops.prims.convert_element_type.default(arg223_1, torch.float32);  arg223_1 = None
        add_101: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_133, 1e-05);  convert_element_type_133 = None
        sqrt_44: "f32[256]" = torch.ops.aten.sqrt.default(add_101);  add_101 = None
        reciprocal_44: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_44);  sqrt_44 = None
        mul_132: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_44, 1);  reciprocal_44 = None
        unsqueeze_354: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_132, -1);  mul_132 = None
        unsqueeze_355: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_354, -1);  unsqueeze_354 = None
        mul_133: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_44, unsqueeze_355);  sub_44 = unsqueeze_355 = None
        unsqueeze_356: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg224_1, -1);  arg224_1 = None
        unsqueeze_357: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_356, -1);  unsqueeze_356 = None
        mul_134: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_133, unsqueeze_357);  mul_133 = unsqueeze_357 = None
        unsqueeze_358: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg225_1, -1);  arg225_1 = None
        unsqueeze_359: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_358, -1);  unsqueeze_358 = None
        add_102: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_134, unsqueeze_359);  mul_134 = unsqueeze_359 = None
        convert_element_type_134: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_102, torch.float16);  add_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_41: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_134);  convert_element_type_134 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_45: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_41, arg226_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_41 = arg226_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_135: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg227_1, torch.float32);  arg227_1 = None
        unsqueeze_360: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_135, -1);  convert_element_type_135 = None
        unsqueeze_361: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_360, -1);  unsqueeze_360 = None
        sub_45: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_45, unsqueeze_361);  convolution_45 = unsqueeze_361 = None
        convert_element_type_136: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg228_1, torch.float32);  arg228_1 = None
        add_103: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_136, 1e-05);  convert_element_type_136 = None
        sqrt_45: "f32[1024]" = torch.ops.aten.sqrt.default(add_103);  add_103 = None
        reciprocal_45: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_45);  sqrt_45 = None
        mul_135: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_45, 1);  reciprocal_45 = None
        unsqueeze_362: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_135, -1);  mul_135 = None
        unsqueeze_363: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_362, -1);  unsqueeze_362 = None
        mul_136: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_45, unsqueeze_363);  sub_45 = unsqueeze_363 = None
        unsqueeze_364: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg229_1, -1);  arg229_1 = None
        unsqueeze_365: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_364, -1);  unsqueeze_364 = None
        mul_137: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_136, unsqueeze_365);  mul_136 = unsqueeze_365 = None
        unsqueeze_366: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg230_1, -1);  arg230_1 = None
        unsqueeze_367: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_366, -1);  unsqueeze_366 = None
        add_104: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_137, unsqueeze_367);  mul_137 = unsqueeze_367 = None
        convert_element_type_137: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_104, torch.float16);  add_104 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_105: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_137, relu_39);  convert_element_type_137 = relu_39 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_42: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_105);  add_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_46: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_42, arg231_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg231_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_138: "f32[256]" = torch.ops.prims.convert_element_type.default(arg232_1, torch.float32);  arg232_1 = None
        unsqueeze_368: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_138, -1);  convert_element_type_138 = None
        unsqueeze_369: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_368, -1);  unsqueeze_368 = None
        sub_46: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_46, unsqueeze_369);  convolution_46 = unsqueeze_369 = None
        convert_element_type_139: "f32[256]" = torch.ops.prims.convert_element_type.default(arg233_1, torch.float32);  arg233_1 = None
        add_106: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_139, 1e-05);  convert_element_type_139 = None
        sqrt_46: "f32[256]" = torch.ops.aten.sqrt.default(add_106);  add_106 = None
        reciprocal_46: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_46);  sqrt_46 = None
        mul_138: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_46, 1);  reciprocal_46 = None
        unsqueeze_370: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_138, -1);  mul_138 = None
        unsqueeze_371: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_370, -1);  unsqueeze_370 = None
        mul_139: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_46, unsqueeze_371);  sub_46 = unsqueeze_371 = None
        unsqueeze_372: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg234_1, -1);  arg234_1 = None
        unsqueeze_373: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_372, -1);  unsqueeze_372 = None
        mul_140: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_139, unsqueeze_373);  mul_139 = unsqueeze_373 = None
        unsqueeze_374: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg235_1, -1);  arg235_1 = None
        unsqueeze_375: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_374, -1);  unsqueeze_374 = None
        add_107: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_140, unsqueeze_375);  mul_140 = unsqueeze_375 = None
        convert_element_type_140: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_107, torch.float16);  add_107 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_43: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_140);  convert_element_type_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_47: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_43, arg236_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_43 = arg236_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_141: "f32[256]" = torch.ops.prims.convert_element_type.default(arg237_1, torch.float32);  arg237_1 = None
        unsqueeze_376: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_141, -1);  convert_element_type_141 = None
        unsqueeze_377: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_376, -1);  unsqueeze_376 = None
        sub_47: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_47, unsqueeze_377);  convolution_47 = unsqueeze_377 = None
        convert_element_type_142: "f32[256]" = torch.ops.prims.convert_element_type.default(arg238_1, torch.float32);  arg238_1 = None
        add_108: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_142, 1e-05);  convert_element_type_142 = None
        sqrt_47: "f32[256]" = torch.ops.aten.sqrt.default(add_108);  add_108 = None
        reciprocal_47: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_47);  sqrt_47 = None
        mul_141: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_47, 1);  reciprocal_47 = None
        unsqueeze_378: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_141, -1);  mul_141 = None
        unsqueeze_379: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_378, -1);  unsqueeze_378 = None
        mul_142: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_47, unsqueeze_379);  sub_47 = unsqueeze_379 = None
        unsqueeze_380: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg239_1, -1);  arg239_1 = None
        unsqueeze_381: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_380, -1);  unsqueeze_380 = None
        mul_143: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_142, unsqueeze_381);  mul_142 = unsqueeze_381 = None
        unsqueeze_382: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg240_1, -1);  arg240_1 = None
        unsqueeze_383: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_382, -1);  unsqueeze_382 = None
        add_109: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_143, unsqueeze_383);  mul_143 = unsqueeze_383 = None
        convert_element_type_143: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_109, torch.float16);  add_109 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_44: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_143);  convert_element_type_143 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_48: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_44, arg241_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_44 = arg241_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_144: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg242_1, torch.float32);  arg242_1 = None
        unsqueeze_384: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_144, -1);  convert_element_type_144 = None
        unsqueeze_385: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_384, -1);  unsqueeze_384 = None
        sub_48: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_48, unsqueeze_385);  convolution_48 = unsqueeze_385 = None
        convert_element_type_145: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg243_1, torch.float32);  arg243_1 = None
        add_110: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_145, 1e-05);  convert_element_type_145 = None
        sqrt_48: "f32[1024]" = torch.ops.aten.sqrt.default(add_110);  add_110 = None
        reciprocal_48: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_48);  sqrt_48 = None
        mul_144: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_48, 1);  reciprocal_48 = None
        unsqueeze_386: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_144, -1);  mul_144 = None
        unsqueeze_387: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_386, -1);  unsqueeze_386 = None
        mul_145: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_48, unsqueeze_387);  sub_48 = unsqueeze_387 = None
        unsqueeze_388: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg244_1, -1);  arg244_1 = None
        unsqueeze_389: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_388, -1);  unsqueeze_388 = None
        mul_146: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_145, unsqueeze_389);  mul_145 = unsqueeze_389 = None
        unsqueeze_390: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg245_1, -1);  arg245_1 = None
        unsqueeze_391: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_390, -1);  unsqueeze_390 = None
        add_111: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_146, unsqueeze_391);  mul_146 = unsqueeze_391 = None
        convert_element_type_146: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_111, torch.float16);  add_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_112: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_146, relu_42);  convert_element_type_146 = relu_42 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_45: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_112);  add_112 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_49: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_45, arg246_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg246_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_147: "f32[256]" = torch.ops.prims.convert_element_type.default(arg247_1, torch.float32);  arg247_1 = None
        unsqueeze_392: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_147, -1);  convert_element_type_147 = None
        unsqueeze_393: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_392, -1);  unsqueeze_392 = None
        sub_49: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_49, unsqueeze_393);  convolution_49 = unsqueeze_393 = None
        convert_element_type_148: "f32[256]" = torch.ops.prims.convert_element_type.default(arg248_1, torch.float32);  arg248_1 = None
        add_113: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_148, 1e-05);  convert_element_type_148 = None
        sqrt_49: "f32[256]" = torch.ops.aten.sqrt.default(add_113);  add_113 = None
        reciprocal_49: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_49);  sqrt_49 = None
        mul_147: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_49, 1);  reciprocal_49 = None
        unsqueeze_394: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_147, -1);  mul_147 = None
        unsqueeze_395: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_394, -1);  unsqueeze_394 = None
        mul_148: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_49, unsqueeze_395);  sub_49 = unsqueeze_395 = None
        unsqueeze_396: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg249_1, -1);  arg249_1 = None
        unsqueeze_397: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_396, -1);  unsqueeze_396 = None
        mul_149: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_148, unsqueeze_397);  mul_148 = unsqueeze_397 = None
        unsqueeze_398: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg250_1, -1);  arg250_1 = None
        unsqueeze_399: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_398, -1);  unsqueeze_398 = None
        add_114: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_149, unsqueeze_399);  mul_149 = unsqueeze_399 = None
        convert_element_type_149: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_114, torch.float16);  add_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_46: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_149);  convert_element_type_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_50: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_46, arg251_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_46 = arg251_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_150: "f32[256]" = torch.ops.prims.convert_element_type.default(arg252_1, torch.float32);  arg252_1 = None
        unsqueeze_400: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_150, -1);  convert_element_type_150 = None
        unsqueeze_401: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_400, -1);  unsqueeze_400 = None
        sub_50: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_50, unsqueeze_401);  convolution_50 = unsqueeze_401 = None
        convert_element_type_151: "f32[256]" = torch.ops.prims.convert_element_type.default(arg253_1, torch.float32);  arg253_1 = None
        add_115: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_151, 1e-05);  convert_element_type_151 = None
        sqrt_50: "f32[256]" = torch.ops.aten.sqrt.default(add_115);  add_115 = None
        reciprocal_50: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_50);  sqrt_50 = None
        mul_150: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_50, 1);  reciprocal_50 = None
        unsqueeze_402: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_150, -1);  mul_150 = None
        unsqueeze_403: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_402, -1);  unsqueeze_402 = None
        mul_151: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_50, unsqueeze_403);  sub_50 = unsqueeze_403 = None
        unsqueeze_404: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg254_1, -1);  arg254_1 = None
        unsqueeze_405: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_404, -1);  unsqueeze_404 = None
        mul_152: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_151, unsqueeze_405);  mul_151 = unsqueeze_405 = None
        unsqueeze_406: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg255_1, -1);  arg255_1 = None
        unsqueeze_407: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_406, -1);  unsqueeze_406 = None
        add_116: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_152, unsqueeze_407);  mul_152 = unsqueeze_407 = None
        convert_element_type_152: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_116, torch.float16);  add_116 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_47: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_152);  convert_element_type_152 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_51: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_47, arg256_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_47 = arg256_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_153: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg257_1, torch.float32);  arg257_1 = None
        unsqueeze_408: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_153, -1);  convert_element_type_153 = None
        unsqueeze_409: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_408, -1);  unsqueeze_408 = None
        sub_51: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_51, unsqueeze_409);  convolution_51 = unsqueeze_409 = None
        convert_element_type_154: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg258_1, torch.float32);  arg258_1 = None
        add_117: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_154, 1e-05);  convert_element_type_154 = None
        sqrt_51: "f32[1024]" = torch.ops.aten.sqrt.default(add_117);  add_117 = None
        reciprocal_51: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_51);  sqrt_51 = None
        mul_153: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_51, 1);  reciprocal_51 = None
        unsqueeze_410: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_153, -1);  mul_153 = None
        unsqueeze_411: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_410, -1);  unsqueeze_410 = None
        mul_154: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_51, unsqueeze_411);  sub_51 = unsqueeze_411 = None
        unsqueeze_412: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg259_1, -1);  arg259_1 = None
        unsqueeze_413: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_412, -1);  unsqueeze_412 = None
        mul_155: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_154, unsqueeze_413);  mul_154 = unsqueeze_413 = None
        unsqueeze_414: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg260_1, -1);  arg260_1 = None
        unsqueeze_415: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_414, -1);  unsqueeze_414 = None
        add_118: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_155, unsqueeze_415);  mul_155 = unsqueeze_415 = None
        convert_element_type_155: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_118, torch.float16);  add_118 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_119: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_155, relu_45);  convert_element_type_155 = relu_45 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_48: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_119);  add_119 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_52: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_48, arg261_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg261_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_156: "f32[256]" = torch.ops.prims.convert_element_type.default(arg262_1, torch.float32);  arg262_1 = None
        unsqueeze_416: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_156, -1);  convert_element_type_156 = None
        unsqueeze_417: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_416, -1);  unsqueeze_416 = None
        sub_52: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_52, unsqueeze_417);  convolution_52 = unsqueeze_417 = None
        convert_element_type_157: "f32[256]" = torch.ops.prims.convert_element_type.default(arg263_1, torch.float32);  arg263_1 = None
        add_120: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_157, 1e-05);  convert_element_type_157 = None
        sqrt_52: "f32[256]" = torch.ops.aten.sqrt.default(add_120);  add_120 = None
        reciprocal_52: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_52);  sqrt_52 = None
        mul_156: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_52, 1);  reciprocal_52 = None
        unsqueeze_418: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_156, -1);  mul_156 = None
        unsqueeze_419: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_418, -1);  unsqueeze_418 = None
        mul_157: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_52, unsqueeze_419);  sub_52 = unsqueeze_419 = None
        unsqueeze_420: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg264_1, -1);  arg264_1 = None
        unsqueeze_421: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_420, -1);  unsqueeze_420 = None
        mul_158: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_157, unsqueeze_421);  mul_157 = unsqueeze_421 = None
        unsqueeze_422: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg265_1, -1);  arg265_1 = None
        unsqueeze_423: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_422, -1);  unsqueeze_422 = None
        add_121: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_158, unsqueeze_423);  mul_158 = unsqueeze_423 = None
        convert_element_type_158: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_121, torch.float16);  add_121 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_49: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_158);  convert_element_type_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_53: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_49, arg266_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_49 = arg266_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_159: "f32[256]" = torch.ops.prims.convert_element_type.default(arg267_1, torch.float32);  arg267_1 = None
        unsqueeze_424: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_159, -1);  convert_element_type_159 = None
        unsqueeze_425: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_424, -1);  unsqueeze_424 = None
        sub_53: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_53, unsqueeze_425);  convolution_53 = unsqueeze_425 = None
        convert_element_type_160: "f32[256]" = torch.ops.prims.convert_element_type.default(arg268_1, torch.float32);  arg268_1 = None
        add_122: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_160, 1e-05);  convert_element_type_160 = None
        sqrt_53: "f32[256]" = torch.ops.aten.sqrt.default(add_122);  add_122 = None
        reciprocal_53: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_53);  sqrt_53 = None
        mul_159: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_53, 1);  reciprocal_53 = None
        unsqueeze_426: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_159, -1);  mul_159 = None
        unsqueeze_427: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_426, -1);  unsqueeze_426 = None
        mul_160: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_53, unsqueeze_427);  sub_53 = unsqueeze_427 = None
        unsqueeze_428: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg269_1, -1);  arg269_1 = None
        unsqueeze_429: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_428, -1);  unsqueeze_428 = None
        mul_161: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_160, unsqueeze_429);  mul_160 = unsqueeze_429 = None
        unsqueeze_430: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg270_1, -1);  arg270_1 = None
        unsqueeze_431: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_430, -1);  unsqueeze_430 = None
        add_123: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_161, unsqueeze_431);  mul_161 = unsqueeze_431 = None
        convert_element_type_161: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_123, torch.float16);  add_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_50: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_161);  convert_element_type_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_54: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_50, arg271_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_50 = arg271_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_162: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg272_1, torch.float32);  arg272_1 = None
        unsqueeze_432: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_162, -1);  convert_element_type_162 = None
        unsqueeze_433: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_432, -1);  unsqueeze_432 = None
        sub_54: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_54, unsqueeze_433);  convolution_54 = unsqueeze_433 = None
        convert_element_type_163: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg273_1, torch.float32);  arg273_1 = None
        add_124: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_163, 1e-05);  convert_element_type_163 = None
        sqrt_54: "f32[1024]" = torch.ops.aten.sqrt.default(add_124);  add_124 = None
        reciprocal_54: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_54);  sqrt_54 = None
        mul_162: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_54, 1);  reciprocal_54 = None
        unsqueeze_434: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_162, -1);  mul_162 = None
        unsqueeze_435: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_434, -1);  unsqueeze_434 = None
        mul_163: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_54, unsqueeze_435);  sub_54 = unsqueeze_435 = None
        unsqueeze_436: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg274_1, -1);  arg274_1 = None
        unsqueeze_437: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_436, -1);  unsqueeze_436 = None
        mul_164: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_163, unsqueeze_437);  mul_163 = unsqueeze_437 = None
        unsqueeze_438: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg275_1, -1);  arg275_1 = None
        unsqueeze_439: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_438, -1);  unsqueeze_438 = None
        add_125: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_164, unsqueeze_439);  mul_164 = unsqueeze_439 = None
        convert_element_type_164: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_125, torch.float16);  add_125 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_126: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_164, relu_48);  convert_element_type_164 = relu_48 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_51: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_126);  add_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_55: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_51, arg276_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg276_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_165: "f32[256]" = torch.ops.prims.convert_element_type.default(arg277_1, torch.float32);  arg277_1 = None
        unsqueeze_440: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_165, -1);  convert_element_type_165 = None
        unsqueeze_441: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_440, -1);  unsqueeze_440 = None
        sub_55: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_55, unsqueeze_441);  convolution_55 = unsqueeze_441 = None
        convert_element_type_166: "f32[256]" = torch.ops.prims.convert_element_type.default(arg278_1, torch.float32);  arg278_1 = None
        add_127: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_166, 1e-05);  convert_element_type_166 = None
        sqrt_55: "f32[256]" = torch.ops.aten.sqrt.default(add_127);  add_127 = None
        reciprocal_55: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_55);  sqrt_55 = None
        mul_165: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_55, 1);  reciprocal_55 = None
        unsqueeze_442: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_165, -1);  mul_165 = None
        unsqueeze_443: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_442, -1);  unsqueeze_442 = None
        mul_166: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_55, unsqueeze_443);  sub_55 = unsqueeze_443 = None
        unsqueeze_444: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg279_1, -1);  arg279_1 = None
        unsqueeze_445: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_444, -1);  unsqueeze_444 = None
        mul_167: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_166, unsqueeze_445);  mul_166 = unsqueeze_445 = None
        unsqueeze_446: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg280_1, -1);  arg280_1 = None
        unsqueeze_447: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_446, -1);  unsqueeze_446 = None
        add_128: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_167, unsqueeze_447);  mul_167 = unsqueeze_447 = None
        convert_element_type_167: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_128, torch.float16);  add_128 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_52: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_167);  convert_element_type_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_56: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_52, arg281_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_52 = arg281_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_168: "f32[256]" = torch.ops.prims.convert_element_type.default(arg282_1, torch.float32);  arg282_1 = None
        unsqueeze_448: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_168, -1);  convert_element_type_168 = None
        unsqueeze_449: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_448, -1);  unsqueeze_448 = None
        sub_56: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_56, unsqueeze_449);  convolution_56 = unsqueeze_449 = None
        convert_element_type_169: "f32[256]" = torch.ops.prims.convert_element_type.default(arg283_1, torch.float32);  arg283_1 = None
        add_129: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_169, 1e-05);  convert_element_type_169 = None
        sqrt_56: "f32[256]" = torch.ops.aten.sqrt.default(add_129);  add_129 = None
        reciprocal_56: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_56);  sqrt_56 = None
        mul_168: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_56, 1);  reciprocal_56 = None
        unsqueeze_450: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_168, -1);  mul_168 = None
        unsqueeze_451: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_450, -1);  unsqueeze_450 = None
        mul_169: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_56, unsqueeze_451);  sub_56 = unsqueeze_451 = None
        unsqueeze_452: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg284_1, -1);  arg284_1 = None
        unsqueeze_453: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_452, -1);  unsqueeze_452 = None
        mul_170: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_169, unsqueeze_453);  mul_169 = unsqueeze_453 = None
        unsqueeze_454: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg285_1, -1);  arg285_1 = None
        unsqueeze_455: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_454, -1);  unsqueeze_454 = None
        add_130: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_170, unsqueeze_455);  mul_170 = unsqueeze_455 = None
        convert_element_type_170: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_130, torch.float16);  add_130 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_53: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_170);  convert_element_type_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_57: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_53, arg286_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_53 = arg286_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_171: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg287_1, torch.float32);  arg287_1 = None
        unsqueeze_456: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_171, -1);  convert_element_type_171 = None
        unsqueeze_457: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_456, -1);  unsqueeze_456 = None
        sub_57: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_57, unsqueeze_457);  convolution_57 = unsqueeze_457 = None
        convert_element_type_172: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg288_1, torch.float32);  arg288_1 = None
        add_131: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_172, 1e-05);  convert_element_type_172 = None
        sqrt_57: "f32[1024]" = torch.ops.aten.sqrt.default(add_131);  add_131 = None
        reciprocal_57: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_57);  sqrt_57 = None
        mul_171: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_57, 1);  reciprocal_57 = None
        unsqueeze_458: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_171, -1);  mul_171 = None
        unsqueeze_459: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_458, -1);  unsqueeze_458 = None
        mul_172: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_57, unsqueeze_459);  sub_57 = unsqueeze_459 = None
        unsqueeze_460: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg289_1, -1);  arg289_1 = None
        unsqueeze_461: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_460, -1);  unsqueeze_460 = None
        mul_173: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_172, unsqueeze_461);  mul_172 = unsqueeze_461 = None
        unsqueeze_462: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg290_1, -1);  arg290_1 = None
        unsqueeze_463: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_462, -1);  unsqueeze_462 = None
        add_132: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_173, unsqueeze_463);  mul_173 = unsqueeze_463 = None
        convert_element_type_173: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_132, torch.float16);  add_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_133: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_173, relu_51);  convert_element_type_173 = relu_51 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_54: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_133);  add_133 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_58: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_54, arg291_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg291_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_174: "f32[256]" = torch.ops.prims.convert_element_type.default(arg292_1, torch.float32);  arg292_1 = None
        unsqueeze_464: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_174, -1);  convert_element_type_174 = None
        unsqueeze_465: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_464, -1);  unsqueeze_464 = None
        sub_58: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_58, unsqueeze_465);  convolution_58 = unsqueeze_465 = None
        convert_element_type_175: "f32[256]" = torch.ops.prims.convert_element_type.default(arg293_1, torch.float32);  arg293_1 = None
        add_134: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_175, 1e-05);  convert_element_type_175 = None
        sqrt_58: "f32[256]" = torch.ops.aten.sqrt.default(add_134);  add_134 = None
        reciprocal_58: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_58);  sqrt_58 = None
        mul_174: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_58, 1);  reciprocal_58 = None
        unsqueeze_466: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_174, -1);  mul_174 = None
        unsqueeze_467: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_466, -1);  unsqueeze_466 = None
        mul_175: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_58, unsqueeze_467);  sub_58 = unsqueeze_467 = None
        unsqueeze_468: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg294_1, -1);  arg294_1 = None
        unsqueeze_469: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_468, -1);  unsqueeze_468 = None
        mul_176: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_175, unsqueeze_469);  mul_175 = unsqueeze_469 = None
        unsqueeze_470: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg295_1, -1);  arg295_1 = None
        unsqueeze_471: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_470, -1);  unsqueeze_470 = None
        add_135: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_176, unsqueeze_471);  mul_176 = unsqueeze_471 = None
        convert_element_type_176: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_135, torch.float16);  add_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_55: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_176);  convert_element_type_176 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_59: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_55, arg296_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_55 = arg296_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_177: "f32[256]" = torch.ops.prims.convert_element_type.default(arg297_1, torch.float32);  arg297_1 = None
        unsqueeze_472: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_177, -1);  convert_element_type_177 = None
        unsqueeze_473: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_472, -1);  unsqueeze_472 = None
        sub_59: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_59, unsqueeze_473);  convolution_59 = unsqueeze_473 = None
        convert_element_type_178: "f32[256]" = torch.ops.prims.convert_element_type.default(arg298_1, torch.float32);  arg298_1 = None
        add_136: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_178, 1e-05);  convert_element_type_178 = None
        sqrt_59: "f32[256]" = torch.ops.aten.sqrt.default(add_136);  add_136 = None
        reciprocal_59: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_59);  sqrt_59 = None
        mul_177: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_59, 1);  reciprocal_59 = None
        unsqueeze_474: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_177, -1);  mul_177 = None
        unsqueeze_475: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_474, -1);  unsqueeze_474 = None
        mul_178: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_59, unsqueeze_475);  sub_59 = unsqueeze_475 = None
        unsqueeze_476: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg299_1, -1);  arg299_1 = None
        unsqueeze_477: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_476, -1);  unsqueeze_476 = None
        mul_179: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_178, unsqueeze_477);  mul_178 = unsqueeze_477 = None
        unsqueeze_478: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg300_1, -1);  arg300_1 = None
        unsqueeze_479: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_478, -1);  unsqueeze_478 = None
        add_137: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_179, unsqueeze_479);  mul_179 = unsqueeze_479 = None
        convert_element_type_179: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_137, torch.float16);  add_137 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_56: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_179);  convert_element_type_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_60: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_56, arg301_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_56 = arg301_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_180: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg302_1, torch.float32);  arg302_1 = None
        unsqueeze_480: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_180, -1);  convert_element_type_180 = None
        unsqueeze_481: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_480, -1);  unsqueeze_480 = None
        sub_60: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_60, unsqueeze_481);  convolution_60 = unsqueeze_481 = None
        convert_element_type_181: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg303_1, torch.float32);  arg303_1 = None
        add_138: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_181, 1e-05);  convert_element_type_181 = None
        sqrt_60: "f32[1024]" = torch.ops.aten.sqrt.default(add_138);  add_138 = None
        reciprocal_60: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_60);  sqrt_60 = None
        mul_180: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_60, 1);  reciprocal_60 = None
        unsqueeze_482: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_180, -1);  mul_180 = None
        unsqueeze_483: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_482, -1);  unsqueeze_482 = None
        mul_181: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_60, unsqueeze_483);  sub_60 = unsqueeze_483 = None
        unsqueeze_484: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg304_1, -1);  arg304_1 = None
        unsqueeze_485: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_484, -1);  unsqueeze_484 = None
        mul_182: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_181, unsqueeze_485);  mul_181 = unsqueeze_485 = None
        unsqueeze_486: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg305_1, -1);  arg305_1 = None
        unsqueeze_487: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_486, -1);  unsqueeze_486 = None
        add_139: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_182, unsqueeze_487);  mul_182 = unsqueeze_487 = None
        convert_element_type_182: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_139, torch.float16);  add_139 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_140: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_182, relu_54);  convert_element_type_182 = relu_54 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_57: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_140);  add_140 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_61: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_57, arg306_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg306_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_183: "f32[256]" = torch.ops.prims.convert_element_type.default(arg307_1, torch.float32);  arg307_1 = None
        unsqueeze_488: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_183, -1);  convert_element_type_183 = None
        unsqueeze_489: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_488, -1);  unsqueeze_488 = None
        sub_61: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_61, unsqueeze_489);  convolution_61 = unsqueeze_489 = None
        convert_element_type_184: "f32[256]" = torch.ops.prims.convert_element_type.default(arg308_1, torch.float32);  arg308_1 = None
        add_141: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_184, 1e-05);  convert_element_type_184 = None
        sqrt_61: "f32[256]" = torch.ops.aten.sqrt.default(add_141);  add_141 = None
        reciprocal_61: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_61);  sqrt_61 = None
        mul_183: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_61, 1);  reciprocal_61 = None
        unsqueeze_490: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_183, -1);  mul_183 = None
        unsqueeze_491: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_490, -1);  unsqueeze_490 = None
        mul_184: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_61, unsqueeze_491);  sub_61 = unsqueeze_491 = None
        unsqueeze_492: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg309_1, -1);  arg309_1 = None
        unsqueeze_493: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_492, -1);  unsqueeze_492 = None
        mul_185: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_184, unsqueeze_493);  mul_184 = unsqueeze_493 = None
        unsqueeze_494: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg310_1, -1);  arg310_1 = None
        unsqueeze_495: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_494, -1);  unsqueeze_494 = None
        add_142: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_185, unsqueeze_495);  mul_185 = unsqueeze_495 = None
        convert_element_type_185: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_142, torch.float16);  add_142 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_58: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_185);  convert_element_type_185 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_62: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_58, arg311_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_58 = arg311_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_186: "f32[256]" = torch.ops.prims.convert_element_type.default(arg312_1, torch.float32);  arg312_1 = None
        unsqueeze_496: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_186, -1);  convert_element_type_186 = None
        unsqueeze_497: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_496, -1);  unsqueeze_496 = None
        sub_62: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_62, unsqueeze_497);  convolution_62 = unsqueeze_497 = None
        convert_element_type_187: "f32[256]" = torch.ops.prims.convert_element_type.default(arg313_1, torch.float32);  arg313_1 = None
        add_143: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_187, 1e-05);  convert_element_type_187 = None
        sqrt_62: "f32[256]" = torch.ops.aten.sqrt.default(add_143);  add_143 = None
        reciprocal_62: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_62);  sqrt_62 = None
        mul_186: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_62, 1);  reciprocal_62 = None
        unsqueeze_498: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_186, -1);  mul_186 = None
        unsqueeze_499: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_498, -1);  unsqueeze_498 = None
        mul_187: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_62, unsqueeze_499);  sub_62 = unsqueeze_499 = None
        unsqueeze_500: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg314_1, -1);  arg314_1 = None
        unsqueeze_501: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_500, -1);  unsqueeze_500 = None
        mul_188: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_187, unsqueeze_501);  mul_187 = unsqueeze_501 = None
        unsqueeze_502: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg315_1, -1);  arg315_1 = None
        unsqueeze_503: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_502, -1);  unsqueeze_502 = None
        add_144: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_188, unsqueeze_503);  mul_188 = unsqueeze_503 = None
        convert_element_type_188: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_144, torch.float16);  add_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_59: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_188);  convert_element_type_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_63: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_59, arg316_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_59 = arg316_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_189: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg317_1, torch.float32);  arg317_1 = None
        unsqueeze_504: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_189, -1);  convert_element_type_189 = None
        unsqueeze_505: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_504, -1);  unsqueeze_504 = None
        sub_63: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_63, unsqueeze_505);  convolution_63 = unsqueeze_505 = None
        convert_element_type_190: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg318_1, torch.float32);  arg318_1 = None
        add_145: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_190, 1e-05);  convert_element_type_190 = None
        sqrt_63: "f32[1024]" = torch.ops.aten.sqrt.default(add_145);  add_145 = None
        reciprocal_63: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_63);  sqrt_63 = None
        mul_189: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_63, 1);  reciprocal_63 = None
        unsqueeze_506: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_189, -1);  mul_189 = None
        unsqueeze_507: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_506, -1);  unsqueeze_506 = None
        mul_190: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_63, unsqueeze_507);  sub_63 = unsqueeze_507 = None
        unsqueeze_508: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg319_1, -1);  arg319_1 = None
        unsqueeze_509: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_508, -1);  unsqueeze_508 = None
        mul_191: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_190, unsqueeze_509);  mul_190 = unsqueeze_509 = None
        unsqueeze_510: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg320_1, -1);  arg320_1 = None
        unsqueeze_511: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_510, -1);  unsqueeze_510 = None
        add_146: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_191, unsqueeze_511);  mul_191 = unsqueeze_511 = None
        convert_element_type_191: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_146, torch.float16);  add_146 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_147: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_191, relu_57);  convert_element_type_191 = relu_57 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_60: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_147);  add_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_64: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_60, arg321_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg321_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_192: "f32[256]" = torch.ops.prims.convert_element_type.default(arg322_1, torch.float32);  arg322_1 = None
        unsqueeze_512: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_192, -1);  convert_element_type_192 = None
        unsqueeze_513: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_512, -1);  unsqueeze_512 = None
        sub_64: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_64, unsqueeze_513);  convolution_64 = unsqueeze_513 = None
        convert_element_type_193: "f32[256]" = torch.ops.prims.convert_element_type.default(arg323_1, torch.float32);  arg323_1 = None
        add_148: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_193, 1e-05);  convert_element_type_193 = None
        sqrt_64: "f32[256]" = torch.ops.aten.sqrt.default(add_148);  add_148 = None
        reciprocal_64: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_64);  sqrt_64 = None
        mul_192: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_64, 1);  reciprocal_64 = None
        unsqueeze_514: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_192, -1);  mul_192 = None
        unsqueeze_515: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_514, -1);  unsqueeze_514 = None
        mul_193: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_64, unsqueeze_515);  sub_64 = unsqueeze_515 = None
        unsqueeze_516: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg324_1, -1);  arg324_1 = None
        unsqueeze_517: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_516, -1);  unsqueeze_516 = None
        mul_194: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_193, unsqueeze_517);  mul_193 = unsqueeze_517 = None
        unsqueeze_518: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg325_1, -1);  arg325_1 = None
        unsqueeze_519: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_518, -1);  unsqueeze_518 = None
        add_149: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_194, unsqueeze_519);  mul_194 = unsqueeze_519 = None
        convert_element_type_194: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_149, torch.float16);  add_149 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_61: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_194);  convert_element_type_194 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_65: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_61, arg326_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_61 = arg326_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_195: "f32[256]" = torch.ops.prims.convert_element_type.default(arg327_1, torch.float32);  arg327_1 = None
        unsqueeze_520: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_195, -1);  convert_element_type_195 = None
        unsqueeze_521: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_520, -1);  unsqueeze_520 = None
        sub_65: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_65, unsqueeze_521);  convolution_65 = unsqueeze_521 = None
        convert_element_type_196: "f32[256]" = torch.ops.prims.convert_element_type.default(arg328_1, torch.float32);  arg328_1 = None
        add_150: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_196, 1e-05);  convert_element_type_196 = None
        sqrt_65: "f32[256]" = torch.ops.aten.sqrt.default(add_150);  add_150 = None
        reciprocal_65: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_65);  sqrt_65 = None
        mul_195: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_65, 1);  reciprocal_65 = None
        unsqueeze_522: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_195, -1);  mul_195 = None
        unsqueeze_523: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_522, -1);  unsqueeze_522 = None
        mul_196: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_65, unsqueeze_523);  sub_65 = unsqueeze_523 = None
        unsqueeze_524: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg329_1, -1);  arg329_1 = None
        unsqueeze_525: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_524, -1);  unsqueeze_524 = None
        mul_197: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_196, unsqueeze_525);  mul_196 = unsqueeze_525 = None
        unsqueeze_526: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg330_1, -1);  arg330_1 = None
        unsqueeze_527: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_526, -1);  unsqueeze_526 = None
        add_151: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_197, unsqueeze_527);  mul_197 = unsqueeze_527 = None
        convert_element_type_197: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_151, torch.float16);  add_151 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_62: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_197);  convert_element_type_197 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_66: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_62, arg331_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_62 = arg331_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_198: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg332_1, torch.float32);  arg332_1 = None
        unsqueeze_528: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_198, -1);  convert_element_type_198 = None
        unsqueeze_529: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_528, -1);  unsqueeze_528 = None
        sub_66: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_66, unsqueeze_529);  convolution_66 = unsqueeze_529 = None
        convert_element_type_199: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg333_1, torch.float32);  arg333_1 = None
        add_152: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_199, 1e-05);  convert_element_type_199 = None
        sqrt_66: "f32[1024]" = torch.ops.aten.sqrt.default(add_152);  add_152 = None
        reciprocal_66: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_66);  sqrt_66 = None
        mul_198: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_66, 1);  reciprocal_66 = None
        unsqueeze_530: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_198, -1);  mul_198 = None
        unsqueeze_531: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_530, -1);  unsqueeze_530 = None
        mul_199: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_66, unsqueeze_531);  sub_66 = unsqueeze_531 = None
        unsqueeze_532: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg334_1, -1);  arg334_1 = None
        unsqueeze_533: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_532, -1);  unsqueeze_532 = None
        mul_200: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_199, unsqueeze_533);  mul_199 = unsqueeze_533 = None
        unsqueeze_534: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg335_1, -1);  arg335_1 = None
        unsqueeze_535: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_534, -1);  unsqueeze_534 = None
        add_153: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_200, unsqueeze_535);  mul_200 = unsqueeze_535 = None
        convert_element_type_200: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_153, torch.float16);  add_153 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_154: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_200, relu_60);  convert_element_type_200 = relu_60 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_63: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_154);  add_154 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_67: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_63, arg336_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg336_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_201: "f32[256]" = torch.ops.prims.convert_element_type.default(arg337_1, torch.float32);  arg337_1 = None
        unsqueeze_536: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_201, -1);  convert_element_type_201 = None
        unsqueeze_537: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_536, -1);  unsqueeze_536 = None
        sub_67: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_67, unsqueeze_537);  convolution_67 = unsqueeze_537 = None
        convert_element_type_202: "f32[256]" = torch.ops.prims.convert_element_type.default(arg338_1, torch.float32);  arg338_1 = None
        add_155: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_202, 1e-05);  convert_element_type_202 = None
        sqrt_67: "f32[256]" = torch.ops.aten.sqrt.default(add_155);  add_155 = None
        reciprocal_67: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_67);  sqrt_67 = None
        mul_201: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_67, 1);  reciprocal_67 = None
        unsqueeze_538: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_201, -1);  mul_201 = None
        unsqueeze_539: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_538, -1);  unsqueeze_538 = None
        mul_202: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_67, unsqueeze_539);  sub_67 = unsqueeze_539 = None
        unsqueeze_540: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg339_1, -1);  arg339_1 = None
        unsqueeze_541: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_540, -1);  unsqueeze_540 = None
        mul_203: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_202, unsqueeze_541);  mul_202 = unsqueeze_541 = None
        unsqueeze_542: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg340_1, -1);  arg340_1 = None
        unsqueeze_543: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_542, -1);  unsqueeze_542 = None
        add_156: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_203, unsqueeze_543);  mul_203 = unsqueeze_543 = None
        convert_element_type_203: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_156, torch.float16);  add_156 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_64: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_203);  convert_element_type_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_68: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_64, arg341_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_64 = arg341_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_204: "f32[256]" = torch.ops.prims.convert_element_type.default(arg342_1, torch.float32);  arg342_1 = None
        unsqueeze_544: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_204, -1);  convert_element_type_204 = None
        unsqueeze_545: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_544, -1);  unsqueeze_544 = None
        sub_68: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_68, unsqueeze_545);  convolution_68 = unsqueeze_545 = None
        convert_element_type_205: "f32[256]" = torch.ops.prims.convert_element_type.default(arg343_1, torch.float32);  arg343_1 = None
        add_157: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_205, 1e-05);  convert_element_type_205 = None
        sqrt_68: "f32[256]" = torch.ops.aten.sqrt.default(add_157);  add_157 = None
        reciprocal_68: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_68);  sqrt_68 = None
        mul_204: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_68, 1);  reciprocal_68 = None
        unsqueeze_546: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_204, -1);  mul_204 = None
        unsqueeze_547: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_546, -1);  unsqueeze_546 = None
        mul_205: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_68, unsqueeze_547);  sub_68 = unsqueeze_547 = None
        unsqueeze_548: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg344_1, -1);  arg344_1 = None
        unsqueeze_549: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_548, -1);  unsqueeze_548 = None
        mul_206: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_205, unsqueeze_549);  mul_205 = unsqueeze_549 = None
        unsqueeze_550: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg345_1, -1);  arg345_1 = None
        unsqueeze_551: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_550, -1);  unsqueeze_550 = None
        add_158: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_206, unsqueeze_551);  mul_206 = unsqueeze_551 = None
        convert_element_type_206: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_158, torch.float16);  add_158 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_65: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_206);  convert_element_type_206 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_69: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_65, arg346_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_65 = arg346_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_207: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg347_1, torch.float32);  arg347_1 = None
        unsqueeze_552: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_207, -1);  convert_element_type_207 = None
        unsqueeze_553: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_552, -1);  unsqueeze_552 = None
        sub_69: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_69, unsqueeze_553);  convolution_69 = unsqueeze_553 = None
        convert_element_type_208: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg348_1, torch.float32);  arg348_1 = None
        add_159: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_208, 1e-05);  convert_element_type_208 = None
        sqrt_69: "f32[1024]" = torch.ops.aten.sqrt.default(add_159);  add_159 = None
        reciprocal_69: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_69);  sqrt_69 = None
        mul_207: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_69, 1);  reciprocal_69 = None
        unsqueeze_554: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_207, -1);  mul_207 = None
        unsqueeze_555: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_554, -1);  unsqueeze_554 = None
        mul_208: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_69, unsqueeze_555);  sub_69 = unsqueeze_555 = None
        unsqueeze_556: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg349_1, -1);  arg349_1 = None
        unsqueeze_557: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_556, -1);  unsqueeze_556 = None
        mul_209: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_208, unsqueeze_557);  mul_208 = unsqueeze_557 = None
        unsqueeze_558: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg350_1, -1);  arg350_1 = None
        unsqueeze_559: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_558, -1);  unsqueeze_558 = None
        add_160: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_209, unsqueeze_559);  mul_209 = unsqueeze_559 = None
        convert_element_type_209: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_160, torch.float16);  add_160 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_161: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_209, relu_63);  convert_element_type_209 = relu_63 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_66: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_161);  add_161 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_70: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_66, arg351_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg351_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_210: "f32[256]" = torch.ops.prims.convert_element_type.default(arg352_1, torch.float32);  arg352_1 = None
        unsqueeze_560: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_210, -1);  convert_element_type_210 = None
        unsqueeze_561: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_560, -1);  unsqueeze_560 = None
        sub_70: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_70, unsqueeze_561);  convolution_70 = unsqueeze_561 = None
        convert_element_type_211: "f32[256]" = torch.ops.prims.convert_element_type.default(arg353_1, torch.float32);  arg353_1 = None
        add_162: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_211, 1e-05);  convert_element_type_211 = None
        sqrt_70: "f32[256]" = torch.ops.aten.sqrt.default(add_162);  add_162 = None
        reciprocal_70: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_70);  sqrt_70 = None
        mul_210: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_70, 1);  reciprocal_70 = None
        unsqueeze_562: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_210, -1);  mul_210 = None
        unsqueeze_563: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_562, -1);  unsqueeze_562 = None
        mul_211: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_70, unsqueeze_563);  sub_70 = unsqueeze_563 = None
        unsqueeze_564: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg354_1, -1);  arg354_1 = None
        unsqueeze_565: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_564, -1);  unsqueeze_564 = None
        mul_212: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_211, unsqueeze_565);  mul_211 = unsqueeze_565 = None
        unsqueeze_566: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg355_1, -1);  arg355_1 = None
        unsqueeze_567: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_566, -1);  unsqueeze_566 = None
        add_163: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_212, unsqueeze_567);  mul_212 = unsqueeze_567 = None
        convert_element_type_212: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_163, torch.float16);  add_163 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_67: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_212);  convert_element_type_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_71: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_67, arg356_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_67 = arg356_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_213: "f32[256]" = torch.ops.prims.convert_element_type.default(arg357_1, torch.float32);  arg357_1 = None
        unsqueeze_568: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_213, -1);  convert_element_type_213 = None
        unsqueeze_569: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_568, -1);  unsqueeze_568 = None
        sub_71: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_71, unsqueeze_569);  convolution_71 = unsqueeze_569 = None
        convert_element_type_214: "f32[256]" = torch.ops.prims.convert_element_type.default(arg358_1, torch.float32);  arg358_1 = None
        add_164: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_214, 1e-05);  convert_element_type_214 = None
        sqrt_71: "f32[256]" = torch.ops.aten.sqrt.default(add_164);  add_164 = None
        reciprocal_71: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_71);  sqrt_71 = None
        mul_213: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_71, 1);  reciprocal_71 = None
        unsqueeze_570: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_213, -1);  mul_213 = None
        unsqueeze_571: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_570, -1);  unsqueeze_570 = None
        mul_214: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_71, unsqueeze_571);  sub_71 = unsqueeze_571 = None
        unsqueeze_572: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg359_1, -1);  arg359_1 = None
        unsqueeze_573: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_572, -1);  unsqueeze_572 = None
        mul_215: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_214, unsqueeze_573);  mul_214 = unsqueeze_573 = None
        unsqueeze_574: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg360_1, -1);  arg360_1 = None
        unsqueeze_575: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_574, -1);  unsqueeze_574 = None
        add_165: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_215, unsqueeze_575);  mul_215 = unsqueeze_575 = None
        convert_element_type_215: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_165, torch.float16);  add_165 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_68: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_215);  convert_element_type_215 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_72: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_68, arg361_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_68 = arg361_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_216: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg362_1, torch.float32);  arg362_1 = None
        unsqueeze_576: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_216, -1);  convert_element_type_216 = None
        unsqueeze_577: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_576, -1);  unsqueeze_576 = None
        sub_72: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_72, unsqueeze_577);  convolution_72 = unsqueeze_577 = None
        convert_element_type_217: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg363_1, torch.float32);  arg363_1 = None
        add_166: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_217, 1e-05);  convert_element_type_217 = None
        sqrt_72: "f32[1024]" = torch.ops.aten.sqrt.default(add_166);  add_166 = None
        reciprocal_72: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_72);  sqrt_72 = None
        mul_216: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_72, 1);  reciprocal_72 = None
        unsqueeze_578: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_216, -1);  mul_216 = None
        unsqueeze_579: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_578, -1);  unsqueeze_578 = None
        mul_217: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_72, unsqueeze_579);  sub_72 = unsqueeze_579 = None
        unsqueeze_580: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg364_1, -1);  arg364_1 = None
        unsqueeze_581: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_580, -1);  unsqueeze_580 = None
        mul_218: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_217, unsqueeze_581);  mul_217 = unsqueeze_581 = None
        unsqueeze_582: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg365_1, -1);  arg365_1 = None
        unsqueeze_583: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_582, -1);  unsqueeze_582 = None
        add_167: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_218, unsqueeze_583);  mul_218 = unsqueeze_583 = None
        convert_element_type_218: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_167, torch.float16);  add_167 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_168: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_218, relu_66);  convert_element_type_218 = relu_66 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_69: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_168);  add_168 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_73: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_69, arg366_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg366_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_219: "f32[256]" = torch.ops.prims.convert_element_type.default(arg367_1, torch.float32);  arg367_1 = None
        unsqueeze_584: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_219, -1);  convert_element_type_219 = None
        unsqueeze_585: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_584, -1);  unsqueeze_584 = None
        sub_73: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_73, unsqueeze_585);  convolution_73 = unsqueeze_585 = None
        convert_element_type_220: "f32[256]" = torch.ops.prims.convert_element_type.default(arg368_1, torch.float32);  arg368_1 = None
        add_169: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_220, 1e-05);  convert_element_type_220 = None
        sqrt_73: "f32[256]" = torch.ops.aten.sqrt.default(add_169);  add_169 = None
        reciprocal_73: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_73);  sqrt_73 = None
        mul_219: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_73, 1);  reciprocal_73 = None
        unsqueeze_586: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_219, -1);  mul_219 = None
        unsqueeze_587: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_586, -1);  unsqueeze_586 = None
        mul_220: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_73, unsqueeze_587);  sub_73 = unsqueeze_587 = None
        unsqueeze_588: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg369_1, -1);  arg369_1 = None
        unsqueeze_589: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_588, -1);  unsqueeze_588 = None
        mul_221: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_220, unsqueeze_589);  mul_220 = unsqueeze_589 = None
        unsqueeze_590: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg370_1, -1);  arg370_1 = None
        unsqueeze_591: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_590, -1);  unsqueeze_590 = None
        add_170: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_221, unsqueeze_591);  mul_221 = unsqueeze_591 = None
        convert_element_type_221: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_170, torch.float16);  add_170 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_70: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_221);  convert_element_type_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_74: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_70, arg371_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_70 = arg371_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_222: "f32[256]" = torch.ops.prims.convert_element_type.default(arg372_1, torch.float32);  arg372_1 = None
        unsqueeze_592: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_222, -1);  convert_element_type_222 = None
        unsqueeze_593: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_592, -1);  unsqueeze_592 = None
        sub_74: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_74, unsqueeze_593);  convolution_74 = unsqueeze_593 = None
        convert_element_type_223: "f32[256]" = torch.ops.prims.convert_element_type.default(arg373_1, torch.float32);  arg373_1 = None
        add_171: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_223, 1e-05);  convert_element_type_223 = None
        sqrt_74: "f32[256]" = torch.ops.aten.sqrt.default(add_171);  add_171 = None
        reciprocal_74: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_74);  sqrt_74 = None
        mul_222: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_74, 1);  reciprocal_74 = None
        unsqueeze_594: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_222, -1);  mul_222 = None
        unsqueeze_595: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_594, -1);  unsqueeze_594 = None
        mul_223: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_74, unsqueeze_595);  sub_74 = unsqueeze_595 = None
        unsqueeze_596: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg374_1, -1);  arg374_1 = None
        unsqueeze_597: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_596, -1);  unsqueeze_596 = None
        mul_224: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_223, unsqueeze_597);  mul_223 = unsqueeze_597 = None
        unsqueeze_598: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg375_1, -1);  arg375_1 = None
        unsqueeze_599: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_598, -1);  unsqueeze_598 = None
        add_172: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_224, unsqueeze_599);  mul_224 = unsqueeze_599 = None
        convert_element_type_224: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_172, torch.float16);  add_172 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_71: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_224);  convert_element_type_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_75: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_71, arg376_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_71 = arg376_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_225: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg377_1, torch.float32);  arg377_1 = None
        unsqueeze_600: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_225, -1);  convert_element_type_225 = None
        unsqueeze_601: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_600, -1);  unsqueeze_600 = None
        sub_75: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_75, unsqueeze_601);  convolution_75 = unsqueeze_601 = None
        convert_element_type_226: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg378_1, torch.float32);  arg378_1 = None
        add_173: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_226, 1e-05);  convert_element_type_226 = None
        sqrt_75: "f32[1024]" = torch.ops.aten.sqrt.default(add_173);  add_173 = None
        reciprocal_75: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_75);  sqrt_75 = None
        mul_225: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_75, 1);  reciprocal_75 = None
        unsqueeze_602: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_225, -1);  mul_225 = None
        unsqueeze_603: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_602, -1);  unsqueeze_602 = None
        mul_226: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_75, unsqueeze_603);  sub_75 = unsqueeze_603 = None
        unsqueeze_604: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg379_1, -1);  arg379_1 = None
        unsqueeze_605: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_604, -1);  unsqueeze_604 = None
        mul_227: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_226, unsqueeze_605);  mul_226 = unsqueeze_605 = None
        unsqueeze_606: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg380_1, -1);  arg380_1 = None
        unsqueeze_607: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_606, -1);  unsqueeze_606 = None
        add_174: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_227, unsqueeze_607);  mul_227 = unsqueeze_607 = None
        convert_element_type_227: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_174, torch.float16);  add_174 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_175: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_227, relu_69);  convert_element_type_227 = relu_69 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_72: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_175);  add_175 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_76: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_72, arg381_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg381_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_228: "f32[256]" = torch.ops.prims.convert_element_type.default(arg382_1, torch.float32);  arg382_1 = None
        unsqueeze_608: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_228, -1);  convert_element_type_228 = None
        unsqueeze_609: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_608, -1);  unsqueeze_608 = None
        sub_76: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_76, unsqueeze_609);  convolution_76 = unsqueeze_609 = None
        convert_element_type_229: "f32[256]" = torch.ops.prims.convert_element_type.default(arg383_1, torch.float32);  arg383_1 = None
        add_176: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_229, 1e-05);  convert_element_type_229 = None
        sqrt_76: "f32[256]" = torch.ops.aten.sqrt.default(add_176);  add_176 = None
        reciprocal_76: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_76);  sqrt_76 = None
        mul_228: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_76, 1);  reciprocal_76 = None
        unsqueeze_610: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_228, -1);  mul_228 = None
        unsqueeze_611: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_610, -1);  unsqueeze_610 = None
        mul_229: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_76, unsqueeze_611);  sub_76 = unsqueeze_611 = None
        unsqueeze_612: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg384_1, -1);  arg384_1 = None
        unsqueeze_613: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_612, -1);  unsqueeze_612 = None
        mul_230: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_229, unsqueeze_613);  mul_229 = unsqueeze_613 = None
        unsqueeze_614: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg385_1, -1);  arg385_1 = None
        unsqueeze_615: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_614, -1);  unsqueeze_614 = None
        add_177: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_230, unsqueeze_615);  mul_230 = unsqueeze_615 = None
        convert_element_type_230: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_177, torch.float16);  add_177 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_73: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_230);  convert_element_type_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_77: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_73, arg386_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_73 = arg386_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_231: "f32[256]" = torch.ops.prims.convert_element_type.default(arg387_1, torch.float32);  arg387_1 = None
        unsqueeze_616: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_231, -1);  convert_element_type_231 = None
        unsqueeze_617: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_616, -1);  unsqueeze_616 = None
        sub_77: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_77, unsqueeze_617);  convolution_77 = unsqueeze_617 = None
        convert_element_type_232: "f32[256]" = torch.ops.prims.convert_element_type.default(arg388_1, torch.float32);  arg388_1 = None
        add_178: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_232, 1e-05);  convert_element_type_232 = None
        sqrt_77: "f32[256]" = torch.ops.aten.sqrt.default(add_178);  add_178 = None
        reciprocal_77: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_77);  sqrt_77 = None
        mul_231: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_77, 1);  reciprocal_77 = None
        unsqueeze_618: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_231, -1);  mul_231 = None
        unsqueeze_619: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_618, -1);  unsqueeze_618 = None
        mul_232: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_77, unsqueeze_619);  sub_77 = unsqueeze_619 = None
        unsqueeze_620: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg389_1, -1);  arg389_1 = None
        unsqueeze_621: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_620, -1);  unsqueeze_620 = None
        mul_233: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_232, unsqueeze_621);  mul_232 = unsqueeze_621 = None
        unsqueeze_622: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg390_1, -1);  arg390_1 = None
        unsqueeze_623: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_622, -1);  unsqueeze_622 = None
        add_179: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_233, unsqueeze_623);  mul_233 = unsqueeze_623 = None
        convert_element_type_233: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_179, torch.float16);  add_179 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_74: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_233);  convert_element_type_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_78: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_74, arg391_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_74 = arg391_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_234: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg392_1, torch.float32);  arg392_1 = None
        unsqueeze_624: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_234, -1);  convert_element_type_234 = None
        unsqueeze_625: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_624, -1);  unsqueeze_624 = None
        sub_78: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_78, unsqueeze_625);  convolution_78 = unsqueeze_625 = None
        convert_element_type_235: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg393_1, torch.float32);  arg393_1 = None
        add_180: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_235, 1e-05);  convert_element_type_235 = None
        sqrt_78: "f32[1024]" = torch.ops.aten.sqrt.default(add_180);  add_180 = None
        reciprocal_78: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_78);  sqrt_78 = None
        mul_234: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_78, 1);  reciprocal_78 = None
        unsqueeze_626: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_234, -1);  mul_234 = None
        unsqueeze_627: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_626, -1);  unsqueeze_626 = None
        mul_235: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_78, unsqueeze_627);  sub_78 = unsqueeze_627 = None
        unsqueeze_628: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg394_1, -1);  arg394_1 = None
        unsqueeze_629: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_628, -1);  unsqueeze_628 = None
        mul_236: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_235, unsqueeze_629);  mul_235 = unsqueeze_629 = None
        unsqueeze_630: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg395_1, -1);  arg395_1 = None
        unsqueeze_631: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_630, -1);  unsqueeze_630 = None
        add_181: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_236, unsqueeze_631);  mul_236 = unsqueeze_631 = None
        convert_element_type_236: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_181, torch.float16);  add_181 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_182: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_236, relu_72);  convert_element_type_236 = relu_72 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_75: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_182);  add_182 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_79: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_75, arg396_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg396_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_237: "f32[256]" = torch.ops.prims.convert_element_type.default(arg397_1, torch.float32);  arg397_1 = None
        unsqueeze_632: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_237, -1);  convert_element_type_237 = None
        unsqueeze_633: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_632, -1);  unsqueeze_632 = None
        sub_79: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_79, unsqueeze_633);  convolution_79 = unsqueeze_633 = None
        convert_element_type_238: "f32[256]" = torch.ops.prims.convert_element_type.default(arg398_1, torch.float32);  arg398_1 = None
        add_183: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_238, 1e-05);  convert_element_type_238 = None
        sqrt_79: "f32[256]" = torch.ops.aten.sqrt.default(add_183);  add_183 = None
        reciprocal_79: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_79);  sqrt_79 = None
        mul_237: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_79, 1);  reciprocal_79 = None
        unsqueeze_634: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_237, -1);  mul_237 = None
        unsqueeze_635: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_634, -1);  unsqueeze_634 = None
        mul_238: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_79, unsqueeze_635);  sub_79 = unsqueeze_635 = None
        unsqueeze_636: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg399_1, -1);  arg399_1 = None
        unsqueeze_637: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_636, -1);  unsqueeze_636 = None
        mul_239: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_238, unsqueeze_637);  mul_238 = unsqueeze_637 = None
        unsqueeze_638: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg400_1, -1);  arg400_1 = None
        unsqueeze_639: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_638, -1);  unsqueeze_638 = None
        add_184: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_239, unsqueeze_639);  mul_239 = unsqueeze_639 = None
        convert_element_type_239: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_184, torch.float16);  add_184 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_76: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_239);  convert_element_type_239 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_80: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_76, arg401_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_76 = arg401_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_240: "f32[256]" = torch.ops.prims.convert_element_type.default(arg402_1, torch.float32);  arg402_1 = None
        unsqueeze_640: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_240, -1);  convert_element_type_240 = None
        unsqueeze_641: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_640, -1);  unsqueeze_640 = None
        sub_80: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_80, unsqueeze_641);  convolution_80 = unsqueeze_641 = None
        convert_element_type_241: "f32[256]" = torch.ops.prims.convert_element_type.default(arg403_1, torch.float32);  arg403_1 = None
        add_185: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_241, 1e-05);  convert_element_type_241 = None
        sqrt_80: "f32[256]" = torch.ops.aten.sqrt.default(add_185);  add_185 = None
        reciprocal_80: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_80);  sqrt_80 = None
        mul_240: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_80, 1);  reciprocal_80 = None
        unsqueeze_642: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_240, -1);  mul_240 = None
        unsqueeze_643: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_642, -1);  unsqueeze_642 = None
        mul_241: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_80, unsqueeze_643);  sub_80 = unsqueeze_643 = None
        unsqueeze_644: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg404_1, -1);  arg404_1 = None
        unsqueeze_645: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_644, -1);  unsqueeze_644 = None
        mul_242: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_241, unsqueeze_645);  mul_241 = unsqueeze_645 = None
        unsqueeze_646: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg405_1, -1);  arg405_1 = None
        unsqueeze_647: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_646, -1);  unsqueeze_646 = None
        add_186: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_242, unsqueeze_647);  mul_242 = unsqueeze_647 = None
        convert_element_type_242: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_186, torch.float16);  add_186 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_77: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_242);  convert_element_type_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_81: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_77, arg406_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_77 = arg406_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_243: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg407_1, torch.float32);  arg407_1 = None
        unsqueeze_648: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_243, -1);  convert_element_type_243 = None
        unsqueeze_649: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_648, -1);  unsqueeze_648 = None
        sub_81: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_81, unsqueeze_649);  convolution_81 = unsqueeze_649 = None
        convert_element_type_244: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg408_1, torch.float32);  arg408_1 = None
        add_187: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_244, 1e-05);  convert_element_type_244 = None
        sqrt_81: "f32[1024]" = torch.ops.aten.sqrt.default(add_187);  add_187 = None
        reciprocal_81: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_81);  sqrt_81 = None
        mul_243: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_81, 1);  reciprocal_81 = None
        unsqueeze_650: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_243, -1);  mul_243 = None
        unsqueeze_651: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_650, -1);  unsqueeze_650 = None
        mul_244: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_81, unsqueeze_651);  sub_81 = unsqueeze_651 = None
        unsqueeze_652: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg409_1, -1);  arg409_1 = None
        unsqueeze_653: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_652, -1);  unsqueeze_652 = None
        mul_245: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_244, unsqueeze_653);  mul_244 = unsqueeze_653 = None
        unsqueeze_654: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg410_1, -1);  arg410_1 = None
        unsqueeze_655: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_654, -1);  unsqueeze_654 = None
        add_188: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_245, unsqueeze_655);  mul_245 = unsqueeze_655 = None
        convert_element_type_245: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_188, torch.float16);  add_188 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_189: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_245, relu_75);  convert_element_type_245 = relu_75 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_78: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_189);  add_189 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_82: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_78, arg411_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg411_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_246: "f32[256]" = torch.ops.prims.convert_element_type.default(arg412_1, torch.float32);  arg412_1 = None
        unsqueeze_656: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_246, -1);  convert_element_type_246 = None
        unsqueeze_657: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_656, -1);  unsqueeze_656 = None
        sub_82: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_82, unsqueeze_657);  convolution_82 = unsqueeze_657 = None
        convert_element_type_247: "f32[256]" = torch.ops.prims.convert_element_type.default(arg413_1, torch.float32);  arg413_1 = None
        add_190: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_247, 1e-05);  convert_element_type_247 = None
        sqrt_82: "f32[256]" = torch.ops.aten.sqrt.default(add_190);  add_190 = None
        reciprocal_82: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_82);  sqrt_82 = None
        mul_246: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_82, 1);  reciprocal_82 = None
        unsqueeze_658: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_246, -1);  mul_246 = None
        unsqueeze_659: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_658, -1);  unsqueeze_658 = None
        mul_247: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_82, unsqueeze_659);  sub_82 = unsqueeze_659 = None
        unsqueeze_660: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg414_1, -1);  arg414_1 = None
        unsqueeze_661: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_660, -1);  unsqueeze_660 = None
        mul_248: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_247, unsqueeze_661);  mul_247 = unsqueeze_661 = None
        unsqueeze_662: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg415_1, -1);  arg415_1 = None
        unsqueeze_663: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_662, -1);  unsqueeze_662 = None
        add_191: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_248, unsqueeze_663);  mul_248 = unsqueeze_663 = None
        convert_element_type_248: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_191, torch.float16);  add_191 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_79: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_248);  convert_element_type_248 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_83: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_79, arg416_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_79 = arg416_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_249: "f32[256]" = torch.ops.prims.convert_element_type.default(arg417_1, torch.float32);  arg417_1 = None
        unsqueeze_664: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_249, -1);  convert_element_type_249 = None
        unsqueeze_665: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_664, -1);  unsqueeze_664 = None
        sub_83: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_83, unsqueeze_665);  convolution_83 = unsqueeze_665 = None
        convert_element_type_250: "f32[256]" = torch.ops.prims.convert_element_type.default(arg418_1, torch.float32);  arg418_1 = None
        add_192: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_250, 1e-05);  convert_element_type_250 = None
        sqrt_83: "f32[256]" = torch.ops.aten.sqrt.default(add_192);  add_192 = None
        reciprocal_83: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_83);  sqrt_83 = None
        mul_249: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_83, 1);  reciprocal_83 = None
        unsqueeze_666: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_249, -1);  mul_249 = None
        unsqueeze_667: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_666, -1);  unsqueeze_666 = None
        mul_250: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_83, unsqueeze_667);  sub_83 = unsqueeze_667 = None
        unsqueeze_668: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg419_1, -1);  arg419_1 = None
        unsqueeze_669: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_668, -1);  unsqueeze_668 = None
        mul_251: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_250, unsqueeze_669);  mul_250 = unsqueeze_669 = None
        unsqueeze_670: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg420_1, -1);  arg420_1 = None
        unsqueeze_671: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_670, -1);  unsqueeze_670 = None
        add_193: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_251, unsqueeze_671);  mul_251 = unsqueeze_671 = None
        convert_element_type_251: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_193, torch.float16);  add_193 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_80: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_251);  convert_element_type_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_84: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_80, arg421_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_80 = arg421_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_252: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg422_1, torch.float32);  arg422_1 = None
        unsqueeze_672: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_252, -1);  convert_element_type_252 = None
        unsqueeze_673: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_672, -1);  unsqueeze_672 = None
        sub_84: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_84, unsqueeze_673);  convolution_84 = unsqueeze_673 = None
        convert_element_type_253: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg423_1, torch.float32);  arg423_1 = None
        add_194: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_253, 1e-05);  convert_element_type_253 = None
        sqrt_84: "f32[1024]" = torch.ops.aten.sqrt.default(add_194);  add_194 = None
        reciprocal_84: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_84);  sqrt_84 = None
        mul_252: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_84, 1);  reciprocal_84 = None
        unsqueeze_674: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_252, -1);  mul_252 = None
        unsqueeze_675: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_674, -1);  unsqueeze_674 = None
        mul_253: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_84, unsqueeze_675);  sub_84 = unsqueeze_675 = None
        unsqueeze_676: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg424_1, -1);  arg424_1 = None
        unsqueeze_677: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_676, -1);  unsqueeze_676 = None
        mul_254: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_253, unsqueeze_677);  mul_253 = unsqueeze_677 = None
        unsqueeze_678: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg425_1, -1);  arg425_1 = None
        unsqueeze_679: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_678, -1);  unsqueeze_678 = None
        add_195: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_254, unsqueeze_679);  mul_254 = unsqueeze_679 = None
        convert_element_type_254: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_195, torch.float16);  add_195 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_196: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_254, relu_78);  convert_element_type_254 = relu_78 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_81: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_196);  add_196 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_85: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_81, arg426_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg426_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_255: "f32[256]" = torch.ops.prims.convert_element_type.default(arg427_1, torch.float32);  arg427_1 = None
        unsqueeze_680: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_255, -1);  convert_element_type_255 = None
        unsqueeze_681: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_680, -1);  unsqueeze_680 = None
        sub_85: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_85, unsqueeze_681);  convolution_85 = unsqueeze_681 = None
        convert_element_type_256: "f32[256]" = torch.ops.prims.convert_element_type.default(arg428_1, torch.float32);  arg428_1 = None
        add_197: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_256, 1e-05);  convert_element_type_256 = None
        sqrt_85: "f32[256]" = torch.ops.aten.sqrt.default(add_197);  add_197 = None
        reciprocal_85: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_85);  sqrt_85 = None
        mul_255: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_85, 1);  reciprocal_85 = None
        unsqueeze_682: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_255, -1);  mul_255 = None
        unsqueeze_683: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_682, -1);  unsqueeze_682 = None
        mul_256: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_85, unsqueeze_683);  sub_85 = unsqueeze_683 = None
        unsqueeze_684: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg429_1, -1);  arg429_1 = None
        unsqueeze_685: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_684, -1);  unsqueeze_684 = None
        mul_257: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_256, unsqueeze_685);  mul_256 = unsqueeze_685 = None
        unsqueeze_686: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg430_1, -1);  arg430_1 = None
        unsqueeze_687: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_686, -1);  unsqueeze_686 = None
        add_198: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_257, unsqueeze_687);  mul_257 = unsqueeze_687 = None
        convert_element_type_257: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_198, torch.float16);  add_198 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_82: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_257);  convert_element_type_257 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_86: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_82, arg431_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_82 = arg431_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_258: "f32[256]" = torch.ops.prims.convert_element_type.default(arg432_1, torch.float32);  arg432_1 = None
        unsqueeze_688: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_258, -1);  convert_element_type_258 = None
        unsqueeze_689: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_688, -1);  unsqueeze_688 = None
        sub_86: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_86, unsqueeze_689);  convolution_86 = unsqueeze_689 = None
        convert_element_type_259: "f32[256]" = torch.ops.prims.convert_element_type.default(arg433_1, torch.float32);  arg433_1 = None
        add_199: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_259, 1e-05);  convert_element_type_259 = None
        sqrt_86: "f32[256]" = torch.ops.aten.sqrt.default(add_199);  add_199 = None
        reciprocal_86: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_86);  sqrt_86 = None
        mul_258: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_86, 1);  reciprocal_86 = None
        unsqueeze_690: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_258, -1);  mul_258 = None
        unsqueeze_691: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_690, -1);  unsqueeze_690 = None
        mul_259: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_86, unsqueeze_691);  sub_86 = unsqueeze_691 = None
        unsqueeze_692: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg434_1, -1);  arg434_1 = None
        unsqueeze_693: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_692, -1);  unsqueeze_692 = None
        mul_260: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_259, unsqueeze_693);  mul_259 = unsqueeze_693 = None
        unsqueeze_694: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg435_1, -1);  arg435_1 = None
        unsqueeze_695: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_694, -1);  unsqueeze_694 = None
        add_200: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_260, unsqueeze_695);  mul_260 = unsqueeze_695 = None
        convert_element_type_260: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_200, torch.float16);  add_200 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_83: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_260);  convert_element_type_260 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_87: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_83, arg436_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_83 = arg436_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_261: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg437_1, torch.float32);  arg437_1 = None
        unsqueeze_696: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_261, -1);  convert_element_type_261 = None
        unsqueeze_697: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_696, -1);  unsqueeze_696 = None
        sub_87: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_87, unsqueeze_697);  convolution_87 = unsqueeze_697 = None
        convert_element_type_262: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg438_1, torch.float32);  arg438_1 = None
        add_201: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_262, 1e-05);  convert_element_type_262 = None
        sqrt_87: "f32[1024]" = torch.ops.aten.sqrt.default(add_201);  add_201 = None
        reciprocal_87: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_87);  sqrt_87 = None
        mul_261: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_87, 1);  reciprocal_87 = None
        unsqueeze_698: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_261, -1);  mul_261 = None
        unsqueeze_699: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_698, -1);  unsqueeze_698 = None
        mul_262: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_87, unsqueeze_699);  sub_87 = unsqueeze_699 = None
        unsqueeze_700: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg439_1, -1);  arg439_1 = None
        unsqueeze_701: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_700, -1);  unsqueeze_700 = None
        mul_263: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_262, unsqueeze_701);  mul_262 = unsqueeze_701 = None
        unsqueeze_702: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg440_1, -1);  arg440_1 = None
        unsqueeze_703: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_702, -1);  unsqueeze_702 = None
        add_202: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_263, unsqueeze_703);  mul_263 = unsqueeze_703 = None
        convert_element_type_263: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_202, torch.float16);  add_202 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_203: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_263, relu_81);  convert_element_type_263 = relu_81 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_84: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_203);  add_203 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_88: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_84, arg441_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg441_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_264: "f32[256]" = torch.ops.prims.convert_element_type.default(arg442_1, torch.float32);  arg442_1 = None
        unsqueeze_704: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_264, -1);  convert_element_type_264 = None
        unsqueeze_705: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_704, -1);  unsqueeze_704 = None
        sub_88: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_88, unsqueeze_705);  convolution_88 = unsqueeze_705 = None
        convert_element_type_265: "f32[256]" = torch.ops.prims.convert_element_type.default(arg443_1, torch.float32);  arg443_1 = None
        add_204: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_265, 1e-05);  convert_element_type_265 = None
        sqrt_88: "f32[256]" = torch.ops.aten.sqrt.default(add_204);  add_204 = None
        reciprocal_88: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_88);  sqrt_88 = None
        mul_264: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_88, 1);  reciprocal_88 = None
        unsqueeze_706: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_264, -1);  mul_264 = None
        unsqueeze_707: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_706, -1);  unsqueeze_706 = None
        mul_265: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_88, unsqueeze_707);  sub_88 = unsqueeze_707 = None
        unsqueeze_708: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg444_1, -1);  arg444_1 = None
        unsqueeze_709: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_708, -1);  unsqueeze_708 = None
        mul_266: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_265, unsqueeze_709);  mul_265 = unsqueeze_709 = None
        unsqueeze_710: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg445_1, -1);  arg445_1 = None
        unsqueeze_711: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_710, -1);  unsqueeze_710 = None
        add_205: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_266, unsqueeze_711);  mul_266 = unsqueeze_711 = None
        convert_element_type_266: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_205, torch.float16);  add_205 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_85: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_266);  convert_element_type_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_89: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_85, arg446_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_85 = arg446_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_267: "f32[256]" = torch.ops.prims.convert_element_type.default(arg447_1, torch.float32);  arg447_1 = None
        unsqueeze_712: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_267, -1);  convert_element_type_267 = None
        unsqueeze_713: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_712, -1);  unsqueeze_712 = None
        sub_89: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_89, unsqueeze_713);  convolution_89 = unsqueeze_713 = None
        convert_element_type_268: "f32[256]" = torch.ops.prims.convert_element_type.default(arg448_1, torch.float32);  arg448_1 = None
        add_206: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_268, 1e-05);  convert_element_type_268 = None
        sqrt_89: "f32[256]" = torch.ops.aten.sqrt.default(add_206);  add_206 = None
        reciprocal_89: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_89);  sqrt_89 = None
        mul_267: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_89, 1);  reciprocal_89 = None
        unsqueeze_714: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_267, -1);  mul_267 = None
        unsqueeze_715: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_714, -1);  unsqueeze_714 = None
        mul_268: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_89, unsqueeze_715);  sub_89 = unsqueeze_715 = None
        unsqueeze_716: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg449_1, -1);  arg449_1 = None
        unsqueeze_717: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_716, -1);  unsqueeze_716 = None
        mul_269: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_268, unsqueeze_717);  mul_268 = unsqueeze_717 = None
        unsqueeze_718: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg450_1, -1);  arg450_1 = None
        unsqueeze_719: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_718, -1);  unsqueeze_718 = None
        add_207: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_269, unsqueeze_719);  mul_269 = unsqueeze_719 = None
        convert_element_type_269: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_207, torch.float16);  add_207 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_86: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_269);  convert_element_type_269 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_90: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_86, arg451_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_86 = arg451_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_270: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg452_1, torch.float32);  arg452_1 = None
        unsqueeze_720: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_270, -1);  convert_element_type_270 = None
        unsqueeze_721: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_720, -1);  unsqueeze_720 = None
        sub_90: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_90, unsqueeze_721);  convolution_90 = unsqueeze_721 = None
        convert_element_type_271: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg453_1, torch.float32);  arg453_1 = None
        add_208: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_271, 1e-05);  convert_element_type_271 = None
        sqrt_90: "f32[1024]" = torch.ops.aten.sqrt.default(add_208);  add_208 = None
        reciprocal_90: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_90);  sqrt_90 = None
        mul_270: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_90, 1);  reciprocal_90 = None
        unsqueeze_722: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_270, -1);  mul_270 = None
        unsqueeze_723: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_722, -1);  unsqueeze_722 = None
        mul_271: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_90, unsqueeze_723);  sub_90 = unsqueeze_723 = None
        unsqueeze_724: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg454_1, -1);  arg454_1 = None
        unsqueeze_725: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_724, -1);  unsqueeze_724 = None
        mul_272: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_271, unsqueeze_725);  mul_271 = unsqueeze_725 = None
        unsqueeze_726: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg455_1, -1);  arg455_1 = None
        unsqueeze_727: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_726, -1);  unsqueeze_726 = None
        add_209: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_272, unsqueeze_727);  mul_272 = unsqueeze_727 = None
        convert_element_type_272: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_209, torch.float16);  add_209 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_210: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_272, relu_84);  convert_element_type_272 = relu_84 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_87: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_210);  add_210 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_91: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_87, arg456_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg456_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_273: "f32[256]" = torch.ops.prims.convert_element_type.default(arg457_1, torch.float32);  arg457_1 = None
        unsqueeze_728: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_273, -1);  convert_element_type_273 = None
        unsqueeze_729: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_728, -1);  unsqueeze_728 = None
        sub_91: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_91, unsqueeze_729);  convolution_91 = unsqueeze_729 = None
        convert_element_type_274: "f32[256]" = torch.ops.prims.convert_element_type.default(arg458_1, torch.float32);  arg458_1 = None
        add_211: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_274, 1e-05);  convert_element_type_274 = None
        sqrt_91: "f32[256]" = torch.ops.aten.sqrt.default(add_211);  add_211 = None
        reciprocal_91: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_91);  sqrt_91 = None
        mul_273: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_91, 1);  reciprocal_91 = None
        unsqueeze_730: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_273, -1);  mul_273 = None
        unsqueeze_731: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_730, -1);  unsqueeze_730 = None
        mul_274: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_91, unsqueeze_731);  sub_91 = unsqueeze_731 = None
        unsqueeze_732: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg459_1, -1);  arg459_1 = None
        unsqueeze_733: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_732, -1);  unsqueeze_732 = None
        mul_275: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_274, unsqueeze_733);  mul_274 = unsqueeze_733 = None
        unsqueeze_734: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg460_1, -1);  arg460_1 = None
        unsqueeze_735: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_734, -1);  unsqueeze_734 = None
        add_212: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_275, unsqueeze_735);  mul_275 = unsqueeze_735 = None
        convert_element_type_275: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_212, torch.float16);  add_212 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_88: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_275);  convert_element_type_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_92: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_88, arg461_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_88 = arg461_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_276: "f32[256]" = torch.ops.prims.convert_element_type.default(arg462_1, torch.float32);  arg462_1 = None
        unsqueeze_736: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_276, -1);  convert_element_type_276 = None
        unsqueeze_737: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_736, -1);  unsqueeze_736 = None
        sub_92: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_92, unsqueeze_737);  convolution_92 = unsqueeze_737 = None
        convert_element_type_277: "f32[256]" = torch.ops.prims.convert_element_type.default(arg463_1, torch.float32);  arg463_1 = None
        add_213: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_277, 1e-05);  convert_element_type_277 = None
        sqrt_92: "f32[256]" = torch.ops.aten.sqrt.default(add_213);  add_213 = None
        reciprocal_92: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_92);  sqrt_92 = None
        mul_276: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_92, 1);  reciprocal_92 = None
        unsqueeze_738: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_276, -1);  mul_276 = None
        unsqueeze_739: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_738, -1);  unsqueeze_738 = None
        mul_277: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_92, unsqueeze_739);  sub_92 = unsqueeze_739 = None
        unsqueeze_740: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg464_1, -1);  arg464_1 = None
        unsqueeze_741: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_740, -1);  unsqueeze_740 = None
        mul_278: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_277, unsqueeze_741);  mul_277 = unsqueeze_741 = None
        unsqueeze_742: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg465_1, -1);  arg465_1 = None
        unsqueeze_743: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_742, -1);  unsqueeze_742 = None
        add_214: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_278, unsqueeze_743);  mul_278 = unsqueeze_743 = None
        convert_element_type_278: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_214, torch.float16);  add_214 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_89: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_278);  convert_element_type_278 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_93: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_89, arg466_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_89 = arg466_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_279: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg467_1, torch.float32);  arg467_1 = None
        unsqueeze_744: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_279, -1);  convert_element_type_279 = None
        unsqueeze_745: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_744, -1);  unsqueeze_744 = None
        sub_93: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_93, unsqueeze_745);  convolution_93 = unsqueeze_745 = None
        convert_element_type_280: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg468_1, torch.float32);  arg468_1 = None
        add_215: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_280, 1e-05);  convert_element_type_280 = None
        sqrt_93: "f32[1024]" = torch.ops.aten.sqrt.default(add_215);  add_215 = None
        reciprocal_93: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_93);  sqrt_93 = None
        mul_279: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_93, 1);  reciprocal_93 = None
        unsqueeze_746: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_279, -1);  mul_279 = None
        unsqueeze_747: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_746, -1);  unsqueeze_746 = None
        mul_280: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_93, unsqueeze_747);  sub_93 = unsqueeze_747 = None
        unsqueeze_748: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg469_1, -1);  arg469_1 = None
        unsqueeze_749: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_748, -1);  unsqueeze_748 = None
        mul_281: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_280, unsqueeze_749);  mul_280 = unsqueeze_749 = None
        unsqueeze_750: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg470_1, -1);  arg470_1 = None
        unsqueeze_751: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_750, -1);  unsqueeze_750 = None
        add_216: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_281, unsqueeze_751);  mul_281 = unsqueeze_751 = None
        convert_element_type_281: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_216, torch.float16);  add_216 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_217: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_281, relu_87);  convert_element_type_281 = relu_87 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_90: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_217);  add_217 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_94: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_90, arg471_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg471_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_282: "f32[256]" = torch.ops.prims.convert_element_type.default(arg472_1, torch.float32);  arg472_1 = None
        unsqueeze_752: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_282, -1);  convert_element_type_282 = None
        unsqueeze_753: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_752, -1);  unsqueeze_752 = None
        sub_94: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_94, unsqueeze_753);  convolution_94 = unsqueeze_753 = None
        convert_element_type_283: "f32[256]" = torch.ops.prims.convert_element_type.default(arg473_1, torch.float32);  arg473_1 = None
        add_218: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_283, 1e-05);  convert_element_type_283 = None
        sqrt_94: "f32[256]" = torch.ops.aten.sqrt.default(add_218);  add_218 = None
        reciprocal_94: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_94);  sqrt_94 = None
        mul_282: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_94, 1);  reciprocal_94 = None
        unsqueeze_754: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_282, -1);  mul_282 = None
        unsqueeze_755: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_754, -1);  unsqueeze_754 = None
        mul_283: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_94, unsqueeze_755);  sub_94 = unsqueeze_755 = None
        unsqueeze_756: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg474_1, -1);  arg474_1 = None
        unsqueeze_757: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_756, -1);  unsqueeze_756 = None
        mul_284: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_283, unsqueeze_757);  mul_283 = unsqueeze_757 = None
        unsqueeze_758: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg475_1, -1);  arg475_1 = None
        unsqueeze_759: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_758, -1);  unsqueeze_758 = None
        add_219: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_284, unsqueeze_759);  mul_284 = unsqueeze_759 = None
        convert_element_type_284: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_219, torch.float16);  add_219 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_91: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_284);  convert_element_type_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_95: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_91, arg476_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_91 = arg476_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_285: "f32[256]" = torch.ops.prims.convert_element_type.default(arg477_1, torch.float32);  arg477_1 = None
        unsqueeze_760: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_285, -1);  convert_element_type_285 = None
        unsqueeze_761: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_760, -1);  unsqueeze_760 = None
        sub_95: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_95, unsqueeze_761);  convolution_95 = unsqueeze_761 = None
        convert_element_type_286: "f32[256]" = torch.ops.prims.convert_element_type.default(arg478_1, torch.float32);  arg478_1 = None
        add_220: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_286, 1e-05);  convert_element_type_286 = None
        sqrt_95: "f32[256]" = torch.ops.aten.sqrt.default(add_220);  add_220 = None
        reciprocal_95: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_95);  sqrt_95 = None
        mul_285: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_95, 1);  reciprocal_95 = None
        unsqueeze_762: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_285, -1);  mul_285 = None
        unsqueeze_763: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_762, -1);  unsqueeze_762 = None
        mul_286: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_95, unsqueeze_763);  sub_95 = unsqueeze_763 = None
        unsqueeze_764: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg479_1, -1);  arg479_1 = None
        unsqueeze_765: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_764, -1);  unsqueeze_764 = None
        mul_287: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_286, unsqueeze_765);  mul_286 = unsqueeze_765 = None
        unsqueeze_766: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg480_1, -1);  arg480_1 = None
        unsqueeze_767: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_766, -1);  unsqueeze_766 = None
        add_221: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_287, unsqueeze_767);  mul_287 = unsqueeze_767 = None
        convert_element_type_287: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_221, torch.float16);  add_221 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_92: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_287);  convert_element_type_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_96: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_92, arg481_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_92 = arg481_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_288: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg482_1, torch.float32);  arg482_1 = None
        unsqueeze_768: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_288, -1);  convert_element_type_288 = None
        unsqueeze_769: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_768, -1);  unsqueeze_768 = None
        sub_96: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_96, unsqueeze_769);  convolution_96 = unsqueeze_769 = None
        convert_element_type_289: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg483_1, torch.float32);  arg483_1 = None
        add_222: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_289, 1e-05);  convert_element_type_289 = None
        sqrt_96: "f32[1024]" = torch.ops.aten.sqrt.default(add_222);  add_222 = None
        reciprocal_96: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_96);  sqrt_96 = None
        mul_288: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_96, 1);  reciprocal_96 = None
        unsqueeze_770: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_288, -1);  mul_288 = None
        unsqueeze_771: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_770, -1);  unsqueeze_770 = None
        mul_289: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_96, unsqueeze_771);  sub_96 = unsqueeze_771 = None
        unsqueeze_772: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg484_1, -1);  arg484_1 = None
        unsqueeze_773: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_772, -1);  unsqueeze_772 = None
        mul_290: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_289, unsqueeze_773);  mul_289 = unsqueeze_773 = None
        unsqueeze_774: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg485_1, -1);  arg485_1 = None
        unsqueeze_775: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_774, -1);  unsqueeze_774 = None
        add_223: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_290, unsqueeze_775);  mul_290 = unsqueeze_775 = None
        convert_element_type_290: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_223, torch.float16);  add_223 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_224: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_290, relu_90);  convert_element_type_290 = relu_90 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_93: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_224);  add_224 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_97: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_93, arg486_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg486_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_291: "f32[256]" = torch.ops.prims.convert_element_type.default(arg487_1, torch.float32);  arg487_1 = None
        unsqueeze_776: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_291, -1);  convert_element_type_291 = None
        unsqueeze_777: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_776, -1);  unsqueeze_776 = None
        sub_97: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_97, unsqueeze_777);  convolution_97 = unsqueeze_777 = None
        convert_element_type_292: "f32[256]" = torch.ops.prims.convert_element_type.default(arg488_1, torch.float32);  arg488_1 = None
        add_225: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_292, 1e-05);  convert_element_type_292 = None
        sqrt_97: "f32[256]" = torch.ops.aten.sqrt.default(add_225);  add_225 = None
        reciprocal_97: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_97);  sqrt_97 = None
        mul_291: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_97, 1);  reciprocal_97 = None
        unsqueeze_778: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_291, -1);  mul_291 = None
        unsqueeze_779: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_778, -1);  unsqueeze_778 = None
        mul_292: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_97, unsqueeze_779);  sub_97 = unsqueeze_779 = None
        unsqueeze_780: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg489_1, -1);  arg489_1 = None
        unsqueeze_781: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_780, -1);  unsqueeze_780 = None
        mul_293: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_292, unsqueeze_781);  mul_292 = unsqueeze_781 = None
        unsqueeze_782: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg490_1, -1);  arg490_1 = None
        unsqueeze_783: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_782, -1);  unsqueeze_782 = None
        add_226: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_293, unsqueeze_783);  mul_293 = unsqueeze_783 = None
        convert_element_type_293: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_226, torch.float16);  add_226 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_94: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_293);  convert_element_type_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_98: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_94, arg491_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_94 = arg491_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_294: "f32[256]" = torch.ops.prims.convert_element_type.default(arg492_1, torch.float32);  arg492_1 = None
        unsqueeze_784: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_294, -1);  convert_element_type_294 = None
        unsqueeze_785: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_784, -1);  unsqueeze_784 = None
        sub_98: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_98, unsqueeze_785);  convolution_98 = unsqueeze_785 = None
        convert_element_type_295: "f32[256]" = torch.ops.prims.convert_element_type.default(arg493_1, torch.float32);  arg493_1 = None
        add_227: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_295, 1e-05);  convert_element_type_295 = None
        sqrt_98: "f32[256]" = torch.ops.aten.sqrt.default(add_227);  add_227 = None
        reciprocal_98: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_98);  sqrt_98 = None
        mul_294: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_98, 1);  reciprocal_98 = None
        unsqueeze_786: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_294, -1);  mul_294 = None
        unsqueeze_787: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_786, -1);  unsqueeze_786 = None
        mul_295: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_98, unsqueeze_787);  sub_98 = unsqueeze_787 = None
        unsqueeze_788: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg494_1, -1);  arg494_1 = None
        unsqueeze_789: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_788, -1);  unsqueeze_788 = None
        mul_296: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_295, unsqueeze_789);  mul_295 = unsqueeze_789 = None
        unsqueeze_790: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg495_1, -1);  arg495_1 = None
        unsqueeze_791: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_790, -1);  unsqueeze_790 = None
        add_228: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_296, unsqueeze_791);  mul_296 = unsqueeze_791 = None
        convert_element_type_296: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_228, torch.float16);  add_228 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_95: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_296);  convert_element_type_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_99: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_95, arg496_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_95 = arg496_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_297: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg497_1, torch.float32);  arg497_1 = None
        unsqueeze_792: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_297, -1);  convert_element_type_297 = None
        unsqueeze_793: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_792, -1);  unsqueeze_792 = None
        sub_99: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_99, unsqueeze_793);  convolution_99 = unsqueeze_793 = None
        convert_element_type_298: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg498_1, torch.float32);  arg498_1 = None
        add_229: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_298, 1e-05);  convert_element_type_298 = None
        sqrt_99: "f32[1024]" = torch.ops.aten.sqrt.default(add_229);  add_229 = None
        reciprocal_99: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_99);  sqrt_99 = None
        mul_297: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_99, 1);  reciprocal_99 = None
        unsqueeze_794: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_297, -1);  mul_297 = None
        unsqueeze_795: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_794, -1);  unsqueeze_794 = None
        mul_298: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_99, unsqueeze_795);  sub_99 = unsqueeze_795 = None
        unsqueeze_796: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg499_1, -1);  arg499_1 = None
        unsqueeze_797: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_796, -1);  unsqueeze_796 = None
        mul_299: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_298, unsqueeze_797);  mul_298 = unsqueeze_797 = None
        unsqueeze_798: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg500_1, -1);  arg500_1 = None
        unsqueeze_799: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_798, -1);  unsqueeze_798 = None
        add_230: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_299, unsqueeze_799);  mul_299 = unsqueeze_799 = None
        convert_element_type_299: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_230, torch.float16);  add_230 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_231: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_299, relu_93);  convert_element_type_299 = relu_93 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_96: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_231);  add_231 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_100: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_96, arg501_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg501_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_300: "f32[256]" = torch.ops.prims.convert_element_type.default(arg502_1, torch.float32);  arg502_1 = None
        unsqueeze_800: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_300, -1);  convert_element_type_300 = None
        unsqueeze_801: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_800, -1);  unsqueeze_800 = None
        sub_100: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_100, unsqueeze_801);  convolution_100 = unsqueeze_801 = None
        convert_element_type_301: "f32[256]" = torch.ops.prims.convert_element_type.default(arg503_1, torch.float32);  arg503_1 = None
        add_232: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_301, 1e-05);  convert_element_type_301 = None
        sqrt_100: "f32[256]" = torch.ops.aten.sqrt.default(add_232);  add_232 = None
        reciprocal_100: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_100);  sqrt_100 = None
        mul_300: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_100, 1);  reciprocal_100 = None
        unsqueeze_802: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_300, -1);  mul_300 = None
        unsqueeze_803: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_802, -1);  unsqueeze_802 = None
        mul_301: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_100, unsqueeze_803);  sub_100 = unsqueeze_803 = None
        unsqueeze_804: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg504_1, -1);  arg504_1 = None
        unsqueeze_805: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_804, -1);  unsqueeze_804 = None
        mul_302: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_301, unsqueeze_805);  mul_301 = unsqueeze_805 = None
        unsqueeze_806: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg505_1, -1);  arg505_1 = None
        unsqueeze_807: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_806, -1);  unsqueeze_806 = None
        add_233: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_302, unsqueeze_807);  mul_302 = unsqueeze_807 = None
        convert_element_type_302: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_233, torch.float16);  add_233 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_97: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_302);  convert_element_type_302 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_101: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_97, arg506_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_97 = arg506_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_303: "f32[256]" = torch.ops.prims.convert_element_type.default(arg507_1, torch.float32);  arg507_1 = None
        unsqueeze_808: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_303, -1);  convert_element_type_303 = None
        unsqueeze_809: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_808, -1);  unsqueeze_808 = None
        sub_101: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_101, unsqueeze_809);  convolution_101 = unsqueeze_809 = None
        convert_element_type_304: "f32[256]" = torch.ops.prims.convert_element_type.default(arg508_1, torch.float32);  arg508_1 = None
        add_234: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_304, 1e-05);  convert_element_type_304 = None
        sqrt_101: "f32[256]" = torch.ops.aten.sqrt.default(add_234);  add_234 = None
        reciprocal_101: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_101);  sqrt_101 = None
        mul_303: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_101, 1);  reciprocal_101 = None
        unsqueeze_810: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_303, -1);  mul_303 = None
        unsqueeze_811: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_810, -1);  unsqueeze_810 = None
        mul_304: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_101, unsqueeze_811);  sub_101 = unsqueeze_811 = None
        unsqueeze_812: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg509_1, -1);  arg509_1 = None
        unsqueeze_813: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_812, -1);  unsqueeze_812 = None
        mul_305: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_304, unsqueeze_813);  mul_304 = unsqueeze_813 = None
        unsqueeze_814: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg510_1, -1);  arg510_1 = None
        unsqueeze_815: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_814, -1);  unsqueeze_814 = None
        add_235: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_305, unsqueeze_815);  mul_305 = unsqueeze_815 = None
        convert_element_type_305: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_235, torch.float16);  add_235 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_98: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_305);  convert_element_type_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_102: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_98, arg511_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_98 = arg511_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_306: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg512_1, torch.float32);  arg512_1 = None
        unsqueeze_816: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_306, -1);  convert_element_type_306 = None
        unsqueeze_817: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_816, -1);  unsqueeze_816 = None
        sub_102: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_102, unsqueeze_817);  convolution_102 = unsqueeze_817 = None
        convert_element_type_307: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg513_1, torch.float32);  arg513_1 = None
        add_236: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_307, 1e-05);  convert_element_type_307 = None
        sqrt_102: "f32[1024]" = torch.ops.aten.sqrt.default(add_236);  add_236 = None
        reciprocal_102: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_102);  sqrt_102 = None
        mul_306: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_102, 1);  reciprocal_102 = None
        unsqueeze_818: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_306, -1);  mul_306 = None
        unsqueeze_819: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_818, -1);  unsqueeze_818 = None
        mul_307: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_102, unsqueeze_819);  sub_102 = unsqueeze_819 = None
        unsqueeze_820: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg514_1, -1);  arg514_1 = None
        unsqueeze_821: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_820, -1);  unsqueeze_820 = None
        mul_308: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_307, unsqueeze_821);  mul_307 = unsqueeze_821 = None
        unsqueeze_822: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg515_1, -1);  arg515_1 = None
        unsqueeze_823: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_822, -1);  unsqueeze_822 = None
        add_237: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_308, unsqueeze_823);  mul_308 = unsqueeze_823 = None
        convert_element_type_308: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_237, torch.float16);  add_237 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_238: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_308, relu_96);  convert_element_type_308 = relu_96 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_99: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_238);  add_238 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_103: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_99, arg516_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg516_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_309: "f32[256]" = torch.ops.prims.convert_element_type.default(arg517_1, torch.float32);  arg517_1 = None
        unsqueeze_824: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_309, -1);  convert_element_type_309 = None
        unsqueeze_825: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_824, -1);  unsqueeze_824 = None
        sub_103: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_103, unsqueeze_825);  convolution_103 = unsqueeze_825 = None
        convert_element_type_310: "f32[256]" = torch.ops.prims.convert_element_type.default(arg518_1, torch.float32);  arg518_1 = None
        add_239: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_310, 1e-05);  convert_element_type_310 = None
        sqrt_103: "f32[256]" = torch.ops.aten.sqrt.default(add_239);  add_239 = None
        reciprocal_103: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_103);  sqrt_103 = None
        mul_309: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_103, 1);  reciprocal_103 = None
        unsqueeze_826: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_309, -1);  mul_309 = None
        unsqueeze_827: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_826, -1);  unsqueeze_826 = None
        mul_310: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_103, unsqueeze_827);  sub_103 = unsqueeze_827 = None
        unsqueeze_828: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg519_1, -1);  arg519_1 = None
        unsqueeze_829: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_828, -1);  unsqueeze_828 = None
        mul_311: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_310, unsqueeze_829);  mul_310 = unsqueeze_829 = None
        unsqueeze_830: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg520_1, -1);  arg520_1 = None
        unsqueeze_831: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_830, -1);  unsqueeze_830 = None
        add_240: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_311, unsqueeze_831);  mul_311 = unsqueeze_831 = None
        convert_element_type_311: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_240, torch.float16);  add_240 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_100: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_311);  convert_element_type_311 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_104: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_100, arg521_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_100 = arg521_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_312: "f32[256]" = torch.ops.prims.convert_element_type.default(arg522_1, torch.float32);  arg522_1 = None
        unsqueeze_832: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_312, -1);  convert_element_type_312 = None
        unsqueeze_833: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_832, -1);  unsqueeze_832 = None
        sub_104: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_104, unsqueeze_833);  convolution_104 = unsqueeze_833 = None
        convert_element_type_313: "f32[256]" = torch.ops.prims.convert_element_type.default(arg523_1, torch.float32);  arg523_1 = None
        add_241: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_313, 1e-05);  convert_element_type_313 = None
        sqrt_104: "f32[256]" = torch.ops.aten.sqrt.default(add_241);  add_241 = None
        reciprocal_104: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_104);  sqrt_104 = None
        mul_312: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_104, 1);  reciprocal_104 = None
        unsqueeze_834: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_312, -1);  mul_312 = None
        unsqueeze_835: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_834, -1);  unsqueeze_834 = None
        mul_313: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_104, unsqueeze_835);  sub_104 = unsqueeze_835 = None
        unsqueeze_836: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg524_1, -1);  arg524_1 = None
        unsqueeze_837: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_836, -1);  unsqueeze_836 = None
        mul_314: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_313, unsqueeze_837);  mul_313 = unsqueeze_837 = None
        unsqueeze_838: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg525_1, -1);  arg525_1 = None
        unsqueeze_839: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_838, -1);  unsqueeze_838 = None
        add_242: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_314, unsqueeze_839);  mul_314 = unsqueeze_839 = None
        convert_element_type_314: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_242, torch.float16);  add_242 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_101: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_314);  convert_element_type_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_105: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_101, arg526_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_101 = arg526_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_315: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg527_1, torch.float32);  arg527_1 = None
        unsqueeze_840: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_315, -1);  convert_element_type_315 = None
        unsqueeze_841: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_840, -1);  unsqueeze_840 = None
        sub_105: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_105, unsqueeze_841);  convolution_105 = unsqueeze_841 = None
        convert_element_type_316: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg528_1, torch.float32);  arg528_1 = None
        add_243: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_316, 1e-05);  convert_element_type_316 = None
        sqrt_105: "f32[1024]" = torch.ops.aten.sqrt.default(add_243);  add_243 = None
        reciprocal_105: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_105);  sqrt_105 = None
        mul_315: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_105, 1);  reciprocal_105 = None
        unsqueeze_842: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_315, -1);  mul_315 = None
        unsqueeze_843: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_842, -1);  unsqueeze_842 = None
        mul_316: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_105, unsqueeze_843);  sub_105 = unsqueeze_843 = None
        unsqueeze_844: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg529_1, -1);  arg529_1 = None
        unsqueeze_845: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_844, -1);  unsqueeze_844 = None
        mul_317: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_316, unsqueeze_845);  mul_316 = unsqueeze_845 = None
        unsqueeze_846: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg530_1, -1);  arg530_1 = None
        unsqueeze_847: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_846, -1);  unsqueeze_846 = None
        add_244: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_317, unsqueeze_847);  mul_317 = unsqueeze_847 = None
        convert_element_type_317: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_244, torch.float16);  add_244 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_245: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_317, relu_99);  convert_element_type_317 = relu_99 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_102: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_245);  add_245 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_106: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_102, arg531_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg531_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_318: "f32[256]" = torch.ops.prims.convert_element_type.default(arg532_1, torch.float32);  arg532_1 = None
        unsqueeze_848: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_318, -1);  convert_element_type_318 = None
        unsqueeze_849: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_848, -1);  unsqueeze_848 = None
        sub_106: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_106, unsqueeze_849);  convolution_106 = unsqueeze_849 = None
        convert_element_type_319: "f32[256]" = torch.ops.prims.convert_element_type.default(arg533_1, torch.float32);  arg533_1 = None
        add_246: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_319, 1e-05);  convert_element_type_319 = None
        sqrt_106: "f32[256]" = torch.ops.aten.sqrt.default(add_246);  add_246 = None
        reciprocal_106: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_106);  sqrt_106 = None
        mul_318: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_106, 1);  reciprocal_106 = None
        unsqueeze_850: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_318, -1);  mul_318 = None
        unsqueeze_851: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_850, -1);  unsqueeze_850 = None
        mul_319: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_106, unsqueeze_851);  sub_106 = unsqueeze_851 = None
        unsqueeze_852: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg534_1, -1);  arg534_1 = None
        unsqueeze_853: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_852, -1);  unsqueeze_852 = None
        mul_320: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_319, unsqueeze_853);  mul_319 = unsqueeze_853 = None
        unsqueeze_854: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg535_1, -1);  arg535_1 = None
        unsqueeze_855: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_854, -1);  unsqueeze_854 = None
        add_247: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_320, unsqueeze_855);  mul_320 = unsqueeze_855 = None
        convert_element_type_320: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_247, torch.float16);  add_247 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_103: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_320);  convert_element_type_320 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_107: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_103, arg536_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_103 = arg536_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_321: "f32[256]" = torch.ops.prims.convert_element_type.default(arg537_1, torch.float32);  arg537_1 = None
        unsqueeze_856: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_321, -1);  convert_element_type_321 = None
        unsqueeze_857: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_856, -1);  unsqueeze_856 = None
        sub_107: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_107, unsqueeze_857);  convolution_107 = unsqueeze_857 = None
        convert_element_type_322: "f32[256]" = torch.ops.prims.convert_element_type.default(arg538_1, torch.float32);  arg538_1 = None
        add_248: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_322, 1e-05);  convert_element_type_322 = None
        sqrt_107: "f32[256]" = torch.ops.aten.sqrt.default(add_248);  add_248 = None
        reciprocal_107: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_107);  sqrt_107 = None
        mul_321: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_107, 1);  reciprocal_107 = None
        unsqueeze_858: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_321, -1);  mul_321 = None
        unsqueeze_859: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_858, -1);  unsqueeze_858 = None
        mul_322: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_107, unsqueeze_859);  sub_107 = unsqueeze_859 = None
        unsqueeze_860: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg539_1, -1);  arg539_1 = None
        unsqueeze_861: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_860, -1);  unsqueeze_860 = None
        mul_323: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_322, unsqueeze_861);  mul_322 = unsqueeze_861 = None
        unsqueeze_862: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg540_1, -1);  arg540_1 = None
        unsqueeze_863: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_862, -1);  unsqueeze_862 = None
        add_249: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_323, unsqueeze_863);  mul_323 = unsqueeze_863 = None
        convert_element_type_323: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_249, torch.float16);  add_249 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_104: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_323);  convert_element_type_323 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_108: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_104, arg541_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_104 = arg541_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_324: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg542_1, torch.float32);  arg542_1 = None
        unsqueeze_864: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_324, -1);  convert_element_type_324 = None
        unsqueeze_865: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_864, -1);  unsqueeze_864 = None
        sub_108: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_108, unsqueeze_865);  convolution_108 = unsqueeze_865 = None
        convert_element_type_325: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg543_1, torch.float32);  arg543_1 = None
        add_250: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_325, 1e-05);  convert_element_type_325 = None
        sqrt_108: "f32[1024]" = torch.ops.aten.sqrt.default(add_250);  add_250 = None
        reciprocal_108: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_108);  sqrt_108 = None
        mul_324: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_108, 1);  reciprocal_108 = None
        unsqueeze_866: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_324, -1);  mul_324 = None
        unsqueeze_867: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_866, -1);  unsqueeze_866 = None
        mul_325: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_108, unsqueeze_867);  sub_108 = unsqueeze_867 = None
        unsqueeze_868: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg544_1, -1);  arg544_1 = None
        unsqueeze_869: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_868, -1);  unsqueeze_868 = None
        mul_326: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_325, unsqueeze_869);  mul_325 = unsqueeze_869 = None
        unsqueeze_870: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg545_1, -1);  arg545_1 = None
        unsqueeze_871: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_870, -1);  unsqueeze_870 = None
        add_251: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_326, unsqueeze_871);  mul_326 = unsqueeze_871 = None
        convert_element_type_326: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_251, torch.float16);  add_251 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_252: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_326, relu_102);  convert_element_type_326 = relu_102 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_105: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_252);  add_252 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_109: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_105, arg546_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg546_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_327: "f32[256]" = torch.ops.prims.convert_element_type.default(arg547_1, torch.float32);  arg547_1 = None
        unsqueeze_872: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_327, -1);  convert_element_type_327 = None
        unsqueeze_873: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_872, -1);  unsqueeze_872 = None
        sub_109: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_109, unsqueeze_873);  convolution_109 = unsqueeze_873 = None
        convert_element_type_328: "f32[256]" = torch.ops.prims.convert_element_type.default(arg548_1, torch.float32);  arg548_1 = None
        add_253: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_328, 1e-05);  convert_element_type_328 = None
        sqrt_109: "f32[256]" = torch.ops.aten.sqrt.default(add_253);  add_253 = None
        reciprocal_109: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_109);  sqrt_109 = None
        mul_327: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_109, 1);  reciprocal_109 = None
        unsqueeze_874: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_327, -1);  mul_327 = None
        unsqueeze_875: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_874, -1);  unsqueeze_874 = None
        mul_328: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_109, unsqueeze_875);  sub_109 = unsqueeze_875 = None
        unsqueeze_876: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg549_1, -1);  arg549_1 = None
        unsqueeze_877: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_876, -1);  unsqueeze_876 = None
        mul_329: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_328, unsqueeze_877);  mul_328 = unsqueeze_877 = None
        unsqueeze_878: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg550_1, -1);  arg550_1 = None
        unsqueeze_879: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_878, -1);  unsqueeze_878 = None
        add_254: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_329, unsqueeze_879);  mul_329 = unsqueeze_879 = None
        convert_element_type_329: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_254, torch.float16);  add_254 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_106: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_329);  convert_element_type_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_110: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_106, arg551_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_106 = arg551_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_330: "f32[256]" = torch.ops.prims.convert_element_type.default(arg552_1, torch.float32);  arg552_1 = None
        unsqueeze_880: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_330, -1);  convert_element_type_330 = None
        unsqueeze_881: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_880, -1);  unsqueeze_880 = None
        sub_110: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_110, unsqueeze_881);  convolution_110 = unsqueeze_881 = None
        convert_element_type_331: "f32[256]" = torch.ops.prims.convert_element_type.default(arg553_1, torch.float32);  arg553_1 = None
        add_255: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_331, 1e-05);  convert_element_type_331 = None
        sqrt_110: "f32[256]" = torch.ops.aten.sqrt.default(add_255);  add_255 = None
        reciprocal_110: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_110);  sqrt_110 = None
        mul_330: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_110, 1);  reciprocal_110 = None
        unsqueeze_882: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_330, -1);  mul_330 = None
        unsqueeze_883: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_882, -1);  unsqueeze_882 = None
        mul_331: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_110, unsqueeze_883);  sub_110 = unsqueeze_883 = None
        unsqueeze_884: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg554_1, -1);  arg554_1 = None
        unsqueeze_885: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_884, -1);  unsqueeze_884 = None
        mul_332: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_331, unsqueeze_885);  mul_331 = unsqueeze_885 = None
        unsqueeze_886: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg555_1, -1);  arg555_1 = None
        unsqueeze_887: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_886, -1);  unsqueeze_886 = None
        add_256: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_332, unsqueeze_887);  mul_332 = unsqueeze_887 = None
        convert_element_type_332: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_256, torch.float16);  add_256 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_107: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_332);  convert_element_type_332 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_111: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_107, arg556_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_107 = arg556_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_333: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg557_1, torch.float32);  arg557_1 = None
        unsqueeze_888: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_333, -1);  convert_element_type_333 = None
        unsqueeze_889: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_888, -1);  unsqueeze_888 = None
        sub_111: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_111, unsqueeze_889);  convolution_111 = unsqueeze_889 = None
        convert_element_type_334: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg558_1, torch.float32);  arg558_1 = None
        add_257: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_334, 1e-05);  convert_element_type_334 = None
        sqrt_111: "f32[1024]" = torch.ops.aten.sqrt.default(add_257);  add_257 = None
        reciprocal_111: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_111);  sqrt_111 = None
        mul_333: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_111, 1);  reciprocal_111 = None
        unsqueeze_890: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_333, -1);  mul_333 = None
        unsqueeze_891: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_890, -1);  unsqueeze_890 = None
        mul_334: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_111, unsqueeze_891);  sub_111 = unsqueeze_891 = None
        unsqueeze_892: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg559_1, -1);  arg559_1 = None
        unsqueeze_893: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_892, -1);  unsqueeze_892 = None
        mul_335: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_334, unsqueeze_893);  mul_334 = unsqueeze_893 = None
        unsqueeze_894: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg560_1, -1);  arg560_1 = None
        unsqueeze_895: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_894, -1);  unsqueeze_894 = None
        add_258: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_335, unsqueeze_895);  mul_335 = unsqueeze_895 = None
        convert_element_type_335: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_258, torch.float16);  add_258 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_259: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_335, relu_105);  convert_element_type_335 = relu_105 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_108: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_259);  add_259 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_112: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_108, arg561_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg561_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_336: "f32[256]" = torch.ops.prims.convert_element_type.default(arg562_1, torch.float32);  arg562_1 = None
        unsqueeze_896: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_336, -1);  convert_element_type_336 = None
        unsqueeze_897: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_896, -1);  unsqueeze_896 = None
        sub_112: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_112, unsqueeze_897);  convolution_112 = unsqueeze_897 = None
        convert_element_type_337: "f32[256]" = torch.ops.prims.convert_element_type.default(arg563_1, torch.float32);  arg563_1 = None
        add_260: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_337, 1e-05);  convert_element_type_337 = None
        sqrt_112: "f32[256]" = torch.ops.aten.sqrt.default(add_260);  add_260 = None
        reciprocal_112: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_112);  sqrt_112 = None
        mul_336: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_112, 1);  reciprocal_112 = None
        unsqueeze_898: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_336, -1);  mul_336 = None
        unsqueeze_899: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_898, -1);  unsqueeze_898 = None
        mul_337: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_112, unsqueeze_899);  sub_112 = unsqueeze_899 = None
        unsqueeze_900: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg564_1, -1);  arg564_1 = None
        unsqueeze_901: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_900, -1);  unsqueeze_900 = None
        mul_338: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_337, unsqueeze_901);  mul_337 = unsqueeze_901 = None
        unsqueeze_902: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg565_1, -1);  arg565_1 = None
        unsqueeze_903: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_902, -1);  unsqueeze_902 = None
        add_261: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_338, unsqueeze_903);  mul_338 = unsqueeze_903 = None
        convert_element_type_338: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_261, torch.float16);  add_261 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_109: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_338);  convert_element_type_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_113: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_109, arg566_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_109 = arg566_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_339: "f32[256]" = torch.ops.prims.convert_element_type.default(arg567_1, torch.float32);  arg567_1 = None
        unsqueeze_904: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_339, -1);  convert_element_type_339 = None
        unsqueeze_905: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_904, -1);  unsqueeze_904 = None
        sub_113: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_113, unsqueeze_905);  convolution_113 = unsqueeze_905 = None
        convert_element_type_340: "f32[256]" = torch.ops.prims.convert_element_type.default(arg568_1, torch.float32);  arg568_1 = None
        add_262: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_340, 1e-05);  convert_element_type_340 = None
        sqrt_113: "f32[256]" = torch.ops.aten.sqrt.default(add_262);  add_262 = None
        reciprocal_113: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_113);  sqrt_113 = None
        mul_339: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_113, 1);  reciprocal_113 = None
        unsqueeze_906: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_339, -1);  mul_339 = None
        unsqueeze_907: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_906, -1);  unsqueeze_906 = None
        mul_340: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_113, unsqueeze_907);  sub_113 = unsqueeze_907 = None
        unsqueeze_908: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg569_1, -1);  arg569_1 = None
        unsqueeze_909: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_908, -1);  unsqueeze_908 = None
        mul_341: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_340, unsqueeze_909);  mul_340 = unsqueeze_909 = None
        unsqueeze_910: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg570_1, -1);  arg570_1 = None
        unsqueeze_911: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_910, -1);  unsqueeze_910 = None
        add_263: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_341, unsqueeze_911);  mul_341 = unsqueeze_911 = None
        convert_element_type_341: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_263, torch.float16);  add_263 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_110: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_341);  convert_element_type_341 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_114: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_110, arg571_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_110 = arg571_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_342: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg572_1, torch.float32);  arg572_1 = None
        unsqueeze_912: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_342, -1);  convert_element_type_342 = None
        unsqueeze_913: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_912, -1);  unsqueeze_912 = None
        sub_114: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_114, unsqueeze_913);  convolution_114 = unsqueeze_913 = None
        convert_element_type_343: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg573_1, torch.float32);  arg573_1 = None
        add_264: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_343, 1e-05);  convert_element_type_343 = None
        sqrt_114: "f32[1024]" = torch.ops.aten.sqrt.default(add_264);  add_264 = None
        reciprocal_114: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_114);  sqrt_114 = None
        mul_342: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_114, 1);  reciprocal_114 = None
        unsqueeze_914: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_342, -1);  mul_342 = None
        unsqueeze_915: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_914, -1);  unsqueeze_914 = None
        mul_343: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_114, unsqueeze_915);  sub_114 = unsqueeze_915 = None
        unsqueeze_916: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg574_1, -1);  arg574_1 = None
        unsqueeze_917: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_916, -1);  unsqueeze_916 = None
        mul_344: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_343, unsqueeze_917);  mul_343 = unsqueeze_917 = None
        unsqueeze_918: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg575_1, -1);  arg575_1 = None
        unsqueeze_919: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_918, -1);  unsqueeze_918 = None
        add_265: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_344, unsqueeze_919);  mul_344 = unsqueeze_919 = None
        convert_element_type_344: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_265, torch.float16);  add_265 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_266: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_344, relu_108);  convert_element_type_344 = relu_108 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_111: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_266);  add_266 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_115: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_111, arg576_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg576_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_345: "f32[256]" = torch.ops.prims.convert_element_type.default(arg577_1, torch.float32);  arg577_1 = None
        unsqueeze_920: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_345, -1);  convert_element_type_345 = None
        unsqueeze_921: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_920, -1);  unsqueeze_920 = None
        sub_115: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_115, unsqueeze_921);  convolution_115 = unsqueeze_921 = None
        convert_element_type_346: "f32[256]" = torch.ops.prims.convert_element_type.default(arg578_1, torch.float32);  arg578_1 = None
        add_267: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_346, 1e-05);  convert_element_type_346 = None
        sqrt_115: "f32[256]" = torch.ops.aten.sqrt.default(add_267);  add_267 = None
        reciprocal_115: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_115);  sqrt_115 = None
        mul_345: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_115, 1);  reciprocal_115 = None
        unsqueeze_922: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_345, -1);  mul_345 = None
        unsqueeze_923: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_922, -1);  unsqueeze_922 = None
        mul_346: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_115, unsqueeze_923);  sub_115 = unsqueeze_923 = None
        unsqueeze_924: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg579_1, -1);  arg579_1 = None
        unsqueeze_925: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_924, -1);  unsqueeze_924 = None
        mul_347: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_346, unsqueeze_925);  mul_346 = unsqueeze_925 = None
        unsqueeze_926: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg580_1, -1);  arg580_1 = None
        unsqueeze_927: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_926, -1);  unsqueeze_926 = None
        add_268: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_347, unsqueeze_927);  mul_347 = unsqueeze_927 = None
        convert_element_type_347: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_268, torch.float16);  add_268 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_112: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_347);  convert_element_type_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_116: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_112, arg581_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_112 = arg581_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_348: "f32[256]" = torch.ops.prims.convert_element_type.default(arg582_1, torch.float32);  arg582_1 = None
        unsqueeze_928: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_348, -1);  convert_element_type_348 = None
        unsqueeze_929: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_928, -1);  unsqueeze_928 = None
        sub_116: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_116, unsqueeze_929);  convolution_116 = unsqueeze_929 = None
        convert_element_type_349: "f32[256]" = torch.ops.prims.convert_element_type.default(arg583_1, torch.float32);  arg583_1 = None
        add_269: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_349, 1e-05);  convert_element_type_349 = None
        sqrt_116: "f32[256]" = torch.ops.aten.sqrt.default(add_269);  add_269 = None
        reciprocal_116: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_116);  sqrt_116 = None
        mul_348: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_116, 1);  reciprocal_116 = None
        unsqueeze_930: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_348, -1);  mul_348 = None
        unsqueeze_931: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_930, -1);  unsqueeze_930 = None
        mul_349: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_116, unsqueeze_931);  sub_116 = unsqueeze_931 = None
        unsqueeze_932: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg584_1, -1);  arg584_1 = None
        unsqueeze_933: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_932, -1);  unsqueeze_932 = None
        mul_350: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_349, unsqueeze_933);  mul_349 = unsqueeze_933 = None
        unsqueeze_934: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg585_1, -1);  arg585_1 = None
        unsqueeze_935: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_934, -1);  unsqueeze_934 = None
        add_270: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_350, unsqueeze_935);  mul_350 = unsqueeze_935 = None
        convert_element_type_350: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_270, torch.float16);  add_270 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_113: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_350);  convert_element_type_350 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_117: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_113, arg586_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_113 = arg586_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_351: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg587_1, torch.float32);  arg587_1 = None
        unsqueeze_936: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_351, -1);  convert_element_type_351 = None
        unsqueeze_937: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_936, -1);  unsqueeze_936 = None
        sub_117: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_117, unsqueeze_937);  convolution_117 = unsqueeze_937 = None
        convert_element_type_352: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg588_1, torch.float32);  arg588_1 = None
        add_271: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_352, 1e-05);  convert_element_type_352 = None
        sqrt_117: "f32[1024]" = torch.ops.aten.sqrt.default(add_271);  add_271 = None
        reciprocal_117: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_117);  sqrt_117 = None
        mul_351: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_117, 1);  reciprocal_117 = None
        unsqueeze_938: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_351, -1);  mul_351 = None
        unsqueeze_939: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_938, -1);  unsqueeze_938 = None
        mul_352: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_117, unsqueeze_939);  sub_117 = unsqueeze_939 = None
        unsqueeze_940: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg589_1, -1);  arg589_1 = None
        unsqueeze_941: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_940, -1);  unsqueeze_940 = None
        mul_353: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_352, unsqueeze_941);  mul_352 = unsqueeze_941 = None
        unsqueeze_942: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg590_1, -1);  arg590_1 = None
        unsqueeze_943: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_942, -1);  unsqueeze_942 = None
        add_272: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_353, unsqueeze_943);  mul_353 = unsqueeze_943 = None
        convert_element_type_353: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_272, torch.float16);  add_272 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_273: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_353, relu_111);  convert_element_type_353 = relu_111 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_114: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_273);  add_273 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_118: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_114, arg591_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg591_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_354: "f32[256]" = torch.ops.prims.convert_element_type.default(arg592_1, torch.float32);  arg592_1 = None
        unsqueeze_944: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_354, -1);  convert_element_type_354 = None
        unsqueeze_945: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_944, -1);  unsqueeze_944 = None
        sub_118: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_118, unsqueeze_945);  convolution_118 = unsqueeze_945 = None
        convert_element_type_355: "f32[256]" = torch.ops.prims.convert_element_type.default(arg593_1, torch.float32);  arg593_1 = None
        add_274: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_355, 1e-05);  convert_element_type_355 = None
        sqrt_118: "f32[256]" = torch.ops.aten.sqrt.default(add_274);  add_274 = None
        reciprocal_118: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_118);  sqrt_118 = None
        mul_354: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_118, 1);  reciprocal_118 = None
        unsqueeze_946: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_354, -1);  mul_354 = None
        unsqueeze_947: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_946, -1);  unsqueeze_946 = None
        mul_355: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_118, unsqueeze_947);  sub_118 = unsqueeze_947 = None
        unsqueeze_948: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg594_1, -1);  arg594_1 = None
        unsqueeze_949: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_948, -1);  unsqueeze_948 = None
        mul_356: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_355, unsqueeze_949);  mul_355 = unsqueeze_949 = None
        unsqueeze_950: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg595_1, -1);  arg595_1 = None
        unsqueeze_951: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_950, -1);  unsqueeze_950 = None
        add_275: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_356, unsqueeze_951);  mul_356 = unsqueeze_951 = None
        convert_element_type_356: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_275, torch.float16);  add_275 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_115: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_356);  convert_element_type_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_119: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_115, arg596_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_115 = arg596_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_357: "f32[256]" = torch.ops.prims.convert_element_type.default(arg597_1, torch.float32);  arg597_1 = None
        unsqueeze_952: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_357, -1);  convert_element_type_357 = None
        unsqueeze_953: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_952, -1);  unsqueeze_952 = None
        sub_119: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_119, unsqueeze_953);  convolution_119 = unsqueeze_953 = None
        convert_element_type_358: "f32[256]" = torch.ops.prims.convert_element_type.default(arg598_1, torch.float32);  arg598_1 = None
        add_276: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_358, 1e-05);  convert_element_type_358 = None
        sqrt_119: "f32[256]" = torch.ops.aten.sqrt.default(add_276);  add_276 = None
        reciprocal_119: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_119);  sqrt_119 = None
        mul_357: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_119, 1);  reciprocal_119 = None
        unsqueeze_954: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_357, -1);  mul_357 = None
        unsqueeze_955: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_954, -1);  unsqueeze_954 = None
        mul_358: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_119, unsqueeze_955);  sub_119 = unsqueeze_955 = None
        unsqueeze_956: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg599_1, -1);  arg599_1 = None
        unsqueeze_957: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_956, -1);  unsqueeze_956 = None
        mul_359: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_358, unsqueeze_957);  mul_358 = unsqueeze_957 = None
        unsqueeze_958: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg600_1, -1);  arg600_1 = None
        unsqueeze_959: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_958, -1);  unsqueeze_958 = None
        add_277: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_359, unsqueeze_959);  mul_359 = unsqueeze_959 = None
        convert_element_type_359: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_277, torch.float16);  add_277 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_116: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_359);  convert_element_type_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_120: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_116, arg601_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_116 = arg601_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_360: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg602_1, torch.float32);  arg602_1 = None
        unsqueeze_960: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_360, -1);  convert_element_type_360 = None
        unsqueeze_961: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_960, -1);  unsqueeze_960 = None
        sub_120: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_120, unsqueeze_961);  convolution_120 = unsqueeze_961 = None
        convert_element_type_361: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg603_1, torch.float32);  arg603_1 = None
        add_278: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_361, 1e-05);  convert_element_type_361 = None
        sqrt_120: "f32[1024]" = torch.ops.aten.sqrt.default(add_278);  add_278 = None
        reciprocal_120: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_120);  sqrt_120 = None
        mul_360: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_120, 1);  reciprocal_120 = None
        unsqueeze_962: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_360, -1);  mul_360 = None
        unsqueeze_963: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_962, -1);  unsqueeze_962 = None
        mul_361: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_120, unsqueeze_963);  sub_120 = unsqueeze_963 = None
        unsqueeze_964: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg604_1, -1);  arg604_1 = None
        unsqueeze_965: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_964, -1);  unsqueeze_964 = None
        mul_362: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_361, unsqueeze_965);  mul_361 = unsqueeze_965 = None
        unsqueeze_966: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg605_1, -1);  arg605_1 = None
        unsqueeze_967: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_966, -1);  unsqueeze_966 = None
        add_279: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_362, unsqueeze_967);  mul_362 = unsqueeze_967 = None
        convert_element_type_362: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_279, torch.float16);  add_279 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_280: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_362, relu_114);  convert_element_type_362 = relu_114 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_117: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_280);  add_280 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_121: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_117, arg606_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg606_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_363: "f32[256]" = torch.ops.prims.convert_element_type.default(arg607_1, torch.float32);  arg607_1 = None
        unsqueeze_968: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_363, -1);  convert_element_type_363 = None
        unsqueeze_969: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_968, -1);  unsqueeze_968 = None
        sub_121: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_121, unsqueeze_969);  convolution_121 = unsqueeze_969 = None
        convert_element_type_364: "f32[256]" = torch.ops.prims.convert_element_type.default(arg608_1, torch.float32);  arg608_1 = None
        add_281: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_364, 1e-05);  convert_element_type_364 = None
        sqrt_121: "f32[256]" = torch.ops.aten.sqrt.default(add_281);  add_281 = None
        reciprocal_121: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_121);  sqrt_121 = None
        mul_363: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_121, 1);  reciprocal_121 = None
        unsqueeze_970: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_363, -1);  mul_363 = None
        unsqueeze_971: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_970, -1);  unsqueeze_970 = None
        mul_364: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_121, unsqueeze_971);  sub_121 = unsqueeze_971 = None
        unsqueeze_972: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg609_1, -1);  arg609_1 = None
        unsqueeze_973: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_972, -1);  unsqueeze_972 = None
        mul_365: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_364, unsqueeze_973);  mul_364 = unsqueeze_973 = None
        unsqueeze_974: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg610_1, -1);  arg610_1 = None
        unsqueeze_975: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_974, -1);  unsqueeze_974 = None
        add_282: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_365, unsqueeze_975);  mul_365 = unsqueeze_975 = None
        convert_element_type_365: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_282, torch.float16);  add_282 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_118: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_365);  convert_element_type_365 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_122: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_118, arg611_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_118 = arg611_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_366: "f32[256]" = torch.ops.prims.convert_element_type.default(arg612_1, torch.float32);  arg612_1 = None
        unsqueeze_976: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_366, -1);  convert_element_type_366 = None
        unsqueeze_977: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_976, -1);  unsqueeze_976 = None
        sub_122: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_122, unsqueeze_977);  convolution_122 = unsqueeze_977 = None
        convert_element_type_367: "f32[256]" = torch.ops.prims.convert_element_type.default(arg613_1, torch.float32);  arg613_1 = None
        add_283: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_367, 1e-05);  convert_element_type_367 = None
        sqrt_122: "f32[256]" = torch.ops.aten.sqrt.default(add_283);  add_283 = None
        reciprocal_122: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_122);  sqrt_122 = None
        mul_366: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_122, 1);  reciprocal_122 = None
        unsqueeze_978: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_366, -1);  mul_366 = None
        unsqueeze_979: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_978, -1);  unsqueeze_978 = None
        mul_367: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_122, unsqueeze_979);  sub_122 = unsqueeze_979 = None
        unsqueeze_980: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg614_1, -1);  arg614_1 = None
        unsqueeze_981: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_980, -1);  unsqueeze_980 = None
        mul_368: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_367, unsqueeze_981);  mul_367 = unsqueeze_981 = None
        unsqueeze_982: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg615_1, -1);  arg615_1 = None
        unsqueeze_983: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_982, -1);  unsqueeze_982 = None
        add_284: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_368, unsqueeze_983);  mul_368 = unsqueeze_983 = None
        convert_element_type_368: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_284, torch.float16);  add_284 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_119: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_368);  convert_element_type_368 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_123: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_119, arg616_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_119 = arg616_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_369: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg617_1, torch.float32);  arg617_1 = None
        unsqueeze_984: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_369, -1);  convert_element_type_369 = None
        unsqueeze_985: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_984, -1);  unsqueeze_984 = None
        sub_123: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_123, unsqueeze_985);  convolution_123 = unsqueeze_985 = None
        convert_element_type_370: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg618_1, torch.float32);  arg618_1 = None
        add_285: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_370, 1e-05);  convert_element_type_370 = None
        sqrt_123: "f32[1024]" = torch.ops.aten.sqrt.default(add_285);  add_285 = None
        reciprocal_123: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_123);  sqrt_123 = None
        mul_369: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_123, 1);  reciprocal_123 = None
        unsqueeze_986: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_369, -1);  mul_369 = None
        unsqueeze_987: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_986, -1);  unsqueeze_986 = None
        mul_370: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_123, unsqueeze_987);  sub_123 = unsqueeze_987 = None
        unsqueeze_988: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg619_1, -1);  arg619_1 = None
        unsqueeze_989: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_988, -1);  unsqueeze_988 = None
        mul_371: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_370, unsqueeze_989);  mul_370 = unsqueeze_989 = None
        unsqueeze_990: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg620_1, -1);  arg620_1 = None
        unsqueeze_991: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_990, -1);  unsqueeze_990 = None
        add_286: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_371, unsqueeze_991);  mul_371 = unsqueeze_991 = None
        convert_element_type_371: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_286, torch.float16);  add_286 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_287: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_371, relu_117);  convert_element_type_371 = relu_117 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_120: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_287);  add_287 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_124: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_120, arg621_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg621_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_372: "f32[256]" = torch.ops.prims.convert_element_type.default(arg622_1, torch.float32);  arg622_1 = None
        unsqueeze_992: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_372, -1);  convert_element_type_372 = None
        unsqueeze_993: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_992, -1);  unsqueeze_992 = None
        sub_124: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_124, unsqueeze_993);  convolution_124 = unsqueeze_993 = None
        convert_element_type_373: "f32[256]" = torch.ops.prims.convert_element_type.default(arg623_1, torch.float32);  arg623_1 = None
        add_288: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_373, 1e-05);  convert_element_type_373 = None
        sqrt_124: "f32[256]" = torch.ops.aten.sqrt.default(add_288);  add_288 = None
        reciprocal_124: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_124);  sqrt_124 = None
        mul_372: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_124, 1);  reciprocal_124 = None
        unsqueeze_994: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_372, -1);  mul_372 = None
        unsqueeze_995: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_994, -1);  unsqueeze_994 = None
        mul_373: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_124, unsqueeze_995);  sub_124 = unsqueeze_995 = None
        unsqueeze_996: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg624_1, -1);  arg624_1 = None
        unsqueeze_997: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_996, -1);  unsqueeze_996 = None
        mul_374: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_373, unsqueeze_997);  mul_373 = unsqueeze_997 = None
        unsqueeze_998: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg625_1, -1);  arg625_1 = None
        unsqueeze_999: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_998, -1);  unsqueeze_998 = None
        add_289: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_374, unsqueeze_999);  mul_374 = unsqueeze_999 = None
        convert_element_type_374: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_289, torch.float16);  add_289 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_121: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_374);  convert_element_type_374 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_125: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_121, arg626_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_121 = arg626_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_375: "f32[256]" = torch.ops.prims.convert_element_type.default(arg627_1, torch.float32);  arg627_1 = None
        unsqueeze_1000: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_375, -1);  convert_element_type_375 = None
        unsqueeze_1001: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1000, -1);  unsqueeze_1000 = None
        sub_125: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_125, unsqueeze_1001);  convolution_125 = unsqueeze_1001 = None
        convert_element_type_376: "f32[256]" = torch.ops.prims.convert_element_type.default(arg628_1, torch.float32);  arg628_1 = None
        add_290: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_376, 1e-05);  convert_element_type_376 = None
        sqrt_125: "f32[256]" = torch.ops.aten.sqrt.default(add_290);  add_290 = None
        reciprocal_125: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_125);  sqrt_125 = None
        mul_375: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_125, 1);  reciprocal_125 = None
        unsqueeze_1002: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_375, -1);  mul_375 = None
        unsqueeze_1003: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1002, -1);  unsqueeze_1002 = None
        mul_376: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_125, unsqueeze_1003);  sub_125 = unsqueeze_1003 = None
        unsqueeze_1004: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg629_1, -1);  arg629_1 = None
        unsqueeze_1005: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1004, -1);  unsqueeze_1004 = None
        mul_377: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_376, unsqueeze_1005);  mul_376 = unsqueeze_1005 = None
        unsqueeze_1006: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg630_1, -1);  arg630_1 = None
        unsqueeze_1007: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1006, -1);  unsqueeze_1006 = None
        add_291: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_377, unsqueeze_1007);  mul_377 = unsqueeze_1007 = None
        convert_element_type_377: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_291, torch.float16);  add_291 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_122: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_377);  convert_element_type_377 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_126: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_122, arg631_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_122 = arg631_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_378: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg632_1, torch.float32);  arg632_1 = None
        unsqueeze_1008: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_378, -1);  convert_element_type_378 = None
        unsqueeze_1009: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1008, -1);  unsqueeze_1008 = None
        sub_126: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_126, unsqueeze_1009);  convolution_126 = unsqueeze_1009 = None
        convert_element_type_379: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg633_1, torch.float32);  arg633_1 = None
        add_292: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_379, 1e-05);  convert_element_type_379 = None
        sqrt_126: "f32[1024]" = torch.ops.aten.sqrt.default(add_292);  add_292 = None
        reciprocal_126: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_126);  sqrt_126 = None
        mul_378: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_126, 1);  reciprocal_126 = None
        unsqueeze_1010: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_378, -1);  mul_378 = None
        unsqueeze_1011: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1010, -1);  unsqueeze_1010 = None
        mul_379: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_126, unsqueeze_1011);  sub_126 = unsqueeze_1011 = None
        unsqueeze_1012: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg634_1, -1);  arg634_1 = None
        unsqueeze_1013: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1012, -1);  unsqueeze_1012 = None
        mul_380: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_379, unsqueeze_1013);  mul_379 = unsqueeze_1013 = None
        unsqueeze_1014: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg635_1, -1);  arg635_1 = None
        unsqueeze_1015: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1014, -1);  unsqueeze_1014 = None
        add_293: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_380, unsqueeze_1015);  mul_380 = unsqueeze_1015 = None
        convert_element_type_380: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_293, torch.float16);  add_293 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_294: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_380, relu_120);  convert_element_type_380 = relu_120 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_123: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_294);  add_294 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_127: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_123, arg636_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg636_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_381: "f32[256]" = torch.ops.prims.convert_element_type.default(arg637_1, torch.float32);  arg637_1 = None
        unsqueeze_1016: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_381, -1);  convert_element_type_381 = None
        unsqueeze_1017: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1016, -1);  unsqueeze_1016 = None
        sub_127: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_127, unsqueeze_1017);  convolution_127 = unsqueeze_1017 = None
        convert_element_type_382: "f32[256]" = torch.ops.prims.convert_element_type.default(arg638_1, torch.float32);  arg638_1 = None
        add_295: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_382, 1e-05);  convert_element_type_382 = None
        sqrt_127: "f32[256]" = torch.ops.aten.sqrt.default(add_295);  add_295 = None
        reciprocal_127: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_127);  sqrt_127 = None
        mul_381: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_127, 1);  reciprocal_127 = None
        unsqueeze_1018: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_381, -1);  mul_381 = None
        unsqueeze_1019: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1018, -1);  unsqueeze_1018 = None
        mul_382: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_127, unsqueeze_1019);  sub_127 = unsqueeze_1019 = None
        unsqueeze_1020: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg639_1, -1);  arg639_1 = None
        unsqueeze_1021: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1020, -1);  unsqueeze_1020 = None
        mul_383: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_382, unsqueeze_1021);  mul_382 = unsqueeze_1021 = None
        unsqueeze_1022: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg640_1, -1);  arg640_1 = None
        unsqueeze_1023: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1022, -1);  unsqueeze_1022 = None
        add_296: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_383, unsqueeze_1023);  mul_383 = unsqueeze_1023 = None
        convert_element_type_383: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_296, torch.float16);  add_296 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_124: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_383);  convert_element_type_383 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_128: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_124, arg641_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_124 = arg641_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_384: "f32[256]" = torch.ops.prims.convert_element_type.default(arg642_1, torch.float32);  arg642_1 = None
        unsqueeze_1024: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_384, -1);  convert_element_type_384 = None
        unsqueeze_1025: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1024, -1);  unsqueeze_1024 = None
        sub_128: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_128, unsqueeze_1025);  convolution_128 = unsqueeze_1025 = None
        convert_element_type_385: "f32[256]" = torch.ops.prims.convert_element_type.default(arg643_1, torch.float32);  arg643_1 = None
        add_297: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_385, 1e-05);  convert_element_type_385 = None
        sqrt_128: "f32[256]" = torch.ops.aten.sqrt.default(add_297);  add_297 = None
        reciprocal_128: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_128);  sqrt_128 = None
        mul_384: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_128, 1);  reciprocal_128 = None
        unsqueeze_1026: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_384, -1);  mul_384 = None
        unsqueeze_1027: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1026, -1);  unsqueeze_1026 = None
        mul_385: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_128, unsqueeze_1027);  sub_128 = unsqueeze_1027 = None
        unsqueeze_1028: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg644_1, -1);  arg644_1 = None
        unsqueeze_1029: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1028, -1);  unsqueeze_1028 = None
        mul_386: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_385, unsqueeze_1029);  mul_385 = unsqueeze_1029 = None
        unsqueeze_1030: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg645_1, -1);  arg645_1 = None
        unsqueeze_1031: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1030, -1);  unsqueeze_1030 = None
        add_298: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_386, unsqueeze_1031);  mul_386 = unsqueeze_1031 = None
        convert_element_type_386: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_298, torch.float16);  add_298 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_125: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_386);  convert_element_type_386 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_129: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_125, arg646_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_125 = arg646_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_387: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg647_1, torch.float32);  arg647_1 = None
        unsqueeze_1032: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_387, -1);  convert_element_type_387 = None
        unsqueeze_1033: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1032, -1);  unsqueeze_1032 = None
        sub_129: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_129, unsqueeze_1033);  convolution_129 = unsqueeze_1033 = None
        convert_element_type_388: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg648_1, torch.float32);  arg648_1 = None
        add_299: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_388, 1e-05);  convert_element_type_388 = None
        sqrt_129: "f32[1024]" = torch.ops.aten.sqrt.default(add_299);  add_299 = None
        reciprocal_129: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_129);  sqrt_129 = None
        mul_387: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_129, 1);  reciprocal_129 = None
        unsqueeze_1034: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_387, -1);  mul_387 = None
        unsqueeze_1035: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1034, -1);  unsqueeze_1034 = None
        mul_388: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_129, unsqueeze_1035);  sub_129 = unsqueeze_1035 = None
        unsqueeze_1036: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg649_1, -1);  arg649_1 = None
        unsqueeze_1037: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1036, -1);  unsqueeze_1036 = None
        mul_389: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_388, unsqueeze_1037);  mul_388 = unsqueeze_1037 = None
        unsqueeze_1038: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg650_1, -1);  arg650_1 = None
        unsqueeze_1039: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1038, -1);  unsqueeze_1038 = None
        add_300: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_389, unsqueeze_1039);  mul_389 = unsqueeze_1039 = None
        convert_element_type_389: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_300, torch.float16);  add_300 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_301: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_389, relu_123);  convert_element_type_389 = relu_123 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_126: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_301);  add_301 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_130: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_126, arg651_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg651_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_390: "f32[256]" = torch.ops.prims.convert_element_type.default(arg652_1, torch.float32);  arg652_1 = None
        unsqueeze_1040: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_390, -1);  convert_element_type_390 = None
        unsqueeze_1041: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1040, -1);  unsqueeze_1040 = None
        sub_130: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_130, unsqueeze_1041);  convolution_130 = unsqueeze_1041 = None
        convert_element_type_391: "f32[256]" = torch.ops.prims.convert_element_type.default(arg653_1, torch.float32);  arg653_1 = None
        add_302: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_391, 1e-05);  convert_element_type_391 = None
        sqrt_130: "f32[256]" = torch.ops.aten.sqrt.default(add_302);  add_302 = None
        reciprocal_130: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_130);  sqrt_130 = None
        mul_390: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_130, 1);  reciprocal_130 = None
        unsqueeze_1042: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_390, -1);  mul_390 = None
        unsqueeze_1043: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1042, -1);  unsqueeze_1042 = None
        mul_391: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_130, unsqueeze_1043);  sub_130 = unsqueeze_1043 = None
        unsqueeze_1044: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg654_1, -1);  arg654_1 = None
        unsqueeze_1045: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1044, -1);  unsqueeze_1044 = None
        mul_392: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_391, unsqueeze_1045);  mul_391 = unsqueeze_1045 = None
        unsqueeze_1046: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg655_1, -1);  arg655_1 = None
        unsqueeze_1047: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1046, -1);  unsqueeze_1046 = None
        add_303: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_392, unsqueeze_1047);  mul_392 = unsqueeze_1047 = None
        convert_element_type_392: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_303, torch.float16);  add_303 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_127: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_392);  convert_element_type_392 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_131: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_127, arg656_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_127 = arg656_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_393: "f32[256]" = torch.ops.prims.convert_element_type.default(arg657_1, torch.float32);  arg657_1 = None
        unsqueeze_1048: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_393, -1);  convert_element_type_393 = None
        unsqueeze_1049: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1048, -1);  unsqueeze_1048 = None
        sub_131: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_131, unsqueeze_1049);  convolution_131 = unsqueeze_1049 = None
        convert_element_type_394: "f32[256]" = torch.ops.prims.convert_element_type.default(arg658_1, torch.float32);  arg658_1 = None
        add_304: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_394, 1e-05);  convert_element_type_394 = None
        sqrt_131: "f32[256]" = torch.ops.aten.sqrt.default(add_304);  add_304 = None
        reciprocal_131: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_131);  sqrt_131 = None
        mul_393: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_131, 1);  reciprocal_131 = None
        unsqueeze_1050: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_393, -1);  mul_393 = None
        unsqueeze_1051: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1050, -1);  unsqueeze_1050 = None
        mul_394: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_131, unsqueeze_1051);  sub_131 = unsqueeze_1051 = None
        unsqueeze_1052: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg659_1, -1);  arg659_1 = None
        unsqueeze_1053: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1052, -1);  unsqueeze_1052 = None
        mul_395: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_394, unsqueeze_1053);  mul_394 = unsqueeze_1053 = None
        unsqueeze_1054: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg660_1, -1);  arg660_1 = None
        unsqueeze_1055: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1054, -1);  unsqueeze_1054 = None
        add_305: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_395, unsqueeze_1055);  mul_395 = unsqueeze_1055 = None
        convert_element_type_395: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_305, torch.float16);  add_305 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_128: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_395);  convert_element_type_395 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_132: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_128, arg661_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_128 = arg661_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_396: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg662_1, torch.float32);  arg662_1 = None
        unsqueeze_1056: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_396, -1);  convert_element_type_396 = None
        unsqueeze_1057: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1056, -1);  unsqueeze_1056 = None
        sub_132: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_132, unsqueeze_1057);  convolution_132 = unsqueeze_1057 = None
        convert_element_type_397: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg663_1, torch.float32);  arg663_1 = None
        add_306: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_397, 1e-05);  convert_element_type_397 = None
        sqrt_132: "f32[1024]" = torch.ops.aten.sqrt.default(add_306);  add_306 = None
        reciprocal_132: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_132);  sqrt_132 = None
        mul_396: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_132, 1);  reciprocal_132 = None
        unsqueeze_1058: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_396, -1);  mul_396 = None
        unsqueeze_1059: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1058, -1);  unsqueeze_1058 = None
        mul_397: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_132, unsqueeze_1059);  sub_132 = unsqueeze_1059 = None
        unsqueeze_1060: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg664_1, -1);  arg664_1 = None
        unsqueeze_1061: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1060, -1);  unsqueeze_1060 = None
        mul_398: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_397, unsqueeze_1061);  mul_397 = unsqueeze_1061 = None
        unsqueeze_1062: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg665_1, -1);  arg665_1 = None
        unsqueeze_1063: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1062, -1);  unsqueeze_1062 = None
        add_307: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_398, unsqueeze_1063);  mul_398 = unsqueeze_1063 = None
        convert_element_type_398: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_307, torch.float16);  add_307 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_308: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_398, relu_126);  convert_element_type_398 = relu_126 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_129: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_308);  add_308 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_133: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_129, arg666_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg666_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_399: "f32[256]" = torch.ops.prims.convert_element_type.default(arg667_1, torch.float32);  arg667_1 = None
        unsqueeze_1064: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_399, -1);  convert_element_type_399 = None
        unsqueeze_1065: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1064, -1);  unsqueeze_1064 = None
        sub_133: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_133, unsqueeze_1065);  convolution_133 = unsqueeze_1065 = None
        convert_element_type_400: "f32[256]" = torch.ops.prims.convert_element_type.default(arg668_1, torch.float32);  arg668_1 = None
        add_309: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_400, 1e-05);  convert_element_type_400 = None
        sqrt_133: "f32[256]" = torch.ops.aten.sqrt.default(add_309);  add_309 = None
        reciprocal_133: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_133);  sqrt_133 = None
        mul_399: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_133, 1);  reciprocal_133 = None
        unsqueeze_1066: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_399, -1);  mul_399 = None
        unsqueeze_1067: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1066, -1);  unsqueeze_1066 = None
        mul_400: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_133, unsqueeze_1067);  sub_133 = unsqueeze_1067 = None
        unsqueeze_1068: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg669_1, -1);  arg669_1 = None
        unsqueeze_1069: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1068, -1);  unsqueeze_1068 = None
        mul_401: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_400, unsqueeze_1069);  mul_400 = unsqueeze_1069 = None
        unsqueeze_1070: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg670_1, -1);  arg670_1 = None
        unsqueeze_1071: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1070, -1);  unsqueeze_1070 = None
        add_310: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_401, unsqueeze_1071);  mul_401 = unsqueeze_1071 = None
        convert_element_type_401: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_310, torch.float16);  add_310 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_130: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_401);  convert_element_type_401 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_134: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_130, arg671_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_130 = arg671_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_402: "f32[256]" = torch.ops.prims.convert_element_type.default(arg672_1, torch.float32);  arg672_1 = None
        unsqueeze_1072: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_402, -1);  convert_element_type_402 = None
        unsqueeze_1073: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1072, -1);  unsqueeze_1072 = None
        sub_134: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_134, unsqueeze_1073);  convolution_134 = unsqueeze_1073 = None
        convert_element_type_403: "f32[256]" = torch.ops.prims.convert_element_type.default(arg673_1, torch.float32);  arg673_1 = None
        add_311: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_403, 1e-05);  convert_element_type_403 = None
        sqrt_134: "f32[256]" = torch.ops.aten.sqrt.default(add_311);  add_311 = None
        reciprocal_134: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_134);  sqrt_134 = None
        mul_402: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_134, 1);  reciprocal_134 = None
        unsqueeze_1074: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_402, -1);  mul_402 = None
        unsqueeze_1075: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1074, -1);  unsqueeze_1074 = None
        mul_403: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_134, unsqueeze_1075);  sub_134 = unsqueeze_1075 = None
        unsqueeze_1076: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg674_1, -1);  arg674_1 = None
        unsqueeze_1077: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1076, -1);  unsqueeze_1076 = None
        mul_404: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_403, unsqueeze_1077);  mul_403 = unsqueeze_1077 = None
        unsqueeze_1078: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg675_1, -1);  arg675_1 = None
        unsqueeze_1079: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1078, -1);  unsqueeze_1078 = None
        add_312: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_404, unsqueeze_1079);  mul_404 = unsqueeze_1079 = None
        convert_element_type_404: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_312, torch.float16);  add_312 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_131: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_404);  convert_element_type_404 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_135: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_131, arg676_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_131 = arg676_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_405: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg677_1, torch.float32);  arg677_1 = None
        unsqueeze_1080: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_405, -1);  convert_element_type_405 = None
        unsqueeze_1081: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1080, -1);  unsqueeze_1080 = None
        sub_135: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_135, unsqueeze_1081);  convolution_135 = unsqueeze_1081 = None
        convert_element_type_406: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg678_1, torch.float32);  arg678_1 = None
        add_313: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_406, 1e-05);  convert_element_type_406 = None
        sqrt_135: "f32[1024]" = torch.ops.aten.sqrt.default(add_313);  add_313 = None
        reciprocal_135: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_135);  sqrt_135 = None
        mul_405: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_135, 1);  reciprocal_135 = None
        unsqueeze_1082: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_405, -1);  mul_405 = None
        unsqueeze_1083: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1082, -1);  unsqueeze_1082 = None
        mul_406: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_135, unsqueeze_1083);  sub_135 = unsqueeze_1083 = None
        unsqueeze_1084: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg679_1, -1);  arg679_1 = None
        unsqueeze_1085: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1084, -1);  unsqueeze_1084 = None
        mul_407: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_406, unsqueeze_1085);  mul_406 = unsqueeze_1085 = None
        unsqueeze_1086: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg680_1, -1);  arg680_1 = None
        unsqueeze_1087: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1086, -1);  unsqueeze_1086 = None
        add_314: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_407, unsqueeze_1087);  mul_407 = unsqueeze_1087 = None
        convert_element_type_407: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_314, torch.float16);  add_314 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_315: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_407, relu_129);  convert_element_type_407 = relu_129 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_132: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_315);  add_315 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_136: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_132, arg681_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg681_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_408: "f32[256]" = torch.ops.prims.convert_element_type.default(arg682_1, torch.float32);  arg682_1 = None
        unsqueeze_1088: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_408, -1);  convert_element_type_408 = None
        unsqueeze_1089: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1088, -1);  unsqueeze_1088 = None
        sub_136: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_136, unsqueeze_1089);  convolution_136 = unsqueeze_1089 = None
        convert_element_type_409: "f32[256]" = torch.ops.prims.convert_element_type.default(arg683_1, torch.float32);  arg683_1 = None
        add_316: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_409, 1e-05);  convert_element_type_409 = None
        sqrt_136: "f32[256]" = torch.ops.aten.sqrt.default(add_316);  add_316 = None
        reciprocal_136: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_136);  sqrt_136 = None
        mul_408: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_136, 1);  reciprocal_136 = None
        unsqueeze_1090: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_408, -1);  mul_408 = None
        unsqueeze_1091: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1090, -1);  unsqueeze_1090 = None
        mul_409: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_136, unsqueeze_1091);  sub_136 = unsqueeze_1091 = None
        unsqueeze_1092: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg684_1, -1);  arg684_1 = None
        unsqueeze_1093: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1092, -1);  unsqueeze_1092 = None
        mul_410: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_409, unsqueeze_1093);  mul_409 = unsqueeze_1093 = None
        unsqueeze_1094: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg685_1, -1);  arg685_1 = None
        unsqueeze_1095: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1094, -1);  unsqueeze_1094 = None
        add_317: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_410, unsqueeze_1095);  mul_410 = unsqueeze_1095 = None
        convert_element_type_410: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_317, torch.float16);  add_317 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_133: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_410);  convert_element_type_410 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_137: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_133, arg686_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_133 = arg686_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_411: "f32[256]" = torch.ops.prims.convert_element_type.default(arg687_1, torch.float32);  arg687_1 = None
        unsqueeze_1096: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_411, -1);  convert_element_type_411 = None
        unsqueeze_1097: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1096, -1);  unsqueeze_1096 = None
        sub_137: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_137, unsqueeze_1097);  convolution_137 = unsqueeze_1097 = None
        convert_element_type_412: "f32[256]" = torch.ops.prims.convert_element_type.default(arg688_1, torch.float32);  arg688_1 = None
        add_318: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_412, 1e-05);  convert_element_type_412 = None
        sqrt_137: "f32[256]" = torch.ops.aten.sqrt.default(add_318);  add_318 = None
        reciprocal_137: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_137);  sqrt_137 = None
        mul_411: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_137, 1);  reciprocal_137 = None
        unsqueeze_1098: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_411, -1);  mul_411 = None
        unsqueeze_1099: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1098, -1);  unsqueeze_1098 = None
        mul_412: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_137, unsqueeze_1099);  sub_137 = unsqueeze_1099 = None
        unsqueeze_1100: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg689_1, -1);  arg689_1 = None
        unsqueeze_1101: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1100, -1);  unsqueeze_1100 = None
        mul_413: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_412, unsqueeze_1101);  mul_412 = unsqueeze_1101 = None
        unsqueeze_1102: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg690_1, -1);  arg690_1 = None
        unsqueeze_1103: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1102, -1);  unsqueeze_1102 = None
        add_319: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_413, unsqueeze_1103);  mul_413 = unsqueeze_1103 = None
        convert_element_type_413: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_319, torch.float16);  add_319 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_134: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_413);  convert_element_type_413 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_138: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_134, arg691_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_134 = arg691_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_414: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg692_1, torch.float32);  arg692_1 = None
        unsqueeze_1104: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_414, -1);  convert_element_type_414 = None
        unsqueeze_1105: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1104, -1);  unsqueeze_1104 = None
        sub_138: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_138, unsqueeze_1105);  convolution_138 = unsqueeze_1105 = None
        convert_element_type_415: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg693_1, torch.float32);  arg693_1 = None
        add_320: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_415, 1e-05);  convert_element_type_415 = None
        sqrt_138: "f32[1024]" = torch.ops.aten.sqrt.default(add_320);  add_320 = None
        reciprocal_138: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_138);  sqrt_138 = None
        mul_414: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_138, 1);  reciprocal_138 = None
        unsqueeze_1106: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_414, -1);  mul_414 = None
        unsqueeze_1107: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1106, -1);  unsqueeze_1106 = None
        mul_415: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_138, unsqueeze_1107);  sub_138 = unsqueeze_1107 = None
        unsqueeze_1108: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg694_1, -1);  arg694_1 = None
        unsqueeze_1109: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1108, -1);  unsqueeze_1108 = None
        mul_416: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_415, unsqueeze_1109);  mul_415 = unsqueeze_1109 = None
        unsqueeze_1110: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg695_1, -1);  arg695_1 = None
        unsqueeze_1111: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1110, -1);  unsqueeze_1110 = None
        add_321: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_416, unsqueeze_1111);  mul_416 = unsqueeze_1111 = None
        convert_element_type_416: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_321, torch.float16);  add_321 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_322: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_416, relu_132);  convert_element_type_416 = relu_132 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_135: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_322);  add_322 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_139: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_135, arg696_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg696_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_417: "f32[256]" = torch.ops.prims.convert_element_type.default(arg697_1, torch.float32);  arg697_1 = None
        unsqueeze_1112: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_417, -1);  convert_element_type_417 = None
        unsqueeze_1113: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1112, -1);  unsqueeze_1112 = None
        sub_139: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_139, unsqueeze_1113);  convolution_139 = unsqueeze_1113 = None
        convert_element_type_418: "f32[256]" = torch.ops.prims.convert_element_type.default(arg698_1, torch.float32);  arg698_1 = None
        add_323: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_418, 1e-05);  convert_element_type_418 = None
        sqrt_139: "f32[256]" = torch.ops.aten.sqrt.default(add_323);  add_323 = None
        reciprocal_139: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_139);  sqrt_139 = None
        mul_417: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_139, 1);  reciprocal_139 = None
        unsqueeze_1114: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_417, -1);  mul_417 = None
        unsqueeze_1115: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1114, -1);  unsqueeze_1114 = None
        mul_418: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_139, unsqueeze_1115);  sub_139 = unsqueeze_1115 = None
        unsqueeze_1116: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg699_1, -1);  arg699_1 = None
        unsqueeze_1117: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1116, -1);  unsqueeze_1116 = None
        mul_419: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_418, unsqueeze_1117);  mul_418 = unsqueeze_1117 = None
        unsqueeze_1118: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg700_1, -1);  arg700_1 = None
        unsqueeze_1119: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1118, -1);  unsqueeze_1118 = None
        add_324: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_419, unsqueeze_1119);  mul_419 = unsqueeze_1119 = None
        convert_element_type_419: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_324, torch.float16);  add_324 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_136: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_419);  convert_element_type_419 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_140: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_136, arg701_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_136 = arg701_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_420: "f32[256]" = torch.ops.prims.convert_element_type.default(arg702_1, torch.float32);  arg702_1 = None
        unsqueeze_1120: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_420, -1);  convert_element_type_420 = None
        unsqueeze_1121: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1120, -1);  unsqueeze_1120 = None
        sub_140: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_140, unsqueeze_1121);  convolution_140 = unsqueeze_1121 = None
        convert_element_type_421: "f32[256]" = torch.ops.prims.convert_element_type.default(arg703_1, torch.float32);  arg703_1 = None
        add_325: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_421, 1e-05);  convert_element_type_421 = None
        sqrt_140: "f32[256]" = torch.ops.aten.sqrt.default(add_325);  add_325 = None
        reciprocal_140: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_140);  sqrt_140 = None
        mul_420: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_140, 1);  reciprocal_140 = None
        unsqueeze_1122: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_420, -1);  mul_420 = None
        unsqueeze_1123: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1122, -1);  unsqueeze_1122 = None
        mul_421: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_140, unsqueeze_1123);  sub_140 = unsqueeze_1123 = None
        unsqueeze_1124: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg704_1, -1);  arg704_1 = None
        unsqueeze_1125: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1124, -1);  unsqueeze_1124 = None
        mul_422: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_421, unsqueeze_1125);  mul_421 = unsqueeze_1125 = None
        unsqueeze_1126: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg705_1, -1);  arg705_1 = None
        unsqueeze_1127: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1126, -1);  unsqueeze_1126 = None
        add_326: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_422, unsqueeze_1127);  mul_422 = unsqueeze_1127 = None
        convert_element_type_422: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_326, torch.float16);  add_326 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_137: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_422);  convert_element_type_422 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_141: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_137, arg706_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_137 = arg706_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_423: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg707_1, torch.float32);  arg707_1 = None
        unsqueeze_1128: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_423, -1);  convert_element_type_423 = None
        unsqueeze_1129: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1128, -1);  unsqueeze_1128 = None
        sub_141: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_141, unsqueeze_1129);  convolution_141 = unsqueeze_1129 = None
        convert_element_type_424: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg708_1, torch.float32);  arg708_1 = None
        add_327: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_424, 1e-05);  convert_element_type_424 = None
        sqrt_141: "f32[1024]" = torch.ops.aten.sqrt.default(add_327);  add_327 = None
        reciprocal_141: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_141);  sqrt_141 = None
        mul_423: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_141, 1);  reciprocal_141 = None
        unsqueeze_1130: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_423, -1);  mul_423 = None
        unsqueeze_1131: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1130, -1);  unsqueeze_1130 = None
        mul_424: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_141, unsqueeze_1131);  sub_141 = unsqueeze_1131 = None
        unsqueeze_1132: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg709_1, -1);  arg709_1 = None
        unsqueeze_1133: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1132, -1);  unsqueeze_1132 = None
        mul_425: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_424, unsqueeze_1133);  mul_424 = unsqueeze_1133 = None
        unsqueeze_1134: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg710_1, -1);  arg710_1 = None
        unsqueeze_1135: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1134, -1);  unsqueeze_1134 = None
        add_328: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_425, unsqueeze_1135);  mul_425 = unsqueeze_1135 = None
        convert_element_type_425: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_328, torch.float16);  add_328 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_329: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_425, relu_135);  convert_element_type_425 = relu_135 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_138: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_329);  add_329 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_142: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_138, arg711_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg711_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_426: "f32[256]" = torch.ops.prims.convert_element_type.default(arg712_1, torch.float32);  arg712_1 = None
        unsqueeze_1136: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_426, -1);  convert_element_type_426 = None
        unsqueeze_1137: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1136, -1);  unsqueeze_1136 = None
        sub_142: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_142, unsqueeze_1137);  convolution_142 = unsqueeze_1137 = None
        convert_element_type_427: "f32[256]" = torch.ops.prims.convert_element_type.default(arg713_1, torch.float32);  arg713_1 = None
        add_330: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_427, 1e-05);  convert_element_type_427 = None
        sqrt_142: "f32[256]" = torch.ops.aten.sqrt.default(add_330);  add_330 = None
        reciprocal_142: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_142);  sqrt_142 = None
        mul_426: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_142, 1);  reciprocal_142 = None
        unsqueeze_1138: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_426, -1);  mul_426 = None
        unsqueeze_1139: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1138, -1);  unsqueeze_1138 = None
        mul_427: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_142, unsqueeze_1139);  sub_142 = unsqueeze_1139 = None
        unsqueeze_1140: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg714_1, -1);  arg714_1 = None
        unsqueeze_1141: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1140, -1);  unsqueeze_1140 = None
        mul_428: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_427, unsqueeze_1141);  mul_427 = unsqueeze_1141 = None
        unsqueeze_1142: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg715_1, -1);  arg715_1 = None
        unsqueeze_1143: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1142, -1);  unsqueeze_1142 = None
        add_331: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_428, unsqueeze_1143);  mul_428 = unsqueeze_1143 = None
        convert_element_type_428: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_331, torch.float16);  add_331 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_139: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_428);  convert_element_type_428 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_143: "f16[32, 256, 14, 14]" = torch.ops.aten.convolution.default(relu_139, arg716_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_139 = arg716_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_429: "f32[256]" = torch.ops.prims.convert_element_type.default(arg717_1, torch.float32);  arg717_1 = None
        unsqueeze_1144: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_429, -1);  convert_element_type_429 = None
        unsqueeze_1145: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1144, -1);  unsqueeze_1144 = None
        sub_143: "f32[32, 256, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_143, unsqueeze_1145);  convolution_143 = unsqueeze_1145 = None
        convert_element_type_430: "f32[256]" = torch.ops.prims.convert_element_type.default(arg718_1, torch.float32);  arg718_1 = None
        add_332: "f32[256]" = torch.ops.aten.add.Tensor(convert_element_type_430, 1e-05);  convert_element_type_430 = None
        sqrt_143: "f32[256]" = torch.ops.aten.sqrt.default(add_332);  add_332 = None
        reciprocal_143: "f32[256]" = torch.ops.aten.reciprocal.default(sqrt_143);  sqrt_143 = None
        mul_429: "f32[256]" = torch.ops.aten.mul.Tensor(reciprocal_143, 1);  reciprocal_143 = None
        unsqueeze_1146: "f32[256, 1]" = torch.ops.aten.unsqueeze.default(mul_429, -1);  mul_429 = None
        unsqueeze_1147: "f32[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1146, -1);  unsqueeze_1146 = None
        mul_430: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(sub_143, unsqueeze_1147);  sub_143 = unsqueeze_1147 = None
        unsqueeze_1148: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg719_1, -1);  arg719_1 = None
        unsqueeze_1149: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1148, -1);  unsqueeze_1148 = None
        mul_431: "f32[32, 256, 14, 14]" = torch.ops.aten.mul.Tensor(mul_430, unsqueeze_1149);  mul_430 = unsqueeze_1149 = None
        unsqueeze_1150: "f16[256, 1]" = torch.ops.aten.unsqueeze.default(arg720_1, -1);  arg720_1 = None
        unsqueeze_1151: "f16[256, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1150, -1);  unsqueeze_1150 = None
        add_333: "f32[32, 256, 14, 14]" = torch.ops.aten.add.Tensor(mul_431, unsqueeze_1151);  mul_431 = unsqueeze_1151 = None
        convert_element_type_431: "f16[32, 256, 14, 14]" = torch.ops.prims.convert_element_type.default(add_333, torch.float16);  add_333 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_140: "f16[32, 256, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_431);  convert_element_type_431 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_144: "f16[32, 1024, 14, 14]" = torch.ops.aten.convolution.default(relu_140, arg721_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_140 = arg721_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_432: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg722_1, torch.float32);  arg722_1 = None
        unsqueeze_1152: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_432, -1);  convert_element_type_432 = None
        unsqueeze_1153: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1152, -1);  unsqueeze_1152 = None
        sub_144: "f32[32, 1024, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_144, unsqueeze_1153);  convolution_144 = unsqueeze_1153 = None
        convert_element_type_433: "f32[1024]" = torch.ops.prims.convert_element_type.default(arg723_1, torch.float32);  arg723_1 = None
        add_334: "f32[1024]" = torch.ops.aten.add.Tensor(convert_element_type_433, 1e-05);  convert_element_type_433 = None
        sqrt_144: "f32[1024]" = torch.ops.aten.sqrt.default(add_334);  add_334 = None
        reciprocal_144: "f32[1024]" = torch.ops.aten.reciprocal.default(sqrt_144);  sqrt_144 = None
        mul_432: "f32[1024]" = torch.ops.aten.mul.Tensor(reciprocal_144, 1);  reciprocal_144 = None
        unsqueeze_1154: "f32[1024, 1]" = torch.ops.aten.unsqueeze.default(mul_432, -1);  mul_432 = None
        unsqueeze_1155: "f32[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1154, -1);  unsqueeze_1154 = None
        mul_433: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(sub_144, unsqueeze_1155);  sub_144 = unsqueeze_1155 = None
        unsqueeze_1156: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg724_1, -1);  arg724_1 = None
        unsqueeze_1157: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1156, -1);  unsqueeze_1156 = None
        mul_434: "f32[32, 1024, 14, 14]" = torch.ops.aten.mul.Tensor(mul_433, unsqueeze_1157);  mul_433 = unsqueeze_1157 = None
        unsqueeze_1158: "f16[1024, 1]" = torch.ops.aten.unsqueeze.default(arg725_1, -1);  arg725_1 = None
        unsqueeze_1159: "f16[1024, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1158, -1);  unsqueeze_1158 = None
        add_335: "f32[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(mul_434, unsqueeze_1159);  mul_434 = unsqueeze_1159 = None
        convert_element_type_434: "f16[32, 1024, 14, 14]" = torch.ops.prims.convert_element_type.default(add_335, torch.float16);  add_335 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_336: "f16[32, 1024, 14, 14]" = torch.ops.aten.add.Tensor(convert_element_type_434, relu_138);  convert_element_type_434 = relu_138 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_141: "f16[32, 1024, 14, 14]" = torch.ops.aten.relu.default(add_336);  add_336 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_145: "f16[32, 512, 14, 14]" = torch.ops.aten.convolution.default(relu_141, arg726_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg726_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_435: "f32[512]" = torch.ops.prims.convert_element_type.default(arg727_1, torch.float32);  arg727_1 = None
        unsqueeze_1160: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_435, -1);  convert_element_type_435 = None
        unsqueeze_1161: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1160, -1);  unsqueeze_1160 = None
        sub_145: "f32[32, 512, 14, 14]" = torch.ops.aten.sub.Tensor(convolution_145, unsqueeze_1161);  convolution_145 = unsqueeze_1161 = None
        convert_element_type_436: "f32[512]" = torch.ops.prims.convert_element_type.default(arg728_1, torch.float32);  arg728_1 = None
        add_337: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_436, 1e-05);  convert_element_type_436 = None
        sqrt_145: "f32[512]" = torch.ops.aten.sqrt.default(add_337);  add_337 = None
        reciprocal_145: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_145);  sqrt_145 = None
        mul_435: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_145, 1);  reciprocal_145 = None
        unsqueeze_1162: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_435, -1);  mul_435 = None
        unsqueeze_1163: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1162, -1);  unsqueeze_1162 = None
        mul_436: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(sub_145, unsqueeze_1163);  sub_145 = unsqueeze_1163 = None
        unsqueeze_1164: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg729_1, -1);  arg729_1 = None
        unsqueeze_1165: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1164, -1);  unsqueeze_1164 = None
        mul_437: "f32[32, 512, 14, 14]" = torch.ops.aten.mul.Tensor(mul_436, unsqueeze_1165);  mul_436 = unsqueeze_1165 = None
        unsqueeze_1166: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg730_1, -1);  arg730_1 = None
        unsqueeze_1167: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1166, -1);  unsqueeze_1166 = None
        add_338: "f32[32, 512, 14, 14]" = torch.ops.aten.add.Tensor(mul_437, unsqueeze_1167);  mul_437 = unsqueeze_1167 = None
        convert_element_type_437: "f16[32, 512, 14, 14]" = torch.ops.prims.convert_element_type.default(add_338, torch.float16);  add_338 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_142: "f16[32, 512, 14, 14]" = torch.ops.aten.relu.default(convert_element_type_437);  convert_element_type_437 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_146: "f16[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_142, arg731_1, None, [2, 2], [1, 1], [1, 1], False, [0, 0], 1);  relu_142 = arg731_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_438: "f32[512]" = torch.ops.prims.convert_element_type.default(arg732_1, torch.float32);  arg732_1 = None
        unsqueeze_1168: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_438, -1);  convert_element_type_438 = None
        unsqueeze_1169: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1168, -1);  unsqueeze_1168 = None
        sub_146: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_146, unsqueeze_1169);  convolution_146 = unsqueeze_1169 = None
        convert_element_type_439: "f32[512]" = torch.ops.prims.convert_element_type.default(arg733_1, torch.float32);  arg733_1 = None
        add_339: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_439, 1e-05);  convert_element_type_439 = None
        sqrt_146: "f32[512]" = torch.ops.aten.sqrt.default(add_339);  add_339 = None
        reciprocal_146: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_146);  sqrt_146 = None
        mul_438: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_146, 1);  reciprocal_146 = None
        unsqueeze_1170: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_438, -1);  mul_438 = None
        unsqueeze_1171: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1170, -1);  unsqueeze_1170 = None
        mul_439: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_146, unsqueeze_1171);  sub_146 = unsqueeze_1171 = None
        unsqueeze_1172: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg734_1, -1);  arg734_1 = None
        unsqueeze_1173: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1172, -1);  unsqueeze_1172 = None
        mul_440: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_439, unsqueeze_1173);  mul_439 = unsqueeze_1173 = None
        unsqueeze_1174: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg735_1, -1);  arg735_1 = None
        unsqueeze_1175: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1174, -1);  unsqueeze_1174 = None
        add_340: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_440, unsqueeze_1175);  mul_440 = unsqueeze_1175 = None
        convert_element_type_440: "f16[32, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_340, torch.float16);  add_340 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_143: "f16[32, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_440);  convert_element_type_440 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_147: "f16[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_143, arg736_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_143 = arg736_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_441: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg737_1, torch.float32);  arg737_1 = None
        unsqueeze_1176: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_441, -1);  convert_element_type_441 = None
        unsqueeze_1177: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1176, -1);  unsqueeze_1176 = None
        sub_147: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_147, unsqueeze_1177);  convolution_147 = unsqueeze_1177 = None
        convert_element_type_442: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg738_1, torch.float32);  arg738_1 = None
        add_341: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_442, 1e-05);  convert_element_type_442 = None
        sqrt_147: "f32[2048]" = torch.ops.aten.sqrt.default(add_341);  add_341 = None
        reciprocal_147: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_147);  sqrt_147 = None
        mul_441: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_147, 1);  reciprocal_147 = None
        unsqueeze_1178: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_441, -1);  mul_441 = None
        unsqueeze_1179: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1178, -1);  unsqueeze_1178 = None
        mul_442: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_147, unsqueeze_1179);  sub_147 = unsqueeze_1179 = None
        unsqueeze_1180: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg739_1, -1);  arg739_1 = None
        unsqueeze_1181: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1180, -1);  unsqueeze_1180 = None
        mul_443: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_442, unsqueeze_1181);  mul_442 = unsqueeze_1181 = None
        unsqueeze_1182: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg740_1, -1);  arg740_1 = None
        unsqueeze_1183: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1182, -1);  unsqueeze_1182 = None
        add_342: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_443, unsqueeze_1183);  mul_443 = unsqueeze_1183 = None
        convert_element_type_443: "f16[32, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_342, torch.float16);  add_342 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:158 in forward, code: identity = self.downsample(x)
        convolution_148: "f16[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_141, arg741_1, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1);  relu_141 = arg741_1 = None
        convert_element_type_444: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg742_1, torch.float32);  arg742_1 = None
        unsqueeze_1184: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_444, -1);  convert_element_type_444 = None
        unsqueeze_1185: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1184, -1);  unsqueeze_1184 = None
        sub_148: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_148, unsqueeze_1185);  convolution_148 = unsqueeze_1185 = None
        convert_element_type_445: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg743_1, torch.float32);  arg743_1 = None
        add_343: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_445, 1e-05);  convert_element_type_445 = None
        sqrt_148: "f32[2048]" = torch.ops.aten.sqrt.default(add_343);  add_343 = None
        reciprocal_148: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_148);  sqrt_148 = None
        mul_444: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_148, 1);  reciprocal_148 = None
        unsqueeze_1186: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_444, -1);  mul_444 = None
        unsqueeze_1187: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1186, -1);  unsqueeze_1186 = None
        mul_445: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_148, unsqueeze_1187);  sub_148 = unsqueeze_1187 = None
        unsqueeze_1188: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg744_1, -1);  arg744_1 = None
        unsqueeze_1189: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1188, -1);  unsqueeze_1188 = None
        mul_446: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_445, unsqueeze_1189);  mul_445 = unsqueeze_1189 = None
        unsqueeze_1190: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg745_1, -1);  arg745_1 = None
        unsqueeze_1191: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1190, -1);  unsqueeze_1190 = None
        add_344: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_446, unsqueeze_1191);  mul_446 = unsqueeze_1191 = None
        convert_element_type_446: "f16[32, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_344, torch.float16);  add_344 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_345: "f16[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_443, convert_element_type_446);  convert_element_type_443 = convert_element_type_446 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_144: "f16[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_345);  add_345 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_149: "f16[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_144, arg746_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg746_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_447: "f32[512]" = torch.ops.prims.convert_element_type.default(arg747_1, torch.float32);  arg747_1 = None
        unsqueeze_1192: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_447, -1);  convert_element_type_447 = None
        unsqueeze_1193: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1192, -1);  unsqueeze_1192 = None
        sub_149: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_149, unsqueeze_1193);  convolution_149 = unsqueeze_1193 = None
        convert_element_type_448: "f32[512]" = torch.ops.prims.convert_element_type.default(arg748_1, torch.float32);  arg748_1 = None
        add_346: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_448, 1e-05);  convert_element_type_448 = None
        sqrt_149: "f32[512]" = torch.ops.aten.sqrt.default(add_346);  add_346 = None
        reciprocal_149: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_149);  sqrt_149 = None
        mul_447: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_149, 1);  reciprocal_149 = None
        unsqueeze_1194: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_447, -1);  mul_447 = None
        unsqueeze_1195: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1194, -1);  unsqueeze_1194 = None
        mul_448: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_149, unsqueeze_1195);  sub_149 = unsqueeze_1195 = None
        unsqueeze_1196: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg749_1, -1);  arg749_1 = None
        unsqueeze_1197: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1196, -1);  unsqueeze_1196 = None
        mul_449: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_448, unsqueeze_1197);  mul_448 = unsqueeze_1197 = None
        unsqueeze_1198: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg750_1, -1);  arg750_1 = None
        unsqueeze_1199: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1198, -1);  unsqueeze_1198 = None
        add_347: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_449, unsqueeze_1199);  mul_449 = unsqueeze_1199 = None
        convert_element_type_449: "f16[32, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_347, torch.float16);  add_347 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_145: "f16[32, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_449);  convert_element_type_449 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_150: "f16[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_145, arg751_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_145 = arg751_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_450: "f32[512]" = torch.ops.prims.convert_element_type.default(arg752_1, torch.float32);  arg752_1 = None
        unsqueeze_1200: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_450, -1);  convert_element_type_450 = None
        unsqueeze_1201: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1200, -1);  unsqueeze_1200 = None
        sub_150: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_150, unsqueeze_1201);  convolution_150 = unsqueeze_1201 = None
        convert_element_type_451: "f32[512]" = torch.ops.prims.convert_element_type.default(arg753_1, torch.float32);  arg753_1 = None
        add_348: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_451, 1e-05);  convert_element_type_451 = None
        sqrt_150: "f32[512]" = torch.ops.aten.sqrt.default(add_348);  add_348 = None
        reciprocal_150: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_150);  sqrt_150 = None
        mul_450: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_150, 1);  reciprocal_150 = None
        unsqueeze_1202: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_450, -1);  mul_450 = None
        unsqueeze_1203: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1202, -1);  unsqueeze_1202 = None
        mul_451: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_150, unsqueeze_1203);  sub_150 = unsqueeze_1203 = None
        unsqueeze_1204: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg754_1, -1);  arg754_1 = None
        unsqueeze_1205: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1204, -1);  unsqueeze_1204 = None
        mul_452: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_451, unsqueeze_1205);  mul_451 = unsqueeze_1205 = None
        unsqueeze_1206: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg755_1, -1);  arg755_1 = None
        unsqueeze_1207: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1206, -1);  unsqueeze_1206 = None
        add_349: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_452, unsqueeze_1207);  mul_452 = unsqueeze_1207 = None
        convert_element_type_452: "f16[32, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_349, torch.float16);  add_349 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_146: "f16[32, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_452);  convert_element_type_452 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_151: "f16[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_146, arg756_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_146 = arg756_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_453: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg757_1, torch.float32);  arg757_1 = None
        unsqueeze_1208: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_453, -1);  convert_element_type_453 = None
        unsqueeze_1209: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1208, -1);  unsqueeze_1208 = None
        sub_151: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_151, unsqueeze_1209);  convolution_151 = unsqueeze_1209 = None
        convert_element_type_454: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg758_1, torch.float32);  arg758_1 = None
        add_350: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_454, 1e-05);  convert_element_type_454 = None
        sqrt_151: "f32[2048]" = torch.ops.aten.sqrt.default(add_350);  add_350 = None
        reciprocal_151: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_151);  sqrt_151 = None
        mul_453: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_151, 1);  reciprocal_151 = None
        unsqueeze_1210: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_453, -1);  mul_453 = None
        unsqueeze_1211: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1210, -1);  unsqueeze_1210 = None
        mul_454: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_151, unsqueeze_1211);  sub_151 = unsqueeze_1211 = None
        unsqueeze_1212: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg759_1, -1);  arg759_1 = None
        unsqueeze_1213: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1212, -1);  unsqueeze_1212 = None
        mul_455: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_454, unsqueeze_1213);  mul_454 = unsqueeze_1213 = None
        unsqueeze_1214: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg760_1, -1);  arg760_1 = None
        unsqueeze_1215: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1214, -1);  unsqueeze_1214 = None
        add_351: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_455, unsqueeze_1215);  mul_455 = unsqueeze_1215 = None
        convert_element_type_455: "f16[32, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_351, torch.float16);  add_351 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_352: "f16[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_455, relu_144);  convert_element_type_455 = relu_144 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_147: "f16[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_352);  add_352 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:146 in forward, code: out = self.conv1(x)
        convolution_152: "f16[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_147, arg761_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  arg761_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:147 in forward, code: out = self.bn1(out)
        convert_element_type_456: "f32[512]" = torch.ops.prims.convert_element_type.default(arg762_1, torch.float32);  arg762_1 = None
        unsqueeze_1216: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_456, -1);  convert_element_type_456 = None
        unsqueeze_1217: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1216, -1);  unsqueeze_1216 = None
        sub_152: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_152, unsqueeze_1217);  convolution_152 = unsqueeze_1217 = None
        convert_element_type_457: "f32[512]" = torch.ops.prims.convert_element_type.default(arg763_1, torch.float32);  arg763_1 = None
        add_353: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_457, 1e-05);  convert_element_type_457 = None
        sqrt_152: "f32[512]" = torch.ops.aten.sqrt.default(add_353);  add_353 = None
        reciprocal_152: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_152);  sqrt_152 = None
        mul_456: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_152, 1);  reciprocal_152 = None
        unsqueeze_1218: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_456, -1);  mul_456 = None
        unsqueeze_1219: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1218, -1);  unsqueeze_1218 = None
        mul_457: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_152, unsqueeze_1219);  sub_152 = unsqueeze_1219 = None
        unsqueeze_1220: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg764_1, -1);  arg764_1 = None
        unsqueeze_1221: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1220, -1);  unsqueeze_1220 = None
        mul_458: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_457, unsqueeze_1221);  mul_457 = unsqueeze_1221 = None
        unsqueeze_1222: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg765_1, -1);  arg765_1 = None
        unsqueeze_1223: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1222, -1);  unsqueeze_1222 = None
        add_354: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_458, unsqueeze_1223);  mul_458 = unsqueeze_1223 = None
        convert_element_type_458: "f16[32, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_354, torch.float16);  add_354 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:148 in forward, code: out = self.relu(out)
        relu_148: "f16[32, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_458);  convert_element_type_458 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:150 in forward, code: out = self.conv2(out)
        convolution_153: "f16[32, 512, 7, 7]" = torch.ops.aten.convolution.default(relu_148, arg766_1, None, [1, 1], [1, 1], [1, 1], False, [0, 0], 1);  relu_148 = arg766_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:151 in forward, code: out = self.bn2(out)
        convert_element_type_459: "f32[512]" = torch.ops.prims.convert_element_type.default(arg767_1, torch.float32);  arg767_1 = None
        unsqueeze_1224: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_459, -1);  convert_element_type_459 = None
        unsqueeze_1225: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1224, -1);  unsqueeze_1224 = None
        sub_153: "f32[32, 512, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_153, unsqueeze_1225);  convolution_153 = unsqueeze_1225 = None
        convert_element_type_460: "f32[512]" = torch.ops.prims.convert_element_type.default(arg768_1, torch.float32);  arg768_1 = None
        add_355: "f32[512]" = torch.ops.aten.add.Tensor(convert_element_type_460, 1e-05);  convert_element_type_460 = None
        sqrt_153: "f32[512]" = torch.ops.aten.sqrt.default(add_355);  add_355 = None
        reciprocal_153: "f32[512]" = torch.ops.aten.reciprocal.default(sqrt_153);  sqrt_153 = None
        mul_459: "f32[512]" = torch.ops.aten.mul.Tensor(reciprocal_153, 1);  reciprocal_153 = None
        unsqueeze_1226: "f32[512, 1]" = torch.ops.aten.unsqueeze.default(mul_459, -1);  mul_459 = None
        unsqueeze_1227: "f32[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1226, -1);  unsqueeze_1226 = None
        mul_460: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(sub_153, unsqueeze_1227);  sub_153 = unsqueeze_1227 = None
        unsqueeze_1228: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg769_1, -1);  arg769_1 = None
        unsqueeze_1229: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1228, -1);  unsqueeze_1228 = None
        mul_461: "f32[32, 512, 7, 7]" = torch.ops.aten.mul.Tensor(mul_460, unsqueeze_1229);  mul_460 = unsqueeze_1229 = None
        unsqueeze_1230: "f16[512, 1]" = torch.ops.aten.unsqueeze.default(arg770_1, -1);  arg770_1 = None
        unsqueeze_1231: "f16[512, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1230, -1);  unsqueeze_1230 = None
        add_356: "f32[32, 512, 7, 7]" = torch.ops.aten.add.Tensor(mul_461, unsqueeze_1231);  mul_461 = unsqueeze_1231 = None
        convert_element_type_461: "f16[32, 512, 7, 7]" = torch.ops.prims.convert_element_type.default(add_356, torch.float16);  add_356 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:152 in forward, code: out = self.relu(out)
        relu_149: "f16[32, 512, 7, 7]" = torch.ops.aten.relu.default(convert_element_type_461);  convert_element_type_461 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:154 in forward, code: out = self.conv3(out)
        convolution_154: "f16[32, 2048, 7, 7]" = torch.ops.aten.convolution.default(relu_149, arg771_1, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1);  relu_149 = arg771_1 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:155 in forward, code: out = self.bn3(out)
        convert_element_type_462: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg772_1, torch.float32);  arg772_1 = None
        unsqueeze_1232: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(convert_element_type_462, -1);  convert_element_type_462 = None
        unsqueeze_1233: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1232, -1);  unsqueeze_1232 = None
        sub_154: "f32[32, 2048, 7, 7]" = torch.ops.aten.sub.Tensor(convolution_154, unsqueeze_1233);  convolution_154 = unsqueeze_1233 = None
        convert_element_type_463: "f32[2048]" = torch.ops.prims.convert_element_type.default(arg773_1, torch.float32);  arg773_1 = None
        add_357: "f32[2048]" = torch.ops.aten.add.Tensor(convert_element_type_463, 1e-05);  convert_element_type_463 = None
        sqrt_154: "f32[2048]" = torch.ops.aten.sqrt.default(add_357);  add_357 = None
        reciprocal_154: "f32[2048]" = torch.ops.aten.reciprocal.default(sqrt_154);  sqrt_154 = None
        mul_462: "f32[2048]" = torch.ops.aten.mul.Tensor(reciprocal_154, 1);  reciprocal_154 = None
        unsqueeze_1234: "f32[2048, 1]" = torch.ops.aten.unsqueeze.default(mul_462, -1);  mul_462 = None
        unsqueeze_1235: "f32[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1234, -1);  unsqueeze_1234 = None
        mul_463: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(sub_154, unsqueeze_1235);  sub_154 = unsqueeze_1235 = None
        unsqueeze_1236: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg774_1, -1);  arg774_1 = None
        unsqueeze_1237: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1236, -1);  unsqueeze_1236 = None
        mul_464: "f32[32, 2048, 7, 7]" = torch.ops.aten.mul.Tensor(mul_463, unsqueeze_1237);  mul_463 = unsqueeze_1237 = None
        unsqueeze_1238: "f16[2048, 1]" = torch.ops.aten.unsqueeze.default(arg775_1, -1);  arg775_1 = None
        unsqueeze_1239: "f16[2048, 1, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1238, -1);  unsqueeze_1238 = None
        add_358: "f32[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(mul_464, unsqueeze_1239);  mul_464 = unsqueeze_1239 = None
        convert_element_type_464: "f16[32, 2048, 7, 7]" = torch.ops.prims.convert_element_type.default(add_358, torch.float16);  add_358 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:160 in forward, code: out += identity
        add_359: "f16[32, 2048, 7, 7]" = torch.ops.aten.add.Tensor(convert_element_type_464, relu_147);  convert_element_type_464 = relu_147 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:161 in forward, code: out = self.relu(out)
        relu_150: "f16[32, 2048, 7, 7]" = torch.ops.aten.relu.default(add_359);  add_359 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:278 in _forward_impl, code: x = self.avgpool(x)
        mean: "f16[32, 2048, 1, 1]" = torch.ops.aten.mean.dim(relu_150, [-1, -2], True);  relu_150 = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:279 in _forward_impl, code: x = torch.flatten(x, 1)
        view: "f16[32, 2048]" = torch.ops.aten.reshape.default(mean, [32, 2048]);  mean = None

        # File: /home/dev/.conda/envs/pytorch-work-b200/lib/python3.12/site-packages/torchvision/models/resnet.py:280 in _forward_impl, code: x = self.fc(x)
        permute: "f16[2048, 1000]" = torch.ops.aten.permute.default(arg776_1, [1, 0]);  arg776_1 = None
        addmm: "f16[32, 1000]" = torch.ops.aten.addmm.default(arg777_1, view, permute);  arg777_1 = view = permute = None
        return (addmm,)
